from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import httpx

from .models import TranscriptResult, TranscriptRevisionKind, TranscriptSegment


MIXED_LANGUAGE_PROMPT = (
    "這是一場包含臺灣華語、English、日本語，以及臺灣話口語"
    "（如：按呢、代誌、歹勢）的商務會議。請忠實保留各語言與專有名詞。"
)


@dataclass(frozen=True, slots=True)
class SpeachesProfile:
    name: str
    model: str
    language: str | None
    vad_filter: bool
    word_timestamps: bool
    prompt: str | None


PROFILES = {
    "breeze": SpeachesProfile(
        name="breeze",
        model="paulpengtw/faster-whisper-Breeze-ASR-26",
        language="zh",
        # 參考專案曾量到 VAD 裁掉開頭語音，因此先關閉並要求逐份素材 A/B 驗證。
        vad_filter=False,
        # 此 Breeze CTranslate2 版本實測的 word 對齊會產生錯誤短時間範圍；
        # Speaches 的 segment timestamp（約 30 秒）反而能維持正確原時間軸。
        word_timestamps=False,
        prompt=None,
    ),
    "large-v3": SpeachesProfile(
        name="large-v3",
        model="Systran/faster-whisper-large-v3",
        # 混語錄音不鎖單一語言，讓模型自行判斷。
        language=None,
        vad_filter=True,
        word_timestamps=False,
        prompt=MIXED_LANGUAGE_PROMPT,
    ),
}


def _quality_flags(text: str, segment: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    tokens = re.findall(r"[\w\u3400-\u9fff]+", text)
    if len(tokens) >= 5:
        dominant = max(tokens.count(token) for token in set(tokens)) / len(tokens)
        if dominant >= 0.4:
            flags.append("suspected_repetition_hallucination")
    no_speech = segment.get("no_speech_prob")
    if no_speech is not None and float(no_speech) >= 0.6:
        flags.append("high_no_speech_probability")
    compression = segment.get("compression_ratio")
    if compression is not None and float(compression) >= 2.4:
        flags.append("high_compression_ratio")
    return flags


def _segments_from_words(
    words: list[dict[str, Any]], max_chars: int = 28, gap_break: float = 0.8
) -> list[TranscriptSegment]:
    """把逐字時間戳重切為適合回聽的短段落。"""
    segments: list[TranscriptSegment] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        text = "".join(str(word.get("word", "")) for word in buffer).strip()
        if text:
            start = float(buffer[0].get("start", 0))
            end = max(float(buffer[-1].get("end", start)), start + 0.3)
            segments.append(
                TranscriptSegment(
                    segment_id=f"segment-{len(segments) + 1}",
                    start_ms=round(start * 1000),
                    end_ms=round(end * 1000),
                    text=text,
                )
            )
        buffer.clear()

    for word in words:
        if buffer:
            gap = float(word.get("start", 0)) - float(buffer[-1].get("end", 0))
            if gap >= gap_break:
                flush()
        buffer.append(word)
        text = "".join(str(item.get("word", "")) for item in buffer).strip()
        if len(text) >= max_chars or text.endswith(("。", "！", "？", ".", "!", "?")):
            flush()
    flush()
    return segments


def _segments_from_response(data: dict[str, Any], word_timestamps: bool) -> list[TranscriptSegment]:
    words = data.get("words")
    valid_words = (
        [word for word in words if isinstance(word, dict)]
        if isinstance(words, list)
        else []
    )
    # Breeze 經部分相容端點回傳的 words 可能是一整個 30 秒句塊，而非真正逐字
    # DTW 結果；這種資料的短起迄時間會誤導回聽定位，應退回 segment timestamp。
    has_real_word_granularity = bool(valid_words) and all(
        len(str(word.get("word", "")).strip()) <= 28 for word in valid_words
    )
    if word_timestamps and has_real_word_granularity:
        return _segments_from_words(valid_words)

    raw_segments = [
        segment for segment in (data.get("segments") or []) if isinstance(segment, dict)
    ]
    audio_duration = float(data.get("duration", 0) or 0)
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(raw_segments, 1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = segment.get("start")
        end = segment.get("end")
        if start is not None and end is not None:
            start_seconds = float(start)
            end_seconds = float(end)
            if len(text) > 28 and end_seconds - start_seconds < 2:
                if index < len(raw_segments):
                    next_start = raw_segments[index].get("start")
                    if next_start is not None:
                        end = next_start
                elif audio_duration > end_seconds:
                    end = audio_duration
        segments.append(
            TranscriptSegment(
                segment_id=f"segment-{index}",
                start_ms=round(float(start) * 1000) if start is not None else None,
                end_ms=round(float(end) * 1000) if end is not None else None,
                text=text,
                quality_flags=_quality_flags(text, segment),
            )
        )
    return segments


class SpeachesTranscriber:
    """呼叫 OpenAI 相容的 Speaches/faster-whisper 轉錄服務。"""

    def __init__(
        self,
        base_url: str,
        profile: SpeachesProfile,
        *,
        vad_filter: bool | None = None,
        prompt: str | None = None,
        timeout_seconds: float = 1800,
        client: httpx.Client | None = None,
    ) -> None:
        self.profile = profile
        self.vad_filter = profile.vad_filter if vad_filter is None else vad_filter
        self.prompt = profile.prompt if prompt is None else prompt
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    def transcribe(
        self, audio_path: Path, engine: object, keywords: list[str]
    ) -> TranscriptResult:
        data: dict[str, str | list[str]] = {
            "model": self.profile.model,
            "response_format": "verbose_json",
            "temperature": "0",
            "vad_filter": str(self.vad_filter).lower(),
        }
        if self.prompt:
            data["prompt"] = self.prompt
        if self.profile.language:
            data["language"] = self.profile.language
        if keywords:
            data["hotwords"] = ", ".join(keywords)
        if self.profile.word_timestamps:
            data["timestamp_granularities"] = ["segment", "word"]

        with audio_path.open("rb") as audio_file:
            response = self._client.post(
                "/audio/transcriptions",
                data=data,
                files={"file": (audio_path.name, audio_file, "application/octet-stream")},
            )
        response.raise_for_status()
        payload = response.json()
        segments = _segments_from_response(payload, self.profile.word_timestamps)
        text = str(payload.get("text", "")).strip()
        if not text:
            text = "\n".join(segment.text for segment in segments)
        language = str(payload.get("language", "")).strip()
        return TranscriptResult(
            provider="speaches",
            model=self.profile.model,
            revision_kind=TranscriptRevisionKind.RAW_ASR,
            source_revision_id=None,
            text=text,
            detected_languages=[language] if language else [],
            segments=segments,
        )


class AsyncSpeachesTranscriber:
    """非同步呼叫 OpenAI 相容的 Speaches/faster-whisper 轉錄服務。"""

    def __init__(
        self,
        base_url: str,
        profile: SpeachesProfile,
        *,
        vad_filter: bool | None = None,
        prompt: str | None = None,
        timeout_seconds: float = 1800,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.profile = profile
        self.vad_filter = profile.vad_filter if vad_filter is None else vad_filter
        self.prompt = profile.prompt if prompt is None else prompt
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    async def transcribe(
        self, audio_path: Path, engine: object, keywords: list[str]
    ) -> TranscriptResult:
        data: dict[str, str | list[str]] = {
            "model": self.profile.model,
            "response_format": "verbose_json",
            "temperature": "0",
            "vad_filter": str(self.vad_filter).lower(),
        }
        if self.prompt:
            data["prompt"] = self.prompt
        if self.profile.language:
            data["language"] = self.profile.language
        if keywords:
            data["hotwords"] = ", ".join(keywords)
        if self.profile.word_timestamps:
            data["timestamp_granularities"] = ["segment", "word"]

        with audio_path.open("rb") as audio_file:
            content = audio_file.read()

        response = await self._client.post(
            "/audio/transcriptions",
            data=data,
            files={"file": (audio_path.name, content, "application/octet-stream")},
        )
        response.raise_for_status()
        payload = response.json()
        segments = _segments_from_response(payload, self.profile.word_timestamps)
        text = str(payload.get("text", "")).strip()
        if not text:
            text = "\n".join(segment.text for segment in segments)
        language = str(payload.get("language", "")).strip()
        return TranscriptResult(
            provider="speaches",
            model=self.profile.model,
            revision_kind=TranscriptRevisionKind.RAW_ASR,
            source_revision_id=None,
            text=text,
            detected_languages=[language] if language else [],
            segments=segments,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
