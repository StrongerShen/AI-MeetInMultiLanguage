from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import Engine, TranscriptResult, TranscriptSegment


class Transcriber(Protocol):
    def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult: ...


class AsyncTranscriber(Protocol):
    async def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult: ...


def _standard_result(response: Any) -> TranscriptResult:
    text = str(response.text).strip()
    detected_languages = [
        str(getattr(language, "code", language))
        for language in (getattr(response, "languages", None) or [])
    ]
    return TranscriptResult(
        model=Engine.TRANSCRIBE.value,
        text=text,
        detected_languages=detected_languages,
        segments=[TranscriptSegment(segment_id="segment-1", text=text)],
    )


def _diarized_result(response: Any) -> TranscriptResult:
    segments = []
    for index, segment in enumerate(getattr(response, "segments", None) or [], 1):
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        segments.append(
            TranscriptSegment(
                segment_id=f"segment-{index}",
                start_ms=round(float(start) * 1000) if start is not None else None,
                end_ms=round(float(end) * 1000) if end is not None else None,
                speaker=getattr(segment, "speaker", None),
                text=str(getattr(segment, "text", "")).strip(),
            )
        )

    text = str(getattr(response, "text", "")).strip()
    if not text:
        text = "\n".join(segment.text for segment in segments)
    return TranscriptResult(model=Engine.DIARIZE.value, text=text, segments=segments)


class OpenAITranscriber:
    def __init__(self, api_key: str, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client

    def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult:
        if engine == Engine.DIARIZE:
            return self._diarize(audio_path)
        return self._standard_transcription(audio_path, keywords)

    def _standard_transcription(
        self, audio_path: Path, keywords: list[str]
    ) -> TranscriptResult:
        context = (
            "這是臺灣的會議錄音，可能在臺灣華語、英語、日語與臺語之間切換。"
            "請忠實轉錄原文，中文使用臺灣繁體字，不要翻譯，也不要補上未說出的內容。"
        )
        request: dict[str, object] = {
            "model": Engine.TRANSCRIBE.value,
            "prompt": context,
            "languages": ["zh-tw", "en", "ja"],
        }
        if keywords:
            request["keywords"] = keywords

        with audio_path.open("rb") as audio_file:
            request["file"] = audio_file
            response = self._client.audio.transcriptions.create(
                **request,
            )

        return _standard_result(response)

    def _diarize(self, audio_path: Path) -> TranscriptResult:
        with audio_path.open("rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model=Engine.DIARIZE.value,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )

        return _diarized_result(response)


class AsyncOpenAITranscriber:
    def __init__(self, api_key: str, client: Any | None = None) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
        self._client = client

    async def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult:
        with audio_path.open("rb") as audio_file:
            if engine == Engine.DIARIZE:
                response = await self._client.audio.transcriptions.create(
                    model=Engine.DIARIZE.value,
                    file=audio_file,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
                return _diarized_result(response)

            context = (
                "這是臺灣的會議錄音，可能在臺灣華語、英語、日語與臺語之間切換。"
                "請忠實轉錄原文，中文使用臺灣繁體字，不要翻譯，也不要補上未說出的內容。"
            )
            request: dict[str, object] = {
                "model": Engine.TRANSCRIBE.value,
                "file": audio_file,
                "prompt": context,
                "languages": ["zh-tw", "en", "ja"],
            }
            if keywords:
                request["keywords"] = keywords
            response = await self._client.audio.transcriptions.create(
                **request,
            )
            return _standard_result(response)
