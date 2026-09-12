from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from typing import Any

from meet_in_multi_language.models import Engine, TranscriptResult, TranscriptSegment
from meet_in_multi_language.pipeline import run_pipeline


def make_test_wav(path: Path, duration_seconds: int = 2) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * (16000 * duration_seconds))
    return path


class FakeTranscriber:
    def transcribe(
        self, audio_path: Path, engine: Engine | str, keywords: list[str]
    ) -> TranscriptResult:
        return TranscriptResult(
            provider="fake",
            model="breeze",
            text="這是一段端到端測試逐字稿內容。",
            detected_languages=["zh"],
            segments=[
                TranscriptSegment(
                    segment_id="seg-001",
                    start_ms=100,
                    end_ms=1800,
                    speaker="SPEAKER_00",
                    text="這是一段端到端測試逐字稿內容。",
                )
            ],
        )


class FakeOllamaClient:
    def summarize(
        self, model: str, transcript: str, keep_alive: Any = None
    ) -> dict[str, Any]:
        payload = {
            "overview": "這是會議總覽。",
            "topics": [{"title": "議題", "summary": "進度報告", "evidence_ids": ["chunk-001-seg-001"]}],
            "decisions": [{"text": "同意上線", "evidence_ids": ["chunk-001-seg-001"]}],
            "action_items": [{"task": "完成檢查", "owner": "測試者", "due_date": "2026-09-20", "evidence_ids": ["chunk-001-seg-001"]}],
            "open_questions": [],
        }
        return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    audio_path = make_test_wav(tmp_path / "meeting.wav", duration_seconds=1)
    output_dir = tmp_path / "output"

    report = run_pipeline(
        source_audio=audio_path,
        output_dir=output_dir,
        engine="breeze",
        summary_model="qwen3.5:9b",
        segment_seconds=600,
        export_formats=["txt", "srt", "vtt", "md", "json"],
        transcriber=FakeTranscriber(),
        ollama_client=FakeOllamaClient(),
    )

    assert report["status"] == "completed"
    assert report["audio"]["chunks_count"] == 1
    assert report["transcript"]["segments_count"] == 1
    assert report["summary"]["model"] == "qwen3.5:9b"

    # 驗證產物檔案
    assert (output_dir / "pipeline_report.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "breeze" / "transcript.json").is_file()

    exports_dir = output_dir / "exports"
    assert (exports_dir / "meeting_raw_asr.txt").is_file()
    assert (exports_dir / "meeting_raw_asr.srt").is_file()
    assert (exports_dir / "meeting_raw_asr.vtt").is_file()
    assert (exports_dir / "meeting_raw_asr.md").is_file()
    assert (exports_dir / "meeting_raw_asr.json").is_file()

    # 檢查 Markdown 內容
    md_content = (exports_dir / "meeting_raw_asr.md").read_text(encoding="utf-8")
    assert "這是會議總覽。" in md_content
    assert "同意上線" in md_content
