from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from typing import Any

import pytest

from meet_in_multi_language.models import Engine, TranscriptResult, TranscriptSegment
from meet_in_multi_language.pipeline import run_pipeline
from meet_in_multi_language import gpu as gpu_module

@pytest.fixture(autouse=True)
def mock_gpu_sync_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock 出去 pipeline 的同步 GPU 卸載，避免觸發網路防線。"""
    monkeypatch.setattr(gpu_module, "sync_unload_all_loaded_ollama_models", lambda *args, **kwargs: None)


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
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def summarize(
        self, model: str, transcript: str, keep_alive: Any = None, num_ctx: int = 24576
    ) -> dict[str, Any]:
        self.calls.append({
            "model": model,
            "transcript": transcript,
            "keep_alive": keep_alive,
            "num_ctx": num_ctx,
        })
        payload = {
            "overview": "這是會議總覽。",
            "topics": [{"title": "議題", "summary": "進度報告", "evidence_ids": ["seg-001"]}],
            "decisions": [{"text": "同意上線", "evidence_ids": ["seg-001"]}],
            "action_items": [{"task": "完成檢查", "owner": "測試者", "due_date": "2026-09-20", "evidence_ids": ["seg-001"]}],
            "open_questions": [],
        }
        return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    audio_path = make_test_wav(tmp_path / "meeting.wav", duration_seconds=1)
    output_dir = tmp_path / "output"

    unloaded_calls: list[tuple[str, str]] = []

    def mock_unloader(url: str, model: str) -> None:
        unloaded_calls.append((url, model))

    fake_ollama = FakeOllamaClient()
    report = run_pipeline(
        source_audio=audio_path,
        output_dir=output_dir,
        engine="breeze",
        summary_model="qwen3.5:9b",
        segment_seconds=600,
        export_formats=["txt", "srt", "vtt", "md", "json"],
        transcriber=FakeTranscriber(),
        ollama_client=fake_ollama,
        speaches_unloader=mock_unloader,
    )

    assert report["status"] == "completed"
    assert report["audio"]["chunks_count"] == 1
    assert report["transcript"]["segments_count"] == 1
    assert report["summary"]["model"] == "qwen3.5:9b"
    assert len(unloaded_calls) == 1
    assert "Breeze-ASR-26" in unloaded_calls[0][1]
    assert len(fake_ollama.calls) == 1
    assert fake_ollama.calls[0]["keep_alive"] == "0m"
    assert fake_ollama.calls[0]["num_ctx"] == 24576

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


def test_pipeline_unloader_failure_aborts_summary(tmp_path: Path) -> None:
    from meet_in_multi_language.gpu import GpuTransitionError

    audio_path = make_test_wav(tmp_path / "fail_unload.wav", duration_seconds=1)
    output_dir = tmp_path / "output_fail"

    def broken_unloader(url: str, model: str) -> None:
        raise GpuTransitionError("顯存釋放失敗，拒絕載入下一模型")

    import pytest

    with pytest.raises(GpuTransitionError, match="顯存釋放失敗"):
        run_pipeline(
            source_audio=audio_path,
            output_dir=output_dir,
            engine="breeze",
            summary_model="qwen3.5:9b",
            transcriber=FakeTranscriber(),
            ollama_client=FakeOllamaClient(),
            speaches_unloader=broken_unloader,
        )


def test_pipeline_ollama_unloader_failure_aborts_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from meet_in_multi_language.gpu import GpuTransitionError
    from meet_in_multi_language import gpu as gpu_module

    def broken_ollama_unloader(url: str, models: set[str]) -> None:
        raise GpuTransitionError("Ollama 卸載失敗，拒絕執行 ASR")

    monkeypatch.setattr(gpu_module, "sync_unload_all_loaded_ollama_models", broken_ollama_unloader)

    audio_path = make_test_wav(tmp_path / "fail_unload2.wav", duration_seconds=1)
    output_dir = tmp_path / "output_fail2"

    with pytest.raises(GpuTransitionError, match="Ollama 卸載失敗"):
        run_pipeline(
            source_audio=audio_path,
            output_dir=output_dir,
            engine="breeze",
            summary_model="qwen3.5:9b",
            transcriber=FakeTranscriber(),
            ollama_client=FakeOllamaClient(),
            speaches_unloader=lambda url, m: None,
        )


def test_pipeline_rejects_unknown_evidence_ids(tmp_path: Path) -> None:
    audio_path = make_test_wav(tmp_path / "bad_evidence.wav", duration_seconds=1)
    output_dir = tmp_path / "output_bad_ev"

    class BadOllamaClient:
        def summarize(self, model: str, transcript: str, keep_alive: Any = None, num_ctx: int = 24576) -> dict[str, Any]:
            payload = {
                "overview": "總覽",
                "topics": [{"title": "題", "summary": "摘", "evidence_ids": ["non-existent-seg"]}],
                "decisions": [],
                "action_items": [],
                "open_questions": [],
            }
            return {"message": {"content": json.dumps(payload)}}

    import pytest

    with pytest.raises(ValueError, match="未知的段落引用識別碼"):
        run_pipeline(
            source_audio=audio_path,
            output_dir=output_dir,
            engine="breeze",
            summary_model="qwen3.5:9b",
            transcriber=FakeTranscriber(),
            ollama_client=BadOllamaClient(),
            speaches_unloader=lambda url, m: None,
        )


def test_network_guard_blocks_external_services() -> None:
    import socket
    import pytest

    with pytest.raises(RuntimeError, match="測試環境防線觸發：禁止直接連線至外部服務"):
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 8001))
        finally:
            s.close()

    with pytest.raises(RuntimeError, match="測試環境防線觸發：禁止直接連線至外部服務"):
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 11434))
        finally:
            s.close()
