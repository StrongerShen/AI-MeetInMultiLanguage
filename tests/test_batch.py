import json
from pathlib import Path

from meet_in_multi_language.batch import transcribe_manifest
from meet_in_multi_language.models import Engine, TranscriptResult, TranscriptSegment


class FakeTranscriber:
    def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult:
        return TranscriptResult(
            model=engine.value,
            text=audio_path.stem,
            segments=[
                TranscriptSegment(
                    segment_id="segment-1",
                    start_ms=100,
                    end_ms=900,
                    speaker="A",
                    text=audio_path.stem,
                )
            ],
        )


def test_batch_transcription_restores_source_timeline(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk-000.mp3").write_bytes(b"first")
    (chunk_dir / "chunk-001.mp3").write_bytes(b"second")
    manifest = {
        "source": {"filename": "meeting.mp3"},
        "chunks": [
            {
                "chunk_id": "chunk-001",
                "path": "chunk-000.mp3",
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "chunk_id": "chunk-002",
                "path": "chunk-001.mp3",
                "start_ms": 1000,
                "end_ms": 2000,
            },
        ],
    }
    manifest_path = chunk_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = transcribe_manifest(
        manifest_path, tmp_path / "results", Engine.DIARIZE, FakeTranscriber()
    )

    transcript = output["transcript"]
    assert transcript["segments"][0]["start_ms"] == 100
    assert transcript["segments"][1]["start_ms"] == 1100
    assert transcript["segments"][1]["segment_id"] == "chunk-002-segment-1"
    assert len(output["completed_chunks"]) == 2
    assert (tmp_path / "results" / Engine.DIARIZE.value / "transcript.json").is_file()


def test_batch_transcription_calls_progress_callback(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk-000.mp3").write_bytes(b"first")
    manifest = {
        "source": {"filename": "meeting.mp3"},
        "chunks": [
            {
                "chunk_id": "chunk-001",
                "path": "chunk-000.mp3",
                "start_ms": 0,
                "end_ms": 1000,
            }
        ],
    }
    manifest_path = chunk_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    progress_events: list[tuple[int, int, str, bool]] = []
    output = transcribe_manifest(
        manifest_path,
        tmp_path / "results",
        Engine.DIARIZE,
        FakeTranscriber(),
        progress_callback=lambda idx, total, cid, cached: progress_events.append(
            (idx, total, cid, cached)
        ),
    )
    assert progress_events == [(1, 1, "chunk-001", False)]

    # 再次執行應命中快取
    cached_events: list[tuple[int, int, str, bool]] = []
    transcribe_manifest(
        manifest_path,
        tmp_path / "results",
        Engine.DIARIZE,
        FakeTranscriber(),
        progress_callback=lambda idx, total, cid, cached: cached_events.append(
            (idx, total, cid, cached)
        ),
    )
    assert cached_events == [(1, 1, "chunk-001", True)]
