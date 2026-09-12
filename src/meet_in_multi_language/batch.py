from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .models import Engine, TranscriptResult, TranscriptSegment
from .transcription import Transcriber


def _shift_segment(
    segment: TranscriptSegment,
    chunk_id: str,
    chunk_start_ms: int,
    chunk_end_ms: int,
) -> TranscriptSegment:
    start_ms = (
        chunk_start_ms + segment.start_ms
        if segment.start_ms is not None
        else chunk_start_ms
    )
    end_ms = (
        chunk_start_ms + segment.end_ms
        if segment.end_ms is not None
        else chunk_end_ms
    )
    return segment.model_copy(
        update={
            "segment_id": f"{chunk_id}-{segment.segment_id}",
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
    )


def transcribe_manifest(
    manifest_path: Path,
    output_dir: Path,
    engine: Engine | str,
    transcriber: Transcriber,
    keywords: list[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("manifest 沒有可轉錄的音訊片段")

    engine_name = engine.value if isinstance(engine, Engine) else engine
    run_dir = output_dir / engine_name
    part_dir = run_dir / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    combined_segments: list[TranscriptSegment] = []
    detected_languages: list[str] = []
    transcript_parts: list[str] = []
    completed_chunks: list[str] = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("manifest 的片段格式不正確")
        chunk_id = str(chunk["chunk_id"])
        audio_path = manifest_path.parent / str(chunk["path"])
        part_path = part_dir / f"{chunk_id}.json"
        if part_path.is_file() and not force:
            result = TranscriptResult.model_validate_json(
                part_path.read_text(encoding="utf-8")
            )
        else:
            result = transcriber.transcribe(audio_path, engine, keywords or [])
            part_path.write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

        chunk_start_ms = int(chunk["start_ms"])
        chunk_end_ms = int(chunk["end_ms"])
        combined_segments.extend(
            _shift_segment(
                segment,
                chunk_id=chunk_id,
                chunk_start_ms=chunk_start_ms,
                chunk_end_ms=chunk_end_ms,
            )
            for segment in result.segments
        )
        transcript_parts.append(result.text)
        for language in result.detected_languages:
            if language not in detected_languages:
                detected_languages.append(language)
        completed_chunks.append(chunk_id)

    transcript = TranscriptResult(
        provider=(result.provider if completed_chunks else "unknown"),
        model=(result.model if completed_chunks else engine_name),
        text="\n".join(part for part in transcript_parts if part),
        detected_languages=detected_languages,
        segments=combined_segments,
    )
    output: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": manifest.get("source", {}),
        "completed_chunks": completed_chunks,
        "transcript": transcript.model_dump(mode="json"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transcript.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output
