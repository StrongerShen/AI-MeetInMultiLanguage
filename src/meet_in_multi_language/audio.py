from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


class AudioToolError(RuntimeError):
    pass


def probe_audio(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=codec_name,sample_rate,channels",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "無法讀取音訊資訊"
        raise AudioToolError(detail)
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        audio_format = payload["format"]
        return {
            "format": audio_format["format_name"],
            "duration_ms": round(float(audio_format["duration"]) * 1000),
            "size_bytes": int(audio_format["size"]),
            "codec": stream["codec_name"],
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioToolError("音訊資訊不完整") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_audio_chunks(
    source: Path, output_dir: Path, segment_seconds: int = 600
) -> dict[str, object]:
    if segment_seconds <= 0:
        raise ValueError("片段秒數必須大於零")
    if not source.is_file():
        raise FileNotFoundError(source)

    source_info = probe_audio(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "chunk-%03d.mp3"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "音訊切段失敗"
        raise AudioToolError(detail)

    chunks: list[dict[str, object]] = []
    start_ms = 0
    for index, chunk_path in enumerate(sorted(output_dir.glob("chunk-*.mp3")), 1):
        chunk_info = probe_audio(chunk_path)
        duration_ms = int(chunk_info["duration_ms"])
        chunks.append(
            {
                "chunk_id": f"chunk-{index:03d}",
                "path": chunk_path.name,
                "start_ms": start_ms,
                "end_ms": start_ms + duration_ms,
                "duration_ms": duration_ms,
                "size_bytes": chunk_info["size_bytes"],
            }
        )
        start_ms += duration_ms

    if not chunks:
        raise AudioToolError("切段完成，但找不到輸出音訊")

    manifest = {
        "schema_version": 1,
        "source": {
            "filename": source.name,
            "sha256": sha256_file(source),
            **source_info,
        },
        "processing": {
            "sample_rate": 16000,
            "channels": 1,
            "bit_rate": 32000,
            "target_segment_seconds": segment_seconds,
        },
        "chunks": chunks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

