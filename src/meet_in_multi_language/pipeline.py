from __future__ import annotations

import json
import os
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .audio import prepare_audio_chunks, probe_audio, sha256_file
from .batch import transcribe_manifest
from .export import ExportFormat, export_payload
from .gpu import host_gpu_lock
from .models import (
    Engine,
    EvaluationRun,
    RunStatus,
    SummaryResult,
    TranscriptResult,
)
from .ollama_eval import (
    OllamaClient,
    format_transcript_for_summary,
    parse_summary_payload,
    validate_summary,
)
from .speaches import PROFILES, SpeachesTranscriber
from .transcription import OpenAITranscriber


def _ensure_single_chunk_manifest(audio_path: Path, output_dir: Path) -> Path:
    """針對短於切段秒數的音訊直接建立單一 chunk 的 manifest，避免額外切段耗損。"""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        return manifest_path

    info = probe_audio(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = int(info["duration_ms"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "filename": audio_path.name,
            "sha256": sha256_file(audio_path),
            **info,
        },
        "processing": {
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "target_segment_seconds": duration_ms // 1000 + 1,
        },
        "chunks": [
            {
                "chunk_id": "chunk-001",
                "path": str(audio_path.resolve()),
                "start_ms": 0,
                "end_ms": duration_ms,
                "duration_ms": duration_ms,
                "size_bytes": info["size_bytes"],
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def sync_unload_speaches(speaches_url: str, model: str) -> None:
    """以同步 HTTP DELETE 通知 Speaches 卸載模型以釋出顯存。"""
    speaches_root = speaches_url.removesuffix("/v1")
    encoded_model = quote(model, safe="")
    try:
        with httpx.Client(timeout=5.0) as http_client:
            response = http_client.delete(f"{speaches_root}/api/ps/{encoded_model}")
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()
    except httpx.ConnectError:
        return
    except Exception as err:
        from .gpu import GpuTransitionError

        raise GpuTransitionError(
            f"無法卸載 Speaches 模型 {model}，中止後續摘要避免顯存溢出：{err}"
        ) from err


def run_pipeline(
    source_audio: Path,
    output_dir: Path,
    engine: str = "breeze",
    summary_model: str | None = None,
    segment_seconds: int = 600,
    keywords: list[str] | None = None,
    export_formats: list[str] | None = None,
    speaches_url: str = "http://127.0.0.1:8001/v1",
    ollama_url: str = "http://127.0.0.1:11434",
    openai_api_key: str | None = None,
    force: bool = False,
    transcriber: Any | None = None,
    ollama_client: Any | None = None,
    speaches_unloader: Any | None = None,
    gpu_lock_file: Path | str | None = None,
) -> dict[str, Any]:
    """端到端批次處理管線：音訊切段、斷點續轉、時間軸對齊、結構化摘要與多格式匯出。"""
    if not source_audio.is_file():
        raise FileNotFoundError(f"找不到音訊檔案：{source_audio}")

    start_time = datetime.now(UTC)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_info = probe_audio(source_audio)
    duration_ms = int(audio_info["duration_ms"])

    # 1. 切段準備
    chunks_dir = output_dir / "chunks"
    if duration_ms > segment_seconds * 1000:
        manifest_data = prepare_audio_chunks(
            source_audio, chunks_dir, segment_seconds=segment_seconds
        )
        manifest_path = chunks_dir / "manifest.json"
        total_chunks = len(manifest_data["chunks"])  # type: ignore[arg-type]
    else:
        manifest_path = _ensure_single_chunk_manifest(source_audio, chunks_dir)
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_chunks = 1

    # 2. 轉錄處理與 3. 摘要處理（受主機級跨行程 GPU 互斥鎖保護）
    requires_gpu = (engine == "breeze") or bool(summary_model)
    gpu_ctx = host_gpu_lock(gpu_lock_file) if requires_gpu else nullcontext()

    summary_result: SummaryResult | None = None
    summary_path: Path | None = None

    with gpu_ctx:
        if engine == "breeze":
            # 跨程序安全：進入 Speaches 前同步卸載 Ollama (不論是否提供自訂 transcriber)
            from .gpu import KNOWN_OLLAMA_MODELS, sync_unload_all_loaded_ollama_models
            fallback_models = set(KNOWN_OLLAMA_MODELS)
            if summary_model:
                fallback_models.add(summary_model)
            sync_unload_all_loaded_ollama_models(ollama_url, fallback_models)

            if transcriber is None:
                profile = PROFILES["breeze"]
                transcriber = SpeachesTranscriber(speaches_url, profile)
        else:
            if transcriber is None:
                api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    raise ValueError(f"使用 {engine} 需要提供 OPENAI_API_KEY")
                transcriber = OpenAITranscriber(api_key)

        def _pipeline_progress(idx: int, total: int, c_id: str, cached: bool) -> None:
            tag = "[快取]" if cached else "[處理]"
            print(f"  {tag} 切段轉錄進度 ({idx}/{total})：{c_id}")

        transcribe_result = transcribe_manifest(
            manifest_path,
            output_dir,
            engine,
            transcriber,
            keywords=keywords or [],
            force=force,
            progress_callback=_pipeline_progress,
        )
        transcript_data = transcribe_result["transcript"]
        assert isinstance(transcript_data, dict)
        transcript_result = TranscriptResult.model_validate(transcript_data)

        # 3. 摘要處理
        if summary_model:
            # 若使用 Breeze ASR，轉錄完成後主動卸載以釋放顯存供 Ollama 摘要使用
            if engine == "breeze":
                unloader = speaches_unloader or sync_unload_speaches
                unloader(speaches_url, PROFILES["breeze"].model)

            client = ollama_client or OllamaClient(ollama_url)
            formatted_transcript = format_transcript_for_summary(transcript_result)
            summary_raw = client.summarize(
                summary_model, formatted_transcript, keep_alive=0, num_ctx=24576
            )
            content = summary_raw.get("message", {}).get("content", "")
            validation = validate_summary(content, formatted_transcript)
            if not validation["schema_valid"]:
                raise ValueError(f"摘要結果不符合結構契約：{validation.get('error')}")
            if validation["unknown_evidence_ids"]:
                raise ValueError(
                    f"摘要包含未知的段落引用識別碼：{validation['unknown_evidence_ids']}"
                )
            if validation["forbidden_terms"]:
                raise ValueError(
                    f"摘要包含非臺灣慣用詞彙：{validation['forbidden_terms']}"
                )
            summary_result = parse_summary_payload(
                content,
                model=summary_model,
                source_revision_id=transcript_result.revision_id,
            )
            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                summary_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )


    # 4. 多格式匯出
    formats_to_export = export_formats or ["txt", "srt", "vtt", "md", "json"]
    dummy_run = EvaluationRun(
        run_id="pipeline-run",
        original_filename=source_audio.name,
        stored_filename=source_audio.name,
        engine=Engine.BREEZE if engine == "breeze" else Engine.DIARIZE,
        status=RunStatus.COMPLETED,
        raw_asr=transcript_result,
        revisions=[transcript_result],
        result=transcript_result,
        summary=summary_result,
    )

    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    exported_files: dict[str, str] = {}

    for fmt in formats_to_export:
        content, _, filename = export_payload(dummy_run, format_name=fmt)  # type: ignore[arg-type]
        target_file = exports_dir / filename
        target_file.write_text(content, encoding="utf-8")
        exported_files[fmt] = str(target_file)

    end_time = datetime.now(UTC)
    report = {
        "status": "completed",
        "audio": {
            "path": str(source_audio.resolve()),
            "duration_ms": duration_ms,
            "chunks_count": total_chunks,
        },
        "engine": engine,
        "transcript": {
            "segments_count": len(transcript_result.segments),
            "text_length": len(transcript_result.text),
            "path": str(output_dir / engine / "transcript.json"),
        },
        "summary": {
            "model": summary_model,
            "path": str(summary_path) if summary_path else None,
            "overview": summary_result.overview if summary_result else None,
        },
        "exports": exported_files,
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }
    (output_dir / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
