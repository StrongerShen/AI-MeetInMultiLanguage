from __future__ import annotations

import asyncio
import json
import re

from .gpu import GpuCategory, GpuWorkQueue
from .models import (
    Engine,
    EvaluationRun,
    RunStatus,
    TranscriptResult,
    TranscriptRevisionKind,
    TranscriptSegment,
)
from .ollama_eval import (
    AsyncOllamaClient,
    format_transcript_for_summary,
    parse_summary_payload,
    validate_summary,
)
from .storage import RunStore
from .transcription import AsyncTranscriber, Transcriber


# 中文通常接近一字一 token，保留足夠空間給提示詞、JSON 與最終輸出。
SUMMARY_CHUNK_CHARS = 12_000


def _split_summary_input(transcript: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in transcript.splitlines():
        evidence_match = re.match(r"^(\[seg-\d+\])", line)
        evidence_prefix = evidence_match.group(1) if evidence_match else ""
        line_parts = [
            line[index : index + SUMMARY_CHUNK_CHARS - len(evidence_prefix)]
            for index in range(
                0, len(line), SUMMARY_CHUNK_CHARS - len(evidence_prefix)
            )
        ] or [""]
        for index, line_part in enumerate(line_parts):
            normalized_part = (
                evidence_prefix + line_part
                if index > 0 and evidence_prefix
                else line_part
            )
            if current and current_length + len(normalized_part) + 1 > SUMMARY_CHUNK_CHARS:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            current.append(normalized_part)
            current_length += len(normalized_part) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [transcript]


def _assert_valid_summary(content: str, transcript: str) -> None:
    validation = validate_summary(content, transcript)
    if not validation["json_valid"] or not validation["schema_valid"]:
        detail = validation.get("error") or validation.get("missing_fields")
        raise ValueError(f"摘要 JSON 格式不符合契約：{detail}")
    if validation["unknown_evidence_ids"]:
        raise ValueError(
            "摘要引用不存在的段落："
            + ", ".join(validation["unknown_evidence_ids"])
        )
    if validation["forbidden_terms"]:
        raise ValueError(
            "摘要含有非臺灣慣用詞彙：" + ", ".join(validation["forbidden_terms"])
        )


async def _summarize_content(
    ollama_client: AsyncOllamaClient, model: str, transcript: str
) -> str:
    chunks = _split_summary_input(transcript)
    if len(chunks) == 1:
        response = await ollama_client.summarize(model, transcript, keep_alive=0)
        return str(response.get("message", {}).get("content", ""))

    partial_summaries: list[dict[str, object]] = []
    for chunk in chunks:
        response = await ollama_client.summarize(model, chunk, keep_alive="5m")
        content = str(response.get("message", {}).get("content", ""))
        _assert_valid_summary(content, chunk)
        partial_summaries.append(json.loads(content))

    synthesis_input = (
        "以下是同一場會議依序產生的分段摘要。"
        "請合併重複內容，保留原 evidence ID，"
        "不得新增未出現在分段摘要中的事實或引用。\n"
        + json.dumps(partial_summaries, ensure_ascii=False)
    )
    response = await ollama_client.summarize(model, synthesis_input, keep_alive=0)
    return str(response.get("message", {}).get("content", ""))


def process_run(run_id: str, store: RunStore, transcriber: Transcriber) -> None:
    run = store.update(run_id, status=RunStatus.TRANSCRIBING, error=None)
    try:
        result = transcriber.transcribe(
            store.audio_path(run.stored_filename), run.engine, run.keywords
        )
    except asyncio.CancelledError:
        store.update(run_id, status=RunStatus.FAILED, error="轉錄工作已取消")
        raise
    except Exception as error:
        store.update(run_id, status=RunStatus.FAILED, error=str(error))
        return
    result.revision_kind = TranscriptRevisionKind.RAW_ASR
    current = store.get(run_id)
    if current.raw_asr is None:
        store.update(
            run_id,
            status=RunStatus.COMPLETED,
            raw_asr=result,
            revisions=[result],
            result=result,
            error=None,
        )
    else:
        store.update(run_id, status=RunStatus.COMPLETED, error=None)


async def process_run_async(
    run_id: str,
    store: RunStore,
    transcriber: AsyncTranscriber,
    gpu_queue: GpuWorkQueue | None = None,
    ollama_client: AsyncOllamaClient | None = None,
) -> None:
    run = store.update(run_id, status=RunStatus.TRANSCRIBING, error=None)
    try:
        if run.engine == Engine.BREEZE and gpu_queue is not None:
            async with gpu_queue.acquire(
                task_name=f"breeze-transcribe:{run_id}",
                category=GpuCategory.SPEACHES,
                model="paulpengtw/faster-whisper-Breeze-ASR-26",
            ):
                result = await transcriber.transcribe(
                    store.audio_path(run.stored_filename), run.engine, run.keywords
                )
        else:
            result = await transcriber.transcribe(
                store.audio_path(run.stored_filename), run.engine, run.keywords
            )
    except asyncio.CancelledError:
        store.update(run_id, status=RunStatus.FAILED, error="轉錄工作已取消")
        raise
    except Exception as error:
        store.update(run_id, status=RunStatus.FAILED, error=str(error))
        return
    finally:
        close = getattr(transcriber, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                # 轉錄結果或原始錯誤比連線池清理錯誤更重要。
                pass

    result.revision_kind = TranscriptRevisionKind.RAW_ASR
    current = store.get(run_id)
    next_status = RunStatus.COMPLETED if not run.auto_summary else RunStatus.SUMMARIZING
    if current.raw_asr is None:
        run = store.update(
            run_id,
            status=next_status,
            raw_asr=result,
            revisions=[result],
            result=result,
            error=None,
        )
    else:
        run = store.update(run_id, status=next_status, error=None)

    if run.auto_summary and ollama_client is not None:
        await summarize_run_async(
            run_id,
            store,
            ollama_client,
            gpu_queue=gpu_queue,
            model=run.summary_model or "qwen3.5:9b",
        )


async def summarize_run_async(
    run_id: str,
    store: RunStore,
    ollama_client: AsyncOllamaClient,
    gpu_queue: GpuWorkQueue | None = None,
    model: str = "qwen3.5:9b",
    source_revision_id: str | None = None,
) -> None:
    run = store.get(run_id)
    source_transcript = None
    if source_revision_id:
        for rev in run.revisions:
            if rev.revision_id == source_revision_id:
                source_transcript = rev
                break
    if source_revision_id and source_transcript is None:
        store.update(
            run_id,
            status=RunStatus.COMPLETED,
            error=f"摘要失敗：找不到指定的逐字稿版本 {source_revision_id}",
        )
        return
    if source_transcript is None:
        source_transcript = run.result or run.raw_asr

    if not source_transcript:
        store.update(run_id, status=RunStatus.FAILED, error="無可用的逐字稿可供摘要")
        return

    store.update(run_id, status=RunStatus.SUMMARIZING, error=None)
    formatted = format_transcript_for_summary(source_transcript)

    try:
        if gpu_queue is not None:
            async with gpu_queue.acquire(
                task_name=f"ollama-summary:{run_id}",
                category=GpuCategory.OLLAMA,
                model=model,
            ):
                content = await _summarize_content(ollama_client, model, formatted)
        else:
            content = await _summarize_content(ollama_client, model, formatted)

        _assert_valid_summary(content, formatted)
        summary = parse_summary_payload(
            content, model=model, source_revision_id=source_transcript.revision_id
        )
        store.update(run_id, status=RunStatus.COMPLETED, summary=summary, error=None)
    except asyncio.CancelledError:
        store.update(run_id, status=RunStatus.COMPLETED, error="摘要工作已取消")
        raise
    except Exception as error:
        store.update(run_id, status=RunStatus.COMPLETED, error=f"摘要失敗：{error}")


def add_corrected_revision(
    run_id: str,
    store: RunStore,
    corrected_text: str,
    source_revision_id: str | None = None,
    corrected_segments: list[TranscriptSegment] | None = None,
) -> EvaluationRun:
    """新增人工校訂版逐字稿，保留原始 raw_asr 絕不覆蓋。"""
    run = store.get(run_id)
    if not run.raw_asr:
        raise ValueError("此工作尚未有原始 ASR 逐字稿")
    corrected_text = corrected_text.strip()
    if not corrected_text:
        raise ValueError("校訂文字不可為空")

    source = run.result or run.raw_asr
    if source_revision_id:
        source = next(
            (revision for revision in run.revisions if revision.revision_id == source_revision_id),
            None,
        )
        if source is None:
            raise ValueError(f"找不到來源逐字稿版本 {source_revision_id}")

    if corrected_segments is None:
        start_ms = source.segments[0].start_ms if source.segments else None
        end_ms = source.segments[-1].end_ms if source.segments else None
        corrected_segments = [
            TranscriptSegment(
                segment_id="seg-001",
                start_ms=start_ms,
                end_ms=end_ms,
                text=corrected_text,
            )
        ]

    corrected_rev = TranscriptResult(
        provider="user",
        model="manual-edit",
        revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
        source_revision_id=source.revision_id,
        text=corrected_text,
        detected_languages=source.detected_languages,
        segments=corrected_segments,
    )
    return store.append_revision(run_id, corrected_rev)


def rename_speaker_revision(
    run_id: str,
    store: RunStore,
    old_speaker: str,
    new_speaker: str,
    source_revision_id: str | None = None,
) -> EvaluationRun:
    """將指定逐字稿版本中的某講者更名，並另存為人工校訂版（human_edited），嚴格保留 raw_asr。"""
    run = store.get(run_id)
    if not run.raw_asr:
        raise ValueError("此工作尚未有原始 ASR 逐字稿")
    new_speaker = new_speaker.strip()
    if not new_speaker:
        raise ValueError("新講者名稱不可為空")

    source = run.result or run.raw_asr
    if source_revision_id:
        source = next(
            (rev for rev in run.revisions if rev.revision_id == source_revision_id),
            None,
        )
        if source is None:
            raise ValueError(f"找不到來源逐字稿版本 {source_revision_id}")

    target_old = old_speaker.strip()
    matched_count = 0
    new_segments: list[TranscriptSegment] = []
    for seg in source.segments:
        current_spk = (seg.speaker or "").strip()
        if current_spk == target_old or (not target_old and not current_spk):
            matched_count += 1
            new_segments.append(seg.model_copy(update={"speaker": new_speaker}))
        else:
            new_segments.append(seg.model_copy())

    if matched_count == 0:
        raise ValueError(f"在版本 {source.revision_id} 中找不到講者「{old_speaker}」的發言段落")

    new_rev = TranscriptResult(
        provider="user",
        model="manual-edit",
        revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
        source_revision_id=source.revision_id,
        text=source.text,
        detected_languages=source.detected_languages,
        segments=new_segments,
    )
    return store.append_revision(run_id, new_rev)


def update_segment_revision(
    run_id: str,
    store: RunStore,
    segment_id: str,
    corrected_text: str,
    speaker: str | None = None,
    source_revision_id: str | None = None,
) -> EvaluationRun:
    """單一段落快速校訂，保留其餘段落之時間軸與講者，並另存為 human_edited 新版本。"""
    run = store.get(run_id)
    if not run.raw_asr:
        raise ValueError("此工作尚未有原始 ASR 逐字稿")
    corrected_text = corrected_text.strip()
    if not corrected_text:
        raise ValueError("校訂文字不可為空")

    source = run.result or run.raw_asr
    if source_revision_id:
        source = next(
            (rev for rev in run.revisions if rev.revision_id == source_revision_id),
            None,
        )
        if source is None:
            raise ValueError(f"找不到來源逐字稿版本 {source_revision_id}")

    target_idx = next(
        (i for i, s in enumerate(source.segments) if s.segment_id == segment_id),
        None,
    )
    if target_idx is None:
        raise ValueError(f"找不到欲校訂的段落：{segment_id}")

    new_segments: list[TranscriptSegment] = []
    for i, seg in enumerate(source.segments):
        if i == target_idx:
            updated_speaker = (
                speaker.strip()
                if speaker is not None and speaker.strip()
                else seg.speaker
            )
            new_segments.append(
                seg.model_copy(
                    update={
                        "text": corrected_text,
                        "speaker": updated_speaker,
                    }
                )
            )
        else:
            new_segments.append(seg.model_copy())

    new_text = "\n".join(s.text for s in new_segments if s.text)
    corrected_rev = TranscriptResult(
        provider="user",
        model="manual-edit",
        revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
        source_revision_id=source.revision_id,
        text=new_text,
        detected_languages=source.detected_languages,
        segments=new_segments,
    )
    return store.append_revision(run_id, corrected_rev)
