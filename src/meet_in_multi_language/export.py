from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from .models import EvaluationRun, SummaryResult, TranscriptResult, TranscriptSegment
from .worker import RevisionNotFoundError


ExportFormat = Literal["txt", "srt", "vtt", "md", "json"]


def format_timestamp_srt(ms: int | None) -> str:
    if ms is None:
        ms = 0
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_timestamp_vtt(ms: int | None) -> str:
    if ms is None:
        ms = 0
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_timestamp_display(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def export_txt(run: EvaluationRun, revision: TranscriptResult) -> str:
    lines: list[str] = [
        "==================================================",
        f"會議錄音逐字稿：{run.original_filename}",
        f"轉錄引擎：{run.engine.value}",
        f"逐字稿版本：{revision.revision_id} ({revision.revision_kind.value})",
        f"匯出時間：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        "==================================================",
        "",
    ]
    if revision.segments:
        for seg in revision.segments:
            start_str = format_timestamp_display(seg.start_ms)
            end_str = format_timestamp_display(seg.end_ms)
            speaker_str = f"[{seg.speaker}] " if seg.speaker else ""
            lines.append(f"[{start_str} - {end_str}] {speaker_str}{seg.text}")
    else:
        lines.append(revision.text)
    lines.append("")
    return "\n".join(lines)


def export_srt(revision: TranscriptResult) -> str:
    lines: list[str] = []
    segments = revision.segments
    if not segments:
        if revision.text.strip():
            lines.extend(["1", "00:00:00,000 --> 00:00:10,000", revision.text.strip(), ""])
        return "\n".join(lines)

    for index, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_srt(seg.start_ms)
        end_ts = format_timestamp_srt(seg.end_ms if seg.end_ms is not None else (seg.start_ms or 0) + 3000)
        speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"{speaker_prefix}{seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def export_vtt(revision: TranscriptResult) -> str:
    lines: list[str] = ["WEBVTT", ""]
    segments = revision.segments
    if not segments:
        if revision.text.strip():
            lines.extend(["1", "00:00:00.000 --> 00:00:10.000", revision.text.strip(), ""])
        return "\n".join(lines)

    for index, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_vtt(seg.start_ms)
        end_ts = format_timestamp_vtt(seg.end_ms if seg.end_ms is not None else (seg.start_ms or 0) + 3000)
        voice_tag = f"<v {seg.speaker}>" if seg.speaker else ""
        lines.append(str(index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"{voice_tag}{seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def export_markdown(
    run: EvaluationRun,
    revision: TranscriptResult,
    summary: SummaryResult | None = None,
) -> str:
    now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    summary = summary or run.summary
    lines: list[str] = [
        f"# 會議逐字稿與摘要報告：{run.original_filename}",
        "",
        "- **原始音訊**：" + run.original_filename,
        f"- **轉錄引擎**：{run.engine.value}",
        f"- **逐字稿版本**：`{revision.revision_id}`（{revision.revision_kind.value}）",
        f"- **匯出時間**：{now_str}",
        "",
        "---",
        "",
    ]

    if summary:
        lines.extend([
            "## 一、會議摘要總覽",
            "",
            f"> 模型：`{summary.model}`（基準版本：`{summary.source_revision_id}`）",
            "",
            summary.overview,
            "",
        ])

        if summary.topics:
            lines.extend(["## 二、討論議題", ""])
            for t in summary.topics:
                evidence_str = f" *（來源：{', '.join(t.evidence_ids)}）*" if t.evidence_ids else ""
                lines.append(f"- **{t.title}**：{t.summary}{evidence_str}")
            lines.append("")

        if summary.decisions:
            lines.extend(["## 三、重要決議", ""])
            for d in summary.decisions:
                evidence_str = f" *（來源：{', '.join(d.evidence_ids)}）*" if d.evidence_ids else ""
                lines.append(f"- {d.text}{evidence_str}")
            lines.append("")

        if summary.action_items:
            lines.extend(["## 四、待辦事項", ""])
            for a in summary.action_items:
                owner_str = f" · 負責人：**{a.owner}**" if a.owner else ""
                due_str = f" · 期限：**{a.due_date or a.original_due_text}**" if (a.due_date or a.original_due_text) else ""
                evidence_str = f" *（來源：{', '.join(a.evidence_ids)}）*" if a.evidence_ids else ""
                lines.append(f"- [ ] **任務**：{a.task}{owner_str}{due_str}{evidence_str}")
            lines.append("")

        if summary.open_questions:
            lines.extend(["## 五、未解問題與待確認事項", ""])
            for q in summary.open_questions:
                evidence_str = f" *（來源：{', '.join(q.evidence_ids)}）*" if q.evidence_ids else ""
                lines.append(f"- {q.text}{evidence_str}")
            lines.append("")

        lines.extend(["---", ""])

    lines.extend([
        "## 逐字稿全文",
        "",
    ])

    if revision.segments:
        for seg in revision.segments:
            time_display = format_timestamp_display(seg.start_ms)
            speaker_display = f"**[{seg.speaker}]** " if seg.speaker else ""
            seg_tag = f"`{seg.segment_id}` "
            flags = f" *[警示: {', '.join(seg.quality_flags)}]*" if seg.quality_flags else ""
            lines.append(f"- [{time_display}] {seg_tag}{speaker_display}{seg.text}{flags}")
    else:
        lines.append(revision.text)

    lines.append("")
    return "\n".join(lines)


def export_payload(
    run: EvaluationRun,
    format_name: ExportFormat,
    revision_id: str | None = None,
) -> tuple[str, str, str]:
    revision = None
    if revision_id:
        revision = next(
            (rev for rev in run.revisions if rev.revision_id == revision_id),
            None,
        )
        if revision is None:
            raise RevisionNotFoundError(f"找不到指定的逐字稿版本：{revision_id}")
    else:
        revision = run.result or run.raw_asr

    if revision is None:
        raise ValueError("此工作尚未有可匯出之逐字稿內容")

    base_name = run.original_filename.rsplit(".", 1)[0]
    rev_suffix = f"_{revision.revision_kind.value}"

    if format_name == "txt":
        content = export_txt(run, revision)
        return content, "text/plain; charset=utf-8", f"{base_name}{rev_suffix}.txt"
    elif format_name == "srt":
        content = export_srt(revision)
        return content, "application/x-subrip; charset=utf-8", f"{base_name}{rev_suffix}.srt"
    elif format_name == "vtt":
        content = export_vtt(revision)
        return content, "text/vtt; charset=utf-8", f"{base_name}{rev_suffix}.vtt"
    elif format_name == "md":
        content = export_markdown(run, revision)
        return content, "text/markdown; charset=utf-8", f"{base_name}{rev_suffix}.md"
    elif format_name == "json":
        data = {
            "run_id": run.run_id,
            "original_filename": run.original_filename,
            "engine": run.engine.value,
            "status": run.status.value,
            "revision": revision.model_dump(mode="json"),
            "summary": run.summary.model_dump(mode="json") if run.summary else None,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"
        return content, "application/json; charset=utf-8", f"{base_name}{rev_suffix}.json"
    else:
        raise ValueError(f"不支援的匯出格式：{format_name}")
