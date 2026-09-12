from __future__ import annotations

import json
from meet_in_multi_language.export import (
    export_markdown,
    export_payload,
    export_srt,
    export_txt,
    export_vtt,
    format_timestamp_display,
    format_timestamp_srt,
    format_timestamp_vtt,
)
from meet_in_multi_language.models import (
    ActionItem,
    DecisionItem,
    Engine,
    EvaluationRun,
    RunStatus,
    SummaryResult,
    TopicItem,
    TranscriptResult,
    TranscriptRevisionKind,
    TranscriptSegment,
)


def make_dummy_run() -> tuple[EvaluationRun, TranscriptResult, SummaryResult]:
    segments = [
        TranscriptSegment(
            segment_id="seg-001",
            start_ms=1250,
            end_ms=4500,
            speaker="SPEAKER_00",
            text="各位同仁早安，會議正式開始。",
            quality_flags=[],
        ),
        TranscriptSegment(
            segment_id="seg-002",
            start_ms=4800,
            end_ms=9200,
            speaker="SPEAKER_01",
            text="關於專案進度，我們預計本週五前完成測試。",
            quality_flags=["high_repetition"],
        ),
    ]
    raw_rev = TranscriptResult(
        revision_id="rev-001",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="各位同仁早安，會議正式開始。 關於專案進度，我們預計本週五前完成測試。",
        segments=segments,
    )
    summary = SummaryResult(
        model="qwen3.5:9b",
        source_revision_id="rev-001",
        overview="本次會議確認專案進度與測試時程。",
        topics=[
            TopicItem(title="專案進度", summary="討論測試時程安排", evidence_ids=["seg-001", "seg-002"])
        ],
        decisions=[
            DecisionItem(text="本週五前完成測試工作", evidence_ids=["seg-002"])
        ],
        action_items=[
            ActionItem(task="完成測試", owner="測試小組", due_date="2026-09-18", evidence_ids=["seg-002"])
        ],
        open_questions=[],
    )
    run = EvaluationRun(
        run_id="test-run-123",
        original_filename="meeting.mp3",
        stored_filename="test-run-123.mp3",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
        raw_asr=raw_rev,
        revisions=[raw_rev],
        result=raw_rev,
        summary=summary,
    )
    return run, raw_rev, summary


def test_timestamp_formatters() -> None:
    assert format_timestamp_srt(1250) == "00:00:01,250"
    assert format_timestamp_srt(3661005) == "01:01:01,005"
    assert format_timestamp_srt(None) == "00:00:00,000"

    assert format_timestamp_vtt(1250) == "00:00:01.250"
    assert format_timestamp_vtt(3661005) == "01:01:01.005"
    assert format_timestamp_vtt(None) == "00:00:00.000"

    assert format_timestamp_display(1250) == "00:01"
    assert format_timestamp_display(3661005) == "01:01:01"
    assert format_timestamp_display(None) == "--:--"


def test_export_txt() -> None:
    run, rev, _ = make_dummy_run()
    txt = export_txt(run, rev)
    assert "會議錄音逐字稿：meeting.mp3" in txt
    assert "[00:01 - 00:04] [SPEAKER_00] 各位同仁早安，會議正式開始。" in txt
    assert "[00:04 - 00:09] [SPEAKER_01] 關於專案進度，我們預計本週五前完成測試。" in txt


def test_export_srt() -> None:
    _, rev, _ = make_dummy_run()
    srt = export_srt(rev)
    assert "1\n00:00:01,250 --> 00:00:04,500\n[SPEAKER_00] 各位同仁早安，會議正式開始。" in srt
    assert "2\n00:00:04,800 --> 00:00:09,200\n[SPEAKER_01] 關於專案進度，我們預計本週五前完成測試。" in srt


def test_export_vtt() -> None:
    _, rev, _ = make_dummy_run()
    vtt = export_vtt(rev)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.250 --> 00:00:04.500" in vtt
    assert "<v SPEAKER_00>各位同仁早安，會議正式開始。" in vtt


def test_export_markdown() -> None:
    run, rev, summary = make_dummy_run()
    md = export_markdown(run, rev, summary)
    assert "# 會議逐字稿與摘要報告：meeting.mp3" in md
    assert "## 一、會議摘要總覽" in md
    assert "本次會議確認專案進度與測試時程。" in md
    assert "## 二、討論議題" in md
    assert "## 三、重要決議" in md
    assert "本週五前完成測試工作" in md
    assert "## 四、待辦事項" in md
    assert "測試小組" in md
    assert "seg-002" in md
    assert "## 逐字稿全文" in md
    assert "[SPEAKER_00]" in md


def test_export_payload_all_formats() -> None:
    run, rev, summary = make_dummy_run()

    for fmt in ("txt", "srt", "vtt", "md", "json"):
        content, media_type, filename = export_payload(run, fmt)  # type: ignore[arg-type]
        assert content
        assert "meeting_raw_asr" in filename
        if fmt == "json":
            payload = json.loads(content)
            assert payload["run_id"] == "test-run-123"
            assert payload["revision"]["revision_id"] == "rev-001"
            assert payload["summary"]["overview"] == summary.overview
