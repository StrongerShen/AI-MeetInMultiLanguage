from __future__ import annotations

from pathlib import Path
import pytest

from meet_in_multi_language.models import (
    Engine,
    EvaluationRun,
    RunStatus,
    TranscriptResult,
    TranscriptRevisionKind,
    TranscriptSegment,
)
from meet_in_multi_language.storage import RunStore
from meet_in_multi_language.worker import rename_speaker_revision


def test_rename_speaker_creates_human_edited_revision_and_protects_raw(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    segments = [
        TranscriptSegment(
            segment_id="seg-001",
            start_ms=0,
            end_ms=2000,
            speaker="SPEAKER_00",
            text="第一句由 SPEAKER_00 發言",
        ),
        TranscriptSegment(
            segment_id="seg-002",
            start_ms=2100,
            end_ms=4000,
            speaker="SPEAKER_01",
            text="第二句由 SPEAKER_01 發言",
        ),
        TranscriptSegment(
            segment_id="seg-003",
            start_ms=4100,
            end_ms=6000,
            speaker="SPEAKER_00",
            text="第三句再次由 SPEAKER_00 發言",
        ),
    ]
    raw_rev = TranscriptResult(
        revision_id="rev-raw",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="完整逐字稿",
        segments=segments,
    )
    run = EvaluationRun(
        run_id="run-rename-test",
        original_filename="sample.wav",
        stored_filename="sample.wav",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
        raw_asr=raw_rev,
        revisions=[raw_rev],
        result=raw_rev,
    )
    store.save(run)

    # 執行更名：將 SPEAKER_00 改為「主席」
    updated_run = rename_speaker_revision(
        run_id="run-rename-test",
        store=store,
        old_speaker="SPEAKER_00",
        new_speaker="主席",
        source_revision_id="rev-raw",
    )

    # 驗證不可變性：raw_asr 與 revisions[0] 完全不變
    assert updated_run.raw_asr.revision_id == "rev-raw"
    assert updated_run.raw_asr.segments[0].speaker == "SPEAKER_00"
    assert updated_run.raw_asr.segments[2].speaker == "SPEAKER_00"
    assert len(updated_run.revisions) == 2

    # 驗證新修訂版
    new_rev = updated_run.revisions[1]
    assert new_rev.revision_kind == TranscriptRevisionKind.HUMAN_EDITED
    assert new_rev.source_revision_id == "rev-raw"
    assert new_rev.segments[0].speaker == "主席"
    assert new_rev.segments[1].speaker == "SPEAKER_01"  # 未更名者保留
    assert new_rev.segments[2].speaker == "主席"
    assert updated_run.result.revision_id == new_rev.revision_id


def test_rename_speaker_invalid_arguments(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    raw_rev = TranscriptResult(
        revision_id="rev-raw",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="測試內容",
        segments=[
            TranscriptSegment(segment_id="seg-001", speaker="SPEAKER_00", text="測試")
        ],
    )
    run = EvaluationRun(
        run_id="run-rename-test-2",
        original_filename="sample.wav",
        stored_filename="sample.wav",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
        raw_asr=raw_rev,
        revisions=[raw_rev],
        result=raw_rev,
    )
    store.save(run)

    # 空新講者名稱
    with pytest.raises(ValueError, match="新講者名稱不可為空"):
        rename_speaker_revision("run-rename-test-2", store, "SPEAKER_00", "   ")

    # 找不到舊講者
    with pytest.raises(ValueError, match="找不到講者"):
        rename_speaker_revision("run-rename-test-2", store, "NON_EXISTENT", "主席")

    # 找不到指定版本
    with pytest.raises(ValueError, match="找不到來源逐字稿版本"):
        rename_speaker_revision("run-rename-test-2", store, "SPEAKER_00", "主席", source_revision_id="rev-fake")


def test_add_corrected_revision_preserves_segment_structure(tmp_path: Path) -> None:
    from meet_in_multi_language.worker import add_corrected_revision

    store = RunStore(tmp_path)
    segments = [
        TranscriptSegment(
            segment_id="chunk-001-seg-1",
            start_ms=1000,
            end_ms=3000,
            speaker="主席",
            text="第一段話。",
        ),
        TranscriptSegment(
            segment_id="chunk-001-seg-2",
            start_ms=3500,
            end_ms=5500,
            speaker="專案經理",
            text="第二段話。",
        ),
    ]
    raw_rev = TranscriptResult(
        revision_id="rev-1",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="第一段話。\n第二段話。",
        segments=segments,
    )
    run = EvaluationRun(
        run_id="run-structure-test",
        original_filename="test.wav",
        stored_filename="test.wav",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
        raw_asr=raw_rev,
        revisions=[raw_rev],
        result=raw_rev,
    )
    store.save(run)

    # 執行多行全篇校訂
    updated_run = add_corrected_revision(
        run_id="run-structure-test",
        store=store,
        corrected_text="校訂後的第一段。\n校訂後的第二段。",
    )

    new_rev = updated_run.result
    assert len(new_rev.segments) == 2
    # 第一段時間戳與講者完整保留
    assert new_rev.segments[0].segment_id == "chunk-001-seg-1"
    assert new_rev.segments[0].start_ms == 1000
    assert new_rev.segments[0].end_ms == 3000
    assert new_rev.segments[0].speaker == "主席"
    assert new_rev.segments[0].text == "校訂後的第一段。"

    # 第二段時間戳與講者完整保留
    assert new_rev.segments[1].segment_id == "chunk-001-seg-2"
    assert new_rev.segments[1].start_ms == 3500
    assert new_rev.segments[1].end_ms == 5500
    assert new_rev.segments[1].speaker == "專案經理"
    assert new_rev.segments[1].text == "校訂後的第二段。"

    assert "保留 2 段結構" in new_rev.description


def test_input_length_limits_and_contract(tmp_path: Path) -> None:
    from meet_in_multi_language.worker import (
        MAX_FULL_TRANSCRIPT_LENGTH,
        MAX_SINGLE_SEGMENT_LENGTH,
        MAX_SPEAKER_NAME_LENGTH,
        SegmentNotFoundError,
        add_corrected_revision,
        update_segment_revision,
    )

    store = RunStore(tmp_path)
    seg = TranscriptSegment(segment_id="seg-1", start_ms=0, end_ms=1000, text="文字")
    raw_rev = TranscriptResult(
        revision_id="rev-1",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="文字",
        segments=[seg],
    )
    run = EvaluationRun(
        run_id="run-limits",
        original_filename="t.wav",
        stored_filename="t.wav",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
        raw_asr=raw_rev,
        revisions=[raw_rev],
        result=raw_rev,
    )
    store.save(run)

    # 1. 講者名稱長度超過限制
    with pytest.raises(ValueError, match="講者名稱長度超過限制"):
        rename_speaker_revision("run-limits", store, "文字", "A" * (MAX_SPEAKER_NAME_LENGTH + 1))

    # 2. 單段校訂文字長度超過限制
    with pytest.raises(ValueError, match="單段校訂文字長度超過限制"):
        update_segment_revision("run-limits", store, "seg-1", "B" * (MAX_SINGLE_SEGMENT_LENGTH + 1))

    # 3. 全篇校訂文字長度超過限制
    with pytest.raises(ValueError, match="校訂文字長度超過限制"):
        add_corrected_revision("run-limits", store, "C" * (MAX_FULL_TRANSCRIPT_LENGTH + 1))

    # 4. 不存在的段落拋出 SegmentNotFoundError
    with pytest.raises(SegmentNotFoundError, match="找不到欲校訂的段落"):
        update_segment_revision("run-limits", store, "seg-not-found", "校訂")
