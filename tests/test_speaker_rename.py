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
