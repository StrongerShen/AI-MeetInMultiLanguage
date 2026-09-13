"""全篇校訂段落結構測試。

涵蓋：
- 行數相同（精確逐段對應，保留 ID、時間戳、講者）
- 行數過少（嚴格拒絕）
- 行數過多（嚴格拒絕）
- 單一來源段落（仍接受單行或多行）
- raw_asr 不可變
"""
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
from meet_in_multi_language.worker import add_corrected_revision


def _setup_run_with_segments(tmp_path: Path, segments: list[TranscriptSegment]) -> tuple[RunStore, str]:
    """建立帶有指定段落的工作並返回 store 與 run_id。"""
    store = RunStore(tmp_path)
    run_id = "run-correct-test"
    text = "\n".join(s.text for s in segments)
    raw = TranscriptResult(
        revision_id="rev-raw",
        provider="test",
        model="test",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text=text,
        segments=segments,
    )
    store.save(
        EvaluationRun(
            run_id=run_id,
            original_filename="test.wav",
            stored_filename="test.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=raw,
            revisions=[raw],
            result=raw,
        )
    )
    return store, run_id


def test_correction_matching_line_count_preserves_structure(tmp_path: Path) -> None:
    """行數相同時精確逐段對應，完整保留 segment_id、時間戳、講者。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="原始第一段"),
        TranscriptSegment(segment_id="seg-002", start_ms=5000, end_ms=10000, speaker="B", text="原始第二段"),
        TranscriptSegment(segment_id="seg-003", start_ms=10000, end_ms=15000, speaker="A", text="原始第三段"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    corrected_text = "校訂第一段\n校訂第二段\n校訂第三段"
    result = add_corrected_revision(run_id, store, corrected_text)

    new_rev = result.result
    assert new_rev is not None
    assert new_rev.revision_kind == TranscriptRevisionKind.HUMAN_EDITED
    assert len(new_rev.segments) == 3

    # 驗證保留原始 ID、時間戳、講者
    assert new_rev.segments[0].segment_id == "seg-001"
    assert new_rev.segments[0].start_ms == 0
    assert new_rev.segments[0].end_ms == 5000
    assert new_rev.segments[0].speaker == "A"
    assert new_rev.segments[0].text == "校訂第一段"

    assert new_rev.segments[1].segment_id == "seg-002"
    assert new_rev.segments[1].start_ms == 5000
    assert new_rev.segments[1].end_ms == 10000
    assert new_rev.segments[1].speaker == "B"
    assert new_rev.segments[1].text == "校訂第二段"

    assert new_rev.segments[2].segment_id == "seg-003"
    assert new_rev.segments[2].start_ms == 10000
    assert new_rev.segments[2].end_ms == 15000
    assert new_rev.segments[2].speaker == "A"
    assert new_rev.segments[2].text == "校訂第三段"

    # source_revision_id 正確記錄
    assert new_rev.source_revision_id == "rev-raw"


def test_correction_fewer_lines_rejected(tmp_path: Path) -> None:
    """行數少於來源段落數時嚴格拒絕。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="第一段"),
        TranscriptSegment(segment_id="seg-002", start_ms=5000, end_ms=10000, speaker="B", text="第二段"),
        TranscriptSegment(segment_id="seg-003", start_ms=10000, end_ms=15000, speaker="A", text="第三段"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    # 只提供 2 行（來源有 3 段）
    with pytest.raises(ValueError, match="校訂行數.*與來源段落數.*不符"):
        add_corrected_revision(run_id, store, "校訂第一段\n校訂第二段")


def test_correction_more_lines_rejected(tmp_path: Path) -> None:
    """行數多於來源段落數時嚴格拒絕。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="第一段"),
        TranscriptSegment(segment_id="seg-002", start_ms=5000, end_ms=10000, speaker="B", text="第二段"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    # 提供 3 行（來源只有 2 段）
    with pytest.raises(ValueError, match="校訂行數.*與來源段落數.*不符"):
        add_corrected_revision(run_id, store, "校訂一\n校訂二\n校訂三")


def test_correction_single_source_segment_accepts_single_line(tmp_path: Path) -> None:
    """單一來源段落仍可接受單行校訂。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="原始單段"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    result = add_corrected_revision(run_id, store, "校訂單段")
    new_rev = result.result
    assert new_rev is not None
    assert len(new_rev.segments) == 1
    assert new_rev.segments[0].segment_id == "seg-001"
    assert new_rev.segments[0].start_ms == 0
    assert new_rev.segments[0].end_ms == 5000
    assert new_rev.segments[0].speaker == "A"
    assert new_rev.segments[0].text == "校訂單段"


def test_correction_single_source_segment_accepts_multiline(tmp_path: Path) -> None:
    """單一來源段落接受多行校訂（合併為單段）。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="原始文字"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    result = add_corrected_revision(run_id, store, "校訂第一行\n校訂第二行")
    new_rev = result.result
    assert new_rev is not None
    assert len(new_rev.segments) == 1
    assert "校訂第一行" in new_rev.segments[0].text
    assert "校訂第二行" in new_rev.segments[0].text


def test_correction_raw_asr_immutable(tmp_path: Path) -> None:
    """校訂後 raw_asr 永遠不可變。"""
    segments = [
        TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, speaker="A", text="原始"),
    ]
    store, run_id = _setup_run_with_segments(tmp_path, segments)

    result = add_corrected_revision(run_id, store, "校訂版")
    assert result.raw_asr is not None
    assert result.raw_asr.text == "原始"
    assert result.raw_asr.revision_kind == TranscriptRevisionKind.RAW_ASR
    assert result.raw_asr.segments[0].text == "原始"

    # raw_asr 仍在 revisions 中
    raw_in_revisions = next(
        (r for r in result.revisions if r.revision_id == "rev-raw"),
        None,
    )
    assert raw_in_revisions is not None
    assert raw_in_revisions.text == "原始"
