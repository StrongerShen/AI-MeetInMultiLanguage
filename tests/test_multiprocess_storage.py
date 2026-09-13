"""真正的多程序儲存測試。

使用獨立程序及各自建立的 RunStore 實例，驗證：
- 多程序同時追加不同修訂版本
- 最終沒有 lost update
- JSON 始終可解析
- 不會遺留暫存檔
- 不會死鎖
- raw_asr 仍完全不變

測試設定合理逾時，失敗時不可永久卡住 pytest。
"""
from __future__ import annotations

import json
import multiprocessing
import os
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


def _worker_append_revision(data_dir_str: str, run_id: str, worker_id: int) -> str:
    """在獨立程序中建立自己的 RunStore 實例並追加一個修訂版本。"""
    store = RunStore(Path(data_dir_str))
    rev = TranscriptResult(
        revision_id=f"rev-proc-{worker_id}-{os.getpid()}",
        provider="test-multiprocess",
        model="test",
        revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
        text=f"多程序修訂文字 #{worker_id} (PID={os.getpid()})",
        segments=[
            TranscriptSegment(
                segment_id=f"seg-proc-{worker_id}",
                start_ms=0,
                end_ms=1000,
                text=f"段落 {worker_id}",
            )
        ],
    )
    try:
        store.append_revision(run_id, rev)
        return f"ok:{worker_id}"
    except Exception as e:
        return f"error:{worker_id}:{e}"


def test_multiprocess_concurrent_append_revisions(tmp_path: Path) -> None:
    """多程序同時追加不同修訂版本，驗證 lost update、JSON 完整性與暫存檔清理。"""
    store = RunStore(tmp_path)
    run_id = "run-multiproc"

    # 建立初始工作與 raw_asr
    raw_asr = TranscriptResult(
        revision_id="rev-raw-multiproc",
        provider="test",
        model="test",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="原始辨識文字",
        segments=[
            TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=5000, text="原始辨識文字")
        ],
    )
    store.save(
        EvaluationRun(
            run_id=run_id,
            original_filename="multiproc.wav",
            stored_filename="multiproc.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=raw_asr,
            revisions=[raw_asr],
            result=raw_asr,
        )
    )

    num_workers = 8

    # 使用 multiprocessing.Pool 啟動真正的獨立程序
    # 3. 獨立程序同時追加版本 (使用 apply_async 以支援逾時)
    with multiprocessing.Pool(processes=num_workers) as pool:
        async_results = [
            pool.apply_async(_worker_append_revision, (str(tmp_path), run_id, i))
            for i in range(1, num_workers + 1)
        ]

        try:
            results = [r.get(timeout=10) for r in async_results]
        except BaseException:
            pool.terminate()
            pool.join()
            raise

    # 驗證所有 worker 都成功
    for result in results:
        assert result.startswith("ok:"), f"Worker 失敗：{result}"

    # 驗證最終結果
    final_run = store.get(run_id)

    # 原始 1 個 + 追加 8 個 = 9 個版本，完全無 lost update
    assert len(final_run.revisions) == num_workers + 1, (
        f"預期 {num_workers + 1} 個版本，實際 {len(final_run.revisions)} 個（lost update）"
    )

    # raw_asr 仍完全不變
    assert final_run.raw_asr is not None
    assert final_run.raw_asr.revision_id == "rev-raw-multiproc"
    assert final_run.raw_asr.text == "原始辨識文字"
    assert final_run.raw_asr.revision_kind == TranscriptRevisionKind.RAW_ASR

    # raw_asr 仍在 revisions 中
    raw_in_revisions = next(
        (r for r in final_run.revisions if r.revision_id == "rev-raw-multiproc"),
        None,
    )
    assert raw_in_revisions is not None
    assert raw_in_revisions == final_run.raw_asr

    # JSON 始終可解析
    run_file = store.run_dir / f"{run_id}.json"
    parsed = json.loads(run_file.read_text(encoding="utf-8"))
    assert parsed["run_id"] == run_id
    assert len(parsed["revisions"]) == num_workers + 1

    # 不會遺留暫存檔
    tmp_files = list(store.run_dir.glob("*.tmp"))
    assert len(tmp_files) == 0, f"遺留暫存檔：{tmp_files}"


def _worker_read_write(data_dir_str: str, run_id: str, idx: int) -> str:
    """在獨立程序中讀取再追加。"""
    s = RunStore(Path(data_dir_str))
    try:
        # 讀取
        s.get(run_id)
        # 追加
        rev = TranscriptResult(
            revision_id=f"rev-dl-{idx}-{os.getpid()}",
            provider="test",
            model="test",
            revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
            text=f"dl-{idx}",
            segments=[],
        )
        s.append_revision(run_id, rev)
        return f"ok:{idx}"
    except Exception as e:
        return f"error:{idx}:{e}"


def test_multiprocess_no_deadlock(tmp_path: Path) -> None:
    """驗證多程序不會死鎖：快速連續取得與釋放鎖。"""
    store = RunStore(tmp_path)
    run_id = "run-deadlock-test"

    raw = TranscriptResult(
        revision_id="rev-dl-raw",
        provider="test",
        model="test",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="原始",
        segments=[],
    )
    store.save(
        EvaluationRun(
            run_id=run_id,
            original_filename="dl.wav",
            stored_filename="dl.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=raw,
            revisions=[raw],
            result=raw,
        )
    )

    # 多程序同時讀寫，5 秒逾時防止死鎖
    num = 5
    with multiprocessing.Pool(processes=num) as pool:
        async_results = [
            pool.apply_async(_worker_read_write, (str(tmp_path), run_id, i))
            for i in range(num)
        ]
        results = [r.get(timeout=10) for r in async_results]

    for r in results:
        assert r.startswith("ok:"), f"Worker 失敗或死鎖：{r}"

    final = store.get(run_id)
    assert len(final.revisions) == num + 1
