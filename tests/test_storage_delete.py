"""儲存層刪除與安全性測試。

涵蓋：
- 正常刪除（紀錄 + 音訊）
- 路徑穿越與 symlink 防護
- 音訊刪除失敗時保留工作紀錄
- 重入鎖與並行更新
- 原子性安全刪除（TOCTOU 競爭防護）
- 狀態在檢查期間改變的並行測試
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

import pytest

from meet_in_multi_language.models import (
    Engine,
    EvaluationRun,
    RunStatus,
    TranscriptResult,
    TranscriptRevisionKind,
)
from meet_in_multi_language.storage import (
    DeleteConflictError,
    RunNotFoundError,
    RunStore,
)


def test_storage_delete_removes_record_and_audio(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    audio_file = store.audio_path("test-delete.wav")
    audio_file.write_bytes(b"dummy audio content")

    run = EvaluationRun(
        run_id="run-to-delete",
        original_filename="sample.wav",
        stored_filename="test-delete.wav",
        engine=Engine.BREEZE,
        status=RunStatus.COMPLETED,
    )
    store.save(run)

    assert (store.run_dir / "run-to-delete.json").is_file()
    assert audio_file.is_file()

    # 執行刪除
    store.delete("run-to-delete", delete_audio=True)

    assert not (store.run_dir / "run-to-delete.json").exists()
    assert not audio_file.exists()

    # 再次刪除或取得應拋出 RunNotFoundError
    with pytest.raises(RunNotFoundError):
        store.get("run-to-delete")

    with pytest.raises(RunNotFoundError):
        store.delete("run-to-delete")


def test_storage_audio_path_blocks_traversal(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    # 1. 拒絕包含路徑穿越字元
    with pytest.raises(ValueError, match="無效的檔案名稱|純檔名"):
        store.audio_path("../etc/passwd")

    with pytest.raises(ValueError, match="無效的檔案名稱|純檔名"):
        store.audio_path("sub/file.wav")

    with pytest.raises(ValueError, match="無效的檔案名稱|純檔名"):
        store.audio_path("..\\windows\\system32")

    with pytest.raises(ValueError, match="無效的檔案名稱|純檔名"):
        store.audio_path("/absolute/path.wav")

    # 2. 正常檔名可正確解析
    valid_path = store.audio_path("meeting.mp3")
    assert valid_path == (store.upload_dir / "meeting.mp3").resolve()

    # 3. 符號連結指向外部目錄時，解析後超出範圍必須被拒絕
    outside_file = tmp_path / "outside.wav"
    outside_file.write_text("secret", encoding="utf-8")
    symlink_file = store.upload_dir / "escaped_symlink.wav"
    symlink_file.symlink_to(outside_file)

    with pytest.raises(ValueError, match="路徑超出上傳目錄範圍"):
        store.audio_path("escaped_symlink.wav")


def test_storage_delete_audio_failure_preserves_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    run_file = store.run_dir / "run-preserve.json"
    run_file.write_text(
        '{"run_id": "run-preserve", "original_filename": "preserve.wav", "stored_filename": "preserve.wav", "engine": "breeze", "status": "completed", "revisions": []}',
        encoding="utf-8",
    )
    audio_file = store.upload_dir / "preserve.wav"
    audio_file.write_text("audio-data", encoding="utf-8")

    # 模擬音訊檔案刪除時發生 OSError
    def fail_unlink(*args, **kwargs) -> None:
        raise OSError("唯讀檔案系統")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="刪除音訊檔案失敗，已保留工作紀錄避免產生孤兒檔案"):
        store.delete("run-preserve", delete_audio=True)

    # 驗證 JSON 紀錄仍被完整保留
    assert run_file.exists()


def test_storage_reentrant_lock_and_concurrent_updates(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    base_rev = TranscriptResult(
        revision_id="rev-0",
        provider="test",
        model="test",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="原始",
        segments=[],
    )
    store.save(
        EvaluationRun(
            run_id="run-concurrent",
            original_filename="c.wav",
            stored_filename="c.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=base_rev,
            revisions=[base_rev],
            result=base_rev,
        )
    )

    # 1. 驗證巢狀重入鎖呼叫不會造成死鎖
    with store._acquire_lock():
        with store._acquire_lock():
            run = store.get("run-concurrent")
            assert run.run_id == "run-concurrent"

    # 2. 驗證並行追加修訂版（多執行緒／多程序情境）
    def add_rev(idx: int) -> None:
        new_rev = TranscriptResult(
            revision_id=f"rev-concurrent-{idx}",
            provider="test",
            model="test",
            revision_kind=TranscriptRevisionKind.HUMAN_EDITED,
            text=f"文字 {idx}",
            segments=[],
        )
        store.append_revision("run-concurrent", new_rev)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(add_rev, i) for i in range(1, 11)]
        concurrent.futures.wait(futures)
        for f in futures:
            f.result()

    final_run = store.get("run-concurrent")
    # 原始 1 個 + 追加 10 個 = 11 個版本，完全無 lost update
    assert len(final_run.revisions) == 11


# === 原子性安全刪除 (safe_delete) ===

def test_safe_delete_completed_run(tmp_path: Path) -> None:
    """safe_delete 在同一鎖內完成狀態檢查、音訊刪除、紀錄移除。"""
    store = RunStore(tmp_path)
    audio_file = store.audio_path("safe-delete.wav")
    audio_file.write_bytes(b"audio")

    store.save(
        EvaluationRun(
            run_id="run-safe-del",
            original_filename="safe.wav",
            stored_filename="safe-delete.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )
    store.safe_delete("run-safe-del")
    assert not audio_file.exists()
    with pytest.raises(RunNotFoundError):
        store.get("run-safe-del")


def test_safe_delete_rejects_active_statuses(tmp_path: Path) -> None:
    """safe_delete 拒絕刪除 queued、transcribing、summarizing 狀態的工作。"""
    store = RunStore(tmp_path)

    for status in (RunStatus.QUEUED, RunStatus.TRANSCRIBING, RunStatus.SUMMARIZING):
        run_id = f"run-active-{status.value}"
        store.save(
            EvaluationRun(
                run_id=run_id,
                original_filename="active.wav",
                stored_filename="active.wav",
                engine=Engine.BREEZE,
                status=status,
            )
        )
        with pytest.raises(DeleteConflictError, match="正在執行或排隊中"):
            store.safe_delete(run_id)
        # 紀錄應完整保留
        assert store.get(run_id).status == status


def test_safe_delete_audio_failure_preserves_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """safe_delete 音訊刪除失敗時保留工作紀錄。"""
    store = RunStore(tmp_path)
    audio_file = store.audio_path("safe-fail.wav")
    audio_file.write_bytes(b"audio")

    store.save(
        EvaluationRun(
            run_id="run-safe-fail",
            original_filename="fail.wav",
            stored_filename="safe-fail.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )

    def fail_unlink(*args, **kwargs) -> None:
        raise OSError("權限不足")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="刪除音訊檔案失敗"):
        store.safe_delete("run-safe-fail")

    # 紀錄應完整保留
    assert store.get("run-safe-fail").run_id == "run-safe-fail"


def test_safe_delete_toctou_race_condition(tmp_path: Path) -> None:
    """驗證 safe_delete 不存在 TOCTOU 競爭：

    模擬狀態在檢查期間從 completed 變為 transcribing。
    由於 safe_delete 在同一個鎖內完成所有操作，外部修改必須等鎖釋放後才能進行。
    """
    store = RunStore(tmp_path)
    audio_file = store.audio_path("toctou.wav")
    audio_file.write_bytes(b"audio")

    store.save(
        EvaluationRun(
            run_id="run-toctou",
            original_filename="toctou.wav",
            stored_filename="toctou.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )

    # 在另一個執行緒嘗試將狀態改為 TRANSCRIBING（模擬背景工作啟動）
    barrier = threading.Barrier(2, timeout=5)
    results: dict[str, object] = {"updater_blocked": False}

    def updater() -> None:
        """嘗試更新狀態。由於 safe_delete 持有鎖，更新必須等候。"""
        barrier.wait()  # 等待主執行緒也就緒
        try:
            store.update("run-toctou", status=RunStatus.TRANSCRIBING)
        except RunNotFoundError:
            # 如果 safe_delete 先完成，紀錄已不存在，預期行為
            results["updater_blocked"] = True

    t = threading.Thread(target=updater)
    t.start()
    barrier.wait()  # 同步起跑

    # 主執行緒執行 safe_delete
    store.safe_delete("run-toctou")

    t.join(timeout=5)

    # 驗證：工作已刪除或更新者因紀錄不存在而失敗
    with pytest.raises(RunNotFoundError):
        store.get("run-toctou")
