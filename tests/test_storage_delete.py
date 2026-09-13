from __future__ import annotations

from pathlib import Path
import pytest

from meet_in_multi_language.models import Engine, EvaluationRun, RunStatus
from meet_in_multi_language.storage import RunNotFoundError, RunStore


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
    import concurrent.futures
    from meet_in_multi_language.models import TranscriptResult, TranscriptRevisionKind

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
