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
