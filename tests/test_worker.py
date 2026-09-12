from pathlib import Path

from meet_in_multi_language.models import (
    Engine,
    EvaluationRun,
    RunStatus,
    TranscriptResult,
)
from meet_in_multi_language.storage import RunStore
from meet_in_multi_language.worker import process_run


class SuccessfulTranscriber:
    def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult:
        return TranscriptResult(model=engine.value, text="測試完成")


class FailingTranscriber:
    def transcribe(
        self, audio_path: Path, engine: Engine, keywords: list[str]
    ) -> TranscriptResult:
        raise RuntimeError("測試錯誤")


def create_run(store: RunStore) -> EvaluationRun:
    store.audio_path("audio.wav").write_bytes(b"audio")
    return store.save(
        EvaluationRun(
            run_id="run-1",
            original_filename="audio.wav",
            stored_filename="audio.wav",
            engine=Engine.TRANSCRIBE,
            status=RunStatus.QUEUED,
        )
    )


def test_worker_saves_successful_result(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    create_run(store)

    process_run("run-1", store, SuccessfulTranscriber())

    run = store.get("run-1")
    assert run.status == RunStatus.COMPLETED
    assert run.result is not None
    assert run.result.text == "測試完成"


def test_worker_saves_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    create_run(store)

    process_run("run-1", store, FailingTranscriber())

    run = store.get("run-1")
    assert run.status == RunStatus.FAILED
    assert run.error == "測試錯誤"
