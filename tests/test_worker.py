import asyncio
import json
from pathlib import Path

import pytest

from meet_in_multi_language.models import (
    Engine,
    EvaluationRun,
    RunStatus,
    TranscriptResult,
)
from meet_in_multi_language.storage import RunStore
from meet_in_multi_language.storage import ImmutableRawAsrError
from meet_in_multi_language.worker import (
    _summarize_content,
    process_run,
    process_run_async,
    summarize_run_async,
)


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


def test_retry_does_not_replace_raw_asr_or_revisions(tmp_path: Path) -> None:
    class DifferentTranscriber:
        def transcribe(self, audio_path, engine, keywords):
            return TranscriptResult(model=engine.value, text="第二次辨識結果")

    store = RunStore(tmp_path)
    create_run(store)
    process_run("run-1", store, SuccessfulTranscriber())
    process_run("run-1", store, DifferentTranscriber())

    run = store.get("run-1")
    assert run.raw_asr is not None
    assert run.raw_asr.text == "測試完成"
    assert len(run.revisions) == 1


def test_store_rejects_direct_raw_asr_mutation(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    create_run(store)
    process_run("run-1", store, SuccessfulTranscriber())

    changed = store.get("run-1")
    assert changed.raw_asr is not None
    changed.raw_asr.text = "遭到竄改"
    with pytest.raises(ImmutableRawAsrError):
        store.save(changed)


def test_summary_rejects_unknown_evidence(tmp_path: Path) -> None:
    class InvalidOllama:
        async def summarize(self, model, transcript, keep_alive=0):
            payload = {
                "overview": "摘要",
                "topics": [],
                "decisions": [{"text": "不存在的引用", "evidence_ids": ["seg-999"]}],
                "action_items": [],
                "open_questions": [],
            }
            return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}

    store = RunStore(tmp_path)
    create_run(store)
    process_run("run-1", store, SuccessfulTranscriber())
    asyncio.run(summarize_run_async("run-1", store, InvalidOllama()))

    run = store.get("run-1")
    assert run.summary is None
    assert run.status == RunStatus.COMPLETED
    assert run.error is not None
    assert "不存在的段落" in run.error


def test_long_summary_is_processed_in_chunks() -> None:
    class RecordingOllama:
        def __init__(self) -> None:
            self.keep_alive_values: list[int | str] = []

        async def summarize(self, model, transcript, keep_alive=0):
            self.keep_alive_values.append(keep_alive)
            payload = {
                "overview": "分段摘要",
                "topics": [],
                "decisions": [{"text": "內容", "evidence_ids": ["seg-001"]}],
                "action_items": [],
                "open_questions": [],
            }
            return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}

    client = RecordingOllama()
    content = asyncio.run(
        _summarize_content(client, "qwen3.5:9b", "[seg-001] " + "字" * 13_000)
    )

    assert json.loads(content)["overview"] == "分段摘要"
    assert client.keep_alive_values == ["5m", "5m", "0m"]


def test_cancelled_async_transcription_updates_status(tmp_path: Path) -> None:
    class WaitingTranscriber:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def transcribe(self, audio_path, engine, keywords):
            self.started.set()
            await asyncio.Event().wait()

    async def exercise() -> None:
        store = RunStore(tmp_path)
        create_run(store)
        transcriber = WaitingTranscriber()
        task = asyncio.create_task(process_run_async("run-1", store, transcriber))
        await transcriber.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        run = store.get("run-1")
        assert run.status == RunStatus.FAILED
        assert run.error == "轉錄工作已取消"

    asyncio.run(exercise())


def test_store_rejects_run_id_with_path_traversal(tmp_path: Path) -> None:
    from meet_in_multi_language.storage import RunNotFoundError

    store = RunStore(tmp_path)
    with pytest.raises(RunNotFoundError, match="無效的工作識別碼"):
        store.get("../../etc/passwd")

    with pytest.raises(RunNotFoundError, match="無效的工作識別碼"):
        store.get("sub/dir")


def test_store_rejects_initial_raw_asr_without_revision_inclusion(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run = EvaluationRun(
        run_id="run-test",
        original_filename="test.mp3",
        stored_filename="test.mp3",
        engine=Engine.BREEZE,
        status=RunStatus.QUEUED,
        raw_asr=TranscriptResult(model="breeze", text="未包含在 revisions 的稿件"),
        revisions=[],
    )
    with pytest.raises(ImmutableRawAsrError, match="revisions 必須完整保留原始 raw_asr 版本"):
        store.save(run)


def test_add_corrected_revision_preserves_raw_and_sets_human_edited(tmp_path: Path) -> None:
    from meet_in_multi_language.models import TranscriptRevisionKind
    from meet_in_multi_language.worker import add_corrected_revision

    store = RunStore(tmp_path)
    create_run(store)
    process_run("run-1", store, SuccessfulTranscriber())

    updated_run = add_corrected_revision("run-1", store, "這是校訂版逐字稿")
    assert updated_run.raw_asr is not None
    assert updated_run.raw_asr.text == "測試完成"
    assert updated_run.raw_asr.revision_kind == TranscriptRevisionKind.RAW_ASR

    assert updated_run.result is not None
    assert updated_run.result.text == "這是校訂版逐字稿"
    assert updated_run.result.revision_kind == TranscriptRevisionKind.HUMAN_EDITED
    assert updated_run.result.source_revision_id == updated_run.raw_asr.revision_id
    assert len(updated_run.revisions) == 2
