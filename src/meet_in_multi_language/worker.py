from __future__ import annotations

from .models import RunStatus
from .storage import RunStore
from .transcription import AsyncTranscriber, Transcriber


def process_run(run_id: str, store: RunStore, transcriber: Transcriber) -> None:
    run = store.update(run_id, status=RunStatus.TRANSCRIBING, error=None)
    try:
        result = transcriber.transcribe(
            store.audio_path(run.stored_filename), run.engine, run.keywords
        )
    except Exception as error:
        store.update(run_id, status=RunStatus.FAILED, error=str(error))
        return
    store.update(run_id, status=RunStatus.COMPLETED, result=result, error=None)


async def process_run_async(
    run_id: str, store: RunStore, transcriber: AsyncTranscriber
) -> None:
    run = store.update(run_id, status=RunStatus.TRANSCRIBING, error=None)
    try:
        result = await transcriber.transcribe(
            store.audio_path(run.stored_filename), run.engine, run.keywords
        )
    except Exception as error:
        store.update(run_id, status=RunStatus.FAILED, error=str(error))
        return
    store.update(run_id, status=RunStatus.COMPLETED, result=result, error=None)
