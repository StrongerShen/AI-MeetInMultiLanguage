from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioToolError, probe_audio
from .config import Settings
from .export import export_payload
from .gpu import GpuWorkQueue
from .models import Engine, EvaluationRun, RunStatus, TranscriptResult
from .ollama_eval import AsyncOllamaClient
from .speaches import PROFILES, AsyncSpeachesTranscriber
from .storage import DeleteConflictError, RunNotFoundError, RunStore
from .transcription import AsyncOpenAITranscriber
from .worker import (
    MAX_MODEL_NAME_LENGTH,
    RevisionNotFoundError,
    SegmentNotFoundError,
    add_corrected_revision,
    process_run_async,
    rename_speaker_revision,
    summarize_run_async,
    update_segment_revision,
)


ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"


async def _probe_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(url)
            return response.is_success
    except httpx.HTTPError:
        return False


def create_app(
    settings: Settings | None = None,
    gpu_queue: GpuWorkQueue | None = None,
    ollama_client: AsyncOllamaClient | None = None,
    service_probe: Callable[[str], Awaitable[bool]] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = RunStore(settings.data_dir)
    gpu_queue = gpu_queue or GpuWorkQueue(
        settings.ollama_url,
        settings.speaches_url,
        settings.ollama_model,
        PROFILES["breeze"].model,
    )
    ollama_client = ollama_client or AsyncOllamaClient(settings.ollama_url)
    service_probe = service_probe or _probe_url

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(ollama_client, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(
        title="AI Meet in Multi-Language", version="0.1.0", lifespan=lifespan
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        html_content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html_content)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        speaches_health_url = settings.speaches_url.removesuffix("/v1") + "/health"
        ollama_health_url = settings.ollama_url.rstrip("/") + "/api/version"
        speaches_available, ollama_available = await asyncio.gather(
            service_probe(speaches_health_url), service_probe(ollama_health_url)
        )
        q_status = gpu_queue.status
        return {
            "status": "ok",
            "openai_configured": bool(settings.openai_api_key),
            "speaches_configured": bool(settings.speaches_url),
            "ollama_configured": bool(settings.ollama_url),
            "speaches_available": speaches_available,
            "ollama_available": ollama_available,
            "max_upload_bytes": settings.max_upload_bytes,
            "ollama_default_model": settings.ollama_model,
            "gpu_queue": {
                "is_busy": q_status.is_busy,
                "active_task": q_status.active_task,
                "active_category": q_status.active_category,
                "queue_length": q_status.queue_length,
                "completed_tasks": q_status.completed_tasks,
            },
        }

    @app.get("/api/runs", response_model=list[EvaluationRun])
    async def list_runs() -> list[EvaluationRun]:
        return store.list()

    @app.get("/api/runs/{run_id}", response_model=EvaluationRun)
    async def get_run(run_id: str) -> EvaluationRun:
        try:
            return store.get(run_id)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error

    @app.get("/api/runs/{run_id}/revisions", response_model=list[TranscriptResult])
    async def get_run_revisions(run_id: str) -> list[TranscriptResult]:
        try:
            run = store.get(run_id)
            return run.revisions
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error

    @app.post("/api/runs", response_model=EvaluationRun, status_code=202)
    async def create_run(
        background_tasks: BackgroundTasks,
        audio: UploadFile = File(...),
        engine: Engine = Form(Engine.DIARIZE),
        keywords: str = Form(""),
        auto_summary: bool = Form(False),
        summary_model: str = Form(""),
    ) -> EvaluationRun:
        if engine in (Engine.TRANSCRIBE, Engine.DIARIZE) and not settings.openai_api_key:
            raise HTTPException(
                status_code=503, detail="此轉錄方式需要伺服器環境設定 OPENAI_API_KEY"
            )

        if summary_model and len(summary_model.strip()) > MAX_MODEL_NAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"模型名稱長度超過限制（最大 {MAX_MODEL_NAME_LENGTH} 字元）",
            )

        original_filename = Path(audio.filename or "audio").name
        extension = Path(original_filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="不支援這個音訊格式")

        run_id = str(uuid4())
        stored_filename = f"{run_id}{extension}"
        destination = store.audio_path(stored_filename)
        written = 0
        try:
            with destination.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413, detail="音訊超過目前 25 MB 原型限制"
                        )
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await audio.close()

        if written == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="音訊檔案是空的")
        try:
            probe_audio(destination)
        except AudioToolError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="無法解碼這個音訊檔案") from error

        parsed_keywords = [
            item.strip() for item in keywords.replace("\r", "\n").split("\n") if item.strip()
        ]
        run = store.save(
            EvaluationRun(
                run_id=run_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                engine=engine,
                status=RunStatus.QUEUED,
                keywords=parsed_keywords,
                auto_summary=auto_summary,
                summary_model=summary_model or settings.ollama_model,
            )
        )

        if engine == Engine.BREEZE:
            transcriber = AsyncSpeachesTranscriber(
                settings.speaches_url, PROFILES["breeze"]
            )
        else:
            transcriber = AsyncOpenAITranscriber(settings.openai_api_key or "")

        background_tasks.add_task(
            process_run_async,
            run_id,
            store,
            transcriber,
            gpu_queue=gpu_queue,
            ollama_client=ollama_client,
        )
        return run

    @app.post("/api/runs/{run_id}/summary", response_model=EvaluationRun, status_code=202)
    async def request_summary(
        run_id: str,
        background_tasks: BackgroundTasks,
        model: str = Form(""),
        source_revision_id: str | None = Form(None),
    ) -> EvaluationRun:
        if model and len(model.strip()) > MAX_MODEL_NAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"模型名稱長度超過限制（最大 {MAX_MODEL_NAME_LENGTH} 字元）",
            )

        with store._acquire_lock():
            try:
                run = store.get(run_id)
            except RunNotFoundError as error:
                raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error

            if not (run.result or run.raw_asr):
                raise HTTPException(
                    status_code=400, detail="工作尚未完成轉錄，無法產生摘要"
                )

            if run.status in (RunStatus.TRANSCRIBING, RunStatus.SUMMARIZING):
                raise HTTPException(
                    status_code=409, detail="工作正在處理中，請稍候再產生摘要"
                )

            effective_model = model.strip() or run.summary_model or settings.ollama_model
            if source_revision_id and not any(
                revision.revision_id == source_revision_id for revision in run.revisions
            ):
                raise HTTPException(status_code=404, detail="找不到指定的逐字稿版本")

            # 原子更新狀態為 SUMMARIZING，防止並行重覆排入
            store.update(
                run_id,
                status=RunStatus.SUMMARIZING,
                summary_model=effective_model,
                error=None,
            )

        background_tasks.add_task(
            summarize_run_async,
            run_id,
            store,
            ollama_client,
            gpu_queue=gpu_queue,
            model=effective_model,
            source_revision_id=source_revision_id,
        )
        return store.get(run_id)

    @app.post("/api/runs/{run_id}/revisions/correct", response_model=EvaluationRun)
    async def correct_revision(
        run_id: str,
        corrected_text: str = Form(...),
        source_revision_id: str | None = Form(None),
    ) -> EvaluationRun:
        try:
            return add_corrected_revision(
                run_id, store, corrected_text, source_revision_id=source_revision_id
            )
        except (RunNotFoundError, RevisionNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/runs/{run_id}/segments/{segment_id}/correct", response_model=EvaluationRun)
    async def correct_segment(
        run_id: str,
        segment_id: str,
        corrected_text: str = Form(...),
        speaker: str | None = Form(None),
        source_revision_id: str | None = Form(None),
    ) -> EvaluationRun:
        try:
            return update_segment_revision(
                run_id,
                store,
                segment_id,
                corrected_text,
                speaker=speaker,
                source_revision_id=source_revision_id,
            )
        except (RunNotFoundError, RevisionNotFoundError, SegmentNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/runs/{run_id}/audio")
    async def get_run_audio(run_id: str) -> FileResponse:
        try:
            run = store.get(run_id)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error

        audio_path = store.audio_path(run.stored_filename)
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="找不到對應的音訊檔案")

        extension = audio_path.suffix.lower()
        media_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".mp4": "video/mp4",
            ".webm": "audio/webm",
        }
        media_type = media_types.get(extension, "application/octet-stream")
        return FileResponse(
            path=audio_path,
            filename=run.original_filename,
            media_type=media_type,
        )

    @app.get("/api/runs/{run_id}/export")
    async def export_run(
        run_id: str,
        format: str = Query("txt", pattern="^(txt|srt|vtt|md|json)$"),
        revision_id: str | None = Query(None),
    ) -> Response:
        try:
            run = store.get(run_id)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error

        try:
            content, media_type, filename = export_payload(
                run, format_name=format, revision_id=revision_id  # type: ignore[arg-type]
            )
        except (RevisionNotFoundError, RunNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        encoded_filename = quote(filename)
        headers = {
            "Content-Disposition": f'attachment; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        }
        return Response(content=content, media_type=media_type, headers=headers)

    @app.post("/api/runs/{run_id}/speakers/rename", response_model=EvaluationRun)
    async def rename_speaker(
        run_id: str,
        old_speaker: str = Form(...),
        new_speaker: str = Form(...),
        source_revision_id: str | None = Form(None),
    ) -> EvaluationRun:
        try:
            return rename_speaker_revision(
                run_id,
                store,
                old_speaker=old_speaker,
                new_speaker=new_speaker,
                source_revision_id=source_revision_id,
            )
        except (RunNotFoundError, RevisionNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/runs/{run_id}", status_code=204)
    async def delete_run(run_id: str) -> Response:
        try:
            store.safe_delete(run_id)
            return Response(status_code=204)
        except DeleteConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail="找不到這筆轉錄工作") from error


    return app


app = create_app()
