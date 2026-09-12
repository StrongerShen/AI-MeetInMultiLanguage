from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from .audio import AudioToolError, probe_audio
from .config import Settings
from .models import Engine, EvaluationRun, RunStatus
from .storage import RunNotFoundError, RunStore
from .transcription import AsyncOpenAITranscriber
from .worker import process_run_async


ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"
STATIC_ASSETS = {
    "app.js": ("text/javascript; charset=utf-8", (STATIC_DIR / "app.js").read_text(encoding="utf-8")),
    "styles.css": ("text/css; charset=utf-8", (STATIC_DIR / "styles.css").read_text(encoding="utf-8")),
}
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = RunStore(settings.data_dir)
    app = FastAPI(title="AI Meet in Multi-Language", version="0.1.0")

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/static/{filename}", include_in_schema=False)
    async def static_asset(filename: str) -> Response:
        asset = STATIC_ASSETS.get(filename)
        if asset is None:
            raise HTTPException(status_code=404, detail="找不到這個靜態資源")
        media_type, content = asset
        return Response(content=content, media_type=media_type)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "openai_configured": bool(settings.openai_api_key),
            "max_upload_bytes": settings.max_upload_bytes,
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

    @app.post("/api/runs", response_model=EvaluationRun, status_code=202)
    async def create_run(
        background_tasks: BackgroundTasks,
        audio: UploadFile = File(...),
        engine: Engine = Form(Engine.DIARIZE),
        keywords: str = Form(""),
    ) -> EvaluationRun:
        if not settings.openai_api_key:
            raise HTTPException(status_code=503, detail="伺服器尚未設定 OPENAI_API_KEY")

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
                        raise HTTPException(status_code=413, detail="音訊超過目前 25 MB 原型限制")
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
            )
        )
        transcriber = AsyncOpenAITranscriber(settings.openai_api_key)
        background_tasks.add_task(process_run_async, run_id, store, transcriber)
        return run

    return app


app = create_app()
