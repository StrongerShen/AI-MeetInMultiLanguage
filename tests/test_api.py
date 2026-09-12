import io
import asyncio
import wave
from pathlib import Path

import httpx

from meet_in_multi_language.config import Settings
from meet_in_multi_language.models import Engine, TranscriptResult


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    return output.getvalue()


def test_health_reports_missing_api_key(tmp_path: Path) -> None:
    from meet_in_multi_language.api import create_app

    app = create_app(Settings(tmp_path, 1024, None))
    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            homepage = await client.get("/")
            response = await client.get("/api/health")
            upload = await client.post(
                "/api/runs",
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
            return homepage, response, upload

    homepage, response, upload = asyncio.run(exercise_api())

    assert homepage.status_code == 200
    assert "多語會議逐字稿" in homepage.text
    assert response.json()["openai_configured"] is False
    assert upload.status_code == 503


def test_upload_runs_transcription_in_background(tmp_path: Path, monkeypatch) -> None:
    from meet_in_multi_language import api

    class FakeTranscriber:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        async def transcribe(
            self, audio_path: Path, engine: Engine, keywords: list[str]
        ) -> TranscriptResult:
            assert audio_path.is_file()
            assert keywords == ["專案名稱"]
            return TranscriptResult(model=engine.value, text="轉錄完成")

    monkeypatch.setattr(api, "AsyncOpenAITranscriber", FakeTranscriber)
    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/runs",
                data={"engine": Engine.TRANSCRIBE.value, "keywords": "專案名稱"},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
            run_id = response.json()["run_id"]
            saved = await client.get(f"/api/runs/{run_id}")
            return response, saved

    response, saved = asyncio.run(exercise_api())

    assert response.status_code == 202
    assert saved.json()["status"] == "completed"
    assert saved.json()["result"]["text"] == "轉錄完成"
