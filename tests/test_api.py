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

    async def fake_probe(url: str) -> bool:
        return "8001" in url

    app = create_app(Settings(tmp_path, 1024, None), service_probe=fake_probe)
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
    assert response.json()["speaches_available"] is True
    assert response.json()["ollama_available"] is False
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
    assert saved.json()["raw_asr"]["text"] == "轉錄完成"
    assert len(saved.json()["revisions"]) == 1


def test_breeze_upload_runs_without_openai_key(tmp_path: Path, monkeypatch) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.gpu import GpuWorkQueue
    from meet_in_multi_language.models import TranscriptRevisionKind, TranscriptSegment

    class FakeBreezeTranscriber:
        def __init__(self, base_url: str, profile: object) -> None:
            pass

        async def transcribe(
            self, audio_path: Path, engine: Engine, keywords: list[str]
        ) -> TranscriptResult:
            assert audio_path.is_file()
            return TranscriptResult(
                provider="speaches",
                model="paulpengtw/faster-whisper-Breeze-ASR-26",
                revision_kind=TranscriptRevisionKind.RAW_ASR,
                text="今仔日開會討論多語逐字稿。",
                segments=[
                    TranscriptSegment(
                        segment_id="seg-001",
                        start_ms=0,
                        end_ms=2500,
                        text="今仔日開會討論多語逐字稿。",
                    )
                ],
            )

    monkeypatch.setattr(api, "AsyncSpeachesTranscriber", FakeBreezeTranscriber)
    gpu_queue = GpuWorkQueue()

    async def fake_unload(model: str) -> None:
        return None

    monkeypatch.setattr(gpu_queue, "unload_ollama", fake_unload)
    # 沒有設定 openai_api_key (None)
    app = api.create_app(Settings(tmp_path, 1024 * 1024, None), gpu_queue=gpu_queue)

    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/runs",
                data={"engine": Engine.BREEZE.value},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
            run_id = response.json()["run_id"]
            saved = await client.get(f"/api/runs/{run_id}")
            return response, saved

    response, saved = asyncio.run(exercise_api())

    assert response.status_code == 202
    run_data = saved.json()
    assert run_data["status"] == "completed"
    assert run_data["engine"] == "breeze"
    assert run_data["raw_asr"] is not None
    assert run_data["raw_asr"]["text"] == "今仔日開會討論多語逐字稿。"
    assert run_data["raw_asr"]["revision_kind"] == "raw_asr"
    assert len(run_data["revisions"]) == 1


def test_revisions_endpoint_and_raw_asr_immutability(tmp_path: Path, monkeypatch) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import TranscriptRevisionKind

    class FakeTranscriber:
        def __init__(self, api_key: str) -> None:
            pass

        async def transcribe(
            self, audio_path: Path, engine: Engine, keywords: list[str]
        ) -> TranscriptResult:
            return TranscriptResult(
                model=engine.value,
                revision_kind=TranscriptRevisionKind.RAW_ASR,
                text="原始轉錄文字（不得覆蓋）",
            )

    monkeypatch.setattr(api, "AsyncOpenAITranscriber", FakeTranscriber)
    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # 建立並完成初始 ASR 轉錄
            res = await client.post(
                "/api/runs",
                data={"engine": Engine.TRANSCRIBE.value},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
            run_id = res.json()["run_id"]

            # 建立人工校訂版
            correct_res = await client.post(
                f"/api/runs/{run_id}/revisions/correct",
                data={"corrected_text": "這是人工校訂後的文字"},
            )

            # 查詢所有修訂版本
            revs_res = await client.get(f"/api/runs/{run_id}/revisions")
            return res, correct_res, revs_res

    _, correct_res, revs_res = asyncio.run(exercise_api())

    assert correct_res.status_code == 200
    run_data = correct_res.json()

    # 核心驗證：raw_asr 絕對不被覆蓋，依舊保留原始 ASR 文字
    assert run_data["raw_asr"]["text"] == "原始轉錄文字（不得覆蓋）"
    assert run_data["raw_asr"]["revision_kind"] == "raw_asr"

    # 當前展示的 result 已切換為校訂版
    assert run_data["result"]["text"] == "這是人工校訂後的文字"
    assert run_data["result"]["revision_kind"] == "human_edited"
    assert run_data["result"]["provider"] == "user"
    assert run_data["result"]["source_revision_id"] == run_data["raw_asr"]["revision_id"]
    assert run_data["result"]["segments"][0]["text"] == "這是人工校訂後的文字"

    # 版本清單包含原始與校訂版
    revisions = revs_res.json()
    assert len(revisions) == 2
    assert revisions[0]["revision_kind"] == "raw_asr"
    assert revisions[1]["revision_kind"] == "human_edited"


def test_summary_endpoint_with_ollama(tmp_path: Path, monkeypatch) -> None:
    import json
    from meet_in_multi_language import api
    from meet_in_multi_language.gpu import GpuWorkQueue
    from meet_in_multi_language.models import TranscriptRevisionKind, TranscriptSegment

    fake_summary_json = {
        "overview": "這是本次會議的重點總覽。",
        "topics": [{"title": "多語逐字稿", "summary": "評測模型品質", "evidence_ids": ["seg-001"]}],
        "decisions": [{"text": "第一版採上傳後處理", "evidence_ids": ["seg-001"]}],
        "action_items": [{"task": "整理測試集", "owner": "Alice", "due_date": "2026-09-18", "original_due_text": "下週五", "evidence_ids": ["seg-001"]}],
        "open_questions": [{"text": "正式部署負責人待定", "evidence_ids": ["seg-001"]}],
    }

    class FakeTranscriber:
        def __init__(self, api_key: str) -> None:
            pass

        async def transcribe(self, audio_path: Path, engine: Engine, keywords: list[str]) -> TranscriptResult:
            return TranscriptResult(
                model=engine.value,
                revision_kind=TranscriptRevisionKind.RAW_ASR,
                text="第一版採上傳後處理。",
                segments=[TranscriptSegment(segment_id="seg-001", start_ms=0, end_ms=1000, text="第一版採上傳後處理。")],
            )

    class FakeOllama:
        def __init__(self, base_url: str) -> None:
            pass

        async def summarize(self, model: str, transcript: str, keep_alive: int | str = 0) -> dict[str, object]:
            assert "[seg-001]" in transcript
            return {"message": {"content": json.dumps(fake_summary_json, ensure_ascii=False)}}

        async def unload(self, model: str = "") -> None:
            pass

    monkeypatch.setattr(api, "AsyncOpenAITranscriber", FakeTranscriber)
    monkeypatch.setattr(api, "AsyncOllamaClient", FakeOllama)

    gpu_queue = GpuWorkQueue()

    async def fake_unload(model: str | None = None) -> None:
        return None

    monkeypatch.setattr(gpu_queue, "unload_speaches", fake_unload)

    app = api.create_app(
        Settings(tmp_path, 1024 * 1024, "test-key"), gpu_queue=gpu_queue
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/runs",
                data={"engine": Engine.TRANSCRIBE.value},
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )
            run_id = upload.json()["run_id"]

            invalid_source = await client.post(
                f"/api/runs/{run_id}/summary",
                data={"source_revision_id": "rev-does-not-exist"},
            )

            summary_res = await client.post(
                f"/api/runs/{run_id}/summary",
                data={"model": "qwen3.5:9b"},
            )
            run_after = await client.get(f"/api/runs/{run_id}")
            return invalid_source, summary_res, run_after

    invalid_source, summary_res, run_after = asyncio.run(exercise_api())

    assert invalid_source.status_code == 404
    assert summary_res.status_code == 202
    data = run_after.json()
    assert data["status"] == "completed"
    assert data["summary"] is not None
    assert data["summary"]["overview"] == "這是本次會議的重點總覽。"
    assert data["summary"]["decisions"][0]["text"] == "第一版採上傳後處理"
    assert data["summary"]["decisions"][0]["evidence_ids"] == ["seg-001"]
    assert data["summary"]["action_items"][0]["owner"] == "Alice"


def test_summary_endpoint_rejects_duplicate_request_when_in_progress(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    raw = TranscriptResult(model="gpt-transcribe", text="已有逐字稿")
    store.save(
        EvaluationRun(
            run_id="run-busy",
            original_filename="test.wav",
            stored_filename="test.wav",
            engine=Engine.TRANSCRIBE,
            status=RunStatus.SUMMARIZING,
            raw_asr=raw,
            revisions=[raw],
            result=raw,
        )
    )

    async def exercise_api() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/runs/run-busy/summary",
                data={"model": "qwen3.5:9b"},
            )

    response = asyncio.run(exercise_api())
    assert response.status_code == 409
    assert "正在處理中" in response.json()["detail"]


def test_async_openai_transcriber_aclose() -> None:
    from meet_in_multi_language.transcription import AsyncOpenAITranscriber

    closed = False

    class FakeAsyncClient:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    transcriber = AsyncOpenAITranscriber("test-key", client=FakeAsyncClient())
    asyncio.run(transcriber.aclose())
    assert closed is True
