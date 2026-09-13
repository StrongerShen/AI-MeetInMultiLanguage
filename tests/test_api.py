import io
import asyncio
import wave
from pathlib import Path

import httpx
import pytest

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

    async def fake_get_loaded() -> set[str]:
        return set()

    monkeypatch.setattr(gpu_queue, "unload_ollama", fake_unload)
    monkeypatch.setattr(gpu_queue, "_get_loaded_ollama_models", fake_get_loaded)
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


def test_get_run_audio_endpoint(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    audio_path = store.audio_path("test-audio.wav")
    audio_path.write_bytes(wav_bytes())

    store.save(
        EvaluationRun(
            run_id="run-audio-test",
            original_filename="original.wav",
            stored_filename="test-audio.wav",
            engine=Engine.TRANSCRIBE,
            status=RunStatus.COMPLETED,
        )
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_found = await client.get("/api/runs/run-audio-test/audio")
            resp_missing = await client.get("/api/runs/non-existent/audio")
            return resp_found, resp_missing

    resp_found, resp_missing = asyncio.run(exercise_api())
    assert resp_found.status_code == 200
    assert resp_found.content == wav_bytes()
    assert resp_missing.status_code == 404


def test_get_run_audio_range_requests(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    data = wav_bytes()
    total_len = len(data)
    audio_path = store.audio_path("range-audio.wav")
    audio_path.write_bytes(data)

    store.save(
        EvaluationRun(
            run_id="run-range-test",
            original_filename="range.wav",
            stored_filename="range-audio.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. 前 10 bytes: bytes=0-9
            resp_part1 = await client.get(
                "/api/runs/run-range-test/audio", headers={"Range": "bytes=0-9"}
            )
            # 2. 中間範圍: bytes=10-19
            resp_part2 = await client.get(
                "/api/runs/run-range-test/audio", headers={"Range": "bytes=10-19"}
            )
            # 3. 超出範圍: bytes=999999-1000000 -> 應回傳 416
            resp_invalid = await client.get(
                "/api/runs/run-range-test/audio", headers={"Range": "bytes=999999-1000000"}
            )
            return resp_part1, resp_part2, resp_invalid

    resp_part1, resp_part2, resp_invalid = asyncio.run(exercise_api())

    # 驗證 206 與 Content-Range
    assert resp_part1.status_code == 206
    assert resp_part1.headers.get("content-range") == f"bytes 0-9/{total_len}"
    assert resp_part1.content == data[0:10]

    assert resp_part2.status_code == 206
    assert resp_part2.headers.get("content-range") == f"bytes 10-19/{total_len}"
    assert resp_part2.content == data[10:20]

    # 驗證 416
    assert resp_invalid.status_code == 416



def test_export_run_endpoint(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus, TranscriptSegment

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    rev = TranscriptResult(
        revision_id="rev-export",
        provider="breeze",
        model="breeze",
        text="測試文字",
        segments=[
            TranscriptSegment(
                segment_id="seg-001",
                start_ms=1000,
                end_ms=3000,
                speaker="SPEAKER_00",
                text="測試文字",
            )
        ],
    )
    store.save(
        EvaluationRun(
            run_id="run-export-test",
            original_filename="sample.mp3",
            stored_filename="sample.mp3",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=rev,
            revisions=[rev],
            result=rev,
        )
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            srt_resp = await client.get("/api/runs/run-export-test/export?format=srt")
            md_resp = await client.get("/api/runs/run-export-test/export?format=md")
            invalid_resp = await client.get("/api/runs/run-export-test/export?format=pdf")
            bad_rev_resp = await client.get("/api/runs/run-export-test/export?format=txt&revision_id=non-existent")
            return srt_resp, md_resp, invalid_resp, bad_rev_resp

    srt_resp, md_resp, invalid_resp, bad_rev_resp = asyncio.run(exercise_api())
    assert srt_resp.status_code == 200
    assert "00:00:01,000 --> 00:00:03,000" in srt_resp.text
    assert md_resp.status_code == 200
    assert "# 會議逐字稿與摘要報告" in md_resp.text
    assert invalid_resp.status_code == 422  # format query regex validation failed
    assert bad_rev_resp.status_code == 404
    assert "找不到指定的逐字稿版本" in bad_rev_resp.json()["detail"]



def test_rename_speaker_endpoint(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import (
        EvaluationRun,
        RunStatus,
        TranscriptRevisionKind,
        TranscriptSegment,
    )

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    rev = TranscriptResult(
        revision_id="rev-spk",
        provider="speaches",
        model="breeze",
        revision_kind=TranscriptRevisionKind.RAW_ASR,
        text="發言內容",
        segments=[
            TranscriptSegment(
                segment_id="seg-001",
                start_ms=1000,
                end_ms=3000,
                speaker="SPEAKER_00",
                text="發言內容",
            )
        ],
    )
    store.save(
        EvaluationRun(
            run_id="run-spk-test",
            original_filename="sample.mp3",
            stored_filename="sample.mp3",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=rev,
            revisions=[rev],
            result=rev,
        )
    )

    async def exercise_api() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/runs/run-spk-test/speakers/rename",
                data={"old_speaker": "SPEAKER_00", "new_speaker": "王董事長"},
            )

    resp = asyncio.run(exercise_api())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["revisions"]) == 2
    assert data["raw_asr"]["segments"][0]["speaker"] == "SPEAKER_00"  # raw_asr 絕對不變
    assert data["revisions"][1]["segments"][0]["speaker"] == "王董事長"
    assert data["result"]["segments"][0]["speaker"] == "王董事長"


def test_delete_run_endpoint(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)
    audio_file = store.audio_path("to-delete.wav")
    audio_file.write_bytes(wav_bytes())

    store.save(
        EvaluationRun(
            run_id="run-api-delete",
            original_filename="sample.wav",
            stored_filename="to-delete.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )

    store.save(
        EvaluationRun(
            run_id="run-running",
            original_filename="busy.wav",
            stored_filename="to-delete.wav",
            engine=Engine.BREEZE,
            status=RunStatus.TRANSCRIBING,
        )
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_conflict = await client.delete("/api/runs/run-running")
            resp_del = await client.delete("/api/runs/run-api-delete")
            resp_missing = await client.delete("/api/runs/non-existent-run")
            return resp_conflict, resp_del, resp_missing

    resp_conflict, resp_del, resp_missing = asyncio.run(exercise_api())
    assert resp_conflict.status_code == 409
    assert "正在執行或排隊中" in resp_conflict.json()["detail"]
    assert resp_del.status_code == 204
    assert not audio_file.exists()
    assert resp_missing.status_code == 404



def test_correct_segment_endpoint(tmp_path: Path) -> None:
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus, TranscriptResult, TranscriptSegment

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)

    rev = TranscriptResult(
        provider="test",
        model="breeze",
        text="第一段原始文字。\n第二段原始文字。",
        segments=[
            TranscriptSegment(
                segment_id="chunk-001-seg-1",
                start_ms=0,
                end_ms=5000,
                speaker="SPEAKER_00",
                text="第一段原始文字。",
            ),
            TranscriptSegment(
                segment_id="chunk-001-seg-2",
                start_ms=5000,
                end_ms=10000,
                speaker="SPEAKER_01",
                text="第二段原始文字。",
            ),
        ],
    )
    store.save(
        EvaluationRun(
            run_id="run-seg-correct-test",
            original_filename="sample.mp3",
            stored_filename="sample.mp3",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
            raw_asr=rev,
            revisions=[rev],
            result=rev,
        )
    )

    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_ok = await client.post(
                "/api/runs/run-seg-correct-test/segments/chunk-001-seg-1/correct",
                data={"corrected_text": "第一段校訂後的文字。", "speaker": "沈志中"},
            )
            resp_bad = await client.post(
                "/api/runs/run-seg-correct-test/segments/non-existent-seg/correct",
                data={"corrected_text": "文字"},
            )
            return resp_ok, resp_bad

    resp_ok, resp_bad = asyncio.run(exercise_api())
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert len(data["revisions"]) == 2
    # raw_asr 絕對不變
    assert data["raw_asr"]["segments"][0]["text"] == "第一段原始文字。"
    assert data["raw_asr"]["segments"][0]["speaker"] == "SPEAKER_00"
    # 新版校訂生效
    new_rev = data["result"]
    assert new_rev["revision_kind"] == "human_edited"
    assert new_rev["segments"][0]["text"] == "第一段校訂後的文字。"
    assert new_rev["segments"][0]["speaker"] == "沈志中"
    assert new_rev["segments"][0]["start_ms"] == 0
    assert new_rev["segments"][0]["end_ms"] == 5000
    # 第二段完全保留
    assert new_rev["segments"][1]["text"] == "第二段原始文字。"
    assert new_rev["segments"][1]["speaker"] == "SPEAKER_01"
    assert new_rev["segments"][1]["start_ms"] == 5000
    assert new_rev["segments"][1]["end_ms"] == 10000

    # 不存在的 segment_id 回傳 404
    assert resp_bad.status_code == 404
    assert "找不到欲校訂的段落" in resp_bad.json()["detail"]


def test_create_run_summary_model_128_chars_accepted(tmp_path: Path) -> None:
    """POST /api/runs 的 summary_model 128 字元接受。"""
    from meet_in_multi_language import api

    class FakeTranscriber:
        def __init__(self, api_key: str) -> None:
            pass

        async def transcribe(self, audio_path: Path, engine: Engine, keywords: list[str]) -> TranscriptResult:
            return TranscriptResult(model=engine.value, text="完成")

    monkeypatch_ctx = pytest.MonkeyPatch()
    monkeypatch_ctx.setattr(api, "AsyncOpenAITranscriber", FakeTranscriber)
    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))

    async def exercise_api() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/runs",
                data={
                    "engine": Engine.TRANSCRIBE.value,
                    "summary_model": "m" * 128,
                    "auto_summary": "false",
                },
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )

    import asyncio
    resp = asyncio.run(exercise_api())
    assert resp.status_code == 202
    assert resp.json()["summary_model"] == "m" * 128
    monkeypatch_ctx.undo()


def test_create_run_summary_model_129_chars_rejected(tmp_path: Path) -> None:
    """POST /api/runs 的 summary_model 129 字元回傳 400。"""
    from meet_in_multi_language import api
    from meet_in_multi_language.config import Settings

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))

    async def exercise_api() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/runs",
                data={
                    "engine": Engine.TRANSCRIBE.value,
                    "summary_model": "m" * 129,
                    "auto_summary": "false",
                },
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )

    import asyncio
    resp = asyncio.run(exercise_api())
    assert resp.status_code == 400
    assert "模型名稱長度超過限制" in resp.json()["detail"]


def test_create_run_summary_model_too_long_even_without_auto_summary(tmp_path: Path) -> None:
    """即使 auto_summary=false，也不得儲存超長模型名稱。"""
    from meet_in_multi_language import api
    from meet_in_multi_language.config import Settings

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))

    async def exercise_api() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/runs",
                data={
                    "engine": Engine.TRANSCRIBE.value,
                    "summary_model": "x" * 200,
                    "auto_summary": "false",
                },
                files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
            )

    import asyncio
    resp = asyncio.run(exercise_api())
    assert resp.status_code == 400


def test_delete_run_uses_atomic_safe_delete(tmp_path: Path) -> None:
    """驗證 DELETE /api/runs/{id} 使用原子性 safe_delete，409 由儲存層統一判定。"""
    from meet_in_multi_language import api
    from meet_in_multi_language.models import EvaluationRun, RunStatus
    from meet_in_multi_language.config import Settings

    app = api.create_app(Settings(tmp_path, 1024 * 1024, "test-key"))
    store = api.RunStore(tmp_path)

    # 排隊中的工作
    store.save(
        EvaluationRun(
            run_id="run-queued",
            original_filename="q.wav",
            stored_filename="q.wav",
            engine=Engine.BREEZE,
            status=RunStatus.QUEUED,
        )
    )

    # 已完成的工作
    audio_file = store.audio_path("done.wav")
    audio_file.write_bytes(wav_bytes())
    store.save(
        EvaluationRun(
            run_id="run-done",
            original_filename="done.wav",
            stored_filename="done.wav",
            engine=Engine.BREEZE,
            status=RunStatus.COMPLETED,
        )
    )

    import asyncio

    async def exercise_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_conflict = await client.delete("/api/runs/run-queued")
            resp_ok = await client.delete("/api/runs/run-done")
            resp_missing = await client.delete("/api/runs/non-existent")
            return resp_conflict, resp_ok, resp_missing

    resp_conflict, resp_ok, resp_missing = asyncio.run(exercise_api())
    assert resp_conflict.status_code == 409
    assert "正在執行或排隊中" in resp_conflict.json()["detail"]
    assert resp_ok.status_code == 204
    assert not audio_file.exists()
    assert resp_missing.status_code == 404
