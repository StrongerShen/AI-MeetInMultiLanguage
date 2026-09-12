from __future__ import annotations

from pathlib import Path

from meet_in_multi_language.doctor import (
    check_binary_tools,
    check_host_memory,
    check_nvidia_gpu,
    check_ollama_service,
    check_speaches_service,
    check_storage_directory,
    diagnose_system,
    format_doctor_report,
)


def test_check_host_memory() -> None:
    res = check_host_memory()
    assert res["status"] in ("ok", "warning")
    assert "detail" in res


def test_check_binary_tools() -> None:
    res = check_binary_tools()
    assert "ffmpeg" in res
    assert "ffprobe" in res
    assert res["status"] in ("ok", "error")


def test_check_storage_directory(tmp_path: Path) -> None:
    res = check_storage_directory(tmp_path)
    assert res["status"] in ("ok", "warning")
    assert res["writable"] is True
    assert "free_gib" in res


def test_check_speaches_service_mock(monkeypatch) -> None:
    import httpx

    class FakeResponse:
        def __init__(self, status_code: int, json_data: object = None) -> None:
            self.status_code = status_code
            self._json = json_data or {}
            self.is_success = 200 <= status_code < 300

        def json(self) -> object:
            return self._json

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str) -> FakeResponse:
            if "health" in url:
                return FakeResponse(200)
            if "models" in url:
                return FakeResponse(200, {"data": [{"id": "paulpengtw/faster-whisper-Breeze-ASR-26"}]})
            return FakeResponse(404)

    monkeypatch.setattr(httpx, "Client", FakeClient)

    res = check_speaches_service("http://fake-speaches:8001/v1")
    assert res["status"] == "ok"
    assert res["available"] is True
    assert res["breeze_ready"] is True


def test_check_ollama_service_mock(monkeypatch) -> None:
    import httpx

    class FakeResponse:
        def __init__(self, status_code: int, json_data: object = None) -> None:
            self.status_code = status_code
            self._json = json_data or {}
            self.is_success = 200 <= status_code < 300

        def json(self) -> object:
            return self._json

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str) -> FakeResponse:
            if "version" in url:
                return FakeResponse(200, {"version": "0.3.0"})
            if "tags" in url:
                return FakeResponse(200, {"models": [{"name": "qwen3.5:9b"}]})
            return FakeResponse(404)

    monkeypatch.setattr(httpx, "Client", FakeClient)

    res = check_ollama_service("http://fake-ollama:11434", target_model="qwen3.5:9b")
    assert res["status"] == "ok"
    assert res["target_ready"] is True


def test_diagnose_system_and_format(tmp_path: Path, monkeypatch) -> None:
    # 模擬全正常環境
    monkeypatch.setattr(
        "meet_in_multi_language.doctor.check_speaches_service",
        lambda *args, **kwargs: {"status": "ok", "detail": "Speaches 正常"},
    )
    monkeypatch.setattr(
        "meet_in_multi_language.doctor.check_ollama_service",
        lambda *args, **kwargs: {"status": "ok", "detail": "Ollama 正常"},
    )
    monkeypatch.setattr(
        "meet_in_multi_language.doctor.check_nvidia_gpu",
        lambda: {"status": "ok", "detail": "RTX 3050 8G 正常"},
    )

    diag = diagnose_system(tmp_path)
    assert diag["overall_status"] == "ok"

    report_text = format_doctor_report(diag)
    assert "AI Meet In Multi-Language 系統環境檢查報告" in report_text
    assert "✅ [正常]" in report_text
