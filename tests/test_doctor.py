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


def test_check_host_memory(monkeypatch) -> None:
    # 1. 實際環境基本驗證
    res = check_host_memory()
    assert res["status"] in ("ok", "warning")
    assert "detail" in res

    # 2. 模擬總記憶體充足 (48 GiB) 但可用記憶體極低 (0.5 GiB) 的情境
    def fake_sysconf(name: str) -> int:
        page_size = 4096
        if name == "SC_PAGE_SIZE":
            return page_size
        elif name == "SC_PHYS_PAGES":
            return int((48 * (1024**3)) / page_size)
        elif name == "SC_AVPHYS_PAGES":
            return int((0.5 * (1024**3)) / page_size)
        return 0

    import os
    monkeypatch.setattr(os, "sysconf", fake_sysconf)

    low_res = check_host_memory()
    assert low_res["status"] == "warning"
    assert low_res["available_gib"] == 0.5
    assert "可用記憶體僅剩 0.5 GiB" in low_res["detail"]



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
    monkeypatch.setattr(
        "meet_in_multi_language.doctor.check_host_memory",
        lambda: {"status": "ok", "total_gib": 32.0, "available_gib": 16.0, "detail": "記憶體正常"},
    )


    diag = diagnose_system(tmp_path)
    assert diag["overall_status"] == "ok"

    report_text = format_doctor_report(diag)
    assert "AI Meet In Multi-Language 系統環境檢查報告" in report_text
    assert "✅ [正常]" in report_text
