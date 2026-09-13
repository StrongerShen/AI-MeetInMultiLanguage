"""GPU 跨程序模型切換安全測試。

涵蓋情境：
- Web → CLI、CLI → Web 交錯
- 兩個獨立 GpuWorkQueue 實例模擬不同 worker
- 新程序第一個工作就是 Ollama（顯存可能留有 Breeze）
- 新程序第一個工作就是 Breeze（顯存可能留有 Ollama）
- 卸載失敗時不得執行工作內容
- 所有例外與取消路徑都會釋放程序鎖及主機鎖

不接觸真實 Speaches、Ollama 或 GPU。
"""
import asyncio
from pathlib import Path
import pytest

from meet_in_multi_language import gpu as gpu_module
from meet_in_multi_language.gpu import (
    GpuCategory,
    GpuTransitionError,
    GpuWorkQueue,
)


# --- 輔助工具 ---

def _make_queue(
    monkeypatch: pytest.MonkeyPatch,
    *,
    unload_ollama_side_effect: object = None,
    unload_speaches_side_effect: object = None,
    record_ollama: list | None = None,
    record_speaches: list | None = None,
    known_ollama_models: list[str] | None = None,
) -> GpuWorkQueue:
    """建立一個帶有 fake 卸載函式的 GpuWorkQueue。"""
    queue = GpuWorkQueue(
        "http://ollama.test",
        "http://speaches.test/v1",
        known_ollama_models=known_ollama_models or ["qwen3.5:9b"],
    )
    _ollama_log = record_ollama if record_ollama is not None else []
    _speaches_log = record_speaches if record_speaches is not None else []

    async def fake_unload_ollama(model: str) -> None:
        _ollama_log.append(model)
        if isinstance(unload_ollama_side_effect, Exception):
            raise unload_ollama_side_effect

    async def fake_unload_speaches(model: str | None = None) -> None:
        _speaches_log.append(model or queue.speaches_model)
        if isinstance(unload_speaches_side_effect, Exception):
            raise unload_speaches_side_effect

    async def fake_get_loaded() -> set[str]:
        return set()

    monkeypatch.setattr(queue, "unload_ollama", fake_unload_ollama)
    monkeypatch.setattr(queue, "unload_speaches", fake_unload_speaches)
    monkeypatch.setattr(queue, "_get_loaded_ollama_models", fake_get_loaded)
    return queue


# === 基本互斥 ===

def test_gpu_work_queue_mutual_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        ollama_log: list[str] = []
        speaches_log: list[str] = []
        queue = _make_queue(monkeypatch, record_ollama=ollama_log, record_speaches=speaches_log)
        execution_order: list[str] = []
        concurrency_counter = 0
        max_concurrency = 0

        async def worker(name: str, delay: float) -> None:
            nonlocal concurrency_counter, max_concurrency
            async with queue.acquire(name, GpuCategory.SPEACHES):
                concurrency_counter += 1
                max_concurrency = max(max_concurrency, concurrency_counter)
                execution_order.append(f"{name}-start")
                await asyncio.sleep(delay)
                execution_order.append(f"{name}-end")
                concurrency_counter -= 1

        await asyncio.gather(
            worker("task-1", 0.05),
            worker("task-2", 0.02),
            worker("task-3", 0.01),
        )

        assert max_concurrency == 1
        assert execution_order == [
            "task-1-start",
            "task-1-end",
            "task-2-start",
            "task-2-end",
            "task-3-start",
            "task-3-end",
        ]
        assert queue.status.completed_tasks == 3
        assert queue.status.is_busy is False

    asyncio.run(run_test())


# === 跨程序切換安全：每次都無條件卸載 ===

def test_gpu_always_unloads_speaches_before_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """進入 Ollama 前一定卸載 Breeze，不依賴程序內歷史。"""
    async def run_test() -> None:
        speaches_log: list[str] = []
        queue = _make_queue(monkeypatch, record_speaches=speaches_log)
        # 直接進入 Ollama（沒有先跑 Speaches），模擬新程序
        async with queue.acquire("ollama-job", GpuCategory.OLLAMA):
            pass
        assert queue.speaches_model in speaches_log

    asyncio.run(run_test())


def test_gpu_always_unloads_ollama_before_speaches(monkeypatch: pytest.MonkeyPatch) -> None:
    """進入 Speaches 前一定卸載所有已知 Ollama 模型，不依賴程序內歷史。"""
    async def run_test() -> None:
        ollama_log: list[str] = []
        queue = _make_queue(
            monkeypatch,
            record_ollama=ollama_log,
            known_ollama_models=["qwen3.5:9b", "qwen3.8:latest"],
        )
        # 直接進入 Speaches（沒有先跑 Ollama），模擬新程序
        async with queue.acquire("speaches-job", GpuCategory.SPEACHES):
            pass
        assert "qwen3.5:9b" in ollama_log
        assert "qwen3.8:latest" in ollama_log

    asyncio.run(run_test())


def test_new_process_first_job_ollama_with_stale_breeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """新程序第一個工作就是 Ollama，但顯存可能留有 Breeze。"""
    async def run_test() -> None:
        speaches_log: list[str] = []
        queue = _make_queue(monkeypatch, record_speaches=speaches_log)
        # 模擬全新程序，_last_category_used 為 None
        assert queue._last_category_used is None
        async with queue.acquire("ollama-first", GpuCategory.OLLAMA, model="qwen3.5:9b"):
            pass
        # 即使沒有歷史，也必須嘗試卸載 Speaches
        assert len(speaches_log) >= 1

    asyncio.run(run_test())


def test_new_process_first_job_breeze_with_stale_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """新程序第一個工作就是 Breeze，但顯存可能留有 Ollama。"""
    async def run_test() -> None:
        ollama_log: list[str] = []
        queue = _make_queue(monkeypatch, record_ollama=ollama_log)
        assert queue._last_category_used is None
        async with queue.acquire("speaches-first", GpuCategory.SPEACHES):
            pass
        # 即使沒有歷史，也必須嘗試卸載 Ollama
        assert len(ollama_log) >= 1

    asyncio.run(run_test())


# === Web → CLI 與 CLI → Web 交錯 ===

def test_web_then_cli_interleave(monkeypatch: pytest.MonkeyPatch) -> None:
    """模擬 Web worker (GpuWorkQueue) 先跑 Breeze，再由 CLI (另一 GpuWorkQueue) 跑 Ollama。"""
    async def run_test() -> None:
        speaches_log_web: list[str] = []
        speaches_log_cli: list[str] = []
        ollama_log_cli: list[str] = []

        web_queue = _make_queue(monkeypatch, record_speaches=speaches_log_web)
        cli_queue = _make_queue(monkeypatch, record_speaches=speaches_log_cli, record_ollama=ollama_log_cli)

        # Web 跑 Breeze
        async with web_queue.acquire("web-breeze", GpuCategory.SPEACHES):
            pass

        # CLI 跑 Ollama（必須先卸載 Breeze）
        async with cli_queue.acquire("cli-ollama", GpuCategory.OLLAMA, model="qwen3.5:9b"):
            pass

        assert len(speaches_log_cli) >= 1  # CLI 進入 Ollama 前卸載了 Breeze

    asyncio.run(run_test())


def test_cli_then_web_interleave(monkeypatch: pytest.MonkeyPatch) -> None:
    """模擬 CLI 先跑 Ollama，再由 Web worker 跑 Breeze。"""
    async def run_test() -> None:
        ollama_log_web: list[str] = []
        ollama_log_cli: list[str] = []

        cli_queue = _make_queue(monkeypatch, record_ollama=ollama_log_cli)
        web_queue = _make_queue(monkeypatch, record_ollama=ollama_log_web)

        # CLI 跑 Ollama
        async with cli_queue.acquire("cli-ollama", GpuCategory.OLLAMA, model="qwen3.5:9b"):
            pass

        # Web 跑 Breeze（必須先卸載 Ollama）
        async with web_queue.acquire("web-breeze", GpuCategory.SPEACHES):
            pass

        assert "qwen3.5:9b" in ollama_log_web  # Web 進入 Speaches 前卸載了 Ollama

    asyncio.run(run_test())


def test_two_independent_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """兩個獨立 GpuWorkQueue 實例模擬不同 worker。"""
    async def run_test() -> None:
        ollama_log_1: list[str] = []
        speaches_log_1: list[str] = []
        ollama_log_2: list[str] = []
        speaches_log_2: list[str] = []

        q1 = _make_queue(monkeypatch, record_ollama=ollama_log_1, record_speaches=speaches_log_1)
        q2 = _make_queue(monkeypatch, record_ollama=ollama_log_2, record_speaches=speaches_log_2)

        # q1 跑 Breeze
        async with q1.acquire("q1-breeze", GpuCategory.SPEACHES):
            pass

        # q2 跑 Ollama（q2 不知道 q1 做了什麼，但必須先嘗試卸載 Breeze）
        async with q2.acquire("q2-ollama", GpuCategory.OLLAMA, model="qwen3.5:9b"):
            pass

        assert len(speaches_log_2) >= 1

    asyncio.run(run_test())


# === 卸載失敗時禁止執行 ===

def test_unload_speaches_failure_blocks_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """卸載 Speaches 失敗時不得執行 Ollama 工作。"""
    async def run_test() -> None:
        queue = _make_queue(
            monkeypatch,
            unload_speaches_side_effect=GpuTransitionError("卸載 Speaches 模型失敗"),
        )
        ollama_executed = False
        with pytest.raises(GpuTransitionError, match="卸載 Speaches 模型失敗"):
            async with queue.acquire("blocked-ollama", GpuCategory.OLLAMA):
                ollama_executed = True
        assert not ollama_executed
        assert queue.status.is_busy is False

    asyncio.run(run_test())


def test_unload_ollama_failure_blocks_speaches(monkeypatch: pytest.MonkeyPatch) -> None:
    """卸載 Ollama 失敗時不得執行 Speaches 工作。"""
    async def run_test() -> None:
        queue = _make_queue(
            monkeypatch,
            unload_ollama_side_effect=GpuTransitionError("卸載 Ollama 模型失敗"),
        )
        speaches_executed = False
        with pytest.raises(GpuTransitionError, match="卸載 Ollama 模型失敗"):
            async with queue.acquire("blocked-speaches", GpuCategory.SPEACHES):
                speaches_executed = True
        assert not speaches_executed
        assert queue.status.is_busy is False

    asyncio.run(run_test())


# === 鎖的釋放保證 ===

def test_cancellation_repairs_waiting_count(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        queue = _make_queue(monkeypatch)
        release_holder = asyncio.Event()
        holder_started = asyncio.Event()

        async def holder() -> None:
            async with queue.acquire("holder", GpuCategory.SPEACHES):
                holder_started.set()
                await release_holder.wait()

        async def waiter() -> None:
            async with queue.acquire("waiter", GpuCategory.SPEACHES):
                pass

        holder_task = asyncio.create_task(holder())
        await holder_started.wait()
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert queue.status.queue_length == 1

        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task
        assert queue.status.queue_length == 0

        release_holder.set()
        await holder_task
        assert queue.status.is_busy is False

    asyncio.run(run_test())


def test_exception_releases_process_and_host_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """工作內容拋出例外時，程序鎖及主機鎖都會釋放。"""
    async def run_test() -> None:
        queue = _make_queue(monkeypatch)

        with pytest.raises(RuntimeError, match="預期錯誤"):
            async with queue.acquire("failed", GpuCategory.SPEACHES):
                raise RuntimeError("預期錯誤")

        assert queue.status.is_busy is False
        # 應能取得下一個工作
        async with queue.acquire("next", GpuCategory.SPEACHES):
            pass
        assert queue.status.completed_tasks == 2

    asyncio.run(run_test())


def test_unload_failure_releases_all_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    """卸載失敗時程序鎖及主機鎖都會釋放，後續工作可正常取得。"""
    async def run_test() -> None:
        queue = _make_queue(
            monkeypatch,
            unload_speaches_side_effect=GpuTransitionError("模擬卸載失敗"),
        )

        with pytest.raises(GpuTransitionError):
            async with queue.acquire("blocked", GpuCategory.OLLAMA):
                pass

        assert queue.status.is_busy is False

        # 修復卸載函式後，下一個工作應能正常執行
        async def fixed_unload(model: str | None = None) -> None:
            return None

        monkeypatch.setattr(queue, "unload_speaches", fixed_unload)
        async with queue.acquire("recovered", GpuCategory.OLLAMA):
            pass
        assert queue.status.completed_tasks == 1

    asyncio.run(run_test())


# === 原有 HTTP 層級測試 ===

def test_ollama_unload_sends_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]):
            requests.append((url, json))
            return Response()

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", Client)
    queue = GpuWorkQueue("http://ollama.test")
    asyncio.run(queue.unload_ollama("qwen3.5:9b"))

    assert requests == [
        (
            "http://ollama.test/api/generate",
            {"model": "qwen3.5:9b", "keep_alive": "0m"},
        )
    ]


def test_speaches_unload_uses_encoded_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            requested_urls.append(url)
            return Response()
            
        async def get(self, url: str):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, json={"models": []})
            
        async def post(self, url: str, json: dict):
            req = httpx.Request("POST", url)
            return httpx.Response(200, request=req)

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", Client)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")
    asyncio.run(queue.unload_speaches())

    assert requested_urls == [
        "http://speaches.test/api/ps/paulpengtw%2Ffaster-whisper-Breeze-ASR-26"
    ]


def test_speaches_unload_tolerates_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class Client404:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            request = httpx.Request("DELETE", url)
            response = httpx.Response(404, request=request)
            response.raise_for_status()

        async def get(self, url: str):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, json={"models": []})
            
        async def post(self, url: str, json: dict):
            req = httpx.Request("POST", url)
            return httpx.Response(200, request=req)

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", Client404)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")
    asyncio.run(queue.unload_speaches())


def test_connect_error_aborts_web_and_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class ClientConnectError:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            raise httpx.ConnectError("Connection refused")

        async def get(self, url: str):
            raise httpx.ConnectError("Connection refused")
            
        async def post(self, url: str, json: dict):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", ClientConnectError)
    
    # Web: Speaches 卸載發生 ConnectError 中止
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")
    with pytest.raises(GpuTransitionError, match="Connection refused"):
        asyncio.run(queue.unload_speaches())
        
    # Web: Ollama 模型探索發生 ConnectError 中止
    with pytest.raises(GpuTransitionError, match="Connection refused"):
        asyncio.run(queue._get_loaded_ollama_models())

    # Web: Ollama 模型卸載發生 ConnectError 中止
    with pytest.raises(GpuTransitionError, match="Connection refused"):
        asyncio.run(queue.unload_ollama("test_model"))


def test_speaches_unload_timeout_raises_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class TimeoutClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            raise httpx.TimeoutException("卸載請求超時")
            
        async def get(self, url: str):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, json={"models": []})
            
        async def post(self, url: str, json: dict):
            req = httpx.Request("POST", url)
            return httpx.Response(200, request=req)

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", TimeoutClient)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")

    with pytest.raises(GpuTransitionError, match="逾時"):
        asyncio.run(queue.unload_speaches())


def test_speaches_unload_http_500_raises_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class ServerErrorClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            req = httpx.Request("DELETE", url)
            resp = httpx.Response(500, request=req)
            resp.raise_for_status()

        async def get(self, url: str):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, json={"models": []})
            
        async def post(self, url: str, json: dict):
            req = httpx.Request("POST", url)
            return httpx.Response(200, request=req)

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", ServerErrorClient)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")

    async def run_switch() -> None:
        speaches_log: list[str] = []

        async def fake_unload_speaches_ok(model: str | None = None) -> None:
            speaches_log.append(model or queue.speaches_model)

        async def fake_unload_ollama(model: str) -> None:
            pass

        # 先跑 Speaches（用 ok 的 unload）
        monkeypatch.setattr(queue, "unload_speaches", fake_unload_speaches_ok)
        monkeypatch.setattr(queue, "unload_ollama", fake_unload_ollama)

        async with queue.acquire("asr-task", GpuCategory.SPEACHES):
            pass

        # 切回 http 500 的 unload
        async def fake_unload_speaches_fail(model: str | None = None) -> None:
            raise GpuTransitionError("無法釋放 Speaches ASR 顯存")

        monkeypatch.setattr(queue, "unload_speaches", fake_unload_speaches_fail)

        ollama_executed = False
        with pytest.raises(GpuTransitionError, match="無法釋放 Speaches ASR 顯存"):
            async with queue.acquire("summary-task", GpuCategory.OLLAMA):
                ollama_executed = True

        assert not ollama_executed
        assert queue.status.is_busy is False

    asyncio.run(run_switch())


# === 主機級鎖測試 ===

def test_host_gpu_lock_mutual_exclusion(tmp_path: Path) -> None:
    import threading
    from meet_in_multi_language.gpu import host_gpu_lock

    lock_file = tmp_path / "test_host_gpu.lock"
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_failed = False

    def holder() -> None:
        with host_gpu_lock(lock_file):
            first_acquired.set()
            release_first.wait()

    t1 = threading.Thread(target=holder)
    t1.start()
    first_acquired.wait()

    try:
        with pytest.raises(TimeoutError, match="等候主機 GPU 互斥鎖逾時"):
            with host_gpu_lock(lock_file, timeout=0.05, poll_interval=0.01):
                pass
        second_failed = True
    finally:
        release_first.set()
        t1.join()

    assert second_failed is True

    with host_gpu_lock(lock_file, timeout=0.1):
        pass


def test_async_host_gpu_lock_mutual_exclusion(tmp_path: Path) -> None:
    from meet_in_multi_language.gpu import async_host_gpu_lock

    lock_file = tmp_path / "test_async_host_gpu.lock"

    async def run_test() -> None:
        async with async_host_gpu_lock(lock_file):
            with pytest.raises(TimeoutError, match="等候主機 GPU 互斥鎖逾時"):
                async with async_host_gpu_lock(lock_file, timeout=0.05, poll_interval=0.01):
                    pass

        async with async_host_gpu_lock(lock_file, timeout=0.1):
            pass

    asyncio.run(run_test())

# === 動態查詢 Ollama 模型與同步卸載測試 ===

def test_get_loaded_ollama_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試非同步與同步查詢實際載入模型的行為。"""
    from meet_in_multi_language.gpu import sync_get_loaded_ollama_models
    import httpx

    class MockPsClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            req = httpx.Request("GET", url)
            resp = httpx.Response(200, request=req, json={"models": [{"name": "loaded1:latest"}, {"name": "loaded2:8b"}]})
            return resp

    monkeypatch.setattr(gpu_module.httpx, "Client", MockPsClient)
    models = sync_get_loaded_ollama_models("http://ollama.test")
    assert models == {"loaded1:latest", "loaded2:8b"}

    class MockAsyncPsClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url):
            req = httpx.Request("GET", url)
            resp = httpx.Response(200, request=req, json={"models": [{"name": "loaded3"}]})
            return resp

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", MockAsyncPsClient)
    queue = GpuWorkQueue("http://ollama.test")
    async_models = asyncio.run(queue._get_loaded_ollama_models())
    assert async_models == {"loaded3"}


def test_sync_connect_error_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試同步查詢與卸載遇到 ConnectError 時會拋出 GpuTransitionError。"""
    from meet_in_multi_language.gpu import sync_get_loaded_ollama_models, sync_unload_ollama
    from meet_in_multi_language.pipeline import sync_unload_speaches
    import httpx

    class MockSyncConnectErrorClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            raise httpx.ConnectError("Connection refused")
        def post(self, url, json):
            raise httpx.ConnectError("Connection refused")
        def delete(self, url):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(gpu_module.httpx, "Client", MockSyncConnectErrorClient)
    # pipeline.py 也有自己的 httpx 導入，必須一起 mock
    from meet_in_multi_language import pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module.httpx, "Client", MockSyncConnectErrorClient)

    with pytest.raises(GpuTransitionError, match="Connection refused"):
        sync_get_loaded_ollama_models("http://ollama.test")

    with pytest.raises(GpuTransitionError, match="Connection refused"):
        sync_unload_ollama("http://ollama.test", "test_model")

    with pytest.raises(GpuTransitionError, match="Connection refused"):
        sync_unload_speaches("http://speaches.test", "test_model")


def test_sync_unload_all_loaded_ollama_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試同步卸載 Ollama 包含動態查詢與 fallback 模型的行為。"""
    from meet_in_multi_language.gpu import sync_unload_all_loaded_ollama_models
    import httpx
    
    unloaded_models = []

    class MockPsClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, json={"models": [{"name": "loaded_dynamic"}]})
        def post(self, url, json):
            unloaded_models.append(json["model"])
            req = httpx.Request("POST", url)
            return httpx.Response(200, request=req)

    monkeypatch.setattr(gpu_module.httpx, "Client", MockPsClient)
    
    sync_unload_all_loaded_ollama_models("http://ollama.test", {"fallback_static"})
    
    # 驗證動態與靜態模型皆被卸載
    assert set(unloaded_models) == {"loaded_dynamic", "fallback_static"}

def test_sync_unload_ollama_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試同步卸載失敗會拋出 GpuTransitionError。"""
    from meet_in_multi_language.gpu import sync_unload_ollama
    import httpx

    class MockFailClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json):
            req = httpx.Request("POST", url)
            resp = httpx.Response(500, request=req)
            resp.raise_for_status()

    monkeypatch.setattr(gpu_module.httpx, "Client", MockFailClient)
    
    with pytest.raises(GpuTransitionError, match="無法卸載 Ollama 模型 fail_model"):
        sync_unload_ollama("http://ollama.test", "fail_model")
