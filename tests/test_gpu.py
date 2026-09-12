import asyncio
import pytest

from meet_in_multi_language import gpu as gpu_module
from meet_in_multi_language.gpu import GpuCategory, GpuWorkQueue


def test_gpu_work_queue_mutual_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        execution_order: list[str] = []
        concurrency_counter = 0
        max_concurrency = 0

        async def fake_unload(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)
        monkeypatch.setattr(queue, "unload_speaches", fake_unload)

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


def test_gpu_work_queue_unloads_ollama_when_switching_to_speaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        unloaded_models: list[str] = []

        async def fake_unload(model: str) -> None:
            unloaded_models.append(model)

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)
        async with queue.acquire("ollama-job", GpuCategory.OLLAMA):
            pass

        # 進入 Speaches 前應卸載本專案設定的 Ollama 模型。
        async with queue.acquire("speaches-job", GpuCategory.SPEACHES):
            pass

        assert unloaded_models == ["qwen3.5:9b"]

    asyncio.run(run_test())


def test_gpu_work_queue_unloads_speaches_before_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        unloaded_models: list[str] = []

        async def fake_unload(model: str | None = None) -> None:
            unloaded_models.append(model or "")

        monkeypatch.setattr(queue, "unload_speaches", fake_unload)
        async with queue.acquire("transcription", GpuCategory.SPEACHES):
            pass
        async with queue.acquire("summary", GpuCategory.OLLAMA):
            pass

        assert unloaded_models == [""]

    asyncio.run(run_test())


def test_gpu_work_queue_cancellation_repairs_waiting_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        release_holder = asyncio.Event()
        holder_started = asyncio.Event()

        async def fake_unload(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)

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


def test_gpu_work_queue_releases_lock_after_task_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")

        async def fake_unload(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)

        with pytest.raises(RuntimeError, match="預期錯誤"):
            async with queue.acquire("failed", GpuCategory.SPEACHES):
                raise RuntimeError("預期錯誤")

        assert queue.status.is_busy is False
        async with queue.acquire("next", GpuCategory.SPEACHES):
            pass
        assert queue.status.completed_tasks == 2

    asyncio.run(run_test())


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
            {"model": "qwen3.5:9b", "keep_alive": 0},
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

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", Client)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")
    asyncio.run(queue.unload_speaches())

    assert requested_urls == [
        "http://speaches.test/api/ps/paulpengtw%2Ffaster-whisper-Breeze-ASR-26"
    ]


def test_gpu_work_queue_unloads_custom_ollama_model_when_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        unloaded_models: list[str] = []

        async def fake_unload(model: str) -> None:
            unloaded_models.append(model)

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)
        # 上一個 Ollama 工作明確使用 qwen3.8:latest
        async with queue.acquire("ollama-job", GpuCategory.OLLAMA, model="qwen3.8:latest"):
            pass

        # 切換到 Speaches 時，應精確卸載 qwen3.8:latest 而非預設的 qwen3.5:9b
        async with queue.acquire("speaches-job", GpuCategory.SPEACHES):
            pass

        assert unloaded_models == ["qwen3.8:latest"]

    asyncio.run(run_test())


def test_gpu_work_queue_unloads_previous_ollama_model_when_switching_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        queue = GpuWorkQueue("http://ollama.test")
        unloaded_models: list[str] = []

        async def fake_unload(model: str) -> None:
            unloaded_models.append(model)

        monkeypatch.setattr(queue, "unload_ollama", fake_unload)
        # 第一個任務使用 qwen3.5:9b
        async with queue.acquire("ollama-1", GpuCategory.OLLAMA, model="qwen3.5:9b"):
            pass

        # 第二個任務切換為 qwen3.8:latest，同屬 OLLAMA 類別但模型不同，應釋放前一個模型
        async with queue.acquire("ollama-2", GpuCategory.OLLAMA, model="qwen3.8:latest"):
            pass

        assert unloaded_models == ["qwen3.5:9b"]

    asyncio.run(run_test())


def test_speaches_unload_tolerates_connect_error_and_404(
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

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", Client404)
    queue = GpuWorkQueue(speaches_url="http://speaches.test/v1")
    # 404 表示模型本就未在 Speaches 顯存中，應順利返回不報錯
    asyncio.run(queue.unload_speaches())

    class ClientConnectError:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def delete(self, url: str):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(gpu_module.httpx, "AsyncClient", ClientConnectError)
    # ConnectError 表示 Speaches 未啟動，顯存無該模型，應順利返回不報錯
    asyncio.run(queue.unload_speaches())
