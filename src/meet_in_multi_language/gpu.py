from __future__ import annotations

import asyncio
import fcntl
import os
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator
from urllib.parse import quote

import httpx

DEFAULT_GPU_LOCK_FILE = Path(
    os.getenv(
        "AI_MEET_GPU_LOCK_FILE",
        str(Path(tempfile.gettempdir()) / "ai_meet_gpu.lock"),
    )
)

# 本專案可能使用的 Ollama 模型清單（用於跨程序安全卸載）
KNOWN_OLLAMA_MODELS: list[str] = [
    "qwen3.5:9b",
    "qwen3.8:latest",
    "gemma4:12b",
    "muse-glimmer:latest",
]


@contextmanager
def host_gpu_lock(
    lock_file: Path | str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """主機跨行程 GPU 互斥鎖（基於 fcntl.flock）。

    確保 CLI、Web worker 或多程序環境下，同一時間只有一個程序進入 GPU 敏感區段。
    """
    path = Path(lock_file) if lock_file is not None else DEFAULT_GPU_LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if timeout is not None and (time.monotonic() - start) >= timeout:
                    raise TimeoutError(f"等候主機 GPU 互斥鎖逾時（超過 {timeout} 秒）：{path}")
                time.sleep(poll_interval)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


@asynccontextmanager
async def async_host_gpu_lock(
    lock_file: Path | str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 0.05,
) -> AsyncIterator[None]:
    """非同步主機跨行程 GPU 互斥鎖，以非阻塞輪詢配合 asyncio.sleep，避免阻塞 event loop。"""
    path = Path(lock_file) if lock_file is not None else DEFAULT_GPU_LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if timeout is not None and (time.monotonic() - start) >= timeout:
                    raise TimeoutError(f"等候主機 GPU 互斥鎖逾時（超過 {timeout} 秒）：{path}")
                await asyncio.sleep(poll_interval)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


class GpuCategory(StrEnum):
    SPEACHES = "speaches"
    OLLAMA = "ollama"


class GpuTransitionError(RuntimeError):
    """GPU 模型無法安全切換。"""


def sync_get_loaded_ollama_models(ollama_url: str) -> set[str]:
    """同步查詢 Ollama 實際載入顯存的模型。"""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{ollama_url.rstrip('/')}/api/ps")
            resp.raise_for_status()
            data = resp.json()
            return {m["name"] for m in data.get("models", [])}
    except httpx.ConnectError:
        return set()
    except Exception as err:
        raise GpuTransitionError(f"無法查詢 Ollama 載入狀態：{err}") from err

def sync_unload_ollama(ollama_url: str, model: str) -> None:
    """同步卸載指定的 Ollama 模型。"""
    if not model:
        return
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={"model": model, "keep_alive": 0},
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        pass
    except httpx.HTTPStatusError as err:
        if err.response.status_code != 404:
            raise GpuTransitionError(f"無法卸載 Ollama 模型 {model}：{err}") from err
    except Exception as err:
        raise GpuTransitionError(f"無法卸載 Ollama 模型 {model}：{err}") from err

def sync_unload_all_loaded_ollama_models(ollama_url: str, fallback_models: set[str]) -> None:
    """同步卸載所有 Ollama 模型（先嘗試查詢實際載入的，再加入 fallback 名單）。"""
    models_to_unload = sync_get_loaded_ollama_models(ollama_url) | fallback_models
    for m in models_to_unload:
        sync_unload_ollama(ollama_url, m)

@dataclass
class GpuQueueStatus:
    is_busy: bool
    active_task: str | None
    active_category: GpuCategory | None
    queue_length: int
    last_category_used: GpuCategory | None
    completed_tasks: int


class GpuWorkQueue:
    """單一行程內的 GPU 互斥工作佇列與模型切換守門員。

    跨程序安全策略：不依賴程序內歷史狀態（如 _last_category_used）。
    每次進入 Ollama 工作前，都會嘗試卸載 Breeze ASR 模型。
    每次進入 Speaches 工作前，都會動態查詢並嘗試卸載所有 Ollama 模型。
    卸載逾時、HTTP 失敗或狀態未知時，禁止開始下一模型。
    """

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        speaches_url: str = "http://127.0.0.1:8001/v1",
        ollama_model: str = "qwen3.5:9b",
        speaches_model: str = "paulpengtw/faster-whisper-Breeze-ASR-26",
        gpu_lock_file: Path | str | None = None,
        known_ollama_models: list[str] | None = None,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.speaches_url = speaches_url.rstrip("/")
        self.ollama_model = ollama_model
        self.speaches_model = speaches_model
        self.gpu_lock_file = gpu_lock_file
        
        # 將傳入的已知模型與設定的 Ollama 模型皆納入防護網
        self.known_ollama_models = set(KNOWN_OLLAMA_MODELS)
        if known_ollama_models:
            self.known_ollama_models.update(known_ollama_models)
        if self.ollama_model:
            self.known_ollama_models.add(self.ollama_model)
            
        self._lock = asyncio.Lock()
        self._waiting_count = 0
        self._active_task: str | None = None
        self._active_category: GpuCategory | None = None
        self._last_category_used: GpuCategory | None = None
        self._completed_tasks = 0

    @property
    def status(self) -> GpuQueueStatus:
        return GpuQueueStatus(
            is_busy=self._lock.locked(),
            active_task=self._active_task,
            active_category=self._active_category,
            queue_length=self._waiting_count,
            last_category_used=self._last_category_used,
            completed_tasks=self._completed_tasks,
        )

    async def _get_loaded_ollama_models(self) -> set[str]:
        """動態查詢 Ollama 實際載入顯存的模型。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/ps")
                response.raise_for_status()
                data = response.json()
                return {m["name"] for m in data.get("models", [])}
        except httpx.ConnectError:
            return set()
        except Exception as err:
            raise GpuTransitionError(f"無法查詢 Ollama 載入狀態：{err}") from err

    async def unload_ollama(self, model: str) -> None:
        """只卸載本工作明確指定的 Ollama 模型。"""
        if not model:
            raise GpuTransitionError("未指定要卸載的 Ollama 模型")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": model, "keep_alive": 0},
                )
                response.raise_for_status()
        except httpx.ConnectError:
            # 服務未啟動時不可能有 Ollama 模型佔用顯存。
            return
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                # 模型未在 Ollama 服務中載入或不存在，顯存中無此模型。
                return
            raise GpuTransitionError(f"無法卸載 Ollama 模型 {model}：{error}") from error
        except (httpx.HTTPError, ValueError) as error:
            raise GpuTransitionError(f"無法卸載 Ollama 模型 {model}：{error}") from error

    async def unload_all_known_ollama_models(self, extra_model: str = "") -> None:
        """動態查詢並嘗試卸載所有 Ollama 模型，確保跨程序安全。

        任何一個模型卸載失敗（非 ConnectError、非 404）時拋出 GpuTransitionError。
        """
        models_to_unload = await self._get_loaded_ollama_models()
        models_to_unload.update(self.known_ollama_models)
        if extra_model:
            models_to_unload.add(extra_model)
            
        for model in models_to_unload:
            await self.unload_ollama(model)

    async def unload_speaches(self, model: str | None = None) -> None:
        """透過 Speaches 實驗性 API 卸載本專案使用的 ASR 模型。"""
        target_model = model or self.speaches_model
        if not target_model:
            raise GpuTransitionError("未指定要卸載的 Speaches 模型")
        api_root = self.speaches_url.removesuffix("/v1")
        encoded_model = quote(target_model, safe="")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.delete(f"{api_root}/api/ps/{encoded_model}")
                status_code = getattr(response, "status_code", 200)
                if status_code not in (200, 204, 404):
                    response.raise_for_status()
        except httpx.ConnectError:
            # 服務未啟動時不可能有 Speaches 模型佔用顯存。
            return
        except httpx.HTTPStatusError as error:
            if error.response.status_code in (404, 204):
                # 模型未載入於 Speaches 顯存中或已成功釋放。
                return
            raise GpuTransitionError(
                "無法釋放 Speaches ASR 顯存；為避免 CUDA OOM，已取消載入 Ollama："
                f"{error}"
            ) from error
        except httpx.TimeoutException as error:
            raise GpuTransitionError(
                f"卸載 Speaches 模型 {target_model} 逾時；為避免 CUDA OOM，已取消後續推論"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise GpuTransitionError(
                "無法釋放 Speaches ASR 顯存；為避免 CUDA OOM，已取消載入 Ollama："
                f"{error}"
            ) from error

    async def _ensure_safe_for_category(
        self, category: GpuCategory, model: str
    ) -> None:
        """無條件確保進入指定類別前，所有可能衝突的模型都已卸載。

        不依賴程序內歷史；每次都主動清理，保證跨程序、程序重啟、崩潰後的安全。
        """
        if category == GpuCategory.OLLAMA:
            # 進入 Ollama 前：卸載 Breeze ASR
            await self.unload_speaches()
        elif category == GpuCategory.SPEACHES:
            # 進入 Speaches 前：卸載所有已知 Ollama 模型
            await self.unload_all_known_ollama_models()

    @asynccontextmanager
    async def acquire(
        self,
        task_name: str,
        category: GpuCategory,
        model: str = "",
        ollama_model_to_unload: str = "",
    ) -> AsyncIterator[None]:
        self._waiting_count += 1
        try:
            await self._lock.acquire()
        except BaseException:
            self._waiting_count -= 1
            raise

        self._waiting_count -= 1
        entered_task = False
        try:
            async with async_host_gpu_lock(self.gpu_lock_file):
                self._active_task = task_name
                self._active_category = category

                # 跨程序安全：無條件清理可能衝突的模型
                await self._ensure_safe_for_category(category, model)

                entered_task = True
                yield
        finally:
            self._active_task = None
            self._active_category = None
            if entered_task:
                self._last_category_used = category
                self._completed_tasks += 1
            self._lock.release()

    async def run(
        self,
        task_name: str,
        category: GpuCategory,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        model: str = "",
        ollama_model: str = "",
        **kwargs: Any,
    ) -> Any:
        effective_model = model or ollama_model
        async with self.acquire(
            task_name,
            category,
            model=effective_model,
            ollama_model_to_unload=effective_model,
        ):
            return await coro_func(*args, **kwargs)
