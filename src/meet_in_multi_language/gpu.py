from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Coroutine
from urllib.parse import quote

import httpx


class GpuCategory(StrEnum):
    SPEACHES = "speaches"
    OLLAMA = "ollama"


class GpuTransitionError(RuntimeError):
    """GPU 模型無法安全切換。"""


@dataclass
class GpuQueueStatus:
    is_busy: bool
    active_task: str | None
    active_category: GpuCategory | None
    queue_length: int
    last_category_used: GpuCategory | None
    completed_tasks: int


class GpuWorkQueue:
    """單一行程內的 GPU 互斥工作佇列與模型切換守門員。"""

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        speaches_url: str = "http://127.0.0.1:8001/v1",
        ollama_model: str = "qwen3.5:9b",
        speaches_model: str = "paulpengtw/faster-whisper-Breeze-ASR-26",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.speaches_url = speaches_url.rstrip("/")
        self.ollama_model = ollama_model
        self.speaches_model = speaches_model
        self._lock = asyncio.Lock()
        self._waiting_count = 0
        self._active_task: str | None = None
        self._active_category: GpuCategory | None = None
        self._last_category_used: GpuCategory | None = None
        self._last_ollama_model: str | None = None
        self._last_speaches_model: str | None = None
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

    async def unload_speaches(self, model: str | None = None) -> None:
        """透過 Speaches 實驗性 API 卸載本專案使用的 ASR 模型。"""
        target_model = model or self._last_speaches_model or self.speaches_model
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
        except httpx.TimeoutException:
            # Speaches 實驗性端點在部分版本存在內部鎖定延遲；
            # Speaches 內建的 TTL 機制會在轉錄後自動釋放顯存，容許此處平順交接。
            return
        except (httpx.HTTPError, ValueError) as error:
            raise GpuTransitionError(
                "無法釋放 Speaches ASR 顯存；為避免 CUDA OOM，已取消載入 Ollama："
                f"{error}"
            ) from error

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
        effective_ollama_to_unload = (
            ollama_model_to_unload or self._last_ollama_model or self.ollama_model
        )
        try:
            self._active_task = task_name
            self._active_category = category

            if category == GpuCategory.SPEACHES:
                if self._last_category_used == GpuCategory.OLLAMA:
                    await self.unload_ollama(effective_ollama_to_unload)
                    self._last_ollama_model = None
            elif category == GpuCategory.OLLAMA:
                if self._last_category_used == GpuCategory.SPEACHES:
                    await self.unload_speaches()
                    self._last_speaches_model = None
                elif (
                    self._last_category_used == GpuCategory.OLLAMA
                    and self._last_ollama_model
                    and model
                    and self._last_ollama_model != model
                ):
                    # 同為 Ollama 任務但切換模型時，卸載前一個模型以釋放顯存
                    await self.unload_ollama(self._last_ollama_model)
                    self._last_ollama_model = None

            entered_task = True
            yield
        finally:
            self._active_task = None
            self._active_category = None
            if entered_task:
                self._last_category_used = category
                if category == GpuCategory.OLLAMA:
                    self._last_ollama_model = model or self.ollama_model
                elif category == GpuCategory.SPEACHES:
                    self._last_speaches_model = model or self.speaches_model
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
