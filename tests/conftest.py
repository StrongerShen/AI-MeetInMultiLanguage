from __future__ import annotations

import socket
from typing import Any

import pytest

# 測試環境允許的本機位址（用於 httpx.ASGITransport、mock 伺服器等）
_ALLOWED_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

# 允許 UNIX domain socket（AF_UNIX）路徑前綴（pytest、coverage 等內部通訊）
_ALLOWED_UNIX_PREFIXES: tuple[str, ...] = ("/tmp/", "/var/run/")


def _is_allowed_connection(address: Any) -> bool:
    """判斷連線目標是否為測試環境允許的位址。

    允許：
    - UNIX domain socket
    - 本機回環位址上的 httpx.ASGITransport、mock 或 fake 使用的埠（排除真實服務埠 8001/11434）
    - 非 tuple 型態的位址（AF_UNIX 等）
    """
    if isinstance(address, (str, bytes)):
        # UNIX domain socket
        return True
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        host_str = str(host)
        # 阻擋所有真實服務埠（Speaches 8001、Ollama 11434）
        if port in (8001, 11434):
            return False
        # 允許本機回環位址
        if host_str in _ALLOWED_LOOPBACK:
            return True
        # 阻擋所有其他外部位址
        return False
    # 未知格式一律允許（避免破壞內部機制）
    return True


@pytest.fixture(autouse=True)
def guard_external_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試環境防線：禁止所有未明確允許的外部網路連線。

    - 阻擋 IPv4、IPv6 的真實 TCP 連線
    - 同時處理 connect() 與 connect_ex()
    - 不破壞 httpx.ASGITransport、mock 或 fake 測試
    - pytest 不得卸載真實 Breeze 或 Ollama 模型
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_allowed_connection(address):
            raise RuntimeError(
                f"測試環境防線觸發：禁止直接連線至外部服務 ({address})，"
                f"測試中請使用 mock 或 fake 物件。"
            )
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_allowed_connection(address):
            raise RuntimeError(
                f"測試環境防線觸發：禁止直接連線至外部服務 ({address})，"
                f"測試中請使用 mock 或 fake 物件。"
            )
        return original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture
def allow_local_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """明確 opt-in fixture：允許本機臨時 socket 測試（最小範圍例外）。

    僅供確實需要本機 socket 的測試使用，例如 IPC 或 subprocess 通訊測試。
    """
    # 此 fixture 存在時，guard_external_network_calls 的 monkeypatch 會被覆蓋
    # 恢復原始 connect/connect_ex
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
