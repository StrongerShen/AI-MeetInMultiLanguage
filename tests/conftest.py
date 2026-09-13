from __future__ import annotations

import socket
from typing import Any
import pytest

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex

def _is_allowed_connection(address: Any) -> bool:
    """判斷連線目標是否為測試環境允許的位址。

    允許：
    - UNIX domain socket
    - 非 tuple 型態的位址（AF_UNIX 等）
    預設拒絕所有 TCP 連線（包含 localhost）。
    """
    if isinstance(address, (str, bytes)):
        # UNIX domain socket
        return True
    if isinstance(address, tuple):
        # 阻擋所有 IP 位址 (包含 localhost)
        return False
    return True

@pytest.fixture(autouse=True)
def guard_external_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試環境防線：禁止所有未明確允許的外部網路連線。"""
    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_allowed_connection(address):
            raise RuntimeError(
                f"測試環境防線觸發：禁止直接連線至外部服務 ({address})，"
                f"測試中請使用 mock 或 fake 物件。"
            )
        return _original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_allowed_connection(address):
            raise RuntimeError(
                f"測試環境防線觸發：禁止直接連線至外部服務 ({address})，"
                f"測試中請使用 mock 或 fake 物件。"
            )
        return _original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

@pytest.fixture
def allow_local_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """明確 opt-in fixture：允許本機臨時 socket 測試（最小範圍例外）。"""
    monkeypatch.setattr(socket.socket, "connect", _original_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _original_connect_ex)
