import socket
import pytest

BLOCKED_PORTS = {8001, 11434}


@pytest.fixture(autouse=True)
def guard_external_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """測試環境防線：禁止任何測試直接連線真實 Speaches、Ollama 或未授權的外部網路服務。"""
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: tuple | str | bytes) -> None:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if port in BLOCKED_PORTS:
                raise RuntimeError(
                    f"測試環境防線觸發：禁止直接連線至外部真實服務 ({host}:{port})，測試中請使用 mock 或 fake 物件。"
                )
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
