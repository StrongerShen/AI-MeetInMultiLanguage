from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int
    openai_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("APP_DATA_DIR", "./var")).resolve(),
            max_upload_bytes=int(os.getenv("APP_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        )

