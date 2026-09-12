from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .models import EvaluationRun


class RunNotFoundError(KeyError):
    pass


class RunStore:
    def __init__(self, data_dir: Path) -> None:
        self.upload_dir = data_dir / "uploads"
        self.run_dir = data_dir / "runs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def audio_path(self, stored_filename: str) -> Path:
        return self.upload_dir / stored_filename

    def save(self, run: EvaluationRun) -> EvaluationRun:
        run.updated_at = datetime.now(UTC)
        destination = self.run_dir / f"{run.run_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        content = run.model_dump(mode="json")
        with self._lock:
            temporary.write_text(
                json.dumps(content, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        return run

    def get(self, run_id: str) -> EvaluationRun:
        path = self.run_dir / f"{run_id}.json"
        if not path.is_file():
            raise RunNotFoundError(run_id)
        return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, run_id: str, **changes: Any) -> EvaluationRun:
        run = self.get(run_id)
        updated = run.model_copy(update=changes)
        return self.save(updated)

    def list(self) -> list[EvaluationRun]:
        runs = [
            EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.run_dir.glob("*.json")
        ]
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

