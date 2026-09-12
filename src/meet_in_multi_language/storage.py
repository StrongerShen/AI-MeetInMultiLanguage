from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .models import EvaluationRun, TranscriptResult


class RunNotFoundError(KeyError):
    pass


class ImmutableRawAsrError(ValueError):
    pass


def _validate_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise RunNotFoundError(f"無效的工作識別碼：{run_id}")


class RunStore:
    def __init__(self, data_dir: Path) -> None:
        self.upload_dir = data_dir / "uploads"
        self.run_dir = data_dir / "runs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def audio_path(self, stored_filename: str) -> Path:
        return self.upload_dir / stored_filename

    def save(self, run: EvaluationRun) -> EvaluationRun:
        _validate_run_id(run.run_id)
        destination = self.run_dir / f"{run.run_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        with self._lock:
            if destination.is_file():
                existing = EvaluationRun.model_validate_json(
                    destination.read_text(encoding="utf-8")
                )
                if existing.raw_asr is not None and run.raw_asr != existing.raw_asr:
                    raise ImmutableRawAsrError("raw_asr 建立後不可修改或清除")
                if existing.raw_asr is not None:
                    preserved_raw = next(
                        (
                            revision
                            for revision in run.revisions
                            if revision.revision_id == existing.raw_asr.revision_id
                        ),
                        None,
                    )
                    if preserved_raw != existing.raw_asr:
                        raise ImmutableRawAsrError(
                            "revisions 必須完整保留原始 raw_asr 版本"
                        )
            if run.raw_asr is not None:
                preserved_raw = next(
                    (
                        revision
                        for revision in run.revisions
                        if revision.revision_id == run.raw_asr.revision_id
                    ),
                    None,
                )
                if preserved_raw != run.raw_asr:
                    raise ImmutableRawAsrError(
                        "revisions 必須完整保留原始 raw_asr 版本"
                    )

            run.updated_at = datetime.now(UTC)
            content = run.model_dump(mode="json")
            temporary.write_text(
                json.dumps(content, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        return run

    def get(self, run_id: str) -> EvaluationRun:
        _validate_run_id(run_id)
        path = self.run_dir / f"{run_id}.json"
        with self._lock:
            if not path.is_file():
                raise RunNotFoundError(run_id)
            return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, run_id: str, **changes: Any) -> EvaluationRun:
        with self._lock:
            run = self.get(run_id)
            requested_raw_asr = changes.get("raw_asr")
            if (
                run.raw_asr is not None
                and "raw_asr" in changes
                and requested_raw_asr != run.raw_asr
            ):
                raise ImmutableRawAsrError("raw_asr 建立後不可修改或清除")

            requested_revisions = changes.get("revisions")
            if run.raw_asr is not None and requested_revisions is not None:
                preserved_raw = next(
                    (
                        revision
                        for revision in requested_revisions
                        if revision.revision_id == run.raw_asr.revision_id
                    ),
                    None,
                )
                if preserved_raw != run.raw_asr:
                    raise ImmutableRawAsrError("revisions 必須完整保留原始 raw_asr 版本")

            updated = run.model_copy(update=changes)
            return self.save(updated)

    def append_revision(self, run_id: str, revision: TranscriptResult) -> EvaluationRun:
        """以單一鎖定操作追加版本，避免並行校訂互相覆蓋。"""
        with self._lock:
            run = self.get(run_id)
            if any(item.revision_id == revision.revision_id for item in run.revisions):
                raise ValueError(f"逐字稿版本識別碼重複：{revision.revision_id}")
            updated = run.model_copy(
                update={"revisions": [*run.revisions, revision], "result": revision}
            )
            return self.save(updated)

    def list(self) -> list[EvaluationRun]:
        with self._lock:
            runs = [
                EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
                for path in self.run_dir.glob("*.json")
            ]
        return sorted(runs, key=lambda item: item.created_at, reverse=True)
