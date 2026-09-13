from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from .models import EvaluationRun, TranscriptResult


class RunNotFoundError(KeyError):
    pass


class ImmutableRawAsrError(ValueError):
    pass


def _validate_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise RunNotFoundError(f"無效的工作識別碼：{run_id}")


def _validate_filename(filename: str) -> None:
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"無效的檔案名稱：{filename}")
    if Path(filename).name != filename:
        raise ValueError(f"檔案名稱必須為純檔名：{filename}")


class RunStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.upload_dir = data_dir / "uploads"
        self.run_dir = data_dir / "runs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._flock_fd: int | None = None
        self._flock_depth: int = 0

    @contextmanager
    def _acquire_lock(self) -> Iterator[None]:
        """跨執行緒（RLock）與跨程序（fcntl.flock）的安全重入鎖。"""
        with self._lock:
            if self._flock_depth == 0:
                lock_path = self.data_dir / "store.lock"
                self._flock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
                fcntl.flock(self._flock_fd, fcntl.LOCK_EX)
            self._flock_depth += 1
            try:
                yield
            finally:
                self._flock_depth -= 1
                if self._flock_depth == 0 and self._flock_fd is not None:
                    try:
                        fcntl.flock(self._flock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    try:
                        os.close(self._flock_fd)
                    except OSError:
                        pass
                    self._flock_fd = None

    def audio_path(self, stored_filename: str) -> Path:
        _validate_filename(stored_filename)
        target = (self.upload_dir / stored_filename).resolve()
        upload_resolved = self.upload_dir.resolve()
        try:
            target.relative_to(upload_resolved)
        except ValueError as err:
            raise ValueError(f"路徑超出上傳目錄範圍：{stored_filename}") from err
        return target

    def save(self, run: EvaluationRun) -> EvaluationRun:
        _validate_run_id(run.run_id)
        destination = self.run_dir / f"{run.run_id}.json"
        temporary = self.run_dir / f"{run.run_id}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        with self._acquire_lock():
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
            try:
                temporary.write_text(
                    json.dumps(content, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return run

    def get(self, run_id: str) -> EvaluationRun:
        _validate_run_id(run_id)
        path = self.run_dir / f"{run_id}.json"
        with self._acquire_lock():
            if not path.is_file():
                raise RunNotFoundError(run_id)
            return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, run_id: str, **changes: Any) -> EvaluationRun:
        with self._acquire_lock():
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
        with self._acquire_lock():
            run = self.get(run_id)
            if any(item.revision_id == revision.revision_id for item in run.revisions):
                raise ValueError(f"逐字稿版本識別碼重複：{revision.revision_id}")
            updated = run.model_copy(
                update={"revisions": [*run.revisions, revision], "result": revision}
            )
            return self.save(updated)

    def list(self) -> list[EvaluationRun]:
        with self._acquire_lock():
            runs = [
                EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
                for path in self.run_dir.glob("*.json")
            ]
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def delete(self, run_id: str, delete_audio: bool = True) -> None:
        """刪除工作紀錄，並可選擇性安全刪除關聯的音訊檔案。"""
        _validate_run_id(run_id)
        with self._acquire_lock():
            run = self.get(run_id)
            if delete_audio and run.stored_filename:
                try:
                    audio_file = self.audio_path(run.stored_filename)
                    audio_file.unlink(missing_ok=True)
                except Exception as err:
                    raise OSError(
                        f"刪除音訊檔案失敗，已保留工作紀錄避免產生孤兒檔案：{err}"
                    ) from err
            run_file = self.run_dir / f"{run_id}.json"
            run_file.unlink(missing_ok=True)
