from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Engine(StrEnum):
    TRANSCRIBE = "gpt-transcribe"
    DIARIZE = "gpt-4o-transcribe-diarize"


class RunStatus(StrEnum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptRevisionKind(StrEnum):
    RAW_ASR = "raw_asr"
    LLM_CORRECTED = "llm_corrected"
    HUMAN_EDITED = "human_edited"


class TranscriptSegment(BaseModel):
    segment_id: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    speaker: str | None = None
    text: str
    quality_flags: list[str] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    provider: str = "openai"
    model: str
    revision_kind: TranscriptRevisionKind = TranscriptRevisionKind.RAW_ASR
    source_revision_id: str | None = None
    text: str
    detected_languages: list[str] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    run_id: str
    original_filename: str
    stored_filename: str
    engine: Engine
    status: RunStatus
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: TranscriptResult | None = None
    error: str | None = None
