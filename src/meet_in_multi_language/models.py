from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Engine(StrEnum):
    TRANSCRIBE = "gpt-transcribe"
    DIARIZE = "gpt-4o-transcribe-diarize"
    BREEZE = "breeze"


class RunStatus(StrEnum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
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
    revision_id: str = Field(default_factory=lambda: f"rev-{uuid4().hex[:8]}")
    provider: str = "openai"
    model: str
    revision_kind: TranscriptRevisionKind = TranscriptRevisionKind.RAW_ASR
    source_revision_id: str | None = None
    text: str
    detected_languages: list[str] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))



class TopicItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    evidence_ids: list[str]


class DecisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_ids: list[str]


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    owner: str | None = None
    due_date: str | None = None
    original_due_text: str | None = None
    evidence_ids: list[str]


class OpenQuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_ids: list[str]


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    source_revision_id: str
    overview: str
    topics: list[TopicItem]
    decisions: list[DecisionItem]
    action_items: list[ActionItem]
    open_questions: list[OpenQuestionItem]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationRun(BaseModel):
    run_id: str
    original_filename: str
    stored_filename: str
    engine: Engine
    status: RunStatus
    keywords: list[str] = Field(default_factory=list)
    auto_summary: bool = False
    summary_model: str = "qwen3.5:9b"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_asr: TranscriptResult | None = None
    revisions: list[TranscriptResult] = Field(default_factory=list)
    result: TranscriptResult | None = None
    summary: SummaryResult | None = None
    error: str | None = None
