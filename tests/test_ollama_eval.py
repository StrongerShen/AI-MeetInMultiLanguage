import json
from pathlib import Path

import httpx

from meet_in_multi_language.ollama_eval import (
    OllamaClient,
    run_ollama_benchmark,
    validate_summary,
)


VALID_SUMMARY = {
    "overview": "第一版採上傳後處理。",
    "topics": [],
    "decisions": [{"text": "上傳後處理", "evidence_ids": ["seg-001"]}],
    "action_items": [],
    "open_questions": [],
}


def test_validation_rejects_unknown_evidence_and_mainland_term() -> None:
    summary = {
        **VALID_SUMMARY,
        "overview": "改善軟件質量",
        "decisions": [{"text": "完成", "evidence_ids": ["seg-999"]}],
    }
    result = validate_summary(json.dumps(summary, ensure_ascii=False), "[seg-001] 原文")

    assert result["json_valid"] is True
    assert result["unknown_evidence_ids"] == ["seg-999"]
    assert set(result["forbidden_terms"]) == {"軟件", "質量"}


def test_validation_rejects_incomplete_nested_schema() -> None:
    summary = {
        **VALID_SUMMARY,
        "decisions": [{"text": "缺少引用欄位"}],
    }
    result = validate_summary(json.dumps(summary, ensure_ascii=False), "[seg-001] 原文")

    assert result["json_valid"] is True
    assert result["schema_valid"] is False
    assert result["error"]


def test_benchmark_saves_model_result(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["think"] is False
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(VALID_SUMMARY, ensure_ascii=False)},
                "total_duration": 100,
                "eval_count": 20,
            },
        )

    transcript = tmp_path / "transcript.txt"
    transcript.write_text("[seg-001] 原文", encoding="utf-8")
    http_client = httpx.Client(
        transport=httpx.MockTransport(respond), base_url="http://ollama.test"
    )

    report = run_ollama_benchmark(
        OllamaClient("http://ollama.test", client=http_client),
        ["model:latest"],
        transcript,
        tmp_path / "output",
    )

    assert report[0]["status"] == "completed"
    assert report[0]["validation"]["schema_valid"] is True
    assert (tmp_path / "output" / "model-latest.json").is_file()


def test_format_transcript_and_parse_summary() -> None:
    from meet_in_multi_language.models import TranscriptResult, TranscriptSegment
    from meet_in_multi_language.ollama_eval import (
        format_transcript_for_summary,
        parse_summary_payload,
    )

    transcript = TranscriptResult(
        model="breeze",
        text="",
        segments=[
            TranscriptSegment(
                segment_id="segment-1",
                start_ms=4000,
                speaker="主持人",
                text="大家好，今仔日討論會議工具。",
            ),
            TranscriptSegment(
                segment_id="segment-2",
                start_ms=22000,
                speaker="Alice",
                text="I propose seven days retention.",
            ),
        ],
    )

    formatted = format_transcript_for_summary(transcript)
    assert "[seg-001][00:00:04][主持人] 大家好，今仔日討論會議工具。" in formatted
    assert "[seg-002][00:00:22][Alice] I propose seven days retention." in formatted

    summary_obj = parse_summary_payload(
        json.dumps(VALID_SUMMARY, ensure_ascii=False),
        model="qwen3.5:9b",
        source_revision_id="rev-1234",
    )
    assert summary_obj.model == "qwen3.5:9b"
    assert summary_obj.source_revision_id == "rev-1234"
    assert summary_obj.overview == "第一版採上傳後處理。"
    assert summary_obj.decisions[0].text == "上傳後處理"
    assert summary_obj.decisions[0].evidence_ids == ["seg-001"]


def test_validation_accepts_bracketed_evidence_id_and_normalizes() -> None:
    from meet_in_multi_language.ollama_eval import (
        normalize_evidence_id,
        parse_summary_payload,
        validate_summary,
    )

    assert normalize_evidence_id("[seg-001]") == "seg-001"
    assert normalize_evidence_id(" seg-002 ") == "seg-002"
    assert normalize_evidence_id("[seg-003] ") == "seg-003"

    summary = {
        "overview": "測試方括號引用容忍度",
        "topics": [{"title": "議題", "summary": "說明", "evidence_ids": [" [seg-001] "]}],
        "decisions": [{"text": "決議內容", "evidence_ids": ["[seg-001]"]}],
        "action_items": [
            {
                "task": "待辦",
                "owner": "Bob",
                "due_date": None,
                "original_due_text": None,
                "evidence_ids": ["[seg-001]"],
            }
        ],
        "open_questions": [{"text": "問題", "evidence_ids": ["[seg-001]"]}],
    }
    raw_content = json.dumps(summary, ensure_ascii=False)
    val_result = validate_summary(raw_content, "[seg-001] 原文內容")
    assert val_result["schema_valid"] is True
    assert val_result["unknown_evidence_ids"] == []

    parsed = parse_summary_payload(raw_content, model="test", source_revision_id="rev-1")
    assert parsed.decisions[0].evidence_ids == ["seg-001"]
    assert parsed.topics[0].evidence_ids == ["seg-001"]
    assert parsed.action_items[0].evidence_ids == ["seg-001"]
    assert parsed.open_questions[0].evidence_ids == ["seg-001"]
