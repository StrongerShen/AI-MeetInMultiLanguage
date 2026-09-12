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
