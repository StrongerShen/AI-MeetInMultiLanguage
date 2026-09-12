from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "summary", "evidence_ids"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due_date": {"type": ["string", "null"]},
                    "original_due_text": {"type": ["string", "null"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "task",
                    "owner",
                    "due_date",
                    "original_due_text",
                    "evidence_ids",
                ],
            },
        },
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
            },
        },
    },
    "required": ["overview", "topics", "decisions", "action_items", "open_questions"],
}

SYSTEM_PROMPT = """你是臺灣的會議紀錄助理。所有衍生內容一律使用臺灣通用繁體中文與臺灣慣用詞彙。
忠實區分提案、決議、修正、撤回及待確認事項。不要把建議寫成決議，也不要自行補上負責人或期限。
每個主題、決議、待辦及未解問題都只能引用輸入內存在的 evidence ID。
相對日期除了保留原說法，也可在會議日期與時區足夠明確時填入 YYYY-MM-DD；不明確就填 null。
只輸出符合指定 JSON schema 的內容，不要加入前言或 Markdown。"""

FORBIDDEN_TERMS = (
    "軟件",
    "網絡",
    "服務器",
    "數據",
    "信息",
    "視頻",
    "質量",
    "用戶",
    "默認",
)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 600,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    def summarize(self, model: str, transcript: str) -> dict[str, Any]:
        response = self._client.post(
            "/api/chat",
            json={
                "model": model,
                "stream": False,
                "think": False,
                "format": SUMMARY_SCHEMA,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                "options": {"temperature": 0, "seed": 42, "num_ctx": 16384},
            },
        )
        response.raise_for_status()
        return response.json()


def validate_summary(content: str, transcript: str) -> dict[str, object]:
    evidence_ids = set(re.findall(r"\[(seg-\d+)\]", transcript))
    try:
        summary = json.loads(content)
    except json.JSONDecodeError as error:
        return {
            "json_valid": False,
            "schema_valid": False,
            "error": str(error),
            "unknown_evidence_ids": [],
            "forbidden_terms": [],
        }

    missing_fields = [field for field in SUMMARY_SCHEMA["required"] if field not in summary]
    unknown_evidence_ids = sorted(
        {
            evidence
            for evidence in _collect_evidence_ids(summary)
            if evidence not in evidence_ids
        }
    )
    forbidden_terms = sorted(term for term in FORBIDDEN_TERMS if term in content)
    return {
        "json_valid": True,
        "schema_valid": not missing_fields,
        "missing_fields": missing_fields,
        "unknown_evidence_ids": unknown_evidence_ids,
        "forbidden_terms": forbidden_terms,
    }


def _collect_evidence_ids(value: object) -> list[str]:
    if isinstance(value, dict):
        evidence = value.get("evidence_ids", [])
        found = [str(item) for item in evidence] if isinstance(evidence, list) else []
        for child in value.values():
            found.extend(_collect_evidence_ids(child))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_collect_evidence_ids(child))
        return found
    return []


def run_ollama_benchmark(
    client: OllamaClient,
    models: list[str],
    transcript_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    transcript = transcript_path.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    for model in models:
        print(f"正在測試 {model}…", flush=True)
        started_at = datetime.now(UTC)
        try:
            response = client.summarize(model, transcript)
            content = str(response.get("message", {}).get("content", ""))
            validation = validate_summary(content, transcript)
            item: dict[str, object] = {
                "model": model,
                "status": "completed",
                "started_at": started_at.isoformat(),
                "total_duration_ns": response.get("total_duration"),
                "load_duration_ns": response.get("load_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "validation": validation,
                "summary": json.loads(content) if validation["json_valid"] else content,
            }
        except Exception as error:
            item = {
                "model": model,
                "status": "failed",
                "started_at": started_at.isoformat(),
                "error": str(error),
            }
        filename = re.sub(r"[^a-zA-Z0-9._-]+", "-", model) + ".json"
        (output_dir / filename).write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report.append(item)

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

