from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audio import prepare_audio_chunks
from .batch import transcribe_manifest
from .metrics import character_error_rate, word_error_rate
from .models import Engine
from .ollama_eval import OllamaClient, run_ollama_benchmark
from .speaches import PROFILES, SpeachesTranscriber
from .transcription import OpenAITranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多語會議品質評測工具")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="將長音檔正規化並切段")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--segment-seconds", type=int, default=600)

    transcribe = commands.add_parser("transcribe", help="依 manifest 整批轉錄並合併")
    transcribe.add_argument("manifest", type=Path)
    transcribe.add_argument("output", type=Path)
    transcribe.add_argument(
        "--engine", choices=tuple(Engine), default=Engine.DIARIZE.value
    )
    transcribe.add_argument("--keyword", action="append", default=[])
    transcribe.add_argument("--force", action="store_true")

    score = commands.add_parser("score", help="比較人工參考稿與轉錄稿")
    score.add_argument("reference", type=Path)
    score.add_argument("hypothesis", type=Path)
    score.add_argument("--unit", choices=("character", "word"), default="character")

    ollama = commands.add_parser(
        "ollama-benchmark", help="比較 Ollama 模型的會議摘要能力"
    )
    ollama.add_argument("transcript", type=Path)
    ollama.add_argument("output", type=Path)
    ollama.add_argument("--host", default="http://Ubuntu.local:11434")
    ollama.add_argument("--model", action="append", required=True)

    local_asr = commands.add_parser(
        "speaches-transcribe", help="使用 Speaches/faster-whisper 轉錄整批音訊"
    )
    local_asr.add_argument("manifest", type=Path)
    local_asr.add_argument("output", type=Path)
    local_asr.add_argument("--host", default="http://127.0.0.1:8001/v1")
    local_asr.add_argument("--profile", choices=tuple(PROFILES), default="breeze")
    local_asr.add_argument("--keyword", action="append", default=[])
    local_asr.add_argument("--force", action="store_true")
    vad = local_asr.add_mutually_exclusive_group()
    vad.add_argument("--vad", dest="vad_filter", action="store_true")
    vad.add_argument("--no-vad", dest="vad_filter", action="store_false")
    local_asr.set_defaults(vad_filter=None)

    pipe = commands.add_parser(
        "pipeline", help="執行端到端批次處理：切段、轉錄、結構化摘要與多格式匯出"
    )
    pipe.add_argument("audio", type=Path, help="音訊檔案路徑")
    pipe.add_argument("output", type=Path, help="輸出目錄路徑")
    pipe.add_argument(
        "--engine",
        choices=("breeze", "gpt-transcribe", "gpt-4o-transcribe-diarize"),
        default="breeze",
        help="轉錄引擎",
    )
    pipe.add_argument(
        "--summary-model",
        default=None,
        help="Ollama 結構化摘要模型名稱（例如 qwen3.5:9b）",
    )
    pipe.add_argument(
        "--segment-seconds",
        type=int,
        default=600,
        help="長音訊切段秒數上限（預設 600 秒）",
    )
    pipe.add_argument("--keyword", action="append", default=[], help="關鍵詞提示（可多次指定）")
    pipe.add_argument("--speaches-url", default="http://127.0.0.1:8001/v1")
    pipe.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    pipe.add_argument(
        "--export",
        default="txt,srt,vtt,md,json",
        help="匯出格式（以逗號分隔，預設 txt,srt,vtt,md,json）",
    )
    pipe.add_argument("--force", action="store_true", help="強制重新轉錄已有快取的切段")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_audio_chunks(args.source, args.output, args.segment_seconds)
        print(
            json.dumps(
                {
                    "manifest": str(args.output / "manifest.json"),
                    "chunks": len(manifest["chunks"]),
                    "duration_ms": manifest["source"]["duration_ms"],
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "transcribe":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("尚未設定 OPENAI_API_KEY")
        engine = Engine(args.engine)
        output = transcribe_manifest(
            args.manifest,
            args.output,
            engine,
            OpenAITranscriber(api_key),
            keywords=args.keyword,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "transcript": str(args.output / engine.value / "transcript.json"),
                    "completed_chunks": len(output["completed_chunks"]),
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "ollama-benchmark":
        report = run_ollama_benchmark(
            OllamaClient(args.host), args.model, args.transcript, args.output
        )
        print(
            json.dumps(
                {
                    "report": str(args.output / "report.json"),
                    "completed": sum(item["status"] == "completed" for item in report),
                    "failed": sum(item["status"] == "failed" for item in report),
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "speaches-transcribe":
        profile = PROFILES[args.profile]
        transcriber = SpeachesTranscriber(
            args.host, profile, vad_filter=args.vad_filter
        )
        output = transcribe_manifest(
            args.manifest,
            args.output,
            profile.name,
            transcriber,
            keywords=args.keyword,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "transcript": str(args.output / profile.name / "transcript.json"),
                    "model": profile.model,
                    "vad_filter": transcriber.vad_filter,
                    "completed_chunks": len(output["completed_chunks"]),
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "pipeline":
        from .pipeline import run_pipeline

        export_formats = [fmt.strip() for fmt in args.export.split(",") if fmt.strip()]
        report = run_pipeline(
            source_audio=args.audio,
            output_dir=args.output,
            engine=args.engine,
            summary_model=args.summary_model,
            segment_seconds=args.segment_seconds,
            keywords=args.keyword,
            export_formats=export_formats,
            speaches_url=args.speaches_url,
            ollama_url=args.ollama_url,
            force=args.force,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    reference = args.reference.read_text(encoding="utf-8")
    hypothesis = args.hypothesis.read_text(encoding="utf-8")
    if args.unit == "word":
        score = word_error_rate(reference, hypothesis)
        metric = "WER"
    else:
        score = character_error_rate(reference, hypothesis)
        metric = "CER"
    print(json.dumps({"metric": metric, "score": score}, ensure_ascii=False))
