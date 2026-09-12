from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audio import prepare_audio_chunks
from .batch import transcribe_manifest
from .metrics import character_error_rate, word_error_rate
from .models import Engine
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

    reference = args.reference.read_text(encoding="utf-8")
    hypothesis = args.hypothesis.read_text(encoding="utf-8")
    if args.unit == "word":
        score = word_error_rate(reference, hypothesis)
        metric = "WER"
    else:
        score = character_error_rate(reference, hypothesis)
        metric = "CER"
    print(json.dumps({"metric": metric, "score": score}, ensure_ascii=False))
