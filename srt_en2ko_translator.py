#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from subtitle_translator import (
    TranslationMetrics,
    append_metrics_log,
    build_translator,
    create_glossary_store,
    load_runtime_config,
    positive_int,
    read_srt,
    translate_srt,
    write_srt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English→Korean SRT translator (LLM-based)")
    parser.add_argument("input", help="Path to input .srt (English)")
    parser.add_argument("-o", "--output", default=None, help="Path to output .srt (Korean)")
    parser.add_argument("--provider", choices=["openai"], default="openai", help="LLM provider")
    parser.add_argument(
        "--model",
        default=None,
        help="Phase1 model name (default: .env SRT_PHASE1_MODEL/SRT_OPENAI_MODEL or gpt-4.1-mini)",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Phase2 repair model name (default: .env SRT_REPAIR_MODEL or gpt-4o)",
    )
    parser.add_argument(
        "--block-max-cues",
        "--batch-size",
        dest="block_max_cues",
        type=positive_int,
        default=None,
        help="Maximum cues per translation block (legacy alias: --batch-size)",
    )
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="Override OpenAI base URL")
    parser.add_argument(
        "--glossary-log-path",
        default=None,
        help="Override glossary JSONL log path (default comes from .env or translation_artifacts/glossary.jsonl)",
    )
    parser.add_argument(
        "--disable-context-window",
        "--disable-history-context",
        dest="disable_context_window",
        action="store_true",
        help="Do not provide surrounding source context to the model",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.provider != "openai":
        raise ValueError("Unsupported provider")

    config = load_runtime_config(glossary_log_path=args.glossary_log_path)
    model = args.model or config.phase1_model
    repair_model = args.repair_model or config.repair_model
    if args.disable_context_window:
        config.use_context_window = False
    if args.block_max_cues is not None:
        config.block_max_cues = max(config.block_min_cues, args.block_max_cues)
    translator = build_translator(config=config, model=model, repair_model=repair_model, base_url=args.openai_base_url)
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)
    metrics = TranslationMetrics()

    cues = read_srt(args.input)
    out_cues = translate_srt(
        cues,
        translator=translator,
        config=config,
        glossary_store=glossary_store,
        metrics=metrics,
    )

    output_path = args.output or os.path.splitext(args.input)[0] + ".ko.srt"
    write_srt(out_cues, output_path)
    append_metrics_log(
        config.metrics_log_path,
        metrics,
        input_path=args.input,
        output_path=output_path,
        phase1_model=model,
        repair_model=repair_model,
    )
    print(f"Metrics: {metrics.summary()}", file=sys.stderr)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
