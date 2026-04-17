#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from subtitle_translator import (
    BaseTranslator,
    BatchTranslationResult,
    Cue,
    GlossaryEntry,
    OpenAIChatTranslator,
    RuntimeConfig,
    SentenceGroup,
    TranslationRequest,
    build_translator,
    create_glossary_store,
    load_runtime_config,
    positive_int,
    read_srt,
    translate_batch_groups,
    translate_srt,
    write_srt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English→Korean SRT translator (LLM-based)")
    parser.add_argument("input", help="Path to input .srt (English)")
    parser.add_argument("-o", "--output", default=None, help="Path to output .srt (Korean)")
    parser.add_argument("--provider", choices=["openai"], default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    parser.add_argument("--batch-size", type=positive_int, default=32, help="Sentence batch size per API call")
    parser.add_argument("--repeat-fill", action="store_true", help="Repeat last fragment to fill empty slots")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="Override OpenAI base URL")
    parser.add_argument(
        "--glossary-log-path",
        default=None,
        help="Override glossary JSONL log path (default comes from .env or translation_artifacts/glossary.jsonl)",
    )
    parser.add_argument(
        "--disable-history-context",
        action="store_true",
        help="Do not feed the previous translated sentence back into later requests",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.provider != "openai":
        raise ValueError("Unsupported provider")

    config = load_runtime_config(glossary_log_path=args.glossary_log_path)
    translator = build_translator(config=config, model=args.model, base_url=args.openai_base_url)
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)

    cues = read_srt(args.input)
    out_cues = translate_srt(
        cues,
        translator=translator,
        batch_size=args.batch_size,
        repeat_fill=args.repeat_fill,
        glossary_store=glossary_store,
        use_previous_context=(config.use_previous_translation_context and not args.disable_history_context),
    )

    output_path = args.output or os.path.splitext(args.input)[0] + ".ko.srt"
    write_srt(out_cues, output_path)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
