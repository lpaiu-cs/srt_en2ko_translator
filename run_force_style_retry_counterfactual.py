#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from subtitle_translator import build_translator, hydrate_translation_block, load_runtime_config, read_srt
from subtitle_translator.models import EmittedCue, TranslationRequest
from subtitle_translator.text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Force a strict style retry on one translated replay row.")
    parser.add_argument("--input", required=True, help="Translated replay eval JSONL")
    parser.add_argument("--id", required=True, help="Exact row id to replay")
    parser.add_argument("--output", required=True, help="Where to write the counterfactual JSON")
    parser.add_argument("--model", default=None, help="Override phase1 model")
    parser.add_argument("--phase1-temperature", type=float, default=0.0, help="Phase1 temperature override")
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Override OpenAI base URL",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    entry = next(row for row in rows if row["id"] == args.id)

    config = load_runtime_config()
    config.phase1_temperature = max(0.0, args.phase1_temperature)
    translator = build_translator(config=config, model=args.model or config.phase1_model, base_url=args.openai_base_url)

    all_cues = read_srt(entry["source_file"])
    block_data = entry["current_block"]
    cue_index_set = set(block_data["cue_indices"])
    block_cues = [cue for cue in all_cues if cue.index in cue_index_set]
    block_lint = block_data.get("block_lint", {})
    block = hydrate_translation_block(
        block_cues,
        all_cues,
        config,
        low_confidence=bool(block_lint.get("low_confidence", False)),
        lint_reasons=list(block_lint.get("lint_reasons", [])),
        lint_actions=list(block_lint.get("lint_actions", [])),
    )

    translated_cues = entry["translation_output"]["translated_cues"]
    previous_emitted_cues = [
        EmittedCue(cue_index=cue["cue_index"], text=normalize_text(cue["text"]))
        for cue in translated_cues
    ]

    source_cues = block_data["source_cues"]
    offending_span = {
        "cue_index": translated_cues[-1]["cue_index"],
        "cue_indices": [translated_cues[-2]["cue_index"], translated_cues[-1]["cue_index"]],
        "span_text": normalize_text(translated_cues[-1]["text"]).rstrip(".,!?;:'\""),
        "left_text": normalize_text(translated_cues[-2]["text"]),
        "right_text": normalize_text(translated_cues[-1]["text"]),
        "source_cue_text": source_cues[-1]["text"],
        "source_tail_type": entry.get("replay_meta", {}).get("source_tail_type"),
        "issue": "duplicate_restatement",
        "preferred_action": "restore_missing_tail",
    }

    request = TranslationRequest(
        block=block,
        strict_style_retry=True,
        style_retry_reasons=["duplicate_restatement"],
        previous_emitted_cues=previous_emitted_cues,
        protected_cue_indices=[translated_cues[-2]["cue_index"]],
        offending_cue_indices=[translated_cues[-1]["cue_index"]],
        offending_spans=[offending_span],
        preferred_actions=["restore_missing_tail"],
    )
    result = translator.translate_block(request)

    payload = {
        "id": entry["id"],
        "source_file": entry["source_file"],
        "replay_meta": entry.get("replay_meta"),
        "current_translated_cues": translated_cues,
        "forced_request": {
            "style_retry_reasons": request.style_retry_reasons,
            "protected_cue_indices": request.protected_cue_indices,
            "offending_cue_indices": request.offending_cue_indices,
            "offending_spans": request.offending_spans,
            "preferred_actions": request.preferred_actions,
        },
        "counterfactual_emitted_cues": [{"cue_index": cue.cue_index, "text": cue.text} for cue in result.emitted_cues],
        "risk_flags": list(result.risk_flags),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote counterfactual output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
