#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from subtitle_translator import (
    TranslationMetrics,
    build_translator,
    create_glossary_store,
    load_runtime_config,
    read_srt,
)
from subtitle_translator.blocks import build_translation_blocks
from subtitle_translator.models import Cue, TranslationBlock
from subtitle_translator.pipeline import _translate_block_recursive
from subtitle_translator.text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the current translator on a reviewed eval-set JSONL.")
    parser.add_argument(
        "--input",
        default="evaluation/cs231n_sp25_eval_review_round1.jsonl",
        help="Reviewed eval-set JSONL to replay against the current pipeline",
    )
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_eval_translated_round2.jsonl",
        help="Where to write translated evaluation records",
    )
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
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Override OpenAI base URL",
    )
    parser.add_argument(
        "--glossary-log-path",
        default=None,
        help="Override glossary JSONL log path",
    )
    return parser


def _load_review_entries(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _match_block(target_indices: List[int], blocks: List[TranslationBlock]) -> TranslationBlock:
    best: Tuple[int, int, float, TranslationBlock] | None = None
    target_set = set(target_indices)
    target_mid = sum(target_indices) / max(len(target_indices), 1)
    for block in blocks:
        block_indices = [cue.index for cue in block.cues]
        overlap = len(target_set & set(block_indices))
        if overlap == 0:
            continue
        size_penalty = -abs(len(block_indices) - len(target_indices))
        block_mid = sum(block_indices) / max(len(block_indices), 1)
        distance_penalty = -abs(block_mid - target_mid)
        candidate = (overlap, size_penalty, distance_penalty, block)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError(f"Could not match block for cue indices {target_indices}")
    return best[3]


def _serialize_cues(cues: Iterable[Cue]) -> List[dict]:
    return [
        {
            "cue_index": cue.index,
            "start": cue.start,
            "end": cue.end,
            "text": cue.text,
        }
        for cue in cues
    ]


def _translated_text(cues: Iterable[Cue]) -> str:
    return " ".join(normalize_text(cue.text) for cue in cues if normalize_text(cue.text))


def _load_blocks_by_lecture(entries: List[dict], config) -> Dict[str, List[TranslationBlock]]:
    blocks_by_lecture: Dict[str, List[TranslationBlock]] = {}
    for lecture in sorted({entry["lecture"] for entry in entries}):
        lecture_entries = [entry for entry in entries if entry["lecture"] == lecture]
        source_file = Path(lecture_entries[0]["source_file"])
        cues = read_srt(source_file)
        blocks_by_lecture[lecture] = build_translation_blocks(cues, config)
    return blocks_by_lecture


def main() -> int:
    args = build_parser().parse_args()
    review_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    config = load_runtime_config(glossary_log_path=args.glossary_log_path)
    translator = build_translator(
        config=config,
        model=args.model or config.phase1_model,
        repair_model=args.repair_model or config.repair_model,
        base_url=args.openai_base_url,
    )
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)

    review_entries = _load_review_entries(review_path)
    blocks_by_lecture = _load_blocks_by_lecture(review_entries, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in review_entries:
            block = _match_block(entry["cue_indices"], blocks_by_lecture[entry["lecture"]])
            metrics = TranslationMetrics()
            translated_cues = _translate_block_recursive(
                block,
                translator,
                config,
                glossary_store,
                metrics,
                depth=0,
                ancestor_failure_signatures=(),
            )
            metrics.note_final_cues(translated_cues)
            record = {
                "id": entry["id"],
                "lecture": entry["lecture"],
                "source_file": entry["source_file"],
                "round1_review": entry.get("review", {}),
                "round1_cue_indices": entry["cue_indices"],
                "current_block": {
                    "cue_indices": [cue.index for cue in block.cues],
                    "source_cues": _serialize_cues(block.cues),
                    "source_text": _translated_text(block.cues),
                    "block_lint": {
                        "low_confidence": block.low_confidence,
                        "lint_reasons": block.lint_reasons,
                        "lint_actions": block.lint_actions,
                    },
                },
                "translation_output": {
                    "translated_cues": _serialize_cues(translated_cues),
                    "translated_text": _translated_text(translated_cues),
                },
                "pipeline_signals": {
                    "phase1_retried": metrics.phase1_retry_blocks > 0,
                    "repair_invoked": metrics.repair_invocations > 0,
                    "repair_accepted": metrics.repair_accepted > 0,
                    "repair_rejected": metrics.repair_rejected > 0,
                    "smaller_block_fallback": metrics.smaller_block_fallbacks > 0,
                    "single_cue_source_fallback": metrics.single_cue_source_fallbacks > 0,
                    "post_wrap_failure": metrics.post_wrap_failure_blocks > 0,
                    "failure_reasons": metrics.failure_reasons,
                    "pre_wrap_failures": metrics.pre_wrap_failures,
                    "post_wrap_failures": metrics.post_wrap_failures,
                    "average_cps": round(metrics.average_cps(), 3),
                },
                "round2_review": {
                    "status": "pending",
                    "failure_tags": [],
                    "notes": "",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote translated eval records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
