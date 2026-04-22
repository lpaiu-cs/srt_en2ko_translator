#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from subtitle_translator import load_runtime_config, read_srt
from subtitle_translator.blocks import build_translation_blocks
from subtitle_translator.models import Cue, TranslationBlock
from subtitle_translator.quality import _source_tail_type
from subtitle_translator.text import normalize_text


def _word_count(text: str) -> int:
    return len([token for token in normalize_text(text).split() if token])


def _score_block(block: TranslationBlock, tail_type: str) -> float:
    tail = normalize_text(block.cues[-1].text)
    head = normalize_text(" ".join(cue.text for cue in block.cues[:-1]))
    score = 0.0
    score += 2.0 if len(block.cues) == 2 else 1.0
    score += max(0.0, 8 - _word_count(tail)) * 0.2
    score += min(2.0, _word_count(head) / 8.0)
    if tail_type in {"purpose_tail", "that_clause_tail", "relative_clause_tail", "comparison_tail"}:
        score += 1.0
    if "carry_context_only" in block.lint_actions:
        score += 0.5
    if "dependent_end" in block.lint_reasons or "dependent_start" in block.lint_reasons:
        score += 0.5
    return round(score, 3)


def _serialize_cues(cues: List[Cue]) -> List[dict]:
    return [
        {
            "cue_index": cue.index,
            "start": cue.start,
            "end": cue.end,
            "text": normalize_text(cue.text),
        }
        for cue in cues
    ]


def _entry(source_path: Path, block_index: int, block: TranslationBlock, tail_type: str, score: float) -> dict:
    return {
        "id": f"{source_path.stem}::block-{block_index:04d}",
        "source_file": str(source_path),
        "lecture": source_path.stem,
        "block_index": block_index,
        "cue_indices": [cue.index for cue in block.cues],
        "source_text": normalize_text(" ".join(cue.text for cue in block.cues)),
        "source_cues": _serialize_cues(block.cues),
        "block_lint": {
            "low_confidence": block.low_confidence,
            "lint_reasons": list(block.lint_reasons),
            "lint_actions": list(block.lint_actions),
        },
        "style_focus": "restore_missing_tail",
        "source_tail_type": tail_type,
        "stress_score": score,
        "tags": ["restore_missing_tail", tail_type],
        "review": {
            "status": "pending",
            "failure_tags": [],
            "notes": "",
        },
    }


def collect_candidates(input_dir: Path) -> List[dict]:
    config = load_runtime_config(glossary_log_path="")
    config.use_context_window = False
    candidates: List[dict] = []
    for source_path in sorted(input_dir.glob("*.srt")):
        cues = read_srt(str(source_path))
        blocks = build_translation_blocks(cues, config)
        for block_index, block in enumerate(blocks, start=1):
            if len(block.cues) < 2:
                continue
            tail_type = _source_tail_type(block.cues[-1].text)
            if not tail_type:
                continue
            score = _score_block(block, tail_type)
            candidates.append(_entry(source_path, block_index, block, tail_type, score))
    return candidates


def select_stress_set(candidates: List[dict], per_tail_type: int) -> List[dict]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for entry in candidates:
        buckets[entry["source_tail_type"]].append(entry)
    selected: List[dict] = []
    for tail_type in sorted(buckets):
        ranked = sorted(
            buckets[tail_type],
            key=lambda item: (
                -float(item["stress_score"]),
                item["lecture"],
                item["block_index"],
            ),
        )
        selected.extend(ranked[:per_tail_type])
    return sorted(selected, key=lambda item: (item["lecture"], item["block_index"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen eval stress set for restore_missing_tail.")
    parser.add_argument("--input-dir", default="cs231n_sp25/eng", help="Directory of English SRT files")
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_restore_missing_tail_stress.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--per-tail-type",
        type=int,
        default=4,
        help="How many blocks to keep per source tail type",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates = collect_candidates(Path(args.input_dir).expanduser())
    selected = select_stress_set(candidates, max(1, args.per_tail_type))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} restore-missing-tail stress rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
