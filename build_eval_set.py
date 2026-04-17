#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from subtitle_translator import load_runtime_config, read_srt
from subtitle_translator.blocks import build_translation_blocks
from subtitle_translator.grouping import ABBREVIATIONS
from subtitle_translator.models import Cue, TranslationBlock
from subtitle_translator.splitting import ts_to_ms
from subtitle_translator.text import normalize_text


CATEGORY_ORDER = [
    "abbrev_decimal",
    "quote_paren",
    "incomplete_clause",
    "long_subordinate",
    "tech_term_dense",
    "numeric_formula",
    "pronoun_ellipsis",
    "tail_heavy_risk",
]

CONNECTOR_PREFIXES = (
    "and",
    "but",
    "because",
    "so",
    "or",
    "then",
    "which",
    "that",
    "who",
    "when",
    "while",
    "if",
    "though",
    "although",
    "where",
    "as",
)

INCOMPLETE_SUFFIXES = (
    "and",
    "or",
    "but",
    "so",
    "because",
    "that",
    "which",
    "who",
    "when",
    "while",
    "if",
    "to",
    "of",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "like",
)

TECH_TERMS = {
    "cnn",
    "cnns",
    "transformer",
    "transformers",
    "resnet",
    "imagenet",
    "pytorch",
    "numpy",
    "gradient",
    "gradients",
    "backpropagation",
    "optimizer",
    "optimizers",
    "sgd",
    "relu",
    "attention",
    "embedding",
    "diffusion",
    "gan",
    "clip",
    "llm",
    "rnn",
    "lstm",
    "gpu",
    "tpu",
    "batchnorm",
}

PRONOUNS = {
    "it",
    "this",
    "that",
    "they",
    "them",
    "these",
    "those",
    "he",
    "she",
    "we",
    "you",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*")
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_NUMERIC_RE = re.compile(r"\d[\d,./%+-]*")
_FORMULA_RE = re.compile(r"(?:=|%|\+|-|/|\bEq\.\s*\(\d+\)|\blayer\s+\d+\b)")
_QUOTE_PAREN_RE = re.compile(r"[\"“”'()\\[\\]{}]")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _duration_ms(cue: Cue) -> int:
    return max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1)


def _normalized_words(text: str) -> List[str]:
    return [token.casefold().strip(".,!?;:'\"") for token in _WORD_RE.findall(normalize_text(text))]


def _tail_heavy_score(block: TranslationBlock) -> float:
    if len(block.cues) < 2:
        return 0.0
    src_chars = [max(len(normalize_text(cue.text)), 1) for cue in block.cues]
    durations = [_duration_ms(cue) for cue in block.cues]
    total_src = sum(src_chars) or 1
    total_dur = sum(durations) or 1
    expected_last = 0.5 * (src_chars[-1] / total_src) + 0.5 * (durations[-1] / total_dur)
    observed_last = src_chars[-1] / total_src
    return max(0.0, observed_last - expected_last)


def score_block(block: TranslationBlock) -> Dict[str, float]:
    joined = normalize_text(" ".join(cue.text for cue in block.cues))
    words = _normalized_words(joined)
    word_counts = Counter(words)
    lowered = joined.casefold()
    scores: Dict[str, float] = {category: 0.0 for category in CATEGORY_ORDER}

    if any(abbrev + "." in lowered for abbrev in ABBREVIATIONS) or _DECIMAL_RE.search(joined):
        scores["abbrev_decimal"] += 1.0
    if "%" in joined:
        scores["abbrev_decimal"] += 0.5

    if _QUOTE_PAREN_RE.search(joined):
        scores["quote_paren"] += 1.0
        if joined.count("(") != joined.count(")") or joined.count('"') % 2 == 1:
            scores["quote_paren"] += 0.5

    if len(block.cues) >= 2:
        first_words = [_normalized_words(cue.text)[:2] for cue in block.cues[1:]]
        if any(tokens and " ".join(tokens).startswith(prefix) for tokens in first_words for prefix in CONNECTOR_PREFIXES):
            scores["incomplete_clause"] += 0.8
    stripped = lowered.rstrip(" .!?,'\"")
    if any(stripped.endswith(" " + suffix) or stripped == suffix for suffix in INCOMPLETE_SUFFIXES):
        scores["incomplete_clause"] += 1.0
    if joined and joined[-1] not in ".!?":
        scores["incomplete_clause"] += 0.3

    subordinate_hits = sum(word_counts.get(token, 0) for token in ("that", "which", "because", "if", "when", "while", "although", "where"))
    if subordinate_hits >= 2:
        scores["long_subordinate"] += 1.0
    if joined.count(",") >= 2 or len(joined) >= 120:
        scores["long_subordinate"] += 0.5

    tech_hits = sum(1 for token in words if token in TECH_TERMS)
    tech_hits += len(_ACRONYM_RE.findall(joined))
    if tech_hits >= 2:
        scores["tech_term_dense"] += min(2.0, tech_hits / 2.0)

    if _NUMERIC_RE.search(joined):
        scores["numeric_formula"] += 0.7
    if _FORMULA_RE.search(joined):
        scores["numeric_formula"] += 0.8

    pronoun_hits = sum(1 for token in words if token in PRONOUNS)
    if pronoun_hits >= 2:
        scores["pronoun_ellipsis"] += min(1.5, pronoun_hits / 2.0)
    if len(block.cues) >= 2 and any(len(_normalized_words(cue.text)) <= 3 for cue in block.cues):
        scores["pronoun_ellipsis"] += 0.3

    tail_score = _tail_heavy_score(block)
    if tail_score > 0.08:
        scores["tail_heavy_risk"] = min(1.5, 4.0 * tail_score)

    return scores


def build_entry(source_path: Path, block_index: int, block: TranslationBlock, scores: Dict[str, float]) -> dict:
    joined = normalize_text(" ".join(cue.text for cue in block.cues))
    tags = [category for category in CATEGORY_ORDER if scores[category] >= 0.8]
    if not tags:
        tags = [max(CATEGORY_ORDER, key=lambda category: scores[category])]

    return {
        "id": f"{source_path.stem}::block-{block_index:04d}",
        "source_file": str(source_path),
        "lecture": source_path.stem,
        "block_index": block_index,
        "cue_indices": [cue.index for cue in block.cues],
        "source_text": joined,
        "source_cues": [
            {
                "cue_index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": normalize_text(cue.text),
            }
            for cue in block.cues
        ],
        "block_lint": {
            "low_confidence": block.low_confidence,
            "lint_reasons": block.lint_reasons,
            "lint_actions": block.lint_actions,
        },
        "tags": tags,
        "tag_scores": {category: round(scores[category], 3) for category in CATEGORY_ORDER if scores[category] > 0},
        "review": {
            "status": "pending",
            "failure_tags": [],
            "notes": "",
        },
    }


def collect_candidates(input_dir: Path, include_all: bool = False) -> List[dict]:
    config = load_runtime_config(glossary_log_path="")
    config.use_context_window = False

    candidates: List[dict] = []
    for source_path in sorted(input_dir.glob("*.srt")):
        cues = read_srt(str(source_path))
        blocks = build_translation_blocks(cues, config)
        for block_index, block in enumerate(blocks, start=1):
            scores = score_block(block)
            if not include_all and max(scores.values()) <= 0:
                continue
            candidates.append(build_entry(source_path, block_index, block, scores))
    return candidates


def select_eval_set(candidates: List[dict], target_count: int) -> List[dict]:
    by_category: Dict[str, List[dict]] = defaultdict(list)
    for entry in candidates:
        for tag in entry["tags"]:
            by_category[tag].append(entry)

    for category in CATEGORY_ORDER:
        by_category[category].sort(
            key=lambda entry: (
                -entry["tag_scores"].get(category, 0.0),
                entry["source_file"],
                entry["block_index"],
            )
        )

    picked: List[dict] = []
    picked_ids = set()
    per_file = Counter()
    per_category_target = max(3, target_count // len(CATEGORY_ORDER))

    for category in CATEGORY_ORDER:
        for entry in by_category[category]:
            if len(picked) >= target_count:
                break
            if entry["id"] in picked_ids:
                continue
            if per_file[entry["source_file"]] >= 4:
                continue
            picked.append(entry)
            picked_ids.add(entry["id"])
            per_file[entry["source_file"]] += 1
            if sum(1 for current in picked if category in current["tags"]) >= per_category_target:
                break

    remaining = sorted(
        (entry for entry in candidates if entry["id"] not in picked_ids),
        key=lambda entry: (
            -sum(entry["tag_scores"].values()),
            len(entry["tags"]),
            entry["source_file"],
            entry["block_index"],
        ),
    )
    for entry in remaining:
        if len(picked) >= target_count:
            break
        if per_file[entry["source_file"]] >= 4:
            continue
        picked.append(entry)
        picked_ids.add(entry["id"])
        per_file[entry["source_file"]] += 1

    picked.sort(key=lambda entry: (entry["source_file"], entry["block_index"]))
    return picked[:target_count]


def select_random_eval_set(candidates: List[dict], target_count: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    picked: List[dict] = []
    per_file = Counter()
    for entry in shuffled:
        if len(picked) >= target_count:
            break
        if per_file[entry["source_file"]] >= 4:
            continue
        picked.append(entry)
        per_file[entry["source_file"]] += 1
    picked.sort(key=lambda entry: (entry["source_file"], entry["block_index"]))
    return picked[:target_count]


def write_jsonl(path: Path, entries: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CS231n subtitle evaluation set from real SRT files")
    parser.add_argument("--input-dir", default="cs231n_sp25/eng", help="Directory containing source English SRT files")
    parser.add_argument("--output", default="evaluation/cs231n_sp25_eval.jsonl", help="Output JSONL path")
    parser.add_argument("--target-count", type=int, default=40, help="Number of blocks to select")
    parser.add_argument(
        "--selection-mode",
        choices=["hard", "random"],
        default="hard",
        help="Select hard-case blocks or a random baseline",
    )
    parser.add_argument("--seed", type=int, default=231, help="Random seed used for random selection mode")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    candidates = collect_candidates(input_dir, include_all=args.selection_mode == "random")
    if args.selection_mode == "random":
        selected = select_random_eval_set(candidates, target_count=max(1, args.target_count), seed=args.seed)
    else:
        selected = select_eval_set(candidates, target_count=max(1, args.target_count))
    write_jsonl(output_path, selected)

    tag_counter = Counter(tag for entry in selected for tag in entry["tags"])
    lecture_counter = Counter(entry["lecture"] for entry in selected)
    print(f"Wrote {len(selected)} eval blocks to {output_path}")
    print("Tag distribution:")
    for tag in CATEGORY_ORDER:
        print(f"  {tag}: {tag_counter.get(tag, 0)}")
    print("Lecture coverage:")
    for lecture, count in lecture_counter.most_common(10):
        print(f"  {lecture}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
