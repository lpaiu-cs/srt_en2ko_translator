#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from subtitle_translator.quality import _source_tail_type


def _load_rows(paths: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _restore_tail_trace(row: dict) -> dict | None:
    trace = row.get("pipeline_signals", {}).get("style_retry_trace", {})
    if "restore_missing_tail" not in trace.get("preferred_actions", []):
        return None
    spans = [
        span
        for span in trace.get("offending_spans", [])
        if span.get("preferred_action") == "restore_missing_tail"
    ]
    if not spans:
        return None
    offending = set(trace.get("offending_cue_indices", []))
    protected = set(trace.get("protected_cue_indices", []))
    if not offending or offending & protected:
        return None
    return trace


def _primary_tail_type(trace: dict) -> str:
    for span in trace.get("offending_spans", []):
        if span.get("preferred_action") == "restore_missing_tail" and span.get("source_tail_type"):
            return str(span["source_tail_type"])
        if span.get("preferred_action") == "restore_missing_tail" and span.get("source_cue_text"):
            inferred = _source_tail_type(str(span["source_cue_text"]))
            if inferred:
                return inferred
    return "unknown"


def _outcome(row: dict) -> str:
    signals = row.get("pipeline_signals", {})
    if signals.get("style_retry_accepted"):
        return "accepted"
    if signals.get("style_retry_rejected"):
        return "rejected"
    return "attempted"


def _trace_quality_score(row: dict, trace: dict, tail_type: str) -> tuple[int, int, int, int]:
    protected = trace.get("protected_cue_indices", [])
    offending = trace.get("offending_cue_indices", [])
    return (
        int(tail_type != "unknown"),
        int(bool(protected)),
        -len(offending or []),
        int(row.get("pipeline_signals", {}).get("style_retry_invoked", False)),
    )


def _cue_text_map(cues: List[dict]) -> Dict[int, str]:
    return {
        int(cue.get("cue_index")): str(cue.get("text", ""))
        for cue in cues
        if isinstance(cue.get("cue_index"), int)
    }


def _select_cue_texts(cues: List[dict], cue_indices: Iterable[int]) -> Dict[int, str]:
    cue_map = _cue_text_map(cues)
    return {cue_index: cue_map.get(cue_index, "") for cue_index in cue_indices}


def _replay_trace(row: dict, trace: dict) -> dict:
    offending_cue_indices = list(trace.get("offending_cue_indices", []))
    return {
        "offending_cue_indices": offending_cue_indices,
        "protected_cue_indices": list(trace.get("protected_cue_indices", [])),
        "offending_spans": list(trace.get("offending_spans", [])),
        "accept_mode": trace.get("accept_mode"),
        "rejection_causes": list(trace.get("rejection_causes", [])),
        "base_phase1_offending_cues": _select_cue_texts(trace.get("base_phase1_emitted_cues", []), offending_cue_indices),
        "strict_candidate_raw_offending_cues": _select_cue_texts(trace.get("strict_candidate_raw_emitted_cues", []), offending_cue_indices),
        "strict_candidate_offending_cues": _select_cue_texts(trace.get("strict_candidate_emitted_cues", []), offending_cue_indices),
        "strict_candidate_post_normalizations": list(trace.get("strict_candidate_post_normalizations", [])),
        "final_offending_cues": _select_cue_texts(trace.get("final_emitted_cues", []), offending_cue_indices),
    }


def select_rows(rows: Iterable[dict], max_per_tail_type: int, max_total: int | None) -> List[dict]:
    best_by_id: Dict[str, tuple[tuple[int, int, int, int], int, dict]] = {}
    for order, row in enumerate(rows):
        trace = _restore_tail_trace(row)
        if trace is None:
            continue
        tail_type = _primary_tail_type(trace)
        enriched = dict(row)
        enriched["replay_meta"] = {
            "style_focus": "restore_missing_tail",
            "source_tail_type": tail_type,
            "style_retry_outcome": _outcome(row),
        }
        enriched["replay_trace"] = _replay_trace(row, trace)
        score = _trace_quality_score(row, trace, tail_type)
        current = best_by_id.get(row["id"])
        if current is None or score > current[0] or (score == current[0] and order > current[1]):
            best_by_id[row["id"]] = (score, order, enriched)

    buckets: Dict[str, List[dict]] = {}
    for _, _, row in best_by_id.values():
        buckets.setdefault(row["replay_meta"]["source_tail_type"], []).append(row)

    selected: List[dict] = []
    for tail_type in sorted(buckets):
        ranked = sorted(
            buckets[tail_type],
            key=lambda item: (
                item["replay_meta"]["style_retry_outcome"] != "rejected",
                item["lecture"],
                item["id"],
            ),
        )
        selected.extend(ranked[:max_per_tail_type] if max_per_tail_type > 0 else ranked)

    selected = sorted(selected, key=lambda item: (item["replay_meta"]["source_tail_type"], item["lecture"], item["id"]))
    if max_total is not None and max_total > 0:
        selected = selected[:max_total]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen replay set from runtime restore_missing_tail cases.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Translated eval JSONL files to scan in order; later files override earlier duplicates.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_restore_missing_tail_replay.jsonl",
        help="Replay-set JSONL output path",
    )
    parser.add_argument(
        "--max-per-tail-type",
        type=int,
        default=4,
        help="How many replay rows to keep per tail type (0 keeps all)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Optional overall cap after per-tail-type selection (0 keeps all)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows([Path(value).expanduser() for value in args.inputs])
    selected = select_rows(rows, max(0, args.max_per_tail_type), args.max_total or None)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} runtime replay rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
