#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MATCH_STAGES = {"strict_retry_overedit", "strict_retry_selector", "strict_retry_unknown"}
MATCH_SUBTYPES = {None, "local_meaning_not_restored"}
_PURPOSE = re.compile(r"^(to )", re.IGNORECASE)
_THAT = re.compile(r"^(that |because |if |while )", re.IGNORECASE)
_REL = re.compile(r"^(which |who |whom |whose |where |when |why )", re.IGNORECASE)
_COMP = re.compile(r"^(as |than |rather than |compared to |like )", re.IGNORECASE)
_CONT = re.compile(r"^(for |by |of |with |without |into |from )", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect stable-head continuation-tail rows that match the current overedit/local-meaning failure signature."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Translated replay JSONL files to scan",
    )
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_continuation_tail_signature_watchlist.jsonl",
        help="Where to write the collected signature rows",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional subset of rejection stages to keep",
    )
    parser.add_argument(
        "--subtypes",
        nargs="*",
        default=None,
        help="Optional subset of rejection subtypes to keep (use 'null' for missing subtype)",
    )
    return parser


def _row_tail_type(row: dict[str, Any]) -> str | None:
    replay_meta = row.get("replay_meta") or {}
    if replay_meta.get("source_tail_type"):
        return replay_meta.get("source_tail_type")
    trace = (row.get("pipeline_signals") or {}).get("style_retry_trace") or {}
    for span in trace.get("offending_spans", []) or []:
        if span.get("source_tail_type"):
            return span.get("source_tail_type")
    cues = ((row.get("current_block") or {}).get("source_cues") or [])
    if not cues:
        return None
    tail_text = " ".join(str(cues[-1].get("text", "")).split())
    if _PURPOSE.match(tail_text):
        return "purpose_tail"
    if _THAT.match(tail_text):
        return "that_clause_tail"
    if _REL.match(tail_text):
        return "relative_clause_tail"
    if _COMP.match(tail_text):
        return "comparison_tail"
    if _CONT.match(tail_text):
        return "continuation_tail"
    return None


def _eval_lane(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    return "full-pipeline" if provenance.get("repair_enabled", True) else "style-only"


def _matches(
    row: dict[str, Any],
    stages: set[str] | None = None,
    subtypes: set[str | None] | None = None,
) -> bool:
    if _row_tail_type(row) != "continuation_tail":
        return False

    signals = row.get("pipeline_signals") or {}
    replay_meta = row.get("replay_meta") or {}
    if replay_meta:
        if signals.get("replay_surface_state") != "surfaced_same_action":
            return False
    elif not signals.get("style_retry_invoked"):
        return False
    stage = signals.get("style_retry_rejection_stage")
    subtype = signals.get("style_retry_rejection_subtype")
    if stage not in MATCH_STAGES:
        return False
    if subtype not in MATCH_SUBTYPES:
        return False
    if stages is not None and stage not in stages:
        return False
    if subtypes is not None and subtype not in subtypes:
        return False

    trace = signals.get("style_retry_trace") or {}
    offending = trace.get("offending_cue_indices") or []
    protected = trace.get("protected_cue_indices") or []
    if len(offending) != 1 or not protected:
        return False

    return True


def _signature_row(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    signals = row.get("pipeline_signals") or {}
    trace = signals.get("style_retry_trace") or {}
    replay_meta = row.get("replay_meta") or {}
    surface_state = signals.get("replay_surface_state") or "current_capture"
    return {
        "id": row.get("id"),
        "lecture": row.get("lecture"),
        "source_file": source_file,
        "eval_lane": _eval_lane(row),
        "tail_type": _row_tail_type(row),
        "historical_outcome": replay_meta.get("style_retry_outcome"),
        "surface_state": surface_state,
        "transition": signals.get("replay_transition"),
        "rejection_stage": signals.get("style_retry_rejection_stage"),
        "rejection_subtype": signals.get("style_retry_rejection_subtype"),
        "offending_cue_indices": trace.get("offending_cue_indices"),
        "protected_cue_indices": trace.get("protected_cue_indices"),
        "base_phase1_emitted_cues": trace.get("base_phase1_emitted_cues"),
        "strict_candidate_emitted_cues": trace.get("strict_candidate_emitted_cues"),
        "final_emitted_cues": trace.get("final_emitted_cues"),
        "rejection_causes": trace.get("rejection_causes"),
        "strict_retry_mode": trace.get("strict_retry_mode"),
    }


def main() -> int:
    args = build_parser().parse_args()
    collected: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str, str]] = set()
    stage_filter = set(args.stages) if args.stages else None
    subtype_filter = None
    if args.subtypes:
        subtype_filter = {None if item == "null" else item for item in args.subtypes}

    for source in args.inputs:
        path = Path(source)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not _matches(row, stages=stage_filter, subtypes=subtype_filter):
                continue
            key = (source, row["id"], _eval_lane(row))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            collected.append(_signature_row(row, source))

    output_path = Path(args.output)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in collected) + ("\n" if collected else ""),
        encoding="utf-8",
    )
    print(f"Collected {len(collected)} continuation signature rows into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
