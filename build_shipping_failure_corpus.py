#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


READABILITY_FAILURES = {
    "line_overflow",
    "cps_overflow_severe",
    "cps_warn",
    "bad_line_break",
    "line_imbalance",
}

ENGLISH_FAILURES = {
    "english_residual",
    "english_residual_warn",
}


def _load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _benchmark_label(path: Path) -> str:
    name = path.name
    if "hard40" in name:
        return "hard40"
    if "random40" in name:
        return "random40"
    if "heldout10_internal" in name:
        return "heldout10_internal"
    return path.stem


def _lane_label(path: Path) -> str:
    name = path.name
    if "style_only" in name:
        return "style-only"
    if "full_pipeline" in name:
        return "full-pipeline"
    return "unknown"


def _selected(ps: dict) -> bool:
    return bool(
        (ps.get("repair_invoked") and not ps.get("repair_accepted"))
        or ps.get("smaller_block_fallback")
        or ps.get("post_wrap_failure")
    )


def _failure_families(ps: dict) -> list[str]:
    families: list[str] = []
    failure_reasons = set((ps.get("failure_reasons") or {}).keys())
    pre_wrap = set((ps.get("pre_wrap_failures") or {}).keys())
    post_wrap = set((ps.get("post_wrap_failures") or {}).keys())

    if failure_reasons & ENGLISH_FAILURES or pre_wrap & ENGLISH_FAILURES:
        families.append("english_residual")
    if ps.get("smaller_block_fallback"):
        families.append("fallback_trigger")
    if (
        ps.get("post_wrap_failure")
        or failure_reasons & READABILITY_FAILURES
        or pre_wrap & READABILITY_FAILURES
        or post_wrap & READABILITY_FAILURES
    ):
        families.append("wrap_readability")
    return families


def _selection_reasons(ps: dict) -> list[str]:
    reasons: list[str] = []
    if ps.get("repair_invoked") and not ps.get("repair_accepted"):
        reasons.append("repair_rejected")
    if ps.get("smaller_block_fallback"):
        reasons.append("smaller_block_fallback")
    if ps.get("post_wrap_failure"):
        reasons.append("post_wrap_failure")
    return reasons


def _shipping_meta(row: dict, source_path: Path) -> dict:
    ps = row.get("pipeline_signals") or {}
    return {
        "benchmarks": [_benchmark_label(source_path)],
        "lanes": [_lane_label(source_path)],
        "source_runs": [source_path.name],
        "selection_reasons": _selection_reasons(ps),
        "failure_families": _failure_families(ps),
        "failure_reasons": ps.get("failure_reasons") or {},
        "pre_wrap_failures": ps.get("pre_wrap_failures") or {},
        "post_wrap_failures": ps.get("post_wrap_failures") or {},
        "repair_invoked": bool(ps.get("repair_invoked")),
        "repair_accepted": bool(ps.get("repair_accepted")),
        "smaller_block_fallback": bool(ps.get("smaller_block_fallback")),
        "post_wrap_failure": bool(ps.get("post_wrap_failure")),
    }


def collect_rows(inputs: Iterable[Path]) -> list[dict]:
    selected_by_id: dict[str, dict] = {}
    for input_path in inputs:
        for row in _load_rows(input_path):
            ps = row.get("pipeline_signals") or {}
            if not _selected(ps):
                continue
            row_id = str(row.get("id"))
            current_meta = _shipping_meta(row, input_path)
            existing = selected_by_id.get(row_id)
            if existing is None:
                enriched = dict(row)
                enriched["shipping_failure_meta"] = current_meta
                selected_by_id[row_id] = enriched
                continue

            meta = existing["shipping_failure_meta"]
            meta["benchmarks"] = sorted(set(meta["benchmarks"]) | set(current_meta["benchmarks"]))
            meta["lanes"] = sorted(set(meta["lanes"]) | set(current_meta["lanes"]))
            meta["source_runs"] = sorted(set(meta["source_runs"]) | set(current_meta["source_runs"]))
            meta["selection_reasons"] = sorted(set(meta["selection_reasons"]) | set(current_meta["selection_reasons"]))
            meta["failure_families"] = sorted(set(meta["failure_families"]) | set(current_meta["failure_families"]))
            for key in ("failure_reasons", "pre_wrap_failures", "post_wrap_failures"):
                target = meta[key]
                for reason, count in current_meta[key].items():
                    target[reason] = max(int(target.get(reason, 0)), int(count))
            meta["repair_invoked"] = meta["repair_invoked"] or current_meta["repair_invoked"]
            meta["repair_accepted"] = meta["repair_accepted"] or current_meta["repair_accepted"]
            meta["smaller_block_fallback"] = meta["smaller_block_fallback"] or current_meta["smaller_block_fallback"]
            meta["post_wrap_failure"] = meta["post_wrap_failure"] or current_meta["post_wrap_failure"]

    return [selected_by_id[row_id] for row_id in sorted(selected_by_id)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a shipping-lane failure corpus from translated eval JSONL outputs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Translated eval JSONL files to scan",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL output path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = [Path(value).expanduser() for value in args.inputs]
    rows = collect_rows(inputs)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} shipping failure rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
