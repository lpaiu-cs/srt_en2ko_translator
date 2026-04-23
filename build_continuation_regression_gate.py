#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_IDS = [
    "Stanford CS231N   Spring 2025   Lecture 11 Large Scale Distributed Training::block-0657",
    "Stanford CS231N Deep Learning for Computer Vision   Spring 2025   Lecture 16 Vision and Language::block-0078",
    "Stanford CS231N Deep Learning for Computer Vision   Spring 2025   Lecture 16 Vision and Language::block-0101",
    "Stanford CS231N Deep Learning for Computer Vision   Spring 2025   Lecture 16 Vision and Language::block-0124",
    "Stanford CS231N Deep Learning for Computer Vision   Spring 2025   Lecture 18 Human-Centered AI::block-0211",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the fixed continuation-tail regression gate from replay outputs.")
    parser.add_argument("--style-only", required=True, help="Style-only replay JSONL")
    parser.add_argument("--full-pipeline", required=True, help="Full-pipeline replay JSONL")
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_continuation_tail_regression_gate.json",
        help="Where to write the regression gate JSON",
    )
    return parser


def _load_rows(path: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _lane_expectation(row: dict) -> dict:
    signals = row.get("pipeline_signals", {})
    return {
        "transition": signals.get("replay_transition"),
        "stage": signals.get("style_retry_rejection_stage"),
        "subtype": signals.get("style_retry_rejection_subtype"),
        "not_invoked_reason": signals.get("style_retry_not_invoked_reason"),
    }


def main() -> int:
    args = build_parser().parse_args()
    style_rows = _load_rows(Path(args.style_only))
    full_rows = _load_rows(Path(args.full_pipeline))

    gate_rows = []
    for row_id in TARGET_IDS:
        style_row = style_rows[row_id]
        full_row = full_rows[row_id]
        gate_rows.append(
            {
                "id": row_id,
                "lecture": style_row.get("lecture"),
                "tail_type": "continuation_tail",
                "style_only": _lane_expectation(style_row),
                "full_pipeline": _lane_expectation(full_row),
            }
        )

    payload = {
        "schema_version": "continuation_regression_gate_v1",
        "source_style_only": args.style_only,
        "source_full_pipeline": args.full_pipeline,
        "rows": gate_rows,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote continuation regression gate to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
