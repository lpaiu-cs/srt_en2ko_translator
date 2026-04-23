#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _matches_family(row: dict, family: str) -> bool:
    meta = row.get("shipping_failure_meta") or {}
    return family in (meta.get("failure_families") or [])


def filter_rows(rows: Iterable[dict], family: str) -> list[dict]:
    return [row for row in rows if _matches_family(row, family)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a shipping failure corpus by failure family.")
    parser.add_argument("--input", required=True, help="Shipping failure corpus JSONL path")
    parser.add_argument(
        "--family",
        required=True,
        choices=["english_residual", "fallback_trigger", "wrap_readability"],
        help="Failure family to keep",
    )
    parser.add_argument("--output", required=True, help="JSONL output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.input).expanduser())
    selected = filter_rows(rows, args.family)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} {args.family} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
