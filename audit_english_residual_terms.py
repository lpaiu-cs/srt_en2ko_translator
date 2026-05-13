#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from subtitle_translator import EmittedCue, TranslationBlock, load_runtime_config, pre_wrap_gate, read_srt


ENGLISH_RESIDUAL_ISSUES = ("english_residual", "english_residual_warn", "english_residual_technical")


def _lecture_key(path: Path) -> str:
    name = path.name.replace(".ko.srt", ".srt")
    marker = "Lecture "
    if marker in name:
        rest = name.split(marker, 1)[1]
        return rest.split()[0]
    return name


def _find_files(path: Path, pattern: str) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        return [path]
    return sorted(path.glob(pattern))


def _source_cues_for(path: Path, eng_by_key: dict[str, Path]) -> list:
    eng_path = eng_by_key.get(_lecture_key(path))
    return read_srt(str(eng_path)) if eng_path and eng_path.exists() else []


def _empty_source_like(cues: Iterable) -> list:
    return [
        type(cue)(index=cue.index, start=cue.start, end=cue.end, text="")
        for cue in cues
    ]


def build_inventory(kor_files: list[Path], eng_folder: Path | None, config) -> list[dict]:
    eng_by_key = {
        _lecture_key(path): path
        for path in (sorted(eng_folder.glob("*.srt")) if eng_folder else [])
    }
    inventory: dict[str, dict] = {}
    for kor_path in kor_files:
        kor_cues = read_srt(str(kor_path))
        source_cues = _source_cues_for(kor_path, eng_by_key) if eng_folder else []
        if len(source_cues) != len(kor_cues):
            source_cues = _empty_source_like(kor_cues)
        block = TranslationBlock(cues=source_cues)
        emitted = [EmittedCue(cue_index=cue.index, text=cue.text) for cue in kor_cues]
        gate = pre_wrap_gate(block, emitted, [], config)
        source_by_index = {cue.index: cue for cue in source_cues}
        for issue in ENGLISH_RESIDUAL_ISSUES:
            for detail in gate.warning_details.get(issue, []):
                normalized = detail["normalized_term"]
                row = inventory.setdefault(
                    normalized,
                    {
                        "term": detail["term"],
                        "normalized_term": normalized,
                        "count": 0,
                        "issues": {},
                        "files": {},
                        "examples": [],
                    },
                )
                row["count"] += 1
                row["issues"][issue] = row["issues"].get(issue, 0) + 1
                row["files"][kor_path.name] = row["files"].get(kor_path.name, 0) + 1
                if len(row["examples"]) < 5:
                    source_cue = source_by_index.get(detail["cue_index"])
                    row["examples"].append(
                        {
                            "file": kor_path.name,
                            "cue_index": detail["cue_index"],
                            "issue": issue,
                            "source_text": source_cue.text if source_cue else "",
                            "output_text": detail.get("output_text", ""),
                        }
                    )
    return sorted(inventory.values(), key=lambda row: (-row["count"], row["normalized_term"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect English residual terms from translated Korean SRT files.")
    parser.add_argument("kor_path", help="Korean .ko.srt file or folder")
    parser.add_argument("--eng-folder", default=None, help="Optional source English SRT folder for examples")
    parser.add_argument("--pattern", default="*.ko.srt", help="Glob pattern when kor_path is a folder")
    parser.add_argument(
        "--english-residual-policy",
        default=None,
        choices=["coarse", "technical_split"],
        help="Override SRT_ENGLISH_RESIDUAL_POLICY for this audit",
    )
    parser.add_argument("--output", default=None, help="Write JSON inventory to this path")
    args = parser.parse_args()

    kor_path = Path(args.kor_path).expanduser().resolve()
    eng_folder = Path(args.eng_folder).expanduser().resolve() if args.eng_folder else None
    config = load_runtime_config(glossary_log_path="")
    if args.english_residual_policy:
        config.english_residual_policy = args.english_residual_policy
    kor_files = _find_files(kor_path, args.pattern)
    if not kor_files:
        raise FileNotFoundError(f"No files matched {args.pattern} under {kor_path}")
    rows = build_inventory(kor_files, eng_folder, config)

    payload = {"terms": rows}
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"terms={len(rows)}")
    for row in rows:
        issue_summary = ", ".join(f"{key}:{value}" for key, value in sorted(row["issues"].items()))
        print(f"{row['term']}\tcount={row['count']}\t{issue_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
