#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from subtitle_translator import Cue, load_runtime_config, read_srt, write_srt
from subtitle_translator.config import _load_english_fallback_terms
from subtitle_translator.english_terms import apply_approved_english_fallbacks
from subtitle_translator.pipeline import _wrap_cue_with_local_rewrap


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


def _rewrap(cues: list[Cue], config) -> list[Cue]:
    return [_wrap_cue_with_local_rewrap(cue, cue.text, config, None) for cue in cues]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply approved English fallback aliases to existing Korean SRT files.")
    parser.add_argument("kor_path", help="Korean .ko.srt file or folder")
    parser.add_argument("--eng-folder", required=True, help="Source English SRT folder")
    parser.add_argument("--fallback-map", required=True, help="JSON fallback map: source English term -> Korean aliases")
    parser.add_argument("--pattern", default="*.ko.srt", help="Glob pattern when kor_path is a folder")
    parser.add_argument("--dry-run", action="store_true", help="Report replacements without writing files")
    args = parser.parse_args()

    kor_path = Path(args.kor_path).expanduser().resolve()
    eng_folder = Path(args.eng_folder).expanduser().resolve()
    fallback_map = Path(args.fallback_map).expanduser().resolve()
    config = load_runtime_config(glossary_log_path="")
    config.english_fallback_terms = _load_english_fallback_terms(fallback_map)
    if not config.english_fallback_terms:
        print("No fallback rules loaded.")
        return 0

    eng_by_key = {_lecture_key(path): path for path in sorted(eng_folder.glob("*.srt"))}
    total_replacements = 0
    kor_files = _find_files(kor_path, args.pattern)
    if not kor_files:
        raise FileNotFoundError(f"No files matched {args.pattern} under {kor_path}")
    for kor_file in kor_files:
        eng_file = eng_by_key.get(_lecture_key(kor_file))
        if not eng_file:
            print(f"SKIP missing English source: {kor_file.name}")
            continue
        source = read_srt(str(eng_file))
        output = read_srt(str(kor_file))
        if len(source) != len(output):
            print(f"SKIP cue count mismatch: {kor_file.name}")
            continue
        rewritten, replacements = apply_approved_english_fallbacks(source, output, config)
        if not replacements:
            continue
        total_replacements += len(replacements)
        print(f"{kor_file.name}: replacements={len(replacements)}")
        for replacement in replacements[:10]:
            print(f"  cue={replacement['cue_index']} {replacement['alias']} -> {replacement['source']}")
        if not args.dry_run:
            write_srt(_rewrap(rewritten, config), str(kor_file))
    print(f"total_replacements={total_replacements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
