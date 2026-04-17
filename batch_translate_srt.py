#!/usr/bin/env python3
"""
Batch translator for a folder of SRT files.

- Reuses the same translator and glossary log across files
- Preserves timestamps/indices; only text is translated/split back
- Reads OPENAI_API_KEY and optional prompt settings from .env
"""
from __future__ import annotations
import argparse
import sys
import traceback
import time
from pathlib import Path
from typing import List

from subtitle_translator import (
    build_translator,
    create_glossary_store,
    load_runtime_config,
    positive_int,
    read_srt,
    translate_srt,
    write_srt,
)


def find_files(root: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(root.rglob(pattern))
    return sorted(root.glob(pattern))


def process_file(path: Path, translator, glossary_store, config) -> Path:
    cues = read_srt(str(path))
    out_cues = translate_srt(
        cues,
        translator=translator,
        config=config,
        glossary_store=glossary_store,
    )
    out_path = path.with_name(path.stem + ".ko.srt")
    write_srt(out_cues, str(out_path))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Batch English→Korean SRT translator (folder)")
    ap.add_argument("folder", help="Folder containing .srt files")
    ap.add_argument("--pattern", default="*.srt", help="Glob pattern for input files (default: *.srt)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument(
        "--model",
        default=None,
        help="Phase1 model name (default: .env SRT_PHASE1_MODEL/SRT_OPENAI_MODEL or gpt-4.1-mini)",
    )
    ap.add_argument(
        "--repair-model",
        default=None,
        help="Phase2 repair model name (default: .env SRT_REPAIR_MODEL or gpt-4o)",
    )
    ap.add_argument(
        "--block-max-cues",
        "--batch-size",
        dest="block_max_cues",
        type=positive_int,
        default=None,
        help="Maximum cues per translation block (legacy alias: --batch-size)",
    )
    ap.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="Override OpenAI base URL")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output .ko.srt already exists")
    ap.add_argument("--retries", type=positive_int, default=3, help="Retries per file on failure (default: 3)")
    ap.add_argument(
        "--glossary-log-path",
        default=None,
        help="Override glossary JSONL log path (shared across the batch run)",
    )
    ap.add_argument(
        "--disable-context-window",
        "--disable-history-context",
        dest="disable_context_window",
        action="store_true",
        help="Do not provide surrounding source context to the model",
    )
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = find_files(root, args.pattern, args.recursive)
    if not files:
        print("No files matched.")
        return

    config = load_runtime_config(glossary_log_path=args.glossary_log_path)
    model = args.model or config.phase1_model
    repair_model = args.repair_model or config.repair_model
    if args.disable_context_window:
        config.use_context_window = False
    if args.block_max_cues is not None:
        config.block_max_cues = max(config.block_min_cues, args.block_max_cues)
    translator = build_translator(config=config, model=model, repair_model=repair_model, base_url=args.openai_base_url)
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)

    total = len(files)
    print(f"Found {total} file(s). Starting...\n")

    skip = 0
    done = 0
    failed = 0
    for i, f in enumerate(files, 1):
        # Compute output path FIRST so it exists for logging/skip checks
        out_path = f.with_name(f.stem + ".ko.srt")

        if args.skip_existing and out_path.exists():
            print(f"[{i}/{total}] SKIP  : {f.name} → {out_path.name} (already exists)")
            done += 1
            skip += 1
            continue

        print(f"[{i}/{total}] START : {f}")
        for attempt in range(1, args.retries + 1):
            try:
                out_path = process_file(
                    f,
                    translator=translator,
                    glossary_store=glossary_store,
                    config=config,
                )
                print(f"[{i}/{total}] DONE  : {f.name} → {out_path.name}")
                done += 1
                break
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                raise
            except Exception as e:
                if attempt < args.retries:
                    wait = min(5 * attempt, 20)
                    print(f"[{i}/{total}] WARN  : {f.name} (attempt {attempt}/{args.retries}) -> {e}")
                    traceback.print_exc()
                    print(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    failed += 1
                    print(f"[{i}/{total}] ERROR : {f}")
                    traceback.print_exc()
                    break

    print("\nSummary:")
    print(f"  Skipped   : {skip}")
    print(f"  Translated: {done - skip}")
    print(f"  Failed    : {failed}")


if __name__ == "__main__":
    main()
