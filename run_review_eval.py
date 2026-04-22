#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from subtitle_translator import (
    TranslationMetrics,
    build_translator,
    create_glossary_store,
    hydrate_translation_block,
    load_runtime_config,
    read_srt,
)
from subtitle_translator.blocks import build_translation_blocks
from subtitle_translator.models import Cue, PhaseTranslationResult, RepairRequest, TranslationBlock, TranslationRequest
from subtitle_translator.pipeline import _translate_block_recursive
from subtitle_translator.text import normalize_text
from subtitle_translator.translators import BaseTranslator


class TracingTranslator(BaseTranslator):
    def __init__(self, inner: BaseTranslator):
        self.inner = inner
        self.phase1_risk_flags: List[str] = []
        self.repair_risk_flags: List[str] = []

    def reset(self) -> None:
        self.phase1_risk_flags = []
        self.repair_risk_flags = []

    def translate_block(self, request: TranslationRequest) -> PhaseTranslationResult:
        result = self.inner.translate_block(request)
        self.phase1_risk_flags.extend(result.risk_flags)
        return result

    def repair_block(self, request: RepairRequest) -> PhaseTranslationResult:
        result = self.inner.repair_block(request)
        self.repair_risk_flags.extend(result.risk_flags)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the current translator on a reviewed eval-set JSONL.")
    parser.add_argument(
        "--input",
        default="evaluation/cs231n_sp25_eval_review_round1.jsonl",
        help="Reviewed eval-set JSONL or prior translated eval JSONL to replay",
    )
    parser.add_argument(
        "--output",
        default="evaluation/cs231n_sp25_eval_translated_round2.jsonl",
        help="Where to write translated evaluation records",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Phase1 model name (default: .env SRT_PHASE1_MODEL/SRT_OPENAI_MODEL or gpt-4.1-mini)",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Phase2 repair model name (default: .env SRT_REPAIR_MODEL or gpt-4o)",
    )
    parser.add_argument(
        "--phase1-temperature",
        type=float,
        default=None,
        help="Override Phase1 temperature (use 0.0 for lower A/B eval noise)",
    )
    parser.add_argument(
        "--repair-temperature",
        type=float,
        default=None,
        help="Override repair temperature",
    )
    parser.add_argument(
        "--prompt-profile",
        default=None,
        help="Override Phase1 prompt profile (for example: fragment_preserving_v2)",
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Turn off Phase2 repair to isolate Phase1/style-retry behavior",
    )
    parser.add_argument(
        "--frozen-blocks",
        action="store_true",
        help="Replay the exact block stored in the input JSONL instead of rematching with the current block builder",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Override OpenAI base URL",
    )
    parser.add_argument(
        "--glossary-log-path",
        default=None,
        help="Override glossary JSONL log path",
    )
    return parser


def _load_review_entries(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _serialize_cues(cues: Iterable[Cue]) -> List[dict]:
    return [
        {
            "cue_index": cue.index,
            "start": cue.start,
            "end": cue.end,
            "text": cue.text,
        }
        for cue in cues
    ]


def _translated_text(cues: Iterable[Cue]) -> str:
    return " ".join(normalize_text(cue.text) for cue in cues if normalize_text(cue.text))


def _git_sha() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd="/Users/lpaiu/study/25-2/Translator",
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or "unknown"
    except Exception:
        return "unknown"


def _block_dict_from_entry(entry: dict) -> dict:
    return entry.get("current_block") or {
        "cue_indices": entry["cue_indices"],
        "source_cues": entry["source_cues"],
        "source_text": entry.get("source_text", ""),
        "block_lint": entry.get("block_lint", {"low_confidence": False, "lint_reasons": [], "lint_actions": []}),
    }


def _match_block(target_indices: List[int], blocks: List[TranslationBlock]) -> TranslationBlock:
    best: Tuple[int, int, float, TranslationBlock] | None = None
    target_set = set(target_indices)
    target_mid = sum(target_indices) / max(len(target_indices), 1)
    for block in blocks:
        block_indices = [cue.index for cue in block.cues]
        overlap = len(target_set & set(block_indices))
        if overlap == 0:
            continue
        size_penalty = -abs(len(block_indices) - len(target_indices))
        block_mid = sum(block_indices) / max(len(block_indices), 1)
        distance_penalty = -abs(block_mid - target_mid)
        candidate = (overlap, size_penalty, distance_penalty, block)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError(f"Could not match block for cue indices {target_indices}")
    return best[3]


def _load_cues_by_source(entries: List[dict]) -> Dict[str, List[Cue]]:
    cues_by_source: Dict[str, List[Cue]] = {}
    for source_file in sorted({entry["source_file"] for entry in entries}):
        cues_by_source[source_file] = read_srt(source_file)
    return cues_by_source


def _build_dynamic_blocks(entries: List[dict], config) -> Dict[str, List[TranslationBlock]]:
    blocks_by_lecture: Dict[str, List[TranslationBlock]] = {}
    for lecture in sorted({entry["lecture"] for entry in entries}):
        lecture_entries = [entry for entry in entries if entry["lecture"] == lecture]
        source_file = Path(lecture_entries[0]["source_file"])
        cues = read_srt(source_file)
        blocks_by_lecture[lecture] = build_translation_blocks(cues, config)
    return blocks_by_lecture


def _hydrate_frozen_block(entry: dict, all_cues: List[Cue], config) -> TranslationBlock:
    block_data = _block_dict_from_entry(entry)
    cue_index_set = set(block_data["cue_indices"])
    block_cues = [cue for cue in all_cues if cue.index in cue_index_set]
    block_lint = block_data.get("block_lint", {})
    return hydrate_translation_block(
        block_cues,
        all_cues,
        config,
        low_confidence=bool(block_lint.get("low_confidence", False)),
        lint_reasons=list(block_lint.get("lint_reasons", [])),
        lint_actions=list(block_lint.get("lint_actions", [])),
    )


def _input_review(entry: dict) -> dict:
    return entry.get("review") or entry.get("round2_review") or entry.get("round1_review") or {}


def _input_cue_indices(entry: dict) -> List[int]:
    if "cue_indices" in entry:
        return entry["cue_indices"]
    if "round1_cue_indices" in entry:
        return entry["round1_cue_indices"]
    block_data = _block_dict_from_entry(entry)
    return block_data.get("cue_indices", [])


def main() -> int:
    args = build_parser().parse_args()
    review_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    config = load_runtime_config(glossary_log_path=args.glossary_log_path)
    if args.phase1_temperature is not None:
        config.phase1_temperature = max(0.0, args.phase1_temperature)
    if args.repair_temperature is not None:
        config.repair_temperature = max(0.0, args.repair_temperature)
    if args.prompt_profile:
        config.phase1_prompt_profile = args.prompt_profile
    if args.disable_repair:
        config.repair_enabled = False

    tracing_translator = TracingTranslator(
        build_translator(
            config=config,
            model=args.model or config.phase1_model,
            repair_model=args.repair_model or config.repair_model,
            base_url=args.openai_base_url,
        )
    )
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)

    review_entries = _load_review_entries(review_path)
    cues_by_source = _load_cues_by_source(review_entries)
    blocks_by_lecture = None if args.frozen_blocks else _build_dynamic_blocks(review_entries, config)

    provenance = {
        "schema_version": "translated_eval_record_v2",
        "phase1_model": args.model or config.phase1_model,
        "repair_model": args.repair_model or config.repair_model,
        "phase1_temperature": config.phase1_temperature,
        "repair_temperature": config.repair_temperature,
        "prompt_profile": config.phase1_prompt_profile,
        "repair_enabled": config.repair_enabled,
        "git_sha": _git_sha(),
        "frozen_block_input": args.frozen_blocks,
        "openai_base_url": args.openai_base_url,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in review_entries:
            all_cues = cues_by_source[entry["source_file"]]
            if args.frozen_blocks:
                block = _hydrate_frozen_block(entry, all_cues, config)
            else:
                block = _match_block(_input_cue_indices(entry), blocks_by_lecture[entry["lecture"]])

            tracing_translator.reset()
            metrics = TranslationMetrics()
            translated_cues = _translate_block_recursive(
                block,
                tracing_translator,
                config,
                glossary_store,
                metrics,
                depth=0,
                ancestor_failure_signatures=(),
            )
            metrics.note_final_cues(translated_cues)
            record = {
                "schema_version": "translated_eval_record_v2",
                "id": entry["id"],
                "lecture": entry["lecture"],
                "source_file": entry["source_file"],
                "source_review": _input_review(entry),
                "input_cue_indices": _input_cue_indices(entry),
                "current_block": {
                    "cue_indices": [cue.index for cue in block.cues],
                    "source_cues": _serialize_cues(block.cues),
                    "source_text": _translated_text(block.cues),
                    "block_lint": {
                        "low_confidence": block.low_confidence,
                        "lint_reasons": block.lint_reasons,
                        "lint_actions": block.lint_actions,
                    },
                },
                "translation_output": {
                    "translated_cues": _serialize_cues(translated_cues),
                    "translated_text": _translated_text(translated_cues),
                },
                "pipeline_signals": {
                    "phase1_retried": metrics.phase1_retry_blocks > 0,
                    "style_retry_invoked": metrics.style_retry_invocations > 0,
                    "style_retry_accepted": metrics.style_retry_accepted > 0,
                    "style_retry_rejected": metrics.style_retry_rejected > 0,
                    "repair_invoked": metrics.repair_invocations > 0,
                    "repair_accepted": metrics.repair_accepted > 0,
                    "repair_rejected": metrics.repair_rejected > 0,
                    "smaller_block_fallback": metrics.smaller_block_fallbacks > 0,
                    "single_cue_source_fallback": metrics.single_cue_source_fallbacks > 0,
                    "post_wrap_failure": metrics.post_wrap_failure_blocks > 0,
                    "failure_reasons": metrics.failure_reasons,
                    "pre_wrap_failures": metrics.pre_wrap_failures,
                    "post_wrap_failures": metrics.post_wrap_failures,
                    "phase1_risk_flags": dict(metrics.phase1_risk_flags),
                    "strict_retry_candidate_risk_flags": dict(metrics.strict_retry_candidate_risk_flags),
                    "style_retry_rejection_causes": dict(metrics.style_retry_rejection_causes),
                    "style_action_attempts": dict(metrics.style_action_attempts),
                    "style_action_accepts": dict(metrics.style_action_accepts),
                    "style_action_rejections": dict(metrics.style_action_rejections),
                    "style_action_remaining_warnings": dict(metrics.style_action_remaining_warnings),
                    "style_action_tail_attempts": dict(metrics.style_action_tail_attempts),
                    "style_action_tail_accepts": dict(metrics.style_action_tail_accepts),
                    "style_action_tail_rejections": dict(metrics.style_action_tail_rejections),
                    "style_action_attempts_by_channel": dict(metrics.style_action_attempts_by_channel),
                    "style_action_accepts_by_channel": dict(metrics.style_action_accepts_by_channel),
                    "style_action_rejections_by_channel": dict(metrics.style_action_rejections_by_channel),
                    "style_action_remaining_warnings_by_channel": dict(metrics.style_action_remaining_warnings_by_channel),
                    "style_action_tail_attempts_by_channel": dict(metrics.style_action_tail_attempts_by_channel),
                    "style_action_tail_accepts_by_channel": dict(metrics.style_action_tail_accepts_by_channel),
                    "style_action_tail_rejections_by_channel": dict(metrics.style_action_tail_rejections_by_channel),
                    "captured_phase1_risk_flags": sorted(set(tracing_translator.phase1_risk_flags)),
                    "average_cps": round(metrics.average_cps(), 3),
                    "style_retry_trace": metrics.style_retry_trace,
                },
                "provenance": provenance,
                "review": {
                    "status": "pending",
                    "failure_tags": [],
                    "notes": "",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote translated eval records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
