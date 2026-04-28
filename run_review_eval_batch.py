#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from subtitle_translator import (
    Cue,
    EmittedCue,
    GlossaryEntry,
    PhaseTranslationResult,
    TranslationBlock,
    TranslationRequest,
    build_translator,
    create_glossary_store,
    load_runtime_config,
    read_srt,
)
from subtitle_translator.openai_batch import OpenAIBatchClient, parse_batch_output_line
from subtitle_translator.pipeline import (
    _apply_deterministic_style_micro_edits,
    _apply_purpose_tail_post_normalization,
    _candidate_is_overedited,
    _choose_better_style_candidate,
    _fallback_source_result,
    _offending_cue_diffs,
    _protected_cue_indices_for_spans,
    _serialize_emitted_cues,
    _strict_accept_mode,
    _style_spans_for_actions,
    _style_warning_action_details,
    _style_retry_candidate_reasons,
    _style_retry_feedback,
    _wrap_phase_result,
)
from subtitle_translator.quality import post_wrap_gate, pre_wrap_gate, validate_phase_structure
from subtitle_translator.splitting import ts_to_ms
from subtitle_translator.text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-oriented frozen-block eval runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_phase1 = subparsers.add_parser("prepare-phase1", help="Build phase1 batch requests for frozen eval blocks.")
    prepare_phase1.add_argument("--input", required=True, help="Eval JSONL input")
    prepare_phase1.add_argument("--requests-out", required=True, help="Batch request JSONL output")
    prepare_phase1.add_argument("--manifest-out", required=True, help="Phase1 manifest JSONL output")
    prepare_phase1.add_argument("--model", default=None, help="Phase1 model override")
    prepare_phase1.add_argument("--repair-model", default=None, help="Repair model override for provenance")
    prepare_phase1.add_argument("--phase1-temperature", type=float, default=None, help="Phase1 temperature override")
    prepare_phase1.add_argument("--prompt-profile", default=None, help="Prompt profile override")
    prepare_phase1.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    prepare_phase1.add_argument("--glossary-log-path", default=None)

    prepare_retry = subparsers.add_parser("prepare-style-retry", help="Build strict style-retry batch requests from phase1 batch output.")
    prepare_retry.add_argument("--phase1-manifest", required=True)
    prepare_retry.add_argument("--phase1-output", required=True)
    prepare_retry.add_argument("--requests-out", required=True)
    prepare_retry.add_argument("--manifest-out", required=True)
    prepare_retry.add_argument("--model", default=None)
    prepare_retry.add_argument("--repair-model", default=None)
    prepare_retry.add_argument("--phase1-temperature", type=float, default=None)
    prepare_retry.add_argument("--prompt-profile", default=None)
    prepare_retry.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    prepare_retry.add_argument("--glossary-log-path", default=None)

    finalize = subparsers.add_parser("finalize", help="Assemble final translated eval JSONL from phase1/strict batch outputs.")
    finalize.add_argument("--retry-manifest", required=True)
    finalize.add_argument("--strict-output", default=None, help="Strict retry batch output JSONL")
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--model", default=None)
    finalize.add_argument("--repair-model", default=None)
    finalize.add_argument("--phase1-temperature", type=float, default=None)
    finalize.add_argument("--prompt-profile", default=None)
    finalize.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    finalize.add_argument("--glossary-log-path", default=None)
    finalize.add_argument("--phase1-batch-json", default=None, help="Optional JSON written by submit/status for phase1 batch provenance")
    finalize.add_argument("--strict-batch-json", default=None, help="Optional JSON written by submit/status for strict batch provenance")

    submit = subparsers.add_parser("submit", help="Upload a batch input file and create a Batch job.")
    submit.add_argument("--input-jsonl", required=True)
    submit.add_argument("--metadata-json", default=None, help="Optional metadata JSON string")
    submit.add_argument("--endpoint", default="/v1/chat/completions")
    submit.add_argument("--completion-window", default="24h")
    submit.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    submit.add_argument("--output-json", default=None, help="Optional path to save create-batch response JSON")

    status = subparsers.add_parser("status", help="Retrieve or optionally wait for a Batch job.")
    status.add_argument("--batch-id", required=True)
    status.add_argument("--wait", action="store_true")
    status.add_argument("--poll-interval-seconds", type=int, default=30)
    status.add_argument("--timeout-seconds", type=int, default=60 * 60 * 24)
    status.add_argument("--openai-base-url", default="https://api.openai.com/v1")

    download = subparsers.add_parser("download", help="Download a Batch output or error file.")
    download.add_argument("--file-id", required=True)
    download.add_argument("--output", required=True)
    download.add_argument("--openai-base-url", default="https://api.openai.com/v1")

    return parser


def _load_entries(path: Path) -> List[dict]:
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


def _serialize_glossary_terms(terms: List[GlossaryEntry]) -> List[dict]:
    return [{"source": term.source, "target": term.target, "note": term.note, "mode": term.mode} for term in terms]


def _deserialize_glossary_terms(items: List[dict]) -> List[GlossaryEntry]:
    return [
        GlossaryEntry(
            source=item["source"],
            target=item["target"],
            note=item.get("note", ""),
            mode=item.get("mode", "soft"),
        )
        for item in items
    ]


def _translated_text(cues: Iterable[Cue]) -> str:
    return " ".join(normalize_text(cue.text) for cue in cues if normalize_text(cue.text))


def _average_cps(cues: Iterable[Cue]) -> float:
    total = 0.0
    count = 0
    for cue in cues:
        duration_ms = max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1)
        visible_chars = len(normalize_text(cue.text).replace(" ", ""))
        total += visible_chars / (duration_ms / 1000.0)
        count += 1
    return total / max(count, 1)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd="/Users/lpaiu/study/25-2/Translator",
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_cues_by_source(entries: List[dict]) -> Dict[str, List[Cue]]:
    cues_by_source: Dict[str, List[Cue]] = {}
    for source_file in sorted({entry["source_file"] for entry in entries}):
        cues_by_source[source_file] = read_srt(source_file)
    return cues_by_source


def _cue_from_dict(item: dict) -> Cue:
    return Cue(index=item["cue_index"], start=item["start"], end=item["end"], text=item["text"])


def _emitted_from_dicts(items: List[dict]) -> List[EmittedCue]:
    return [EmittedCue(cue_index=item["cue_index"], text=item["text"]) for item in items]


def _phase_result_from_dict(data: dict) -> PhaseTranslationResult:
    return PhaseTranslationResult(
        emitted_cues=_emitted_from_dicts(data.get("emitted_cues", [])),
        risk_flags=list(data.get("risk_flags", [])),
    )


def _phase_result_to_dict(result: PhaseTranslationResult) -> dict:
    return {
        "emitted_cues": _serialize_emitted_cues(result),
        "risk_flags": list(result.risk_flags),
    }


def _hydrate_block(record: dict) -> TranslationBlock:
    block_data = record["current_block"]
    lint = block_data.get("block_lint", {})
    return TranslationBlock(
        cues=[_cue_from_dict(cue) for cue in block_data["source_cues"]],
        previous_source_sentences=list(block_data.get("previous_source_sentences", [])),
        next_source_sentences=list(block_data.get("next_source_sentences", [])),
        low_confidence=bool(lint.get("low_confidence", False)),
        lint_reasons=list(lint.get("lint_reasons", [])),
        lint_actions=list(lint.get("lint_actions", [])),
    )


def _parse_batch_output(path: Path) -> Dict[str, dict]:
    outputs: Dict[str, dict] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        row = parse_batch_output_line(raw_line)
        if row and row.get("custom_id"):
            outputs[row["custom_id"]] = row
    return outputs


def _parse_batch_phase_result(translator, batch_row: dict) -> tuple[PhaseTranslationResult | None, List[str]]:
    if not batch_row:
        return None, ["missing_batch_result"]
    if batch_row.get("error"):
        return None, ["batch_error"]
    response = batch_row.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body")
    if status_code != 200 or not isinstance(body, dict):
        return None, [f"http_{status_code or 'unknown'}"]
    try:
        return translator.parse_chat_completion_response_body(body), []
    except Exception as exc:
        return None, [f"parse_error:{type(exc).__name__}"]


def _style_action_counter(spans: List[dict]) -> tuple[Dict[str, int], Dict[str, int]]:
    action_counts: Dict[str, int] = {}
    tail_counts: Dict[str, int] = {}
    for span in spans:
        action = span.get("preferred_action")
        if not action:
            continue
        action_counts[action] = action_counts.get(action, 0) + 1
        tail_type = span.get("source_tail_type")
        if tail_type:
            keyed = f"{action}|{tail_type}"
            tail_counts[keyed] = tail_counts.get(keyed, 0) + 1
    return action_counts, tail_counts


def _style_action_counter_with_accept_mode(spans: List[dict], accept_mode: str) -> tuple[Dict[str, int], Dict[str, int]]:
    action_counts: Dict[str, int] = {}
    tail_counts: Dict[str, int] = {}
    for span in spans:
        action = span.get("preferred_action")
        if not action:
            continue
        action_key = f"{action}|{accept_mode}"
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        tail_type = span.get("source_tail_type")
        if tail_type:
            tail_key = f"{action}|{tail_type}|{accept_mode}"
            tail_counts[tail_key] = tail_counts.get(tail_key, 0) + 1
    return action_counts, tail_counts


def _load_batch_provenance(path_str: str | None, prefix: str) -> dict:
    if not path_str:
        return {}
    payload = json.loads(Path(path_str).expanduser().read_text(encoding="utf-8"))
    batch = payload.get("batch", payload if isinstance(payload, dict) else {})
    file_payload = payload.get("file", {}) if isinstance(payload, dict) else {}
    if not isinstance(batch, dict):
        return {}
    return {
        f"{prefix}_batch_id": batch.get("id"),
        f"{prefix}_input_file_id": batch.get("input_file_id") or file_payload.get("id"),
        f"{prefix}_output_file_id": batch.get("output_file_id"),
        f"{prefix}_error_file_id": batch.get("error_file_id"),
        f"{prefix}_batch_status": batch.get("status"),
    }


def _provenance(config, args) -> dict:
    return {
        "schema_version": "translated_eval_record_v2",
        "phase1_model": args.model or config.phase1_model,
        "repair_model": args.repair_model or config.repair_model,
        "phase1_temperature": config.phase1_temperature,
        "repair_temperature": config.repair_temperature,
        "repair_policy": config.repair_policy,
        "prompt_profile": config.phase1_prompt_profile,
        "repair_enabled": False,
        "max_chars_per_line": config.max_chars_per_line,
        "max_lines_per_cue": config.max_lines_per_cue,
        "max_cps": config.max_cps,
        "wrap_policy": config.wrap_policy,
        "english_residual_policy": config.english_residual_policy,
        "git_sha": _git_sha(),
        "frozen_block_input": True,
        "openai_base_url": args.openai_base_url,
        "eval_lane": "batch_phase1_style_retry",
    }


def _load_config_from_args(args):
    config = load_runtime_config(glossary_log_path=getattr(args, "glossary_log_path", None))
    if getattr(args, "phase1_temperature", None) is not None:
        config.phase1_temperature = max(0.0, args.phase1_temperature)
    if getattr(args, "prompt_profile", None):
        config.phase1_prompt_profile = args.prompt_profile
    config.repair_enabled = False
    return config


def cmd_prepare_phase1(args) -> int:
    config = _load_config_from_args(args)
    translator = build_translator(
        config=config,
        model=args.model or config.phase1_model,
        repair_model=args.repair_model or config.repair_model,
        base_url=args.openai_base_url,
    )
    glossary_store = create_glossary_store(config=config, glossary_log_path=args.glossary_log_path)
    entries = _load_entries(Path(args.input).expanduser())
    cues_by_source = _load_cues_by_source(entries)
    requests_path = Path(args.requests_out).expanduser()
    manifest_path = Path(args.manifest_out).expanduser()
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(config, args)

    with requests_path.open("w", encoding="utf-8") as req_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for idx, entry in enumerate(entries, 1):
            all_cues = cues_by_source[entry["source_file"]]
            cue_index_set = set(entry.get("cue_indices") or entry.get("input_cue_indices") or entry["current_block"]["cue_indices"])
            block_cues = [cue for cue in all_cues if cue.index in cue_index_set]
            block = TranslationBlock(
                cues=block_cues,
                previous_source_sentences=list(entry.get("current_block", {}).get("previous_source_sentences", [])),
                next_source_sentences=list(entry.get("current_block", {}).get("next_source_sentences", [])),
                low_confidence=bool(entry.get("current_block", {}).get("block_lint", {}).get("low_confidence", False)),
                lint_reasons=list(entry.get("current_block", {}).get("block_lint", {}).get("lint_reasons", [])),
                lint_actions=list(entry.get("current_block", {}).get("block_lint", {}).get("lint_actions", [])),
            )
            glossary_terms = glossary_store.relevant_terms([cue.text for cue in block.cues])
            request = TranslationRequest(block=block, glossary_terms=glossary_terms)
            body = translator.build_phase1_request_body(request)
            custom_id = f"phase1-{idx:05d}"
            req_handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_handle.write(
                json.dumps(
                    {
                        "schema_version": "review_eval_batch_phase1_manifest_v1",
                        "custom_id": custom_id,
                        "id": entry["id"],
                        "lecture": entry["lecture"],
                        "source_file": entry["source_file"],
                        "source_review": entry.get("review") or entry.get("source_review") or {},
                        "input_cue_indices": [cue.index for cue in block.cues],
                        "current_block": {
                            "cue_indices": [cue.index for cue in block.cues],
                            "source_cues": _serialize_cues(block.cues),
                            "source_text": _translated_text(block.cues),
                            "previous_source_sentences": list(block.previous_source_sentences),
                            "next_source_sentences": list(block.next_source_sentences),
                            "block_lint": {
                                "low_confidence": block.low_confidence,
                                "lint_reasons": list(block.lint_reasons),
                                "lint_actions": list(block.lint_actions),
                            },
                        },
                        "glossary_terms": _serialize_glossary_terms(glossary_terms),
                        "provenance": provenance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote phase1 batch requests to {requests_path}")
    print(f"Wrote phase1 manifest to {manifest_path}")
    return 0


def cmd_prepare_style_retry(args) -> int:
    config = _load_config_from_args(args)
    translator = build_translator(
        config=config,
        model=args.model or config.phase1_model,
        repair_model=args.repair_model or config.repair_model,
        base_url=args.openai_base_url,
    )
    phase1_manifest = _load_entries(Path(args.phase1_manifest).expanduser())
    phase1_outputs = _parse_batch_output(Path(args.phase1_output).expanduser())
    requests_path = Path(args.requests_out).expanduser()
    manifest_path = Path(args.manifest_out).expanduser()
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests_path.open("w", encoding="utf-8") as req_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for row in phase1_manifest:
            block = _hydrate_block(row)
            glossary_terms = _deserialize_glossary_terms(row.get("glossary_terms", []))
            phase1_result, phase1_errors = _parse_batch_phase_result(translator, phase1_outputs.get(row["custom_id"]))
            raw_phase1_result = phase1_result
            validation_reasons: List[str] = []
            if phase1_result is not None:
                validation = validate_phase_structure(block, phase1_result.emitted_cues)
                if not validation.valid:
                    validation_reasons = validation.reasons
                    phase1_result = None
            style_retry_reasons: List[str] = []
            offending_cue_indices: List[int] = []
            offending_spans: List[dict] = []
            preferred_actions: List[str] = []
            protected_cue_indices: List[int] = []
            strict_custom_id = None
            micro_edit_trace = {
                "attempted": False,
                "accepted": False,
                "rejection_causes": [],
                "candidate_emitted_cues": [],
                "spans": [],
            }

            if phase1_result is not None:
                pre_gate = pre_wrap_gate(block, phase1_result.emitted_cues, glossary_terms, config)
                wrapped = _wrap_phase_result(block, phase1_result, config, None)
                post_gate = post_wrap_gate(wrapped, config)
                style_retry_reasons = _style_retry_candidate_reasons(phase1_result, pre_gate, post_gate)
                initial_offending_cue_indices, initial_offending_spans, initial_preferred_actions = _style_retry_feedback(
                    block,
                    pre_gate,
                    post_gate,
                    style_retry_reasons,
                )
                micro_edit_spans = _style_spans_for_actions(
                    initial_offending_spans,
                    ["drop_head_marker", "trim_explanatory_tail"],
                )
                if micro_edit_spans:
                    micro_edit_trace["attempted"] = True
                    micro_edit_trace["spans"] = micro_edit_spans
                    deterministic_candidate = _apply_deterministic_style_micro_edits(phase1_result, micro_edit_spans)
                    if deterministic_candidate is not None:
                        deterministic_pre = pre_wrap_gate(block, deterministic_candidate.emitted_cues, glossary_terms, config)
                        wrapped_deterministic = _wrap_phase_result(block, deterministic_candidate, config, None)
                        deterministic_post = post_wrap_gate(wrapped_deterministic, config)
                        deterministic_reasons = list(
                            dict.fromkeys(
                                str(span.get("issue"))
                                for span in micro_edit_spans
                                if span.get("issue")
                            )
                        )
                        deterministic_protected = _protected_cue_indices_for_spans(block, micro_edit_spans)
                        (
                            phase1_result,
                            pre_gate,
                            post_gate,
                            micro_accepted,
                            micro_rejection_causes,
                        ) = _choose_better_style_candidate(
                            phase1_result,
                            pre_gate,
                            post_gate,
                            deterministic_candidate,
                            deterministic_pre,
                            deterministic_post,
                            deterministic_reasons,
                            deterministic_protected,
                            offending_spans=micro_edit_spans,
                        )
                        micro_edit_trace["candidate_emitted_cues"] = _serialize_emitted_cues(deterministic_candidate)
                        micro_edit_trace["accepted"] = micro_accepted
                        micro_edit_trace["rejection_causes"] = micro_rejection_causes
                    else:
                        micro_edit_trace["rejection_causes"] = ["deterministic_noop"]

                style_retry_reasons = _style_retry_candidate_reasons(phase1_result, pre_gate, post_gate)
                if style_retry_reasons:
                    offending_cue_indices, offending_spans, preferred_actions = _style_retry_feedback(
                        block,
                        pre_gate,
                        post_gate,
                        style_retry_reasons,
                    )
                    protected_cue_indices = [
                        cue.index
                        for cue in block.cues
                        if cue.index not in set(offending_cue_indices)
                    ]
                    strict_request = TranslationRequest(
                        block=block,
                        glossary_terms=glossary_terms,
                        strict_style_retry=True,
                        style_retry_reasons=style_retry_reasons,
                        previous_emitted_cues=phase1_result.emitted_cues,
                        protected_cue_indices=protected_cue_indices,
                        offending_cue_indices=offending_cue_indices,
                        offending_spans=offending_spans,
                        preferred_actions=preferred_actions,
                    )
                    strict_custom_id = f"{row['custom_id']}:strict"
                    req_handle.write(
                        json.dumps(
                            {
                                "custom_id": strict_custom_id,
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": translator.build_phase1_request_body(strict_request),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            manifest_handle.write(
                json.dumps(
                    {
                        **row,
                        "schema_version": "review_eval_batch_retry_manifest_v1",
                        "raw_phase1_result": _phase_result_to_dict(raw_phase1_result) if raw_phase1_result is not None else None,
                        "phase1_result": _phase_result_to_dict(phase1_result) if phase1_result is not None else None,
                        "phase1_errors": phase1_errors,
                        "phase1_validation_reasons": validation_reasons,
                        "micro_edit_trace": micro_edit_trace,
                        "strict_custom_id": strict_custom_id,
                        "style_retry_reasons": style_retry_reasons,
                        "offending_cue_indices": offending_cue_indices,
                        "protected_cue_indices": protected_cue_indices,
                        "offending_spans": offending_spans,
                        "preferred_actions": preferred_actions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote strict-retry batch requests to {requests_path}")
    print(f"Wrote retry manifest to {manifest_path}")
    return 0


def cmd_finalize(args) -> int:
    config = _load_config_from_args(args)
    translator = build_translator(
        config=config,
        model=args.model or config.phase1_model,
        repair_model=args.repair_model or config.repair_model,
        base_url=args.openai_base_url,
    )
    retry_manifest = _load_entries(Path(args.retry_manifest).expanduser())
    strict_outputs = _parse_batch_output(Path(args.strict_output).expanduser()) if args.strict_output else {}
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_provenance = {
        **_load_batch_provenance(args.phase1_batch_json, "phase1"),
        **_load_batch_provenance(args.strict_batch_json, "strict"),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for row in retry_manifest:
            block = _hydrate_block(row)
            glossary_terms = _deserialize_glossary_terms(row.get("glossary_terms", []))
            phase1_result = _phase_result_from_dict(row["phase1_result"]) if row.get("phase1_result") else None
            phase1_errors = list(row.get("phase1_errors", []))
            validation_reasons = list(row.get("phase1_validation_reasons", []))

            final_result: PhaseTranslationResult
            pre_gate = post_gate_result = None
            strict_result: PhaseTranslationResult | None = None
            style_retry_trace = {}
            style_retry_rejection_causes: Dict[str, int] = {}
            style_retry_invoked = bool(row.get("strict_custom_id"))
            style_retry_accepted = False
            style_retry_rejected = False

            if phase1_result is None:
                final_result = _fallback_source_result(block, "phase1_structure_failure")
                pre_gate = pre_wrap_gate(block, final_result.emitted_cues, glossary_terms, config)
                wrapped_final = _wrap_phase_result(block, final_result, config, None)
                post_gate_result = post_wrap_gate(wrapped_final, config)
            else:
                pre_gate = pre_wrap_gate(block, phase1_result.emitted_cues, glossary_terms, config)
                wrapped_phase1 = _wrap_phase_result(block, phase1_result, config, None)
                post_gate_result = post_wrap_gate(wrapped_phase1, config)
                final_result = phase1_result

            if style_retry_invoked or row.get("micro_edit_trace", {}).get("attempted"):
                style_retry_trace = {
                    "reasons": list(row.get("style_retry_reasons", [])),
                    "offending_cue_indices": list(row.get("offending_cue_indices", [])),
                    "protected_cue_indices": list(row.get("protected_cue_indices", [])),
                    "offending_spans": list(row.get("offending_spans", [])),
                    "preferred_actions": list(row.get("preferred_actions", [])),
                    "effective_strict_prompt_profile": (
                        "fragment_preserving_v3"
                        if (
                            row["provenance"].get("prompt_profile") == "fragment_preserving_v2"
                            and len(row.get("offending_cue_indices", [])) == 1
                            and row.get("protected_cue_indices")
                            and row.get("offending_spans")
                            and all(
                                span.get("preferred_action") == "restore_missing_tail"
                                and span.get("source_tail_type") == "continuation_tail"
                                for span in row.get("offending_spans", [])
                            )
                        )
                        else row["provenance"].get("prompt_profile")
                    ),
                    "base_phase1_emitted_cues": _serialize_emitted_cues(
                        _phase_result_from_dict(row["raw_phase1_result"])
                    ) if row.get("raw_phase1_result") else (_serialize_emitted_cues(phase1_result) if phase1_result else []),
                    "micro_edit_trace": row.get("micro_edit_trace", {}),
                }
                if phase1_result is not None and style_retry_invoked:
                    strict_result, strict_errors = _parse_batch_phase_result(translator, strict_outputs.get(row["strict_custom_id"]))
                    if strict_result is None:
                        style_retry_rejected = True
                        style_retry_trace["strict_candidate_raw_emitted_cues"] = []
                        style_retry_trace["strict_candidate_emitted_cues"] = []
                        style_retry_trace["strict_candidate_risk_flags"] = []
                        style_retry_trace["strict_candidate_post_normalizations"] = []
                        style_retry_trace["accept_mode"] = None
                        style_retry_trace["accepted"] = False
                        style_retry_trace["rejection_causes"] = list(strict_errors)
                        for cause in strict_errors:
                            style_retry_rejection_causes[cause] = style_retry_rejection_causes.get(cause, 0) + 1
                    else:
                        style_retry_trace["strict_candidate_raw_emitted_cues"] = _serialize_emitted_cues(strict_result)
                        strict_result, strict_post_normalizations = _apply_purpose_tail_post_normalization(
                            strict_result,
                            row.get("offending_spans", []),
                        )
                        style_retry_trace["strict_candidate_emitted_cues"] = _serialize_emitted_cues(strict_result)
                        style_retry_trace["strict_candidate_risk_flags"] = list(strict_result.risk_flags)
                        style_retry_trace["strict_candidate_post_normalizations"] = list(strict_post_normalizations)
                        strict_validation = validate_phase_structure(block, strict_result.emitted_cues)
                        if not strict_validation.valid:
                            rejection_causes = [f"strict_retry_{reason}" for reason in strict_validation.reasons]
                            style_retry_rejected = True
                            style_retry_trace["accept_mode"] = None
                            style_retry_trace["accepted"] = False
                            style_retry_trace["rejection_causes"] = rejection_causes
                            for cause in rejection_causes:
                                style_retry_rejection_causes[cause] = style_retry_rejection_causes.get(cause, 0) + 1
                        else:
                            strict_pre = pre_wrap_gate(block, strict_result.emitted_cues, glossary_terms, config)
                            wrapped_strict = _wrap_phase_result(block, strict_result, config, None)
                            strict_post = post_wrap_gate(wrapped_strict, config)
                            if _candidate_is_overedited(
                                block,
                                phase1_result,
                                pre_gate,
                                post_gate_result,
                                strict_result,
                                strict_pre,
                                strict_post,
                                row.get("style_retry_reasons", []),
                            ):
                                rejection_causes = ["overedited_candidate"]
                                style_retry_rejected = True
                                style_retry_trace["accept_mode"] = None
                                style_retry_trace["accepted"] = False
                                style_retry_trace["rejection_causes"] = rejection_causes
                                style_retry_trace["strict_candidate_emitted_cues"] = _serialize_emitted_cues(strict_result)
                                for cause in rejection_causes:
                                    style_retry_rejection_causes[cause] = style_retry_rejection_causes.get(cause, 0) + 1
                            else:
                                final_result, pre_gate, post_gate_result, accepted, rejection_causes = _choose_better_style_candidate(
                                    phase1_result,
                                    pre_gate,
                                    post_gate_result,
                                    strict_result,
                                    strict_pre,
                                    strict_post,
                                    row.get("style_retry_reasons", []),
                                    row.get("protected_cue_indices", []),
                                    offending_spans=row.get("offending_spans", []),
                                )
                            style_retry_accepted = accepted
                            style_retry_rejected = not accepted
                            style_retry_trace["accepted"] = accepted
                            style_retry_trace["accept_mode"] = _strict_accept_mode(strict_post_normalizations) if accepted else None
                            style_retry_trace["rejection_causes"] = rejection_causes
                            for cause in rejection_causes:
                                style_retry_rejection_causes[cause] = style_retry_rejection_causes.get(cause, 0) + 1

                style_retry_trace["final_emitted_cues"] = _serialize_emitted_cues(final_result)
                style_retry_trace["final_risk_flags"] = list(final_result.risk_flags)
                style_retry_trace["offending_cue_diffs"] = _offending_cue_diffs(
                    phase1_result if phase1_result is not None else final_result,
                    strict_result,
                    final_result,
                    row.get("offending_cue_indices", []),
                )

            wrapped_final = _wrap_phase_result(block, final_result, config, None)
            final_post = post_wrap_gate(wrapped_final, config)
            remaining_warning_spans = _style_warning_action_details(pre_gate, final_post)
            style_action_attempts: Dict[str, int] = {}
            style_action_accepts: Dict[str, int] = {}
            style_action_rejections: Dict[str, int] = {}
            style_action_remaining_warnings, _ = _style_action_counter(remaining_warning_spans)
            style_action_tail_attempts: Dict[str, int] = {}
            style_action_tail_accepts: Dict[str, int] = {}
            style_action_tail_rejections: Dict[str, int] = {}
            style_action_accept_modes: Dict[str, int] = {}
            style_action_tail_accept_modes: Dict[str, int] = {}
            style_action_attempts_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_accepts_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_rejections_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_remaining_warnings_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_tail_attempts_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_tail_accepts_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_tail_rejections_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_accept_modes_by_channel: Dict[str, Dict[str, int]] = {}
            style_action_tail_accept_modes_by_channel: Dict[str, Dict[str, int]] = {}

            def merge_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
                for key, value in source.items():
                    target[key] = target.get(key, 0) + value

            def merge_channel_counts(target: Dict[str, Dict[str, int]], channel: str, source: Dict[str, int]) -> None:
                per_channel = target.setdefault(channel, {})
                for key, value in source.items():
                    per_channel[key] = per_channel.get(key, 0) + value

            def merge_accept_mode_counts(spans: List[dict], accept_mode: str, channel: str) -> None:
                if not accept_mode:
                    return
                action_counts, tail_counts = _style_action_counter_with_accept_mode(spans, accept_mode)
                merge_counts(style_action_accept_modes, action_counts)
                merge_counts(style_action_tail_accept_modes, tail_counts)
                merge_channel_counts(style_action_accept_modes_by_channel, channel, action_counts)
                merge_channel_counts(style_action_tail_accept_modes_by_channel, channel, tail_counts)

            micro_trace = row.get("micro_edit_trace", {})
            micro_attempts, micro_tail_attempts = _style_action_counter(micro_trace.get("spans", []))
            merge_counts(style_action_attempts, micro_attempts)
            merge_counts(style_action_tail_attempts, micro_tail_attempts)
            merge_channel_counts(style_action_attempts_by_channel, "micro_edit", micro_attempts)
            merge_channel_counts(style_action_tail_attempts_by_channel, "micro_edit", micro_tail_attempts)
            if micro_trace.get("attempted"):
                if micro_trace.get("accepted"):
                    merge_counts(style_action_accepts, micro_attempts)
                    merge_counts(style_action_tail_accepts, micro_tail_attempts)
                    merge_channel_counts(style_action_accepts_by_channel, "micro_edit", micro_attempts)
                    merge_channel_counts(style_action_tail_accepts_by_channel, "micro_edit", micro_tail_attempts)
                else:
                    merge_counts(style_action_rejections, micro_attempts)
                    merge_counts(style_action_tail_rejections, micro_tail_attempts)
                    merge_channel_counts(style_action_rejections_by_channel, "micro_edit", micro_attempts)
                    merge_channel_counts(style_action_tail_rejections_by_channel, "micro_edit", micro_tail_attempts)

            strict_attempts, strict_tail_attempts = _style_action_counter(row.get("offending_spans", []))
            merge_counts(style_action_attempts, strict_attempts)
            merge_counts(style_action_tail_attempts, strict_tail_attempts)
            merge_channel_counts(style_action_attempts_by_channel, "strict_retry", strict_attempts)
            merge_channel_counts(style_action_tail_attempts_by_channel, "strict_retry", strict_tail_attempts)
            if style_retry_invoked:
                if style_retry_accepted:
                    merge_counts(style_action_accepts, strict_attempts)
                    merge_counts(style_action_tail_accepts, strict_tail_attempts)
                    merge_channel_counts(style_action_accepts_by_channel, "strict_retry", strict_attempts)
                    merge_channel_counts(style_action_tail_accepts_by_channel, "strict_retry", strict_tail_attempts)
                    merge_accept_mode_counts(row.get("offending_spans", []), style_retry_trace.get("accept_mode"), "strict_retry")
                else:
                    merge_counts(style_action_rejections, strict_attempts)
                    merge_counts(style_action_tail_rejections, strict_tail_attempts)
                    merge_channel_counts(style_action_rejections_by_channel, "strict_retry", strict_attempts)
                    merge_channel_counts(style_action_tail_rejections_by_channel, "strict_retry", strict_tail_attempts)

            micro_remaining, _ = _style_action_counter(
                [span for span in remaining_warning_spans if span.get("preferred_action") in {"drop_head_marker", "trim_explanatory_tail"}]
            )
            strict_remaining, _ = _style_action_counter(
                [span for span in remaining_warning_spans if span.get("preferred_action") in {"delete_repeat_local", "restore_missing_tail"}]
            )
            merge_channel_counts(style_action_remaining_warnings_by_channel, "micro_edit", micro_remaining)
            merge_channel_counts(style_action_remaining_warnings_by_channel, "strict_retry", strict_remaining)

            record = {
                "schema_version": "translated_eval_record_v2",
                "id": row["id"],
                "lecture": row["lecture"],
                "source_file": row["source_file"],
                "source_review": row.get("source_review", {}),
                "input_cue_indices": row["input_cue_indices"],
                "current_block": row["current_block"],
                "translation_output": {
                    "translated_cues": _serialize_cues(wrapped_final),
                    "translated_text": _translated_text(wrapped_final),
                },
                "pipeline_signals": {
                    "phase1_retried": False,
                    "style_retry_invoked": style_retry_invoked,
                    "style_retry_accepted": style_retry_accepted,
                    "style_retry_rejected": style_retry_rejected,
                    "repair_invoked": False,
                    "repair_accepted": False,
                    "repair_rejected": False,
                    "smaller_block_fallback": False,
                    "single_cue_source_fallback": phase1_result is None,
                    "post_wrap_failure": bool(final_post.repair_reasons or final_post.warning_reasons),
                    "failure_reasons": {
                        reason: 1
                        for reason in (
                            phase1_errors
                            + validation_reasons
                            + list(pre_gate.repair_reasons)
                            + list(final_post.repair_reasons)
                        )
                    },
                    "pre_wrap_failures": {reason: 1 for reason in (list(pre_gate.repair_reasons) + list(pre_gate.warning_reasons))},
                    "post_wrap_failures": {reason: 1 for reason in (list(final_post.repair_reasons) + list(final_post.warning_reasons))},
                    "phase1_risk_flags": {flag: 1 for flag in final_result.risk_flags},
                    "strict_retry_candidate_risk_flags": {
                        flag: 1
                        for flag in style_retry_trace.get("strict_candidate_risk_flags", [])
                    } if style_retry_trace.get("accepted") else {},
                    "style_retry_rejection_causes": style_retry_rejection_causes,
                    "style_action_attempts": style_action_attempts,
                    "style_action_accepts": style_action_accepts,
                    "style_action_rejections": style_action_rejections,
                    "style_action_remaining_warnings": style_action_remaining_warnings,
                    "style_action_tail_attempts": style_action_tail_attempts,
                    "style_action_tail_accepts": style_action_tail_accepts,
                    "style_action_tail_rejections": style_action_tail_rejections,
                    "style_action_accept_modes": style_action_accept_modes,
                    "style_action_tail_accept_modes": style_action_tail_accept_modes,
                    "style_action_attempts_by_channel": style_action_attempts_by_channel,
                    "style_action_accepts_by_channel": style_action_accepts_by_channel,
                    "style_action_rejections_by_channel": style_action_rejections_by_channel,
                    "style_action_remaining_warnings_by_channel": style_action_remaining_warnings_by_channel,
                    "style_action_tail_attempts_by_channel": style_action_tail_attempts_by_channel,
                    "style_action_tail_accepts_by_channel": style_action_tail_accepts_by_channel,
                    "style_action_tail_rejections_by_channel": style_action_tail_rejections_by_channel,
                    "style_action_accept_modes_by_channel": style_action_accept_modes_by_channel,
                    "style_action_tail_accept_modes_by_channel": style_action_tail_accept_modes_by_channel,
                    "captured_phase1_risk_flags": list(row.get("phase1_result", {}).get("risk_flags", [])) if row.get("phase1_result") else [],
                    "average_cps": round(_average_cps(wrapped_final), 3),
                    "effective_strict_prompt_profile": (
                        style_retry_trace.get("effective_strict_prompt_profile")
                        or style_retry_trace.get("prompt_profile")
                    ),
                    "effective_repair_profile": row["provenance"].get("repair_policy"),
                    "style_retry_trace": style_retry_trace,
                },
                "provenance": {**row["provenance"], **batch_provenance},
                "review": {
                    "status": "pending",
                    "failure_tags": [],
                    "notes": "",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote finalized batch eval records to {output_path}")
    return 0


def _make_batch_client(args) -> OpenAIBatchClient:
    config = load_runtime_config()
    return OpenAIBatchClient(
        api_key=config.openai_api_key,
        base_url=args.openai_base_url,
        timeout=config.request_timeout,
        max_attempts=config.request_max_attempts,
        backoff_min_seconds=config.request_backoff_min_seconds,
        backoff_max_seconds=config.request_backoff_max_seconds,
    )


def cmd_submit(args) -> int:
    client = _make_batch_client(args)
    uploaded = client.upload_batch_file(Path(args.input_jsonl).expanduser())
    metadata = json.loads(args.metadata_json) if args.metadata_json else {}
    batch = client.create_batch(
        input_file_id=uploaded["id"],
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata,
    )
    print(json.dumps({"file": uploaded, "batch": batch}, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).expanduser().write_text(
            json.dumps({"file": uploaded, "batch": batch}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


def cmd_status(args) -> int:
    client = _make_batch_client(args)
    batch = (
        client.wait_for_batch(
            args.batch_id,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        if args.wait
        else client.retrieve_batch(args.batch_id)
    )
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    return 0


def cmd_download(args) -> int:
    client = _make_batch_client(args)
    output_path = client.download_file_to_path(args.file_id, Path(args.output).expanduser())
    print(f"Downloaded {args.file_id} to {output_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare-phase1":
        return cmd_prepare_phase1(args)
    if args.command == "prepare-style-retry":
        return cmd_prepare_style_retry(args)
    if args.command == "finalize":
        return cmd_finalize(args)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "download":
        return cmd_download(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
