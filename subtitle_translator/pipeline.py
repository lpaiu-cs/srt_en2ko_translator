from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .blocks import build_translation_blocks
from .config import RuntimeConfig
from .glossary import GlossaryStore
from .metrics import TranslationMetrics
from .models import Cue, EmittedCue, PhaseTranslationResult, RepairRequest, TranslationBlock, TranslationRequest
from .quality import post_wrap_gate, pre_wrap_gate, validate_phase_structure
from .text import normalize_text, warn, wrap_lines, wrap_lines_candidates
from .translators import BaseTranslator


LOCAL_ONLY_FAILURES = {
    "line_overflow",
    "bad_line_break",
    "cps_overflow",
    "cps_overflow_severe",
}

BOUNDARY_LINT_REASONS = {
    "dependent_start",
    "dependent_end",
    "numeric_orphan",
    "comparison_midstart",
    "qa_fragment",
}

BOUNDARY_SENSITIVE_REPAIR_REASONS = {
    "numeric_anchor_loss",
    "delimiter_anchor_loss",
    "identifier_anchor_loss",
}

STYLE_WARNING_REASONS = {
    "fragment_overclosure",
    "duplicate_restatement",
}

STYLE_RISK_TO_REASON = {
    "fragment_overclosure_risk": "fragment_overclosure",
    "duplicate_restatement_risk": "duplicate_restatement",
}

STYLE_REASON_TO_RISK = {
    reason: risk_flag
    for risk_flag, reason in STYLE_RISK_TO_REASON.items()
}


def create_glossary_store(config: RuntimeConfig, glossary_log_path: Optional[str] = None) -> GlossaryStore:
    path = Path(glossary_log_path).expanduser() if glossary_log_path is not None else config.glossary_log_path
    return GlossaryStore(path=path, max_terms=config.glossary_max_terms)


def _fallback_source_result(block: TranslationBlock, reason: str) -> PhaseTranslationResult:
    return PhaseTranslationResult(
        emitted_cues=[EmittedCue(cue_index=cue.index, text=normalize_text(cue.text)) for cue in block.cues],
        risk_flags=[reason] if reason else [],
    )


def _failure_signature(reasons: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(reason for reason in reasons if reason)))


def _gate_score(pre_gate, post_gate) -> tuple[int, int]:
    return (
        len(pre_gate.repair_reasons) + len(post_gate.repair_reasons),
        len(pre_gate.warning_reasons) + len(post_gate.warning_reasons),
    )


def _gate_repair_reason_set(pre_gate, post_gate) -> set[str]:
    return set(pre_gate.repair_reasons) | set(post_gate.repair_reasons)


def _gate_warning_occurrences(pre_gate, post_gate, reasons: Sequence[str]) -> int:
    total = 0
    for reason in reasons:
        detail_count = len(pre_gate.warning_details.get(reason, [])) + len(post_gate.warning_details.get(reason, []))
        if detail_count:
            total += detail_count
        elif reason in pre_gate.warning_reasons or reason in post_gate.warning_reasons:
            total += 1
    return total


def _post_wrap_score(post_gate) -> tuple[int, int]:
    return (len(post_gate.repair_reasons), len(post_gate.warning_reasons))


def _style_risk_occurrences(result: PhaseTranslationResult, reasons: Sequence[str]) -> int:
    allowed_risks = {STYLE_REASON_TO_RISK[reason] for reason in reasons if reason in STYLE_REASON_TO_RISK}
    return sum(1 for risk_flag in result.risk_flags if risk_flag in allowed_risks)


def _single_cue_wrap_score(cue: Cue, config: RuntimeConfig) -> tuple[tuple[int, int, int, int, int], object]:
    gate = post_wrap_gate([cue], config)
    lines = cue.text.splitlines() if cue.text else []
    max_line_len = max((len(line) for line in lines), default=0)
    imbalance = abs(len(lines[0].strip()) - len(lines[1].strip())) if len(lines) == 2 else 0
    score = (
        len(gate.repair_reasons),
        len(gate.warning_reasons),
        max_line_len,
        imbalance,
        len(lines),
    )
    return score, gate


def _wrap_cue_with_local_rewrap(cue: Cue, text: str, config: RuntimeConfig, metrics: Optional[TranslationMetrics]) -> Cue:
    normalized = normalize_text(text)
    primary = Cue(
        index=cue.index,
        start=cue.start,
        end=cue.end,
        text=wrap_lines(normalized, width=config.max_chars_per_line),
    )
    primary_score, primary_gate = _single_cue_wrap_score(primary, config)
    if not primary_gate.repair_reasons and not primary_gate.warning_reasons:
        return primary

    if metrics:
        metrics.local_rewrap_attempts += 1

    best = primary
    best_score = primary_score
    for candidate_text in wrap_lines_candidates(normalized, width=config.max_chars_per_line, max_lines=config.max_lines_per_cue):
        candidate = Cue(
            index=cue.index,
            start=cue.start,
            end=cue.end,
            text=candidate_text,
        )
        candidate_score, _ = _single_cue_wrap_score(candidate, config)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    if metrics and best.text != primary.text:
        metrics.local_rewrap_successes += 1
    return best


def _wrap_phase_result(
    block: TranslationBlock,
    result: PhaseTranslationResult,
    config: RuntimeConfig,
    metrics: Optional[TranslationMetrics],
) -> List[Cue]:
    text_by_index = {cue.cue_index: normalize_text(cue.text) for cue in result.emitted_cues}
    wrapped: List[Cue] = []
    for cue in block.cues:
        wrapped.append(_wrap_cue_with_local_rewrap(cue, text_by_index.get(cue.index, ""), config, metrics))
    return wrapped


def _repair_candidate_reasons(
    block: TranslationBlock,
    pre_gate,
    post_gate,
) -> List[str]:
    reasons = list(dict.fromkeys(pre_gate.repair_reasons + post_gate.repair_reasons))
    blocked = set(BOUNDARY_LINT_REASONS)
    if "carry_context_only" in block.lint_actions:
        blocked.update(BOUNDARY_SENSITIVE_REPAIR_REASONS)
    return [reason for reason in reasons if reason not in blocked]


def _style_retry_candidate_reasons(
    phase1_result: PhaseTranslationResult,
    pre_gate,
    post_gate,
) -> List[str]:
    reasons: List[str] = []
    warning_set = set(pre_gate.warning_reasons + post_gate.warning_reasons)
    for reason in STYLE_WARNING_REASONS:
        if reason in warning_set:
            reasons.append(reason)
    for risk_flag in phase1_result.risk_flags:
        mapped = STYLE_RISK_TO_REASON.get(risk_flag)
        if mapped and mapped not in reasons:
            reasons.append(mapped)
    return reasons


def _style_retry_feedback(
    block: TranslationBlock,
    pre_gate,
    post_gate,
    style_retry_reasons: Sequence[str],
) -> tuple[List[int], List[dict], List[str]]:
    offending_cues: List[int] = []
    offending_spans: List[dict] = []
    preferred_actions: List[str] = []
    default_actions = {
        "fragment_overclosure": "keep_fragment_open",
        "duplicate_restatement": "delete_repeat_local",
    }
    seen_spans: set[str] = set()

    for reason in style_retry_reasons:
        details = pre_gate.warning_details.get(reason, []) + post_gate.warning_details.get(reason, [])
        if not details and reason in default_actions:
            if reason == "duplicate_restatement" and len(block.cues) >= 2:
                tail_cue = block.cues[-1]
                normalized_tail = tail_cue.text.strip().lower()
                fallback_action = (
                    "restore_missing_tail"
                    if normalized_tail.startswith(
                        ("to ", "that ", "which ", "who ", "where ", "when ", "because ", "if ", "while ", "as ", "than ", "for ", "by ", "of ")
                    )
                    else default_actions[reason]
                )
                offending_cues.append(tail_cue.index)
                offending_spans.append(
                    {
                        "cue_index": tail_cue.index,
                        "cue_indices": [tail_cue.index],
                        "span_text": "",
                        "source_cue_text": tail_cue.text,
                        "issue": "duplicate_restatement",
                        "preferred_action": fallback_action,
                    }
                )
                preferred_actions.append(fallback_action)
                continue
            preferred_actions.append(default_actions[reason])
            continue
        for detail in details:
            cue_indices = detail.get("cue_indices")
            if isinstance(cue_indices, list):
                for cue_index in cue_indices:
                    if isinstance(cue_index, int) and cue_index not in offending_cues and cue_index >= 0:
                        offending_cues.append(cue_index)
            cue_index = detail.get("cue_index")
            if isinstance(cue_index, int) and cue_index not in offending_cues and cue_index >= 0:
                offending_cues.append(cue_index)
            action = detail.get("preferred_action")
            if isinstance(action, str) and action:
                preferred_actions.append(action)
            signature = json.dumps(detail, ensure_ascii=False, sort_keys=True)
            if signature in seen_spans:
                continue
            seen_spans.add(signature)
            offending_spans.append(detail)

    return sorted(offending_cues), offending_spans, list(dict.fromkeys(preferred_actions))


def _glossary_terms_for_block(block: TranslationBlock, glossary_store: Optional[GlossaryStore]) -> list:
    if not glossary_store:
        return []
    return glossary_store.relevant_terms([cue.text for cue in block.cues])


def _split_block(block: TranslationBlock) -> List[TranslationBlock]:
    if len(block.cues) <= 1:
        return [block]
    midpoint = max(1, len(block.cues) // 2)
    left_cues = block.cues[:midpoint]
    right_cues = block.cues[midpoint:]
    left_bridge = normalize_text(" ".join(cue.text for cue in right_cues))
    right_bridge = normalize_text(" ".join(cue.text for cue in left_cues))
    left_next = ([left_bridge] if left_bridge else []) + block.next_source_sentences[:1]
    right_prev = (block.previous_source_sentences + ([right_bridge] if right_bridge else []))[-2:]
    return [
        TranslationBlock(
            cues=left_cues,
            previous_source_sentences=block.previous_source_sentences,
            next_source_sentences=left_next[:1],
        ),
        TranslationBlock(
            cues=right_cues,
            previous_source_sentences=right_prev,
            next_source_sentences=block.next_source_sentences,
        ),
    ]


def _run_phase1_once(block: TranslationBlock, translator: BaseTranslator, glossary_terms: list) -> PhaseTranslationResult:
    request = TranslationRequest(block=block, glossary_terms=glossary_terms)
    return translator.translate_block(request)


def _run_phase1_with_retry(
    block: TranslationBlock,
    translator: BaseTranslator,
    glossary_terms: list,
    config: RuntimeConfig,
) -> tuple[Optional[PhaseTranslationResult], int, List[str]]:
    attempts = 0
    last_reasons: List[str] = []
    seen_signatures: set[Tuple[str, ...]] = set()
    for attempt in range(config.phase1_max_retries):
        attempts = attempt + 1
        try:
            result = _run_phase1_once(block, translator, glossary_terms)
        except Exception as exc:
            last_reasons = ["phase1_exception"]
            signature = _failure_signature(last_reasons)
            warn(f"Phase1 attempt {attempts} failed for cues {[cue.index for cue in block.cues]}: {exc}")
            if signature in seen_signatures:
                break
            seen_signatures.add(signature)
            continue

        validation = validate_phase_structure(block, result.emitted_cues)
        if validation.valid:
            return result, attempts, last_reasons

        last_reasons = validation.reasons
        signature = _failure_signature(last_reasons)
        warn(
            f"Phase1 attempt {attempts} returned invalid structure for cues {[cue.index for cue in block.cues]}: "
            f"{', '.join(validation.reasons)}"
        )
        if signature in seen_signatures:
            break
        seen_signatures.add(signature)
    return None, attempts, last_reasons


def _run_phase2_repair(
    block: TranslationBlock,
    phase1_result: PhaseTranslationResult,
    failure_reasons: List[str],
    translator: BaseTranslator,
    glossary_terms: list,
    config: RuntimeConfig,
    metrics: Optional[TranslationMetrics],
) -> Optional[PhaseTranslationResult]:
    seen_signatures: set[Tuple[str, ...]] = set()
    for attempt in range(config.phase2_max_repairs):
        repair_request = RepairRequest(
            block=block,
            phase1_result=phase1_result,
            glossary_terms=glossary_terms,
            failure_reasons=failure_reasons,
        )
        if metrics:
            metrics.repair_invocations += 1
        try:
            repaired = translator.repair_block(repair_request)
        except Exception as exc:
            signature = ("phase2_exception",)
            warn(f"Phase2 repair attempt {attempt + 1} failed for cues {[cue.index for cue in block.cues]}: {exc}")
            if signature in seen_signatures:
                break
            seen_signatures.add(signature)
            continue

        validation = validate_phase_structure(block, repaired.emitted_cues)
        if validation.valid:
            return repaired

        signature = _failure_signature(validation.reasons)
        warn(
            f"Phase2 repair attempt {attempt + 1} returned invalid structure for cues {[cue.index for cue in block.cues]}: "
            f"{', '.join(validation.reasons)}"
        )
        if signature in seen_signatures:
            break
        seen_signatures.add(signature)
    return None


def _run_phase1_style_retry(
    block: TranslationBlock,
    phase1_result: PhaseTranslationResult,
    style_retry_reasons: List[str],
    protected_cue_indices: List[int],
    offending_cue_indices: List[int],
    offending_spans: List[dict],
    preferred_actions: List[str],
    translator: BaseTranslator,
    glossary_terms: list,
    metrics: Optional[TranslationMetrics],
) -> Optional[PhaseTranslationResult]:
    if metrics:
        metrics.style_retry_invocations += 1
    request = TranslationRequest(
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
    try:
        retried = translator.translate_block(request)
    except Exception as exc:
        warn(
            f"Strict Phase1 retry failed for cues {[cue.index for cue in block.cues]} "
            f"after style warnings {style_retry_reasons}: {exc}"
        )
        return None

    validation = validate_phase_structure(block, retried.emitted_cues)
    if validation.valid:
        return retried

    warn(
        f"Strict Phase1 retry returned invalid structure for cues {[cue.index for cue in block.cues]}: "
        f"{', '.join(validation.reasons)}"
    )
    return None


def _changed_chars(left: str, right: str) -> int:
    matcher = SequenceMatcher(a=left, b=right)
    total = 0
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "equal":
            total += max(left_end - left_start, right_end - right_start)
    return total


def _repair_change_stats(current: PhaseTranslationResult, candidate: PhaseTranslationResult) -> tuple[int, int]:
    current_by_index = {cue.cue_index: normalize_text(cue.text) for cue in current.emitted_cues}
    candidate_by_index = {cue.cue_index: normalize_text(cue.text) for cue in candidate.emitted_cues}
    changed_cues = 0
    changed_chars = 0
    for cue_index, current_text in current_by_index.items():
        candidate_text = candidate_by_index.get(cue_index, "")
        if current_text != candidate_text:
            changed_cues += 1
            changed_chars += _changed_chars(current_text, candidate_text)
    return changed_cues, changed_chars


def _changed_cue_indices(current: PhaseTranslationResult, candidate: PhaseTranslationResult) -> List[int]:
    current_by_index = {cue.cue_index: normalize_text(cue.text) for cue in current.emitted_cues}
    candidate_by_index = {cue.cue_index: normalize_text(cue.text) for cue in candidate.emitted_cues}
    changed: List[int] = []
    for cue_index, current_text in current_by_index.items():
        if current_text != candidate_by_index.get(cue_index, ""):
            changed.append(cue_index)
    return changed


def _candidate_is_overedited(
    block: TranslationBlock,
    current_result: PhaseTranslationResult,
    current_pre,
    current_post,
    candidate_result: PhaseTranslationResult,
    candidate_pre,
    candidate_post,
    failure_reasons: List[str],
) -> bool:
    changed_cues, changed_chars = _repair_change_stats(current_result, candidate_result)
    if changed_cues == 0:
        return False

    current_score = _gate_score(current_pre, current_post)
    candidate_score = _gate_score(candidate_pre, candidate_post)
    total_chars = sum(len(normalize_text(cue.text)) for cue in current_result.emitted_cues) or 1
    change_threshold = max(24, total_chars // 2)

    if len(failure_reasons) <= 2 and len(block.cues) >= 3:
        if changed_cues >= max(2, (len(block.cues) * 3 + 3) // 4):
            return True
    if set(failure_reasons).issubset(LOCAL_ONLY_FAILURES) and changed_cues == len(block.cues) and changed_chars > change_threshold:
        return True
    if candidate_score >= current_score and changed_chars > max(12, total_chars // 3):
        return True
    return False


def _choose_better_candidate(
    current_result: PhaseTranslationResult,
    current_pre,
    current_post,
    candidate_result: PhaseTranslationResult,
    candidate_pre,
    candidate_post,
):
    current_score = _gate_score(current_pre, current_post)
    candidate_score = _gate_score(candidate_pre, candidate_post)
    if candidate_score < current_score:
        return candidate_result, candidate_pre, candidate_post, True
    return current_result, current_pre, current_post, False


def _choose_better_style_candidate(
    current_result: PhaseTranslationResult,
    current_pre,
    current_post,
    candidate_result: PhaseTranslationResult,
    candidate_pre,
    candidate_post,
    style_retry_reasons: Sequence[str],
    protected_cue_indices: Sequence[int],
):
    if protected_cue_indices:
        changed_protected = set(_changed_cue_indices(current_result, candidate_result)) & set(protected_cue_indices)
        if changed_protected:
            return current_result, current_pre, current_post, False

    current_repair_reasons = _gate_repair_reason_set(current_pre, current_post)
    candidate_repair_reasons = _gate_repair_reason_set(candidate_pre, candidate_post)
    if candidate_repair_reasons - current_repair_reasons:
        return current_result, current_pre, current_post, False
    if _post_wrap_score(candidate_post) > _post_wrap_score(current_post):
        return current_result, current_pre, current_post, False

    current_targeted = _gate_warning_occurrences(current_pre, current_post, style_retry_reasons)
    candidate_targeted = _gate_warning_occurrences(candidate_pre, candidate_post, style_retry_reasons)
    current_risk_targeted = _style_risk_occurrences(current_result, style_retry_reasons)
    candidate_risk_targeted = _style_risk_occurrences(candidate_result, style_retry_reasons)
    if candidate_targeted < current_targeted:
        return candidate_result, candidate_pre, candidate_post, True
    if candidate_targeted == current_targeted and candidate_risk_targeted < current_risk_targeted:
        return candidate_result, candidate_pre, candidate_post, True

    if candidate_targeted == current_targeted and _gate_score(candidate_pre, candidate_post) < _gate_score(current_pre, current_post):
        return candidate_result, candidate_pre, candidate_post, True

    return current_result, current_pre, current_post, False


def _serialize_emitted_cues(result: PhaseTranslationResult) -> List[dict]:
    return [{"cue_index": cue.cue_index, "text": normalize_text(cue.text)} for cue in result.emitted_cues]


def _offending_cue_diffs(
    base_result: PhaseTranslationResult,
    strict_result: Optional[PhaseTranslationResult],
    final_result: PhaseTranslationResult,
    offending_cue_indices: Sequence[int],
) -> List[dict]:
    base_map = {cue.cue_index: normalize_text(cue.text) for cue in base_result.emitted_cues}
    strict_map = {cue.cue_index: normalize_text(cue.text) for cue in strict_result.emitted_cues} if strict_result else {}
    final_map = {cue.cue_index: normalize_text(cue.text) for cue in final_result.emitted_cues}
    diffs: List[dict] = []
    for cue_index in offending_cue_indices:
        diffs.append(
            {
                "cue_index": cue_index,
                "base_text": base_map.get(cue_index, ""),
                "strict_candidate_text": strict_map.get(cue_index, ""),
                "final_text": final_map.get(cue_index, ""),
            }
        )
    return diffs


def _translate_block_recursive(
    block: TranslationBlock,
    translator: BaseTranslator,
    config: RuntimeConfig,
    glossary_store: Optional[GlossaryStore],
    metrics: Optional[TranslationMetrics],
    depth: int,
    ancestor_failure_signatures: Tuple[Tuple[str, ...], ...],
) -> List[Cue]:
    if metrics:
        metrics.blocks_started += 1

    glossary_terms = _glossary_terms_for_block(block, glossary_store)
    phase1_result, phase1_attempts, phase1_failure_reasons = _run_phase1_with_retry(block, translator, glossary_terms, config)

    if metrics:
        if phase1_attempts > 1:
            metrics.phase1_retry_blocks += 1
        if phase1_result is None:
            metrics.phase1_failed_blocks += 1
            metrics.add_reasons("failure", phase1_failure_reasons)
        else:
            metrics.phase1_success_blocks += 1

    if phase1_result is None:
        signature = _failure_signature(phase1_failure_reasons or ["phase1_structure_failure"])
        if len(block.cues) > 1 and depth < config.max_split_depth and signature not in ancestor_failure_signatures:
            if metrics:
                metrics.smaller_block_fallbacks += 1
            output: List[Cue] = []
            for child in _split_block(block):
                output.extend(
                    _translate_block_recursive(
                        child,
                        translator,
                        config,
                        glossary_store,
                        metrics,
                        depth + 1,
                        ancestor_failure_signatures + ((signature,) if signature else ()),
                    )
                )
            return output

        if len(block.cues) == 1 and metrics:
            metrics.single_cue_source_fallbacks += 1
        warn(f"Preserving source text for cues {[cue.index for cue in block.cues]} after Phase1 structure failure")
        return _wrap_phase_result(block, _fallback_source_result(block, "phase1_structure_failure"), config, metrics)

    pre_gate = pre_wrap_gate(block, phase1_result.emitted_cues, glossary_terms, config)
    if metrics:
        metrics.add_reasons("pre_wrap", pre_gate.repair_reasons + pre_gate.warning_reasons)
        metrics.add_reasons("failure", pre_gate.repair_reasons)

    wrapped_phase1 = _wrap_phase_result(block, phase1_result, config, metrics)
    post_gate = post_wrap_gate(wrapped_phase1, config)
    if metrics:
        if post_gate.repair_reasons or post_gate.warning_reasons:
            metrics.post_wrap_failure_blocks += 1
        metrics.add_reasons("post_wrap", post_gate.repair_reasons + post_gate.warning_reasons)
        metrics.add_reasons("failure", post_gate.repair_reasons)

    final_result = phase1_result
    final_pre = pre_gate
    final_post = post_gate
    strict_phase1: Optional[PhaseTranslationResult] = None

    style_retry_reasons = _style_retry_candidate_reasons(phase1_result, pre_gate, post_gate)
    if style_retry_reasons:
        offending_cue_indices, offending_spans, preferred_actions = _style_retry_feedback(
            block,
            pre_gate,
            post_gate,
            style_retry_reasons,
        )
        offending_cue_index_set = set(offending_cue_indices)
        protected_cue_indices = (
            [
                cue.index
                for cue in block.cues
                if cue.index not in offending_cue_index_set
            ]
            if offending_cue_index_set
            else []
        )
        if metrics:
            metrics.style_retry_trace = {
                "reasons": list(style_retry_reasons),
                "offending_cue_indices": list(offending_cue_indices),
                "protected_cue_indices": list(protected_cue_indices),
                "offending_spans": list(offending_spans),
                "preferred_actions": list(preferred_actions),
                "base_phase1_emitted_cues": _serialize_emitted_cues(phase1_result),
            }
        strict_phase1 = _run_phase1_style_retry(
            block,
            phase1_result,
            style_retry_reasons,
            protected_cue_indices,
            offending_cue_indices,
            offending_spans,
            preferred_actions,
            translator,
            glossary_terms,
            metrics,
        )
        if strict_phase1 is not None:
            if metrics:
                metrics.add_strict_retry_candidate_risk_flags(strict_phase1.risk_flags)
                metrics.style_retry_trace["strict_candidate_emitted_cues"] = _serialize_emitted_cues(strict_phase1)
                metrics.style_retry_trace["strict_candidate_risk_flags"] = list(strict_phase1.risk_flags)
                metrics.style_retry_trace["accepted"] = False
            strict_pre = pre_wrap_gate(block, strict_phase1.emitted_cues, glossary_terms, config)
            wrapped_strict = _wrap_phase_result(block, strict_phase1, config, metrics)
            strict_post = post_wrap_gate(wrapped_strict, config)
            if _candidate_is_overedited(
                block,
                phase1_result,
                pre_gate,
                post_gate,
                strict_phase1,
                strict_pre,
                strict_post,
                style_retry_reasons,
            ):
                if metrics:
                    metrics.style_retry_rejected += 1
                warn(
                    f"Rejected strict style retry for cues {[cue.index for cue in block.cues]} "
                    f"after style warnings {style_retry_reasons}"
                )
            else:
                final_result, final_pre, final_post, accepted = _choose_better_style_candidate(
                    final_result,
                    final_pre,
                    final_post,
                    strict_phase1,
                    strict_pre,
                    strict_post,
                    style_retry_reasons,
                    protected_cue_indices,
                )
                if metrics:
                    if accepted:
                        metrics.style_retry_accepted += 1
                    else:
                        metrics.style_retry_rejected += 1
                    metrics.style_retry_trace["accepted"] = accepted
        elif metrics:
            metrics.style_retry_trace["strict_candidate_emitted_cues"] = []
            metrics.style_retry_trace["strict_candidate_risk_flags"] = []
            metrics.style_retry_trace["accepted"] = False

    if metrics:
        metrics.add_phase1_risk_flags(final_result.risk_flags)
        if metrics.style_retry_trace:
            metrics.style_retry_trace["final_emitted_cues"] = _serialize_emitted_cues(final_result)
            metrics.style_retry_trace["final_risk_flags"] = list(final_result.risk_flags)
            metrics.style_retry_trace["offending_cue_diffs"] = _offending_cue_diffs(
                phase1_result,
                strict_phase1,
                final_result,
                metrics.style_retry_trace.get("offending_cue_indices", []),
            )

    if config.repair_enabled and (final_pre.repair_needed or final_post.repair_needed):
        failure_reasons = _repair_candidate_reasons(block, final_pre, final_post)
        repaired = None
        if failure_reasons:
            repaired = _run_phase2_repair(block, final_result, failure_reasons, translator, glossary_terms, config, metrics)
        if repaired is not None:
            repaired_pre = pre_wrap_gate(block, repaired.emitted_cues, glossary_terms, config)
            wrapped_repaired = _wrap_phase_result(block, repaired, config, metrics)
            repaired_post = post_wrap_gate(wrapped_repaired, config)
            if _candidate_is_overedited(
                block,
                final_result,
                final_pre,
                final_post,
                repaired,
                repaired_pre,
                repaired_post,
                failure_reasons,
            ):
                if metrics:
                    metrics.repair_rejected += 1
                warn(
                    f"Rejected over-edited repair for cues {[cue.index for cue in block.cues]} "
                    f"after changing too many cues"
                )
            else:
                final_result, final_pre, final_post, accepted = _choose_better_candidate(
                    final_result,
                    final_pre,
                    final_post,
                    repaired,
                    repaired_pre,
                    repaired_post,
                )
                if metrics:
                    if accepted:
                        metrics.repair_accepted += 1
                    else:
                        metrics.repair_rejected += 1

    final_signature = _failure_signature(final_pre.repair_reasons + final_post.repair_reasons)
    if final_signature and final_signature in ancestor_failure_signatures:
        warn(
            f"Stopping recursive fallback for cues {[cue.index for cue in block.cues]} "
            f"because failure signature repeated: {', '.join(final_signature)}"
        )
        wrapped_final = _wrap_phase_result(block, final_result, config, metrics)
        if len(block.cues) == 1 and metrics:
            metrics.single_cue_issue_keeps += 1
        return wrapped_final

    if (final_pre.repair_needed or final_post.repair_needed) and len(block.cues) > 1:
        if depth < config.max_split_depth:
            if metrics:
                metrics.smaller_block_fallbacks += 1
            output: List[Cue] = []
            for child in _split_block(block):
                output.extend(
                    _translate_block_recursive(
                        child,
                        translator,
                        config,
                        glossary_store,
                        metrics,
                        depth + 1,
                        ancestor_failure_signatures + ((final_signature,) if final_signature else ()),
                    )
                )
            return output
        warn(
            f"Max split depth reached for cues {[cue.index for cue in block.cues]}; "
            f"keeping best available result"
        )

    wrapped_final = _wrap_phase_result(block, final_result, config, metrics)
    if final_pre.repair_needed or final_post.repair_needed:
        warn(
            f"Keeping cues {[cue.index for cue in block.cues]} despite remaining issues: "
            f"{', '.join(final_pre.repair_reasons + final_post.repair_reasons)}"
        )
        if len(block.cues) == 1 and metrics:
            metrics.single_cue_issue_keeps += 1

    return wrapped_final


def translate_srt(
    cues: List[Cue],
    translator: BaseTranslator,
    config: RuntimeConfig,
    glossary_store: Optional[GlossaryStore] = None,
    metrics: Optional[TranslationMetrics] = None,
) -> List[Cue]:
    blocks = build_translation_blocks(cues, config)
    output: List[Cue] = []
    for block in blocks:
        output.extend(
            _translate_block_recursive(
                block,
                translator,
                config,
                glossary_store,
                metrics,
                depth=0,
                ancestor_failure_signatures=(),
            )
        )
    if metrics:
        metrics.note_final_cues(output)
    return output
