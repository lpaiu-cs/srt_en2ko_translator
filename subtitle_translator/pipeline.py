from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .blocks import build_translation_blocks
from .config import RuntimeConfig
from .glossary import GlossaryStore
from .metrics import TranslationMetrics
from .models import Cue, EmittedCue, PhaseTranslationResult, RepairRequest, TranslationBlock, TranslationRequest
from .quality import _looks_like_duplicate_proposition, post_wrap_gate, pre_wrap_gate, validate_phase_structure
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
    "unsupported_head_marker",
    "unsupported_explanatory_tail",
    "duplicate_restatement",
}

STYLE_RISK_TO_REASON = {
    "unsupported_head_marker_risk": "unsupported_head_marker",
    "unsupported_explanatory_tail_risk": "unsupported_explanatory_tail",
    "duplicate_restatement_risk": "duplicate_restatement",
}

STYLE_REASON_TO_RISK = {
    reason: risk_flag
    for risk_flag, reason in STYLE_RISK_TO_REASON.items()
}

_STRONG_CLOSURE_RE = re.compile(r"(입니다|합니다|됩니다|거죠|겁니다|있습니다|없습니다|이었다|이다|한다|했다)(?:[.!?\"'”’)\]]*)$")
_PURPOSE_MARKER_RE = re.compile(r"(위해|위해서|하기 위해|하려고|하도록|하려면)")
_THAT_CLAUSE_MARKER_RE = re.compile(r"(라고|라는|다고|다는|하면|된다면|때문에|인지|라는 점)")
_RELATIVE_CLAUSE_MARKER_RE = re.compile(r"(하는|되는|했던|였던|인|할|될|보여주는|가지는)")
_COMPARISON_MARKER_RE = re.compile(r"(처럼|같이|보다|만큼|에 비해|와 달리|대신)")
_CONTINUATION_TAIL_RE = re.compile(r"(으로|에서|와|과|로|에|의|중|부터|까지|하며|하면서|한 채|및)$")
_FRAGMENTARY_CONTINUATION_RE = re.compile(r"(의|와|과|로|으로|에|에서|부터|까지|만|및|처럼|보다|중|쪽의)$")
_CONTINUATION_OVERLAP_SUFFIX_RE = re.compile(r"(입니다|입니다만|이죠|예요|에요|거죠|겁니다|합니다|합니다만|됩니다|되죠|돼요|요)$")
_PURPOSE_TAIL_NORMALIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"기 위해서(?:입니다|인 거죠|인 겁니다|이죠|예요|에요|요)?(?:[.!?\"'”’),，、\s]*)$"), "기 위해"),
    (re.compile(r"기 위해(?:입니다|인 거죠|인 겁니다|이죠|예요|에요|요)?(?:[.!?\"'”’),，、\s]*)$"), "기 위해"),
    (re.compile(r"기 위한(?:\s*(?:것입니다|것이죠|것이에요|겁니다|거죠|거예요|거에요))?(?:[.!?\"'”’),，、\s]*)$"), "기 위해"),
    (re.compile(r"(하려고)(?:\s*(?:합니다|한 것입니다|한 거죠|한 겁니다|해요|했어요))?(?:[.!?\"'”’),，、\s]*)$"), r"\1"),
    (re.compile(r"(하도록)(?:\s*(?:합니다|한 것입니다|한 거죠|한 겁니다|해요|했어요))?(?:[.!?\"'”’),，、\s]*)$"), r"\1"),
    (re.compile(r"(하려면)(?:\s*(?:됩니다|합니다|하는 겁니다|하는 거죠))?(?:[.!?\"'”’),，、\s]*)$"), r"\1"),
)


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
        "unsupported_head_marker": "drop_head_marker",
        "unsupported_explanatory_tail": "trim_explanatory_tail",
        "duplicate_restatement": "delete_repeat_local",
    }
    seen_spans: set[str] = set()

    for reason in style_retry_reasons:
        details = pre_gate.warning_details.get(reason, []) + post_gate.warning_details.get(reason, [])
        if not details and reason in default_actions:
            if reason == "duplicate_restatement" and len(block.cues) >= 2:
                tail_cue = block.cues[-1]
                normalized_tail = tail_cue.text.strip().lower()
                if normalized_tail.startswith("to "):
                    source_tail_type = "purpose_tail"
                elif normalized_tail.startswith(("that ", "because ", "if ", "while ")):
                    source_tail_type = "that_clause_tail"
                elif normalized_tail.startswith(("which ", "who ", "whom ", "whose ", "where ", "when ", "why ")):
                    source_tail_type = "relative_clause_tail"
                elif normalized_tail.startswith(("as ", "than ", "rather than ", "compared to ", "like ")):
                    source_tail_type = "comparison_tail"
                else:
                    source_tail_type = "continuation_tail" if normalized_tail.startswith(("for ", "by ", "of ", "with ", "without ", "into ", "from ")) else None
                fallback_action = (
                    "restore_missing_tail"
                    if source_tail_type
                    else default_actions[reason]
                )
                offending_cues.append(tail_cue.index)
                offending_spans.append(
                    {
                        "cue_index": tail_cue.index,
                        "cue_indices": [tail_cue.index],
                        "span_text": "",
                        "source_cue_text": tail_cue.text,
                        "source_tail_type": source_tail_type,
                        "issue": "duplicate_restatement",
                        "preferred_action": fallback_action,
                    }
                )
                preferred_actions.append(fallback_action)
                continue
            preferred_actions.append(default_actions[reason])
            continue
        for detail in details:
            cue_index = detail.get("cue_index")
            if isinstance(cue_index, int) and cue_index not in offending_cues and cue_index >= 0:
                offending_cues.append(cue_index)
            elif not isinstance(cue_index, int):
                cue_indices = detail.get("cue_indices")
                if isinstance(cue_indices, list):
                    for nested_cue_index in cue_indices:
                        if isinstance(nested_cue_index, int) and nested_cue_index not in offending_cues and nested_cue_index >= 0:
                            offending_cues.append(nested_cue_index)
            action = detail.get("preferred_action")
            if isinstance(action, str) and action:
                preferred_actions.append(action)
            signature = json.dumps(detail, ensure_ascii=False, sort_keys=True)
            if signature in seen_spans:
                continue
            seen_spans.add(signature)
            offending_spans.append(detail)

    return sorted(offending_cues), offending_spans, list(dict.fromkeys(preferred_actions))


def _source_tail_type_from_text(source_text: str) -> str | None:
    normalized = normalize_text(source_text).casefold()
    if not normalized:
        return None
    if normalized.startswith("to "):
        return "purpose_tail"
    if normalized.startswith(("that ", "because ", "if ", "while ")):
        return "that_clause_tail"
    if normalized.startswith(("which ", "who ", "whom ", "whose ", "where ", "when ", "why ")):
        return "relative_clause_tail"
    if normalized.startswith(("as ", "than ", "rather than ", "compared to ", "like ")):
        return "comparison_tail"
    if normalized.startswith(("for ", "by ", "of ", "with ", "without ", "into ", "from ")):
        return "continuation_tail"
    return None


def _looks_fragmentary_continuation(text: str) -> bool:
    normalized = normalize_text(text).rstrip(".,!?;:'\"")
    if not normalized:
        return True
    if _STRONG_CLOSURE_RE.search(normalized):
        return False
    if len(normalized) <= 18 and _FRAGMENTARY_CONTINUATION_RE.search(normalized):
        return True
    return False


def _continuation_overlap_stem(text: str) -> str:
    normalized = normalize_text(text).rstrip(".,!?;:'\"")
    normalized = _CONTINUATION_OVERLAP_SUFFIX_RE.sub("", normalized).rstrip(" ,，")
    return normalized


def _continuation_detector_miss_feedback(
    block: TranslationBlock,
    result: PhaseTranslationResult,
) -> tuple[List[int], List[dict], List[str]] | None:
    if len(block.cues) < 2 or "carry_context_only" not in block.lint_actions:
        return None

    tail_source_type = _source_tail_type_from_text(block.cues[-1].text)
    if tail_source_type != "continuation_tail":
        return None

    text_map = _result_text_map(result)
    tail_cue = block.cues[-1]
    prev_cue = block.cues[-2]
    tail_text = text_map.get(tail_cue.index, "")
    prev_text = text_map.get(prev_cue.index, "")
    overlap_stem = _continuation_overlap_stem(tail_text)
    if not (
        _looks_fragmentary_continuation(tail_text)
        or (tail_text and prev_text and _looks_like_duplicate_proposition(prev_text, tail_text))
        or (overlap_stem and len(overlap_stem) >= 8 and overlap_stem in prev_text)
    ):
        return None

    span_text = tail_text.rstrip(".,!?;:'\"")
    span = {
        "cue_index": tail_cue.index,
        "cue_indices": [prev_cue.index, tail_cue.index],
        "span_text": span_text,
        "left_text": prev_text,
        "right_text": tail_text,
        "source_cue_text": tail_cue.text,
        "source_tail_type": "continuation_tail",
        "issue": "duplicate_restatement",
        "preferred_action": "restore_missing_tail",
        "trigger_reason": "detector_miss",
    }
    return [tail_cue.index], [span], ["restore_missing_tail"]


def _classify_not_invoked_reason(
    block: TranslationBlock,
    result: PhaseTranslationResult,
) -> str:
    if not block.cues:
        return "no_offending_cue_after_phase1"

    text_map = _result_text_map(result)
    tail_source_type = _source_tail_type_from_text(block.cues[-1].text)
    if tail_source_type == "continuation_tail":
        tail_text = text_map.get(block.cues[-1].index, "")
        prev_text = text_map.get(block.cues[-2].index, "") if len(block.cues) >= 2 else ""
        overlap_stem = _continuation_overlap_stem(tail_text)
        if _looks_fragmentary_continuation(tail_text) or (
            "carry_context_only" in block.lint_actions
            and tail_text
            and prev_text
            and (
                _looks_like_duplicate_proposition(prev_text, tail_text)
                or (overlap_stem and len(overlap_stem) >= 8 and overlap_stem in prev_text)
            )
        ):
            return "detector_miss" if "carry_context_only" in block.lint_actions else "low_tail_confidence"
        if tail_text:
            return "acceptable_absorption"
        return "already_absorbed_by_cue1"
    if text_map.get(block.cues[-1].index, ""):
        return "acceptable_absorption"
    return "no_offending_cue_after_phase1"


def _effective_style_retry_feedback(
    block: TranslationBlock,
    result: PhaseTranslationResult,
    pre_gate,
    post_gate,
) -> tuple[List[str], List[int], List[dict], List[str], str | None]:
    style_retry_reasons = _style_retry_candidate_reasons(result, pre_gate, post_gate)
    if style_retry_reasons:
        offending_cue_indices, offending_spans, preferred_actions = _style_retry_feedback(
            block,
            pre_gate,
            post_gate,
            style_retry_reasons,
        )
        return style_retry_reasons, offending_cue_indices, offending_spans, preferred_actions, None

    forced_feedback = _continuation_detector_miss_feedback(block, result)
    if forced_feedback is not None:
        offending_cue_indices, offending_spans, preferred_actions = forced_feedback
        return ["duplicate_restatement"], offending_cue_indices, offending_spans, preferred_actions, "detector_miss"

    return [], [], [], [], _classify_not_invoked_reason(block, result)


def _style_warning_spans(pre_gate, post_gate, reasons: Sequence[str]) -> List[dict]:
    spans: List[dict] = []
    seen: set[str] = set()
    for reason in reasons:
        for detail in pre_gate.warning_details.get(reason, []) + post_gate.warning_details.get(reason, []):
            signature = json.dumps(detail, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            spans.append(detail)
    return spans


def _style_warning_action_details(pre_gate, post_gate) -> List[dict]:
    spans: List[dict] = []
    seen: set[str] = set()
    for details in list(pre_gate.warning_details.values()) + list(post_gate.warning_details.values()):
        for detail in details:
            signature = json.dumps(detail, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            spans.append(detail)
    return spans


def _style_spans_for_actions(spans: Sequence[dict], actions: Sequence[str]) -> List[dict]:
    allowed = {action for action in actions if action}
    return [span for span in spans if span.get("preferred_action") in allowed]


def _protected_cue_indices_for_spans(block: TranslationBlock, spans: Sequence[dict]) -> List[int]:
    offending: set[int] = set()
    for span in spans:
        cue_index = span.get("cue_index")
        if isinstance(cue_index, int):
            offending.add(cue_index)
            continue
        cue_indices = span.get("cue_indices")
        if isinstance(cue_indices, list):
            for nested_cue_index in cue_indices:
                if isinstance(nested_cue_index, int):
                    offending.add(nested_cue_index)
    return [cue.index for cue in block.cues if cue.index not in offending]


def _remove_head_marker_once(text: str, span_text: str) -> str:
    normalized = normalize_text(text)
    marker = normalize_text(span_text)
    if not marker:
        return normalized
    if normalized.startswith(marker):
        trimmed = normalized[len(marker):].lstrip(" ,，")
        return normalize_text(trimmed)
    pattern = re.compile(rf"^{re.escape(marker)}\s*[,，]?\s*")
    return normalize_text(pattern.sub("", normalized, count=1))


def _remove_tail_once(text: str, span_text: str) -> str:
    normalized = normalize_text(text)
    target = normalize_text(span_text)
    if not target:
        return normalized
    pattern = re.compile(rf"{re.escape(target)}\s*$")
    updated = pattern.sub("", normalized, count=1).rstrip(" ,，")
    return normalize_text(updated)


def _normalize_purpose_tail_fragment(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return normalized
    updated = normalized
    for pattern, replacement in _PURPOSE_TAIL_NORMALIZATION_PATTERNS:
        candidate = normalize_text(pattern.sub(replacement, updated, count=1))
        if candidate != updated:
            updated = candidate
            break
    return updated.rstrip(" ,，")


def _apply_purpose_tail_post_normalization(
    candidate_result: PhaseTranslationResult,
    offending_spans: Sequence[dict],
) -> tuple[PhaseTranslationResult, List[dict]]:
    purpose_spans = [
        span
        for span in offending_spans
        if span.get("preferred_action") == "restore_missing_tail" and span.get("source_tail_type") == "purpose_tail"
    ]
    if not purpose_spans:
        return candidate_result, []

    text_by_index = {cue.cue_index: normalize_text(cue.text) for cue in candidate_result.emitted_cues}
    applied: List[dict] = []
    for span in purpose_spans:
        cue_index = span.get("cue_index")
        if not isinstance(cue_index, int):
            continue
        current_text = text_by_index.get(cue_index, "")
        if not current_text:
            continue
        normalized = _normalize_purpose_tail_fragment(current_text)
        if normalized and normalized != current_text:
            text_by_index[cue_index] = normalized
            applied.append(
                {
                    "cue_index": cue_index,
                    "before": current_text,
                    "after": normalized,
                    "source_tail_type": "purpose_tail",
                    "preferred_action": "restore_missing_tail",
                    "normalization": "purpose_tail_fragment",
                }
            )

    if not applied:
        return candidate_result, []

    return (
        PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=cue.cue_index, text=text_by_index.get(cue.cue_index, normalize_text(cue.text)))
                for cue in candidate_result.emitted_cues
            ],
            risk_flags=list(candidate_result.risk_flags),
        ),
        applied,
    )


def _apply_deterministic_style_micro_edits(
    current_result: PhaseTranslationResult,
    spans: Sequence[dict],
) -> PhaseTranslationResult | None:
    if not spans:
        return None
    text_by_index = {cue.cue_index: normalize_text(cue.text) for cue in current_result.emitted_cues}
    changed = False
    edited_actions = {"drop_head_marker", "trim_explanatory_tail"}
    applied_reasons = {span.get("issue") for span in spans if span.get("preferred_action") in edited_actions}
    for span in spans:
        action = span.get("preferred_action")
        cue_index = span.get("cue_index")
        if action not in edited_actions or not isinstance(cue_index, int):
            continue
        current_text = text_by_index.get(cue_index, "")
        if not current_text:
            continue
        if action == "drop_head_marker":
            updated = _remove_head_marker_once(current_text, str(span.get("span_text", "")))
        else:
            updated = _remove_tail_once(current_text, str(span.get("span_text", "")))
        if updated and updated != current_text:
            text_by_index[cue_index] = updated
            changed = True
    if not changed:
        return None
    edited_risks = {STYLE_REASON_TO_RISK[reason] for reason in applied_reasons if reason in STYLE_REASON_TO_RISK}
    filtered_risks = [risk for risk in current_result.risk_flags if risk not in edited_risks]
    return PhaseTranslationResult(
        emitted_cues=[
            EmittedCue(cue_index=cue.cue_index, text=text_by_index.get(cue.cue_index, normalize_text(cue.text)))
            for cue in current_result.emitted_cues
        ],
        risk_flags=filtered_risks,
    )


def _result_text_map(result: PhaseTranslationResult) -> dict[int, str]:
    return {cue.cue_index: normalize_text(cue.text) for cue in result.emitted_cues}


def _style_warning_signature(span: dict) -> str:
    return json.dumps(span, ensure_ascii=False, sort_keys=True)


def _matching_warning_span(candidate_pre, candidate_post, issue: str, cue_index: int) -> dict | None:
    for detail in candidate_pre.warning_details.get(issue, []) + candidate_post.warning_details.get(issue, []):
        if detail.get("cue_index") == cue_index:
            return detail
    return None


def _action_specific_style_rejection_causes(
    current_result: PhaseTranslationResult,
    candidate_result: PhaseTranslationResult,
    candidate_pre,
    candidate_post,
    offending_spans: Sequence[dict],
) -> List[str]:
    current_text_map = _result_text_map(current_result)
    candidate_text_map = _result_text_map(candidate_result)
    rejection_causes: List[str] = []

    for span in offending_spans:
        action = span.get("preferred_action")
        cue_index = span.get("cue_index")
        if not isinstance(action, str) or not isinstance(cue_index, int):
            continue

        current_text = current_text_map.get(cue_index, "")
        candidate_text = candidate_text_map.get(cue_index, "")

        if action == "restore_missing_tail":
            if candidate_text == current_text:
                rejection_causes.append("restore_tail_not_changed")
                continue
            if not candidate_text:
                rejection_causes.append("restore_tail_empty")
                continue
            left_text = normalize_text(str(span.get("left_text", "")))
            if left_text and _looks_like_duplicate_proposition(left_text, candidate_text):
                rejection_causes.append("restore_tail_duplicate_persisted")
            source_tail_type = span.get("source_tail_type")
            rejection_causes.extend(_tail_type_specific_restore_rejection_causes(candidate_text, source_tail_type))
            matched_warning = _matching_warning_span(candidate_pre, candidate_post, "duplicate_restatement", cue_index)
            if matched_warning and matched_warning.get("preferred_action") == "restore_missing_tail":
                rejection_causes.append("restore_tail_warning_persisted")
        elif action == "delete_repeat_local":
            if candidate_text == current_text:
                rejection_causes.append("delete_repeat_not_changed")
        elif action == "drop_head_marker":
            span_text = normalize_text(str(span.get("span_text", "")))
            if span_text and candidate_text.startswith(span_text):
                rejection_causes.append("head_marker_persisted")
        elif action == "trim_explanatory_tail":
            span_text = normalize_text(str(span.get("span_text", "")))
            if span_text and candidate_text.endswith(span_text):
                rejection_causes.append("explanatory_tail_persisted")

    return list(dict.fromkeys(rejection_causes))


def _tail_type_specific_restore_rejection_causes(candidate_text: str, source_tail_type: str | None) -> List[str]:
    normalized = normalize_text(candidate_text)
    if not source_tail_type or not normalized:
        return []

    rejection_causes: List[str] = []
    strong_closure = _STRONG_CLOSURE_RE.search(normalized) is not None

    if source_tail_type == "purpose_tail":
        if _PURPOSE_MARKER_RE.search(normalized) is None:
            rejection_causes.append("restore_tail_purpose_marker_missing")
        if strong_closure:
            rejection_causes.append("restore_tail_overclosed_for_purpose")
    elif source_tail_type == "that_clause_tail":
        if _THAT_CLAUSE_MARKER_RE.search(normalized) is None:
            rejection_causes.append("restore_tail_that_clause_shape_missing")
        if strong_closure:
            rejection_causes.append("restore_tail_overclosed_for_that_clause")
    elif source_tail_type == "relative_clause_tail":
        if _RELATIVE_CLAUSE_MARKER_RE.search(normalized) is None:
            rejection_causes.append("restore_tail_relative_clause_shape_missing")
        if strong_closure:
            rejection_causes.append("restore_tail_overclosed_for_relative_clause")
    elif source_tail_type == "comparison_tail":
        if _COMPARISON_MARKER_RE.search(normalized) is None:
            rejection_causes.append("restore_tail_comparison_shape_missing")
        if strong_closure:
            rejection_causes.append("restore_tail_overclosed_for_comparison")
    elif source_tail_type == "continuation_tail":
        if _CONTINUATION_TAIL_RE.search(normalized) is None and strong_closure:
            rejection_causes.append("restore_tail_overclosed_for_continuation")

    return rejection_causes


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
) -> tuple[Optional[PhaseTranslationResult], List[str]]:
    offending_cue_index_set = {
        int(span["cue_index"])
        for span in offending_spans
        if isinstance(span.get("cue_index"), int)
    }
    strict_retry_mode = "full_block"
    if (
        len(offending_cue_index_set) == 1
        and protected_cue_indices
        and any(span.get("trigger_reason") == "detector_miss" for span in offending_spans)
        and all(
            span.get("preferred_action") == "restore_missing_tail"
            and span.get("source_tail_type") == "continuation_tail"
            for span in offending_spans
        )
    ):
        strict_retry_mode = "offending_cue_only"
    if metrics:
        metrics.style_retry_invocations += 1
    request = TranslationRequest(
        block=block,
        glossary_terms=glossary_terms,
        strict_style_retry=True,
        strict_retry_mode=strict_retry_mode,
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
        return None, ["strict_retry_exception"]

    validation = validate_phase_structure(block, retried.emitted_cues)
    if validation.valid:
        return retried, []

    warn(
        f"Strict Phase1 retry returned invalid structure for cues {[cue.index for cue in block.cues]}: "
        f"{', '.join(validation.reasons)}"
    )
    return None, [f"strict_retry_{reason}" for reason in validation.reasons]


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
    offending_spans: Sequence[dict] | None = None,
):
    rejection_causes: List[str] = []
    forced_detector_retry = any(span.get("trigger_reason") == "detector_miss" for span in (offending_spans or []))
    offending_cue_indices = {
        int(span["cue_index"])
        for span in (offending_spans or [])
        if isinstance(span.get("cue_index"), int)
    }
    if protected_cue_indices:
        changed_protected = set(_changed_cue_indices(current_result, candidate_result)) & set(protected_cue_indices)
        if changed_protected:
            rejection_causes.append("protected_cue_touched")
            return current_result, current_pre, current_post, False, rejection_causes

    current_repair_reasons = _gate_repair_reason_set(current_pre, current_post)
    candidate_repair_reasons = _gate_repair_reason_set(candidate_pre, candidate_post)
    if candidate_repair_reasons - current_repair_reasons:
        rejection_causes.append("new_repair_reason_introduced")
        return current_result, current_pre, current_post, False, rejection_causes
    if _post_wrap_score(candidate_post) > _post_wrap_score(current_post):
        rejection_causes.append("post_wrap_regression")
        return current_result, current_pre, current_post, False, rejection_causes

    action_rejections = _action_specific_style_rejection_causes(
        current_result,
        candidate_result,
        candidate_pre,
        candidate_post,
        offending_spans or [],
    )
    if action_rejections:
        rejection_causes.extend(action_rejections)
        return current_result, current_pre, current_post, False, list(dict.fromkeys(rejection_causes))

    current_targeted = _gate_warning_occurrences(current_pre, current_post, style_retry_reasons)
    candidate_targeted = _gate_warning_occurrences(candidate_pre, candidate_post, style_retry_reasons)
    current_risk_targeted = _style_risk_occurrences(current_result, style_retry_reasons)
    candidate_risk_targeted = _style_risk_occurrences(candidate_result, style_retry_reasons)
    changed_offending = bool(set(_changed_cue_indices(current_result, candidate_result)) & offending_cue_indices)
    if (
        forced_detector_retry
        and changed_offending
        and current_targeted == 0
        and candidate_targeted == 0
        and candidate_risk_targeted <= current_risk_targeted
        and _gate_score(candidate_pre, candidate_post) <= _gate_score(current_pre, current_post)
    ):
        return candidate_result, candidate_pre, candidate_post, True, []
    if candidate_targeted < current_targeted:
        return candidate_result, candidate_pre, candidate_post, True, []
    if candidate_targeted == current_targeted and candidate_risk_targeted < current_risk_targeted:
        return candidate_result, candidate_pre, candidate_post, True, []

    if candidate_targeted == current_targeted and _gate_score(candidate_pre, candidate_post) < _gate_score(current_pre, current_post):
        return candidate_result, candidate_pre, candidate_post, True, []

    if candidate_targeted >= current_targeted:
        rejection_causes.append("no_targeted_style_reduction")
        for reason in style_retry_reasons:
            current_reason_count = _gate_warning_occurrences(current_pre, current_post, [reason])
            candidate_reason_count = _gate_warning_occurrences(candidate_pre, candidate_post, [reason])
            if candidate_reason_count >= current_reason_count and current_reason_count > 0:
                rejection_causes.append(f"{reason}_persisted")
    if candidate_risk_targeted >= current_risk_targeted and current_risk_targeted > 0:
        rejection_causes.append("style_risk_not_reduced")

    return current_result, current_pre, current_post, False, list(dict.fromkeys(rejection_causes))


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


def _strict_accept_mode(post_normalizations: Sequence[dict]) -> str:
    return "postnorm_salvaged_accept" if post_normalizations else "strict_direct_accept"


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

    (
        style_retry_reasons,
        initial_offending_cue_indices,
        initial_offending_spans,
        initial_preferred_actions,
        initial_not_invoked_reason,
    ) = _effective_style_retry_feedback(
        block,
        final_result,
        final_pre,
        final_post,
    )
    if style_retry_reasons:
        if metrics:
            metrics.style_retry_trace = {
                "reasons": list(style_retry_reasons),
                "offending_cue_indices": list(initial_offending_cue_indices),
                "protected_cue_indices": list(_protected_cue_indices_for_spans(block, initial_offending_spans)),
                "offending_spans": list(initial_offending_spans),
                "preferred_actions": list(initial_preferred_actions),
                "forced_invocation_reason": initial_not_invoked_reason,
                "base_phase1_emitted_cues": _serialize_emitted_cues(phase1_result),
            }
        deterministic_spans = _style_spans_for_actions(
            initial_offending_spans,
            ["drop_head_marker", "trim_explanatory_tail"],
        )
        if deterministic_spans:
            if metrics:
                metrics.note_style_action_attempts(deterministic_spans, channel="micro_edit")
                metrics.style_retry_trace["micro_edit_attempted"] = True
                metrics.style_retry_trace["micro_edit_spans"] = list(deterministic_spans)
            deterministic_candidate = _apply_deterministic_style_micro_edits(final_result, deterministic_spans)
            if deterministic_candidate is not None:
                deterministic_pre = pre_wrap_gate(block, deterministic_candidate.emitted_cues, glossary_terms, config)
                wrapped_deterministic = _wrap_phase_result(block, deterministic_candidate, config, metrics)
                deterministic_post = post_wrap_gate(wrapped_deterministic, config)
                deterministic_reasons = list(
                    dict.fromkeys(
                        str(span.get("issue"))
                        for span in deterministic_spans
                        if span.get("issue")
                    )
                )
                deterministic_protected = _protected_cue_indices_for_spans(block, deterministic_spans)
                (
                    final_result,
                    final_pre,
                    final_post,
                    deterministic_accepted,
                    deterministic_rejection_causes,
                ) = _choose_better_style_candidate(
                    final_result,
                    final_pre,
                    final_post,
                    deterministic_candidate,
                    deterministic_pre,
                    deterministic_post,
                    deterministic_reasons,
                    deterministic_protected,
                    offending_spans=deterministic_spans,
                )
                if metrics:
                    metrics.note_style_action_outcome(deterministic_spans, deterministic_accepted, channel="micro_edit")
                    metrics.style_retry_trace["micro_edit_candidate_emitted_cues"] = _serialize_emitted_cues(deterministic_candidate)
                    metrics.style_retry_trace["micro_edit_accepted"] = deterministic_accepted
                    metrics.style_retry_trace["micro_edit_rejection_causes"] = list(deterministic_rejection_causes)
            elif metrics:
                metrics.note_style_action_outcome(deterministic_spans, False, channel="micro_edit")
                metrics.style_retry_trace["micro_edit_candidate_emitted_cues"] = []
                metrics.style_retry_trace["micro_edit_accepted"] = False
                metrics.style_retry_trace["micro_edit_rejection_causes"] = ["deterministic_noop"]

        (
            style_retry_reasons,
            offending_cue_indices,
            offending_spans,
            preferred_actions,
            not_invoked_reason,
        ) = _effective_style_retry_feedback(
            block,
            final_result,
            final_pre,
            final_post,
        )
        if style_retry_reasons:
            protected_cue_indices = _protected_cue_indices_for_spans(block, offending_spans)
            if metrics:
                metrics.note_style_action_attempts(offending_spans, channel="strict_retry")
                metrics.style_retry_trace["reasons"] = list(style_retry_reasons)
                metrics.style_retry_trace["offending_cue_indices"] = list(offending_cue_indices)
                metrics.style_retry_trace["protected_cue_indices"] = list(protected_cue_indices)
                metrics.style_retry_trace["offending_spans"] = list(offending_spans)
                metrics.style_retry_trace["preferred_actions"] = list(preferred_actions)
                metrics.style_retry_trace["forced_invocation_reason"] = not_invoked_reason
            strict_phase1, strict_retry_failures = _run_phase1_style_retry(
                block,
                final_result,
                style_retry_reasons,
                protected_cue_indices,
                offending_cue_indices,
                offending_spans,
                preferred_actions,
                translator,
                glossary_terms,
                metrics,
            )
            if metrics and metrics.style_retry_trace is not None:
                metrics.style_retry_trace["strict_retry_mode"] = (
                    "offending_cue_only"
                    if (
                        len({int(span["cue_index"]) for span in offending_spans if isinstance(span.get("cue_index"), int)}) == 1
                        and protected_cue_indices
                        and any(span.get("trigger_reason") == "detector_miss" for span in offending_spans)
                        and all(
                            span.get("preferred_action") == "restore_missing_tail"
                            and span.get("source_tail_type") == "continuation_tail"
                            for span in offending_spans
                        )
                    )
                    else "full_block"
                )
            if strict_phase1 is not None:
                raw_strict_phase1 = strict_phase1
                strict_phase1, strict_post_normalizations = _apply_purpose_tail_post_normalization(
                    strict_phase1,
                    offending_spans,
                )
                if metrics:
                    metrics.style_retry_trace["strict_candidate_raw_emitted_cues"] = _serialize_emitted_cues(raw_strict_phase1)
                    metrics.style_retry_trace["strict_candidate_emitted_cues"] = _serialize_emitted_cues(strict_phase1)
                    metrics.style_retry_trace["strict_candidate_risk_flags"] = list(strict_phase1.risk_flags)
                    metrics.style_retry_trace["strict_candidate_post_normalizations"] = list(strict_post_normalizations)
                    metrics.style_retry_trace["accepted"] = False
                strict_pre = pre_wrap_gate(block, strict_phase1.emitted_cues, glossary_terms, config)
                wrapped_strict = _wrap_phase_result(block, strict_phase1, config, metrics)
                strict_post = post_wrap_gate(wrapped_strict, config)
                if _candidate_is_overedited(
                    block,
                    final_result,
                    final_pre,
                    final_post,
                    strict_phase1,
                    strict_pre,
                    strict_post,
                    style_retry_reasons,
                ):
                    if metrics:
                        metrics.style_retry_rejected += 1
                        metrics.note_style_action_outcome(offending_spans, False, channel="strict_retry")
                        metrics.add_style_retry_rejection_causes(["overedited_candidate"])
                    warn(
                        f"Rejected strict style retry for cues {[cue.index for cue in block.cues]} "
                        f"after style warnings {style_retry_reasons}"
                    )
                    if metrics.style_retry_trace:
                        metrics.style_retry_trace["accept_mode"] = None
                        metrics.style_retry_trace["rejection_causes"] = ["overedited_candidate"]
                else:
                    final_result, final_pre, final_post, accepted, rejection_causes = _choose_better_style_candidate(
                        final_result,
                        final_pre,
                        final_post,
                        strict_phase1,
                        strict_pre,
                        strict_post,
                        style_retry_reasons,
                        protected_cue_indices,
                        offending_spans=offending_spans,
                    )
                    if metrics:
                        metrics.note_style_action_outcome(offending_spans, accepted, channel="strict_retry")
                        if accepted:
                            metrics.style_retry_accepted += 1
                            metrics.add_strict_retry_candidate_risk_flags(strict_phase1.risk_flags)
                            accept_mode = _strict_accept_mode(strict_post_normalizations)
                            metrics.note_style_action_accept_mode(offending_spans, accept_mode, channel="strict_retry")
                            metrics.style_retry_trace["accept_mode"] = accept_mode
                        else:
                            metrics.style_retry_rejected += 1
                            metrics.add_style_retry_rejection_causes(rejection_causes)
                            metrics.style_retry_trace["accept_mode"] = None
                        metrics.style_retry_trace["accepted"] = accepted
                        metrics.style_retry_trace["rejection_causes"] = rejection_causes
            elif metrics:
                metrics.style_retry_rejected += 1
                metrics.note_style_action_outcome(offending_spans, False, channel="strict_retry")
                metrics.style_retry_trace["strict_candidate_raw_emitted_cues"] = []
                metrics.style_retry_trace["strict_candidate_emitted_cues"] = []
                metrics.style_retry_trace["strict_candidate_risk_flags"] = []
                metrics.style_retry_trace["strict_candidate_post_normalizations"] = []
                metrics.style_retry_trace["accept_mode"] = None
                metrics.style_retry_trace["accepted"] = False
                metrics.style_retry_trace["rejection_causes"] = list(strict_retry_failures)
                metrics.add_style_retry_rejection_causes(strict_retry_failures)
        elif metrics:
            metrics.style_retry_trace["not_invoked_reason"] = not_invoked_reason
            metrics.style_retry_trace["strict_candidate_raw_emitted_cues"] = []
            metrics.style_retry_trace["strict_candidate_emitted_cues"] = []
            metrics.style_retry_trace["strict_candidate_risk_flags"] = []
            metrics.style_retry_trace["strict_candidate_post_normalizations"] = []
            metrics.style_retry_trace["accept_mode"] = None
            metrics.style_retry_trace["accepted"] = False
            metrics.style_retry_trace["rejection_causes"] = ["resolved_by_micro_edit"]
    elif metrics:
        metrics.style_retry_trace = {
            "not_invoked_reason": initial_not_invoked_reason,
        }

    if metrics:
        metrics.add_phase1_risk_flags(final_result.risk_flags)
        remaining_style_spans = _style_warning_action_details(final_pre, final_post)
        metrics.note_style_action_remaining_warnings(remaining_style_spans)
        metrics.note_style_action_remaining_warnings(
            _style_spans_for_actions(remaining_style_spans, ["drop_head_marker", "trim_explanatory_tail"]),
            channel="micro_edit",
        )
        metrics.note_style_action_remaining_warnings(
            _style_spans_for_actions(remaining_style_spans, ["delete_repeat_local", "restore_missing_tail"]),
            channel="strict_retry",
        )
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
