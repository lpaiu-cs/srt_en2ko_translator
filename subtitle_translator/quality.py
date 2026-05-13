from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Dict, Iterable, List

from .config import RuntimeConfig
from .models import Cue, EmittedCue, GlossaryEntry, QualityGateResult, SchemaValidationResult, TranslationBlock
from .splitting import ts_to_ms
from .text import normalize_text


_ENGLISH_SPAN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./+-]{3,}")
_ACRONYM_RE = re.compile(r"^[A-Z]{2,5}$")
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,./%+-]*")
_MODEL_TOKEN_RE = re.compile(r"\b[A-Za-z]+(?:[-]?\d+[A-Za-z0-9-]*)+\b")
_PARTICLE_ONLY_RE = re.compile(
    r"^(은|는|이|가|을|를|에|의|도|만|와|과|로|으로|에서|부터|까지|보다|처럼|하고|및|또는|그리고|하지만|입니다|합니다|죠|겁니다)$"
)
_SENTENCE_END_RE = re.compile(r"(입니다|합니다|됩니다|거죠|겁니다|있습니다|없습니다|했다|한다|했다는|means\.?)$")
_EXPLANATORY_CLOSER_RE = re.compile(
    r"(라는 뜻입니다|거죠|겁니다|라고 볼 수 있습니다|라고 할 수 있습니다)(?:[\"'”’),，、\s]*)$"
)
_RESTATEMENT_MARKER_RE = re.compile(r"^(즉|다시 말해|다시 말하면|말하자면|바로)\b")
_PROPOSITION_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)|[,;:](?:\s*|$)|\s*[—–-]\s+|\n+")
_PURPOSE_TAIL_RE = re.compile(r"^(to\b|in order to\b)", re.IGNORECASE)
_THAT_TAIL_RE = re.compile(r"^(that\b|because\b|if\b|while\b)", re.IGNORECASE)
_RELATIVE_TAIL_RE = re.compile(r"^(which\b|who\b|whom\b|whose\b|where\b|when\b|why\b)", re.IGNORECASE)
_COMPARISON_TAIL_RE = re.compile(r"^(as\b|than\b|rather than\b|compared to\b|like\b)", re.IGNORECASE)
_GENERIC_CONTINUATION_TAIL_RE = re.compile(
    r"^(for\b|by\b|of\b|with\b|without\b|into\b|from\b)",
    re.IGNORECASE,
)
_DISCOURSE_MARKER_HEAD_RE = re.compile(r"^(즉|다시 말해|다시 말하면|말하자면|바로)\s*[,，]?\s*")
_TECHNICAL_CAMEL_RE = re.compile(r"^(?:[A-Z][a-z]+(?:Net|GAN)|[A-Z]{2,}[A-Za-z]+(?:Net|GAN|Prop)?|[A-Z][A-Za-z]+Net|[A-Z][A-Za-z]+GAN)$")
_TECHNICAL_HYPHEN_DIGIT_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9./+-]+$")
_TECHNICAL_CARRY_THROUGH_HINTS = {
    "adam",
    "alexnet",
    "binning",
    "c3d",
    "convnet",
    "dc-gan",
    "gan",
    "gflops",
    "gpt-4",
    "rmsprop",
    "sgd",
    "vgg-16",
}


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def validate_phase_structure(block: TranslationBlock, emitted_cues: List[EmittedCue]) -> SchemaValidationResult:
    reasons: List[str] = []
    source_indices = [cue.index for cue in block.cues]
    emitted_indices = [cue.cue_index for cue in emitted_cues]

    if len(emitted_cues) != len(block.cues):
        reasons.append("cue_count_mismatch")
    if len(set(emitted_indices)) != len(emitted_indices):
        reasons.append("cue_index_duplicate")
    if set(emitted_indices) != set(source_indices):
        reasons.append("cue_index_mismatch")
    elif emitted_indices != source_indices:
        reasons.append("cue_order_mismatch")
    return SchemaValidationResult(valid=not reasons, reasons=_unique(reasons))


def _cue_duration_ms(cue: Cue) -> int:
    return max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1)


def _cue_gap_after_ms(cues: List[Cue], idx: int) -> int:
    if idx + 1 >= len(cues):
        return 0
    return max(ts_to_ms(cues[idx + 1].start) - ts_to_ms(cues[idx].end), 0)


def _expected_shares(block: TranslationBlock) -> List[float]:
    src_chars = [max(len(normalize_text(cue.text)), 1) for cue in block.cues]
    durations = [_cue_duration_ms(cue) for cue in block.cues]
    total_src = sum(src_chars) or 1
    total_dur = sum(durations) or 1
    shares = []
    for src, dur in zip(src_chars, durations):
        src_share = src / total_src
        dur_share = dur / total_dur
        shares.append(0.5 * src_share + 0.5 * dur_share)
    return shares


def _target_shares(emitted_cues: List[EmittedCue]) -> tuple[List[int], List[float]]:
    tgt_chars = [len(normalize_text(cue.text).replace(" ", "")) for cue in emitted_cues]
    total_tgt = sum(tgt_chars) or 1
    return tgt_chars, [chars / total_tgt for chars in tgt_chars]


def _hard_soft_glossary_reasons(source_text: str, output_text: str, glossary_terms: List[GlossaryEntry]) -> tuple[List[str], List[str]]:
    repair: List[str] = []
    warnings: List[str] = []
    src_casefold = source_text.casefold()
    out_casefold = output_text.casefold()
    for term in glossary_terms:
        if normalize_text(term.source).casefold() not in src_casefold:
            continue
        if normalize_text(term.target).casefold() in out_casefold:
            continue
        if term.mode == "hard":
            repair.append("glossary_violation")
        else:
            warnings.append("glossary_soft_miss")
    return repair, warnings


def _allowed_english_terms(glossary_terms: List[GlossaryEntry], config: RuntimeConfig) -> set[str]:
    allowed = set(config.allowed_english_terms)
    for term in glossary_terms:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9./+-]*", f"{term.source} {term.target}"):
            allowed.add(token.casefold())
    return allowed


def _is_technical_carry_through_span(span: str, allowed: set[str]) -> bool:
    span_casefold = span.casefold()
    if span_casefold in allowed:
        return True
    if _ACRONYM_RE.fullmatch(span):
        return True
    if span_casefold in _TECHNICAL_CARRY_THROUGH_HINTS:
        return True
    if _TECHNICAL_HYPHEN_DIGIT_RE.fullmatch(span):
        return True
    if _TECHNICAL_CAMEL_RE.fullmatch(span):
        return True
    return False


def _english_residual_reasons(output_text: str, glossary_terms: List[GlossaryEntry], config: RuntimeConfig) -> tuple[List[str], List[str]]:
    allowed = _allowed_english_terms(glossary_terms, config)
    spans = []
    technical_spans = []
    for match in _ENGLISH_SPAN_RE.finditer(output_text):
        span = match.group(0)
        if _is_technical_carry_through_span(span, allowed):
            if config.english_residual_policy == "technical_split" and span.casefold() not in allowed:
                technical_spans.append(span)
            continue
        spans.append(span)
    if not spans and technical_spans:
        return [], ["english_residual_technical"]
    if not spans:
        return [], []
    non_space_chars = len(output_text.replace(" ", "")) or 1
    total_span_chars = sum(len(span) for span in spans)
    if len(spans) >= 2 or (total_span_chars / non_space_chars) > 0.30:
        return ["english_residual"], []
    return [], ["english_residual_warn"]


def _english_residual_span_details(
    emitted_cues: List[EmittedCue],
    glossary_terms: List[GlossaryEntry],
    config: RuntimeConfig,
) -> tuple[List[dict], List[dict]]:
    allowed = _allowed_english_terms(glossary_terms, config)
    residual_details: List[dict] = []
    technical_details: List[dict] = []
    for cue in emitted_cues:
        cue_text = normalize_text(cue.text)
        for match in _ENGLISH_SPAN_RE.finditer(cue_text):
            span = match.group(0)
            detail = {
                "cue_index": cue.cue_index,
                "term": span,
                "normalized_term": span.casefold(),
                "span_text": span,
                "output_text": cue_text,
            }
            if _is_technical_carry_through_span(span, allowed):
                if config.english_residual_policy == "technical_split" and span.casefold() not in allowed:
                    technical_details.append({**detail, "issue": "english_residual_technical"})
                continue
            residual_details.append({**detail, "issue": "english_residual"})
    return residual_details, technical_details


def _normalize_numeric_token(token: str) -> str:
    return token.strip(" \t\r\n.,!?;:'\"-–—")


def _numeric_anchor_reasons(source_text: str, output_text: str) -> List[str]:
    source_tokens = {
        normalized
        for normalized in (_normalize_numeric_token(token) for token in _NUMERIC_TOKEN_RE.findall(source_text))
        if normalized
    }
    output_tokens = {
        normalized
        for normalized in (_normalize_numeric_token(token) for token in _NUMERIC_TOKEN_RE.findall(output_text))
        if normalized
    }
    if source_tokens and not source_tokens.issubset(output_tokens):
        return ["numeric_anchor_loss"]
    return []


def _delimiter_anchor_reasons(source_text: str, output_text: str) -> List[str]:
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if source_text.count(opening) != output_text.count(opening) or source_text.count(closing) != output_text.count(closing):
            return ["delimiter_anchor_loss"]
    return []


def _identifier_anchor_reasons(source_text: str, output_text: str) -> List[str]:
    source_identifiers = {token for token in _MODEL_TOKEN_RE.findall(source_text)}
    if not source_identifiers:
        return []
    output_casefold = output_text.casefold()
    for token in source_identifiers:
        if token.casefold() not in output_casefold:
            return ["identifier_anchor_loss"]
    return []


def _cue_cps(text: str, cue: Cue) -> float:
    visible_chars = len(normalize_text(text).replace(" ", ""))
    duration_ms = _cue_duration_ms(cue)
    return visible_chars / (duration_ms / 1000.0)


def _cps_thresholds(config: RuntimeConfig) -> tuple[float, float, float]:
    if config.wrap_policy == "cps_relaxed_v1":
        return (config.max_cps + 1.0, config.max_cps + 2.0, config.max_cps + 4.0)
    return (config.max_cps, config.max_cps + 2.0, config.max_cps + 4.0)


def _fragment_overclosure_details(block: TranslationBlock, tgt_texts: List[str]) -> List[dict]:
    if "carry_context_only" not in block.lint_actions and "dependent_end" not in block.lint_reasons:
        return []
    details: List[dict] = []
    for cue, text in zip(block.cues, tgt_texts):
        normalized = normalize_text(text)
        match = _EXPLANATORY_CLOSER_RE.search(normalized)
        if normalized and match:
            details.append(
                {
                    "cue_index": cue.index,
                    "span_text": match.group(0),
                    "issue": "unsupported_explanatory_tail",
                    "preferred_action": "trim_explanatory_tail",
                }
            )
        discourse_match = _DISCOURSE_MARKER_HEAD_RE.match(normalized)
        if normalized and discourse_match and "dependent_end" in block.lint_reasons:
            details.append(
                {
                    "cue_index": cue.index,
                    "span_text": discourse_match.group(0).strip(),
                    "issue": "unsupported_head_marker",
                    "preferred_action": "drop_head_marker",
                }
            )
    return details


def _split_output_clauses(output_text: str) -> List[str]:
    return [
        normalize_text(fragment)
        for fragment in re.split(r"[.!?]\s+|\n+", output_text)
        if normalize_text(fragment)
    ]


def _duplicate_restatement_warnings(source_text: str, output_text: str) -> List[str]:
    return ["duplicate_restatement"] if _duplicate_restatement_details(source_text, output_text, [], {}) else []


def _split_propositions(text: str) -> List[dict]:
    propositions: List[dict] = []
    last_end = 0
    for match in _PROPOSITION_SPLIT_RE.finditer(text):
        segment = normalize_text(text[last_end:match.start()])
        if segment:
            propositions.append(
                {
                    "text": segment,
                    "start": last_end,
                    "end": match.start(),
                }
            )
        last_end = match.end()
    tail = normalize_text(text[last_end:])
    if tail:
        propositions.append(
            {
                "text": tail,
                "start": last_end,
                "end": len(text),
            }
        )
    return propositions


def _looks_like_duplicate_proposition(left: str, right: str) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    right_wo_marker = normalize_text(_RESTATEMENT_MARKER_RE.sub("", right_norm))
    if left_norm == right_norm and len(left_norm) >= 6:
        return True
    if right_wo_marker and len(right_wo_marker) >= 6:
        if left_norm == right_wo_marker:
            return True
        if right_wo_marker in left_norm or left_norm in right_wo_marker:
            shorter = min(len(left_norm), len(right_wo_marker))
            longer = max(len(left_norm), len(right_wo_marker), 1)
            if shorter >= 8 and shorter / longer >= 0.70:
                return True
        if SequenceMatcher(a=left_norm, b=right_wo_marker).ratio() >= 0.70 and min(len(left_norm), len(right_wo_marker)) >= 8:
            return True
    if right_norm in left_norm or left_norm in right_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm), 1)
        if shorter >= 8 and shorter / longer >= 0.72:
            return True
    return SequenceMatcher(a=left_norm, b=right_norm).ratio() >= 0.78 and min(len(left_norm), len(right_norm)) >= 10


def _proposition_entries(emitted_cues: List[EmittedCue]) -> List[dict]:
    entries: List[dict] = []
    for cue in emitted_cues:
        cue_text = normalize_text(cue.text)
        for proposition in _split_propositions(cue_text):
            entries.append(
                {
                    "cue_index": cue.cue_index,
                    "text": proposition["text"],
                    "span_text": cue_text[proposition["start"] : proposition["end"]].strip(),
                    "start": proposition["start"],
                    "end": proposition["end"],
                }
            )
    return entries


def _source_tail_type(source_text: str) -> str | None:
    normalized = normalize_text(source_text)
    if not normalized:
        return None
    if _PURPOSE_TAIL_RE.match(normalized):
        return "purpose_tail"
    if _THAT_TAIL_RE.match(normalized):
        return "that_clause_tail"
    if _RELATIVE_TAIL_RE.match(normalized):
        return "relative_clause_tail"
    if _COMPARISON_TAIL_RE.match(normalized):
        return "comparison_tail"
    if _GENERIC_CONTINUATION_TAIL_RE.match(normalized):
        return "continuation_tail"
    return None


def _duplicate_restatement_details(
    source_text: str,
    output_text: str,
    emitted_cues: List[EmittedCue],
    source_cue_map: Dict[int, str],
) -> List[dict]:
    source_clause_count = len(_split_output_clauses(source_text))
    details: List[dict] = []

    proposition_entries = _proposition_entries(emitted_cues) if emitted_cues else []
    if not proposition_entries:
        proposition_entries = [
            {
                "cue_index": -1,
                "text": proposition["text"],
                "span_text": proposition["text"],
                "start": proposition["start"],
                "end": proposition["end"],
            }
            for proposition in _split_propositions(output_text)
        ]

    for left, right in zip(proposition_entries, proposition_entries[1:]):
        left_text = left["text"]
        right_text = right["text"]
        if len(left_text) < 6 or len(right_text) < 6:
            continue
        right_source_text = source_cue_map.get(right["cue_index"], "")
        source_tail_type = _source_tail_type(right_source_text)
        preferred_action = (
            "restore_missing_tail"
            if left["cue_index"] != right["cue_index"] and source_tail_type
            else "delete_repeat_local"
        )
        if (
            preferred_action == "restore_missing_tail"
            and right["cue_index"] >= 0
            and right_text
            and right_text in left_text
            and len(right_text) >= 8
        ):
            details.append(
                {
                    "cue_index": right["cue_index"],
                    "cue_indices": [left["cue_index"], right["cue_index"]],
                    "span_text": right["span_text"],
                    "left_text": left_text,
                    "right_text": right_text,
                    "source_cue_text": right_source_text,
                    "source_tail_type": source_tail_type,
                    "issue": "duplicate_restatement",
                    "preferred_action": preferred_action,
                }
            )
            continue
        if source_clause_count <= 1 and _RESTATEMENT_MARKER_RE.match(right_text):
            details.append(
                {
                    "cue_index": right["cue_index"],
                    "cue_indices": [left["cue_index"], right["cue_index"]],
                    "span_text": right["span_text"],
                    "left_text": left_text,
                    "right_text": right_text,
                    "source_cue_text": right_source_text,
                    "source_tail_type": source_tail_type,
                    "issue": "duplicate_restatement",
                    "preferred_action": preferred_action,
                }
            )
            continue
        if _looks_like_duplicate_proposition(left_text, right_text):
            details.append(
                {
                    "cue_index": right["cue_index"],
                    "cue_indices": [left["cue_index"], right["cue_index"]],
                    "span_text": right["span_text"],
                    "left_text": left_text,
                    "right_text": right_text,
                    "source_cue_text": right_source_text,
                    "source_tail_type": source_tail_type,
                    "issue": "duplicate_restatement",
                    "preferred_action": preferred_action,
                }
            )

    return details


def pre_wrap_gate(
    block: TranslationBlock,
    emitted_cues: List[EmittedCue],
    glossary_terms: List[GlossaryEntry],
    config: RuntimeConfig,
) -> QualityGateResult:
    repair: List[str] = []
    warnings: List[str] = list(block.lint_reasons)
    warning_details: Dict[str, List[dict]] = {}

    tgt_texts = [normalize_text(cue.text) for cue in emitted_cues]
    output_text = " ".join(text for text in tgt_texts if text)
    source_text = " ".join(normalize_text(cue.text) for cue in block.cues if normalize_text(cue.text))

    if not tgt_texts or not tgt_texts[0]:
        repair.append("first_cue_empty")

    expected = _expected_shares(block)
    tgt_chars, tgt_shares = _target_shares(emitted_cues)
    total_tgt = sum(tgt_chars)
    if len(block.cues) >= 3 and total_tgt >= 18:
        if tgt_chars[0] < 4 and tgt_shares[0] < 0.35 * expected[0]:
            repair.append("front_sparse")
        last_expected = expected[-1]
        last_share = tgt_shares[-1]
        first_two_sum = sum(tgt_chars[:2]) if len(tgt_chars) >= 2 else 0
        support = (
            _SENTENCE_END_RE.search(tgt_texts[-1]) is not None
            or any(_PARTICLE_ONLY_RE.fullmatch(text) for text in tgt_texts[:-1] if text)
            or _cue_cps(tgt_texts[-1], block.cues[-1]) > config.max_cps
        )
        if (
            last_share > max(0.50, 1.8 * last_expected)
            and tgt_chars[-1] > first_two_sum
            and support
        ):
            repair.append("tail_heavy")

    glossary_repair, glossary_warn = _hard_soft_glossary_reasons(source_text, output_text, glossary_terms)
    repair.extend(glossary_repair)
    warnings.extend(glossary_warn)
    repair.extend(_numeric_anchor_reasons(source_text, output_text))
    repair.extend(_delimiter_anchor_reasons(source_text, output_text))
    repair.extend(_identifier_anchor_reasons(source_text, output_text))
    english_repair, english_warn = _english_residual_reasons(output_text, glossary_terms, config)
    repair.extend(english_repair)
    warnings.extend(english_warn)
    residual_details, technical_details = _english_residual_span_details(emitted_cues, glossary_terms, config)
    if "english_residual" in english_repair and residual_details:
        warning_details["english_residual"] = [
            {**detail, "issue": "english_residual"} for detail in residual_details
        ]
    if "english_residual_warn" in english_warn and residual_details:
        warning_details["english_residual_warn"] = [
            {**detail, "issue": "english_residual_warn"} for detail in residual_details
        ]
    if "english_residual_technical" in english_warn and technical_details:
        warning_details["english_residual_technical"] = technical_details
    fragment_details = _fragment_overclosure_details(block, tgt_texts)
    if fragment_details:
        for issue in _unique(detail["issue"] for detail in fragment_details):
            warnings.append(issue)
            warning_details[issue] = [detail for detail in fragment_details if detail["issue"] == issue]
    duplicate_details = _duplicate_restatement_details(
        source_text,
        output_text,
        emitted_cues,
        {cue.index: cue.text for cue in block.cues},
    )
    if duplicate_details:
        warnings.append("duplicate_restatement")
        warning_details["duplicate_restatement"] = duplicate_details

    warn_threshold, repair_threshold, severe_threshold = _cps_thresholds(config)
    for source_cue, emitted in zip(block.cues, emitted_cues):
        cps = _cue_cps(emitted.text, source_cue)
        if cps > severe_threshold:
            repair.append("cps_overflow_severe")
        elif cps > repair_threshold:
            repair.append("cps_overflow")
        elif cps > warn_threshold:
            warnings.append("cps_warn")

    return QualityGateResult(
        repair_needed=bool(repair),
        repair_reasons=_unique(repair),
        warning_reasons=_unique(warnings),
        warning_details=warning_details,
    )


def _line_length_balance(lines: List[str]) -> bool:
    if len(lines) != 2:
        return True
    lengths = [len(line.strip()) for line in lines]
    shortest = min(lengths)
    longest = max(lengths) or 1
    return shortest >= max(3, int(longest * 0.25))


def post_wrap_gate(wrapped_cues: List[Cue], config: RuntimeConfig) -> QualityGateResult:
    repair: List[str] = []
    warnings: List[str] = []
    warn_threshold, repair_threshold, severe_threshold = _cps_thresholds(config)
    for cue in wrapped_cues:
        lines = cue.text.splitlines() if cue.text else []
        if len(lines) > config.max_lines_per_cue:
            repair.append("line_overflow")
        for line in lines:
            if len(line) > config.max_chars_per_line:
                repair.append("line_overflow")
            if _PARTICLE_ONLY_RE.fullmatch(normalize_text(line)):
                repair.append("bad_line_break")
        if lines and not _line_length_balance(lines):
            warnings.append("line_imbalance")
        cps = _cue_cps(cue.text, cue)
        if cps > severe_threshold:
            repair.append("cps_overflow_severe")
        elif cps > repair_threshold:
            repair.append("cps_overflow")
        elif cps > warn_threshold:
            warnings.append("cps_warn")
    return QualityGateResult(
        repair_needed=bool(repair),
        repair_reasons=_unique(repair),
        warning_reasons=_unique(warnings),
    )
