from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Iterable, List

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
_EXPLANATORY_CLOSER_RE = re.compile(r"(라는 뜻입니다|거죠|겁니다|라고 볼 수 있습니다|라고 할 수 있습니다)$")
_RESTATEMENT_MARKER_RE = re.compile(r"^(즉|다시 말해|다시 말하면|말하자면|바로)\b")


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


def _english_residual_reasons(output_text: str, glossary_terms: List[GlossaryEntry], config: RuntimeConfig) -> tuple[List[str], List[str]]:
    allowed = _allowed_english_terms(glossary_terms, config)
    spans = []
    for match in _ENGLISH_SPAN_RE.finditer(output_text):
        span = match.group(0)
        if span.casefold() in allowed:
            continue
        if _ACRONYM_RE.fullmatch(span):
            continue
        spans.append(span)
    if not spans:
        return [], []
    non_space_chars = len(output_text.replace(" ", "")) or 1
    total_span_chars = sum(len(span) for span in spans)
    if len(spans) >= 2 or (total_span_chars / non_space_chars) > 0.30:
        return ["english_residual"], []
    return [], ["english_residual_warn"]


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


def _fragment_overclosure_warnings(block: TranslationBlock, tgt_texts: List[str]) -> List[str]:
    if "carry_context_only" not in block.lint_actions and "dependent_end" not in block.lint_reasons:
        return []
    for text in tgt_texts:
        normalized = normalize_text(text)
        if normalized and _EXPLANATORY_CLOSER_RE.search(normalized):
            return ["fragment_overclosure"]
    return []


def _split_output_clauses(output_text: str) -> List[str]:
    return [
        normalize_text(fragment)
        for fragment in re.split(r"[.!?]\s+|\n+", output_text)
        if normalize_text(fragment)
    ]


def _duplicate_restatement_warnings(source_text: str, output_text: str) -> List[str]:
    clauses = _split_output_clauses(output_text)
    if len(clauses) < 2:
        return []
    if len(_split_output_clauses(source_text)) <= 1:
        for clause in clauses[1:]:
            if _RESTATEMENT_MARKER_RE.match(clause):
                return ["duplicate_restatement"]
    for left, right in zip(clauses, clauses[1:]):
        if len(left) < 10 or len(right) < 10:
            continue
        if _RESTATEMENT_MARKER_RE.match(right):
            return ["duplicate_restatement"]
        if SequenceMatcher(a=left, b=right).ratio() >= 0.74:
            return ["duplicate_restatement"]
    return []


def pre_wrap_gate(
    block: TranslationBlock,
    emitted_cues: List[EmittedCue],
    glossary_terms: List[GlossaryEntry],
    config: RuntimeConfig,
) -> QualityGateResult:
    repair: List[str] = []
    warnings: List[str] = list(block.lint_reasons)

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
    warnings.extend(_fragment_overclosure_warnings(block, tgt_texts))
    warnings.extend(_duplicate_restatement_warnings(source_text, output_text))

    for source_cue, emitted in zip(block.cues, emitted_cues):
        cps = _cue_cps(emitted.text, source_cue)
        if cps > config.max_cps + 4.0:
            repair.append("cps_overflow_severe")
        elif cps > config.max_cps + 2.0:
            repair.append("cps_overflow")
        elif cps > config.max_cps:
            warnings.append("cps_warn")

    return QualityGateResult(
        repair_needed=bool(repair),
        repair_reasons=_unique(repair),
        warning_reasons=_unique(warnings),
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
        if cps > config.max_cps + 4.0:
            repair.append("cps_overflow_severe")
        elif cps > config.max_cps + 2.0:
            repair.append("cps_overflow")
        elif cps > config.max_cps:
            warnings.append("cps_warn")
    return QualityGateResult(
        repair_needed=bool(repair),
        repair_reasons=_unique(repair),
        warning_reasons=_unique(warnings),
    )
