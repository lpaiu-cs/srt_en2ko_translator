from __future__ import annotations

import re
from typing import Iterable, List

from .config import RuntimeConfig
from .models import EmittedCue, GlossaryEntry, QualityGateResult, TranslationBlock
from .splitting import ts_to_ms
from .text import normalize_text, wrap_lines


_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[./+-][A-Za-z0-9]+)*")
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,./%+-]*")
_MEANINGFUL_TOKEN_RE = re.compile(r"[A-Za-z]+|[가-힣]+")
_PARTICLE_ONLY_RE = re.compile(
    r"^(은|는|이|가|을|를|에|의|도|만|와|과|로|으로|에서|부터|까지|보다|처럼|하고|및|또는|그리고|하지만|입니다|합니다|죠|겁니다)$"
)


def _unique_reasons(reasons: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(reason for reason in reasons if reason))


def _joined_text(cues: Iterable[EmittedCue]) -> str:
    return " ".join(normalize_text(cue.text) for cue in cues if normalize_text(cue.text))


def _required_glossary_violations(source_text: str, output_text: str, glossary_terms: List[GlossaryEntry]) -> bool:
    source_casefold = source_text.casefold()
    output_casefold = output_text.casefold()
    for term in glossary_terms:
        if normalize_text(term.source).casefold() not in source_casefold:
            continue
        if normalize_text(term.target).casefold() not in output_casefold:
            return True
    return False


def _allowed_english_tokens(glossary_terms: List[GlossaryEntry]) -> set[str]:
    allowed = set()
    for term in glossary_terms:
        for token in _ENGLISH_TOKEN_RE.findall(f"{term.source} {term.target}"):
            allowed.add(token.casefold())
    return allowed


def _english_residual_too_high(output_text: str, glossary_terms: List[GlossaryEntry]) -> bool:
    allowed = _allowed_english_tokens(glossary_terms)
    english_tokens = []
    for token in _ENGLISH_TOKEN_RE.findall(output_text):
        if token.casefold() in allowed:
            continue
        if token.isupper() and len(token) <= 5:
            continue
        english_tokens.append(token)
    meaningful_tokens = [token for token in _MEANINGFUL_TOKEN_RE.findall(output_text) if not token.isdigit()]
    if len(english_tokens) < 3:
        return False
    if not meaningful_tokens:
        return False
    return (len(english_tokens) / len(meaningful_tokens)) > 0.35


def _missing_numeric_or_bracket_content(source_text: str, output_text: str) -> bool:
    for token in set(_NUMERIC_TOKEN_RE.findall(source_text)):
        if token not in output_text:
            return True
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if source_text.count(opening) != output_text.count(opening):
            return True
        if source_text.count(closing) != output_text.count(closing):
            return True
    return False


def _cps(text: str, start_ts: str, end_ts: str) -> float:
    duration_ms = max(ts_to_ms(end_ts) - ts_to_ms(start_ts), 1)
    visible_chars = len(normalize_text(text).replace(" ", ""))
    return visible_chars / (duration_ms / 1000.0)


def evaluate_quality(
    block: TranslationBlock,
    emitted_cues: List[EmittedCue],
    glossary_terms: List[GlossaryEntry],
    config: RuntimeConfig,
) -> QualityGateResult:
    reasons: List[str] = []
    source_indices = [cue.index for cue in block.cues]
    emitted_indices = [cue.cue_index for cue in emitted_cues]

    if len(emitted_cues) != len(block.cues):
        reasons.append("cue_count_mismatch")
    if emitted_indices != source_indices:
        reasons.append("cue_index_mismatch")

    emitted_texts = [normalize_text(cue.text) for cue in emitted_cues]
    output_text = " ".join(text for text in emitted_texts if text)
    source_text = " ".join(normalize_text(cue.text) for cue in block.cues if normalize_text(cue.text))

    if not emitted_texts or not emitted_texts[0]:
        reasons.append("first_cue_empty")
    if any(not text for text in emitted_texts):
        reasons.append("empty_cue")

    total_chars = sum(len(text) for text in emitted_texts)
    if total_chars >= 20 and emitted_texts:
        if len(emitted_texts[0]) < max(1, int(total_chars * 0.05)):
            reasons.append("front_sparse")
        if len(emitted_texts[-1]) >= int(total_chars * 0.55):
            reasons.append("tail_heavy")
        if len(emitted_texts) >= 3 and len(emitted_texts[-1]) > (len(emitted_texts[0]) + len(emitted_texts[1])):
            reasons.append("tail_heavy")

    if _required_glossary_violations(source_text, output_text, glossary_terms):
        reasons.append("glossary_violation")
    if _english_residual_too_high(output_text, glossary_terms):
        reasons.append("english_residual")
    if _missing_numeric_or_bracket_content(source_text, output_text):
        reasons.append("anchor_loss")

    for source_cue, emitted in zip(block.cues, emitted_cues):
        wrapped = wrap_lines(normalize_text(emitted.text), width=config.max_chars_per_line)
        lines = wrapped.splitlines() if wrapped else []
        if len(lines) > config.max_lines_per_cue:
            reasons.append("line_overflow")
        for line in lines:
            if len(line) > config.max_chars_per_line:
                reasons.append("line_overflow")
            if _PARTICLE_ONLY_RE.fullmatch(normalize_text(line)):
                reasons.append("bad_line_break")
        if _cps(emitted.text, source_cue.start, source_cue.end) > config.max_cps:
            reasons.append("cps_overflow")

    return QualityGateResult(passed=not reasons, reasons=_unique_reasons(reasons))
