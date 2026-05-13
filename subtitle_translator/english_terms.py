from __future__ import annotations

import re
from typing import Iterable, Sequence

from .config import RuntimeConfig
from .models import Cue, GlossaryEntry
from .text import normalize_text


_ENGLISH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9./+-]{1,}")


def _contains_latin_term(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(normalized_term)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _approved_source_terms_in_text(text: str, config: RuntimeConfig) -> list[str]:
    approved = set(config.allowed_english_terms)
    terms: list[str] = []
    for match in _ENGLISH_TERM_RE.finditer(text):
        term = match.group(0)
        if term.casefold() in approved and term not in terms:
            terms.append(term)
    for rule in config.english_fallback_terms:
        source = normalize_text(str(rule.get("source", "")))
        if source and _contains_latin_term(text, source) and source not in terms:
            terms.append(source)
    return terms


def approved_english_glossary_terms_for_block(block_cues: Sequence[Cue], config: RuntimeConfig) -> list[GlossaryEntry]:
    source_text = " ".join(normalize_text(cue.text) for cue in block_cues)
    return [
        GlossaryEntry(
            source=term,
            target=term,
            note="Approved English carry-through term; keep this term in English.",
            mode="hard",
        )
        for term in _approved_source_terms_in_text(source_text, config)
    ]


def merge_glossary_terms(base_terms: Iterable[GlossaryEntry], approved_terms: Iterable[GlossaryEntry]) -> list[GlossaryEntry]:
    merged: dict[str, GlossaryEntry] = {}
    for term in base_terms:
        merged[normalize_text(term.source).casefold()] = term
    for term in approved_terms:
        merged[normalize_text(term.source).casefold()] = term
    return list(merged.values())


def apply_approved_english_fallbacks(
    source_cues: Sequence[Cue],
    output_cues: Sequence[Cue],
    config: RuntimeConfig,
) -> tuple[list[Cue], list[dict]]:
    if not config.english_fallback_terms:
        return list(output_cues), []

    source_by_index = {cue.index: cue for cue in source_cues}
    rewritten: list[Cue] = []
    replacements: list[dict] = []
    for cue in output_cues:
        source_cue = source_by_index.get(cue.index)
        if not source_cue:
            rewritten.append(cue)
            continue
        source_text = normalize_text(source_cue.text)
        target_text = cue.text
        for rule in config.english_fallback_terms:
            source_term = normalize_text(str(rule.get("source", "")))
            aliases = [normalize_text(str(alias)) for alias in rule.get("aliases", []) if normalize_text(str(alias))]
            if not source_term or not aliases:
                continue
            if not _contains_latin_term(source_text, source_term):
                continue
            if _contains_latin_term(target_text, source_term):
                continue
            for alias in aliases:
                if alias not in normalize_text(target_text):
                    continue
                before = target_text
                target_text = target_text.replace(alias, source_term)
                if target_text != before:
                    replacements.append(
                        {
                            "cue_index": cue.index,
                            "source": source_term,
                            "alias": alias,
                            "before": before,
                            "after": target_text,
                        }
                    )
                    break
        rewritten.append(Cue(index=cue.index, start=cue.start, end=cue.end, text=target_text))
    return rewritten, replacements
