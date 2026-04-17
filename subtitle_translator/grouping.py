from __future__ import annotations

import re
from typing import List

from .models import Cue, SentenceGroup
from .text import normalize_text, warn


ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
}

CONNECTOR_PREFIXES = (
    "and",
    "but",
    "because",
    "so",
    "or",
    "then",
    "which",
    "that",
    "who",
    "when",
    "while",
    "if",
    "though",
    "although",
    "where",
    "as",
)

FILLER_PREFIXES = (
    "well",
    "you know",
    "i mean",
    "okay",
    "ok",
    "all right",
    "right",
)

INCOMPLETE_SUFFIXES = (
    "and",
    "or",
    "but",
    "so",
    "because",
    "that",
    "which",
    "who",
    "when",
    "while",
    "if",
    "to",
    "of",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "like",
    "such as",
    "kind of",
    "sort of",
)

_WORD_RE = re.compile(r"[A-Za-z']+")


def _next_non_space(text: str, start: int) -> str:
    for idx in range(start, len(text)):
        if not text[idx].isspace():
            return text[idx]
    return ""


def _previous_token(text: str, idx: int) -> str:
    start = idx
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".'"):
        start -= 1
    return text[start:idx].strip(" .'\"").casefold()


def _sentence_boundary_positions(text: str) -> List[int]:
    boundaries: List[int] = []
    for idx, ch in enumerate(text):
        if ch not in ".!?":
            continue
        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if ch == "." and prev_char.isdigit() and next_char.isdigit():
            continue
        if ch == "." and _previous_token(text, idx) in ABBREVIATIONS:
            continue
        next_visible = _next_non_space(text, idx + 1)
        if next_visible and next_visible.islower() and ch == ".":
            continue
        cut = idx + 1
        while cut < len(text) and text[cut].isspace():
            cut += 1
        if _should_hold_boundary(text, cut):
            continue
        boundaries.append(cut)
    return boundaries


def _normalize_ascii(text: str) -> str:
    return normalize_text(text).casefold()


def _has_unclosed_delimiter(segment: str) -> bool:
    if segment.count("(") > segment.count(")"):
        return True
    if segment.count("[") > segment.count("]"):
        return True
    if segment.count("{") > segment.count("}"):
        return True
    quote_count = segment.count('"') + segment.count("“") + segment.count("”")
    return quote_count % 2 == 1


def _starts_with_prefix(fragment: str, prefixes: tuple[str, ...]) -> bool:
    lowered = _normalize_ascii(fragment)
    return any(lowered.startswith(prefix + " ") or lowered == prefix for prefix in prefixes)


def _ends_with_incomplete_suffix(segment: str) -> bool:
    lowered = _normalize_ascii(segment).rstrip(" .!?,'\"")
    return any(lowered.endswith(" " + suffix) or lowered == suffix for suffix in INCOMPLETE_SUFFIXES)


def _looks_filler_sentence(segment: str) -> bool:
    lowered = _normalize_ascii(segment).rstrip(" .!?,'\"")
    if _starts_with_prefix(lowered, FILLER_PREFIXES):
        words = _WORD_RE.findall(lowered)
        return len(words) <= 4
    return False


def _should_hold_boundary(text: str, cut: int) -> bool:
    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        return False
    if _has_unclosed_delimiter(left):
        return True
    if _ends_with_incomplete_suffix(left):
        return True
    if _starts_with_prefix(right, CONNECTOR_PREFIXES):
        return True
    if _starts_with_prefix(right, FILLER_PREFIXES):
        return True
    if _looks_filler_sentence(left):
        return True
    return False


def group_cues_into_sentences(cues: List[Cue]) -> List[SentenceGroup]:
    groups: List[SentenceGroup] = []
    buffer_text: List[str] = []
    buffer_indices: List[int] = []

    for cue in cues:
        piece = normalize_text(cue.text)
        if not piece:
            groups.append(SentenceGroup(text="", cue_indices=[cue.index]))
            warn(f"empty cue at {cue.start} --> {cue.end}")
            continue
        buffer_text.append(piece)
        buffer_indices.append(cue.index)
        joined = " ".join(buffer_text)
        while True:
            boundaries = _sentence_boundary_positions(joined)
            if not boundaries:
                break
            cut_pos = boundaries[0]
            sentence = joined[:cut_pos].strip()
            if sentence:
                groups.append(SentenceGroup(text=sentence, cue_indices=buffer_indices.copy()))
            tail = joined[cut_pos:].strip()
            buffer_text = [tail] if tail else []
            buffer_indices = [buffer_indices[-1] if buffer_indices else cue.index] if tail else []
            joined = " ".join(buffer_text)

    if buffer_text:
        residual = " ".join(buffer_text).strip()
        if residual:
            fallback_index = cues[-1].index if cues else 1
            groups.append(
                SentenceGroup(
                    text=residual,
                    cue_indices=buffer_indices.copy() if buffer_indices else [fallback_index],
                )
            )
    return groups
