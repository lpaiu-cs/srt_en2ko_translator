from __future__ import annotations

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
        boundaries.append(cut)
    return boundaries


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
