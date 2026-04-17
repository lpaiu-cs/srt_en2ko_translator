from __future__ import annotations

import re
from typing import Dict, List

from .models import Cue, SentenceGroup
from .text import normalize_text, wrap_lines


SRT_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
LEADING_PUNCT = "、，,。．.?!！？；;:·•"


def ts_to_ms(ts: str) -> int:
    match = SRT_TS_RE.match(ts)
    if not match:
        return 0
    hh, mm, ss, ms = map(int, match.groups())
    return (((hh * 60) + mm) * 60 + ss) * 1000 + ms


def _is_numeric_separator(text: str, idx: int) -> bool:
    if idx <= 0 or idx + 1 >= len(text):
        return False
    return text[idx] in {",", "."} and text[idx - 1].isdigit() and text[idx + 1].isdigit()


def _candidate_scores(text: str) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    idx = 0
    while idx < len(text) - 1:
        ch = text[idx]
        position = None
        score = None
        if ch.isspace():
            position = idx + 1
            score = 1.6
        elif ch in ".!?":
            if not _is_numeric_separator(text, idx):
                position = idx + 1
                score = 6.0
        elif ch in ",;:·•":
            if not _is_numeric_separator(text, idx):
                position = idx + 1
                score = 3.0 if ch != "," else 2.2
        if position is not None and score is not None:
            while position < len(text) and text[position].isspace():
                score += 0.1
                position += 1
            if 0 < position < len(text):
                scores[position] = max(scores.get(position, float("-inf")), score)
        idx += 1
    return scores


def _best_cut(start: int, target: int, text: str, candidate_scores: Dict[int, float]) -> int:
    best_position = None
    best_score = float("-inf")
    for position, weight in candidate_scores.items():
        if position <= start or position >= len(text):
            continue
        score = weight - (abs(position - target) * 0.25)
        if position - start <= 1:
            score -= 1.5
        if text[position] in LEADING_PUNCT:
            score -= 2.5
        if score > best_score:
            best_position = position
            best_score = score
    if best_position is None or best_position <= start:
        return min(max(start + 1, target), len(text) - 1)
    return best_position


def proportional_split_by_time(text: str, durations_ms: List[int]) -> List[str]:
    text = normalize_text(text)
    slots = len(durations_ms)
    if slots <= 1:
        return [text]
    if not text:
        return [""] * slots

    weights = [max(int(duration), 1) for duration in durations_ms]
    total_weight = sum(weights)
    length = len(text)

    cumulative = 0
    targets = []
    for weight in weights[:-1]:
        cumulative += weight
        targets.append(round(length * (cumulative / total_weight)))

    candidate_scores = _candidate_scores(text)
    parts: List[str] = []
    start = 0
    for target in targets:
        cut = _best_cut(start, target, text, candidate_scores)
        parts.append(text[start:cut].strip())
        start = cut
    parts.append(text[start:].strip())

    for idx in range(1, len(parts)):
        match = re.match(rf"^[{re.escape(LEADING_PUNCT)}]+", parts[idx])
        if match and parts[idx - 1]:
            leading = match.group(0)
            parts[idx - 1] = (parts[idx - 1] + leading).rstrip()
            parts[idx] = parts[idx][len(leading):].lstrip()
    return parts


def groups_to_cues_by_time(
    translated_groups: List[SentenceGroup],
    cues: List[Cue],
    repeat_fill: bool = False,
    width: int = 42,
) -> List[Cue]:
    idx_to_cue: Dict[int, Cue] = {cue.index: cue for cue in cues}
    idx_to_fragments: Dict[int, List[str]] = {cue.index: [] for cue in cues}

    for group in translated_groups:
        if not group.cue_indices:
            continue
        durations = [
            max(ts_to_ms(idx_to_cue[idx].end) - ts_to_ms(idx_to_cue[idx].start), 1)
            if idx in idx_to_cue
            else 1
            for idx in group.cue_indices
        ]
        parts = proportional_split_by_time(group.text, durations)
        if repeat_fill and len(parts) > 1:
            last_nonempty = None
            for idx, part in enumerate(parts):
                if part.strip():
                    last_nonempty = part
                elif last_nonempty is not None:
                    parts[idx] = last_nonempty
            first_nonempty = next((part for part in parts if part.strip()), "")
            if first_nonempty:
                for idx, part in enumerate(parts):
                    if part.strip():
                        break
                    parts[idx] = first_nonempty
        for cue_index, fragment in zip(group.cue_indices, parts):
            idx_to_fragments[cue_index].append(fragment)

    output: List[Cue] = []
    for cue in cues:
        fragments = [normalize_text(fragment) for fragment in idx_to_fragments.get(cue.index, []) if fragment.strip()]
        output.append(
            Cue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                text=wrap_lines(" ".join(fragments), width=width),
            )
        )
    return output
