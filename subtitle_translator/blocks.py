from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import RuntimeConfig
from .grouping import group_cues_into_sentences
from .models import Cue, TranslationBlock
from .splitting import ts_to_ms
from .text import normalize_text


@dataclass
class _SentenceWindow:
    text: str
    start_pos: int
    end_pos: int


def _cue_gap_ms(left: Cue, right: Cue) -> int:
    return max(ts_to_ms(right.start) - ts_to_ms(left.end), 0)


def _block_duration_ms(cues: List[Cue]) -> int:
    if not cues:
        return 0
    return max(ts_to_ms(cues[-1].end) - ts_to_ms(cues[0].start), 0)


def _block_source_chars(cues: List[Cue]) -> int:
    return sum(len(normalize_text(cue.text)) for cue in cues)


def _ends_with_hard_punctuation(text: str) -> bool:
    return normalize_text(text).endswith((".", "!", "?", ".”", "!”", "?”"))


def _looks_incomplete(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if cleaned.endswith((".", "!", "?", ":", ";")):
        return False
    lowered = cleaned.casefold()
    dangling_suffixes = (
        "and",
        "or",
        "but",
        "so",
        "because",
        "that",
        "which",
        "who",
        "when",
        "if",
        "then",
        "than",
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
    )
    return lowered.endswith(dangling_suffixes) or cleaned.endswith((",", "-", "(", "[", "{", "/"))


def _is_short_supportive_cue(text: str) -> bool:
    cleaned = normalize_text(text)
    return 0 < len(cleaned) <= 40


def _allow_one_more_cue(block_cues: List[Cue], next_cue: Cue, gap_ms: int, config: RuntimeConfig) -> bool:
    joined = normalize_text(" ".join(cue.text for cue in block_cues))
    return (
        _looks_incomplete(joined)
        and gap_ms <= config.block_max_gap_ms
        and _is_short_supportive_cue(next_cue.text)
    )


def _sentence_windows(cues: List[Cue]) -> List[_SentenceWindow]:
    cue_pos = {cue.index: pos for pos, cue in enumerate(cues)}
    windows: List[_SentenceWindow] = []
    for sentence in group_cues_into_sentences(cues):
        positions = [cue_pos[idx] for idx in sentence.cue_indices if idx in cue_pos]
        if not positions:
            continue
        windows.append(
            _SentenceWindow(
                text=sentence.text,
                start_pos=min(positions),
                end_pos=max(positions),
            )
        )
    return windows


def _context_for_block(start_pos: int, end_pos: int, windows: List[_SentenceWindow], config: RuntimeConfig) -> tuple[List[str], List[str]]:
    if not config.use_context_window:
        return [], []
    previous = [window.text for window in windows if window.end_pos < start_pos]
    following = [window.text for window in windows if window.start_pos > end_pos]
    return previous[-2:], following[:1]


def build_translation_blocks(cues: List[Cue], config: RuntimeConfig) -> List[TranslationBlock]:
    if not cues:
        return []

    windows = _sentence_windows(cues)
    blocks: List[TranslationBlock] = []
    index = 0
    while index < len(cues):
        block_cues = [cues[index]]
        next_index = index
        remaining = len(cues) - index
        target_min_cues = min(config.block_min_cues, remaining)

        while next_index + 1 < len(cues) and len(block_cues) < config.block_max_cues:
            candidate = cues[next_index + 1]
            proposed = block_cues + [candidate]
            if _block_duration_ms(proposed) > config.block_max_duration_ms:
                break
            if _block_source_chars(proposed) > config.block_max_source_chars:
                break

            gap_ms = _cue_gap_ms(block_cues[-1], candidate)
            force_minimum = len(block_cues) < target_min_cues and gap_ms <= config.block_max_gap_ms
            if not force_minimum:
                joined = normalize_text(" ".join(cue.text for cue in block_cues))
                if _ends_with_hard_punctuation(joined) and not _allow_one_more_cue(block_cues, candidate, gap_ms, config):
                    break
                if gap_ms > config.block_max_gap_ms and not _allow_one_more_cue(block_cues, candidate, gap_ms, config):
                    break

            block_cues.append(candidate)
            next_index += 1

        previous_context, next_context = _context_for_block(index, next_index, windows, config)
        blocks.append(
            TranslationBlock(
                cues=block_cues,
                previous_source_sentences=previous_context,
                next_source_sentences=next_context,
            )
        )
        index = next_index + 1
    return blocks
