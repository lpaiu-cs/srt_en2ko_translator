from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .config import RuntimeConfig
from .grouping import group_cues_into_sentences
from .models import Cue, TranslationBlock
from .splitting import ts_to_ms
from .text import normalize_text


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_NUMERIC_RE = re.compile(r"\d[\d,./%+-]*")

DANGLING_SUFFIXES = (
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
)

DEPENDENCY_STARTERS = (
    "of",
    "a",
    "an",
    "the",
    "that",
    "which",
    "because",
    "if",
    "when",
    "while",
    "who",
    "whose",
    "where",
    "as",
)

COMPARISON_STARTERS = (
    "like",
    "than",
    "rather than",
    "as compared to",
    "compared to",
    "similar to",
)

BACKCHANNEL_STARTERS = (
    "yeah",
    "yep",
    "right",
    "okay",
    "ok",
    "sure",
    "uh-huh",
    "mm-hmm",
)

COMMON_VERBS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "takes",
    "take",
    "using",
    "use",
    "called",
    "look",
    "looks",
    "get",
    "gets",
    "compute",
    "compute",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
}


@dataclass
class _SentenceWindow:
    text: str
    start_pos: int
    end_pos: int


@dataclass
class _BlockDraft:
    cues: List[Cue]
    low_confidence: bool = False
    lint_reasons: List[str] = field(default_factory=list)
    lint_actions: List[str] = field(default_factory=list)


def _cue_gap_ms(left: Cue, right: Cue) -> int:
    return max(ts_to_ms(right.start) - ts_to_ms(left.end), 0)


def _block_duration_ms(cues: List[Cue]) -> int:
    if not cues:
        return 0
    return max(ts_to_ms(cues[-1].end) - ts_to_ms(cues[0].start), 0)


def _block_source_chars(cues: List[Cue]) -> int:
    return sum(len(normalize_text(cue.text)) for cue in cues)


def _block_text(cues: List[Cue]) -> str:
    return normalize_text(" ".join(cue.text for cue in cues))


def _ends_with_hard_punctuation(text: str) -> bool:
    return normalize_text(text).endswith((".", "!", "?", ".”", "!”", "?”"))


def _looks_incomplete(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if cleaned.endswith((".", "!", "?", ":", ";")):
        return False
    lowered = cleaned.casefold().rstrip(" .!?,'\"")
    if any(lowered.endswith(" " + suffix) or lowered == suffix for suffix in DANGLING_SUFFIXES):
        return True
    if cleaned.endswith((",", "-", "(", "[", "{", "/")):
        return True
    words = [word.casefold() for word in _WORD_RE.findall(cleaned)]
    if not words:
        return True
    if _starts_comparison_midthought(cleaned):
        return True
    if any(word in {"if", "when", "while", "because", "that", "which", "where", "whenever", "than"} for word in words[:-1]):
        return True
    last_token = cleaned.split()[-1].strip(".,!?;:'\"")
    if _NUMERIC_RE.fullmatch(last_token) is not None:
        return True
    return False


def _starts_comparison_midthought(text: str) -> bool:
    cleaned = normalize_text(text).casefold()
    if not cleaned:
        return False
    return any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in COMPARISON_STARTERS)


def _starts_dependency_heavy(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    words = _WORD_RE.findall(cleaned)
    if not words:
        return False
    first = words[0].casefold()
    if first in {"a", "an", "the"}:
        early = [word.casefold() for word in words[:8]]
        if not any(word in COMMON_VERBS for word in early):
            return True
        return False
    if first in DEPENDENCY_STARTERS:
        return True
    if cleaned[0].islower() and first not in {"i"}:
        return True
    return False


def _has_unbalanced_delimiters(text: str) -> bool:
    if text.count("(") != text.count(")"):
        return True
    if text.count("[") != text.count("]"):
        return True
    if text.count("{") != text.count("}"):
        return True
    quote_count = text.count('"') + text.count("“") + text.count("”")
    return quote_count % 2 == 1


def _is_orphan_numeric_acronym_fragment(cues: List[Cue]) -> bool:
    text = _block_text(cues)
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    if not text or len(cues) > 2 or len(text) > 55:
        return False
    if not (_NUMERIC_RE.search(text) or _ACRONYM_RE.search(text)):
        return False
    if any(word in COMMON_VERBS for word in words):
        return False
    return _starts_dependency_heavy(text) or len(words) <= 6


def _is_short_qa_fragment(cues: List[Cue]) -> bool:
    text = _block_text(cues)
    lowered = text.casefold()
    if len(text) > 24:
        return False
    if "?" in text:
        return True
    return any(lowered == starter or lowered.startswith(starter + " ") for starter in BACKCHANNEL_STARTERS)


def _is_short_supportive_cue(text: str) -> bool:
    cleaned = normalize_text(text)
    return 0 < len(cleaned) <= 40


def _allow_one_more_cue(block_cues: List[Cue], next_cue: Cue, gap_ms: int, config: RuntimeConfig) -> bool:
    joined = _block_text(block_cues)
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


def hydrate_translation_block(
    block_cues: List[Cue],
    all_cues: List[Cue],
    config: RuntimeConfig,
    *,
    low_confidence: bool = False,
    lint_reasons: List[str] | None = None,
    lint_actions: List[str] | None = None,
) -> TranslationBlock:
    if not block_cues:
        return TranslationBlock(
            cues=[],
            low_confidence=low_confidence,
            lint_reasons=lint_reasons or [],
            lint_actions=lint_actions or [],
        )
    windows = _sentence_windows(all_cues or block_cues)
    cue_pos = {cue.index: pos for pos, cue in enumerate(all_cues or block_cues)}
    positions = [cue_pos[cue.index] for cue in block_cues if cue.index in cue_pos]
    start_pos = min(positions) if positions else 0
    end_pos = max(positions) if positions else len(block_cues) - 1
    previous_context, next_context = _context_for_block(start_pos, end_pos, windows, config)
    return TranslationBlock(
        cues=block_cues,
        previous_source_sentences=previous_context,
        next_source_sentences=next_context,
        low_confidence=low_confidence,
        lint_reasons=lint_reasons or [],
        lint_actions=lint_actions or [],
    )


def _draft_reasons(cues: List[Cue]) -> List[str]:
    text = _block_text(cues)
    reasons: List[str] = []
    if _looks_incomplete(text):
        reasons.append("dependent_end")
    if _starts_comparison_midthought(text):
        reasons.append("dependent_start")
        reasons.append("comparison_midstart")
    elif _starts_dependency_heavy(text):
        reasons.append("dependent_start")
    if _has_unbalanced_delimiters(text):
        reasons.append("unbalanced_delimiter")
    if _is_orphan_numeric_acronym_fragment(cues):
        reasons.append("numeric_orphan")
    if _is_short_qa_fragment(cues):
        reasons.append("qa_fragment")
    return list(dict.fromkeys(reasons))


def _draft_actions(cues: List[Cue], reasons: List[str]) -> List[str]:
    if not reasons:
        return []
    prompt_worthy_fragment_reasons = {
        "dependent_end",
        "numeric_orphan",
        "comparison_midstart",
        "qa_fragment",
    }
    if any(reason in prompt_worthy_fragment_reasons for reason in reasons):
        return ["carry_context_only"]
    return []


def _pair_score(left: List[Cue], right: List[Cue]) -> tuple[int, int]:
    left_reasons = _draft_reasons(left)
    right_reasons = _draft_reasons(right)
    return (len(left_reasons) + len(right_reasons), len(left) + len(right))


def _combined_feasible(cues: List[Cue], config: RuntimeConfig) -> bool:
    return (
        1 <= len(cues) <= config.block_max_cues
        and _block_duration_ms(cues) <= config.block_max_duration_ms
        and _block_source_chars(cues) <= config.block_max_source_chars
    )


def _speaker_switch_risk(left: List[Cue], right: List[Cue]) -> bool:
    left_text = _block_text(left)
    right_text = _block_text(right).casefold()
    if "?" not in left_text:
        return False
    if len(right) == 1 and any(right_text == starter or right_text.startswith(starter + " ") for starter in BACKCHANNEL_STARTERS):
        return True
    if len(left_text) <= 20 and len(right_text) <= 32:
        return True
    return False


def _repair_block_boundaries(drafts: List[_BlockDraft], config: RuntimeConfig) -> List[_BlockDraft]:
    if len(drafts) < 2:
        for draft in drafts:
            draft.lint_reasons = _draft_reasons(draft.cues)
            draft.lint_actions = _draft_actions(draft.cues, draft.lint_reasons)
            draft.low_confidence = bool(draft.lint_reasons)
        return drafts

    index = 0
    while index < len(drafts) - 1:
        left = drafts[index]
        right = drafts[index + 1]
        current_score = _pair_score(left.cues, right.cues)
        best_action = None
        best_pair: tuple[List[Cue], List[Cue]] | None = None
        best_score = current_score

        pair_reasons = set(_draft_reasons(left.cues) + _draft_reasons(right.cues))
        actionable = pair_reasons & {
            "dependent_end",
            "dependent_start",
            "unbalanced_delimiter",
            "numeric_orphan",
            "comparison_midstart",
            "qa_fragment",
        }
        if not actionable:
            index += 1
            continue

        merged = left.cues + right.cues
        if _combined_feasible(merged, config) and not _speaker_switch_risk(left.cues, right.cues):
            merge_score = (len(_draft_reasons(merged)), len(merged))
            if merge_score < best_score:
                best_action = "merge"
                best_pair = (merged, [])
                best_score = merge_score

        if len(left.cues) > 1:
            shifted_right_left = left.cues[:-1]
            shifted_right_right = [left.cues[-1]] + right.cues
            if _combined_feasible(shifted_right_left, config) and _combined_feasible(shifted_right_right, config):
                shift_score = _pair_score(shifted_right_left, shifted_right_right)
                if shift_score < best_score:
                    best_action = "shift_right"
                    best_pair = (shifted_right_left, shifted_right_right)
                    best_score = shift_score

        if len(right.cues) > 1 and not _speaker_switch_risk(left.cues, right.cues):
            shifted_left_left = left.cues + [right.cues[0]]
            shifted_left_right = right.cues[1:]
            if _combined_feasible(shifted_left_left, config) and _combined_feasible(shifted_left_right, config):
                shift_score = _pair_score(shifted_left_left, shifted_left_right)
                if shift_score < best_score:
                    best_action = "shift_left"
                    best_pair = (shifted_left_left, shifted_left_right)
                    best_score = shift_score

        if best_action == "merge" and best_pair is not None:
            drafts[index] = _BlockDraft(cues=best_pair[0])
            del drafts[index + 1]
            if index > 0:
                index -= 1
            continue

        if best_action in {"shift_right", "shift_left"} and best_pair is not None:
            drafts[index] = _BlockDraft(cues=best_pair[0])
            drafts[index + 1] = _BlockDraft(cues=best_pair[1])
            if index > 0:
                index -= 1
            continue

        left.lint_reasons = _draft_reasons(left.cues)
        right.lint_reasons = _draft_reasons(right.cues)
        left.lint_actions = _draft_actions(left.cues, left.lint_reasons)
        right.lint_actions = _draft_actions(right.cues, right.lint_reasons)
        left.low_confidence = bool(left.lint_reasons)
        right.low_confidence = bool(right.lint_reasons)
        index += 1

    for draft in drafts:
        draft.lint_reasons = _draft_reasons(draft.cues)
        draft.lint_actions = _draft_actions(draft.cues, draft.lint_reasons)
        draft.low_confidence = bool(draft.lint_reasons)
    return drafts


def build_translation_blocks(cues: List[Cue], config: RuntimeConfig) -> List[TranslationBlock]:
    if not cues:
        return []

    windows = _sentence_windows(cues)
    drafts: List[_BlockDraft] = []
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
                joined = _block_text(block_cues)
                if _ends_with_hard_punctuation(joined) and not _allow_one_more_cue(block_cues, candidate, gap_ms, config):
                    break
                if gap_ms > config.block_max_gap_ms and not _allow_one_more_cue(block_cues, candidate, gap_ms, config):
                    break

            block_cues.append(candidate)
            next_index += 1

        drafts.append(_BlockDraft(cues=block_cues))
        index = next_index + 1

    repaired_drafts = _repair_block_boundaries(drafts, config)

    blocks: List[TranslationBlock] = []
    cue_pos = {cue.index: pos for pos, cue in enumerate(cues)}
    for draft in repaired_drafts:
        positions = [cue_pos[cue.index] for cue in draft.cues if cue.index in cue_pos]
        if not positions:
            continue
        blocks.append(
            hydrate_translation_block(
                draft.cues,
                cues,
                config,
                low_confidence=draft.low_confidence,
                lint_reasons=draft.lint_reasons,
                lint_actions=draft.lint_actions,
            )
        )
    return blocks
