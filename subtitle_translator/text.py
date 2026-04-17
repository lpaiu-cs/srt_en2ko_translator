from __future__ import annotations

import re
import sys
from typing import List


def compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value: str) -> str:
    return compact_spaces(value.replace("\n", " "))


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _wrap_lines_greedy(text: str, width: int) -> str:
    if not text:
        return ""
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for word in words:
        next_len = cur_len + len(word) + (1 if cur else 0)
        if cur and next_len > width:
            lines.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur_len = next_len if cur else len(word)
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _balanced_two_line_wrap(text: str, width: int) -> str:
    words = text.split()
    if len(words) < 2:
        return _wrap_lines_greedy(text, width)

    best = _wrap_lines_greedy(text, width)
    best_score = None
    for split_at in range(1, len(words)):
        left = " ".join(words[:split_at]).strip()
        right = " ".join(words[split_at:]).strip()
        if not left or not right:
            continue
        left_len = len(left)
        right_len = len(right)
        overflow = max(0, left_len - width) + max(0, right_len - width)
        short_penalty = int(min(left_len, right_len) < 3)
        punctuation_penalty = 0 if left.endswith((",", ".", "!", "?", ":", ";")) else 1
        balance_penalty = abs(left_len - right_len)
        max_len_penalty = max(left_len, right_len)
        score = (overflow, short_penalty, punctuation_penalty, balance_penalty, max_len_penalty)
        if best_score is None or score < best_score:
            best = f"{left}\n{right}"
            best_score = score
    return best


def _punctuation_two_line_wrap(text: str, width: int) -> str:
    words = text.split()
    if len(words) < 2:
        return _wrap_lines_greedy(text, width)

    preferred_splits: List[int] = []
    for idx, word in enumerate(words[:-1], start=1):
        if word.endswith((",", ".", "!", "?", ":", ";")):
            preferred_splits.append(idx)
    if not preferred_splits:
        return _balanced_two_line_wrap(text, width)

    best = _wrap_lines_greedy(text, width)
    best_score = None
    for split_at in preferred_splits:
        left = " ".join(words[:split_at]).strip()
        right = " ".join(words[split_at:]).strip()
        if not left or not right:
            continue
        left_len = len(left)
        right_len = len(right)
        overflow = max(0, left_len - width) + max(0, right_len - width)
        balance_penalty = abs(left_len - right_len)
        max_len_penalty = max(left_len, right_len)
        score = (overflow, balance_penalty, max_len_penalty)
        if best_score is None or score < best_score:
            best = f"{left}\n{right}"
            best_score = score
    return best


def wrap_lines(text: str, width: int = 42) -> str:
    return _wrap_lines_greedy(text, width)


def wrap_lines_candidates(text: str, width: int = 42, max_lines: int = 2) -> List[str]:
    normalized = compact_spaces(text)
    if not normalized:
        return [""]

    candidates = [
        _wrap_lines_greedy(normalized, width),
        _balanced_two_line_wrap(normalized, width),
        _punctuation_two_line_wrap(normalized, width),
    ]
    if max_lines >= 2:
        if width > 10:
            candidates.append(_wrap_lines_greedy(normalized, width - 2))
        if width > 14:
            candidates.append(_wrap_lines_greedy(normalized, width - 4))

    unique: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique or [normalized]
