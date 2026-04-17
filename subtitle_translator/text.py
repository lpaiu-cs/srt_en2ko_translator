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


def wrap_lines(text: str, width: int = 42) -> str:
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
