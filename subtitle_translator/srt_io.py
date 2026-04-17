from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import Cue


TIMING_LINE_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(?:\s*.*)?$"
)


def read_srt(path: str) -> List[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip(), flags=re.MULTILINE)
    cues: List[Cue] = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        try:
            index = int(lines[0].strip())
            timing_line_index = 1
        except ValueError:
            index = len(cues) + 1
            timing_line_index = 0
        match = TIMING_LINE_RE.match(lines[timing_line_index])
        if not match:
            raise ValueError(f"Invalid timing line in block starting with: {lines[:2]}")
        start, end = match.group(1), match.group(2)
        text_lines = lines[timing_line_index + 1:]
        cues.append(Cue(index=index, start=start, end=end, text="\n".join(text_lines)))
    return cues


def write_srt(cues: List[Cue], path: str) -> None:
    parts = []
    for cue in cues:
        parts.append(str(cue.index))
        parts.append(f"{cue.start} --> {cue.end}")
        parts.append(cue.text if cue.text.strip() else "")
        parts.append("")
    Path(path).write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
