from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Cue:
    index: int
    start: str
    end: str
    text: str


@dataclass
class SentenceGroup:
    text: str
    cue_indices: List[int]


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    note: str = ""


@dataclass
class TranslationRequest:
    sentences: List[str]
    previous_translation: str = ""
    glossary_terms: List[GlossaryEntry] = field(default_factory=list)


@dataclass
class BatchTranslationResult:
    translations: List[str]
    glossary_updates: List[GlossaryEntry] = field(default_factory=list)
