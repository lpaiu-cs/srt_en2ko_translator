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
    mode: str = "soft"


@dataclass
class EmittedCue:
    cue_index: int
    text: str


@dataclass
class TranslationBlock:
    cues: List[Cue]
    previous_source_sentences: List[str] = field(default_factory=list)
    next_source_sentences: List[str] = field(default_factory=list)
    low_confidence: bool = False
    lint_reasons: List[str] = field(default_factory=list)
    lint_actions: List[str] = field(default_factory=list)


@dataclass
class TranslationRequest:
    block: TranslationBlock
    glossary_terms: List[GlossaryEntry] = field(default_factory=list)


@dataclass
class PhaseTranslationResult:
    emitted_cues: List[EmittedCue]
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class RepairRequest:
    block: TranslationBlock
    phase1_result: PhaseTranslationResult
    glossary_terms: List[GlossaryEntry] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class QualityGateResult:
    repair_needed: bool
    repair_reasons: List[str] = field(default_factory=list)
    warning_reasons: List[str] = field(default_factory=list)


@dataclass
class SchemaValidationResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)
