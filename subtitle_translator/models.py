from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    strict_style_retry: bool = False
    strict_retry_mode: str = "full_block"
    prompt_profile_override: str | None = None
    style_retry_reasons: List[str] = field(default_factory=list)
    previous_emitted_cues: List[EmittedCue] = field(default_factory=list)
    protected_cue_indices: List[int] = field(default_factory=list)
    offending_cue_indices: List[int] = field(default_factory=list)
    offending_spans: List[Dict[str, Any]] = field(default_factory=list)
    preferred_actions: List[str] = field(default_factory=list)


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
    warning_details: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class SchemaValidationResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)
