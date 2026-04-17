from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .blocks import build_translation_blocks
from .config import RuntimeConfig
from .glossary import GlossaryStore
from .models import Cue, EmittedCue, PhaseTranslationResult, RepairRequest, TranslationRequest, TranslationBlock
from .quality import evaluate_quality
from .text import normalize_text, warn, wrap_lines
from .translators import BaseTranslator


def create_glossary_store(config: RuntimeConfig, glossary_log_path: Optional[str] = None) -> GlossaryStore:
    path = Path(glossary_log_path).expanduser() if glossary_log_path is not None else config.glossary_log_path
    return GlossaryStore(path=path, max_terms=config.glossary_max_terms)


def _fallback_phase1_result(block: TranslationBlock, reason: str) -> PhaseTranslationResult:
    return PhaseTranslationResult(
        emitted_cues=[
            EmittedCue(cue_index=cue.index, text=normalize_text(cue.text))
            for cue in block.cues
        ],
        risk_flags=[reason],
    )


def _result_to_cues(block: TranslationBlock, result: PhaseTranslationResult, config: RuntimeConfig) -> List[Cue]:
    text_by_index = {emitted.cue_index: normalize_text(emitted.text) for emitted in result.emitted_cues}
    output: List[Cue] = []
    for cue in block.cues:
        text = wrap_lines(text_by_index.get(cue.index, ""), width=config.max_chars_per_line)
        output.append(Cue(index=cue.index, start=cue.start, end=cue.end, text=text))
    return output


def _pick_better_result(
    phase1_result: PhaseTranslationResult,
    phase1_reasons: List[str],
    repair_result: PhaseTranslationResult,
    repair_reasons: List[str],
) -> PhaseTranslationResult:
    if not repair_reasons and phase1_reasons:
        return repair_result
    if not phase1_reasons and repair_reasons:
        return phase1_result
    if len(repair_reasons) < len(phase1_reasons):
        return repair_result
    return phase1_result


def translate_srt(
    cues: List[Cue],
    translator: BaseTranslator,
    config: RuntimeConfig,
    glossary_store: Optional[GlossaryStore] = None,
) -> List[Cue]:
    blocks = build_translation_blocks(cues, config)
    output_cues: List[Cue] = []

    for block in blocks:
        glossary_terms = glossary_store.relevant_terms([cue.text for cue in block.cues]) if glossary_store else []
        phase1_request = TranslationRequest(block=block, glossary_terms=glossary_terms)

        try:
            phase1_result = translator.translate_block(phase1_request)
        except Exception as exc:
            warn(f"Phase1 translation failed for cues {[cue.index for cue in block.cues]}: {exc}")
            phase1_result = _fallback_phase1_result(block, "phase1_exception")

        phase1_gate = evaluate_quality(block, phase1_result.emitted_cues, glossary_terms, config)
        final_result = phase1_result

        should_repair = config.repair_enabled and (not phase1_gate.passed or bool(phase1_result.risk_flags))
        if should_repair:
            failure_reasons = list(dict.fromkeys(phase1_gate.reasons + phase1_result.risk_flags))
            repair_request = RepairRequest(
                block=block,
                phase1_result=phase1_result,
                glossary_terms=glossary_terms,
                failure_reasons=failure_reasons,
            )
            try:
                repair_result = translator.repair_block(repair_request)
            except Exception as exc:
                warn(f"Phase2 repair failed for cues {[cue.index for cue in block.cues]}: {exc}")
            else:
                repair_gate = evaluate_quality(block, repair_result.emitted_cues, glossary_terms, config)
                final_result = _pick_better_result(
                    phase1_result=phase1_result,
                    phase1_reasons=phase1_gate.reasons,
                    repair_result=repair_result,
                    repair_reasons=repair_gate.reasons,
                )

        output_cues.extend(_result_to_cues(block, final_result, config))

    return output_cues
