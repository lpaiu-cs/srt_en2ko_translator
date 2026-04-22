from __future__ import annotations

import unittest

from subtitle_translator.config import load_runtime_config
from subtitle_translator.models import Cue, EmittedCue, PhaseTranslationResult, TranslationBlock
from subtitle_translator.pipeline import (
    _apply_deterministic_style_micro_edits,
    _apply_purpose_tail_post_normalization,
    _choose_better_style_candidate,
    _protected_cue_indices_for_spans,
    _wrap_phase_result,
)
from subtitle_translator.quality import post_wrap_gate, pre_wrap_gate


class StyleActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_runtime_config(glossary_log_path="")

    def test_deterministic_micro_edit_removes_head_marker_and_tail(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(
                    index=1,
                    start="00:00:00,000",
                    end="00:00:01,000",
                    text="which are convolutional layers",
                )
            ],
            lint_reasons=["dependent_end"],
            lint_actions=["carry_context_only"],
        )
        base = PhaseTranslationResult(
            emitted_cues=[EmittedCue(cue_index=1, text="즉, 이런 것들이 컨볼루션 레이어입니다,")],
            risk_flags=["unsupported_head_marker_risk", "unsupported_explanatory_tail_risk"],
        )
        spans = [
            {
                "cue_index": 1,
                "span_text": "즉,",
                "issue": "unsupported_head_marker",
                "preferred_action": "drop_head_marker",
            },
            {
                "cue_index": 1,
                "span_text": "레이어입니다,",
                "issue": "unsupported_explanatory_tail",
                "preferred_action": "trim_explanatory_tail",
            },
        ]

        edited = _apply_deterministic_style_micro_edits(base, spans)
        self.assertIsNotNone(edited)
        self.assertEqual(edited.emitted_cues[0].text, "이런 것들이 컨볼루션")
        self.assertEqual(edited.risk_flags, [])

    def test_restore_missing_tail_selector_accepts_local_tail_rewrite(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(
                    index=1020,
                    start="00:38:52,147",
                    end="00:38:53,980",
                    text="But we need a little bit more structure here",
                ),
                Cue(
                    index=1021,
                    start="00:38:53,980",
                    end="00:38:56,900",
                    text="to actually make progress.",
                ),
            ]
        )
        current = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=1020, text="하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."),
                EmittedCue(cue_index=1021, text="조금 더 구조가 필요합니다."),
            ],
            risk_flags=["duplicate_restatement_risk"],
        )
        candidate = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=1020, text="하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."),
                EmittedCue(cue_index=1021, text="실제로 진전을 이루기 위해서요."),
            ],
            risk_flags=[],
        )
        offending_spans = [
            {
                "cue_index": 1021,
                "cue_indices": [1020, 1021],
                "span_text": "조금 더 구조가 필요합니다.",
                "left_text": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다.",
                "right_text": "조금 더 구조가 필요합니다.",
                "source_cue_text": "to actually make progress.",
                "source_tail_type": "purpose_tail",
                "issue": "duplicate_restatement",
                "preferred_action": "restore_missing_tail",
            }
        ]

        current_pre = pre_wrap_gate(block, current.emitted_cues, [], self.config)
        current_post = post_wrap_gate(_wrap_phase_result(block, current, self.config, None), self.config)
        candidate_pre = pre_wrap_gate(block, candidate.emitted_cues, [], self.config)
        candidate_post = post_wrap_gate(_wrap_phase_result(block, candidate, self.config, None), self.config)

        (
            final_result,
            _,
            _,
            accepted,
            rejection_causes,
        ) = _choose_better_style_candidate(
            current,
            current_pre,
            current_post,
            candidate,
            candidate_pre,
            candidate_post,
            ["duplicate_restatement"],
            _protected_cue_indices_for_spans(block, offending_spans),
            offending_spans=offending_spans,
        )

        self.assertTrue(accepted)
        self.assertEqual(rejection_causes, [])
        self.assertEqual(final_result.emitted_cues[1].text, "실제로 진전을 이루기 위해서요.")
        self.assertEqual(_protected_cue_indices_for_spans(block, offending_spans), [1020])

    def test_purpose_tail_post_normalization_salvages_overclosed_candidate(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(index=1020, start="00:38:52,147", end="00:38:53,980", text="But we need a little bit more structure here"),
                Cue(index=1021, start="00:38:53,980", end="00:38:56,900", text="to actually make progress."),
            ]
        )
        current = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=1020, text="하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."),
                EmittedCue(cue_index=1021, text="조금 더 구조가 필요합니다."),
            ],
            risk_flags=["duplicate_restatement_risk"],
        )
        overclosed = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=1020, text="하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."),
                EmittedCue(cue_index=1021, text="실제로 진전을 이루기 위해서입니다."),
            ],
            risk_flags=[],
        )
        offending_spans = [
            {
                "cue_index": 1021,
                "cue_indices": [1020, 1021],
                "span_text": "조금 더 구조가 필요합니다.",
                "left_text": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다.",
                "right_text": "조금 더 구조가 필요합니다.",
                "source_cue_text": "to actually make progress.",
                "source_tail_type": "purpose_tail",
                "issue": "duplicate_restatement",
                "preferred_action": "restore_missing_tail",
            }
        ]
        normalized, applied = _apply_purpose_tail_post_normalization(overclosed, offending_spans)
        self.assertEqual(normalized.emitted_cues[1].text, "실제로 진전을 이루기 위해")
        self.assertEqual(applied[0]["cue_index"], 1021)

        current_pre = pre_wrap_gate(block, current.emitted_cues, [], self.config)
        current_post = post_wrap_gate(_wrap_phase_result(block, current, self.config, None), self.config)
        candidate_pre = pre_wrap_gate(block, normalized.emitted_cues, [], self.config)
        candidate_post = post_wrap_gate(_wrap_phase_result(block, normalized, self.config, None), self.config)

        _, _, _, accepted, rejection_causes = _choose_better_style_candidate(
            current,
            current_pre,
            current_post,
            normalized,
            candidate_pre,
            candidate_post,
            ["duplicate_restatement"],
            _protected_cue_indices_for_spans(block, offending_spans),
            offending_spans=offending_spans,
        )
        self.assertTrue(accepted)
        self.assertEqual(rejection_causes, [])

    def test_purpose_tail_post_normalization_handles_variants_and_punctuation(self) -> None:
        offending_spans = [
            {
                "cue_index": 11,
                "source_tail_type": "purpose_tail",
                "preferred_action": "restore_missing_tail",
            }
        ]
        candidate = PhaseTranslationResult(
            emitted_cues=[EmittedCue(cue_index=11, text="메모리를 절약하기 위한 것입니다,")],
            risk_flags=[],
        )
        normalized, applied = _apply_purpose_tail_post_normalization(candidate, offending_spans)
        self.assertEqual(normalized.emitted_cues[0].text, "메모리를 절약하기 위해")
        self.assertEqual(applied[0]["after"], "메모리를 절약하기 위해")

    def test_purpose_tail_post_normalization_keeps_existing_fragment(self) -> None:
        offending_spans = [
            {
                "cue_index": 11,
                "source_tail_type": "purpose_tail",
                "preferred_action": "restore_missing_tail",
            }
        ]
        candidate = PhaseTranslationResult(
            emitted_cues=[EmittedCue(cue_index=11, text="실제로 진전을 이루기 위해")],
            risk_flags=[],
        )
        normalized, applied = _apply_purpose_tail_post_normalization(candidate, offending_spans)
        self.assertEqual(normalized.emitted_cues[0].text, "실제로 진전을 이루기 위해")
        self.assertEqual(applied, [])

    def test_restore_missing_tail_rejects_purpose_tail_without_purpose_marker(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(index=10, start="00:00:00,000", end="00:00:01,500", text="We need more structure"),
                Cue(index=11, start="00:00:01,500", end="00:00:03,000", text="to make progress."),
            ]
        )
        current = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=10, text="더 많은 구조가 필요합니다."),
                EmittedCue(cue_index=11, text="더 많은 구조가 필요합니다."),
            ],
            risk_flags=["duplicate_restatement_risk"],
        )
        candidate = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=10, text="더 많은 구조가 필요합니다."),
                EmittedCue(cue_index=11, text="실제로 진전을 이룹니다."),
            ],
            risk_flags=[],
        )
        offending_spans = [
            {
                "cue_index": 11,
                "cue_indices": [10, 11],
                "span_text": "더 많은 구조가 필요합니다.",
                "left_text": "더 많은 구조가 필요합니다.",
                "right_text": "더 많은 구조가 필요합니다.",
                "source_cue_text": "to make progress.",
                "source_tail_type": "purpose_tail",
                "issue": "duplicate_restatement",
                "preferred_action": "restore_missing_tail",
            }
        ]
        current_pre = pre_wrap_gate(block, current.emitted_cues, [], self.config)
        current_post = post_wrap_gate(_wrap_phase_result(block, current, self.config, None), self.config)
        candidate_pre = pre_wrap_gate(block, candidate.emitted_cues, [], self.config)
        candidate_post = post_wrap_gate(_wrap_phase_result(block, candidate, self.config, None), self.config)

        _, _, _, accepted, rejection_causes = _choose_better_style_candidate(
            current,
            current_pre,
            current_post,
            candidate,
            candidate_pre,
            candidate_post,
            ["duplicate_restatement"],
            _protected_cue_indices_for_spans(block, offending_spans),
            offending_spans=offending_spans,
        )

        self.assertFalse(accepted)
        self.assertIn("restore_tail_purpose_marker_missing", rejection_causes)

    def test_restore_missing_tail_rejects_overclosed_relative_clause(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(index=20, start="00:00:00,000", end="00:00:01,500", text="50 is a pretty good one"),
                Cue(index=21, start="00:00:01,500", end="00:00:03,000", text="that most people use in practice."),
            ]
        )
        current = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=20, text="50이 꽤 좋은 값입니다."),
                EmittedCue(cue_index=21, text="대부분의 사람들이 실제로 사용합니다."),
            ],
            risk_flags=["duplicate_restatement_risk"],
        )
        candidate = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=20, text="50이 꽤 좋은 값입니다."),
                EmittedCue(cue_index=21, text="대부분의 사람들이 실제로 사용하는 값입니다."),
            ],
            risk_flags=[],
        )
        offending_spans = [
            {
                "cue_index": 21,
                "cue_indices": [20, 21],
                "span_text": "대부분의 사람들이 실제로 사용합니다.",
                "left_text": "50이 꽤 좋은 값입니다.",
                "right_text": "대부분의 사람들이 실제로 사용합니다.",
                "source_cue_text": "that most people use in practice.",
                "source_tail_type": "relative_clause_tail",
                "issue": "duplicate_restatement",
                "preferred_action": "restore_missing_tail",
            }
        ]
        current_pre = pre_wrap_gate(block, current.emitted_cues, [], self.config)
        current_post = post_wrap_gate(_wrap_phase_result(block, current, self.config, None), self.config)
        candidate_pre = pre_wrap_gate(block, candidate.emitted_cues, [], self.config)
        candidate_post = post_wrap_gate(_wrap_phase_result(block, candidate, self.config, None), self.config)

        _, _, _, accepted, rejection_causes = _choose_better_style_candidate(
            current,
            current_pre,
            current_post,
            candidate,
            candidate_pre,
            candidate_post,
            ["duplicate_restatement"],
            _protected_cue_indices_for_spans(block, offending_spans),
            offending_spans=offending_spans,
        )

        self.assertFalse(accepted)
        self.assertIn("restore_tail_overclosed_for_relative_clause", rejection_causes)


if __name__ == "__main__":
    unittest.main()
