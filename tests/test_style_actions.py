from __future__ import annotations

import unittest

from subtitle_translator.config import load_runtime_config
from subtitle_translator.english_terms import (
    apply_approved_english_fallbacks,
    approved_english_glossary_terms_for_block,
    merge_glossary_terms,
)
from subtitle_translator.models import Cue, EmittedCue, GlossaryEntry, PhaseTranslationResult, RepairRequest, TranslationBlock, TranslationRequest
from subtitle_translator.pipeline import (
    _apply_deterministic_style_micro_edits,
    _apply_purpose_tail_post_normalization,
    _classify_not_invoked_reason,
    _choose_better_style_candidate,
    _continuation_detector_miss_feedback,
    _protected_cue_indices_for_spans,
    _strict_retry_prompt_profile_override,
    _wrap_phase_result,
)
from subtitle_translator.quality import post_wrap_gate, pre_wrap_gate
from subtitle_translator.translators import (
    OpenAIChatTranslator,
    _repair_profile_for_request,
    _splice_offending_cue_rewrites,
    build_repair_system_prompt_for_profile,
)


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

    def test_continuation_restore_missing_tail_uses_v3_prompt_override(self) -> None:
        offending_spans = [
            {
                "cue_index": 192,
                "source_tail_type": "continuation_tail",
                "preferred_action": "restore_missing_tail",
            }
        ]
        override = _strict_retry_prompt_profile_override(
            self.config,
            offending_spans,
            protected_cue_indices=[191],
        )
        self.assertEqual(override, "fragment_preserving_v3")

    def test_non_continuation_restore_missing_tail_does_not_use_v3_override(self) -> None:
        offending_spans = [
            {
                "cue_index": 1021,
                "source_tail_type": "purpose_tail",
                "preferred_action": "restore_missing_tail",
            }
        ]
        override = _strict_retry_prompt_profile_override(
            self.config,
            offending_spans,
            protected_cue_indices=[1020],
        )
        self.assertIsNone(override)

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

    def test_compact_technical_fragment_repair_profile_matches_0404_shape(self) -> None:
        self.config.repair_policy = "compact_technical_fragment_v1"
        request = RepairRequest(
            block=TranslationBlock(
                cues=[
                    Cue(
                        index=405,
                        start="00:16:50,000",
                        end="00:16:58,640",
                        text="So for AlexNet, it takes 0.7 GFLOPS. For VGG-16, it takes like 13.6 GFLOPS. But for C3D,",
                    )
                ],
                lint_reasons=["dependent_end"],
            ),
            phase1_result=PhaseTranslationResult(
                emitted_cues=[
                    EmittedCue(
                        cue_index=405,
                        text="그래서 AlexNet은 0.7 GFLOPS가 필요합니다. VGG-16은 약 13.6 GFLOPS가 필요하고, C3D는",
                    )
                ]
            ),
            failure_reasons=["line_overflow"],
        )
        self.assertEqual(_repair_profile_for_request(self.config, request), "compact_technical_fragment_v1")

    def test_compact_technical_fragment_repair_profile_ignores_nonmatching_rows(self) -> None:
        self.config.repair_policy = "compact_technical_fragment_v1"
        request = RepairRequest(
            block=TranslationBlock(
                cues=[
                    Cue(
                        index=1,
                        start="00:00:00,000",
                        end="00:00:03,000",
                        text="This is just a normal unfinished clause,",
                    )
                ],
                lint_reasons=["dependent_end"],
            ),
            phase1_result=PhaseTranslationResult(
                emitted_cues=[EmittedCue(cue_index=1, text="이건 그냥 일반적인 미완결 절이고,")]
            ),
            failure_reasons=["line_overflow"],
        )
        self.assertEqual(_repair_profile_for_request(self.config, request), "baseline")

    def test_local_rewrap_splits_dense_korean_into_two_valid_lines(self) -> None:
        self.config.max_lines_per_cue = 2
        dense_text = "가" * (self.config.max_chars_per_line + 6)
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:10,000", text="dense source")]
        )
        result = PhaseTranslationResult(emitted_cues=[EmittedCue(cue_index=1, text=dense_text)])

        wrapped = _wrap_phase_result(block, result, self.config, None)
        lines = wrapped[0].text.splitlines()

        self.assertLessEqual(len(lines), self.config.max_lines_per_cue)
        self.assertTrue(all(len(line) <= self.config.max_chars_per_line for line in lines))
        self.assertFalse(post_wrap_gate(wrapped, self.config).repair_needed)

    def test_repair_prompt_and_payload_expose_line_constraints(self) -> None:
        prompt = build_repair_system_prompt_for_profile(self.config, "baseline")
        self.assertIn(f"at most {self.config.max_lines_per_cue} line", prompt)
        self.assertIn(f"at most {self.config.max_chars_per_line} characters", prompt)
        self.assertIn("line_overflow", prompt)

        request = RepairRequest(
            block=TranslationBlock(
                cues=[Cue(index=1, start="00:00:00,000", end="00:00:03,000", text="source")]
            ),
            phase1_result=PhaseTranslationResult(emitted_cues=[EmittedCue(cue_index=1, text="번역")]),
            failure_reasons=["line_overflow"],
        )
        translator = OpenAIChatTranslator.__new__(OpenAIChatTranslator)
        translator.config = self.config
        payload = translator._payload_for_repair(request)
        self.assertEqual(
            payload["line_constraints"],
            {
                "max_lines_per_cue": self.config.max_lines_per_cue,
                "max_chars_per_line": self.config.max_chars_per_line,
            },
        )

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

    def test_technical_split_policy_downgrades_allowed_technical_english(self) -> None:
        self.config.english_residual_policy = "technical_split"
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="GPT-4 and Adam and RMSProp")],
        )
        emitted = [EmittedCue(cue_index=1, text="GPT-4와 Adam, RMSProp")]
        gate = pre_wrap_gate(block, emitted, [], self.config)
        self.assertFalse(gate.repair_needed)
        self.assertIn("english_residual_technical", gate.warning_reasons)
        self.assertNotIn("english_residual", gate.repair_reasons)

    def test_technical_split_policy_keeps_actual_residual_english_as_repair(self) -> None:
        self.config.english_residual_policy = "technical_split"
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="we need some weird framework semantics")],
        )
        emitted = [EmittedCue(cue_index=1, text="weird framework semantics를 생각해보면")]
        gate = pre_wrap_gate(block, emitted, [], self.config)
        self.assertTrue(gate.repair_needed)
        self.assertIn("english_residual", gate.repair_reasons)

    def test_english_residual_warning_details_include_terms(self) -> None:
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="source")],
        )
        emitted = [EmittedCue(cue_index=1, text="framework를 한국어 문장 안에서 아주 짧게 한 번만 언급하고 넘어갑니다")]
        gate = pre_wrap_gate(block, emitted, [], self.config)

        self.assertIn("english_residual_warn", gate.warning_reasons)
        self.assertEqual(gate.warning_details["english_residual_warn"][0]["term"], "framework")
        self.assertEqual(gate.warning_details["english_residual_warn"][0]["cue_index"], 1)

    def test_technical_english_warning_details_include_terms(self) -> None:
        self.config.english_residual_policy = "technical_split"
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="AlexNet and RMSProp")],
        )
        emitted = [EmittedCue(cue_index=1, text="AlexNet과 RMSProp")]
        gate = pre_wrap_gate(block, emitted, [], self.config)

        self.assertIn("english_residual_technical", gate.warning_reasons)
        self.assertEqual(
            [detail["term"] for detail in gate.warning_details["english_residual_technical"]],
            ["AlexNet", "RMSProp"],
        )

    def test_approved_english_terms_become_hard_glossary_terms(self) -> None:
        self.config.allowed_english_terms.append("alexnet")
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="AlexNet was the first example")],
        )

        terms = approved_english_glossary_terms_for_block(block.cues, self.config)

        self.assertEqual(terms[0].source, "AlexNet")
        self.assertEqual(terms[0].target, "AlexNet")
        self.assertEqual(terms[0].mode, "hard")

    def test_approved_english_glossary_overrides_existing_korean_alias(self) -> None:
        merged = merge_glossary_terms(
            [GlossaryEntry(source="AlexNet", target="알렉스넷", mode="hard")],
            [GlossaryEntry(source="AlexNet", target="AlexNet", mode="hard")],
        )

        self.assertEqual(merged[0].target, "AlexNet")

    def test_approved_english_fallback_replaces_declared_korean_alias(self) -> None:
        self.config.english_fallback_terms = [{"source": "AlexNet", "aliases": ["알렉스넷"]}]
        source = [Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="AlexNet was the first example")]
        output = [Cue(index=1, start="00:00:00,000", end="00:00:02,000", text="알렉스넷이 첫 예시였습니다.")]

        rewritten, replacements = apply_approved_english_fallbacks(source, output, self.config)

        self.assertEqual(rewritten[0].text, "AlexNet이 첫 예시였습니다.")
        self.assertEqual(replacements[0]["source"], "AlexNet")

    def test_cps_relaxed_wrap_policy_suppresses_warn_near_threshold(self) -> None:
        block = TranslationBlock(
            cues=[Cue(index=1, start="00:00:00,000", end="00:00:01,000", text="x")],
        )
        emitted = [EmittedCue(cue_index=1, text="가" * 19)]

        baseline_gate = pre_wrap_gate(block, emitted, [], self.config)
        self.assertIn("cps_warn", baseline_gate.warning_reasons)

        self.config.wrap_policy = "cps_relaxed_v1"
        relaxed_gate = pre_wrap_gate(block, emitted, [], self.config)
        self.assertNotIn("cps_warn", relaxed_gate.warning_reasons)

    def test_continuation_detector_miss_feedback_surfaces_fragmentary_tail(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(
                    index=191,
                    start="00:07:58,470",
                    end="00:08:02,070",
                    text="like model is that it can be trained with just associations",
                ),
                Cue(
                    index=192,
                    start="00:08:02,070",
                    end="00:08:03,650",
                    text="of images and text.",
                ),
            ],
            lint_reasons=["dependent_start", "comparison_midstart"],
            lint_actions=["carry_context_only"],
        )
        result = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=191, text="이 모델의 장점은 단지 연관성만으로도 학습할 수 있다는 점입니다."),
                EmittedCue(cue_index=192, text="이미지와 텍스트의."),
            ],
            risk_flags=[],
        )
        feedback = _continuation_detector_miss_feedback(block, result)
        self.assertIsNotNone(feedback)
        offending_cues, offending_spans, preferred_actions = feedback
        self.assertEqual(offending_cues, [192])
        self.assertEqual(preferred_actions, ["restore_missing_tail"])
        self.assertEqual(offending_spans[0]["trigger_reason"], "detector_miss")

    def test_not_invoked_reason_marks_acceptable_absorption(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(
                    index=147,
                    start="00:06:08,700",
                    end="00:06:13,180",
                    text="was that you want to pull together again,",
                ),
                Cue(
                    index=148,
                    start="00:06:13,180",
                    end="00:06:14,240",
                    text="transformations of the same image.",
                ),
            ],
            lint_reasons=["dependent_start"],
            lint_actions=[],
        )
        result = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=147, text="다시 한 번 모으고 싶은 것은, 변환된"),
                EmittedCue(cue_index=148, text="같은 이미지의 것들입니다."),
            ],
            risk_flags=[],
        )
        self.assertEqual(_classify_not_invoked_reason(block, result), "acceptable_absorption")

    def test_not_invoked_reason_marks_continuation_duplicate_as_detector_miss(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(
                    index=191,
                    start="00:07:58,470",
                    end="00:08:02,070",
                    text="like model is that it can be trained with just associations",
                ),
                Cue(
                    index=192,
                    start="00:08:02,070",
                    end="00:08:03,650",
                    text="of images and text.",
                ),
            ],
            lint_reasons=["dependent_start", "comparison_midstart"],
            lint_actions=["carry_context_only"],
        )
        result = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=191, text="모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,"),
                EmittedCue(cue_index=192, text="이미지와 텍스트의 연관성입니다."),
            ],
            risk_flags=[],
        )
        self.assertEqual(_classify_not_invoked_reason(block, result), "detector_miss")
        self.assertIsNotNone(_continuation_detector_miss_feedback(block, result))

    def test_forced_detector_miss_candidate_can_be_accepted_without_warning_reduction(self) -> None:
        block = TranslationBlock(
            cues=[
                Cue(index=191, start="00:07:58,470", end="00:08:02,070", text="like model is that it can be trained with just associations"),
                Cue(index=192, start="00:08:02,070", end="00:08:03,650", text="of images and text."),
            ],
            lint_reasons=["dependent_start", "comparison_midstart"],
            lint_actions=["carry_context_only"],
        )
        current = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=191, text="모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,"),
                EmittedCue(cue_index=192, text="이미지와 텍스트의 연관성입니다."),
            ],
            risk_flags=[],
        )
        candidate = PhaseTranslationResult(
            emitted_cues=[
                EmittedCue(cue_index=191, text="모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,"),
                EmittedCue(cue_index=192, text="이미지와 텍스트의 연관성으로요."),
            ],
            risk_flags=[],
        )
        offending_spans = [
            {
                "cue_index": 192,
                "cue_indices": [191, 192],
                "span_text": "이미지와 텍스트의 연관성입니다",
                "left_text": "모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,",
                "right_text": "이미지와 텍스트의 연관성입니다.",
                "source_cue_text": "of images and text.",
                "source_tail_type": "continuation_tail",
                "issue": "duplicate_restatement",
                "preferred_action": "restore_missing_tail",
                "trigger_reason": "detector_miss",
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

        self.assertTrue(accepted)
        self.assertEqual(rejection_causes, [])

    def test_splice_offending_cue_rewrites_preserves_protected_cues(self) -> None:
        request = TranslationRequest(
            block=TranslationBlock(
                cues=[
                    Cue(index=191, start="00:00:00,000", end="00:00:01,000", text="one nice thing about the model is"),
                    Cue(index=192, start="00:00:01,000", end="00:00:02,000", text="of images and text."),
                ]
            ),
            strict_style_retry=True,
            strict_retry_mode="offending_cue_only",
            previous_emitted_cues=[
                EmittedCue(cue_index=191, text="모델의 장점 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."),
                EmittedCue(cue_index=192, text="이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."),
            ],
            protected_cue_indices=[191],
            offending_cue_indices=[192],
        )

        result = _splice_offending_cue_rewrites(
            request,
            rewrites=[{"cue_index": 192, "text": "이미지와 텍스트의 연관성으로요."}],
            risk_flags=["duplicate_restatement_risk"],
        )

        self.assertEqual(result.emitted_cues[0].text, "모델의 장점 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.")
        self.assertEqual(result.emitted_cues[1].text, "이미지와 텍스트의 연관성으로요.")

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
