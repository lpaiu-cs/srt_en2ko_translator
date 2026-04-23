from __future__ import annotations

import unittest

from collect_continuation_signature_rows import _effective_strict_prompt_profile, _signature_row
from collect_continuation_signature_rows import _matches as _watchlist_matches
from run_review_eval import (
    _current_accept_mode,
    _current_surfaced_actions,
    _replay_surface_info,
    _replay_transition,
    _style_retry_not_invoked_reason,
    _style_retry_rejection_stage,
    _style_retry_rejection_subtype,
)


class ReplaySurfaceStateTests(unittest.TestCase):
    def test_surface_state_same_action(self) -> None:
        entry = {
            "replay_meta": {
                "style_focus": "restore_missing_tail",
                "source_tail_type": "purpose_tail",
                "style_retry_outcome": "accepted",
            }
        }
        trace = {
            "offending_spans": [
                {
                    "preferred_action": "restore_missing_tail",
                    "source_tail_type": "purpose_tail",
                }
            ]
        }
        info = _replay_surface_info(entry, trace)
        self.assertEqual(info["surface_state"], "surfaced_same_action")
        self.assertEqual(info["current_actions"], ["restore_missing_tail"])

    def test_surface_state_other_action(self) -> None:
        entry = {
            "replay_meta": {
                "style_focus": "restore_missing_tail",
                "source_tail_type": "purpose_tail",
                "style_retry_outcome": "rejected",
            }
        }
        trace = {
            "offending_spans": [
                {
                    "preferred_action": "trim_explanatory_tail",
                }
            ]
        }
        info = _replay_surface_info(entry, trace)
        self.assertEqual(info["surface_state"], "surfaced_other_action")
        self.assertEqual(info["current_actions"], ["trim_explanatory_tail"])

    def test_surface_state_unsurfaced(self) -> None:
        entry = {
            "replay_meta": {
                "style_focus": "restore_missing_tail",
                "source_tail_type": "continuation_tail",
                "style_retry_outcome": "rejected",
            }
        }
        info = _replay_surface_info(entry, {})
        self.assertEqual(info["surface_state"], "unsurfaced")
        self.assertEqual(info["current_actions"], [])

    def test_surface_actions_fallback_to_preferred_actions(self) -> None:
        trace = {"preferred_actions": ["restore_missing_tail", "restore_missing_tail", "delete_repeat_local"]}
        self.assertEqual(_current_surfaced_actions(trace), ["delete_repeat_local", "restore_missing_tail"])

    def test_replay_transition_and_rejection_stage(self) -> None:
        entry = {
            "replay_meta": {
                "style_focus": "restore_missing_tail",
                "source_tail_type": "continuation_tail",
                "style_retry_outcome": "rejected",
            }
        }
        signals = {
            "style_retry_invoked": True,
            "style_retry_accepted": False,
            "style_retry_rejected": True,
            "replay_surface_state": "surfaced_same_action",
            "style_retry_trace": {
                "rejection_causes": ["restore_tail_warning_persisted"],
            },
        }
        self.assertEqual(_current_accept_mode(signals), "rejected")
        self.assertEqual(_style_retry_rejection_stage(signals), "strict_retry_selector")
        self.assertEqual(
            _replay_transition(entry, signals),
            "rejected->surfaced_same_action->rejected",
        )

    def test_replay_transition_unsurfaced_not_invoked(self) -> None:
        entry = {
            "replay_meta": {
                "style_focus": "restore_missing_tail",
                "source_tail_type": "purpose_tail",
                "style_retry_outcome": "accepted",
            }
        }
        signals = {
            "style_retry_invoked": False,
            "style_retry_accepted": False,
            "style_retry_rejected": False,
            "replay_surface_state": "unsurfaced",
            "style_retry_trace": {},
        }
        self.assertEqual(_current_accept_mode(signals), "not_invoked")
        self.assertEqual(_style_retry_rejection_stage(signals), "not_invoked")
        self.assertEqual(
            _replay_transition(entry, signals),
            "accepted->unsurfaced->not_invoked",
        )

    def test_not_invoked_reason(self) -> None:
        signals = {
            "style_retry_invoked": False,
            "style_retry_accepted": False,
            "style_retry_rejected": False,
            "style_retry_trace": {"not_invoked_reason": "acceptable_absorption"},
        }
        self.assertEqual(_style_retry_not_invoked_reason(signals), "acceptable_absorption")

    def test_rejection_subtypes(self) -> None:
        empty_signals = {
            "style_retry_invoked": True,
            "style_retry_accepted": False,
            "style_retry_rejected": True,
            "style_retry_trace": {"rejection_causes": ["restore_tail_empty"]},
        }
        self.assertEqual(_style_retry_rejection_subtype(empty_signals), "empty_tail_collapse")

        protected_signals = {
            "style_retry_invoked": True,
            "style_retry_accepted": False,
            "style_retry_rejected": True,
            "style_retry_trace": {"rejection_causes": ["protected_cue_touched"]},
        }
        self.assertEqual(_style_retry_rejection_subtype(protected_signals), "protected_cue_touched")

        local_signals = {
            "style_retry_invoked": True,
            "style_retry_accepted": False,
            "style_retry_rejected": True,
            "style_retry_trace": {"rejection_causes": ["restore_tail_overclosed_for_continuation"]},
        }
        self.assertEqual(_style_retry_rejection_subtype(local_signals), "local_meaning_not_restored")

    def test_watchlist_match_with_stage_and_subtype_filters(self) -> None:
        row = {
            "id": "lecture::block-1",
            "provenance": {"prompt_profile": "fragment_preserving_v2"},
            "current_block": {
                "source_cues": [
                    {"cue_index": 1, "text": "setup"},
                    {"cue_index": 2, "text": "from these interactions."},
                ]
            },
            "pipeline_signals": {
                "style_retry_invoked": True,
                "style_retry_rejection_stage": "strict_retry_selector",
                "style_retry_rejection_subtype": "local_meaning_not_restored",
                "style_retry_trace": {
                    "offending_cue_indices": [2],
                    "protected_cue_indices": [1],
                    "offending_spans": [
                        {
                            "preferred_action": "restore_missing_tail",
                            "source_tail_type": "continuation_tail",
                        }
                    ],
                },
            },
        }
        self.assertTrue(
            _watchlist_matches(
                row,
                stages={"strict_retry_selector"},
                subtypes={"local_meaning_not_restored"},
            )
        )
        self.assertFalse(
            _watchlist_matches(
                row,
                stages={"strict_retry_overedit"},
                subtypes={"local_meaning_not_restored"},
            )
        )
        self.assertTrue(
            _watchlist_matches(
                row,
                stages={"strict_retry_selector"},
                subtypes={"local_meaning_not_restored"},
                effective_profiles={"fragment_preserving_v3"},
            )
        )
        self.assertFalse(
            _watchlist_matches(
                row,
                stages={"strict_retry_selector"},
                subtypes={"local_meaning_not_restored"},
                effective_profiles={"fragment_preserving_v2"},
            )
        )

    def test_watchlist_signature_row_keeps_effective_strict_prompt_profile(self) -> None:
        row = {
            "id": "lecture::block-2",
            "lecture": "Lecture X",
            "pipeline_signals": {
                "style_retry_rejection_stage": "strict_retry_selector",
                "style_retry_trace": {
                    "effective_strict_prompt_profile": "fragment_preserving_v3",
                },
            },
        }
        signature = _signature_row(row, "dummy.jsonl")
        self.assertEqual(signature["effective_strict_prompt_profile"], "fragment_preserving_v3")

    def test_effective_strict_prompt_profile_backfills_continuation_v3(self) -> None:
        row = {
            "provenance": {"prompt_profile": "fragment_preserving_v2"},
            "pipeline_signals": {
                "style_retry_trace": {
                    "offending_cue_indices": [2],
                    "protected_cue_indices": [1],
                    "offending_spans": [
                        {
                            "preferred_action": "restore_missing_tail",
                            "source_tail_type": "continuation_tail",
                        }
                    ],
                }
            },
        }
        self.assertEqual(_effective_strict_prompt_profile(row), "fragment_preserving_v3")

    def test_pipeline_signal_can_surface_effective_strict_prompt_profile(self) -> None:
        signals = {
            "style_retry_trace": {
                "effective_strict_prompt_profile": "fragment_preserving_v3",
            }
        }
        self.assertEqual(
            signals["style_retry_trace"].get("effective_strict_prompt_profile"),
            "fragment_preserving_v3",
        )


if __name__ == "__main__":
    unittest.main()
