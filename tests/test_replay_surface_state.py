from __future__ import annotations

import unittest

from run_review_eval import (
    _current_accept_mode,
    _current_surfaced_actions,
    _replay_surface_info,
    _replay_transition,
    _style_retry_rejection_stage,
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


if __name__ == "__main__":
    unittest.main()
