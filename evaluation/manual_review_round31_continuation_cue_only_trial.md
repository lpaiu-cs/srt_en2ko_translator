## Round31 Continuation Cue-Only Trial

This round tested a broader `offending_cue_only` strict-retry contract for `restore_missing_tail|continuation_tail`.
The mode was widened from detector-miss-only rows to all continuation-tail rows with a single offending cue and protected cues.

### Outcome

- The broader contract did **not** improve the target failure cleanly.
- `Lecture 16::block-0101` still failed under the cue-only contract:
  - style-only: `rejected -> surfaced_same_action -> rejected`
  - full-pipeline: `rejected -> surfaced_same_action -> rejected`
- The new failures were not invocation misses anymore. They became cue-only strict-retry quality failures:
  - style-only `block-0101`: `strict_retry_overedit`
  - full-pipeline `block-0101`: `strict_retry_selector | local_meaning_not_restored`

### Regression Signals

- `Lecture 18::block-0211` regressed from direct accept to `strict_retry_selector | local_meaning_not_restored` in both lanes.
- `Lecture 11::block-0657` no longer stayed on the previous direct-accept path:
  - style-only became `surfaced_other_action -> micro_edit_only`
  - full-pipeline also moved to `surfaced_other_action -> micro_edit_only`

### Decision

The broad continuation cue-only trigger was reverted.
The repo state remains on the narrower detector-miss-only trigger that produced the round30 baseline:

- `block-0101` fixed in full-pipeline
- `block-0078` preserved as `acceptable_absorption`
- `block-0124` and `block-0211` remained direct-accept on the stable path
- no continuation-wide deterministic salvage was added

### Files

- style-only trial:
  `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round31_style_only_eval.jsonl`
- full-pipeline trial:
  `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round31_full_pipeline_eval.jsonl`
- continuation debug:
  `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_lane_debug_round31.jsonl`
