## Round 29: Continuation Detector-Miss Invocation

### Change

Added runtime `style_retry_not_invoked_reason` logging and a narrow continuation-tail detector-miss invocation path.

Scope:

- keep `purpose_tail` deterministic salvage unchanged
- keep `continuation_tail` acceptable-absorption rows unsurfaced
- surface only narrow `carry_context_only + continuation_tail + duplicated/local-tail-miss` cases

Files changed:

- `subtitle_translator/pipeline.py`
- `run_review_eval.py`
- `run_force_style_retry_counterfactual.py`
- `tests/test_style_actions.py`
- `tests/test_replay_surface_state.py`

### Counterfactual

- `evaluation/cs231n_sp25_block0101_force_counterfactual_round26.json`
- `evaluation/cs231n_sp25_block0101_force_counterfactual_round26_current.json`

Both counterfactuals showed that `Lecture 16::block-0101` can produce a better continuation-tail candidate such as `이미지와 텍스트의 연관성으로요.` when strict retry is forced.

### Eval Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round29_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round29_full_pipeline_eval.jsonl`
- `evaluation/cs231n_sp25_continuation_tail_lane_debug_round29.jsonl`

### Continuation 5-row Result

#### Style-only

- `block-0078`: `unsurfaced -> not_invoked`, `acceptable_absorption`
- `block-0101`: `surfaced_same_action -> strict_direct_accept`
- `block-0124`: `surfaced_same_action -> strict_direct_accept`
- `block-0211`: `surfaced_same_action -> strict_direct_accept`
- `block-0657`: `unsurfaced -> not_invoked`, `acceptable_absorption`

#### Full-pipeline

- `block-0078`: `unsurfaced -> not_invoked`, `acceptable_absorption`
- `block-0101`: `surfaced_same_action -> rejected`, `strict_retry_overedit`
- `block-0124`: `surfaced_same_action -> strict_direct_accept`
- `block-0211`: `surfaced_same_action -> strict_direct_accept`
- `block-0657`: `surfaced_same_action -> strict_direct_accept`

### What Improved

- `block-0101` is no longer opaque. It now records:
  - `style-only`: detector miss surfaced and accepted
  - `full-pipeline`: detector miss surfaced, but rejected as overedit

- `block-0078` stayed protected as `acceptable_absorption`.

- The runtime output now records `style_retry_not_invoked_reason`, which makes the continuation `not_invoked` split inspectable without manual-only labeling.

### Remaining Issue

- `block-0101` in full-pipeline is no longer an invocation miss.
- The remaining failure is now strict retry overediting cue 1 in the full-pipeline path.

### Interpretation

The continuation-tail problem has moved one step downstream:

- before: `0101` was blocked at invocation
- now: `0101` reaches strict retry, but the full-pipeline candidate still rewrites the protected cue and gets rejected as overedited

So the next continuation-specific patch should target **full-pipeline strict candidate overedit on `block-0101`-like rows**, not broader invocation.
