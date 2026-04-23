## Round 22: Continuation-Tail Focus

### Goal

Replay-evaluate the widened 17-row replay corpus again after adding selector-rejection subtypes, then re-freeze the 5-row continuation debug set.

### Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round22_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round22_full_pipeline_eval.jsonl`
- `evaluation/cs231n_sp25_continuation_tail_lane_debug_round22.jsonl`

### Widened Replay: Continuation Slice

#### Style-only

- `surfaced_same_action | strict_direct_accept | accepted`: `1`
- `surfaced_same_action | rejected | strict_retry_selector | local_meaning_not_restored`: `1`
- `unsurfaced | not_invoked`: `3`

#### Full-pipeline

- `surfaced_same_action | strict_direct_accept | accepted`: `1`
- `surfaced_same_action | rejected | strict_retry_selector | local_meaning_not_restored`: `2`
- `unsurfaced | not_invoked`: `2`

### Continuation 5-row Debug Set

1. `Lecture 11::block-0657`
   - both lanes accept
   - label: `positive_control`

2. `Lecture 16::block-0078`
   - both lanes `unsurfaced -> not_invoked`
   - label: `acceptable_absorption`

3. `Lecture 16::block-0101`
   - style-only: `unsurfaced -> not_invoked`
   - full-pipeline: `selector_reject | local_meaning_not_restored`
   - labels: `detector_miss` / `selector_reject`

4. `Lecture 16::block-0124`
   - both lanes: `selector_reject | local_meaning_not_restored`
   - label: `selector_reject`

5. `Lecture 18::block-0211`
   - both lanes: `unsurfaced -> not_invoked`
   - label: `detector_miss`

### Interpretation

- `continuation_tail` is now the clearest next code target.
- The main subproblems are:
  - `acceptable_absorption`: rows like `block-0078`
  - `detector_miss`: rows like `block-0101` style-only and `block-0211`
  - `selector_reject / local_meaning_not_restored`: rows like `block-0124` and `block-0101` full-pipeline
- `empty_tail_collapse` did not remain the dominant continuation failure in round22; the more common subtype is now `local_meaning_not_restored`, plus a separate unsurfaced/not-invoked branch.

### Next Step

- If behavior changes resume, they should target continuation-tail invocation/selection first.
- Deterministic salvage should still remain purpose-only.
