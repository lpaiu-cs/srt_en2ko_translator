## Round 21: Widened Replay Eval

### Goal

Replay-evaluate the widened 17-row runtime replay corpus in both lanes and inspect the 5-row `continuation_tail` slice directly.

### Inputs

- Replay corpus:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round20b.jsonl`

### Outputs

- Style-only:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round21_style_only_eval.jsonl`
- Full-pipeline:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round21_full_pipeline_eval.jsonl`
- Continuation fixed debug set:
  - `evaluation/cs231n_sp25_continuation_tail_lane_debug_round21.jsonl`

### Widened Replay Summary

#### Style-only

`tail_type × surface_state × accept_mode × rejection_stage`

- `purpose_tail`
  - `surfaced_same_action | postnorm_salvaged_accept | accepted`: `2`
  - `surfaced_same_action | rejected | strict_retry_selector`: `3`
  - `unsurfaced | not_invoked | not_invoked`: `3`
- `continuation_tail`
  - `surfaced_same_action | strict_direct_accept | accepted`: `1`
  - `surfaced_same_action | rejected | strict_retry_selector`: `2`
  - `unsurfaced | not_invoked | not_invoked`: `2`
- `relative_clause_tail`
  - `surfaced_same_action | strict_direct_accept | accepted`: `1`
  - `unsurfaced | not_invoked | not_invoked`: `1`
- `that_clause_tail`
  - `surfaced_same_action | rejected | strict_retry_selector`: `1`
- `comparison_tail`
  - `unsurfaced | not_invoked | not_invoked`: `1`

#### Full-pipeline

`tail_type × surface_state × accept_mode × rejection_stage`

- `purpose_tail`
  - `surfaced_same_action | postnorm_salvaged_accept | accepted`: `3`
  - `surfaced_same_action | rejected | strict_retry_selector`: `3`
  - `unsurfaced | not_invoked | not_invoked`: `2`
- `continuation_tail`
  - `surfaced_same_action | strict_direct_accept | accepted`: `2`
  - `surfaced_same_action | rejected | strict_retry_selector`: `2`
  - `unsurfaced | not_invoked | not_invoked`: `1`
- `relative_clause_tail`
  - `surfaced_same_action | strict_direct_accept | accepted`: `1`
  - `unsurfaced | not_invoked | not_invoked`: `1`
- `that_clause_tail`
  - `unsurfaced | not_invoked | not_invoked`: `1`
- `comparison_tail`
  - `unsurfaced | not_invoked | not_invoked`: `1`

### Interpretation

- The widened corpus confirms the earlier split:
  - `purpose_tail` is still weak, but not purely because of rejection; it mixes surfaced salvage, surfaced rejection, and unsurfaced rows.
  - `continuation_tail` is now better grounded with `5` rows and shows a clear lane-specific pattern.
- `continuation_tail` is now the more direct next behavior target:
  - style-only: mixed `selector_reject` and `not_invoked`
  - full-pipeline: mixed `selector_reject`, `not_invoked`, and some direct accepts
- `relative_clause_tail` remains promising but under-sampled.
- `that_clause_tail` and `comparison_tail` are still not ready for behavior work.

### Continuation 5-Row Debug Set

The fixed continuation debug set now has `5` rows:

1. `Lecture 11::block-0657`
   - both lanes accepted
   - positive control
2. `Lecture 16::block-0078`
   - both lanes `unsurfaced -> not_invoked`
   - manual label: `acceptable_absorption`
3. `Lecture 16::block-0101`
   - style-only: `not_invoked`
   - full-pipeline: `selector_reject`
   - manual labels: `detector_miss` / `selector_reject`
4. `Lecture 16::block-0124`
   - style-only: `selector_reject`
   - full-pipeline: direct accept
   - strongest lane-divergence row
5. `Lecture 18::block-0211`
   - both lanes `selector_reject`
   - current strict candidate collapses to empty tail

### Next Step

- Keep deterministic salvage restricted to `purpose_tail`.
- If behavior changes resume, they should target `continuation_tail` first:
  - selector-stage continuation candidate quality
  - not-invoked vs acceptable-absorption split
