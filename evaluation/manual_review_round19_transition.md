## Round 19: Transition And Rejection Stage

### Goal

Add replay-level transition and rejection-stage fields so replay rows can be interpreted as:

- historical outcome
- current surface state
- current accept mode
- current rejection stage

### Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round19_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round19_full_pipeline_eval.jsonl`

### New Fields

Each replay row now includes:

- `pipeline_signals.replay_transition`
- `pipeline_signals.replay_current_accept_mode`
- `pipeline_signals.style_retry_rejection_stage`

### Transition Summary

#### Style-only

- `purpose_tail`
  - `accepted -> surfaced_same_action -> postnorm_salvaged_accept`: `1`
  - `accepted -> surfaced_same_action -> rejected`: `1`
  - `rejected -> surfaced_same_action -> rejected`: `1`
  - `rejected -> unsurfaced -> not_invoked`: `2`
- `continuation_tail`
  - `accepted -> surfaced_same_action -> rejected`: `1`
  - `rejected -> surfaced_same_action -> rejected`: `2`
- `relative_clause_tail`
  - `accepted -> surfaced_same_action -> strict_direct_accept`: `1`
  - `accepted -> unsurfaced -> not_invoked`: `1`
- `that_clause_tail`
  - `rejected -> surfaced_same_action -> rejected`: `1`
- `comparison_tail`
  - `rejected -> surfaced_same_action -> rejected`: `1`

#### Full-pipeline

- `purpose_tail`
  - `accepted -> surfaced_same_action -> postnorm_salvaged_accept`: `2`
  - `rejected -> surfaced_same_action -> postnorm_salvaged_accept`: `1`
  - `rejected -> surfaced_same_action -> rejected`: `1`
  - `rejected -> unsurfaced -> not_invoked`: `1`
- `continuation_tail`
  - `accepted -> surfaced_same_action -> rejected`: `1`
  - `rejected -> unsurfaced -> not_invoked`: `2`
- `relative_clause_tail`
  - `accepted -> surfaced_same_action -> strict_direct_accept`: `1`
  - `accepted -> surfaced_same_action -> rejected`: `1`
- `that_clause_tail`
  - `rejected -> surfaced_same_action -> rejected`: `1`
- `comparison_tail`
  - `rejected -> surfaced_same_action -> rejected`: `1`

### Continuation Lane Divergence

`continuation_tail` now reads more clearly:

- style-only:
  - all `3` rows still `surfaced_same_action`
  - all current failures are `style_retry_rejection_stage = strict_retry_selector`
- full-pipeline:
  - `1` row is still `surfaced_same_action -> rejected`
  - `2` rows moved to `unsurfaced -> not_invoked`

So the current divergence is not a post-wrap-stage issue. It is a split between:

- selector rejection in style-only
- no current `restore_missing_tail` surfacing in full-pipeline

### Purpose Unsurfaced Manual Read

Two style-only `purpose_tail` rows became `unsurfaced -> not_invoked`:

1. `Lecture 11::block-0178`
   - current output: `저는 K40을 손에 쥐어본 적이 있고, B100을 쥐어볼 기회는 없었습니다.`
   - judgment: likely acceptable without `restore_missing_tail`
   - interpretation: this looks closer to genuine resolution than detector miss

2. `Lecture 17::block-0198`
   - current output: `컴퓨터 비전에서 이미 보셨던 것인데, 인스턴스 분할을 시도하는 기술입니다.`
   - judgment: the purpose meaning is folded into a noun-phrase rewrite
   - interpretation: this also looks closer to acceptable reformulation than obvious detector miss

This means the current `purpose_tail` weakness is not purely missed detection. It is a mix of:

- rows that still surface and need salvage
- rows that no longer surface because the current phrasing absorbed the local tail meaning acceptably

### Interpretation

- `purpose_tail` remains the weakest tail type, but the weakness now splits into:
  - unstable surfaced acceptance
  - a smaller set of `unsurfaced -> not_invoked` rows
- `continuation_tail` is the next target, and its problem is now better defined:
  - style-only: selector-stage rejection
  - full-pipeline: some rows do not surface at all
- `relative_clause_tail` is still promising, but not yet stable enough to call solved

### Next Step

- Keep using `tail_type × surface_state × accept_mode`, not totals.
- Prioritize `purpose_tail` and `continuation_tail`.
- Do not broaden deterministic salvage beyond `purpose_tail` yet.
