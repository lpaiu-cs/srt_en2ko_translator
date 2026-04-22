## Round 18: Replay Surface State

### Goal

Add `surface_state` to replay re-eval outputs so runtime replay rows can be interpreted as:

- `surfaced_same_action`
- `surfaced_other_action`
- `unsurfaced`

This should be read together with `tail_type × accept_mode`.

### Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round18_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round18_full_pipeline_eval.jsonl`

### Style-only

`tail_type × surface_state`

- `purpose_tail | surfaced_same_action`: `3`
- `purpose_tail | unsurfaced`: `2`
- `continuation_tail | surfaced_same_action`: `3`
- `relative_clause_tail | surfaced_same_action`: `2`
- `that_clause_tail | surfaced_same_action`: `1`
- `comparison_tail | unsurfaced`: `1`

`tail_type × surface_state × accept_mode`

- `purpose_tail | surfaced_same_action | postnorm_salvaged_accept`: `2`
- `purpose_tail | surfaced_same_action | rejected`: `1`
- `purpose_tail | unsurfaced | not_invoked`: `2`
- `continuation_tail | surfaced_same_action | strict_direct_accept`: `2`
- `continuation_tail | surfaced_same_action | rejected`: `1`
- `relative_clause_tail | surfaced_same_action | strict_direct_accept`: `1`
- `relative_clause_tail | surfaced_same_action | rejected`: `1`
- `that_clause_tail | surfaced_same_action | rejected`: `1`
- `comparison_tail | unsurfaced | not_invoked`: `1`

### Full-pipeline

`tail_type × surface_state`

- `purpose_tail | surfaced_same_action`: `4`
- `purpose_tail | unsurfaced`: `1`
- `continuation_tail | surfaced_same_action`: `3`
- `relative_clause_tail | surfaced_same_action`: `2`
- `that_clause_tail | surfaced_same_action`: `1`
- `comparison_tail | surfaced_same_action`: `1`

`tail_type × surface_state × accept_mode`

- `purpose_tail | surfaced_same_action | postnorm_salvaged_accept`: `2`
- `purpose_tail | surfaced_same_action | rejected`: `2`
- `purpose_tail | unsurfaced | not_invoked`: `1`
- `continuation_tail | surfaced_same_action | strict_direct_accept`: `1`
- `continuation_tail | surfaced_same_action | rejected`: `2`
- `relative_clause_tail | surfaced_same_action | strict_direct_accept`: `2`
- `that_clause_tail | surfaced_same_action | rejected`: `1`
- `comparison_tail | surfaced_same_action | rejected`: `1`

### Interpretation

- The replay corpus is now large enough to distinguish acceptance from surfacing.
- In this 12-row corpus, no historical `restore_missing_tail` row moved into `surfaced_other_action`.
- The current divergence is mostly between:
  - still surfacing under `restore_missing_tail`
  - no longer surfacing at all
- `purpose_tail` remains the weakest tail type, but the new lens shows that part of the problem is not just rejection; some historical rows are now `unsurfaced`.
- `relative_clause_tail` looks relatively stable because it keeps surfacing and reaches direct accepts.
- `continuation_tail` is unstable in a different way: it keeps surfacing, but acceptance varies by lane.

### Next Step

- Keep collecting runtime replay rows, but read them with `surface_state` first.
- The next style interpretation should prioritize `purpose_tail` and `continuation_tail`.
- Do not broaden deterministic salvage beyond `purpose_tail` yet.
