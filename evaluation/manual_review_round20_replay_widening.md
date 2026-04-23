## Round 20: Replay Corpus Widening

### Goal

Widen the `restore_missing_tail` runtime replay corpus beyond the round19 size while prioritizing:

- `purpose_tail`
- `continuation_tail`

### Probe Runs

#### Priority Probe

Built a priority-weighted frozen probe:

- `evaluation/cs231n_sp25_restore_missing_tail_probe_priority300.jsonl`

Composition:

- `purpose_tail`: `120`
- `continuation_tail`: `120`
- `relative_clause_tail`: `30`
- `that_clause_tail`: `15`
- `comparison_tail`: `15`

Ran style-only API eval:

- `evaluation/cs231n_sp25_restore_missing_tail_probe_priority300_round20_style_only.jsonl`

Surfaced `restore_missing_tail` rows:

- total: `10`
- `purpose_tail`: `6`
- `continuation_tail`: `2`
- `relative_clause_tail`: `1`
- `that_clause_tail`: `1`

#### Continuation-only Probe

Built an extra continuation-focused frozen probe:

- `evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200.jsonl`

Ran style-only API eval:

- `evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round20_style_only.jsonl`

Surfaced `restore_missing_tail` rows:

- total: `5`
- `continuation_tail`: `5`

### Replay Corpus Result

Rebuilt replay corpus with all prior inputs plus the two new probe outputs:

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round20b.jsonl`

Corpus size:

- round19 baseline: `12`
- round20 widened corpus: `17`

Tail-type composition:

- `purpose_tail`: `8`
- `continuation_tail`: `5`
- `relative_clause_tail`: `2`
- `that_clause_tail`: `1`
- `comparison_tail`: `1`

Historical outcomes in the widened replay corpus:

- `accepted`: `6`
- `rejected`: `11`

### Interpretation

- The main widening goal succeeded.
- More importantly, `continuation_tail` is no longer stuck at `3`; it is now at `5`.
- The corpus is still skewed toward `purpose_tail`, but it is now large enough to keep `purpose_tail` and `continuation_tail` as the two active interpretation targets.
- `that_clause_tail` and `comparison_tail` are still under-sampled.

### Next Step

- Use the widened corpus together with the fixed continuation-tail debug set.
- Do not broaden deterministic salvage beyond `purpose_tail`.
- The next analysis should read:
  - `tail_type × surface_state × accept_mode`
  - continuation debug labels
  together, before changing behavior.
