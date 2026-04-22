## Round 17: Replay Corpus Expansion To 10+

### Goal

Expand the runtime `restore_missing_tail` replay corpus to at least 10 rows without changing translation behavior.

### Data Collection

- Built a larger frozen probe set:
  - `evaluation/cs231n_sp25_restore_missing_tail_probe300.jsonl`
- Ran style-only API eval on the 300-row probe:
  - `evaluation/cs231n_sp25_restore_missing_tail_probe300_round17_style_only.jsonl`
- Rebuilt the replay corpus from prior translated eval outputs plus the new probe output:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round17.jsonl`

### Replay Corpus Size

- Replay corpus rows: `12`
- Prior round16 replay corpus rows: `4`

Tail-type composition:

- `purpose_tail`: `5`
- `continuation_tail`: `3`
- `relative_clause_tail`: `2`
- `that_clause_tail`: `1`
- `comparison_tail`: `1`

Historical harvested outcomes inside the replay corpus:

- `accepted`: `5`
- `rejected`: `7`

### Current-System Replay Evaluation

Replayed the 12-row frozen replay corpus through both lanes:

- Style-only:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round17_style_only_eval.jsonl`
- Full-pipeline:
  - `evaluation/cs231n_sp25_restore_missing_tail_replay_round17_full_pipeline_eval.jsonl`

#### Style-only

`tail_type × accept_mode`

- `purpose_tail | postnorm_salvaged_accept`: `1`
- `purpose_tail | rejected`: `2`
- `relative_clause_tail | strict_direct_accept`: `2`
- `continuation_tail | rejected`: `3`
- `that_clause_tail | rejected`: `1`

Notes:

- `purpose_tail` still splits into one salvaged accept and two rejects.
- `relative_clause_tail` now has two direct accepts.
- `continuation_tail` remains unresolved in this lane.

#### Full-pipeline

`tail_type × accept_mode`

- `purpose_tail | postnorm_salvaged_accept`: `1`
- `purpose_tail | rejected`: `3`
- `relative_clause_tail | strict_direct_accept`: `2`
- `continuation_tail | strict_direct_accept`: `1`
- `continuation_tail | rejected`: `1`

Notes:

- `purpose_tail` remains the most fragile tail type.
- `relative_clause_tail` stays clean in full-pipeline too.
- `continuation_tail` now has one direct accept, but is not yet stable.

### Interpretation

- This round achieved the operational goal: the runtime replay corpus is now above the `10`-row threshold.
- The replay corpus is no longer purpose-only; it now includes `relative_clause_tail`, `continuation_tail`, `that_clause_tail`, and `comparison_tail`.
- The main unresolved style action is still `restore_missing_tail`, with `purpose_tail` remaining the weakest tail type.
- `purpose_tail` deterministic salvage should stay narrowly scoped. There is still only one reliable `postnorm_salvaged_accept`.

### Next Step

- Keep harvesting replay rows until there is enough data to interpret `tail_type × accept_mode` more confidently, especially for `that_clause_tail`, `comparison_tail`, and `continuation_tail`.
- Do not broaden deterministic salvage beyond `purpose_tail` yet.
