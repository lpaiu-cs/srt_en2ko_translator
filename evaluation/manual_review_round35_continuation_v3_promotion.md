## Round35 Continuation v3 Narrow Promotion

This round does **not** promote `fragment_preserving_v3` as the global default prompt profile.

Instead, the runtime now auto-promotes only this narrow strict-retry branch onto `v3`:

- `preferred_action = restore_missing_tail`
- `source_tail_type = continuation_tail`
- single offending cue
- protected cue present

The global `SRT_PHASE1_PROMPT_PROFILE` default remains `fragment_preserving_v2`.

### Inputs

- replay lane:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round20b.jsonl`
- continuation harvest lane:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round33_style_only.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round33_full_pipeline.jsonl`

### Outputs

- replay:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round35_style_only_eval.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round35_full_pipeline_eval.jsonl`
- continuation harvest:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round35_style_only_eval.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round35_full_pipeline_eval.jsonl`
- canonical stable-head artifacts:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_regression_gate.json`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_signature_watchlist.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_signature_watchlist_local_meaning.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_signature_watchlist_overedit.jsonl`

### Regression Gate

Current gate, after narrow promotion:

- `block-0078`
  - style-only: `unsurfaced -> not_invoked -> acceptable_absorption`
  - full-pipeline: `unsurfaced -> not_invoked -> acceptable_absorption`
- `block-0101`
  - style-only: `strict_direct_accept`
  - full-pipeline: `strict_direct_accept`
- `block-0124`
  - style-only: `strict_direct_accept`
  - full-pipeline: `strict_direct_accept`
- `block-0211`
  - style-only: `strict_direct_accept`
  - full-pipeline: `strict_direct_accept`
- `block-0657`
  - style-only: `unsurfaced -> not_invoked -> acceptable_absorption`
  - full-pipeline: `unsurfaced -> not_invoked -> acceptable_absorption`

Interpretation:

- no new `selector_reject` branch appeared inside the 5-row gate
- no `surfaced_other_action` regression appeared
- `0657` shifted to acceptable absorption in full-pipeline instead of remaining direct accept, but it did **not** regress into a bad branch (`selector_reject` or `strict_retry_overedit`)

### Continuation Signature Watchlist

Round35 stable-head watchlist:

- total rows: `2`
- `strict_retry_selector | local_meaning_not_restored`: `1`
- `strict_retry_overedit`: `1`

Rows:

1. `Lecture 17 Robot Learning::block-0612`
   - `strict_retry_selector | local_meaning_not_restored`
2. `Lecture 8 Attention and Transformers::block-0764`
   - `strict_retry_overedit`

### Interpretation

This is enough to promote `fragment_preserving_v3` for the narrow continuation-selector branch only.

It is **not** enough to reopen continuation behavior generally.

Current policy after round35:

- keep global default prompt profile at `fragment_preserving_v2`
- keep purpose-tail deterministic salvage purpose-only
- treat round35 as the new continuation stable-head baseline
- keep harvesting continuation subtype watchlists separately
- do not reopen continuation behavior again until a stable-head subtype reaches the same evidence threshold again
