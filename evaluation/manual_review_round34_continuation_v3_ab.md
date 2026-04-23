## Round34 Continuation v3 A/B

This round introduced an experimental prompt/example profile only:

- `fragment_preserving_v3`

It does **not** change the structure contract.
The change is limited to stricter continuation-tail `restore_missing_tail` guidance and 3 new local-meaning restoration examples.

### A/B Inputs

- continuation harvest lane:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round33_style_only.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round33_full_pipeline.jsonl`
- continuation regression lane:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round20b.jsonl`

### Outputs

- continuation harvest:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round34_style_only_v3.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_probe_continuation200_round34_full_pipeline_v3.jsonl`
- continuation signature watchlist:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_continuation_tail_signature_watchlist_round34_v3.jsonl`
- regression lane:
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round34_style_only_v3.jsonl`
  - `/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_restore_missing_tail_replay_round34_full_pipeline_v3.jsonl`

### Watchlist Result

Stable-head continuation signature rows changed from:

- round33: `7`
- round34 v3: `2`

Round34 v3 distribution:

- by lane:
  - `full-pipeline`: `2`
- by stage:
  - `strict_retry_selector`: `1`
  - `strict_retry_overedit`: `1`
- by subtype:
  - `local_meaning_not_restored`: `1`
  - `null`: `1`

Remaining rows:

- `Stanford CS231N Deep Learning for Computer Vision   Spring 2025   Lecture 17 Robot Learning::block-0612`
  - `strict_retry_selector | local_meaning_not_restored`
- `Stanford CS231N   Spring 2025   Lecture 8 Attention and Transformers::block-0764`
  - `strict_retry_overedit`

### Regression Gate

The 5-row continuation regression lane stayed within the intended boundary:

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
  - full-pipeline: `strict_direct_accept`

No new `selector_reject` branch appeared inside the 5-row gate.
No new `surfaced_other_action` regression appeared inside the 5-row gate.

### Interpretation

This is enough evidence to keep `fragment_preserving_v3` as the better continuation-selector A/B profile.

The remaining continuation work is now narrower:

- one selector/local-meaning row
- one overedit row

The next decision is no longer “should continuation be reopened”.
It is whether to promote `fragment_preserving_v3` beyond A/B, or keep harvesting a bit longer before switching the default profile.
