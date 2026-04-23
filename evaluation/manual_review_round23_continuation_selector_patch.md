## Round 23: Continuation Selector Patch

### Change

Added strict continuation-tail retry guidance and strict examples aimed at the `selector_reject | local_meaning_not_restored` branch.

Files changed:

- `subtitle_translator/translators.py`

### Eval Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round23_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round23_full_pipeline_eval.jsonl`
- `evaluation/cs231n_sp25_continuation_tail_lane_debug_round23.jsonl`

### Continuation 5-row Result

#### Style-only

- `accepted`: `1`
- `selector_reject | empty_tail_collapse`: `1`
- `strict_retry_overedit`: `1`
- `not_invoked`: `2`

#### Full-pipeline

- `accepted`: `2`
- `selector_reject | local_meaning_not_restored`: `0`
- `not_invoked`: `3`

### What Improved

- `Lecture 16::block-0124`
  - before: selector reject
  - now: direct accept in both lanes

- `Lecture 16::block-0101`
  - before: full-pipeline selector reject
  - now: full-pipeline direct accept with
    `이미지와 텍스트의 연관성만으로요.`

This means the original target branch, `selector_reject | local_meaning_not_restored`, was reduced substantially.

### New / Remaining Issues

- `Lecture 11::block-0657`
  - style-only regressed into `strict_retry_overedit`
  - full-pipeline became `unsurfaced -> not_invoked`

- `Lecture 18::block-0211`
  - style-only still fails as `empty_tail_collapse`
  - full-pipeline still does not invoke `restore_missing_tail`

### Interpretation

- The targeted continuation selector branch improved.
- The remaining continuation problems are no longer centered on `local_meaning_not_restored`.
- The next continuation-specific issues are now:
  - `strict_retry_overedit`
  - `empty_tail_collapse`
  - `not_invoked` rows that still look like detector miss

### Conclusion

This patch was directionally successful for the intended branch, but it also surfaced a new tradeoff:

- better continuation-tail local restoration on some rows
- a regression on an accepted control row

So the next continuation patch should not broaden salvage. It should focus on preventing overedit / collapse while preserving the gains on `block-0124` and `block-0101`.
