## Round 25: Continuation Adverb-Anchor Patch

### Change

Adjusted strict continuation-tail guidance to avoid repeating the same adverbial anchor already carried by the protected cue.

Files changed:

- `subtitle_translator/translators.py`

### Eval Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round25_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round25_full_pipeline_eval.jsonl`
- `evaluation/cs231n_sp25_continuation_tail_lane_debug_round25.jsonl`

### Continuation 5-row Result

#### Style-only

- `strict_direct_accept`: `3`
- `not_invoked`: `2`
- `selector_reject`: `0`

#### Full-pipeline

- `strict_direct_accept`: `3`
- `not_invoked`: `2`
- `selector_reject`: `0`

### What Improved

- `Lecture 11::block-0657`
  - before: style-only `strict_retry_overedit`, full-pipeline `not_invoked`
  - now: direct accept in both lanes with
    `여러분의 신경망을 다른 방식으로,`

- `Lecture 16::block-0124`
  - before: full-pipeline regressed to duplicated `바로 바로`
  - now: direct accept in both lanes with
    `즉시 사용할 수 있으면 좋겠죠.`

- `Lecture 18::block-0211`
  - before: style-only `empty_tail_collapse`
  - now: direct accept in both lanes with
    `시각 장면 자체의.`

### Remaining Issues

- `Lecture 16::block-0078`
  - both lanes remain `unsurfaced -> not_invoked`
  - still reads as acceptable absorption rather than a clear restore miss

- `Lecture 16::block-0101`
  - both lanes remain `unsurfaced -> not_invoked`
  - still looks closer to detector/invocation miss than selector weakness

### Interpretation

- The continuation-tail selector branch is no longer the main bottleneck on this 5-row debug slice.
- The previous `local_meaning_not_restored`, `strict_retry_overedit`, and `empty_tail_collapse` cases were resolved on the replay rows that surfaced.
- The remaining continuation-tail issue is now concentrated in `not_invoked` rows, which still need manual triage between acceptable absorption and detector miss.

### Conclusion

This patch meaningfully reduced continuation-tail selector failures without broadening deterministic salvage beyond `purpose_tail`.
The next continuation-specific work should move from selector tuning to `not_invoked` triage and invocation logic.
