# Round 14: Purpose-Tail Post-Normalization

## Setup

- Patch focus:
  - deterministic post-normalization for `restore_missing_tail|purpose_tail`
  - runtime replay builder for surfaced `restore_missing_tail` cases
  - style-only lane and full-pipeline lane re-run after the patch
- Eval mode:
  - frozen-block
  - phase1 temperature `0.0`
  - style-only: repair disabled
  - full-pipeline: repair enabled

## Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round14_style_only.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round14_full_pipeline.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round14_style_only.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round14_style_only.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round14_full_pipeline.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round14_full_pipeline.jsonl`

## Runtime replay set

- replay rows: `1`
- source: previously surfaced `restore_missing_tail` runtime case(s)
- current replay coverage is still sparse; this is a selector benchmark, not a broad stress set

### style-only replay lane

- `style_retry_invoked`: `1`
- `style_retry_accepted`: `1`
- `repair_invoked`: `0`
- accepted action:
  - `strict_retry.restore_missing_tail|purpose_tail = 1`

Final text:

- cue 1: `하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다.`
- cue 2: `실제로 진전을 이루기 위해`

Interpretation:

- The previous rejected purpose-tail candidate `실제로 진전을 이루기 위해서입니다.` is now salvaged by deterministic post-normalization.
- The accepted candidate keeps cue-local ownership intact and removes the over-closed copular ending.

### full-pipeline replay lane

- `style_retry_invoked`: `1`
- `style_retry_accepted`: `1`
- `repair_invoked`: `0`
- final text matches the style-only lane

Interpretation:

- The purpose-tail fix survives the shipping lane as-is.
- Repair is not involved in this target case.

## hard40 / random40 regression lanes

### style-only hard40

- `style_retry_invoked`: `2`
- `style_retry_accepted`: `2`
- `smaller_block_fallback`: `4`
- `post_wrap_failure`: `3`

Accepted actions:

- `micro_edit.drop_head_marker = 2`
- `micro_edit.trim_explanatory_tail = 1`
- `strict_retry.delete_repeat_local = 2`

Interpretation:

- The purpose-tail patch does not disturb the existing hard40 style fixes.
- No additional `restore_missing_tail` case surfaced in hard40 on this run.

### style-only random40

- `style_retry_invoked`: `1`
- `style_retry_accepted`: `1`
- `smaller_block_fallback`: `0`
- `post_wrap_failure`: `0`

Accepted actions:

- `micro_edit.drop_head_marker = 1`
- `strict_retry.restore_missing_tail|purpose_tail = 1`

Interpretation:

- The earlier round13 rejection on the random40 purpose-tail case is now resolved cleanly.

### full-pipeline hard40

- `style_retry_invoked`: `2`
- `style_retry_accepted`: `2`
- `repair_invoked`: `5`
- `repair_accepted`: `0`
- `smaller_block_fallback`: `3`
- `post_wrap_failure`: `3`

Interpretation:

- Shipping-lane behavior remains dominated by the same non-style issues as before.
- The purpose-tail patch does not create new repair traffic or new style regressions in hard40.

### full-pipeline random40

- `style_retry_invoked`: `1`
- `style_retry_accepted`: `1`
- `repair_invoked`: `0`
- `smaller_block_fallback`: `0`
- `post_wrap_failure`: `0`

Interpretation:

- The purpose-tail target case remains fixed in the shipping lane.

## Conclusion

- This round is a real targeted improvement.
- The remaining problem from round13 was correctly isolated as `restore_missing_tail|purpose_tail`, and the new deterministic post-normalization fixes that case without reopening boundary or protected-cue issues.
- The runtime replay set is still too small to serve as a broad benchmark by itself, so future replay harvesting should continue.
- The next useful step is to expand the runtime replay corpus for `restore_missing_tail` while leaving boundary and long-context untouched.
