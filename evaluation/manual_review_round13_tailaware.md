# Round 13: Tail-Type-Aware Restore-Missing-Tail

## Setup

- Patch focus:
  - tail-type-aware strict retry guidance for `restore_missing_tail`
  - tail-type-aware selector rejection causes
  - protected-cue guard for `restore_missing_tail`
  - channel-split action metrics (`micro_edit` vs `strict_retry`)
- Eval mode:
  - frozen-block
  - repair disabled
  - phase1 temperature `0.0`

## Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_stress_round13b_tailaware.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round13b_tailaware.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round13b_tailaware.jsonl`

## Aggregate

### restore_missing_tail stress set

- records: `20`
- `style_retry_invoked`: `0`
- `style_retry_accepted`: `0`
- `smaller_block_fallback`: `2`
- `post_wrap_failure`: `3`

Observed actions:

- `micro_edit.drop_head_marker`: `1 accepted`

Interpretation:

- The source-only stress set does not yet surface many real `restore_missing_tail` failures.
- It is useful for tail-type coverage, but weak as a selector benchmark by itself.

### hard40

- records: `40`
- `style_retry_invoked`: `2`
- `style_retry_accepted`: `2`
- `style_retry_rejected`: `0`
- `smaller_block_fallback`: `4`
- `post_wrap_failure`: `3`

Channel-split actions:

- `micro_edit`
  - `drop_head_marker`: `2 accepted`
  - `trim_explanatory_tail`: `1 accepted`
- `strict_retry`
  - `delete_repeat_local`: `2 accepted`

Interpretation:

- The new tail-type-aware rules did not hurt the existing hard40 style fixes.
- No `restore_missing_tail` case surfaced in hard40 on this run.

### random40

- records: `40`
- `style_retry_invoked`: `1`
- `style_retry_accepted`: `0`
- `style_retry_rejected`: `1`
- `smaller_block_fallback`: `0`
- `post_wrap_failure`: `0`

Channel-split actions:

- `micro_edit`
  - `drop_head_marker`: `1 accepted`
- `strict_retry`
  - `restore_missing_tail`: `1 rejected`
  - tail rejection bucket: `restore_missing_tail|purpose_tail = 1`

Rejection cause:

- `restore_tail_overclosed_for_purpose`

## Key example

`Lecture 13 Generative Models 1::block-0482`

- base cue 2: `조금 더 구조가 필요합니다.`
- strict candidate cue 2: `실제로 진전을 이루기 위해서입니다.`
- final decision: reject strict candidate
- reason: `restore_tail_overclosed_for_purpose`

Interpretation:

- The system now preserves cue-local ownership correctly: cue 1 stayed protected and cue 2 remained the only editable target.
- The remaining problem is no longer routing or protected-cue leakage.
- It is now specifically a purpose-tail wording problem: the model restores the meaning, but still over-closes the fragment.

## Conclusion

- This patch improved diagnosability and control.
- `restore_missing_tail` failures are now classified by tail type instead of being lumped into generic candidate-quality failure.
- The next high-value patch should target `purpose_tail` wording specifically, not general style control or boundary logic.
