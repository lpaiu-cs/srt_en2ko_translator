# Round 15: Restore-Missing-Tail Accept Modes

## Setup

- Patch focus:
  - record `strict_direct_accept` vs `postnorm_salvaged_accept`
  - preserve replay rows with base/raw/post-normalized/final offending cue text
  - add negative tests for purpose-tail normalization variants
- Eval mode:
  - frozen-block
  - phase1 temperature `0.0`
  - both `style-only` and `full-pipeline`

## Outputs

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round15_style_only.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round15_full_pipeline.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round15_style_only.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round15_style_only.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round15_full_pipeline.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round15_full_pipeline.jsonl`

## Replay lane result

- replay corpus rows: `1`
- tail type: `purpose_tail`
- style-only:
  - `style_retry_invoked = 1`
  - `style_retry_accepted = 1`
  - `strict_retry.restore_missing_tail|purpose_tail|postnorm_salvaged_accept = 1`
- full-pipeline:
  - `style_retry_invoked = 1`
  - `style_retry_accepted = 1`
  - `strict_retry.restore_missing_tail|purpose_tail|postnorm_salvaged_accept = 1`

Key trace:

- base offending cue: `조금 더 구조가 필요합니다.`
- raw strict candidate: `실제로 진전을 이루기 위해서입니다.`
- post-normalized candidate: `실제로 진전을 이루기 위해`
- final cue: `실제로 진전을 이루기 위해`

Interpretation:

- This acceptance is not a direct strict-retry win.
- It is a `postnorm_salvaged_accept`, which is exactly the distinction that was missing in round14.

## hard40 / random40

### style-only

- hard40:
  - `style_retry_invoked = 2`
  - `style_retry_accepted = 2`
  - accepted modes: `strict_retry.delete_repeat_local|strict_direct_accept = 2`
  - `smaller_block_fallback = 3`
  - `post_wrap_failure = 2`
- random40:
  - `style_retry_invoked = 1`
  - `style_retry_accepted = 1`
  - accepted modes: `strict_retry.restore_missing_tail|purpose_tail|postnorm_salvaged_accept = 1`
  - `smaller_block_fallback = 0`
  - `post_wrap_failure = 0`

### full-pipeline

- hard40:
  - `style_retry_invoked = 2`
  - `style_retry_accepted = 2`
  - accepted modes: `strict_retry.delete_repeat_local|strict_direct_accept = 2`
  - `repair_invoked = 5`
  - `repair_accepted = 0`
  - `smaller_block_fallback = 3`
  - `post_wrap_failure = 3`
- random40:
  - `style_retry_invoked = 1`
  - `style_retry_accepted = 1`
  - accepted modes: `strict_retry.restore_missing_tail|purpose_tail|postnorm_salvaged_accept = 1`
  - `repair_invoked = 0`
  - `smaller_block_fallback = 0`
  - `post_wrap_failure = 0`

Interpretation:

- `delete_repeat_local` remains a direct strict-retry action.
- `restore_missing_tail|purpose_tail` is currently succeeding through deterministic salvage, not through raw strict-candidate quality alone.
- full-pipeline still has the same non-style bottlenecks in hard40; this round does not solve those.

## Conclusion

- The main gain in round15 is observability.
- We can now distinguish:
  - direct strict-retry acceptance
  - post-normalization salvage acceptance
- For the currently surfaced `purpose_tail` case, the system is succeeding via `postnorm_salvaged_accept`.
- The next useful step is still replay-corpus expansion, not boundary or long-context changes.
