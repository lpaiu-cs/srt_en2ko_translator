# Round 8 Style Retry + Prompt v2 Eval

This round evaluated the new `fragment_preserving_v2` prompt profile together with same-model strict style retry.

## Commands

```bash
python3 run_review_eval.py \
  --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl \
  --output evaluation/cs231n_sp25_eval_hard40_translated_round8_style_retry_v2.jsonl \
  --frozen-blocks \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v2

python3 run_review_eval.py \
  --input evaluation/cs231n_sp25_eval_random40.jsonl \
  --output evaluation/cs231n_sp25_eval_random40_translated_round8_style_retry_v2.jsonl \
  --frozen-blocks \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v2
```

## Round 7 -> Round 8 Summary

### Hard 40

- `style_retry_invoked`: `0 -> 2`
- `style_retry_accepted`: `0 -> 1`
- `repair_invoked`: `5 -> 5`
- `repair_accepted`: `0 -> 1`
- `smaller_block_fallback`: `3 -> 2`
- `post_wrap_failure`: `3 -> 2`
- pre-wrap `duplicate_restatement`: `2 -> 1`
- pre-wrap `fragment_overclosure`: `0 -> 1`

### Random 40

- `style_retry_invoked`: `0 -> 2`
- `style_retry_accepted`: `0 -> 0`
- `repair_invoked`: `0 -> 0`
- `smaller_block_fallback`: `0 -> 0`
- `post_wrap_failure`: `0 -> 0`

## Representative Improvements

- Random `Lecture 9::block-0357` improved directly under `fragment_preserving_v2`:
  - round 7: `그리고 다시 이미지 해상도로 돌아가는 업샘플링 단계가 있습니다. 이미지 해상도로 돌아가는 단계입니다.`
  - round 8: `그리고 다시 이미지 해상도로 돌아가는 업샘플링 단계가 있습니다.`
- Hard `Lecture 6::block-0291` triggered strict style retry and improved:
  - round 7: `그래서 이 주황색 블록들은 여기서 흔한 것들인데, 3x3 컨볼루션 레이어입니다. 즉, 이런 것들이 컨볼루션 레이어라는 겁니다,`
  - round 8: `여기서 흔히 볼 수 있는 주황색 블록들은, 3x3 컨볼루션 레이어입니다. 컨볼루션 레이어로서,`

## Representative Non-Improvement

- Random `Lecture 13::block-0482` still repeats the same proposition even after strict retry was attempted and rejected:
  - round 7: `하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.`
  - round 8: `하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.`

## Interpretation

- The detector fix worked: `fragment_overclosure` is now observed in round 8 instead of being silently missed.
- `fragment_preserving_v2` helps directly on some duplicate-restatement cases even before strict retry.
- Same-model strict style retry is now real and can improve style behavior, but acceptance is still low.
- The remaining bottleneck is still Phase1 style behavior, especially repeated restatement that survives both the base prompt and the strict retry candidate.
- The next lever is not more boundary routing. It is a tighter strict-retry candidate policy and stronger contrastive examples for duplicate restatement.
