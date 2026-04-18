# Round 7 Style-Control Eval

This round used the live OpenAI API through the current `run_review_eval.py` pipeline.

## Commands

```bash
python3 run_review_eval.py \
  --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl \
  --output evaluation/cs231n_sp25_eval_hard40_translated_round7_style_control.jsonl \
  --frozen-blocks \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v1

python3 run_review_eval.py \
  --input evaluation/cs231n_sp25_eval_random40.jsonl \
  --output evaluation/cs231n_sp25_eval_random40_translated_round7_style_control.jsonl \
  --frozen-blocks \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v1
```

## Hard 40 Summary

- Rows: 40
- Phase1 retried: 0
- Repair invoked: 5
- Repair accepted: 0
- Repair rejected: 5
- Smaller-block fallback: 3
- Single-cue source fallback: 0
- Post-wrap failure: 3

Pre-wrap warning / failure counts:

- `dependent_end`: 21
- `dependent_start`: 18
- `english_residual`: 6
- `english_residual_warn`: 5
- `numeric_orphan`: 2
- `duplicate_restatement`: 2
- `cps_warn`: 1

Post-wrap counts:

- `line_overflow`: 2
- `cps_warn`: 1

Phase1 self-reported risk flags:

- `context_uncertain`: 1

## Random 40 Summary

- Rows: 40
- Phase1 retried: 0
- Repair invoked: 0
- Repair accepted: 0
- Repair rejected: 0
- Smaller-block fallback: 0
- Single-cue source fallback: 0
- Post-wrap failure: 0

Pre-wrap warning / failure counts:

- `dependent_start`: 20
- `dependent_end`: 6
- `english_residual_warn`: 1

Phase1 self-reported risk flags:

- `duplicate_restatement_risk`: 2

## Representative Remaining Failures

- Hard case `Lecture 4::block-0581` still duplicates a definition-like clause:
  - `국소 그래디언트와 상류 그래디언트입니다. 상류 그래디언트는 종종 그래디언트입니다.`
- Hard case `Lecture 6::block-0291` still over-closes with an explanatory ending:
  - `그래서 이 주황색 블록들은 여기서 흔한 것들인데, 3x3 컨볼루션 레이어입니다. 즉, 이런 것들이 컨볼루션 레이어라는 겁니다,`
- Random case `Lecture 9::block-0357` still repeats the same proposition:
  - `그리고 다시 이미지 해상도로 돌아가는 업샘플링 단계가 있습니다. 이미지 해상도로 돌아가는 단계입니다.`
- Random case `Lecture 13::block-0482` still repeats the sentence almost verbatim:
  - `하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.`

## Interpretation

- The run did not regress random-baseline routing: repair and fallback stayed at zero on random 40.
- The remaining issue is still Phase1 style behavior, not boundary routing.
- The new prompt reduced obvious fragment over-closure enough that no `fragment_overclosure` warning fired in this run, but duplicate restatement remains visible in both hard and random slices.
- Repair remains a marginal tool in this failure regime; the better next lever is stricter Phase1 prompt/examples or a same-model strict retry path for style warnings before Phase2.
