# Round 9: Style Selector and Strict Retry Payload

## Setup

- Inputs:
  - `evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl`
  - `evaluation/cs231n_sp25_eval_random40.jsonl`
- Command:
  - `python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl --output evaluation/cs231n_sp25_eval_hard40_translated_round9_style_selector.jsonl --frozen-blocks --phase1-temperature 0.0 --prompt-profile fragment_preserving_v2`
  - `python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_random40.jsonl --output evaluation/cs231n_sp25_eval_random40_translated_round9_style_selector.jsonl --frozen-blocks --phase1-temperature 0.0 --prompt-profile fragment_preserving_v2`
- Code changes under test:
  - accepted-only `phase1_risk_flags`
  - separate `strict_retry_candidate_risk_flags`
  - style-specific strict-retry selector
  - offending cue/span payload for strict retry
  - cue-level + proposition-level duplicate detector

## Aggregate Summary

### Hard 40

- Round 8:
  - `style_retry_invoked = 2`
  - `style_retry_accepted = 1`
  - `repair_invoked = 5`
  - `repair_accepted = 1`
  - `smaller_block_fallback = 2`
- Round 9:
  - `style_retry_invoked = 5`
  - `style_retry_accepted = 4`
  - `repair_invoked = 4`
  - `repair_accepted = 0`
  - `smaller_block_fallback = 2`

Interpretation:
- style retry now catches more real duplicate/over-closure cases
- some failures that previously leaked into repair no longer do
- remaining failures are still mostly style/wording, not boundary routing

### Random 40

- Round 8:
  - `style_retry_invoked = 2`
  - `style_retry_accepted = 0`
  - `repair/fallback = 0`
  - `phase1_risk_flags = {'fragment_overclosure_risk': 2, 'duplicate_restatement_risk': 2}`
- Round 9:
  - `style_retry_invoked = 1`
  - `style_retry_accepted = 0`
  - `repair/fallback = 0`
  - `phase1_risk_flags = {'context_uncertain': 1, 'duplicate_restatement_risk': 1}`
  - `strict_retry_candidate_risk_flags = {'duplicate_restatement_risk': 1}`

Interpretation:
- random baseline stayed stable
- the accepted-only risk metric is cleaner now
- rejected strict candidates no longer pollute `phase1_risk_flags`

## Representative Improvements

### Duplicate-restatement cleanup

- `Lecture 11::block-0074`
- Round 8:
  - `그리고 진짜 핵심은 이 132개의 스트리밍 멀티프로세서, 즉 SM입니다.`
- Round 9:
  - `그리고 진짜 핵심은 이 132개의 스트리밍 멀티프로세서입니다.`
- Signal:
  - strict retry invoked and accepted

### Duplicate noun phrase removal

- `Lecture 12::block-0075`
- Round 8:
  - `다운스트림 작업, 다운스트림 목표는, 여러분이 관심 있는 응용 분야입니다.`
- Round 9:
  - `다운스트림 작업, 여러분이 관심 있는 응용 분야입니다.`
- Signal:
  - strict retry invoked and accepted

### Unsupported tail trimmed instead of closed

- `Lecture 16::block-0107`
- Round 8:
  - `셀프슈퍼바이즈드 학습 수업에서, 1단계에서 그 사전 학습을 하는 거죠,`
- Round 9:
  - `셀프슈퍼바이즈드 학습 수업에서, 1단계에서 그 사전 학습을 하는`
- Signal:
  - strict retry invoked and accepted

## Remaining Problem Case

- `Lecture 13::block-0482`
- Round 8:
  - `하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.`
- Round 9:
  - unchanged
- Signal:
  - `style_retry_invoked = True`
  - `style_retry_accepted = False`
  - `phase1_risk_flags = {'duplicate_restatement_risk': 1}`
  - `strict_retry_candidate_risk_flags = {'duplicate_restatement_risk': 1}`

Interpretation:
- the detector and strict path now isolate the duplicate problem cleanly
- candidate selection is no longer polluting accepted metrics
- the remaining bottleneck is candidate quality for repetition deletion, not routing

## Regression To Watch

- `Lecture 6::block-0291`
- Round 8:
  - `여기서 흔히 볼 수 있는 주황색 블록들은, 3x3 컨볼루션 레이어입니다. 컨볼루션 레이어로서,`
- Round 9:
  - `여기서 흔히 볼 수 있는 주황색 블록들은, 3x3 컨볼루션 레이어입니다. 즉, 이런 것들이 컨볼루션 레이어라는 겁니다,`
- Signal:
  - strict retry invoked but rejected
  - final block remained on the worse base Phase1 wording

Interpretation:
- the selector behaves correctly for the retry candidate itself
- but the base Phase1 prompt still occasionally produces a worse fragment before strict retry even runs
- next work should stay focused on strict retry input quality and repeat-tail deletion, not on boundary routing
