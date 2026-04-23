## Round 19: Continuation-Tail Lane-Diff Debug Set

### Goal

Freeze the three current `continuation_tail` replay rows as a small lane-diff debug set and attach direct manual labels:

- `selector_reject`
- `not_invoked`
- `acceptable_absorption`
- `detector_miss`

### Debug Set

- `evaluation/cs231n_sp25_continuation_tail_lane_debug_round19.jsonl`

Rows:

1. `Lecture 16::block-0101`
2. `Lecture 16::block-0124`
3. `Lecture 18::block-0211`

### Per-row Triage

#### `Lecture 16::block-0101`

- style-only:
  - `rejected -> surfaced_same_action -> rejected`
  - label: `selector_reject`
- full-pipeline:
  - `rejected -> unsurfaced -> not_invoked`
  - label: `acceptable_absorption`

Why:

- style-only still surfaces `restore_missing_tail`, but the strict candidate keeps the same proposition and does not preserve continuation shape.
- full-pipeline rewrites the idea into a single acceptable sentence: `모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.`

#### `Lecture 16::block-0124`

- style-only:
  - `accepted -> surfaced_same_action -> rejected`
  - label: `selector_reject`
- full-pipeline:
  - `accepted -> surfaced_same_action -> rejected`
  - label: `selector_reject`

Why:

- Both lanes still surface `restore_missing_tail`.
- Both reject at selector stage because the candidate keeps turning the continuation into an over-closed clause.
- This is the cleanest continuation-tail patch target.

#### `Lecture 18::block-0211`

- style-only:
  - `rejected -> surfaced_same_action -> rejected`
  - label: `selector_reject`
- full-pipeline:
  - `rejected -> unsurfaced -> not_invoked`
  - label: `detector_miss`

Why:

- style-only surfaces the action, but the strict candidate becomes empty and is rejected.
- full-pipeline no longer surfaces `restore_missing_tail`, yet the final output still leaves a duplicate proposition:
  `시각 장면의 풍부함 덕분에 아주 밀집된 장면 그래프를 만들 수 있습니다. 시각 장면의 풍부함입니다.`
- That makes this row more consistent with detector miss than acceptable absorption.

### Interpretation

- `continuation_tail` is no longer a recall-only problem.
- The current split is:
  - selector-stage rejection on surfaced rows
  - unsurfaced rows that must be triaged as either acceptable absorption or detector miss
- Within the current 3-row debug set:
  - `selector_reject`: 4 lane-cases
  - `acceptable_absorption`: 1 lane-case
  - `detector_miss`: 1 lane-case

### Next Step

- Use this 3-row set as the fixed continuation-tail debug lane before expanding replay corpus further.
- Do not broaden deterministic salvage.
