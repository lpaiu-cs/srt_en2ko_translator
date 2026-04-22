# Round 16: Runtime Replay Corpus Expansion

## Setup

- Goal:
  - expand the runtime replay corpus for surfaced `restore_missing_tail` cases
  - keep deterministic salvage limited to `purpose_tail`
  - re-evaluate the expanded replay corpus in both `style-only` and `full-pipeline`
- Inputs:
  - previous translated eval outputs
  - `evaluation/cs231n_sp25_restore_missing_tail_probe100_round16_style_only.jsonl`
- Eval mode:
  - `style-only`: frozen-block, repair disabled, phase1 temperature `0.0`
  - `full-pipeline`: frozen-block, repair enabled, phase1 temperature `0.0`

## Replay Corpus Growth

- previous replay corpus size: `1`
- new replay corpus size: `4`

Tail-type composition:

- `purpose_tail = 2`
- `continuation_tail = 1`
- `that_clause_tail = 1`

Files:

- `evaluation/cs231n_sp25_restore_missing_tail_probe100.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_probe100_round16_style_only.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round16.jsonl`

Interpretation:

- The corpus is no longer a single-case benchmark.
- It is still below the desired `10+` rows, but it now contains multiple tail types.

## Replay Eval

Outputs:

- `evaluation/cs231n_sp25_restore_missing_tail_replay_round16_style_only_eval.jsonl`
- `evaluation/cs231n_sp25_restore_missing_tail_replay_round16_full_pipeline_eval.jsonl`

### style-only replay eval

- `purpose_tail`
  - `postnorm_salvaged_accept = 1`
  - `rejected = 1`
- `continuation_tail`
  - no accepted `restore_missing_tail`
  - current run did not keep the action surfaced as a strict-retry acceptance
- `that_clause_tail`
  - no accepted `restore_missing_tail`

### full-pipeline replay eval

- `purpose_tail`
  - `postnorm_salvaged_accept = 1`
  - `rejected = 1`
- `that_clause_tail`
  - `rejected = 1`
  - rejection causes:
    - `restore_tail_that_clause_shape_missing`
    - `restore_tail_overclosed_for_that_clause`
- `continuation_tail`
  - no accepted `restore_missing_tail`

## Key Reading

`purpose_tail` is now split into two distinct cases:

1. `Lecture 13 ... block-0482`
   - accepted via `postnorm_salvaged_accept`
   - raw strict candidate is still over-closed
   - deterministic salvage converts it into a viable fragment
2. `Lecture 16 ... block-0069`
   - still rejected
   - current rejection cause: `restore_tail_purpose_marker_missing`

Interpretation:

- `purpose_tail` is not “solved”.
- The current system can salvage at least one surfaced purpose-tail case.
- Another surfaced purpose-tail case still fails before salvage is applicable.

## Conclusion

- This round expands the replay corpus and makes the next target clearer.
- `purpose_tail` remains the only tail type with an observed acceptance path, and that path is still `postnorm_salvaged_accept`, not direct strict-candidate quality.
- `that_clause_tail` and `continuation_tail` now exist in the replay corpus, but they are not ready for deterministic salvage.
- The next step is still replay-corpus growth until the corpus reaches at least `10` surfaced rows before widening deterministic salvage beyond `purpose_tail`.
