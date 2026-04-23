## Round 42 Wrap Policy A/B

Target set:
- [cs231n_sp25_shipping_failure_corpus_round40_wrap_readability.jsonl](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round40_wrap_readability.jsonl:1)

Lane:
- frozen-block full-pipeline
- `SRT_ENGLISH_RESIDUAL_POLICY=technical_split`
- `SRT_PHASE1_TEMPERATURE=0.0`

Variants:
- baseline
- `SRT_WRAP_POLICY=cps_relaxed_v1`
- `SRT_MAX_CHARS_PER_LINE=28`

Outputs:
- [baseline](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_wrap_readability_round42_baseline.jsonl:1)
- [cps_relaxed_v1](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_wrap_readability_round42_cps_relaxed_v1.jsonl:1)
- [width28](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_wrap_readability_round42_width28.jsonl:1)

## Aggregate Result

Baseline:
- `post_wrap_failure = 3`
- `line_overflow = 2`
- `cps_overflow_severe = 1`

`cps_relaxed_v1`:
- `post_wrap_failure = 3`
- no measurable improvement over baseline on this 4-row set

`width28`:
- `post_wrap_failure = 2`
- `line_overflow = 1`
- `cps_overflow_severe = 1`

## Row-Level Read

### `Lecture 11 Large Scale Distributed Training::block-0083`

Baseline:
- `post_wrap_failure = line_overflow`

`width28`:
- failure cleared

Interpretation:
- this row responds to line-packing width
- it looks like a real line-packing problem, not a prompt problem

### `Lecture 12 Self-Supervised Learning::block-0528`

Baseline:
- no post-wrap failure in this controlled rerun

Interpretation:
- this row is unstable enough that it should not be used as the main wrap-policy anchor

### `Lecture 8 Attention and Transformers::block-0381`

All variants:
- `post_wrap_failure = cps_overflow_severe`

Interpretation:
- this is not a simple wrap-width problem
- likely needs either split-first handling or a stronger CPS-specific shipping rule

### `Lecture 10 Video Understanding::block-0404`

All variants:
- `post_wrap_failure = line_overflow`

Interpretation:
- width alone does not solve it
- this row is a single-cue numeric/model-name-heavy case and likely needs something beyond ordinary local re-wrap

## Conclusion

1. `cps_relaxed_v1` is not useful on the current wrap set.
2. `max_chars_per_line=28` has real leverage, but only on one subtype.
3. wrap/readability is still the right shipping target, but it is not one problem:
   - `0083` looks width-sensitive
   - `0381` looks CPS/split-sensitive
   - `0404` looks like a harder single-cue line-overflow case

## Next Practical Step

Do not globalize a wrap policy yet.

Next move should be narrower:
- keep `technical_split` as the shipping A/B baseline
- treat wrap rows by subtype rather than as one bucket
- if another wrap patch is opened, prefer a width-sensitive policy experiment over `cps_relaxed_v1`
