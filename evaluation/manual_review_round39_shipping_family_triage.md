## Round 39 Shipping Family Triage

Worksets:
- [english_residual](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round37_english_residual.jsonl)
- [fallback_trigger](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round37_fallback_trigger.jsonl)
- [wrap_readability](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round37_wrap_readability.jsonl)

This is a row-level read of the round37 shipping failure corpus. It is not a new API benchmark; it is a family split and triage pass.

## Main Read

The broad shipping bottleneck is not continuation. It is a mixed shipping lane made of:
- coarse `english_residual` routing
- low repair utility on rows that probably should not have gone to repair first
- `smaller_block_fallback` on long unfinished dependent-end blocks
- post-wrap readability issues that look more like line-packing policy than prompt failures

## Family 1: english_residual

Rows: `5`

Rows in this family:
- `Lecture 11 Large Scale Distributed Training::block-0083`
- `Lecture 12 Self-Supervised Learning::block-0175`
- `Lecture 3 Regularization and Optimization::block-0704`
- `Lecture 10 Video Understanding::block-0404`
- `Lecture 14 Generative Models 2::block-0301`

### Read

Under the current CS231n preset, these rows are mostly **allowed technical carry-through**, not obvious “translate this English away” failures.

Examples:
- `GPT-4`
- `Adam`, `RMSProp`, `SGD`
- `AlexNet`, `VGG-16`, `GFLOPS`, `C3D`
- `GAN`, `DC-GAN`, `ConvNet`
- `binning` is closer to a glossary / term-policy decision than a generic repair target

### Implication

Current `english_residual` routing is too coarse for shipping use.

The next step here is not “make repair better at English residual.” It is:
- split `english_residual` into
  - `allowed_technical_carry_through`
  - `actual_residual_english`
- stop sending the first class into repair by default

## Family 2: fallback_trigger

Rows: `3`

Rows in this family:
- `Lecture 11 Large Scale Distributed Training::block-0083`
- `Lecture 3 Regularization and Optimization::block-0704`
- `Lecture 14 Generative Models 2::block-0301`

### Read

All three rows share the same broad shape:
- unfinished dependent-end block
- technical term carry-through present
- repair did not help
- row eventually still needed fallback or remained fallback-prone

### Common signature

Current broad signature is:
- `dependent_end`
- plus technical carry-through / `english_residual(_warn)`
- plus unfinished trailing clause / comma / non-final tail shape

### Implication

This does **not** argue for reopening boundary general logic.
It argues for a narrower shipping-lane experiment:

- for this signature, compare `repair-first` vs `preemptive split-first`

The likely next question is not “how do we repair these better?” but “should these rows skip repair and split earlier?”

## Family 3: wrap_readability

Rows: `3`

Rows in this family:
- `Lecture 11 Large Scale Distributed Training::block-0083`
- `Lecture 8 Attention and Transformers::block-0381`
- `Lecture 10 Video Understanding::block-0404`

### Read

These rows do not point first to a prompt problem.

They point to two readability subtypes:

1. `line_overflow`
   - numeric-heavy / proper-noun-heavy lines
   - example: `AlexNet / VGG-16 / GFLOPS / C3D`
   - example: long causal clause with `binning`

2. `cps_overflow_severe`
   - discourse is translated reasonably but the packed Korean still exceeds comfortable subtitle speed
   - example: `Lecture 8 Attention and Transformers::block-0381`

### Implication

This family should be treated as a wrap/readability policy target:
- line packing for numeric-heavy / model-name-heavy rows
- CPS-aware wrapping / possibly earlier split on long two-sentence rows

The next experiment here should be a wrap-policy A/B, not a new prompt-profile A/B.

## Practical Priority Order

1. `english_residual` policy split
   - separate allowed technical carry-through from actual residual English

2. `fallback_trigger` trigger study
   - inspect `repair-first` vs `split-first` on the shared dependent-end signature

3. `wrap_readability` A/B
   - line packing and CPS policy, especially for numeric-heavy and long discourse rows

## What Not To Reopen

Still out of scope for the next shipping pass:
- continuation behavior
- purpose-tail deterministic salvage beyond purpose-only
- long-context expansion
- broad boundary retuning

Those are not what round37/38/39 are pointing to.
