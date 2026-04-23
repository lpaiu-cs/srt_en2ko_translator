## Round 41 English Residual Policy A/B

Goal:
- reduce shipping-lane repair traffic caused by technical carry-through being routed as `english_residual`

Policy under test:
- `SRT_ENGLISH_RESIDUAL_POLICY=technical_split`

Behavior:
- keep current coarse behavior as baseline
- downgrade allowed technical carry-through to warning-only (`english_residual_technical`)
- still keep actual residual English as repair-eligible `english_residual`

This is a routing policy A/B, not a prompt-profile A/B.

## Manual Label Basis

The round37 shipping failure corpus was manually labeled in:
- [manual_review_round40_shipping_labels.md](/Users/lpaiu/study/25-2/Translator/evaluation/manual_review_round40_shipping_labels.md:1)

That 6-row read suggested:
- most current `english_residual` shipping rows were actually technical carry-through
- broad shipping failures were being over-routed into repair

## Broad Full-Pipeline Comparison

### hard40

Round37 baseline:
- `repair_invoked = 6`
- `repair_accepted = 1`
- `repair_rejected = 5`
- `smaller_block_fallback = 3`
- `post_wrap_failure = 3`
- `failure_reasons.english_residual = 6`

Round40 `technical_split`:
- `repair_invoked = 3`
- `repair_accepted = 1`
- `repair_rejected = 2`
- `smaller_block_fallback = 1`
- `post_wrap_failure = 4`
- `failure_reasons.english_residual = 0`
- `pre_wrap_failures.english_residual_technical = 9`

### random40

Round37 baseline:
- `style_retry_invoked = 2`
- `repair_invoked = 0`
- no fallback/post-wrap failures

Round40 `technical_split`:
- `style_retry_invoked = 1`
- `repair_invoked = 0`
- no fallback/post-wrap failures

No obvious shipping-lane regression surfaced here.

### heldout10_internal

Round37 baseline:
- `repair_invoked = 2`
- `repair_accepted = 0`
- `repair_rejected = 2`
- `smaller_block_fallback = 1`
- `post_wrap_failure = 1`
- `failure_reasons.english_residual = 3`

Round40 `technical_split`:
- `repair_invoked = 1`
- `repair_accepted = 0`
- `repair_rejected = 1`
- `smaller_block_fallback = 0`
- `post_wrap_failure = 1`
- `failure_reasons.english_residual = 0`
- `pre_wrap_failures.english_residual_technical = 3`

## Shipping Failure Corpus Comparison

Round37 shipping corpus:
- [cs231n_sp25_shipping_failure_corpus_round37.jsonl](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round37.jsonl:1)
- unique rows: `6`
- families:
  - `english_residual = 5`
  - `fallback_trigger = 3`
  - `wrap_readability = 3`

Round40 shipping corpus:
- [cs231n_sp25_shipping_failure_corpus_round40_technical_split.jsonl](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round40_technical_split.jsonl:1)
- unique rows: `4`
- families:
  - `english_residual = 1`
  - `fallback_trigger = 1`
  - `wrap_readability = 4`

Round40 family splits:
- [english_residual](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round40_english_residual.jsonl:1)
- [fallback_trigger](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round40_fallback_trigger.jsonl:1)
- [wrap_readability](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round40_wrap_readability.jsonl:1)

## What Actually Changed

Rows that stopped going to repair:
- `Lecture 12 Self-Supervised Learning::block-0175`
  - `GPT-4` carry-through is now treated as technical warning, not repair
- `Lecture 3 Regularization and Optimization::block-0704`
  - `Adam / RMSProp / SGD` carry-through no longer forces repair
- `Lecture 14 Generative Models 2::block-0301`
  - `GAN / DC-GAN / ConvNet` carry-through no longer forces repair

Rows that still remain:
- `Lecture 11 Large Scale Distributed Training::block-0083`
  - still mixed `fallback_trigger + wrap_readability`
- `Lecture 8 Attention and Transformers::block-0381`
  - still `wrap_readability`
- `Lecture 10 Video Understanding::block-0404`
  - still `wrap_readability`, plus repair was still not useful
- new round40 row:
  - `Lecture 12 Self-Supervised Learning::block-0528`
  - `english_residual + wrap_readability`
  - dominant live issue is post-wrap readability, not repair traffic
  - the remaining English is `end-to-end`, so this row is not strong evidence that repair-first should come back

## Read

This A/B supports the shipping interpretation:

1. splitting technical carry-through out of `english_residual` reduces repair traffic
2. the next broad bottleneck shifts even more clearly toward wrap/readability
3. fallback also improves when coarse English residual routing is removed

But it is **not** yet enough to promote blindly as a global default without caution:
- `post_wrap_failure` rose from `3 -> 4` on `hard40`
- a new wrap-heavy row appeared in `hard40`
- full-pipeline still runs with normal shipping randomness, so some single-row drift should be read carefully

## Recommendation

The policy A/B is directionally good.

Immediate next target:
- treat `wrap_readability` as the next focused shipping-lane workset
- inspect whether the remaining `english_residual` row is a real residual-English case or another technical carry-through miss
- keep continuation frozen

Conservative promotion policy:
- do not reopen continuation
- keep deterministic salvage purpose-only
- if this policy is promoted, do it because repair routing gets cleaner, not because prompt behavior changed
