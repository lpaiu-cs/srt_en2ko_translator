## Round 38 Shipping Failure Corpus

This corpus is built from broad **full-pipeline** outputs, not continuation replay.

Inputs:
- `evaluation/cs231n_sp25_eval_hard40_translated_round37_full_pipeline.jsonl`
- `evaluation/cs231n_sp25_eval_heldout10_internal_translated_round37_full_pipeline.jsonl`

Selection rule:
- `repair_invoked && !repair_accepted`
- or `smaller_block_fallback == true`
- or `post_wrap_failure == true`

Builder:

```bash
python3 build_shipping_failure_corpus.py \
  --inputs \
    evaluation/cs231n_sp25_eval_hard40_translated_round37_full_pipeline.jsonl \
    evaluation/cs231n_sp25_eval_heldout10_internal_translated_round37_full_pipeline.jsonl \
  --output evaluation/cs231n_sp25_shipping_failure_corpus_round37.jsonl
```

Output:
- `evaluation/cs231n_sp25_shipping_failure_corpus_round37.jsonl`

## Result

The shipping failure corpus contains **6 unique rows**.

Benchmark membership:
- `hard40 = 6`
- `heldout10_internal = 2`

Two rows appear in both benchmark slices and are merged by stable `id`:
- `Lecture 10 Video Understanding::block-0404`
- `Lecture 14 Generative Models 2::block-0301`

## Failure Families

Family counts over unique rows:
- `english_residual = 5`
- `fallback_trigger = 3`
- `wrap_readability = 3`

Selection-reason counts over unique rows:
- `repair_rejected = 5`
- `smaller_block_fallback = 3`
- `post_wrap_failure = 3`

## Current Read

This corpus supports the current broad-shipping interpretation:

1. continuation is no longer the next broad shipping target
2. the dominant remaining family is `english_residual`
3. smaller-block fallback and post-wrap readability remain secondary but real shipping issues
4. low repair utility is still visible because `repair_rejected` dominates the corpus

## Rows

1. `Lecture 11 Large Scale Distributed Training::block-0083`
   - families: `english_residual`, `fallback_trigger`, `wrap_readability`
   - selection reasons: `repair_rejected`, `smaller_block_fallback`, `post_wrap_failure`

2. `Lecture 12 Self-Supervised Learning::block-0175`
   - families: `english_residual`
   - selection reasons: `repair_rejected`

3. `Lecture 3 Regularization and Optimization::block-0704`
   - families: `english_residual`, `fallback_trigger`
   - selection reasons: `repair_rejected`, `smaller_block_fallback`

4. `Lecture 8 Attention and Transformers::block-0381`
   - families: `wrap_readability`
   - selection reasons: `post_wrap_failure`

5. `Lecture 10 Video Understanding::block-0404`
   - benchmark membership: `hard40`, `heldout10_internal`
   - families: `english_residual`, `wrap_readability`
   - selection reasons: `repair_rejected`, `post_wrap_failure`

6. `Lecture 14 Generative Models 2::block-0301`
   - benchmark membership: `hard40`, `heldout10_internal`
   - families: `english_residual`, `fallback_trigger`
   - selection reasons: `repair_rejected`, `smaller_block_fallback`

## Next Step

Use this corpus as the narrow shipping-lane workset for the next redesign step:

- split `english_residual` into
  - allowed technical carry-through
  - actual residual English that should be translated or rewritten
- inspect whether `fallback_trigger` rows should split earlier instead of going through repair
- treat `wrap_readability` as a wrap policy problem before treating it as a prompt problem
