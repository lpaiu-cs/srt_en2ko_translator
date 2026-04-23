# Round 46: Final-Postwrap Shipping Filter

## Why

Round44 shipping corpus selection used:

- `repair_invoked && !repair_accepted`
- `smaller_block_fallback == true`
- `post_wrap_failure == true`

That catches useful shipping-lane work, but it also mixes:

- rows whose final emitted cues are still unresolved
- rows that had an intermediate post-wrap failure before repair/local re-wrap, but whose final emitted cues are now acceptable

After re-checking `block-0381` against the current `post_wrap_gate`, it turns out to be the second case:

- round44 record still shows `post_wrap_failures = {"cps_overflow_severe": 1}`
- but the **final translated cues** only produce `warning = cps_warn`, not a repair-level failure

So `block-0381` should not drive the next shipping behavior patch.

## Tooling

`build_shipping_failure_corpus.py` now supports:

```bash
python3 build_shipping_failure_corpus.py \
  --inputs \
    evaluation/cs231n_sp25_eval_hard40_translated_round44_full_pipeline_default28.jsonl \
    evaluation/cs231n_sp25_eval_heldout10_internal_translated_round44_full_pipeline_default28.jsonl \
  --output evaluation/cs231n_sp25_shipping_failure_corpus_round44_default28_final_postwrap.jsonl \
  --final-postwrap-only
```

This keeps only rows whose **final translated cues still fail** the current post-wrap gate.

## Result

Current HEAD, round44 full-pipeline runs:

- filtered corpus size: `1`
- remaining row:
  - `Lecture 10 Video Understanding::block-0404`
  - selection reasons: `post_wrap_failure`, `repair_rejected`
  - final post-wrap reason: `line_overflow`
  - family: `wrap_readability`

Rows removed by this filter:

- `Lecture 8 Attention and Transformers::block-0381`
  - intermediate `cps_overflow_severe` still appears in the record
  - final emitted cues are no longer repair-level unresolved
  - treat as `transient_wrap_issue_resolved_by_repair`, not as the next shipping target

## Implication

The next shipping-lane patch should not be a generic split-first rule for `0381`-like rows.

The current unresolved shipping target is now narrower:

- single-cue, technical-carry-through-heavy `line_overflow`
- `repair_rejected`
- no smaller-block fallback

That points to `block-0404`-style wrap/readability handling, not continuation and not broad fallback retuning.
