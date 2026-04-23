# Round 47: Compact Technical Repair A/B

## Goal

Target the only remaining **final unresolved** shipping row from the round44 `final-postwrap-only` corpus:

- `Lecture 10 Video Understanding::block-0404`
- single cue
- `dependent_end`
- technical carry-through heavy (`AlexNet`, `VGG-16`, `C3D`, `GFLOPS`)
- `line_overflow`
- `repair_rejected`

This round did **not** reopen continuation or fallback logic. It tested a narrow Phase2Repair branch only.

## Patch

Added `SRT_REPAIR_POLICY=compact_technical_fragment_v1`.

This branch applies only when all of these are true:

- single-cue block
- `failure_reasons` includes `line_overflow`
- block lint includes `dependent_end`
- the source/Phase1 text is technical carry-through heavy

The compact repair prompt tells the model to:

- preserve technical names / numbers / units exactly
- compress repeated predicates into a compact list-like fragment
- keep the cue unfinished if the source cue is unfinished
- prefer commas and compact phrasing over repeating `...필요합니다`

Example target shape:

- bad: `AlexNet은 0.7 GFLOPS가 필요합니다. VGG-16은 약 13.6 GFLOPS가 필요하고, C3D는`
- good: `AlexNet은 0.7 GFLOPS, VGG-16은 13.6 GFLOPS, C3D는`

## Targeted Result

Input:

- `evaluation/cs231n_sp25_shipping_failure_corpus_round44_default28_final_postwrap.jsonl`

Runtime pinned for the A/B:

- `SRT_MAX_CHARS_PER_LINE=28`
- `SRT_ENGLISH_RESIDUAL_POLICY=technical_split`
- `SRT_WRAP_POLICY=baseline`

### Baseline repair

- output: `AlexNet은 0.7 GFLOPS가\n필요합니다. VGG-16은 약 13.6\nGFLOPS가 필요하고, C3D는`
- still fails final post-wrap (`line_overflow`)
- `repair_accepted = false`

### `compact_technical_fragment_v1`

- output: `AlexNet은 0.7 GFLOPS, VGG-16은\n13.6 GFLOPS, C3D는`
- passes final post-wrap under the pinned runtime thresholds
- `repair_accepted = true`
- `pipeline_signals.effective_repair_profile = compact_technical_fragment_v1`

## Broad Result

Re-ran:

- `hard40` full-pipeline
- `random40` full-pipeline
- `heldout10_internal` full-pipeline

with the same pinned runtime:

- `SRT_MAX_CHARS_PER_LINE=28`
- `SRT_ENGLISH_RESIDUAL_POLICY=technical_split`
- `SRT_WRAP_POLICY=baseline`
- `SRT_REPAIR_POLICY=compact_technical_fragment_v1`

### Aggregate

Compared with round44 `default28`:

- `hard40`
  - `repair_invoked: 2 -> 1`
  - `repair_accepted: 1 -> 1`
  - `repair_rejected: 1 -> 0`
  - `smaller_block_fallback: 0 -> 0`
  - `post_wrap_failure: 2 -> 2` in the raw row metrics, but the remaining `0381` signal is now only an intermediate/transient wrap issue
- `heldout10_internal`
  - `repair_invoked: 1 -> 1`
  - `repair_accepted: 0 -> 1`
  - `repair_rejected: 1 -> 0`
  - `post_wrap_failure: 1 -> 1` in raw row metrics, but no final unresolved row remains under the pinned runtime thresholds
- `random40`
  - still clean

## Final-Postwrap Interpretation

Important caveat:

- raw replay metrics still record intermediate `post_wrap_failure` counts
- for shipping decisions, the more relevant question is whether the **final translated cues** still fail the current post-wrap gate

When `build_shipping_failure_corpus.py --final-postwrap-only` is rerun under the same pinned runtime (`width28 + technical_split + baseline wrap`), the result is:

- `0` remaining rows across `hard40 + heldout10_internal`

So the compact repair patch appears to clear the last remaining **final unresolved** internal shipping row.

## What This Means

On the current internal CS231n shipping benchmarks, the main shipping-lane bottleneck is no longer continuation and is no longer the `0404` technical fragment row either.

The next step should not be another immediate shipping behavior patch. The higher-value next step is:

- external held-out generalization
- or observability cleanup so `final-postwrap-only` always evaluates against the row's runtime thresholds without relying on the current shell env
