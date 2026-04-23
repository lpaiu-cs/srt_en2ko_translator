## Round 37 Broad Re-Baseline

Current HEAD was re-evaluated on broad shipping-lane benchmarks rather than continuation-only replay.

Benchmarks:
- `hard40`
- `random40`
- `heldout10_internal`

Lanes:
- `style-only`: frozen blocks, `--disable-repair`, `phase1_temperature=0.0`
- `full-pipeline`: frozen blocks, repair enabled, current HEAD defaults

Notes:
- `heldout10_internal` is a CS231n-internal fallback slice, not an external non-CS231n corpus.
- Continuation behavior remained frozen during this run.

### Aggregate Results

| benchmark | lane | style retry | repair | smaller-block fallback | post-wrap failure |
| --- | --- | ---: | ---: | ---: | ---: |
| hard40 | style-only | 2 invoked / 2 accepted | 0 | 4 | 3 |
| hard40 | full-pipeline | 2 invoked / 2 accepted | 6 invoked / 1 accepted | 3 | 3 |
| random40 | style-only | 1 invoked / 1 accepted | 0 | 0 | 0 |
| random40 | full-pipeline | 2 invoked / 2 accepted | 0 | 0 | 0 |
| heldout10_internal | style-only | 0 | 0 | 1 | 1 |
| heldout10_internal | full-pipeline | 0 | 2 invoked / 0 accepted | 1 | 1 |

### Main Read

1. Continuation is no longer the broad shipping bottleneck.
   The continuation-specific branch stayed quiet in broad evaluation and did not create visible shipping regressions.

2. The remaining broad bottlenecks are shipping-lane issues:
   - `english_residual`
   - `smaller_block_fallback`
   - `post_wrap_failure`
   - low `Phase2Repair` utility

3. `Phase2Repair` is still weak on broad shipping benchmarks.
   - `hard40 full-pipeline`: `repair_invoked=6`, `repair_accepted=1`
   - `heldout10_internal full-pipeline`: `repair_invoked=2`, `repair_accepted=0`

4. Random broad slices remain relatively clean.
   - `random40` had no repair traffic and no fallback/post-wrap failures in either lane.

### Failure Distribution

#### hard40
- dominant failure reasons:
  - `english_residual = 6`
  - `line_overflow = 2`
  - `cps_overflow_severe = 2`
- post-wrap failures:
  - `line_overflow = 2`
  - `cps_overflow_severe = 1`

#### heldout10_internal
- dominant failure reasons:
  - `english_residual = 3`
  - `line_overflow = 1`

### Interpretation

This run does **not** say the translator is production-grade or domain-general.
It says current HEAD has moved past continuation-tail micro-tuning as the immediate priority, and the next practical shipping target is the broader lane:
- repair usefulness
- fallback rate
- post-wrap stability
- residual English leakage

### Recommended Next Step

Do **not** reopen continuation behavior yet.

Next target:
1. inspect a small set of `hard40` / `heldout10_internal` rows where:
   - `repair_invoked = true && repair_accepted = false`
   - or `smaller_block_fallback = true`
   - or `post_wrap_failure = true`
2. classify whether the next shipping improvement should focus on:
   - `Phase2Repair` routing/utility
   - block sizing / fallback triggers
   - wrap/readability policy
   - English residual handling
