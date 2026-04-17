## Round 4: Boundary-Aware Lint + Carry-Context Re-evaluation

Scope:
- Added finer lint labels at the block layer: `dependent_start`, `dependent_end`, `numeric_orphan`, `comparison_midstart`, `qa_fragment`
- Added `carry_context_only` as a lint action
- Injected block lint state into Phase1 input and prompt
- Removed boundary-driven issues from Phase2 repair routing
- Re-ran:
  - reviewed hard-case 40
  - random baseline 40

Hard-case comparison, before vs after:
- Previous hard-case run (`round2` translated eval):
  - `repair_invoked`: 13
  - `repair_accepted`: 0
  - `smaller_block_fallback`: 9
  - dominant failure: `anchor_loss: 15`
- Current boundary-aware hard-case run:
  - `repair_invoked`: 5
  - `repair_accepted`: 1
  - `smaller_block_fallback`: 2
  - dominant remaining failures:
    - `english_residual: 4`
    - `line_overflow: 2`
    - `cps_overflow_severe: 2`

What changed:
- The old coarse `anchor_loss` bucket disappeared from the hard-case slice.
- The same problematic blocks are now exposed as lint-state instead of being treated as repair-time failures:
  - `dependent_end`: 21
  - `dependent_start`: 20
  - `numeric_orphan`: 2
- `carry_context_only` was attached to 31/40 hard-case blocks.

Mapping of previous `anchor_loss` cases:
- Previous unique `anchor_loss` records: 9
- Current lint mapping across those 9 records:
  - `dependent_start`: 7
  - `dependent_end`: 6
  - `numeric_orphan`: 1
  - `carry_context_only`: 9

Random baseline 40:
- `repair_invoked`: 0
- `smaller_block_fallback`: 0
- `post_wrap_failure`: 0
- Remaining signals are lint-state only:
  - `dependent_start`: 20
  - `dependent_end`: 6
  - `english_residual_warn`: 1
- `carry_context_only` was attached to 23/40 random blocks.

Interpretation:
- The patch did what it was supposed to do.
- Boundary-driven problems are no longer misrouted into Phase2 repair.
- The hard-case slice still contains many fragmentary blocks, but they are now carried as explicit boundary state into Phase1 instead of inflating fallback and repair traffic.
- The random baseline stayed clean, which is the main evidence that this did not obviously damage ordinary cases.

Notable hard-case outcomes:
1. `Lecture 5 CNNs::block-0676`
   Old behavior: singleton fragment with `anchor_loss`-style routing and rejected repair.
   New behavior: block widened to `[1424, 1425]`, marked `dependent_start + carry_context_only`, no repair, no fallback.
2. `Lecture 8 Attention and Transformers::block-0673`
   Old behavior: `anchor_loss` + fallback.
   New behavior: marked `dependent_end + carry_context_only`, no repair, no fallback.
3. `Lecture 3 Regularization::block-0758`
   Old behavior: incomplete numeric tail not explicitly surfaced.
   New behavior: marked `dependent_end + carry_context_only`, translated without repair/fallback.

Remaining issues:
- Some hard-case Korean outputs are still awkward even when the routing is now correct.
- `dependent_start` / `dependent_end` is now high-recall but still coarse; `comparison_midstart` and `qa_fragment` did not meaningfully surface in this slice after local repair.
- Repair still has low utility on the hard-case slice; it is now mostly reserved for real readability and residual-English problems, which is the intended direction.
