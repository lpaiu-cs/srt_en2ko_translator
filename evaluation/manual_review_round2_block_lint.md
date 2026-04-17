## Round 2: Block Lint/Repair Pass Re-evaluation

Scope:
- Re-checked the same 40 hard-case source blocks from `cs231n_sp25_eval_review_round1.jsonl`
- Compared them against the current `build_translation_blocks()` output after the block lint/repair pass
- This is a failure-taxonomy pass on an adversarial sample, not a corpus prevalence estimate

Headline:
- 34/40 reviewed blocks were unchanged
- 6/40 reviewed blocks changed their cue boundaries
- 3 changed blocks look clearly improved and are no longer marked low-confidence
- 3 changed blocks are still unresolved and remain low-confidence
- 28/40 blocks in the after-lint hard-case sample are still marked `block_lint.low_confidence=true`

Old grouping-error set vs current lint signal:
- 32 blocks had `grouping_error` in round 1
- 27 of those are still flagged as low-confidence by the new lint pass
- 5 are no longer flagged
- Of those 5, 3 look like genuine fixes and 2 look like misses that the current lint heuristics still fail to catch

Current lint reason counts in `cs231n_sp25_eval_after_lint.jsonl`:
- `dependency_start`: 18
- `dangling_end`: 13
- `orphan_numeric_fragment`: 2

Clear improvements:
1. `Lecture 12 Self-Supervised Learning::block-0752`
   Old: `[978, 979]` -> `are considered. / In order to implement actually we`
   New: `[979, 980]` -> `In order to implement actually we / use this we use the batch learning framework.`
   Assessment: old block mixed a completed sentence with a restart; new block removes that contamination.
2. `Lecture 3 Regularization and Optimization::block-0329`
   Old: `[568, 569]` -> `and you get 99.7% accuracy. / So clearly, it's not bad, but it's also, I wouldn't say,`
   New: `[569, 570]` -> `So clearly, it's not bad, but it's also, I wouldn't say, / particularly good.`
   Assessment: old block started after a full stop and ended unfinished; new block closes the hedge properly.
3. `Lecture 16 Vision and Language::block-0408`
   Old: `[782, 783, 784]` -> `The only parts that are trained are / these perceiver sampler components / and this cross-attention layer that`
   New: `[782, 783]` -> `The only parts that are trained are / these perceiver sampler components`
   Assessment: old block trailed into an incomplete relative clause; new block trims that tail.

Changed but still unresolved:
1. `Lecture 2 Image Classification with Linear Classifiers::block-0400`
   Old: `[511, 512]`
   New: `[512, 513]`
   Lint: `dependency_start`
   Assessment: improved the left edge, but the block still starts inside an `if` clause and ends unfinished.
2. `Lecture 3 Regularization and Optimization::block-0651`
   Old: `[1184, 1185]`
   New: `[1183, 1184, 1185]`
   Lint: `dependency_start`
   Assessment: adding the prior cue helps a bit, but the Q/A fragment is still awkward and not a stable translation unit.
3. `Lecture 5 Image Classification with CNNs::block-0676`
   Old: `[1425, 1426]`
   New: `[1425]`
   Lint: `dependency_start`
   Assessment: removing the dangling `that` from the right edge is useful, but the new singleton block still starts mid-comparison (`like dividing...`).

Likely lint misses:
1. `Lecture 3 Regularization and Optimization::block-0758`
   Still ends at `taking 1/10` and likely needs one more cue.
2. `Lecture 8 Attention and Transformers::block-0673`
   Still ends at `we end up with a 0` inside an explanation that likely needs one more cue.

Interpretation:
- The new block lint/repair pass is acting conservatively, which is the right default.
- It fixes some obvious local boundary contamination without forcing unsafe merges.
- The remaining failure mode is still block-boundary quality, not long-context.
- The next high-value step is to run the current translation pipeline on this same reviewed 40-block set and tag:
  - `translation_error`
  - `glossary_inconsistency`
  - `repair_invoked`
  - `smaller_block_fallback`
