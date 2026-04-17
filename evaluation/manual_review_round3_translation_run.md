## Round 3: Translation-Side Re-evaluation After Block Lint/Repair

Scope:
- Ran `run_review_eval.py` on the same 40 hard-case records from `cs231n_sp25_eval_review_round1.jsonl`
- Used the current block builder, current lint/repair pass, current Phase1/Phase2 pipeline, and the configured OpenAI models
- This is still an adversarial eval slice, not a corpus prevalence estimate

Pipeline-signal summary over 40 reviewed records:
- `repair_invoked`: 13
- `repair_accepted`: 0
- `repair_rejected`: 13
- `smaller_block_fallback`: 9
- `post_wrap_failure`: 4
- Mean final CPS: `7.692`

Dominant failure signals:
- `anchor_loss`: 15
- `cps_overflow_severe`: 4
- `line_overflow`: 3
- `english_residual`: 2

Interpretation:
- The pipeline now runs end-to-end on the reviewed set.
- The dominant remaining mechanical issue is still block-boundary quality, showing up downstream as `anchor_loss`.
- Repair is being invoked, but in this hard-case slice it was rejected every time because the candidate rewrites were not strictly better or were over-edited.
- This means the current repair guardrails are conservative and working as designed, but also that upstream block quality is still the main lever.

Changed block spot-checks:
1. `Lecture 12 Self-Supervised Learning::block-0752`
   Result: improved.
   Source block is now locally complete and the Korean output is natural enough without repair.
2. `Lecture 3 Regularization and Optimization::block-0329`
   Result: boundary improved, translation still slightly awkward.
   The new block is structurally better, but the Korean output repeats the hedge (`특별히 좋다고...`) and still needs prompt/model tuning.
3. `Lecture 16 Vision and Language::block-0408`
   Result: improved.
   The incomplete trailing relative clause is gone and the Korean output is locally stable.
4. `Lecture 2 Image Classification with Linear Classifiers::block-0400`
   Result: still unresolved.
   The block remains a bad translation unit and the Korean output inherits that incompleteness.
5. `Lecture 3 Regularization and Optimization::block-0651`
   Result: still unresolved.
   The Q/A fragment is more stable than before, but the emitted Korean duplicates `네, 네.` and is still not a good subtitle unit.
6. `Lecture 5 Image Classification with CNNs::block-0676`
   Result: still unresolved.
   The current singleton block is cleaner on the right edge, but it still starts mid-comparison. Repair fired here and was correctly rejected.

Conclusion:
- The block lint/repair pass did help. It produced a few real fixes and reduced some obvious contamination.
- The next bottleneck is still boundary quality, not long-context.
- The next code change should target the remaining local misses:
  - incomplete numeric endings such as `taking 1/10`
  - explanation tails like `we end up with a 0`
  - mid-comparison starters such as `like dividing...`
  - short Q/A fragments that should prefer context carry-over over forced local closure
