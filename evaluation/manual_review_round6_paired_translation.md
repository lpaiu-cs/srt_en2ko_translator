## Round 6: Paired Translation Review

Scope:
- Reviewed the current round-5 translations on a paired sample made of:
  - 9 records that previously showed `anchor_loss` in the old hard-case run
  - 6 records that still carry `carry_context_only` in the calibrated random baseline
  - 10 unflagged control records from the same random baseline
- Review tags used:
  - `translation_error`
  - `awkward_local_closure`
  - `omission_addition`
  - `glossary_mismatch`
  - `english_residual`

## Cohort Summary

Previous `anchor_loss` cohort, 9 samples:
- clearly acceptable or improved: 6
- still awkward or wrong: 3
- main remaining pattern:
  - incomplete-fragment blocks still sometimes get over-closed with invented subject/completion

Random `carry_context_only` cohort, 6 samples:
- clearly acceptable: 2
- mixed or awkward: 4
- main remaining pattern:
  - the routing is now reasonable, but Phase1 still tends to smooth an unfinished clause into a locally complete Korean statement

Unflagged control cohort, 10 samples:
- clearly acceptable: 7
- awkward despite no boundary signal: 3
- main remaining pattern:
  - paraphrastic duplication or unnecessary second-clause restatement in otherwise normal blocks

Interpretation:
- The boundary-aware routing change is real. The old mechanical misrouting problem is much smaller now.
- But this paired review shows the next bottleneck is no longer lint taxonomy. It is translation behavior inside Phase1.
- Incomplete fragments are now being routed more sanely, but the model still sometimes:
  - over-closes an unfinished thought
  - adds a generic explanatory ending
  - repeats content in a second paraphrase

## Detailed Review

1. `Lecture 11 Large Scale Distributed Training::block-0158`
   Tags: `translation_error`, `awkward_local_closure`, `omission_addition`
   Note: incomplete English fragment is turned into a complete Korean statement with an invented subject and a filled-in ending.

2. `Lecture 2 Linear Classifiers::block-0719`
   Tags: none
   Note: numeric fragment is preserved cleanly enough; the old anchor-loss style failure is gone.

3. `Lecture 4 Backpropagation::block-0716`
   Tags: none
   Note: Korean keeps the unfinished local shape without forcing a wrong completion.

4. `Lecture 5 CNNs::block-0625`
   Tags: none
   Note: locally awkward source, but current Korean remains acceptable and does not collapse mechanically.

5. `Lecture 5 CNNs::block-0676`
   Tags: `translation_error`
   Note: `divide by 2` becomes a more awkward `2배씩 나누는`, which is not the cleanest rendering of the source relation.

6. `Lecture 8 Attention and Transformers::block-0673`
   Tags: none
   Note: previously fell into boundary/fallback trouble; current output is acceptable enough.

7. `Lecture 10 Video Understanding::block-0404`
   Tags: `english_residual`
   Note: content is locally fine, but this is still one of the few remaining mechanical problem blocks due residual English tokens and wrapping pressure.

8. `Lecture 16 Vision and Language::block-0106`
   Tags: none
   Note: output is locally stable and no obvious meaning drift shows up.

9. `Lecture 16 Vision and Language::block-0166`
   Tags: `awkward_local_closure`
   Note: the Korean closes the phrase a little too neatly for a trailing English fragment.

10. `Lecture 12 Self-Supervised Learning::block-0849`
    Tags: `translation_error`, `awkward_local_closure`, `omission_addition`
    Note: the unfinished source tail is over-completed and introduces unsupported wording.

11. `Lecture 12 Self-Supervised Learning::block-0894`
    Tags: none
    Note: short fragment is translated conservatively and reads fine.

12. `Lecture 2 Linear Classifiers::block-0567`
    Tags: `awkward_local_closure`
    Note: incomplete explanatory tail is converted into a complete “...라는 뜻입니다” style closure.

13. `Lecture 4 Backpropagation::block-0308`
    Tags: none
    Note: the conditional remains appropriately open.

14. `Lecture 7 RNNs::block-0646`
    Tags: `awkward_local_closure`
    Note: output is understandable but turns a running clause into a more closed explanatory sentence.

15. `Lecture 10 Video Understanding::block-0019`
    Tags: `translation_error`, `omission_addition`
    Note: the trailing “that you can ...” tail is effectively dropped.

16. `Lecture 12 Self-Supervised Learning::block-0338`
    Tags: none
    Note: dependent-start-only block is fine without prompt intervention.

17. `Lecture 12 Self-Supervised Learning::block-0603`
    Tags: none
    Note: clean control.

18. `Lecture 2 Linear Classifiers::block-0376`
    Tags: none
    Note: clean control.

19. `Lecture 3 Regularization::block-0142`
    Tags: `omission_addition`, `awkward_local_closure`
    Note: output adds an extra paraphrastic restatement that is not needed.

20. `Lecture 3 Regularization::block-0466`
    Tags: none
    Note: dependent-start-only block is acceptable without carry-context prompting.

21. `Lecture 3 Regularization::block-0583`
    Tags: none
    Note: clean enough despite low-confidence state label.

22. `Lecture 4 Backpropagation::block-0084`
    Tags: none
    Note: dependent-start-only block is acceptable without intervention.

23. `Lecture 4 Backpropagation::block-0665`
    Tags: none
    Note: clean control.

24. `Lecture 7 RNNs::block-0309`
    Tags: `omission_addition`, `awkward_local_closure`
    Note: output repeats the sentence ending unnecessarily.

25. `Lecture 8 Attention and Transformers::block-0064`
    Tags: `omission_addition`, `awkward_local_closure`
    Note: output adds a second redundant restatement of the same idea.

## Conclusion

What this round says:
- `carry_context_only` precision was worth calibrating. Many `dependent_start`-only blocks are fine without prompt intervention.
- The next bottleneck is not more boundary recall. It is Phase1 style control.
- The most important remaining prompt/model failures are:
  - avoid over-closing unfinished fragments
  - avoid unsupported filler completions
  - avoid redundant paraphrastic restatement in otherwise normal blocks

Recommended next move:
- Keep the current boundary routing structure.
- Do not widen `carry_context_only` yet.
- Tune Phase1 prompt/examples specifically against:
  - unfinished dependent tails
  - fragment-preserving Korean
  - no duplicate restatement
