## Round 5: Carry-Context Precision Audit and Action Calibration

Scope:
- Audited the random-40 baseline entries that were marked `carry_context_only` in round 4
- Tightened `carry_context_only` so it no longer fires on `dependent_start` alone
- Re-ran:
  - reviewed hard-case 40
  - random baseline 40

Manual precision audit of round-4 random baseline:
- Flagged records in random-40 before calibration: 23
- Manual classification by direct inspection:
  - true positive: 10
  - borderline: 6
  - false positive: 7

Observed false-positive pattern:
- Most false positives were `dependent_start`-only cases that were locally understandable and did not need prompt-level carry-context treatment.
- Typical examples:
  - `encodes the input image into a representation`
  - `in a random order.`
  - `in practice, we derive analytical gradients.`
  - `used in different setups.`
  - `is larger and it contains all of the possible models`

Action calibration change:
- `dependent_start` remains a high-recall state label
- `carry_context_only` now fires only for stronger fragment classes:
  - `dependent_end`
  - `numeric_orphan`
  - `comparison_midstart`
  - `qa_fragment`

Effect on routing:
- Random baseline:
  - `carry_context_only`: `23 -> 6`
  - `repair_invoked`: stayed `0`
  - `smaller_block_fallback`: stayed `0`
- Hard-case reviewed 40:
  - `carry_context_only`: `31 -> 23`
  - `repair_invoked`: stayed `5`
  - `smaller_block_fallback`: `2 -> 3`

Interpretation:
- The precision problem was real.
- Tightening `carry_context_only` removed a large number of prompt-level interventions from the random baseline without causing repair/fallback regressions there.
- The hard-case slice did not collapse back to the pre-boundary-aware behavior, which means the calibration did not undo the main routing fix.
- There is a small hard-case tradeoff signal (`repair_accepted 1 -> 0`, `cps_overflow_severe 2 -> 4`) that should be judged with paired translation review rather than from counters alone.

Conclusion:
- The next step should stay focused on paired translation review, not recall expansion.
- `dependent_start` is currently better treated as descriptive state than as an automatic prompt intervention.
- `comparison_midstart` and `qa_fragment` should be validated on a small targeted stress set before any broader recall work.
