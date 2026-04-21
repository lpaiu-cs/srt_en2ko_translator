# Round 10: Style Actions and Repair-Off Micro Check

## Scope

- Full `hard40` / `random40` repair-off reruns were attempted with:
  - `--frozen-blocks`
  - `--phase1-temperature 0.0`
  - `--prompt-profile fragment_preserving_v2`
  - `--disable-repair`
- Those full reruns were heavily contaminated by repeated `429` responses, so they are not reliable as aggregate comparison runs.
- A smaller four-block micro-set was used to verify the new action taxonomy and debug trace fields:
  - `Lecture 12::block-0075`
  - `Lecture 16::block-0107`
  - `Lecture 6::block-0291`
  - `Lecture 13::block-0482`

## Code Changes Under Test

- `duplicate_restatement` strict-retry action split:
  - `delete_repeat_local`
  - `restore_missing_tail`
- protected cue indices added to strict retry payload
- offending-cue / offending-span trace added to eval artifact
- fragment-overclosure detector extended to catch unsupported discourse-marker heads such as `즉,`

## Micro-Set Findings

### 1. Local duplicate deletion still works

- `Lecture 12::block-0075`
- Style trace now isolates only cue `91` as offending and protects cue `92`.
- Preferred action:
  - `delete_repeat_local`
- Strict candidate:
  - `다운스트림 작업,`
- Final acceptance:
  - accepted when strict retry is available

Interpretation:
- local duplicate trimming remains compatible with protected-cue locking

### 2. Relative-clause tail now surfaces as a style warning

- `Lecture 6::block-0291`
- Before this patch, the discourse-marker hallucination was not reliably surfaced.
- Now the block is tagged with:
  - `fragment_overclosure`
  - offending cue `538`
  - offending span `즉,`
  - preferred action `keep_fragment_open`

Interpretation:
- the new detector catches unsupported discourse-marker openings in `dependent_end` blocks
- strict retry can now target this case directly instead of missing it completely

### 3. Duplicate + displaced tail is now routed differently

- `Lecture 13::block-0482`
- Before this patch, the block had no offending-cue detail and defaulted to duplicate deletion behavior.
- Now the trace records:
  - offending cue `1021`
  - protected cue `1020`
  - preferred action `restore_missing_tail`
  - source cue text `to actually make progress.`
- Strict candidate:
  - `실제로 진전을 이루기 위해서입니다.`

Interpretation:
- the system now distinguishes “delete the repeated local proposition” from “restore the missing local tail meaning”
- the remaining issue is candidate quality and selector acceptance, not missing routing/action information

### 4. Fragment overclosure still depends on API availability

- `Lecture 16::block-0107`
- The block was correctly traced as:
  - `fragment_overclosure`
  - offending cue `208`
  - preferred action `keep_fragment_open`
- But the strict retry itself failed with a transient `429`, so no clean accept/reject judgment is available from this micro run.

Interpretation:
- routing and trace are correct
- this micro run does not provide a clean model-quality read because the strict call never completed

## Conclusion

- The new action taxonomy is now visible in the runtime traces.
- `block-0482` is the clearest proof:
  - the system now recognizes it as `restore_missing_tail`, not just generic duplicate deletion.
- `block-0291` is also improved at the diagnostic layer:
  - unsupported `즉,` heads are now surfaced as `fragment_overclosure`.
- The next blocker is no longer missing routing metadata.
- The next blocker is:
  - strict candidate quality
  - and clean API evaluation without `429` contamination
