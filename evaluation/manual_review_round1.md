# Manual Review Round 1

Reviewed file:

- `evaluation/cs231n_sp25_eval_review_round1.jsonl`

Scope:

- 40 sampled blocks from `cs231n_sp25/eng`
- Source-side manual review only
- Focused on `grouping_error`, `context_inconsistency`, and `wrap_readability`

Why source-side only:

- The current pipeline was not re-run against these blocks inside this review pass.
- Existing `.ko.srt` files are older prototype outputs and are not treated as ground truth for the current system.
- Because of that, `translation_error` and `glossary_inconsistency` were not assigned in this round.

## Round 1 Counts

- `grouping_error`: 32 / 40
- `context_inconsistency`: 19 / 40
- `wrap_readability`: 2 / 40
- control / locally acceptable blocks: 8 / 40

## Main Finding

The dominant failure mode in the sampled CS231n blocks is still block-boundary quality, not long-context inconsistency.

Typical bad cases:

- block starts with a subordinate clause or lowercase continuation
- block ends on `that`, `which`, `so`, or another dangling connector
- orphan numeric / acronym fragment becomes a standalone block
- short audience Q&A fragments create unstable cue balance

## Examples

- incomplete trailing clause:
  - `Why is that? ... so`
  - `... the first device that`
  - `... this cross-attention layer that`
- left-context dependency:
  - `that it's, first, general enough ...`
  - `of 13% or 0.13.`
  - `a non-stationary distribution.`
- likely front/tail instability:
  - `more? / Yeah, yeah.`
  - `to use? / The short answer to your question`

## Implication

The next highest-value step is not longer raw context. It is:

1. tighten block boundary rules further
2. rerun the current translator on this reviewed eval set
3. only then start assigning `translation_error` and `glossary_inconsistency`
