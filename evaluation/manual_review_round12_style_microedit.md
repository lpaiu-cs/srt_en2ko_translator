# Round 12: Deterministic Micro-Edit + Action-Aware Style Retry

## Run setup

- Mode: frozen-block eval
- Repair: disabled
- Phase1 temperature: `0.0`
- Prompt profile: `fragment_preserving_v2`
- Inputs:
  - `evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl`
  - `evaluation/cs231n_sp25_eval_random40.jsonl`
- Outputs:
  - `evaluation/cs231n_sp25_eval_hard40_translated_round12_style_microedit.jsonl`
  - `evaluation/cs231n_sp25_eval_random40_translated_round12_style_microedit.jsonl`

## Execution notes

- A first `random40` run surfaced a runtime bug in the new `restore_missing_tail` selector path:
  - `_looks_like_duplicate_proposition` was referenced without import.
- The bug was fixed in `subtitle_translator/pipeline.py`, then both `hard40` and `random40` were rerun on the same build.
- No `429` rate-limit failure occurred in the final successful runs.

## Aggregate results

### hard40

- Records: `40`
- `style_retry_invoked`: `2`
- `style_retry_accepted`: `2`
- `style_retry_rejected`: `0`
- `repair_invoked`: `0`
- `smaller_block_fallback`: `3`
- `post_wrap_failure`: `3`

Action counts:

- Attempts:
  - `delete_repeat_local`: `2`
  - `drop_head_marker`: `2`
  - `trim_explanatory_tail`: `1`
- Accepts:
  - `delete_repeat_local`: `2`
  - `drop_head_marker`: `2`
  - `trim_explanatory_tail`: `1`

### random40

- Records: `40`
- `style_retry_invoked`: `1`
- `style_retry_accepted`: `1`
- `style_retry_rejected`: `0`
- `repair_invoked`: `0`
- `smaller_block_fallback`: `0`
- `post_wrap_failure`: `0`

Action counts:

- Attempts:
  - `drop_head_marker`: `1`
  - `restore_missing_tail`: `1`
- Accepts:
  - `drop_head_marker`: `1`
  - `restore_missing_tail`: `1`

## Interpretation

- The deterministic micro-edit pass is now taking care of span-local style fixes without sending them through Phase1 strict retry.
- The remaining strict retry traffic is small and accepted cleanly in this run.
- `random40` stayed stable with no repair or fallback traffic.
