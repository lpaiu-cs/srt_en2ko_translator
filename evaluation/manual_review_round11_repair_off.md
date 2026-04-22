Round 11 reran the frozen-block API eval with Phase2 repair disabled to isolate Phase1 and strict style retry.

Command profile
- `--frozen-blocks`
- `--phase1-temperature 0.0`
- `--prompt-profile fragment_preserving_v2`
- `--disable-repair`

429 status
- No `429` rate-limit error was observed in the rerun.
- The micro-set check, `hard40`, and `random40` all completed and wrote output files successfully.

Outputs
- `evaluation/cs231n_sp25_eval_micro_style_actions_round11_repair_off_check.jsonl`
- `evaluation/cs231n_sp25_eval_hard40_translated_round11_repair_off.jsonl`
- `evaluation/cs231n_sp25_eval_random40_translated_round11_repair_off.jsonl`

Hard 40 summary
- Records: `40`
- `style_retry_invoked`: `6`
- `style_retry_accepted`: `4`
- `style_retry_rejected`: `1`
- `repair_invoked`: `0` because repair was disabled
- `smaller_block_fallback`: `3`
- `post_wrap_failure`: `4`
- Remaining failure reasons: `english_residual=4`, `line_overflow=2`
- Style actions surfaced: `delete_repeat_local=3`, `keep_fragment_open=3`

Random 40 summary
- Records: `40`
- `style_retry_invoked`: `4`
- `style_retry_accepted`: `2`
- `style_retry_rejected`: `2`
- `repair_invoked`: `0` because repair was disabled
- `smaller_block_fallback`: `0`
- `post_wrap_failure`: `0`
- Style actions surfaced: `keep_fragment_open=2`, `delete_repeat_local=1`, `restore_missing_tail=1`

Key observations
- The rerun confirms that the current bottleneck is not boundary routing or Phase2 repair. Repair was fully off and the main moving part was still Phase1 plus strict retry.
- `Lecture 13 Generative Models 1::block-0482` now routes through `restore_missing_tail` and the strict retry is accepted, but the final text still sounds repetitive. The action taxonomy is now correct, while candidate quality remains imperfect.
- `Lecture 6 CNN Architectures::block-0291` still shows the unsupported discourse-marker tail (`즉, ...`) pattern and the strict retry does not win there yet.
- `random40` staying at zero fallback with repair off suggests the current style controls are not destabilizing ordinary blocks.
