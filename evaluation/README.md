# Evaluation

This directory holds manually reviewable evaluation artifacts built from the real `cs231n_sp25/eng` SRT files.

## Files

- `cs231n_sp25_eval.jsonl`: sampled translation blocks for review.
- `*_translated*.jsonl`: pipeline replay outputs with translated cues, pipeline signals, and provenance for manual comparison.

## JSONL Schema

Each line contains:

- `id`: stable block identifier.
- `source_file`: absolute source SRT path.
- `lecture`: lecture filename stem.
- `block_index`: block order inside that lecture.
- `cue_indices`: original cue ids for that block.
- `source_text`: normalized joined source text.
- `source_cues`: cue-level timing and text.
- `tags`: heuristic sampling tags.
- `tag_scores`: category scores used only for sampling.
- `review`: manual annotation stub.

Translated replay JSONL also includes:

- `current_block`: the exact block that was translated in that run.
- `translation_output`: translated cues and joined translated text.
- `pipeline_signals`: retry/repair/fallback signals plus captured `phase1_risk_flags`.
- `provenance`: `phase1_model`, `repair_model`, temperatures, `prompt_profile`, `git_sha`, and whether `--frozen-blocks` was used.

Action-oriented replay JSONL also includes:

- `pipeline_signals.style_action_*`: aggregate action counts.
- `pipeline_signals.style_action_*_by_channel`: the same counts split into `micro_edit` and `strict_retry`.
- `pipeline_signals.style_action_accept_modes*`: whether a strict-retry acceptance was direct or salvaged by post-normalization.

## Review Tags

Use the `review.failure_tags` array for manual labeling:

- `translation_error`
- `awkward_local_closure`
- `omission_addition`
- `glossary_mismatch`
- `english_residual`

If you need source-side boundary diagnosis, keep it in `review.notes` or in a separate source-review pass instead of mixing it into the translation tags.

Add short notes in `review.notes`.

## Rebuild

```bash
python3 build_eval_set.py --input-dir cs231n_sp25/eng --output evaluation/cs231n_sp25_eval.jsonl --target-count 40
```

The sampler is heuristic. It is meant to surface likely failure cases, not to be a gold benchmark by itself.

## Replay

Dynamic replay against the current block builder:

```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated.jsonl
```

Frozen-block prompt A/B replay:

```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated_frozen.jsonl --frozen-blocks --phase1-temperature 0.0 --prompt-profile fragment_preserving_v2
```

Use frozen-block mode when the variable under test is prompt/profile/model behavior rather than block-boundary changes.

Two replay lanes are useful and should be kept separate:

- `style-only`: `--frozen-blocks --disable-repair --phase1-temperature 0.0`
- `full-pipeline`: frozen or dynamic replay with repair enabled

Do not compare the two lanes as if they were the same benchmark. The style-only lane isolates Phase1 + deterministic micro-edit + strict retry behavior; the full-pipeline lane measures shipping behavior.

`restore_missing_tail` also has a dedicated stress-set builder:

```bash
python3 build_restore_tail_stress_set.py \
  --input-dir cs231n_sp25/eng \
  --output evaluation/cs231n_sp25_restore_missing_tail_stress.jsonl \
  --per-tail-type 4
```

There is also a runtime replay builder for real `restore_missing_tail` cases that actually surfaced in previous translated eval outputs:

```bash
python3 build_restore_tail_replay_set.py \
  --inputs \
    evaluation/cs231n_sp25_eval_random40_translated_round11_repair_off.jsonl \
    evaluation/cs231n_sp25_eval_random40_translated_round12_style_microedit.jsonl \
    evaluation/cs231n_sp25_eval_random40_translated_round13b_tailaware.jsonl \
  --output evaluation/cs231n_sp25_restore_missing_tail_replay.jsonl
```

This replay set is not a coverage set. It is meant to benchmark actual `restore_missing_tail` selector decisions with preserved offending/protected cue metadata.
Each replay row also keeps a `replay_trace` snapshot with offending/protected cue ids, base Phase1 offending cue text, raw strict candidate text, post-normalization edits, final offending cue text, and accept/reject outcome.
Replayed outputs should now also be read with `surface_state`:

- `surfaced_same_action`
- `surfaced_other_action`
- `unsurfaced`

Use `tail_type × accept_mode × surface_state` together. Acceptance alone can be misleading when a historical replay row no longer surfaces under the same action in the current build.
Replay outputs also expose:

- `replay_transition`: `historical_outcome -> current surface_state -> current accept_mode`
- `style_retry_rejection_stage`: coarse current-lane stage such as `strict_retry_selector`, `strict_retry_generation`, `strict_retry_postwrap`, `strict_retry_overedit`, or `not_invoked`

For larger frozen evals where synchronous rate limits add noise, use the Batch lane:

```bash
python3 run_review_eval_batch.py prepare-phase1 \
  --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl \
  --requests-out evaluation/batch/hard40_phase1_requests.jsonl \
  --manifest-out evaluation/batch/hard40_phase1_manifest.jsonl \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v2
python3 run_review_eval_batch.py submit --input-jsonl evaluation/batch/hard40_phase1_requests.jsonl
python3 run_review_eval_batch.py prepare-style-retry \
  --phase1-manifest evaluation/batch/hard40_phase1_manifest.jsonl \
  --phase1-output evaluation/batch/hard40_phase1_output.jsonl \
  --requests-out evaluation/batch/hard40_strict_requests.jsonl \
  --manifest-out evaluation/batch/hard40_retry_manifest.jsonl
python3 run_review_eval_batch.py finalize \
  --retry-manifest evaluation/batch/hard40_retry_manifest.jsonl \
  --strict-output evaluation/batch/hard40_strict_output.jsonl \
  --output evaluation/cs231n_sp25_eval_hard40_translated_batch.jsonl
```
