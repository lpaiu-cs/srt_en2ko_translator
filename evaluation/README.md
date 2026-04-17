# Evaluation

This directory holds manually reviewable evaluation artifacts built from the real `cs231n_sp25/eng` SRT files.

## Files

- `cs231n_sp25_eval.jsonl`: sampled translation blocks for review.

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

## Review Tags

Use the `review.failure_tags` array for manual labeling:

- `grouping_error`
- `translation_error`
- `glossary_inconsistency`
- `wrap_readability`
- `context_inconsistency`

Add short notes in `review.notes`.

## Rebuild

```bash
python3 build_eval_set.py --input-dir cs231n_sp25/eng --output evaluation/cs231n_sp25_eval.jsonl --target-count 40
```

The sampler is heuristic. It is meant to surface likely failure cases, not to be a gold benchmark by itself.
