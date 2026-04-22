# Translator

English SRT to Korean subtitle translator with cue-preserving output, structured model responses, bounded repair, and batch processing.

## Features

- Preserves original cue timing and cue indices.
- Builds small dynamic translation blocks instead of translating the whole sentence at once.
- Separates `context window`, `translation unit`, and `emission unit`.
- Uses Phase1 structured output for exact cue-count preservation.
- Applies deterministic quality gates before and after line wrapping.
- Tries local re-wrap strategies before escalating post-wrap failures to repair or smaller-block fallback.
- Sends only failed blocks to a bounded Phase2 repair model.
- Caps retry, repair, and recursive split depth to avoid oscillating fallback behavior.
- Supports reusable glossary logs with `hard` and `soft` term modes.
- Writes per-file JSONL metrics for tuning prompt, block size, and repair policy.
- Supports single-file and folder-based batch translation.

## Pipeline

1. Build a translation block of 2-4 cues using punctuation, gap, duration, and source-length limits.
2. Send the block to Phase1 and require structured output with `emitted_cues` and `risk_flags`.
3. Validate schema and cue structure. If structure fails, retry Phase1 and then fall back to smaller blocks.
4. Run a pre-wrap gate for alignment, glossary, anchor, English residual, and front/tail concentration checks.
5. Apply line wrapping, then try local re-wrap strategies if the first wrap fails post-wrap checks.
6. Run a post-wrap gate for line overflow, CPS, and bad line break checks.
7. If needed, send only that block to Phase2 repair with a bounded rewrite prompt.
8. If the block still fails, split it into smaller blocks instead of using proportional redistribution.
9. Stop recursive fallback when split depth is exhausted or the same failure signature repeats.

## Repository Layout

- `srt_en2ko_translator.py`: single-file CLI entrypoint.
- `batch_translate_srt.py`: folder-based batch runner.
- `subtitle_translator/`: core package.
- `translation_artifacts/`: local glossary logs and other generated artifacts.

## Setup

1. Create a virtual environment if you want isolation.
2. Install dependencies:

```bash
python3 -m pip install requests
```

3. Copy `.env.example` to `.env` and fill in at least `OPENAI_API_KEY`.

## Environment Variables

- `OPENAI_API_KEY`: required API key.
- `SRT_PHASE1_MODEL`: default Phase1 model. Defaults to `gpt-4.1-mini`.
- `SRT_OPENAI_MODEL`: backward-compatible alias for `SRT_PHASE1_MODEL`.
- `SRT_REPAIR_MODEL`: default Phase2 repair model. Defaults to `gpt-4o`.
- `SRT_PHASE1_TEMPERATURE`, `SRT_REPAIR_TEMPERATURE`: generation temperatures. For eval A/B runs, `0.0` reduces comparison noise.
- `SRT_PHASE1_PROMPT_PROFILE`: Phase1 prompt/example profile. Defaults to `fragment_preserving_v2`. Keep `fragment_preserving_v1` for frozen baseline comparisons.
- `SRT_TRANSLATION_CONTEXT`: optional domain/context hint.
- `SRT_TRANSLATION_STYLE`: optional tone/style hint.
- `SRT_USE_CONTEXT_WINDOW`: whether to provide left/right source context.
- `SRT_ENABLE_REPAIR`: enables bounded Phase2 repair.
- `SRT_PHASE1_MAX_RETRIES`, `SRT_PHASE2_MAX_REPAIRS`, `SRT_MAX_SPLIT_DEPTH`: retry and recursion guardrails.
- `SRT_GLOSSARY_LOG_PATH`: glossary JSONL log path.
- `SRT_METRICS_LOG_PATH`: per-file JSONL metrics log path.
- `SRT_GLOSSARY_MAX_TERMS`: max glossary entries injected per request.
- `SRT_ALLOWED_ENGLISH_TERMS`: comma-separated technical terms allowed to remain in output.
- `SRT_REQUEST_TIMEOUT`: request timeout in seconds.
- `SRT_REQUEST_MAX_ATTEMPTS`, `SRT_REQUEST_BACKOFF_MIN_SECONDS`, `SRT_REQUEST_BACKOFF_MAX_SECONDS`: synchronous debug/eval retry backoff for rate limits and transient server errors.
- `SRT_BLOCK_MIN_CUES`, `SRT_BLOCK_MAX_CUES`, `SRT_BLOCK_MAX_DURATION_MS`, `SRT_BLOCK_MAX_SOURCE_CHARS`, `SRT_BLOCK_MAX_GAP_MS`: block builder controls.
- `SRT_MAX_CHARS_PER_LINE`, `SRT_MAX_LINES_PER_CUE`, `SRT_MAX_CPS`: readability thresholds.

## CS231n Preset

The old hard-coded lecture preset is now expressed through `.env`:

- `SRT_TRANSLATION_CONTEXT=These subtitles are the Stanford CS231n lecture on computer vision and deep learning.`
- `SRT_TRANSLATION_STYLE=Translate in a spoken lecture style for Korean subtitles. Use polite sentence-final endings only when the source thought is actually complete. For unfinished fragments, incomplete clauses, or carry-context blocks, non-final fragment endings are acceptable and preferred. Do not add generic explanatory endings that are not directly supported by the source.`

## Single File Usage

```bash
python srt_en2ko_translator.py input.srt -o output.ko.srt --model gpt-4.1-mini --repair-model gpt-4o
```

Useful flags:

- `--block-max-cues 4`
- `--glossary-log-path translation_artifacts/cs231n.jsonl`
- `--disable-context-window`
- `--openai-base-url ...`

## Batch Usage

```bash
python batch_translate_srt.py ./cs231n_sp25/eng --model gpt-4.1-mini --repair-model gpt-4o --skip-existing --recursive
```

Batch runs reuse one translator instance and one glossary log, which helps keep terminology consistent across a lecture series.

## Evaluation

Build a real review set from the CS231n Spring 2025 English SRT files:

```bash
python3 build_eval_set.py --input-dir cs231n_sp25/eng --output evaluation/cs231n_sp25_eval.jsonl --target-count 40
```

Replay the current pipeline against a reviewed set:

```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated.jsonl
```

For prompt A/B runs, freeze the original block boundaries and lower Phase1 temperature:

```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated_frozen.jsonl --frozen-blocks --phase1-temperature 0.0 --prompt-profile fragment_preserving_v2
```

For larger frozen-block evals, keep synchronous replay for micro debugging and use the Batch lane for clean rate-limit isolation:

```bash
python3 run_review_eval_batch.py prepare-phase1 \
  --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl \
  --requests-out evaluation/batch/hard40_phase1_requests.jsonl \
  --manifest-out evaluation/batch/hard40_phase1_manifest.jsonl \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v2
```

Then upload/create the batch, download the phase1 output, prepare the strict-retry batch, and finalize:

```bash
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

The translated eval JSONL records provenance (`phase1_model`, `repair_model`, temperatures, `prompt_profile`, `git_sha`) together with block lint state and captured Phase1 risk flags. Current manual review tags are centered on translation behavior rather than the older grouping-only pass:

- `translation_error`
- `awkward_local_closure`
- `omission_addition`
- `glossary_mismatch`
- `english_residual`

Use those tags in each entry's `review.failure_tags`, and note whether the issue came from a frozen-block A/B run or a dynamic-block replay.

## Core Modules

- `config.py`: `.env` loading and runtime config.
- `blocks.py`: dynamic block building and context window selection.
- `srt_io.py`: SRT parsing and writing.
- `grouping.py`: sentence grouping heuristics used for context windows.
- `glossary.py`: glossary persistence and retrieval.
- `quality.py`: structure validation plus pre-wrap/post-wrap gates.
- `translators.py`: model adapter and structured output handling.
- `pipeline.py`: retry, repair, split fallback, and final orchestration.
