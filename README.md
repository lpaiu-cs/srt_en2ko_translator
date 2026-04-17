# Translator

English SRT to Korean subtitle translator with cue-preserving output, batch processing, prompt configuration via `.env`, rolling context, and a reusable glossary log.

## Features

- Preserves original cue timing and cue indices.
- Groups adjacent cues into sentence-level translation units before re-splitting them.
- Reuses the previous translated sentence as lightweight discourse context.
- Maintains a JSONL glossary log for recurring terminology across files.
- Avoids splitting inside numeric separators such as `1,000` and `3.14` when redistributing text back into cues.
- Supports single-file and folder-based batch translation.

## Repository Layout

- `srt_en2ko_translator.py`: single-file CLI entrypoint for one SRT.
- `batch_translate_srt.py`: folder-based batch runner.
- `subtitle_translator/`: refactored core package.
- `translation_artifacts/`: generated glossary logs and other local artifacts.

## Setup

1. Create a virtual environment if you want isolation.
2. Install dependencies:

```bash
python3 -m pip install requests
```

3. Copy `.env.example` to `.env` and fill in at least `OPENAI_API_KEY`.

## Environment Variables

- `OPENAI_API_KEY`: required API key.
- `SRT_OPENAI_MODEL`: default model when `--model` is omitted. Defaults to `gpt-4.1-mini`.
- `SRT_TRANSLATION_CONTEXT`: optional domain/context hint for the translator.
- `SRT_TRANSLATION_STYLE`: optional tone/style hint for the translator.
- `SRT_USE_PREVIOUS_CONTEXT`: `true` by default. Reuses the last translated sentence as context.
- `SRT_GLOSSARY_LOG_PATH`: glossary JSONL log path. Defaults to `translation_artifacts/glossary.jsonl`.
- `SRT_GLOSSARY_MAX_TERMS`: number of relevant glossary entries injected per request. Defaults to `12`.
- `SRT_REQUEST_TIMEOUT`: request timeout in seconds. Defaults to `120`.

## CS231n Preset

The previously hard-coded prompt values were:

- `SRT_TRANSLATION_CONTEXT=These subtitles are the Stanford CS231n lecture on computer vision and deep learning.`
- `SRT_TRANSLATION_STYLE=Translate in a spoken, explanatory lecture style (like a professor talking to students). Use polite Korean sentence endings consistently (e.g. 습니다체; '~입니다', '~할 수 있습니다'), but allow occasional softer variations such as '~하는 거죠', '~라는 겁니다' to sound natural in lecture context.`

`.env.example` now includes those values directly so you can keep the old behavior by copying it as-is.

## Single File Usage

```bash
python srt_en2ko_translator.py input.srt -o output.ko.srt --model gpt-4.1-mini
```

Useful flags:

- `--batch-size 16`
- `--repeat-fill`
- `--glossary-log-path translation_artifacts/cs231n.jsonl`
- `--disable-history-context`

## Batch Usage

```bash
python batch_translate_srt.py ./cs231n_sp25/eng --model gpt-4.1-mini --repeat-fill --skip-existing --recursive
```

Batch runs reuse one translator instance and one glossary log, which improves terminology consistency across a lecture series.

## Refactoring Notes

Core logic is now split into small modules:

- `config.py`: `.env` loading and runtime config
- `srt_io.py`: SRT parsing and writing
- `grouping.py`: sentence grouping heuristics
- `splitting.py`: weighted time-based re-splitting
- `glossary.py`: JSONL glossary persistence
- `translators.py`: model adapter and response parsing
- `pipeline.py`: orchestration and fallback behavior

This keeps the CLI thin and makes it easier to swap the translation backend later for self-hosted or deployable models.
