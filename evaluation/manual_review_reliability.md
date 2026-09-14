# Reliability and replay audit follow-up

Date: 2026-09-14

## Resolved in code

- Transport/API failures propagate out of Phase1, strict retry and Phase2 instead
  of being classified as malformed translation output and converted to English.
- Authentication, permanent request errors and exhausted credits abort folder
  processing; transient rate limits and server failures retain bounded retries.
- Source fallback counts include multi-cue leaves. The shipping entry point refuses
  to save an incomplete result, including when the caller did not provide metrics.
- Folder scans exclude generated `.ko.srt` files and directories. File failures
  return exit code 1; existing output is not overwritten by incomplete translation.
- Multipart retries rewind each stream to its original position for both retryable
  HTTP responses and connection exceptions.
- Batch and synchronous paths share snapshot hydration, approved glossary selection,
  effective style feedback, retry profile/mode selection and final normalization.
- Batch manifests record the actual request profile/mode and credential-free runtime
  settings. Finalization restores those settings and records its own code/settings.
- Offending-cue-only Batch responses are parsed and spliced into protected cues.
- Overedited Batch candidates are rejected without an uninitialized variable or
  double-counted rejection metric.
- Frozen snapshots supply cue text, timing, order and context without reading the
  original SRT. Legacy missing context is empty, not silently reconstructed.
- Watchlist collection no longer guesses v3 from a continuation-tail signature.
  Missing profile evidence remains unknown; historical results are not relabeled.
- Git provenance resolves relative to the executing checkout, not a developer path.
- Dependencies and Python support are documented; CI runs the existing standard-
  library unittest suite on Python 3.10, 3.12 and 3.14 without API credentials.

## Validation

Independent Python 3.12.4 virtual environment, installed from `requirements.txt`:

```text
python -m unittest discover -s tests -q
Ran 53 tests
OK
git diff --check
(no errors)
```

The regression tests mock network boundaries, but execute the real request builders,
multipart encoding, Batch preparation/finalization, frozen CLI path and selectors.
The original 42 tests still run; the old test that asserted guessed v3 provenance
now asserts that missing evidence remains unknown.

## External validation: blocked, not passed

The four excerpts in `external_domain_smoke.jsonl` were selected from
[NASA](https://www.nasa.gov/podcasts/gravity-assist/gravity-assist-the-moon-with-sarah-noble/),
[USGS](https://www.usgs.gov/faqs/what-difference-between-earthquake-magnitude-and-earthquake-intensity-what-modified-mercalli),
[NPS](https://www.nps.gov/grba/learn/nature/bats.htm?fullweb=1), and the
[US Forest Service](https://research.fs.usda.gov/treesearch/40227).
The source-linked semantic expectations were fixed before requesting translation.
These are external-domain text excerpts with synthetic cue boundaries and timing,
not a representative real-subtitle held-out benchmark.

A live frozen replay was attempted on commit `b27970abe85a8710cf6bb4a9083f4d3fb6517850`,
using `gpt-4.1-mini` / `gpt-4o`, temperature 0, one attempt per phase, split depth 0
and 30-second request timeout. The first request failed with HTTP 429. One bounded
diagnostic request confirmed this sanitized error:

```json
{"status":429,"error_type":"insufficient_quota","error_code":"credit_balance_exhausted"}
```

No translation rows completed. No successful meaning review or out-of-domain quality
claim is made. No keys, billing settings or credits were changed. After the user
restores API credit, rerun the command in `evaluation/README.md` and review each
semantic expectation. Broader generalization still requires a separately selected
real-subtitle evaluation set; passing four examples would not prove it.
