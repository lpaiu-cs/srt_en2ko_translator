## Round 40 Shipping Failure Labels

This is a manual row-level labeling pass over the 6-row shipping failure corpus.

Corpus:
- [cs231n_sp25_shipping_failure_corpus_round37.jsonl](/Users/lpaiu/study/25-2/Translator/evaluation/cs231n_sp25_shipping_failure_corpus_round37.jsonl)

Label dimensions:
- `allowed_technical_carry_through`
- `actual_residual_english`
- `split_first_candidate`
- `wrap_first_candidate`

## Row Labels

### Lecture 11 Large Scale Distributed Training::block-0083
- `allowed_technical_carry_through`: yes
  - `binning` is closer to domain term carry-through / glossary policy than generic bad English.
- `actual_residual_english`: no
- `split_first_candidate`: yes
  - long dependent-end causal tail
- `wrap_first_candidate`: yes
  - line overflow remains after wrapping

### Lecture 12 Self-Supervised Learning::block-0175
- `allowed_technical_carry_through`: yes
  - `GPT-4`
- `actual_residual_english`: no
- `split_first_candidate`: no
- `wrap_first_candidate`: no

### Lecture 3 Regularization and Optimization::block-0704
- `allowed_technical_carry_through`: yes
  - `Adam`, `RMSProp`, `SGD`
- `actual_residual_english`: no
- `split_first_candidate`: yes
  - unfinished dependent-end tail
- `wrap_first_candidate`: no

### Lecture 8 Attention and Transformers::block-0381
- `allowed_technical_carry_through`: yes
  - `RNN` usage is technical carry-through
- `actual_residual_english`: no
- `split_first_candidate`: no
- `wrap_first_candidate`: yes
  - dominant issue is `cps_overflow_severe`

### Lecture 10 Video Understanding::block-0404
- `allowed_technical_carry_through`: yes
  - `AlexNet`, `VGG-16`, `GFLOPS`, `C3D`
- `actual_residual_english`: no
- `split_first_candidate`: no
- `wrap_first_candidate`: yes
  - numeric-heavy / model-name-heavy line packing

### Lecture 14 Generative Models 2::block-0301
- `allowed_technical_carry_through`: yes
  - `GAN`, `DC-GAN`, `ConvNet`
- `actual_residual_english`: no
- `split_first_candidate`: yes
  - unfinished dependent-end block
- `wrap_first_candidate`: no

## Summary

This 6-row corpus does **not** currently show strong evidence for `actual_residual_english`.

What it mostly shows is:
- technical carry-through being mixed into the same routing bucket as bad English
- unfinished dependent-end rows that may prefer `split-first`
- wrap/readability rows that likely need line packing or CPS policy work more than prompt work

That makes `english residual policy split` the right next A/B before touching continuation or broader prompt behavior again.
