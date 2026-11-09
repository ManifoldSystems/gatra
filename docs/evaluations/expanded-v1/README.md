# Expanded-mixture checkpoint evaluation

## Method

Run on the V100 after step 88,138 completed. All three checkpoints use the same
expanded validation bins, FP32 inference and 100 identically seeded batches of
16 × 256 byte tokens (409,600 evaluated target positions, sampled with possible
overlap). Validation context is 256 so the initial checkpoint is supported.
These numbers are not directly comparable to the context-512 training log.
The 12 fixed prompts have two generation seeds each, 192 new byte tokens,
temperature 0.7, top-k 40 and top-p 0.9. Generation uses native context 256 for
the initial checkpoint and 512 for the continuations. This comparison therefore
does not isolate the effect of training from the effect of generation context.

## Paired validation

| Checkpoint | Step | Cumulative processed byte tokens | Loss | Byte perplexity |
| --- | ---: | ---: | ---: | ---: |
| Initial latest | 34,212 | 280,264,704 | 1.074185 | 2.927607 |
| Expanded best | 80,000 | 655,360,000 | 0.974524 | 2.649906 |
| Expanded latest | 88,138 | 722,026,496 | 0.971670 | 2.642353 |

Latest improves loss by about 9.5% relative to the initial checkpoint on this
fixed sample. It also improves slightly over the checkpoint named best, which
was selected using different randomly sampled context-512 evaluations during
training. File names alone do not settle the comparison. Prefer latest as the
working baseline for the next experiment while retaining both checkpoints.

## Qualitative findings

Inspect all 72 outputs in [samples.md](samples.md); machine-readable settings,
checkpoint SHA-256 hashes and per-batch losses are in [results.json](results.json).

- Recipe continuations become more recipe-like, using cooking actions and
  ingredients. They can still repeat steps, skip numbering and omit the rice.
- News and story continuations have Indonesian sentence structure but frequently
  lose the prompt's meaning or introduce unrelated events.
- Both expanded checkpoints fail the direct Tokyo, 2 + 3, named-cat retrieval,
  three-fruit listing, exact-copy and good-morning translation prompts in both
  tested seeds. Summaries introduce unsupported details. Identity responses do
  not reliably identify Gatra.
- Repetition remains prominent. Lower language-model loss does not establish
  instruction-following or factual accuracy.

This is an exploratory inspection, not a standardized benchmark or an estimated
population accuracy. Prompts have not been checked against the training corpus.
The instruction prompts mainly use Indonesian labels; existing prepared
instruction examples use `User:` / `Assistant:`. Test that exact template before
attributing every instruction failure to model capacity. Per-source validation
is not included here; aggregate improvement may hide weaker individual domains.

## Recommended next experiment

Use latest as the baseline for a small Indonesian assistant-only SFT pilot with
one consistent conversation format and short, clean examples that fit context
512. Mask user/prompt tokens in the loss, preserve held-out conversation groups,
and compare the same prompts plus template-matched prompts and held-out
assistant loss. Watch for worsening general-text validation. This pilot tests
whether task alignment helps; it does not guarantee factual knowledge or
reasoning. These evaluation results do not yet justify another blind long
pretraining run.

## Backup

Local backup destination:
`checkpoints/gatra-10m-expanded-ctx512-fp16/`.
Keep `best.pt`, `latest.pt`, logs, launch metadata, saved TOML config and bin
metadata together. `backup-sha256.json` records transfer verification on the Mac.
