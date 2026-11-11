# All prepared server sources: mixed continued pretraining

User clarified that the large acquired collections should be combined, not left
unused while EOS alone is repeated. `configs/server-all-data.toml` now explicitly
lists ten original corpus files and eighteen prepared source directories.

Includes Aya Collection, EOS, IndoSum, ITQA, AMR-NLI, Indo-NER, Indo-Bloom,
STEM/non-STEM, original Indonesian web/wiki/news/instruction corpora, community
corpus/recipes/Javanese lexicon, Alpaca, multilingual OASST, IndoNLU, IndoQA,
PRDECT and commerce. Previous SFT mixtures are not included again as independent
sources. OASST Indonesia is covered by multilingual OASST. The undeveloped bulk
catalog and its partially downloaded sources are not claimed to be included.

## Preparation

`scripts/train_all_server.py` runs only in `/workspace/gatra`. Preserve explicit
source splits and use the historical original-corpus document-hash split for base
files. Reserve all test and validation normalized text, user prompts and source
groups before writing train bins. Test is never written into training/validation
bins. Deduplicate full normalized text across all included sources. This is not
near-dedup and does not resolve historical model exposure or template variants.

For ITQA (no validation) and non-STEM (empty validation), reserve 5% of the sorted
hashed training groups, at least one, while preserving test. Book groups stay
together. Every declared source must retain training rows or the build fails.
Record per-source document/byte counts, input hashes, derived groups and filtering
counts in `data/bin/gatra-10m-all-data-v1/meta.json`.

No 512/1,024-byte document-length filtering: entire documents/messages enter the
byte bins and the trainer samples 1,024-byte windows. Instruction examples use
full-text CE including prompts here. This is continued pretraining, not
assistant-only SFT, and does not promise learning full long-document dependencies.
Source files are included once after exact dedup; sampling is proportional to
retained bytes. All sources contribute, but Aya/EOS can dominate by size.

## Training

Start from original expanded pretraining step 88,138, not the EOS-specialized
checkpoint whose fixed general-text validation regressed. Fresh optimizer/scaler,
context 1,024, batch 32, FP16, peak LR 3e-5, 200 warmup updates, 30% final WSD
decay. Budget: one retained-corpus-size token pass (random windows with replacement),
maximum eight hours. Save/evaluate every 1,000 updates; generation every 5,000.

Launched preparation-to-training job PID 21222 on 2026-11-11. Log:
`checkpoints/gatra-10m-all-data-v1.log`. Checkpoints and resolved training manifest:
`checkpoints/gatra-10m-all-data-v1/`. Fixed evaluation follows completion using
the same original comparison windows; test remains reserved. All datasets and
model artifacts stay on server; only scripts/config/documentation live locally.

Small synthetic verification passed for held-out group exclusion, cross-source
exact text exclusion, reserved test omission and byte counts. Real source totals
and elapsed-time estimates are available only after the server build completes.
