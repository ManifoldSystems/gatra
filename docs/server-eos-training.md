# EOS continued pretraining, server-only

Configuration: `configs/server-eos-train.toml`. Entry point:

```bash
cd /workspace/gatra
.venv/bin/python -u scripts/train_server_eos.py
```

Launched on 2026-11-10 as PID 20103. Preparation and training log:
`checkpoints/gatra-10m-eos-v1.log`. Checkpoints and the resolved launch manifest:
`checkpoints/gatra-10m-eos-v1/`. All dataset, bin and model files remain on server.

Preparation completed: **658,043,570 train byte tokens**, **34,924,759 validation
byte tokens**, **80,328 planned updates**. Initial EOS validation loss: **0.8862**.
CUDA initialization and initial evaluation/checkpoint succeeded. A process-event
waiter runs fixed general-text evaluation for best/latest after successful
completion; output stays at `checkpoints/eval-eos-v1/fixed/` on the server.

## Scope

This stage uses the full prepared EOS corpus (130,649 train and 7,040 validation
documents before any additional exact overlap removal). Preserve the exported
split, verify input SHA-256 and let validation win normalized-text duplicates.
Write UTF-8 bytes plus newline separators to uint8 token bins, with counts and
hashes in `data/bin/gatra-10m-eos-v1/meta.json`.

Initialize from pretraining step 88,138, not the task-adapted v3 checkpoint.
Preserve model architecture/context (10.7M parameters, byte tokenizer, 512-byte
windows). Use a fresh AdamW and GradScaler for this corpus, FP16, batch 16,
peak LR 5e-5, 100-step warmup and the existing WSD decay. Validation best tracking
resets because the validation corpus changed. Stage-local counters start at
zero; original counters and checkpoint hash are preserved in the launch manifest.

Budget is `ceil(train_bytes / 8192)` updates: one corpus-size token budget.
The existing trainer samples random windows with replacement, so this does not
guarantee seeing every document or every byte exactly once. Long EOS documents
are available to window sampling rather than discarded for exceeding 512 bytes.

The small bin verification exercised hash validation, byte counts and exclusion
of a training document also present in validation. Training uses the existing
CUDA AMP trainer. Periodic logs include training/validation loss and generations.

This is full-text pretraining, not assistant-only SFT. Aya and the other newly
acquired instruction sources are not included in this stage. SFT preparation and
selection follow the pretraining result; there is no claim that all downloaded
collections have already been trained. Compare original fixed general-text
validation and generation after completion before choosing the next SFT base.
