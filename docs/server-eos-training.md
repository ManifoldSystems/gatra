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

## Overnight context extension

### First EOS pass completed

Completed 80,328 updates / 658,046,976 attempted byte tokens in 69.6 minutes,
averaging 157,559 byte tok/s. 30 AMP updates were skipped (~0.037%). Final EOS
validation loss was 0.7778 versus 0.8862 at initialization; best was selected at
step 63,000. Sampled evaluation windows vary during training.

Fixed general-text validation on the original shared windows regressed:

| Checkpoint | Fixed general-text loss |
| --- | ---: |
| Original pretraining control | 0.971670 |
| EOS best (step 63,000) | 1.167840 |
| EOS latest (step 80,328) | 1.175220 |

Latest is approximately 20.94% worse than the control on this metric. Lower EOS
loss does not imply better general language performance. The already queued
overnight run uses EOS latest per its declared plan; it remains an EOS adaptation
experiment, not a validated replacement for the original pretraining control.
Use the final paired evaluation before choosing an SFT base. Raw outputs remain
in `checkpoints/eval-eos-v1/fixed/` on the server. The evaluator's different-mix
warning reflects deliberately evaluating EOS weights on the original corpus.

Best SHA-256: `e3af2de8a6b2320679da23d44498d1fbe7ce65ae48c82f44a6058a78c2ada23f`.
Latest SHA-256: `adfea8ac652fb066c3051d3f65871026cda3a2b3252770c303e50807f19ee1f0`.

User approved a longer overnight run (approximately 23:00 to 09:00 window).
`configs/server-eos-overnight.toml` and `scripts/train_eos_overnight.py` queue an
automatic continuation after PID 20103 exits successfully. First-pass checkpoints
remain in their original directory. The new directory is
`checkpoints/gatra-10m-eos-ctx1024-overnight/`; stdout is the adjacent `.log`.

Configuration: context 1,024 **bytes**, batch 32 (32,768 tokens/update), FP16,
LR 3e-5, 200-step warmup, WSD with 30% final decay to 3e-6, and 120,492 updates
(3,948,281,856 attempted tokens, about six corpus-size token budgets). Maximum
runtime is eight hours for the continuation; fixed evaluation follows. At the
measured synthetic speed it takes about 5.9 hours; allow roughly 6–7 hours for
real batches/evaluation plus the remaining first pass. This is an estimate, not
a promised wall-clock completion time.

Weights and optimizer moments are retained from the completed first pass.
Step/token counters and best-validation tracking reset for the changed context;
parent counters/hash are retained in the manifest. GradScaler restarts at its
default scale. Model size stays 10.7M. Epoch-equivalent budgets use random window
sampling, not guaranteed full document traversal. Longer training is not evidence
of better generalization; retain best and latest and compare fixed general-text
validation after completion.

Synthetic CUDA benchmark (25 updates, last 20 timed; original process paused
briefly and resumed in `finally`):

| Context | Batch | Byte tok/s | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| 1,024 | 8 | 153,440 | 1,655 |
| 1,024 | 16 | 174,268 | 3,135 |
| 1,024 | 32 | 185,778 | 6,093 |
| 2,048 | 8 | 146,717 | 3,136 |

Each probe had one skipped FP16 update during scaler adaptation. These are random
byte-input performance probes, not quality evaluations. Actual total GPU memory
includes the paused process and CUDA overhead. Context 1,024/batch 32 was selected.

Correction: the first EOS run inherited `decay_frac=0` from its base checkpoint,
so its LR remains flat after warmup despite the earlier intended decay description.
The queued overnight run explicitly sets `decay_frac=0.30`; warmup peak and final
LR were verified against `schedule_lr` locally.
