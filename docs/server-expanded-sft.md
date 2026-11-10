# Server-expanded single-line SFT

User requested more training data, with new datasets and checkpoints retained
only on the V100 server. Source code and summary documentation remain in Git.

## Launch: 2026-11-10

Launched PID 17298 on V100 after preparation and six relevant tests passed.
25,834 training examples (2.77x v2), 3,481 development validation, and 1,306
reserved IndoNLU test examples. Three epochs give 4,845 attempted updates.
Log: `checkpoints/gatra-10m-sft-expanded-v3.log`; launch metadata is adjacent.
An event-driven process waiter runs paired task/fixed evaluation after successful
training, under `checkpoints/eval-sft-expanded-v3/`.

| Task | Train | Validation |
| --- | ---: | ---: |
| Synthetic extraction | 5,262 | 570 |
| Copy | 1,088 | 112 |
| TyDi QA | 1,511 | 157 |
| FacQA | 1,196 | 175 |
| NERGrit | 3,609 | 593 |
| SmSA | 7,136 | 1,231 |
| E-commerce sentiment | 3,000 | 360 |
| PRDECT sentiment/emotion | 3,000 | 267 |
| IndoQA | 32 | 16 |

New resolved revisions:

- `jakartaresearch/indoqa`: converted Parquet revision
  `98a61b0101111eadddbc3a1f061898f9e2c163e1`.
- `ZakyF/PRDECT-ID`: `32343a60b0257bf6d14964efbc32f5097201ce2b`.
- `AIbnuHibban/e-commerce-sentiment-bahasa-indonesia`:
  `295cbc457b8282dde9cf43910ecb57efe57320b8`.

IndoQA originally had 3,309 train / 1,104 validation records; upstream context
overlap removed 1,320 train records. Even after shortening only the instruction
prefix, 3,045 remaining rows exceed 512 bytes. No passage or answer was cropped.
Commerce contained extensive exact duplicates (14,178 training duplicates were
removed by acquisition). Size alone is not evidence of diverse content.

Nine hash-ranked new-source training examples were inspected. IndoQA answers
were supported by passages, though one passage has an incomplete birth-date
phrase. Commerce examples were short and template-like. PRDECT has subjective
labels: a review praising softness but complaining about color is labeled
Positive/Happy. These remain source-annotated experimental candidates.

## Experiment contract

- Acquire IndoQA, PRDECT-ID and e-commerce sentiment using
  `scripts/prepare_local_tasks.py`; resolved HF revisions, original split/index
  and raw Arrow snapshots are retained on the server.
- Mix with prepared IndoNLU FacQA/NERGrit/SmSA and v2 copy, synthetic extraction
  and TyDi QA. `scripts/prepare_server_mix.py` records input/output hashes.
- Full context and answer must fit 512 UTF-8 bytes; no truncation. New answers
  must be single-line. Supervise assistant bytes plus terminal newline only.
- Keep upstream held-out groups. Test > validation > train; discard lower split
  group/prompt conflicts. Reserve too-long held-out groups for new acquisitions.
- Train caps: SmSA 3,000 per label; commerce 1,000 per label; IndoQA and PRDECT
  3,000 each; NERGrit empty answers 500 per entity type. Other tasks retain their
  eligible examples. These are maximum counts, not oversampling targets.
- Exclude arithmetic from this mixture following v2's poor exact-match result.
- Initialize from the original expanded pretraining `latest.pt` (step 88,138),
  rather than accumulating v2's general-text regression. Fresh AdamW/GradScaler,
  FP16, batch 16, peak LR 1e-5, three epochs. Context stays at 512 bytes.
- Compare greedy newline-stopped exact match on development tasks and the
  fixed FP32 general-text validation. Use identical decoding for base and SFT.
  No held-out test is used for checkpoint selection.

## Paths and commands (server)

```bash
cd /workspace/gatra
.venv/bin/python scripts/prepare_local_tasks.py \
  --sources indoqa prdect commerce \
  --out data/instruction/server-expanded-v1/tasks
.venv/bin/python scripts/prepare_server_mix.py
.venv/bin/python -u scripts/train_sft.py --prepared --epochs 3 --lr 1e-5 \
  --data data/instruction/server-expanded-v1/mix \
  --checkpoint checkpoints/gatra-10m-expanded-ctx512-fp16/latest.pt \
  --out checkpoints/gatra-10m-sft-expanded-v3
```

The preparation script requires `python-dotenv` in addition to existing HF
dependencies. Install it without changing the server's pinned CUDA Torch build.
Long-running jobs write a server-side log and launch metadata.

## Interpretation

This is still a 10.7M-parameter byte-token model with a short context. More data
does not imply broad conversational competence. Source labels are not fully
human-verified; exact split/group checks are not exhaustive cross-source semantic
deduplication. Some tasks/validation were already used in earlier experiments,
and pretraining contamination is not exhaustively audited. Report per-source
accuracy as well as answer loss and general-text loss. No text-replay objective
is introduced in this run; measure the tradeoff before choosing a default model.
