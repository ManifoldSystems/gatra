---
title: "T4: Training loop WSD-token + DDP + checkpoint-state"
tags: [gatra, wayfinder, ticket]
wayfinder: task
status: resolved
blocked-by: []
---

# T4: Training loop WSD-token + DDP + checkpoint-state

## Question

Tulis ulang loop training untuk 300M: `wsd_lr(tokens_seen)`, grad-accum ke 2M token, bf16+DDP, checkpoint berisi scheduler-state + tokens_seen, `continue_train.py` restore ketiganya. Plus profil memori z-loss chunked @V32k.

Keputusan yang harus keluar dari tiket ini:
1. Diff `train.py` + `continue_train.py` + field `config.py` baru (`schedule`, `batch`, `precision`, `qat`, `eval`).
2. Tabel akumulasi per profil (Spark 1×, cluster 8×A100/H100): micro/GPU × GPU × accum = global batch.
3. Hasil pilot: proxy μP 40M sweep {2,3,4}e-3 (di Spark) → LR kunci untuk 300M.
4. Profil memori: z-loss tiap-step vs tiap-k-step @seq 2048/4096.

Konteks: `docs/research/gatra-300m-training-2026-10-22.md` §1 (P0/P1) + §2 (compute) + §4 (checklist); current `train.py` (cosine-iters, 8k-token batch, fp32).

Unblocks: T7 (spec). Jalan paralel dengan T1–T3 (independen).
