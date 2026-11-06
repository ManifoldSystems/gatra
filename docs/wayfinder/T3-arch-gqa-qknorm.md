---
title: "T3: Diff GQA + QK-norm + config 300M"
tags: [gatra, wayfinder, ticket]
wayfinder: prototype
status: resolved
blocked-by: []
---

# T3: Diff GQA + QK-norm + config 300M

## Question

Buat diff `model.py` (MHA → GQA 15Q-3KV + QK-norm + hook z-loss/soft-cap OFF) + `config.py` (field baru) + `configs/gatra-300m.toml` (28L/d960/V32k/θ=500k/ctx-4096/dropout-0.0), dengan gate `param_count() ≈ 299.9M`.

Keputusan yang harus keluar dari tiket ini:
1. Diff final per file (siap review, bukan konsep).
2. `param_count()` terukur + KV/tok + Q4_K size (rekalkulasi, bukan copy).
3. Hasil smoke-run (gatra-smoke + 1M config tetap hijau; test baru untuk GQA equivalence saat n_kv == n_head).
4. Catat divergensi dari arch-note bila ada (θ=500k vs 100k — diputuskan di T6, tapi field-nya disiapkan di sini).

Konteks: `docs/research/gatra-300m-arch-2026-10-22.md` §1 + §4 (checklist kode); `src/gatra/model.py`, `config.py` saat ini.

Unblocks: T7 (spec). Prototype di branch, bukan main.
