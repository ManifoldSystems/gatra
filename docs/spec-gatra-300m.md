---
title: "Spec Gatra-300M (ringkas, siap eksekusi)"
tags: [gatra, spec, wayfinder]
---

# Spec Gatra-300M

Dokumen ringkas siap eksekusi. Detail per tiket di `docs/wayfinder/` (T1–T7) dan
catatan riset di `docs/research/` — bagian ini **merujuk, bukan mengulang**.
Konfig runnable: `configs/gatra-300m.toml`.

## 1. Arsitektur (T3)

28L / d960 / 15Q-3KV GQA (grup 5, head_dim 64) / SwiGLU 2560 / V32768 tied /
RoPE θ=500k / ctx-4096 / dropout 0.0 / QK-norm ON / z-loss + logit-cap OFF.
`param_count()` = 299885504 ≈ 299.9M; embedding tax 10.5%; KV 21 KiB/tok
(84 MiB @4K); Q4_K ≈ 169 MB. Detail: [T3](wayfinder/T3-arch-gqa-qknorm.md),
[arch-note](research/gatra-300m-arch-2026-10-22.md). Divergensi dari arch-note
awal: θ=500k (bukan 100k) — diputuskan di [T6](wayfinder/T6-context-curriculum.md).

## 2. Tokenizer (T1)

BPE-32k byte-level ID-first (own-BPE-32k, bukan SmolLM2-49k → logit-KD butuh
teacher same-tokenizer). Korpus latih §3.3 data-note (web-ID HQ + wiki dominan).
Gate: fertility ≤1.5 tok/word di wiki-id + web-id
(`gatra-fertility-gate`, exit 1 bila gagal). Artefak:
`data/tokenizer/gatra-300m-bpe.json` via `gatra-train-tokenizer`.
Detail: [T1](wayfinder/T1-tokenizer-bpe-32k.md).

## 3. Data (T2)

Mix full-3T: ID 15 / EN 52 / code 15 / math 10 / sintetik 7 / instr-decay 1
(`[data.profiles]` di toml; bobot AWAL pra-audit, T2 mengunci ulang setelah
token unik terukur). Epoch cap 4.0 fail-closed (`gatra-build` gagal bila
sumber terulang >4x). `cosmopedia-id` satu-satunya profil
unblocked-by-data-acquisition (terjemahan §3.4). Fallback: unik-ID 40–80B →
budget 2T; <40B → mode CPT. Detail: [T2](wayfinder/T2-data-audit-mix.md),
[data-note](research/gatra-300m-data-2026-10-22.md).

## 4. Training (T4)

WSD (warmup → stabil → decay linear 20% token terakhir), peak LR 3e-3 (sweep
proxy {2,3,4}e-3 di Spark, P0), warmup 2000 iters ≈ 4.2B token (dihitung dalam
TOKEN), batch global 2M token/step, AdamW β=(0.9,0.95) wd 0.1, clip 1.0.
Budget: 512 × 4096 × 1430511 = 3.0000T token. Follow-up: DDP/bf16/
checkpoint-state penuh. Detail: [T4](wayfinder/T4-training-loop.md),
[training-note](research/gatra-300m-training-2026-10-22.md).

## 5. Konteks (T6)

2048 bulk (~2.6T) → 4096 (~300B) → 8K (~80B) → 16K (~20B); θ=500k dari step-0
tetap; SWA 5:1+W=1024 hanya-16K + gate recall-ID; dokumen panjang ID terbukti
ada (hukum-ID). File ini = tahap 4096. Detail:
[T6](wayfinder/T6-context-curriculum.md).

## 6. Compute (T5)

Spark = P0 + A/B + SFT/DPO/QAT + eval; cluster = full 3T. Angka di
[T5-RESULT](wayfinder/T5-compute-RESULT.md) (bound dari bandwidth + referensi
training; ukur langsung bila Spark tersedia).

## 7. Gates G0–G6 + eval

G3 go/no-go cluster; G5 threshold eval ID (IndoMMLU/NusaX/IFEval-ID, dibangun
sebelum pilot-50B). Harness eval + QAT per-block + GGUF Q4_K_M: fog yang
dibawa ke eksekusi (lihat MAP). Detail: [T7](wayfinder/T7-spec-assembly.md),
[MAP](wayfinder/MAP-gatra-300m-spec.md).

## 8. Post-training + distilasi

SFT staged (alignment → reasoning, NO RL) → DPO terakhir; distilasi default
MiniPLM data-side. Detail: training-note §P3/P4.

## 9. A/B plan (maks 3, ≤250B) + promosi

Kriteria promosi dikunci sebelum pilot-50B (lihat MAP fog). Detail:
[T7](wayfinder/T7-spec-assembly.md).

## 10. Risiko + fallback

Fallback ladder: 3T → 2T → CPT (lihat §3); Spark-full hanya ~1T (keluar rezim
overtrain 10k). Alternatif ditolak (Mamba/MLA/sparse) naik ke meja hanya bila
produk menuntut ≥16K. Detail: [alternatives](research/gatra-300m-alternatives-2026-10-22.md).
