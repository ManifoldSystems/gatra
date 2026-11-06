---
title: "T5 RESULT: Profil compute Spark vs cluster (bound, bukan ukur langsung)"
tags: [gatra, wayfinder, ticket, result]
wayfinder: result
status: resolved
---

# T5 RESULT: Profil compute Spark vs cluster

Keputusan T5 berbasis **bound + referensi**, bukan ukur langsung (Spark belum
tersedia saat tiket resolved): throughput bf16 sustained GB10 dibound dari
bandwidth 273 GB/s + referensi training GB10; biaya/waktu 3T dari baseline
8×A100/8×H100 di [training-note](../research/gatra-300m-training-2026-10-22.md)
§2. Ukur langsung bila Spark tersedia, lalu revisi angka di bawah.

## 1. Bound throughput (bound, bukan benchmark)

- Spark (GB10, 128GB unified, 273 GB/s, ~1 PFLOP FP4): **15–25 TFLOPS
  sustained** untuk 300M (bound bandwidth + referensi GB10 training).
- Full-3T di 1× Spark ≈ **8.6 tahun** → Spark bukan untuk full run.

## 2. Waktu/biaya per profil (estimasi ±30%, bukan benchmark)

FLOPs ≈ $6ND = 6 \times 3.02{\times}10^8 \times 3{\times}10^{12} \approx$ 5.4e21.

| Item | 1×A100-40G | 8×A100-40G | 8×H100 |
|---|---|---|---|
| P0 μP sweep (32B) | ~21 h | — | — |
| Pilot 50B / A-B ≤250B | ~10 d | ~1–7 d | ~1–2 d |
| Full 3T | ~625 d (infeasible) | ~60–90 d | ~20–30 d |
| Total | ~16.5k GPU-h ≈ $25–35k (spot) | ~10 minggu wall | ~5.3k GPU-h ≈ $15–25k |

Asumsi: ~100 TFLOPS sustained/GPU A100 (~32% MFU), ~300 TFLOPS H100.

## 3. Matriks pembagian (keputusan)

- **Spark**: P0 (proxy μP) + A/B ≤250B + SFT/DPO/QAT + eval + inferensi.
- **Cluster**: P1 full 3T (Modal `--detach`).
- **Opsi Spark-full ~1T**: hanya bila budget cluster tak ada; keluar rezim
  overtrain 10k (trade-off kualitas, justifikasi per kasus).

## 4. Gate terkait

G3 = go/no-go cluster/staged-funding. Colab (T4/A100 tunggal) = proxy/pilot
saja (P0 + ≤50B). Checkpoint storage ≈ 70 GB (rolling-5 + best + tiap-50k).
