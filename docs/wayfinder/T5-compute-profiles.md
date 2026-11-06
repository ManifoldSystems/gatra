---
title: "T5: Profil compute Spark vs cluster"
tags: [gatra, wayfinder, ticket]
wayfinder: research
status: resolved
blocked-by: []
---

# T5: Profil compute Spark vs cluster

## Question

Kunci pembagian kerja Spark-vs-cluster berbasis angka terukur, bukan asumsi: throughput bf16 sustained GB10 untuk 300M (target ukur langsung bila Spark tersedia; bila belum, bound dari bandwidth 273 GB/s + referensi GB10 training), biaya/waktu 3T di cluster (Modal/dll), dan batas kelayakan Spark-full ~1T.

Keputusan yang harus keluar dari tiket ini:
1. Throughput terukur/ter-bound per profil (TFLOPS sustained, MFU) + waktu P0/pilot-50B/full-3T/full-1T per profil.
2. Matriks pembagian: Spark = P0 + A/B + SFT/DPO/QAT + eval; cluster = P1 full — atau revisi bila angka berkata lain.
3. Opsi Spark-full ~1T: trade-off kualitas (keluar rezim overtrain 10k) + justifikasi kapan diambil.
4. Estimasi biaya per profil (GPU-hours × tarif).

Konteks: `docs/research/gatra-300m-training-2026-10-22.md` §2 (baseline 8×A100/8×H100); spesifikasi GB10 (128GB unified, 273 GB/s, ~1 PFLOP FP4).

Unblocks: T7 (spec). Research AFK — bisa jalan paralel sekarang.
