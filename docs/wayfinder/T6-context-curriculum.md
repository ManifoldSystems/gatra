---
title: "T6: Kurikulum 4K→8K→16K + data panjang"
tags: [gatra, wayfinder, ticket]
wayfinder: grilling
status: resolved
blocked-by: []
---

# T6: Kurikulum 4K→8K→16K + data panjang

## Question

Kunci kurikulum konteks (tahap token per panjang + θ per tahap + fase extension) dan jawab pertanyaan data: apakah dokumen ID panjang (buku/skripsi/laporan, >8K token) cukup untuk fase 8K/16K? Bila tidak, apa fallback-nya (EN panjang? sintetik gabung? potong target 16K?).

## Resolution (2026-10-22, grilling dengan manusia — sesuai rekomendasi)

1. Kurikulum final: **2048 bulk (~2.6T) → 4096 (~300B) → 8K (~80B) → 16K (~20B)**.
2. θ: **500k dari step-0, tetap** (16K-ready; tanpa retune).
3. SWA 5:1+W=1024: **hanya 16K** + gate recall-ID; 4K/8K full attention.
4. Sumber dokumen panjang ID (cek HF 2026-10-22): **(a) TERBUKTI ADA** — `endomorphosis/ipfs_indonesia_laws` (105.645 UU JDIH BPK), `Azzindani/ID_REG` (250rb+ dokumen hukum), `evoreign/indonesia-regulation` (PERDA/POJK + markdown, s.d. 23M chars/dok). Skripsi-UI hanya abstrak (terlalu pendek); whole-books 20.9B token tapi Inggris (fallback b). Strategi: **(a)+(c)** — tambah 1–2 sumber hukum ID + packing dokumen pendek untuk sisa; (b) EN-panjang sebagai fallback.
5. Amendemen: T3 buffer RoPE 16K-ready (θ=500k di gatra-300m.toml); T4 batch-math @16K dicatat di spec.

Keputusan yang harus keluar dari tiket ini (bersama manusia — grilling):
1. Kurikulum final: mis. 2048 bulk → 4096 tail → 8K ext → 16K ext (batas token per tahap).
2. θ per tahap (usulan: 500k dari step-0; naik ke 1e6 di 16K atau tidak).
3. Varian SWA 5:1+W=1024: aktif di tahap mana (usulan: hanya 16K) + gate recall-ID.
4. Bukti dokumen panjang ID dari T2 (distribusi panjang per sumber) + fallback bila kurang.
5. Dampak ke T3 (buffer RoPE) + T4 (batch math @16K) — catat sebagai amendemen.

Konteks: hitungan KV (@4K 84 / @8K 168 / @16K 336 MiB; SWA-1024 @16K ~66 MiB); preseden SmolLM2 2k→8k + θ 130k; Qwen3 θ=1e6 @32K.

Unblocks: T7 (spec). Blocked by: T2. HITL — butuh pertukaran langsung dengan manusia.
