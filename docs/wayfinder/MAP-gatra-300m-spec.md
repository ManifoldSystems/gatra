---
title: "Wayfinder Map: Gatra-300M Spec Siap Eksekusi"
tags: [gatra, wayfinder, planning, spec-300m]
wayfinder: map
destination: "spec 300M siap eksekusi"
---

# Map: Gatra-300M Spec Siap Eksekusi

## Destination

Spec 300M siap eksekusi: arsitektur + data + training + post-train/QAT/export + eval gates (G0–G6), dengan kurikulum konteks 4K → 8K → 16K dan pembagian data fleksibel per profil GPU (Spark vs cluster). Selesai = tidak ada keputusan tersisa sebelum implementasi dimulai.

## Notes

- Domain: pretrain_llms/gatra (nested git `ManifoldSystems/gatra`, tracker GitHub read-only → peta lokal di `docs/wayfinder/`).
- Skills per sesi: `research` (klaim primer + kutip+sitasi), `prototype` (diff kode kecil), `grilling` + `domain-modeling` (keputusan).
- Standing preferences: dense+GQA+full-attention sebagai base (alternatif ditolak kecuali SWA varian); tidak ada sumber >4 epoch; QAT int4 + GGUF Q4_K_M; gates sebagai merge-gate.
- Riset dasar (8 catatan, jangan riset ulang tanpa alasan): `docs/research/gatra-300m-{arch,data,training,alternatives}-2026-10-22.md`, `arch-efficient-sub500m`, `pretrain-best-practices-sub500m`, `edge-deployment-quant-runtime`, `gatra-repo-audit`.

## Decisions so far

- [T5: Spark bound 15–25 TFLOPS sustained; full-3T di 1× Spark = 8.6 tahun; Spark = pilot/post-train, cluster = full](work/t5-compute @38b03a3): memperkuat Q6; opsi Spark-full hanya dengan budget ~1T.
- [T3: GQA 15Q-3KV + QK-norm + gatra-300m.toml, param_count 299885504 ≈ 299.9M](work/t3-arch @4268498): gate arch-note lolos.
- [T4 (reduced scope): wsd_lr(tokens_seen) + TrainConfig schedule/batch/precision](work/t4-trainloop @8cac54d): DDP/bf16/checkpoint-state jadi follow-up.
- [Integration: T3+T4+T5 merged, 42 tests green](integration/gatra-300m-spec @54fe54a)
- [T1: BPETokenizer V32k + train/fertility scripts, 51 tests green, merged](integration/gatra-300m-spec @bf620e5): own-BPE-32k locked (vs SmolLM2-49k); `tokenizers` dep added.
- [T2: data pipeline 300M, 60 tests green](work/t2-data @698665e, child of bf620e5): uint16 bins + meta tokenizer + epoch-cap hard-fail + score columns; mix full-3T 15/52/15/10/7/1 + Spark-~1T 30/35/15/12/7/1 di gatra-300m.toml.
- [T6: kurikulum 4K→8K→16K resolved](T6-context-curriculum.md): 2048 bulk → 4096 → 8K → 16K; θ=500k; SWA hanya-16K; sumber hukum-ID terbukti ada (ipfs_indonesia_laws, ID_REG, indonesia-regulation).
- [T7: spec-gatra-300m.md (1286 kata, 10 seksi), 60 tests green](work/t7-spec @d832e09, child of 698665e): θ=500k + mix_long + komentar kurikulum di toml; review lolos (3 mix sum 1.0, hanya 4 file tersentuh).
- [FINALISASI DONE] rantai commit lengkap: bf620e5 (T1+T3+T4+T5) → 698665e (T2) → d832e09 (T7). Pointer `integration/gatra-300m-spec` = d832e09; worktree /tmp/gatra-wt/* dihapus; main == origin @314f466.
- [PR #1 dibuka](https://github.com/ManifoldSystems/gatra/pull/1): amaldevice:integration/gatra-300m-spec → ManifoldSystems/gatra@main (via fork amaldevice/gatra; akses upstream pull-only).

- [Arsitektur kunci: arch-note 28L/d960/15Q-3KV/V32k 299.9M](T3-arch-gqa-qknorm.md): embedding tax 10.5% vs 14.6% (V49k); KV 21 KiB/tok; Q4_K ~169 MB.
- [Konteks kurikulum 4K → 8K → 16K, 16K-ready dari step-0](T6-context-curriculum.md): θ=500k dari awal; SWA 5:1+W=1024 sebagai varian (trigger ≥8K/16K).
- [Budget 3T + fallback 2T + elaborasi karakteristik data](T2-data-audit-mix.md): gate audit otomatis; unik-ID ~100–150B vs butuh 750B.
- [Spec end-to-end sampai gates G0–G6](T7-spec-assembly.md): termasuk SFT→DPO→QAT→GGUF.
- [English replay 52% diterima, narasi positioning ID-first = evaluasi+SFT](T2-data-audit-mix.md): ID 15% = porsi maksimum aman dalam cap 4 epoch.
- [Compute: pembagian data fleksibel per profil GPU](T5-compute-profiles.md): Spark = pilot/post-train/inferensi; cluster = full 3T; opsi Spark-full hanya dengan budget ~1T.
- [Alternatif ditolak sebagai base](../research/gatra-300m-alternatives-2026-10-22.md): Mamba/conv/hybrid-head/MLA/linear/sparse; SWA satu-satunya A/B layak.

## Frontier (open, unblocked)

_(empty — all tickets T1–T7 content-done)_

## Blocked

_(empty — T2/T6/T7 resolved)_

## Not yet specified (fog) — dibawa ke eksekusi (di luar destination spec)

- Distilasi final: default MiniPLM data-side; logit-KD hanya bila A/B 50B menang (T1 mengunci own-BPE-32k → KD butuh teacher same-tokenizer).
- Post-training ID: kurasi Aya/Cendol/NusaWrites + filter kapasitas (detail set saat eksekusi P3).
- Eval harness ID: subset IndoMMLU/NusaX/IFEval-ID + threshold G5 (dibangun sebelum pilot-50B).
- QAT int4: per-block (llama.cpp path) sebagai default; A/B per-channel bila delta >1pp.
- T4 follow-up: DDP/bf16/checkpoint-state penuh; MinHash lintas-sumber skala penuh.

## Out of scope

- Mamba/SSM, conv-hybrid, hybrid-head, MLA, linear/gated, sparse-trainable sebagai base 300M — ditolak di alternatives-note; naik ke meja hanya bila produk menuntut ≥16K DAN runtime-pair lolos sustained-5min + fallback 0.
- Scale >300M, multilingual di luar ID+EN, multimodal — di luar destination spec-300M.
