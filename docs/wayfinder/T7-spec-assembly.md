---
title: "T7: Perakitan spec gatra-300m"
tags: [gatra, wayfinder, ticket]
wayfinder: task
status: resolved
blocked-by: []
---

# T7: Perakitan spec gatra-300m

## Question

Rakit `docs/spec-gatra-300m.md`: satu dokumen yang membuat pelaksana tidak perlu bertanya lagi. Konsolidasi resolusi T1–T6 + revisi training-note ke arch-note/V32k.

Struktur spec (final):
1. Konfig arsitektur kunci + angka terhitung (dari T3).
2. Tokenizer + fertility (dari T1).
3. Mix data per profil + karakteristik per komponen + gate 3T/2T (dari T2; Q2: elaborasi isi tiap sumber).
4. Recipe training per fase P0–P4 + hyperparam exact (dari T4).
5. Kurikulum konteks + varian SWA (dari T6).
6. Profil compute + pembagian Spark/cluster (dari T5; Q6).
7. Gates G0–G6 + harness eval (graduasi fog: pilihan task ID + threshold).
8. Post-training ID + distilasi (graduasi fog: hasil T1/T2).
9. A/B plan (maks 3, ≤250B) + kriteria promosi.
10. Risiko + fallback ladder.

Done = map closed (semua tiket closed, fog tergraduasi atau dipindah out-of-scope).

Konteks: semua catatan `docs/research/` + resolusi T1–T6.

Blocked by: T1–T6 (semua).
