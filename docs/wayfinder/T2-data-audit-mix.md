---
title: "T2: Audit data + mix per profil GPU"
tags: [gatra, wayfinder, ticket]
wayfinder: task
status: resolved
blocked-by: []
---

# T2: Audit data + mix per profil GPU

## Question

Jalankan `gatra-build` di semua sumber + ukur token unik per sumber dengan tokenizer T1, lalu kunci mix final untuk dua profil: (a) full 3T (cluster), (b) Spark-full ~1T. Termasuk elaborasi karakteristik tiap komponen: domain, bahasa, panjang dokumen, kualitas, epoch count.

Keputusan yang harus keluar dari tiket ini:
1. Tabel supply per sumber (token unik terukur, bukan estimasi) + total vs 750B (3T) / 250B (1T).
2. Mix final per profil (ID% / EN% / code% / math% / sintetik%) + epoch per komponen (tidak ada >4.0).
3. Karakteristik tiap komponen (satu paragraf per komponen: apa isinya, kenapa porsinya segitu).
4. Gate otomatis: 3T vs fallback 2T (aturan angka, bukan diskusi ulang).
5. Perubahan `prepare.py` (sumber baru + kolom skor) + `data.py` (MinHash + epoch-cap + uint16 + streaming) — sebagai diff-spec, bukan implementasi.

Konteks: `docs/research/gatra-300m-data-2026-10-22.md` §1–§5 (mix awal 15/52/15/10/7); fallback ladder §5.

Unblocks: T6 (bukti dokumen panjang ID), T7 (spec). Blocked by: T1.
