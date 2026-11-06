---
title: "T1: Tokenizer BPE-32k + fertility gate"
tags: [gatra, wayfinder, ticket]
wayfinder: task
status: resolved
blocked-by: []
---

# T1: Tokenizer BPE-32k + fertility gate

## Question

Latih tokenizer BPE 32k ID-first (HuggingFace `tokenizers`, byte-level BPE) di korpus §3.3 data-note, dan kunci keputusan vocab: BPE-32k sendiri vs adopsi SmolLM2-49k (membuka opsi logit-KD). Gate: fertility ≤1.5 tok/word di wiki-id + web-id.

Keputusan yang harus keluar dari tiket ini:
1. Korpus latih tokenizer final (komposisi + bobot).
2. Vocab final: 32k-sendiri atau 49k-SmolLM2 (dengan justifikasi fertility + implikasi KD).
3. Script `scripts/train_tokenizer_300m.py` + `scripts/fertility_gate.py` (gagal bila >1.5).

Konteks: `docs/research/gatra-300m-data-2026-10-22.md` §3.3; `docs/research/gatra-repo-audit-2026-10-22.md` §2.1 (baseline byte ~6.5 tok/word); preseden Aksara (wiki-ID 1.357, web-ID 1.215).

Unblocks: T2 (estimasi token per sumber), T3 (vocab_size config), T7 (spec).
