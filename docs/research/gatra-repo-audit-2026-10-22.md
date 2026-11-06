---
title: "Gatra Repo Audit vs Efficient-Edge Architecture Playbook (2026-10-22)"
tags: [gatra, pretraining, audit, efficient-llm, indonesian-lm]
---

# Gatra Repo Audit (2026-10-22)

Audit of `pretrain_llms/gatra` against pretraining best practice and
`docs/efficient-llm-architecture-edge-under-500m-2026-10-20.md` §§7–8.
Method: `rg` + targeted reads of `src/gatra/{model,config,train,data,tokenizer,prepare}.py`,
`configs/*.toml`, `tests/`. No full-directory reads.

## 1. What the repo implements today

### 1.1 Architecture knobs

| Knob | Implemented | Evidence |
|---|---|---|
| Decoder-only Transformer | ✅ | `model.py`: `Gatra` — tok_emb → N×Block → RMSNorm → lm_head |
| Depth / width | Smoke 2L/d64; 1M 4L/d128; 10M **6L/d384** | `configs/*.toml` |
| Attention | ❌ **MHA only** (`qkv` fused `3*d`, `n_head` KV heads) — no GQA | `model.py: Attention` |
| Norm | ✅ pre-RMSNorm (`attn_norm`, `mlp_norm`, final `norm`), eps 1e-6 | `model.py: Block` |
| FFN | ✅ SwiGLU (`w1/w2/w3`), `n_ff = 2.67d` rounded to 64 (1024 @d384) | `config.py: __post_init__`, `model.py` |
| Positional | ✅ RoPE, base θ=10_000, precomputed cos/sin buffer | `model.py: RotaryEmbedding` |
| Biases | ✅ none (`bias=False` everywhere), `attention_bias=false` equivalent | `model.py` |
| Dropout | ⚠️ **0.1 in gatra-1m/10m pretraining** (0.0 only in smoke) | configs; playbook §5.5: 0.0 for pretraining |
| Weight tying | ✅ `lm_head.weight = tok_emb.weight` (playbook §8 hedged "belum tentu ada" — it **is** there) | `model.py: Gatra.__init__` |
| Vocab | ❌ **byte-level, V=256** (`ByteTokenizer`: utf-8 bytes) | `tokenizer.py` |
| Init | ✅ N(0, 0.02) + GPT-2-style residual rescale `/√(2L)` on `proj`/`w2` | `model.py` |
| Precision / parallel | ❌ fp32 default, **no autocast/amp, no DDP/FSDP, no `compile`** | `rg` over `src/`, `tests/` |
| Stability extras | ❌ no QK-norm, no z-loss, no logit cap | `model.py` |

Param counts (analytic, tied: $N = Vd + L(4d^2+3dd_{ff}+2d) + d$; verified formula against code):

| Config | d / L / heads / block | $d_{ff}$ | Params | Nameplate |
|---|---|---|---|---|
| smoke | 64 / 2 / 4 / 32 | 192 | 123,200 | — |
| 1m | 128 / 4 / 4 / 128 | 384 | 885,888 | ~0.9M ✅ |
| 10m | 384 / 6 / 6 / 256 | 1024 | 10,720,128 | ~10.7M ✅ |

### 1.2 Training setup

| Knob | Value (10M tier) | Playbook §5.5 reference |
|---|---|---|
| Optimizer | AdamW β=(0.9, 0.95), wd=0.1 | ✅ matches |
| LR schedule | cosine, peak **3e-4**, `min_lr_ratio` 0.1 | ⚠️ 10× below SmolLM2-360M 3e-3 / MobileLLM-350M 2e-3 (WSD/cosine) |
| Warmup | 100 iters (2% of 5000) | ✅ fraction reasonable |
| Batch | 32 × 256 = **8,192 tokens/step** | ❌ vs ±1–2M tokens/batch in released models (~200× smaller) |
| Grad clip | 1.0 | ✅ matches |
| Dropout | 0.1 | ❌ should be 0.0 pretraining |
| Precision | fp32, single device | ❌ bf16 + DDP needed before any scale-up |
| Eval | train/val loss + ppl every 200 iters, qualitative samples | ❌ no benchmark harness (see §1.4) |

### 1.3 Data pipeline

- **Sources (planned mix, `gatra-10m.toml`)**: CulturaX-id, FineWeb2-HQ `ind_Latn`, wiki-id, Aya-id (×2.0), Cendol wikihow + puisi, Kopi-news-id, newspapers-2018-id, sample. Indonesia-first; identity file at ×20 weight. Only `sample.jsonl` + `gatra-identity.jsonl` exist locally — the large corpora are fetched via `prepare.py` (HF: CulturaX, FineWeb-2/HQ, Wikipedia, Aya, Cendol, Kopi).
- **Dedup**: ✅ exact doc-hash (`blake2b`) within build and across files (`--dedup-against`), first-source-wins ordering. No near-dedup (MinHash) — acceptable at this scale.
- **Split**: ✅ hash-based train/val split (`val_ratio` 0.1, per-source val share, stable across rebuilds) + tail-split fallback for tiny corpora.
- **Weighting**: ✅ fractional repeat weights in `write_corpus` (e.g. 2.5× = 2 copies + 50% third copy).
- **Filtering**: ⚠️ only `min_chars=80` + language/config selection. **No quality classifier** (no FineWeb-Edu/DCLM-style filter), no toxicity/quality gate.
- **Mix gaps**: no English replay, no code, no math, no synthetic textbook (Cosmopedia-style). For an Indonesia-first model the ID-heavy mix is intentional, but §5.2 evidence says code/math + English replay measurably help even non-English targets — worth a 5–10% ablation.
- **Packing**: `gatra-build` → `train.bin`/`val.bin` uint8 memmap + `meta.json`; trainer samples random windows. Clean and correct for byte tokens.

### 1.4 Eval / benchmark harness

**None.** `train.py` logs train/val loss, perplexity, tok/s, VRAM, and free-text samples to `metrics.jsonl`. Tests cover shapes, causality, tokenizer roundtrip, data mixing, checkpoint resume — no HellaSwag/ARC/MMLU/IndoMMLU-style harness, no fertility script, no latency benchmark. (Playbook §9 notes phone tok/s numbers are rare everywhere, but a local `lm-eval-harness` + IndoBench gate is the cheapest quality control before any 30M+ spend.)

## 2. Measured numbers

### 2.1 Tokenizer fertility (stdlib measurement, `sample.jsonl` + `gatra-identity.jsonl`)

| Corpus | docs | bytes/char | bytes/word | chars/word |
|---|---|---|---|---|
| `sample.jsonl` (ID) | 20 | 1.000 | 6.70 | 6.70 |
| `gatra-identity.jsonl` (ID+EN) | 54 | 1.000 | 6.49 | 6.49 |

Indonesian is pure-ASCII Latin, so 1 byte ≈ 1 char and **1 byte-token ≈ 1 char**.
A BPE tokenizer does ~3.5–4.5 chars/token on Latin text → byte tokenizer inflates
sequence length by **≈3.5–4×** (consistent with playbook §3.6 "2–4×").
Consequences, measured in the repo's own units:

- `block_size=256` bytes ≈ **~60 words ≈ ~60–70 BPE-tokens** of effective context. The "256 context" is really a ~64-token context wearing a trenchcoat.
- KV cache per generated unit of *meaning* is 3.5–4× larger than a BPE model with the same architecture; total training tokens to cover the same text is 3.5–4× higher.
- At 10M scale the absolute waste is trivial (KV @256 ctx ≈ 2.3 MB bf16), but the habit compounds: a 100M run on byte tokens pays ~4× the token budget for the same corpus coverage.

### 2.2 Effective token budgets vs Chinchilla / overtraining

`tokens = max_iters × batch_size × block_size`:

| Config | Budget | Params | tokens/param | Chinchilla 20× | Overtrain 10k× (playbook §5.1 default) |
|---|---|---|---|---|---|
| smoke | 2,560 | 0.12M | 0.02 | — | — |
| 1m | 4.1M | 0.89M | **4.6** | 17.8M (4.3×) | — |
| 10m | **41M** | 10.7M | **3.8** | 214M (5×) | **107B (2,600×)** |

The 10M config is ~5× short of even Chinchilla-optimal and ~3 orders of magnitude
short of the 4T-for-350M regime (§5.1: SmolLM2-360M 11k/param). This is a **compute/data
decision, not an architecture decision** — exactly the playbook's headline message.

### 2.3 Depth check (MobileLLM ablation, §3.1)

Playbook: ~30 layers optimal at 125M; <10 layers "perform poorly". Gatra-10M has **6 layers**
at d384 (wide-shallow). Direction of error is confirmed: depth needs to go up 2–5×,
width stays flat or shrinks.

## 3. Gaps ranked by impact

1. **Token budget 3 orders of magnitude short** (§2.2). Nothing else matters until there is a plan to go from 41M → 100B+ tokens. Binding constraint is unique Indonesian data volume (FineWeb-2 `ind_Latn` ≈ 80–100B tokens est.) + repetition ≤4 epochs (§5.1). P1 action: quantify the full build (`gatra-build` on all 10M sources, count bytes) before any architecture work.
2. **Byte tokenizer (V=256)** (§2.1). 3.5–4× sequence inflation; makes every token-budget, KV-cache, and latency number worse by that factor. P1 action: train/adopt BPE 32k–49k, measure fertility on the ID corpus, then commit. (Embedding cost at V=32k/d384 ≈ 12.6M params is real — accounted for in §4 configs.)
3. **Depth 6 layers, should be 12–24+** (§2.3). Cheapest architectural win; no new ops, no runtime risk.
4. **No GQA.** Irrelevant at 10M (KV is MBs) but required for the 30M/100M tiers where KV @4K reaches tens of MB. Implement now while MHA→GQA is a ~20-line change (`n_kv` heads, repeat_interleave); uptraining conversion costs only ~5% of pretraining (§3.2).
5. **Dropout 0.1 + LR 3e-4 + 8k-token batches.** Three pretraining-hygiene misses in one row: dropout→0.0; LR is 10× below released-model practice (sweep 1–3e-3 via 40M μP proxy per §5.5); batch needs gradient accumulation toward ≥256k tokens before a 100M run, else gradient noise dominates.
6. **No benchmark harness.** No external validity signal beyond val ppl. Add `lm-eval-harness` (ARC-ID / HellaSwag-ID / IndoMMLU subset) + fertility + tok/s script as a merge gate for any config change.
7. **Single-device fp32, no DDP/amp.** Fine today; blocks everything above ~30M on Colab-class hardware. Add bf16 autocast + DDP before the 30M run.
8. **No quality filtering / no code-math-English replay** (§1.3). Lower priority than 1–7, but the cheapest data win after dedup: a FastText edu-filter + 5–10% code/math/English ablation.

Explicitly **not** gaps (already match playbook): SwiGLU 2.67× ✅, pre-RMSNorm ✅, no biases ✅, tied embeddings ✅, RoPE ✅, AdamW(0.9,0.95)/wd 0.1/clip 1.0 ✅, hash dedup + hash split ✅.

## 4. Concrete 30M / 100M configs (consistent with §7.1)

Shared: decoder-only, pre-RMSNorm, RoPE θ=10k (ctx ≤4K), SwiGLU 2.67×, GQA,
tied embeddings, `attention_bias=false`, dropout 0.0, bf16, AdamW β=(0.9,0.95),
wd=0.1, clip 1.0. Param formula (tied):
$N = Vd + L(2d^2 + 2d\,n_{kv}d_{head} + 3dd_{ff} + 2d) + d$.

| | **Gatra-30M** | **Gatra-100M** |
|---|---|---|
| Layers $L$ | 12 | 20 |
| $d_{model}$ | 384 | 576 |
| Q / KV heads ($d_{head}$=64) | 6 / 3 | 9 / 3 |
| $d_{ff}$ | 1024 (2.67×) | 1536 (2.67×) |
| Vocab (BPE, tied) | 32,768 | 49,152 |
| **Params** | **≈32.1M** (emb 12.6M + 19.5M) | **≈99.1M** (emb 28.3M + 70.8M) |
| Block / context target | 2048 (BPE) | 2048–4096 (BPE) |
| KV bf16/token | 9 KiB | 15 KiB |
| KV @4K | 36 MiB | 60 MiB |
| Token floor (Chinchilla 20×) | 640M | 2B |
| Token target (10k/param, §5.1) | **~300B** | **~1T** |
| LR / schedule | sweep 1–3e-3, WSD or cosine, warmup 2k | same (μP transfer from 30M) |
| Batch | ≥256k tokens (accumulate) | 1–2M tokens (DDP required) |

Notes:

- Depth 12/20 (not 30) is deliberate: at 30–100M the embedding (12–28M) eats a large
  share, so 30 layers would force $d$ so thin that $d_{head}=64$ breaks. Depth still
  rises 2–3× vs today's 6 layers, following the §3.1 direction within the embedding tax.
- Vocab 32k at 30M (not 49k): embedding is already 39% of the budget; 49k would
  leave too little for transformer blocks. Revisit 49k at 100M+ where it matches §7.1.
- The 100M tier is a stepping stone to the §7.1 125M tier (30L/d576/9Q-3KV/V49k);
  promote to it once the 1T-token data story (repetition ≤4 epochs over ≥250B unique
  tokens) is demonstrated.
- Data prerequisite dominates: run `gatra-build` over the full source list first.
  If unique ID bytes < target/4, the answer is more sources (FineWeb-2 full `ind_Latn`,
  CulturaX full, Leipzig, MADLAD-400 ID) or English/code replay — not a smaller target.
- Do not scale past 100M on byte tokens under any circumstances: at 4× inflation the
  1T-token target becomes 4T byte-tokens of compute for the same text.
