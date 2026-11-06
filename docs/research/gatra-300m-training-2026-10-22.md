---
title: "Gatra-300M Training Recipe: Pretrain + Post-Train + QAT + Export at ~3T Tokens"
tags: [gatra, pretraining, post-training, qat, gguf, indonesian-lm, distillation, eval]
---

# Gatra-300M Training Recipe (~3T tokens, Indonesia-first, llama.cpp GGUF Q4_K_M)

> [!note] Graph
> - [[../efficient-llm-architecture-edge-under-500m-2026-10-20|Efficient Edge Architecture (baseline)]] §§5, 7.2
> - [[pretrain-best-practices-sub500m-2026-10-22|Sub-500M Best Practices]] (validates/extends baseline §5)
> - [[gatra-repo-audit-2026-10-22|Gatra Repo Audit]] §1.2 (current: AdamW(0.9,0.95) wd 0.1, cosine 3e-4, warmup 100, batch 8k tokens, clip 1.0, dropout 0.1, fp32 single-device)

**Scope:** training recipe only (architecture fixed per baseline §7.1 tier: 30L / d896 / 14Q-4KV / d_ff 2432 / V49,152 tied / RoPE θ=100k / ctx 4K ≈ 302M params; KV 30 KiB/tok, 120 MiB @4K; Q4_K ≈ 170 MB). Every key claim carries one verbatim quote + one primary-source citation. Unverifiable items are marked `[OPEN]` and kept out of the defaults.

**Model definition (frozen for this recipe).** L=30, d=896, 14 query heads / 4 KV heads ($d_{head}$=64), SwiGLU $d_{ff}$=2432 (2.67×), vocab 49,152 tied, pre-RMSNorm, no biases, RoPE θ=100k, QK-norm ON, block 2048 (extend to 4096 late, §1). Param check: emb $49{,}152{\times}896 \approx 44.0$M; per layer $2d^2 + 2d{\cdot}4{\cdot}64 + 3dd_{ff} + 2d \approx 8.60$M; ×30 ≈ 258M; total ≈ **302M**.

---

## 1) Phase table (exact hyperparameters)

### P0 — Tokenizer + μP proxy (prerequisite, ~1 week on 1×A100)

| Item | Value |
|---|---|
| Tokenizer decision | Adopt SmolLM2 49,152 BPE **or** train own ID BPE-32k — decided by AB1 (§3). Default lean: adopt SmolLM2 vocab (unlocks logit-KD option, §6) |
| Fertility gate | BPE tok/word on held-out ID ≤ 2.0 (byte baseline 6.70, audit §2.1) |
| μP proxy | ~40M model, same arch family/vocab; sweep LR {1,2,3,4,6}e-3 + init scale + wd; ~4B tokens each |
| Transfer rule | μP: "tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model" — "by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning cost only 7% of total pretraining cost." ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466)) |
| LR default if sweep ties | **3e-3** (SmolLM2-small direct evidence, §1.1) |

### P1 — Pretraining (3T tokens, single-stage, WSD)

| Item | Value | Evidence |
|---|---|---|
| Schedule | **WSD**: 2k-step warmup → stable at peak → **linear decay to 0 over final 20%** | SmolLM2-135M/360M: "trained using the WSD scheduler with 20% decay and a learning rate of $3.0\times10^{-3}$." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)); WSD is "conducive to continuous training and domain adaptation" ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)). WSD beats cosine here operationally: a 1.5M-step run must support early-stop/continue without a fixed horizon (cosine forces one; `continue_train.py` exists for exactly this) |
| Peak LR | **3e-3** (sweep {2,3,4}e-3 on proxy; 4e-3 only if proxy-clean) | R1 shows 4e-3 pretrain LR with "linear decay to $0.1\times$ the peak value" after 2k warmup, and higher LR → higher RankMe utilization (21.98 vs 7.81) ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)). 3e-3 is the conservative pick at 300M/3T; 4e-3 is the aggressive arm |
| Warmup | **2,000 optimizer steps** (≈4B tokens @2M/step), **counted in tokens, not iters** | Both SmolLM2-1.7B ("2,000-step warmup phase") and R1 ("2k-step warmup schedule") use 2k ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737), [arXiv:2509.24945](https://arxiv.org/abs/2509.24945)). Current repo warmup (100 iters @8k tok = 0.8M tokens) is ~5,000× too short in token terms — rescale on any batch change |
| Global batch | **2M tokens/step** (range 1–2M; 1M for single-GPU pilots) | SmolLM2 ablations: "global batch size of approximately 2 million tokens." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)); R1 runs ≈4M/step at 128 GPUs. PaLM ramps "batch size 512 (1M tokens) … 1024 (2M tokens) … 2048 (4M tokens)" ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) |
| Optimizer | AdamW β=(0.9,0.95), wd=0.1, grad-clip 1.0 | Matches repo today; PaLM reference uses "global norm gradient clipping … 1.0" ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) |
| Dropout | **0.0** (repo 0.1 → change) | Pretraining hygiene per baseline §5.5; 0.1 only if SFT overfits |
| Precision | **bf16 autocast + DDP** (FSDP unnecessary — 300M fits one GPU); T4 pilots fp16+loss-scale (no bf16 on Turing) | bf16 "uses 2 bytes per model and optimizer parameter — effectively half the memory of fp32" (torchtune precision docs); grad-accum identity: `total = batch × accum × devices` ([HF TRL customization](https://huggingface.co/docs/trl/en/customization)) |
| Seq len | **2048** for first ~2.8T; **final ~200B tokens at 4096** (context-extension phase, RoPE θ=100k from step 0) | Precedent: SmolLM2-1.7B extends 2k→8k late with raised RoPE θ; R1 mid-trains at 4k. Product ctx cap is 4K (edge doc §7.2); 8K out of scope |
| Data mix (single-stage from step 0) | ~10% ID (FineWeb-2 `ind_Latn` + CulturaX-id + wiki-id + Aya-id + Kopi/news; ≤3–4 epochs of ~100B unique ID) / ~50% FineWeb-Edu+DCLM-60/40 / ~24% code (Stack-Edu-equiv) / ~14% math (FineMath/InfiMM-equiv) / ~4% synthetic (Cosmopedia-v2-equiv) | Single-stage wins at small scale: "these smaller models benefited from a single-stage training approach with consistently high-quality data" ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)); repetition cap: "up to 4 epochs of repeated data yields negligible changes to loss … with more repetition, the value of adding compute eventually decays to zero" ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)). ID unique (~60B words FineWeb-2 `ind_Latn`, [HF card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)) forces the 10% cap: 300B ID tokens ÷ ~100B unique ≈ 3 epochs — inside the cap; more ID% needs more unique ID sources, not more epochs |
| Stability kit | **QK-norm ON, z-loss 1e-4 ON, logit soft-cap ON** (hooks §4) | QK-norm: Qwen3 adds "QK-Norm to the attention mechanism to ensure stable training" at all sizes incl. 0.6B ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)); "QK-layernorm enables stable training across three orders of magnitude of learning rate (LR) variation" ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)). z-loss: "auxiliary loss of z_loss = 10⁻⁴·log²Z … which increases the stability of training" ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)). Expect spikes anyway (SmolLM2: "a noticeable loss spike … remained even after rewinding … most evaluation metrics recovered by the end of the stage") — ride them out, don't rewind-and-skip ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) |
| Steps / checkpoints | 3T ÷ 2M = **1.5M optimizer steps**; eval every ~1k steps; keep rolling-5 + best + every-50k; SIGTERM-safe save (extends `continue_train.py`: scheduler state + tokens_seen-based schedule) | Current `train.py` checkpoints only on `eval_interval` and stores no scheduler state — breaks under WSD + preemption (§4) |

### P2 — Distillation A/B at 50B (conditional, before full commit)

Default **MiniPLM-style data-side** (tokenizer-agnostic, "reducing the data demand by 2.4 times" ([arXiv:2410.17215](https://arxiv.org/abs/2410.17215))). Logit-KD only if AB1 adopts SmolLM2 vocab **and** the 50B A/B wins (§6 decision table). If teacher is strong, prefer Qwen3-style on-policy distillation over RL (1,800 vs 17,920 GPU-hours, better Pass@1 *and* Pass@64 — §6).

### P3 — Post-training (staged; NO RL)

| Stage | Data | Hyperparams |
|---|---|---|
| Alignment SFT | SmolTalk-style general (EN) **filtered** (drop function-calling/hard, per SmolLM2-small) + Aya-id curated + Cendol-Instruct (ID task NLP) + Cendol-Chat-v2-style ID chat | 2 epochs, batch 128, seq 4096, LR **3e-4**, AdamW wd 0.1→0; SFT recipe: "for 2 epochs, using a global batch size of 128, sequence length of 8192, and a learning rate of $3.0\times10^{-4}$", small models use "a filtered version of SmolTalk … to better align with the models' capacity" ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)). (R1's 5e-6 general-SFT LR is post-KD-mid-training regime, not from-scratch — do not copy blindly.) Cendol precedent: mT5-small **300M fully-finetuned at 3e-4** on Cendol-Instruct, competitive <1B vs 7B models ([IndoNLP/cendol](https://github.com/IndoNLP/cendol)) |
| Reasoning SFT | Math/science/code reasoning sets (EN ok — reasoning transfers), **1 epoch, AFTER alignment SFT** | Staged order mandatory: "Decoupling alignment and reasoning proves essential: a staged approach (Tulu first, then reasoning data) consistently outperforms joint training" ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)). Expectation at 300M — near the capacity floor (R1-140M MATH 4.8/GSM8K 3.7 vs R1-360M 19.2/23.8): **prioritize IFEval/instruction-following over MATH**; any MATH gain is upside, not the gate |
| DPO (last) | UltraFeedback-style (EN) + translated/RLAIF ID pairs `[OPEN: no major ID preference set found]` | "2 epochs with a learning rate of $1.0\times10^{-6}$, beta of 0.5, global batch size of 128, and sequence length of 1024 tokens" ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) |
| RL | **Skipped** | "small model trained with SFT data outperforms base+RL (final GSM8K 57.0 RL vs 74.0 SFT for 950M-base), and additional RL can degrade small models optimized with SFT" — "small models, being capacity-constrained, are better suited to imitating curated demonstrations than performing exploration" ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) |

### P4 — QAT int4 (5k steps) + GGUF export

Follow Gemma 3 exactly: "finetuning each model for a small number of steps, typically 5,000, using Quantization Aware Training (QAT) … probabilities from the non-quantized checkpoint as targets … per-channel int4, per-block int4" ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786)). Use **per-block int4** (llama.cpp path), data matched to pretrain+post distributions → export via `convert_hf_to_gguf` → `quantize Q4_K_M` → eval gates (§5).

---

## 2) Compute budget table

FLOPs ≈ $6ND = 6 \times 3.02{\times}10^8 \times 3{\times}10^{12} \approx$ **5.4e21** (pretrain; SFT/DPO/QAT add <2%).

| Item | Tokens | FLOPs | 1×A100-40G* | 8×A100-40G | 8×H100* |
|---|---|---|---|---|---|
| P0 μP sweep (8 cfg × 4B @40M) | 32B | 7.7e18 | ~21 h | — | — |
| AB1–AB3 pilots (§3, ~250B @300M) | 250B | 4.5e20 | ~52 d | ~7 d | ~2 d |
| P1 full pretrain 3T | 3T | 5.4e21 | ~625 d (infeasible → gate) | **~60–90 d** | **~20–30 d** |
| P3 SFT+reasoning+DPO (~10B) | ~10B | ~2e19 | ~2 d | hours | hours |
| P4 QAT 5k steps × 2M (+teacher fwd ≈2×) | ~10B equiv | ~4e19 | ~3–4 d | ~12 h | ~4 h |
| **Total** | | | **~16.5k GPU-h ≈ $25–35k** (spot ~$1.5–2/h) | **~10 weeks wall** | **~5.3k GPU-h ≈ $15–25k** (@~$3–4/h) |

\* Assumes ~100 TFLOPS sustained/GPU on A100 (~32% MFU — small-model realistic), ~300 TFLOPS on H100. Treat as ±30% engineering estimates, not benchmarks.
**Colab verdict:** T4 (no bf16, 65 TFLOPS peak → ~9 GPU-years for 3T) and single-A100 Colab are **proxy/pilot-only** (P0 + ≤50B pilots). The 3T run needs a cluster (Modal `--detach`, per repo routing) or staged funding — this is go/no-go gate G3 (§5). Checkpoint storage policy: bf16 weights 0.6 GB + fp32 master+m+v ≈ 3.6 GB + grads; keep rolling-5 + best + every-50k (~15 × ~4.5 GB ≈ **70 GB**), not every eval.

### Gradient-accumulation math (from today's 8k-token micro-batch to 1–2M)

`global_tokens = micro_seqs × seq_len × n_gpus × accum_steps` ([HF TRL](https://huggingface.co/docs/trl/en/customization)).

| Setup | micro/GPU | × GPUs | accum | Global batch |
|---|---|---|---|---|
| Today (10M cfg) | 32 × 256 = 8k | 1 | 1 | **8k** (baseline; ~250× under target) |
| Colab T4-16G pilot (fp16) | 2 × 2048 = 4k | 1 | 64–128 | 256–512k (smoke/pilot only) |
| Colab A100-40G pilot | 8 × 2048 = 16k | 1 | 64 | **~1M** |
| 8×A100-40G full run | 8 × 2048 = 16k | 8 | 16 | **~2M** ✓ |
| 8×H100-80G full run | 16 × 2048 = 32k | 8 | 8 | **~2M** ✓ |

Seq 2048 vs 4096: doubling seq halves micro-batch at fixed memory (KV/activations scale with seq) while attention-per-token cost rises; hence 2048 for the bulk + 4096 only for the final ~200B. RoPE θ=100k set from step 0 so no retune at extension.

### What breaks moving from fp32 single-device (repo today → recipe)

1. **Schedule units**: cosine-in-iters + warmup-100-iters silently changes meaning when batch grows 250× — convert everything to tokens (warmup_tokens, decay_tokens).
2. **Batch noise**: 8k-token steps at LR 3e-3 will diverge at 300M; LR 3e-3 is only valid *with* 1–2M-token batches + warmup + stability kit.
3. **Memory**: fp32 AdamW @300M ≈ 3.6 GB states + 1.2 GB weights before a single activation — bf16 halves this; without it, micro-batch collapses and accum-steps explode.
4. **T4 has no bf16** (Turing) → fp16 + loss scaling for pilots; bf16 only on Ampere+ (A100/Colab-Pro).
5. **Checkpoint/resume**: `continue_train.py` restores model+optimizer+RNG but no scheduler state and `cosine_lr` is step-indexed — under WSD + 1.5M steps + preemption (spot/Colab), add scheduler state + tokens_seen-indexed schedule + SIGTERM hook + step-based (not only eval-based) saves.
6. **FSDP is NOT needed** (300M fits one GPU); DDP + grad-accum is the whole distribution story.

---

## 3) Ablation plan (max 3; total ≤ ~250B tokens, <8% of budget)

**AB1 — Tokenizer / KD-option (2 × 50B pilots).** Q: adopt SmolLM2-49k vocab vs own ID-BPE-32k? Arms: (a) adopted-49k, (b) own-32k trained on ID-heavy corpus. Metrics: fertility on held-out ID, 50B pilot ppl, downstream delta. Decision: pick (a) iff fertility within 10% of (b) — (a) unlocks SmolLM2-1.7B logit-KD (§6); else (b) + data-side distillation only. Cost ~100B.
**AB2 — Data mix (4 × 25B pilots, fractional).** Q: ID% {5, 15} × code/math {off, on} at fixed 10%-ID-center later. Metrics: ID-eval (IndoMMLU-primary, hellaswag_id) + EN sanity (hellaswag_en) + ppl. Decision: highest ID-eval without EN regression; sets P1 mix. Cost ~100B.
**AB3 — Distillation mode (≤3 × 50B, only the winner arm of AB1/AB2).** Q: none vs MiniPLM-data-side vs logit-KD (logit arm only if AB1→adopted vocab + same-tokenizer teacher). Decision rule: adopt iff >2pp ID-eval or >3% rel-ppl win — else skip (MobileLLM precedent: KD "comparable to or even lower" accuracy at "2.6−3.2× slower" training ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905))). Cost ≤50B (run only 2 arms if logit-KD ineligible).

---

## 4) Code-change checklist

### `src/gatra/config.py` — new fields (all with defaults preserving current behavior)

- `[schedule]`: `type="wsd"`, `warmup_tokens=4e9`, `decay_frac=0.2`, `min_lr=0.0` (replaces iter-based cosine; keep `cosine` as legacy option).
- `[batch]`: `global_tokens=2_000_000`, `micro_batch_seqs=8`, `accum_steps` (auto-derived = global ÷ (micro×seq×world), validated divisible).
- `[precision]`: `bf16=true`, `ddp=true`, `fp16_fallback=true` (T4).
- `[model]`: `n_kv`, `qk_norm=true`, `z_loss=1e-4`, `logit_cap=30.0`, `rope_base=100000`, `vocab_size=49152`.
- `[tokenizer]`: `name="smollm2-49k"|"id-bpe-32k"`, `path=...` (AB1 outcome).
- `[qat]`: `steps=5000`, `teacher="best_bf16"`, `format="per-block-int4"`.
- `[eval]`: harness task list + gate thresholds (§5).

### `src/gatra/train.py` — surgical changes

1. Replace `cosine_lr(step)` with `wsd_lr(tokens_seen)`: linear warmup over `warmup_tokens`, stable, linear decay to `min_lr` over final `decay_frac` of planned tokens; LR set per **optimizer step** (not micro-step).
2. Grad-accum loop: `loss/accum_steps`, backward each micro-step, `clip_grad_norm_` + `optimizer.step()` + schedule-step every `accum_steps`; `tokens_seen += global_tokens` per optimizer step.
3. bf16: `torch.autocast("cuda", torch.bfloat16)` around forward (fp16+GradScaler only when `fp16_fallback`); keep AdamW fp32 states (default).
4. DDP: `torchrun` entry, `DistributedDataParallel`, rank-0-only logging/checkpoint/metrics, per-rank RNG offset (`seed + rank`).
5. z-loss: `loss += cfg * logsumexp(logits).mean()²` — chunked over seq (`[OPEN: profile 49k-vocab logsumexp memory @2048 before enabling every step; fallback every-k-steps]`).
6. Checkpointing: save every N optimizer steps (default 500) + SIGTERM/SIGUSR1 handler; include scheduler state + `tokens_seen`; extend `continue_train.py` to restore all three (it already handles model+optimizer+RNG).
7. Dropout default 0.0 for pretrain configs; LR default 3e-3 (from P0 sweep, not hardcoded 3e-4).
8. Deterministic data order under DDP (`DistributedSampler`-equivalent shard offsets in `data.py`).

### `src/gatra/data.py` — the two blockers + scale

1. **Remove `ByteTokenizer`-only assert**; support BPE (`uint16` bins for V<65k, meta `tokenizer` field, mix-mismatch check extended to tokenizer id). This is the single biggest blocker — nothing else works without it.
2. **Streaming for 3T**: `gatra-build` pre-tokenizing 3T BPE tokens (6–12 TB bins) is infeasible — add on-the-fly tokenize-from-shards dataset (compressed JSONL shards, deterministic shuffle by doc-hash, per-source epoch counters). Keep `build_bins` for ≤100B pilots.
3. Per-source token accounting + repetition guard: log tokens/source/epoch, hard-fail if any source exceeds 4 epochs (Muennighoff cap).
4. Add mix slots: English replay (FineWeb-Edu/DCLM), code, math, synthetic — weights from AB2.

### `src/gatra/model.py` — hooks (all behind config flags, default ON for 300M)

1. **GQA**: `n_kv` heads (`q`: `d→d`, `k/v`: `d→n_kv·64`), `repeat_interleave(n_head//n_kv)` (require divisibility: 14/4 → use 14Q/7KV? No — 14Q/**2KV** (Qwen2.5-0.5B pattern) or 12Q/4KV; **fix heads to 12Q/4KV @d768? No.** Keep d896/14Q: KV ∈ {2, 7, 14}. Pick **14Q/7KV** (GQA 2:1, KV 43 KiB/tok… recompute: 2·30·7·64·2 = 53,760 B = 52.5 KiB/tok, 210 MiB @4K — acceptable) or **14Q/2KV** (15 KiB/tok, 60 MiB @4K, Qwen-proven). Default **14Q/2KV** for edge KV; ablate 7KV only if quality gap appears. (Audit's 15Q/5KV assumed d960; at d896 use divisors of 14.)
2. **QK-norm**: `RMSNorm(64)` on q,k after projection, before RoPE (Qwen3 placement).
3. **Logit soft-cap**: `tanh(logits/cap)*cap` (cap≈30) before softmax/loss (Gemma-2-style; cheap insurance, no controlled sub-500M study — flag `[UNVERIFIED-standalone, kept as insurance]`).
4. RoPE θ from config (100k); block_size 2048→4096 switch without re-init (buffer sized to max).

---

## 5) Eval harness + go/no-go gates

Harness: `lm-evaluation-harness` + custom scripts, run as **merge gates** (not in-loop except ppl).

| # | Signal | Tool | Gate |
|---|---|---|---|
| G0 | Tokenizer fertility (ID held-out) | tok/word script | ≤2.0 (byte=6.70); else AB1 arm (b) |
| G1 | μP sweep clean | 40M proxy curves | No divergence at chosen LR; RankMe non-collapsed |
| G2 | 50B pilot | ppl + hellaswag_id (okapi `hellaswag_id`, [harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/okapi/hellaswag_multilingual/README.md)) | Within scaling-law prediction ±10%; no irrecoverable spike |
| G3 | Data audit | `gatra-build --audit` (new: unique tokens/source, epoch projection) | ≥750B unique total (3T ÷ 4-epoch cap); ID unique ≥100B for 10% mix; **NO-GO to full run if unmet** — cut budget, don't exceed 4 epochs |
| G4 | Distillation A/B | AB3 | Adopt iff >2pp ID-eval or >3% ppl; else train supervised (Busbridge rule) |
| G5 | Post-train | IndoMMLU-primary ([14,981 Qs, 64 tasks](https://aclanthology.org/2023.emnlp-main.760/); note GPT-3.5 "only manages to pass the Indonesian primary school level" — calibrate expectations), NusaX-senti-id ([parallel 10-local-language sentiment](https://aclanthology.org/2023.eacl-main.57/)), IFEval-style ID | Monotonic vs base; IFEval prioritized over MATH (capacity floor, §P3) |
| G6 | QAT + export | ppl-delta bf16→int4; `llama.cpp` Q4_K_M tok/s + RSS | ppl <3% rel degradation, downstream <2pp drop; **≥30 tok/s sustained** (non-burst) + RSS <600 MB @4K on reference device |

---

## 6) Distillation decision (teacher table)

Constraint: logit-KD requires **identical tokenizers**; data-side (MiniPLM) does not.

| Teacher | ID capability | Tokenizer | Verdict |
|---|---|---|---|
| SmolLM2-1.7B (Apache-2.0) | Weak (EN-centric) | 49,152 — adoptable (AB1-a) | Logit-KD A/B eligible; expect small ID lift |
| Qwen2.5-0.5B/1.5B (Apache-2.0) | ID supported | 151,936 | **Reject for logit-KD**: emb tax 152k×896 ≈ 136M params ≈ **45% of a 300M budget** (same arithmetic as baseline §3.6: 27% @500M). Data-side only |
| Llama-3.1-8B-Instruct (R1's KD teacher) / Qwen3 / Cendol-LLaMA | Good | 128k / mismatched | Cross-tokenizer → **data-side only** |
| `[OPEN]` ID-strong teacher sharing adopted vocab | — | — | None identified; if AB1→own-BPE, logit-KD is off the table entirely |

Rule (Busbridge): "if only one student is to be distilled and a teacher also requires training, supervised learning is generally preferable" — all candidate teachers exist, so the A/B is legitimate; but MobileLLM's negative result at 125M/350M same-tokenizer regime means the prior is **skip unless AB3 wins** ([arXiv:2502.08606](https://arxiv.org/abs/2502.08606), [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)).

---

## References (primary only; all accessed 2026-10-22)

- SmolLM2 — https://arxiv.org/abs/2502.02737
- MobileLLM — https://arxiv.org/abs/2402.14905
- MobileLLM-R1 — https://arxiv.org/abs/2509.24945
- MiniCPM + WSD — https://arxiv.org/abs/2404.06395
- PaLM (z-loss, clip, batch ramp) — https://arxiv.org/abs/2204.02311
- Qwen3 (distill≫RL, QK-norm) — https://arxiv.org/abs/2505.09388
- Gemma 3 (QAT 5k) — https://arxiv.org/abs/2503.19786
- Gemma 2 (logit-KD) — https://arxiv.org/abs/2408.00118
- MiniPLM (2.4×) — https://arxiv.org/abs/2410.17215
- Distillation scaling laws — https://arxiv.org/abs/2502.08606
- Data-Constrained (4 epochs) — https://arxiv.org/abs/2305.16264
- Sardana overtraining — https://arxiv.org/abs/2401.00448
- Chinchilla — https://arxiv.org/abs/2203.15556
- μP — https://arxiv.org/abs/2203.03466
- Small-scale instabilities/QK-norm — https://arxiv.org/abs/2309.14322
- FineWeb-2 card — https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- Cendol — https://github.com/IndoNLP/cendol
- IndoMMLU — https://aclanthology.org/2023.emnlp-main.760/
- NusaX — https://aclanthology.org/2023.eacl-main.57/
- lm-eval-harness okapi ID tasks — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/okapi/hellaswag_multilingual/README.md
- HF TRL (bf16/grad-accum) — https://huggingface.co/docs/trl/en/customization
