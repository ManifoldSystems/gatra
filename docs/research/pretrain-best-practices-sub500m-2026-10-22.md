---
title: "Pretraining + Post-Training Best Practices for Small LMs Under 500M"
tags: [pretraining, post-training, small-language-models, distillation, data-curation, hyperparameters, stability, quantization]
aliases:
  - Sub-500M Pretraining Best Practices
  - Best Practices Pretraining LM Kecil
---

# Pretraining + Post-Training Best Practices for Small LMs (<500M)

> [!note] Konteks
> Validasi + ekstensi §5 dari [[../efficient-llm-architecture-edge-under-500m-2026-10-20|Efficient LLM Architecture Edge <500M (2026-10-20)]].
> Induk: [[pretrain_llms/gatra/README|Gatra: Indonesia-first small LM]]. Topik terkait: [[llm_architecture/frontier-efficient-llm-architectures-2026|Frontier Efficient Architectures]], [[quantization-precision/README|Quantization & Precision]].

**Scope:** pretraining + post-training only (architecture §1–4 and deployment §6 of the baseline doc are out of scope except where training interacts with them). Every key claim carries one verbatim quote + one primary-source URL. Claims without a citable primary source are marked `[UNVERIFIED]` and excluded from recommendations.

---

## 1. Validated claims (baseline doc §5 holds)

### 1a. Token budgets: Chinchilla baseline + overtraining regime

- **Chinchilla = 20 tok/param compute-optimal.** Hoffmann et al. train "from 70 million to over 16 billion parameters on 5 to 500 billion tokens" and find "for every doubling of model size the number of training tokens should also be doubled." ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556))
- **Beyond-Chinchilla: no saturation up to ~10,000 tok/param.** Sardana et al. train "47 models ranging from 150M to 6B parameters, on various data budgets from 10 to 10,000 tokens per parameter. We find that model quality continues to improve as we scale token ratios. We do not find evidence of a 'saturation point,' beyond which models do not improve, even with additional data." ([arXiv:2401.00448](https://arxiv.org/abs/2401.00448))
- **Caveat on fitting:** "when fitting Chinchilla coefficients using only typical token ratio runs, this formula overestimates the impact of additional training data as we move to the extreme ratio regime." Same source. So treat 4T/10k-per-param as a validated *default*, not a law.
- **Repetition limit ≈4 epochs.** Muennighoff et al.: "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero." ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264))
- **Observed budgets (all validated):** SmolLM2-135M 2T / 360M 4T ([HF card 135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [HF card 360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)); Qwen2.5 family "scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens" ([arXiv:2412.15115](https://arxiv.org/abs/2412.15115)) — family-level, per-size split unpublished; Qwen3 family "pre-trained on 36 trillion tokens covering up to 119 languages and dialects" ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) — same family-level caveat; MobileLLM-R1 "pre-training with 4.2T tokens on the dataset resampled from these ~2T tokens" ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)); phi-1-small "a smaller model with 350M parameters trained with the same pipeline as phi-1 that still achieves 45% on HumanEval" with phi-1 using "textbook quality data from the web (6B tokens) and synthetically generated textbooks and exercises with GPT-3.5 (1B tokens)" ([arXiv:2306.11644](https://arxiv.org/abs/2306.11644)).
- **WSD enables the overtraining regime.** MiniCPM: "we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler (LRS), conducive to continuous training and domain adaptation" and "derive the much higher compute optimal data-model ratio than Chinchilla Optimal." ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395))

### 1b. Data mix: quality > volume; single-stage for small models

- **Curation matters more at small scale.** SmolLM2: "data curation has an especially outsized influence for smaller models, as their limited capacity must be carefully optimized." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737))
- **Single-stage wins at 135M/360M; multi-stage only for 1.7B.** "Unlike SmolLM2-1.7B, where we leveraged a multi-stage training strategy, these smaller models benefited from a single-stage training approach with consistently high-quality data. We incorporated Stack-Edu from the start, alongside InfiMM-WebMath, FineMath, and Cosmopedia." Same source.
- **60/40 FineWeb-Edu/DCLM web mix (stage 1, 1.7B).** "We adopted a 60% FineWeb-Edu and 40% DCLM ratio for web data" (stage 1, 0→6T). Same source.
- **Decay mix (1.7B reference):** "English web data at 58% … and Cosmopedia v2 at 4%, which provides 30B tokens of high-quality synthetic textbooks" with math 14% total and Stack-Edu 24%. Same source.
- **FineWeb-Edu effect size.** "FineWeb-Edu achieves a 33.6% accuracy on the MMLU benchmark at only 38 billion tokens, significantly outperforming Matrix … which reaches similar accuracy at 300 billion tokens" and at 1.71B/350B scale "MMLU score increases from 33% to 37% … ARC score goes from 46% to 57%." ([arXiv:2406.17557](https://arxiv.org/abs/2406.17557))
- **DCLM effect size.** "DCLM-Baseline, enables training a 7B parameter language model from scratch to 64% 5-shot accuracy on MMLU with 2.6T training tokens" from a "standardized corpus of 240T tokens extracted from Common Crawl." ([arXiv:2406.11794](https://arxiv.org/abs/2406.11794))
- **Cosmopedia volume.** v0.1: "dataset of synthetic textbooks, blogposts, stories, posts and WikiHow articles generated by Mixtral-8x7B-Instruct-v0.1 … over 30 million files and 25 billion tokens." ([HF card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)) Note v2 used by SmolLM2 = 30B tokens (see §2 correction).
- **TinyStories.** "TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters) … yet still produce fluent and consistent stories." ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759))
- **FineWeb-2 ind_Latn volume (validated).** Card lists `ind_Latn`: 100,238,529 documents, 348.65GB UTF-8, 60,264,322,142 words. ([HF card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)) Token estimate ≈80–100B depends on tokenizer; the baseline doc's "±80–100B token estimasi" is reasonable but tokenizer-dependent — measure with your own tokenizer, do not quote as fact.

### 1c. Distillation: conditional, exactly as baseline doc states

- **Gemma 2 logit-KD positive (2B scale).** "distilling from a larger model improves performance compared to training from scratch. Note that 500B is 10× more than the compute-optimal number of tokens for a 2B model" with ablation "Average (3 bench.) 60.3 → 67.7" and PPL "200M: 23→21, 400M: 19→17, 1B: 17→15" from a 7B teacher. ([arXiv:2408.00118](https://arxiv.org/abs/2408.00118))
- **MobileLLM logit-KD negative (125M/350M).** "adding KD loss is comparable to or even lower than solely using the next token as labels. However, it's noteworthy that the training time using KD is 2.6−3.2× slower than training from scratch using labels." ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905))
- **Distillation scaling laws (Busbridge).** "In settings involving many students or an existing teacher, distillation outperforms supervised learning up to a compute level that scales predictably with student size. Conversely, if only one student is to be distilled and a teacher also requires training, supervised learning is generally preferable." ([arXiv:2502.08606](https://arxiv.org/abs/2502.08606)) Fit uses teachers NT∈{546M,…,7.75B}; students below ~500M are extrapolation (dashed lines in Fig. 1), so **no validated law exists for sub-500M students** — baseline doc's claim stands.
- **MiniPLM data-side distillation.** "MiniPLM enhances pre-training data utilization, reducing the data demand by 2.4 times." ([arXiv:2410.17215](https://arxiv.org/abs/2410.17215)) The baseline doc's "50B ≈ 118B, 200MB vs 30PB logits" specifics were **not** re-verified here — treat as `[UNVERIFIED]`, keep only the 2.4× figure.
- **Qwen3: distill >> RL for small models.** "directly distilling the output logits from teacher models into lightweight student models … achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method" with better Pass@1 *and* Pass@64. ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388))

### 1d. Schedules + hyperparameters

- **SmolLM2-small exact recipe.** "trained using the WSD scheduler with 20% decay and a learning rate of 3.0×10⁻³" (135M/360M); 1.7B uses "2,000-step warmup phase … peak learning rate of 5.0×10⁻⁴ (stable phase) … reducing the learning rate to zero over 10% of the total training steps." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737))
- **MobileLLM cosine baseline.** "The initial learning rate is set to 2e-3 and follows a cosine learning-rate decay strategy" with "Adam … weight decay of 0.1." ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905))
- **MobileLLM-R1 high-LR evidence.** Pretrain LR 4.0E-03 with "linear decay to 0.1× the peak value" after 2k warmup; mid-training LR 3.60E-04 "decays linearly to zero"; post-training Adam "with zero weight decay." LR ablation: pretrain 4e-3 → post-mid-training MMLU 36.31 vs 4e-4 → 27.55, with RankMe 21.98 vs 7.81 — "higher learning rates lead to higher representational rank … a larger fraction of parameters are effectively utilized." ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945))
- **AdamW (0.9, 0.95).** SmolLM2-1.7B: "AdamW optimizer with (β,β₂)=(0.9,0.95)"; MobileLLM-R1: "(β₁,β₂,ϵ)=(0.9,0.95,10⁻⁸) and weight decay 0.1." Same sources as above.
- **Batch scale.** SmolLM2 ablations: "global batch size of approximately 2 million tokens." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) PaLM reference schedule ramps "batch size 512 (1M tokens) … 1024 (2M tokens) … 2048 (4M tokens)" with "global norm gradient clipping … 1.0." ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311))
- **μP transfer.** "by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning cost only 7% of total pretraining cost." ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466))

### 1e. Stability at small scale

- **Instabilities reproduce at small scale at high LR.** "these instabilities also appear in small models when training at high learning rates, and … mitigations previously employed at large scales are equally effective in this regime." Methods studied: "warm-up, weight decay, and the μParam." ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322))
- **QK-norm.** "QK-layernorm enables stable training across three orders of magnitude of learning rate (LR) variation." Same source. Production confirmation: Qwen3 adds "QK-Norm to the attention mechanism to ensure stable training for Qwen3" at *all* sizes incl. 0.6B. ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388))
- **z-loss.** PaLM: "auxiliary loss of z_loss = 10⁻⁴·log²Z to encourage the softmax normalizer log(Z) to be close to 0, which we found increases the stability of training." ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311))
- **Loss spikes happen even in SOTA runs.** SmolLM2 stage 3: "we observed a noticeable loss spike during this phase which remained even after rewinding training and skipping data associated with the spike. The exact cause remains undetermined but most evaluation metrics recovered by the end of the stage." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737))

### 1f. Post-training for small models

- **SFT exact (1.7B; small models use filtered variant).** "supervised fine-tuning … for 2 epochs, using a global batch size of 128, sequence length of 8192, and a learning rate of 3.0×10⁻⁴." For 135M/360M: "SFT using a filtered version of SmolTalk, removing complex instruction-following tasks (e.g., function calling) and hard examples from MagPie-Ultra to better align with the models' capacity." ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737))
- **DPO exact.** "UltraFeedback proved the most consistently effective … We trained for 2 epochs with a learning rate of 1.0×10⁻⁶, beta of 0.5, global batch size of 128, and sequence length of 1024 tokens." Same source.
- **Reasoning-SFT capacity floor.** Same-corpus comparison (OpenMathReasoning + OpenScienceReasoning-2 + OpenCodeReasoning-2, 1 epoch): 140M MATH 4.8 / GSM8K 3.7 vs 360M MATH 19.2 / GSM8K 23.8; base checkpoints show the same gap (140M-base MATH500 4.6/GSM8K 16.3 vs 360M-base 13.4/39.4). "Decoupling alignment and reasoning proves essential: a staged approach (Tulu first, then reasoning data) consistently outperforms joint training." ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) Baseline doc's "±300M floor" holds.
- **SFT > RL at small scale; RL can hurt after SFT.** "small model trained with SFT data outperforms base+RL (final GSM8K 57.0 RL vs 74.0 SFT for 950M-base), and additional RL can degrade small models optimized with SFT." Figure 10 + §D.2, same source. Mechanism per authors: "small models, being capacity-constrained, are better suited to imitating curated demonstrations rather than performing exploration directly."
- **Two-stage post-training order.** "(1) Instruction-following supervision (Tulu-SFT) provides crucial alignment signals that make subsequent reasoning adaptation significantly more effective than starting directly with reasoning data … (4) Decoupling alignment and reasoning proves essential." Same source.

### 1g. QAT int4 post-pretraining

- Gemma 3: "quantized versions … obtained by finetuning each model for a small number of steps, typically 5,000, using Quantization Aware Training (QAT). We use probabilities from the non-quantized checkpoint as targets" covering "per-channel int4, per-block int4, and switched fp8." ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786)) Baseline doc's "±5.000 step" holds.

---

## 2. Corrected / updated claims

| # | Baseline doc claim | Verdict + fix |
|---|---|---|
| C1 | §5.1 table: "Qwen2.5-0.5B 18T*, Qwen3-0.6B 36T*" with family-level footnote | **Confirmed, keep asterisk.** Qwen2.5 report states 7T→18T; Qwen3 states "approximately 36 trillion tokens" with "three-stage strategy … 30T … 5T … long-context" — neither breaks out per-size allocation. ([arXiv:2412.15115](https://arxiv.org/abs/2412.15115), [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) |
| C2 | §5.2 "Cosmopedia: 25B token sintetis dari Mixtral-8x7B" | **Version split.** 25B = v0.1 ([HF card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)); SmolLM2-1.7B decay uses "Cosmopedia v2 … 30B tokens." Use v2/30B for new recipes. ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) |
| C3 | §5.2 "FineWeb-2 … ind_Latn ±100,2M dokumen / 348,65GB … (±80–100B token estimasi)" | **Volume confirmed, token count tokenizer-dependent.** Card: 100,238,529 docs / 348.65GB / 60.26B *words*. Token estimate only valid after measuring with your tokenizer. ([HF card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)) |
| C4 | §5.3 "fitted student ≥546M, tidak ada law untuk sub-500M" | **Confirmed with precision.** Busbridge fits teachers NT∈{546M,…,7.75B}; sub-500M *student* behavior is dashed-line extrapolation in their Fig. 1. Decision rule "A/B 50B token before commit" stands. ([arXiv:2502.08606](https://arxiv.org/abs/2502.08606)) |
| C5 | §5.3 "MiniPLM … 50B ≈ 118B (2,36×) … 200MB vs 30PB logits" | **Partially verified.** Only "reducing the data demand by 2.4 times" re-verified. Storage-ratio specifics `[UNVERIFIED]` — drop from recipe until checked against the MiniPLM paper text. ([arXiv:2410.17215](https://arxiv.org/abs/2410.17215)) |
| C6 | §5.4 "Tidak ada hasil ORPO/GRPO terkontrol di ≤500M" | **Outdated — one GRPO ablation now exists (at 950M).** MobileLLM-R1 §D.2: GRPO on NuminaMath-TIR, "LR 3×10⁻⁶, 100 steps, batch 32 prompts, 8 generations, KL β=0"; result SFT>RL, RL-after-SFT degrades. Still **no ORPO** and **no ≤500M GRPO** found. ([arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) |
| C7 | §5.5 "QK-norm … bukti empirisnya di 8B/22B, jadi di ≤500M … asuransi murah" | **Upgrade to default.** Qwen3 ships QK-Norm at all sizes incl. 0.6B "to ensure stable training"; MobileLLM-R1 ships QK-norm at 140M/360M/950M. Adopt QK-norm by default for new sub-500M runs. ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388), [arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) |
| C8 | §5.5 table: "LR 3,45e-4 (Sardana 370M)", "Dropout 0,0/0,1", "Sardana batch" | **`[UNVERIFIED]` — remove from recipe.** Not re-verified against a primary source in this pass. Replacements: SmolLM2-small 3e-3/WSD-20%, R1 4e-3/linear-to-0.1×, MobileLLM 2e-3/cosine (all cited in §1d). |
| C9 | §5.4 "Sardana … " (implicit: overtraining numbers) | **No change needed**, but add the fitting caveat: Chinchilla coefficients fit at ≤~100 tok/param "overestimates the impact of additional training data" at extreme ratios. Budget 4T as default, verify with ablations. ([arXiv:2401.00448](https://arxiv.org/abs/2401.00448)) |
| C10 | §7.1 "RoPE θ=100k (8K)" for 350M tier | **Approximate.** SmolLM2-1.7B extends 2k→8k with "RoPE value of 130k"; 360M card θ not re-verified. Keep 100k as starting point, ablate 100k vs 130k. ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) |
| C11 | §3.1 "optimal depth near 30 layers" vs R1 15-layer 140M/360M | **Nuance, not contradiction.** MobileLLM depth sweep (commonsense, 1T tokens, cosine) favors ~30L; MobileLLM-R1 reasoning recipe uses 15L/576/2048 (140M) and 15L/1024/4096 (360M) with QK-norm + 128k vocab + 4.2T tokens and wins on reasoning. Depth rule is objective-dependent; for a reasoning-targeted 100M model, A/B depth (e.g. 15L-wide vs 24–30L-thin) instead of assuming 30L. ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905), [arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) |
| C12 | §4 table scores (MMLU/GSM8K/HumanEval per model) | **Carried over, not re-verified except:** SmolLM2-360M card (HellaSwag 54.5, ARC 53.0, MMLU-cloze 35.8, GSM8K 3.2) and 135M card (HellaSwag 42.1, ARC 43.9, MMLU 31.5, GSM8K 1.4) confirmed; R1 Table 8/9 supersedes several numbers (e.g. SmolLM2-360M MATH500 1.8/GSM8K 5.0 base; R1-360M-base 13.4/39.4). Do not mix harnesses. ([HF cards](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [arXiv:2509.24945](https://arxiv.org/abs/2509.24945)) |
| C13 | §5.1 "MiniCPM4-0.5B 1T", §4 "LFM2 10T / LFM2.5 28T / Granite 15T / Falcon token n/d" | **`[UNVERIFIED]` in this pass.** MiniCPM paper covers 1.2B/2.4B (non-embedding) + WSD, not MiniCPM4-0.5B's 1T. Keep in tables only with explicit "not verified" flag. ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)) |

---

## 3. Concrete recipe deltas for a 100M Indonesian-first model (Gatra)

Assumes dense transformer, tied embeddings, GQA, pre-RMSNorm, RoPE, SwiGLU (arch per baseline §7). Deltas below are *training-only* changes vs baseline §7.2 (which targets 350M/4T).

1. **Budget: 1T tokens (≈10k tok/param), ≥250B unique.** 100M × 10,000 = 1T (Sardana regime, §1a); ÷4-epoch cap (Muennighoff, §1a) → ≥250B unique tokens. FineWeb-2 `ind_Latn` (~60B words / ~80–100B tokens tokenizer-dependent, §1b) covers at most ~40% of unique need at 5–15% mix → the rest must be English replay (FineWeb-Edu/DCLM 60/40) + code/math. If unique <250B, cut budget before exceeding 4 epochs.
2. **Single-stage curated mix from step 0.** Per SmolLM2-small (§1b): no multi-stage phasing; include Stack-Edu-equivalent code + FineMath/InfiMM-WebMath-equivalent math + Cosmopedia-v2-style synthetic from the start; filter DCLM-style web with an edu-classifier (drop score-0, downsample 1–2). Re-run the mix ablation at 100M scale — do not copy 1.7B stage ratios.
3. **Distillation order: data-side first, logit-KD only by A/B.** Default to MiniPLM-style corpus refinement (2.4× data-demand reduction, §1c). Logit-KD only if (a) teacher exists (no teacher-training cost, Busbridge rule) AND (b) same tokenizer; A/B 50B tokens, expect negative if student already overtrained or tokenizer mismatched (MobileLLM result, §1c). If teacher is strong: prefer Qwen3-style off-policy→on-policy logit distillation over RL (1/10 GPU hours, better Pass@1 *and* Pass@64, §1c).
4. **Schedule: WSD + 20% decay, peak LR 3–4e-3, warmup 2k.** SmolLM2-small 3e-3/WSD-20% (§1d); R1 shows 4e-3 > 2e-3 > 1e-3 > 4e-4 post-mid-training with RankMe tracking (21.98 vs 7.81). Sweep {2e-3, 3e-3, 4e-3} on a ≤40M μP proxy (7%-cost transfer, §1d), then one 100M confirmation. Batch ≈1–2M tokens (SmolLM2 ≈2M; PaLM ramp 1M→2M→4M, §1d). AdamW (0.9, 0.95), wd 0.1 pretrain / 0 post-training, clip 1.0.
5. **Stability kit: QK-norm ON, z-loss 1e-4 ON, soft-cap optional.** QK-norm now default at 140M–0.6B in production (C7); z-loss 1e-4·log²Z (PaLM, §1e); expect and ride out spikes (SmolLM2 recovered despite undetermined cause, §1e) — rewind-and-skip is not guaranteed to help.
6. **Post-training: capacity-filtered SFT → DPO; reasoning-SFT minimal at 100M.** SFT on SmolTalk-style mix minus function-calling/hard items, 2 epochs, batch 128, seq 8192, LR 3e-4 (§1f); DPO UltraFeedback 2 epochs, LR 1e-6, β=0.5, batch 128, seq 1024. At ~100–140M expect MATH single-digits even after reasoning-SFT (R1-140M 4.8 vs R1-360M 19.2, §1f) → prioritize instruction-following/IFEval over math; staged Tulu-first-then-reasoning if reasoning attempted. **No RL** (SFT>RL, RL-after-SFT degrades, §1f).
7. **QAT int4 ≈5k steps, then export.** KD-style targets from the bf16 checkpoint, data matched to pretrain+post distributions, per-block int4 for llama.cpp (Gemma 3 recipe, §1g).

---

## 4. Open questions

1. **Distillation scaling below 500M students.** Busbridge fits teachers ≥546M; sub-500M student behavior is extrapolation. Need: KD-vs-supervised IsoFLOP at 100–360M with Indonesian mix. ([arXiv:2502.08606](https://arxiv.org/abs/2502.08606))
2. **Indonesian data quality transfer.** FineWeb-2 ablations cover 9 languages (zh/fr/ar/ru/th/hi/tr/sw/te) — Indonesian not among them; Swahili (low-resource) filtered data underperformed dedup-only data (~1B tokens). Need: id_Latn ablation (filtered vs filtered+removed, rehydration weights on/off). ([HF card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2))
3. **Depth vs width for reasoning at 100M.** MobileLLM commonsense optimum (~30L) vs R1 reasoning recipe (15L, wider, QK-norm, 4.2T). Need: 15L-wide vs 24–30L-thin A/B on reasoning + commonsense at 100M/1T. ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905), [arXiv:2509.24945](https://arxiv.org/abs/2509.24945))
4. **ORPO / GRPO at ≤500M.** One GRPO ablation at 950M (SFT>RL); none at ≤500M; no ORPO found. Need: GRPO/ORPO vs DPO at 100–360M. (C6)
5. **QAT ablation at ≤500M.** Gemma 3 QAT recipe validated at 1B+; 270M INT4 claim is a product statement ("minimal performance degradation") without published ablation deltas. Need: bf16 vs PTQ vs QAT-5k perplexity + downstream at 100–360M. ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786))
6. **Vocab/fertility for Indonesian.** 32–49k BPE sweet-spot claim is Latin-general; no id fertility measurement cited. Need: fertility + downstream A/B (32k vs 49k vs 128k) on id-heavy mix. (C12-adjacent; no primary source found)
7. **LR/batch scaling at 100M with id-heavy mix.** 3–4e-3 evidence comes from English/code/math mixes; μP transfer to id-heavy 100M untested. Need: proxy sweep with the actual mix. ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466))
8. **MoE / MTP / early-exit / fp8 at ≤500M.** Still no controlled study found — remains "skip" per baseline §7.3. Any adoption needs a from-scratch A/B, not an import from large-scale literature.

---

## References (primary only; all accessed 2026-10-22)

- SmolLM2 — https://arxiv.org/abs/2502.02737 ; cards: https://huggingface.co/HuggingFaceTB/SmolLM2-360M , https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Chinchilla — https://arxiv.org/abs/2203.15556
- Beyond-Chinchilla (Sardana) — https://arxiv.org/abs/2401.00448
- Data-Constrained (Muennighoff) — https://arxiv.org/abs/2305.16264
- MiniCPM + WSD — https://arxiv.org/abs/2404.06395
- FineWeb / FineWeb-Edu — https://arxiv.org/abs/2406.17557
- FineWeb-2 + card — https://arxiv.org/abs/2506.20920 ; https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- DCLM — https://arxiv.org/abs/2406.11794
- Cosmopedia card — https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- TinyStories — https://arxiv.org/abs/2305.07759
- phi-1 — https://arxiv.org/abs/2306.11644
- MobileLLM — https://arxiv.org/abs/2402.14905
- MobileLLM-R1 — https://arxiv.org/abs/2509.24945
- Gemma 2 (KD) — https://arxiv.org/abs/2408.00118
- Gemma 3 (QAT) — https://arxiv.org/abs/2503.19786
- Distillation scaling laws (Busbridge) — https://arxiv.org/abs/2502.08606
- MiniPLM — https://arxiv.org/abs/2410.17215
- Qwen2.5 — https://arxiv.org/abs/2412.15115
- Qwen3 — https://arxiv.org/abs/2505.09388
- μP — https://arxiv.org/abs/2203.03466
- Small-scale instabilities (Wortsman) — https://arxiv.org/abs/2309.14322
- QK-norm origin (ViT-22B) — https://arxiv.org/abs/2302.05442
- PaLM (z-loss, clip, batch ramp) — https://arxiv.org/abs/2204.02311
