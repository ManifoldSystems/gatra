---
title: "Edge Deployment Stack for Sub-500M LLMs: Runtimes, Quantization, Measured Numbers (2026-10-22)"
tags: [edge-deployment, quantization, runtime, llama-cpp, executorch, qnn, kleidiai, sub-500m]
---

# Edge Deployment Stack for Sub-500M LLMs: Runtimes, Quantization, Measured Numbers

> [!tip] Terkait (Graph)
> - [[efficient-llm-architecture-edge-under-500m-2026-10-20|Efficient LLM Architecture Edge Under 500M (Baseline 2026-10-20)]]
> - [[quantization-precision/README|Quantization & Precision]]

Baseline validated: `pretrain_llms/gatra/docs/efficient-llm-architecture-edge-under-500m-2026-10-20.md` §6.
This note validates §6.1–§6.4 against primary sources (access 2026-10-22), corrects stale entries, and adds a runtime recommendation matrix.

---

## 1) Validated claims (baseline §6 holds)

### 1.1 Runtimes — license, format, backend

**llama.cpp — MIT, GGUF, CPU-first, K-quants with explicit bpw.**

> [!quote] llama.cpp PR #1684 (ikawrakow, merged 2023-06-05)
> "`GGML_TYPE_Q4_K` - \"type-1\" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using `4.5` bpw."
> — https://github.com/ggml-org/llama.cpp/pull/1684

Full K-quant costs from the same PR: Q2_K `2.5625`, Q3_K `3.4375`, Q4_K `4.5`, Q5_K `5.5`, Q6_K `6.5625` bpw; mixes (e.g. Q4_K_M uses Q6_K for half of `attention.wv` and `feed_forward.w2`, else Q4_K). Baseline §6.2 numbers (Q2_K 2.5625 … Q6_K 6.5625) **validated**. License MIT: https://github.com/ggml-org/llama.cpp

**ExecuTorch — BSD, `.pte`, export-per-backend, quantization backend-specific.**

> [!quote] ExecuTorch LICENSE (pytorch/executorch, main)
> "BSD License / For \"ExecuTorch\" software / Copyright (c) Meta Platforms, Inc. and affiliates."
> — https://github.com/pytorch/executorch/blob/main/LICENSE

> [!quote] ExecuTorch docs, Quantization Overview
> "Quantization in ExecuTorch is backend-specific. Each backend defines how models should be quantized based on its hardware capabilities."
> — https://docs.pytorch.org/executorch/stable/quantization-overview.html

> [!quote] ExecuTorch docs, Export and Lowering
> "This process yields a .pte file which is optimized for on-device execution using a particular backend… In order to deploy to multiple backends, such as Core ML on iOS and Arm CPU on Android, it is common to generate a dedicated .pte file for each."
> — https://docs.pytorch.org/executorch/main/export-overview.html (via web_search)

Baseline "butuh export per backend" **validated**. Backends covered: XNNPACK, Vulkan, CoreML, QNN (each with own `*Quantizer`, e.g. XNNPACKQuantizer, CoreMLQuantizer).

**MLC-LLM — Apache-2.0, TVM compiler, q4f16_1, iOS/Android/Web/WebGPU.**

> [!quote] MLC-LLM docs, Compile Model Libraries
> "`mlc_llm gen_config … --quantization q4f16_1` … `mlc_llm compile … --device android … --device webgpu … --device iphone"
> — https://llm.mlc.ai/docs/compilation/compile_models.html

License Apache-2.0: https://github.com/mlc-ai/mlc-llm/blob/main/LICENSE. Baseline row **validated** (compiler-driven; one compiled lib per `--device` target).

**ONNX Runtime GenAI — MIT, ONNX, EP-dependent (XNNPACK/QNN/CoreML).**

> [!quote] ONNX Runtime docs
> "The QNN providers are built as shared libraries… `--use_qnn [QNN_LIBRARY_KIND]`… Usage of NNAPI on Android is via the NNAPI Execution Provider… Usage of CoreML on iOS and macOS is via the CoreML EP."
> — https://onnxruntime.ai/docs/execution-providers/ (via web_search; GenAI entry https://onnxruntime.ai/docs/genai/)

License MIT: https://github.com/microsoft/onnxruntime-genai/blob/main/LICENSE. Baseline "EP-dependent, Windows/Android/iOS, API preview" **validated**.

**LiteRT-LM — Apache-2.0, `.litertlm`, INT4, cross-platform (not Gemma-only).**

> [!quote] Google AI Edge, LiteRT-LM Overview
> "Cross-Platform Support: Run on Android, iOS, Web, Desktop, and IoT (e.g. Raspberry Pi). Hardware Acceleration: Get peak performance … by leveraging GPU and NPU accelerators."
> — https://developers.google.com/edge/litert-lm

> [!quote] LiteRT Community, Gemma-3n-E2B card
> "Quantization: quantized model with int4 weights and float activations. The inference on CPU is accelerated via the LiteRT XNNPACK delegate with 4 threads."
> — https://huggingface.co/google/gemma-3n-E2B-it-litert-lm

License Apache-2.0 (LiteRT repo): https://github.com/google-ai-edge/LiteRT. Baseline format `.litertlm` + INT4 **validated**.

**Core ML — proprietary, `.mlpackage`, CPU/GPU/ANE, PTQ/GPTQ/QAT.**

> [!quote] Apple coremltools docs, Quantization Algorithms
> "Post-training (data-free) weight quantization … Post-training (data calibration) activation quantization … GPTQ algorithm for weight quantization … Fine-tuning based algorithm for quantizing weight and/or activations."
> — https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html

> [!quote] Apple coremltools docs
> "Core ML models now also use a directory format, called `.mlpackage`, rather than just a protobuf file."
> — https://apple.github.io/coremltools/docs-guides/source/ (release notes)

Baseline row **validated**; iOS-only, proprietary toolchain.

**MLX — MIT, Apple silicon only, not a phone runtime.**

> [!quote] ml-explore/mlx
> "MLX: An array framework for machine learning on Apple silicon … License: MIT License … Operations can run on any of the supported devices (currently the CPU and the GPU)."
> — https://github.com/ml-explore/mlx

> [!quote] ml-explore/mlx LICENSE
> "MIT License / Copyright © 2023 Apple Inc."
> — https://github.com/ml-explore/mlx/blob/main/LICENSE

Baseline "Mac/laptop, bukan ponsel" **validated**.

**Qualcomm QNN / AI Hub — QNN SDK proprietary, AI Hub Models BSD-3.**

> [!quote] qualcomm/ai-hub-models
> "License: BSD 3-Clause … On-Device Runtimes: Qualcomm AI Engine Direct / LiteRT / ONNX … NPU (includes Hexagon DSP, HTP): FP16*, INT16, INT8"
> — https://github.com/qualcomm/ai-hub-models

Baseline "vendor graph, lock-in" **validated**; ceiling path for Snapdragon NPU is QNN/HTP, not mainline llama.cpp.

**KleidiAI — Apache-2.0 micro-kernels, not a runtime.**

> [!quote] ARM-software/kleidiai, docs/matmul_qsi4cx/README.md
> "SPDX-License-Identifier: Apache-2.0 … we will explore the use of the integer 4-bit (int4) matrix multiplication matmul micro-kernel with the per-channel quantization (cx) … Arm CPUs with FEAT_I8MM extension."
> — https://github.com/ARM-software/kleidiai/blob/main/docs/matmul_qsi4cx/README.md

Baseline "bukan runtime; lapisan kernel" **validated**. Consumed via llama.cpp/XNNPACK/ExecuTorch-XNNPACK, not directly.

### 1.2 Quantization methods

**GPTQ — 3–4 bit, negligible degradation (large-model evidence), 3.25×/4.5× speedup.**

> [!quote] GPTQ, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), Abstract
> "GPTQ can quantize GPT models with 175 billion parameters in approximately four GPU hours, reducing the bitwidth down to 3 or 4 bits per weight, with negligible accuracy degradation relative to the uncompressed baseline… speedups … of around 3.25x when using high-end GPUs (NVIDIA A100) and 4.5x when using more cost-effective ones (NVIDIA A6000)."

**AWQ — protect 1% salient weights, >3× speedup desktop + mobile GPU.**

> [!quote] AWQ, MIT HAN Lab ([paper](https://arxiv.org/abs/2306.00978), [project](https://hanlab.mit.edu/projects/awq))
> "Our method is based on the observation that weights are not equally important: protecting only 1% of salient weights can greatly reduce quantization error… offering more than 3x speedup over the Huggingface FP16 implementation on both desktop and mobile GPUs."

**SmoothQuant — W8A8, up to 1.56× speedup, 2× memory.**

> [!quote] SmoothQuant, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438), Abstract
> "We demonstrate up to 1.56× speedup and 2× memory reduction for LLMs with negligible loss in accuracy."

Note scope: OPT/BLOOM/Llama-1/2/Falcon/Mistral/Mixtral; server-GPU numbers (FasterTransformer/A100), not phone SoCs.

**LLM.int8() — half memory via vector-wise + mixed-precision outlier path.**

> [!quote] LLM.int8(), [arXiv:2208.07339](https://arxiv.org/abs/2208.07339), Abstract
> "We develop a procedure for Int8 matrix multiplication … which cut the memory needed for inference by half while retaining full precision performance… for the emergent outliers, we also include a new mixed-precision decomposition scheme, which isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit."

Caveat (same paper's follow-up §): SmoothQuant authors note "LLM.int8() is almost always slower than the FP16 baseline, which is due to the large overhead of the mixed-precision activation representation" — https://arxiv.org/abs/2211.10438. So memory win ≠ speed win on edge.

**QAT INT4 (Gemma 3) — ~5k steps, −54% perplexity drop, minimal degradation.**

> [!quote] Google Developers Blog, Introducing Gemma 3 270M
> "We applied QAT on ~5,000 steps using probabilities from the non-quantized checkpoint as targets. We reduce the perplexity drop by 54% (using llama.cpp perplexity evaluation) when quantizing down to Q4_0."
> — https://developers.googleblog.com/en/introducing-gemma-3-270m/

> [!quote] google/gemma-3-270m-it-qat-q4_0-unquantized model card
> "Thanks to QAT, the model is able to preserve similar quality as `bfloat16` while significantly reducing the memory requirements to load the model."
> — https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized

Baseline "QAT int4 pasca-pretraining (±5k step)" **validated** as cheapest reliable path for <500M.

**FP8 — training/inference evidence exists but floor is ~111M, all server-side.**

> [!quote] FP8 training/inference, [arXiv:2309.17224](https://arxiv.org/html/2309.17224), Abstract
> "We apply this methodology to train and validate large language models of the type of GPT and Llama 2 using FP8, for model sizes ranging from 111M to 70B."

No phone-NPU FP8 inference numbers found for sub-500M; treat FP8 as training-efficiency knob, not an edge deployment format in 2026.

### 1.3 Measured on-device numbers (first-party, device + runtime + quant specified)

**MobileLLM-125M — iPhone 13, ExecuTorch, MPS, FP16: execute 15.6 ms; load 39.2 ms; init 1361.7 ms.**

> [!quote] MobileLLM, [arXiv:2402.14905](https://arxiv.org/abs/2402.14905), §3.6 Table 7
> "We measure the latency for MobileLLM-125M and MobileLLM-LS-125M FP16 models via ExecuTorch on iPhone 13 (iOS 17.2.1), with Metal Performance Shaders (MPS) backend… | MobileLLM | 39.2 ms | 1361.7 ms | 15.6 ms |"

Absolute measurement, first-party. Note: per-forward execute latency, not end-to-end tok/s.

**MobileLLM-R1 — Galaxy S22 (8 GB), ExecuTorch, W4A8 (linear 4-bit weights + 8-bit dynamic activations, group 128; embeddings 4-bit group 32): 140M >100 tok/s to 8K; 360M ~61–77 tok/s.**

> [!quote] MobileLLM-R1, [arXiv:2509.24945](https://arxiv.org/abs/2509.24945), §8 Table 10
> "| MobileLLM-R1-140M | 140M | 129.67 | 116.16 | 111.42 | 110.47 | 96.38 | 79.71 | / | MobileLLM-R1-350M | 360M | 77.23 | 61.54 | 61.24 | 61.27 | 61.13 | OOM |"

> [!quote] Same paper, §8 methodology
> "We benchmark … using ExecuTorch on a Samsung Galaxy S22 (8 GB RAM)… models are benchmarked under quantized settings: linear layers are quantized using 8-bit dynamic activations and 4-bit weights… group size of 128. Embedding layers are quantized to 4-bit with a group size of 32."

Absolute, first-party, warm-up + mean of 3 runs. Strongest sub-500M phone evidence in the set.

**Gemma 3 270M — Pixel 9 Pro (Tensor G4), INT4 QAT: 25 conversations on 0.75% battery.**

> [!quote] Google Developers Blog
> "Internal tests on a Pixel 9 Pro SoC show the INT4-quantized model used just 0.75% of the battery for 25 conversations, making it our most power-efficient Gemma model."
> — https://developers.googleblog.com/en/introducing-gemma-3-270m/

Absolute battery measurement, first-party; **not** tok/s. Conversation length undefined → cannot convert to tok/s or mJ/token.

**LFM2-350M — vendor-relative "2×", ExecuTorch 8da4w / llama.cpp Q4_0, Galaxy S24 Ultra + AMD Ryzen HX370.**

> [!quote] Liquid AI blog, Introducing LFM2 (2025-07-10)
> "Built on a new hybrid architecture, LFM2 delivers 2x faster decode and prefill performance than Qwen3 on CPU."
> — https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models

> [!quote] Same blog, methodology note
> "The LFM2 models were evaluated using each platform's recommended quantization schemes — 8da4w for ExecuTorch and Q4_0 for llama.cpp … Target hardware included the Samsung Galaxy S24 Ultra (Qualcomm Snapdragon SoC) and AMD Ryzen (HX370) platforms."

**Vendor-relative claim** (denominator = Qwen3, unspecified size/config in headline). Do not quote "2×" without "vs Qwen3 on CPU, Liquid's own harness".

**LiteRT-LM first-party phone numbers (reference tiers, mostly >500M).**

> [!quote] google/gemma-3n-E2B-it-litert-lm card
> "| Gemma3n-E2B | Samsung S24 Ultra | CPU | 110.5 | 16.1 | / | Gemma3n-E2B | Samsung S24 Ultra | GPU | 816.4 | 15.6 | / | Gemma3n-E2B | Vivo X300 Pro | NPU | 1671.3 | 28.4 |"
> — https://huggingface.co/google/gemma-3n-E2B-it-litert-lm

> [!quote] Google AI Edge, LiteRT-LM models table
> "| Gemma3-1B | Chat | 1005 | … | Samsung S24 Ultra | 177 | 33 | 1191 | 24 | / | FunctionGemma | Base | 289 | … | Samsung S25 Ultra | 2238 | 154 | - | - |"
> — https://developers.google.com/edge/litert-lm/models/gemma-4

Absolute, first-party, but E2B/E4B/1B tiers are above 500M; only FunctionGemma-289M sits in scope (154 tok/s decode CPU). Useful as LiteRT ceiling reference, not a <500M chat-LM data point.

**Qualcomm geniex-bench — Qwen3-0.6B Q4_0, Snapdragon X Elite (laptop-class): CPU 18.5 tok/s decode → NPU 23.1 → hybrid 27.4.**

> [!quote] Qualcomm geniex-bench docs
> "[ok] cell plugin=llama_cpp device=cpu ngl=0 ttft=476.2ms prefill=25.3tps decode=18.5tps … [ok] device=npu … ttft=340.1ms prefill=62.8tps decode=23.1tps … [ok] device=hybrid … ttft=198.4ms prefill=91.5tps decode=27.4tps (Qwen_Qwen3-1.7B-GGUF:Q4_0, Snapdragon X Elite)"
> — https://github.com/qualcomm/ai-hub-models (geniex-bench; same harness shows Qwen3-0.6B Q4_0 NPU ttft≈349–358 ms, prefill≈59–60 tps, decode≈20–22 tps)

Absolute, first-party harness, but Snapdragon X Elite (not phone SoC) and 0.6B slightly above 500M. Best available evidence that NPU/hybrid helps prefill far more than decode.

**Third-party neutral (dispatchAI, Snapdragon 865, llama-cli, 4 threads, ctx 512, 20 tokens): SmolLM2-135M fp16 18.3 t/s; SmolLM2-360M fp16 12.1 t/s; Qwen2.5-0.5B int4 9.8 t/s.**

> [!quote] dispatchAI/per-chip-benchmark-matrix
> "Models run via `llama-cli` (llama.cpp) on Samsung S20 FE 5G devices: Chipset: Qualcomm Snapdragon 865 … Threads: 4, Context: 512 … | SmolLM2-135M | 135M | fp16 | 18.3 | 270MB | / | SmolLM2-360M | 360M | fp16 | 12.1 | 720MB | / | Qwen2.5-0.5B | 500M | int4 | 9.8 | 350MB |"
> — https://huggingface.co/datasets/dispatchAI/per-chip-benchmark-matrix

Absolute, third-party, old SoC (865, 7 nm) + fp16 for SmolLM2 (not quant-comparable to int4 rows). Useful neutral floor, not SOTA.

### 1.4 Edge constraints that change design

**Cold start dominates short sessions (1.36 s init vs 39 ms load).**

> [!quote] MobileLLM §3.6 (same Table 7)
> "Model loading, initialization, and execution time are reported in Table 7… execution time is averaged over 50 iterations."

Design consequence (baseline §6.4): warm-up at app start matters more than a 10% decode gain.

**Background execution is CPU-restricted on iOS; Android throttles background work.**

> [!quote] Apple staff (anilkatti) on Core ML background issue
> "One thing to keep in mind is background mode is enabled only on CPU. You can do this by setting MLModelConfiguration.computeUnits."
> — https://github.com/apple/coremltools (issue thread; API: https://developer.apple.com/documentation/coreml/mlcomputeunits)

> [!quote] Android Developers, Background tasks overview
> "If the task will take longer than a few seconds, it may be appropriate to start a foreground service… foreground services that use the short service type must complete quickly, within about three minutes."
> — https://developer.android.com/develop/background-work/background-tasks

Design consequence: background inference must assume CPU-only + strict time budgets; NPU/GPU path is foreground-only.

**Thermal throttling is a four-state OS contract; burst benchmarks mislead.**

> [!quote] Apple Developer Documentation, Thermal States (Table 18-1)
> "Serious … System performance may also be impacted as the system begins enacting countermeasures… Critical … The device needs to cool down."
> — https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum (also https://developer.apple.com/documentation/foundation/processinfo/thermalstate)

Design consequence: report sustained 5-min numbers, not 10-s bursts (baseline §6.4 holds).

**NPU is op-specific; one unsupported op forces CPU fallback.**

> [!quote] ExecuTorch Quantization Overview (cited above)
> "Not all quantization options are supported by all backends. Consult backend-specific guides for supported quantization modes."

Design consequence for conv/SSM hybrids (§2 below): dense transformer is the portable default.

**Vocab/embedding share of file size.**

> [!quote] Google Developers Blog, Gemma 3 270M
> "Our new model has a total of 270 million parameters: 170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks. Thanks to the large vocabulary of 256k tokens…"
> — https://developers.googleblog.com/en/introducing-gemma-3-270m/

63% of parameters in embeddings. At INT4 that is still ~95 MB of the ~152 MB file before any transformer weight. Baseline §6.4 "vocab mendominasi ukuran file" **validated**.

**KV cache vs weights at long context.**

> [!quote] MobileLLM-R1 §8
> "When context length becomes large, the KV cache size approaches the size of the weights. Since we are already memory-bound when loading weights, the additional cost of loading the KV cache is no longer negligible."

OOM row in Table 10 (360M OOM at 32K; 950M OOM at 16K on 8 GB phone) is the empirical counterpart. Baseline cap-4K guidance **validated**.

---

## 2) Corrected / updated claims (what changed vs 2026-10-20)

1. **"SmolLM2/Qwen2.5 have no phone latency numbers" — now partially stale.** Neutral third-party numbers exist (dispatchAI Snapdragon 865: 18.3 / 12.1 / 9.8 tok/s) and `executorch-community/SmolLM2-135M` `.pte` recipes exist — https://huggingface.co/executorch-community/SmolLM2-135M. Still true that **first-party** SmolLM2/Qwen2.5 phone tok/s are missing, but "no numbers at all" is no longer accurate.
2. **Gemma 3 vocab 262k vs 256k — rounding inconsistency flagged.** Gemma 3 family tokenizer is 262,208 ("vocabulary size has been changed to 262k" — https://developers.google.com/ai/gemma); the 270M blog rounds to "256k tokens" (quote §1.4). Use 262,208 for size math; the 170M-embedding figure already reflects it (262,208 × 640 ≈ 168M).
3. **LiteRT-LM is no longer "Gemma-first".** Supported list now includes Llama, Phi-4, Qwen, FunctionGemma with published CPU/GPU/NPU tables (quote §1.3). Keep as a viable NPU-upgrade path, not a Google-only silo.
4. **llama.cpp "Hexagon" needs a qualifier.** Mainline llama.cpp backend coverage is CPU/Metal/Vulkan/CUDA; Hexagon-NPU acceleration for GGUF on Snapdragon goes through Qualcomm's QNN EP / geniex fork (`--device npu/hybrid`, quote §1.3), not stock `llama-cli`. Baseline table cell "Hexagon" overstates mainline.
5. **Granite-4.0-H-350M on llama.cpp/MLX — "ongoing work" confirmed still incomplete.** IBM tracker lists Mamba2 layers done, hybrid attention/recurrent cache partially done, but "Support for `mamba2` in non-CPU backends" open and "Metal backend … still doesn't work on my M3 (assertion error about non-contiguous data)", with no end-to-end `GraniteMoEHybrid` support yet — https://github.com/ggml-org/llama.cpp/issues (IBM Granite 4 tracking issue, gabe-l-hart). Do not plan a Granite-class hybrid unless shipping CPU-only.
6. **Mamba in llama.cpp — supported but second-class (strengthens baseline fallback warning).** Mamba support merged in PR #5328, but author notes: "I did not implement GPU kernels for the new operators" and "the more a Mamba model is quantized, the more likely the model will output gibberish (especially with smaller models like `mamba-130m` at `Q4_K`)" — https://github.com/ggml-org/llama.cpp/pull/5328. Conv1d/selective-scan kernels are the exact fallback risk baseline §3.7/§6.4 warns about.
7. **LFM2.5-350M 28T-token row — carry-over without primary paper.** The LFM2 blog (quote §1.3) covers LFM2; the "LFM2.5 28T" figure in baseline Table §4 has no linked report as of 2026-10-22. Keep flagged as vendor claim until a report/model card lands.
8. **MobileLLM license "FAIR Noncommercial" and LFM2 `lfm1.0` — carried over, not re-validated here.** No primary license file was re-fetched for these two in this pass; do not treat as validated. (Qwen2.5/SmolLM2 Apache-2.0 and Gemma Terms are per their HF cards and unchanged.)
9. **Qwen3-0.6B is 752M (over-tier) — baseline correctly marks § as reference.** Keep out of the ≤500M size table (§3 below); use only for NPU-harness evidence.

---

## 3) Runtime recommendation matrix (CPU-first baseline + NPU upgrade path)

Default: **ship llama.cpp GGUF Q4_K_M CPU-first; add exactly one NPU path only after sustained-benchmark gates pass.**

| Stage | Runtime / format | When | Quant | Why (primary evidence) |
|---|---|---|---|---|
| Baseline (all devices) | **llama.cpp / GGUF `Q4_K_M`** | Day 0, all tiers | Q4_K 4.5 bpw (PR #1684 quote §1.1) | Widest device coverage; deterministic file sizes (§4); Mamba/SSM caveat documented (PR #5328 quote §2.6) |
| iOS upgrade | **ExecuTorch `.pte` + CoreML EP** (separate artifact from CPU `.pte`) | Need ANE/GPU or background-CPU pinning | Backend-specific PTQ/QAT via torchao PT2E (quote §1.1) | Per-backend `.pte` required; background = CPU-only (Apple quote §1.4) |
| Android upgrade | **ExecuTorch `.pte` + QNN EP**, else **LiteRT-LM `.litertlm`** | Snapdragon target with measured prefill win | 8da4w (ExecuTorch) / INT4 (LiteRT) | QNN HTP precision FP16/INT16/INT8 (AI Hub quote §1.1); geniex hybrid shows prefill ≫ decode gains (quote §1.3) |
| Web / demo | **MLC-LLM (TVM)** | Browser or cross-OS demo without native app | q4f16_1 | One compile per `--device` (quote §1.1) |
| Apple laptop dev | **MLX** | Profiling/ablation on Mac only | FP16/quantized | MIT, Apple-silicon-only (quote §1.1); not a phone path |
| Kernels (invisible) | **KleidiAI via XNNPACK/llama.cpp** | Never a direct dependency | INT4/INT8 micro-kernels, FEAT_I8MM | Apache-2.0, CPU micro-kernels (quote §1.1) |

NPU-gate rule (from §1.3–§1.4 evidence): promote an NPU path only if it beats the llama.cpp CPU baseline on **sustained 5-min** runs for TTFT (prompt 512–1K), decode tok/s at 4K context, peak RSS, and battery per 1K tokens — with unsupported-op fallback rate = 0 (ExecuTorch partition report). A prefill-only win (typical: geniex NPU prefill 2–3×, decode ~1.2×) justifies NPU only for long-prompt products.

Operator-fallback risks for non-attention ops:

- **Conv1d / depthwise / gated-conv (LFM2-style):** no dedicated NPU kernel in CoreML/QNN partitions by default; falls back to CPU inside an otherwise-NPU graph (ExecuTorch backend-specific quote §1.1). Keep hybrid CPU-only unless the target NPU partition log shows 100% delegation.
- **SSM selective-scan / Mamba-2 (Granite/Falcon-H1-style):** llama.cpp CPU support exists but GPU kernels were never implemented by the contributor (PR #5328 quote); Granite hybrid end-to-end + Metal still open (IBM tracker quote). Treat as CPU-only research path.
- **Sliding-window + sink cache:** safe on all paths (pure attention masking); preferred long-context lever over hybrid ops.

### Model size table — Q4_K (4.5 bpw) and Q8 (8.5 bpw) file weights-only

Formula: `bytes = params × bpw / 8`. Excludes tokenizer, runtime, KV cache.

| Tier | Params | Q4_K (4.5 bpw) | Q8_0 (8.5 bpw) |
|---|---:|---:|---:|
| 135M | 135M | **76 MB** | **143 MB** |
| 270M | 270M | **152 MB** | **287 MB** |
| 350M | 350M | **197 MB** | **372 MB** |
| 500M | 500M | **281 MB** | **531 MB** |

bpw source: PR #1684 quote §1.1 (Q4_K 4.5; Q8_0 block-32 + fp16 scale = 8.5 bpw convention used in baseline §6.2). Add ~5–15 MB for tokenizer + runtime fixed cost; add KV from baseline §2.2 (e.g. 350M-tier bf16 ≈ 39 KiB/token → 155 MiB @4K).

---

## 4) Open questions

1. **QAT-vs-PTQ gap under 500M is still single-vendor.** Only Gemma 3 (−54% PPL drop via 5k-step QAT, quote §1.2) quantifies it; no controlled QAT ablation for SmolLM2/MobileLLM/Qwen2.5 tiers. Needs a 135M/350M A/B (PTQ Q4_K_M vs 5k-step QAT → Q4_K_M) on identical evals.
2. **No first-party tok/s for SmolLM2-135M/360M or Qwen2.5-0.5B.** DispatchAI fills the gap with third-party Snapdragon 865 numbers (quote §1.3) but on an old SoC with fp16-vs-int4 mixing. A 2026-phone (S24/S25-class + iPhone 15/16-class) llama.cpp Q4_K_M sweep is still missing.
3. **Sustained-vs-burst delta is unquantified for every sub-500M model.** All tok/s rows above are short runs (20–128 tokens or 3-iteration means); none reports a 5-min thermal-soaked curve against `thermalState` (Apple quote §1.4).
4. **NPU decode economics unresolved.** geniex shows hybrid/NPU prefill 2–3× but decode only ~1.2× (quote §1.3); whether any NPU path beats a KleidiAI-accelerated CPU at batch-1 decode for ≤500M is still open.
5. **Hybrid-vs-dense on real NPU partitions.** LFM2's "2× vs Qwen3 on CPU" (vendor-relative, quote §1.3) does not answer NPU-delegated hybrid vs dense; Granite hybrid NPU support is still open (quote §2.5). Needs ExecuTorch partition-rate + tok/s head-to-head on the same SoC.
6. **License re-validation outstanding.** MobileLLM FAIR-noncommercial and LFM2 `lfm1.0` rows were carried from baseline without re-fetching primary license files in this pass.
7. **FP8 / int4-mixed training at <500M.** FP8 evidence floor is 111M but server-only (quote §1.2); no edge-relevant FP8→INT4-QAT recipe with quality deltas at 135–500M.

---

## Sources (primary only; access 2026-10-22)

- llama.cpp K-quants PR #1684 — https://github.com/ggml-org/llama.cpp/pull/1684
- llama.cpp Mamba PR #5328 — https://github.com/ggml-org/llama.cpp/pull/5328
- ExecuTorch LICENSE — https://github.com/pytorch/executorch/blob/main/LICENSE
- ExecuTorch Quantization Overview — https://docs.pytorch.org/executorch/stable/quantization-overview.html
- MLC-LLM compile docs — https://llm.mlc.ai/docs/compilation/compile_models.html ; repo https://github.com/mlc-ai/mlc-llm
- ONNX Runtime GenAI — https://onnxruntime.ai/docs/genai/ ; https://github.com/microsoft/onnxruntime-genai
- LiteRT-LM overview — https://developers.google.com/edge/litert-lm ; models table https://developers.google.com/edge/litert-lm/models/gemma-4
- Gemma-3n-E2B LiteRT card — https://huggingface.co/google/gemma-3n-E2B-it-litert-lm
- Core ML compute units — https://developer.apple.com/documentation/coreml/mlcomputeunits ; quantization — https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
- MLX — https://github.com/ml-explore/mlx ; LICENSE https://github.com/ml-explore/mlx/blob/main/LICENSE
- Qualcomm AI Hub Models — https://github.com/qualcomm/ai-hub-models
- KleidiAI int4 matmul — https://github.com/ARM-software/kleidiai/blob/main/docs/matmul_qsi4cx/README.md
- GPTQ — https://arxiv.org/abs/2210.17323 ; AWQ — https://arxiv.org/abs/2306.00978 ; SmoothQuant — https://arxiv.org/abs/2211.10438 ; LLM.int8() — https://arxiv.org/abs/2208.07339
- Gemma 3 270M blog — https://developers.googleblog.com/en/introducing-gemma-3-270m/ ; QAT card https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized
- MobileLLM — https://arxiv.org/abs/2402.14905 ; MobileLLM-R1 — https://arxiv.org/abs/2509.24945
- LFM2 blog — https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
- dispatchAI benchmark matrix — https://huggingface.co/datasets/dispatchAI/per-chip-benchmark-matrix
- executorch-community SmolLM2-135M — https://huggingface.co/executorch-community/SmolLM2-135M
- Apple ThermalState — https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum
- Android background tasks — https://developer.android.com/develop/background-work/background-tasks
- FP8 training/inference — https://arxiv.org/html/2309.17224
