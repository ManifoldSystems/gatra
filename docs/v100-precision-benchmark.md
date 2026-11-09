# V100 batch and precision probe

2026-11-09, Tesla V100-SXM2-32GB, PyTorch 2.7.1+cu126. Gatra 10.721M,
context 512, expanded-mixture checkpoint step 49,500. Each configuration reloads
the same checkpoint and optimizer. Ten warmup steps followed by twenty measured
steps; one trial per configuration, sequentially FP32 then FP16. Production
training was paused and resumed automatically after the probe.

| Precision | Batch | Tokens/s | Seconds/step | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 16 | 93,246 | 0.0879 | 2,054 |
| FP32 | 32 | 100,686 | 0.1627 | 3,963 |
| FP32 | 64 | 107,260 | 0.3055 | 7,770 |
| FP16 AMP | 16 | 160,559 | 0.0510 | 1,656 |
| FP16 AMP | 32 | 186,581 | 0.0878 | 3,136 |
| FP16 AMP | 64 | 201,773 | 0.1624 | 6,094 |

All measured losses and gradient norms were finite; zero skipped AMP updates.
FP16 uses autocast with FP32 master parameters and AdamW state plus GradScaler;
it does not cast the entire model to half precision. Losses use different sampled
windows across batch sizes, so they are not a quality ranking. A short stable
probe does not establish long-run convergence or accuracy equivalence. Timing
does not include checkpoint writes, generation, or periodic validation.

Relative to FP32 batch 16, FP16 batch 16 improves throughput 1.72x with the same
token batch; FP16 batch 64 improves it 2.16x but quadruples tokens per update.
The low-disruption next experiment is FP16 batch 16. Production training remains
FP32 batch 16 until AMP is integrated into train/evaluation and scaler state is
saved/restored in checkpoints. A larger batch also needs an explicit token-budget
and learning-rate schedule decision.

Reproduce with `scripts/benchmark_precision.py`; local raw outputs are in
`checkpoints/precision-benchmark/results.json` and the server has one JSON file
per configuration under the corresponding checkpoint directory.
