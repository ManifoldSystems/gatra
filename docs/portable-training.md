# Continue on Colab or Kaggle

The current V100 mixed-data run uses 10.721M parameters, byte context 1,024,
batch 32, FP16 and 3,017,763,515 training / 259,786,725 validation byte tokens.
The GPU was measured at 99% utilization and about 6.5 GiB device memory usage.
Filling the remaining VRAM is not itself a throughput objective. Estimated one
corpus-size budget is about five hours using prior measured throughput; actual
ETA comes from this run's training log.

## Assets to retain outside the expiring server

Transfer directly from the server to a user-chosen durable destination; do not
copy datasets/checkpoints to the Mac. No destination/upload is configured yet.

- `checkpoints/gatra-10m-all-data-v1/latest.pt`: model plus optimizer, scaler,
  RNG state, configuration and counters; primary resume artifact.
- `best.pt`, `launch-manifest.json`, metrics and training log for comparison.
- `data/bin/gatra-10m-all-data-v1/{train.bin,val.bin,meta.json}`: roughly 3.28 GB
  uncompressed combined. Byte bins are portable across OS/GPU architectures.
- Original pretraining control `gatra-10m-expanded-ctx512-fp16/latest.pt`.
- Git revision and environment package versions. Hash each completed artifact
  at a stable checkpoint boundary, then verify hashes at the destination.

Use durable storage (for example a private HF repository or Drive), not the
temporary Colab/Kaggle runtime disk as the only copy. A full training checkpoint
is needed to preserve optimizer state; model-only weights are insufficient.
Source raw exports are useful for future remixes and should be transferred too
if storage allows, but the byte bins suffice to resume the exact data mixture.

## Runtime recipe

Clone the repository and install it into the notebook runtime's environment.
Use a CUDA-enabled PyTorch build supported by the assigned GPU; do not copy the
V100 virtual environment. FP16 works on T4/V100; actual batch throughput and
memory need a short check on the assigned GPU. The checkpoint's batch/context
are retained by default. This is a single-GPU recipe even if two GPUs are assigned.

After transferring and hash-checking the artifacts:

```bash
python scripts/resume_portable.py \
  --checkpoint /path/to/latest.pt \
  --bin-dir /path/to/gatra-10m-all-data-v1 \
  --out-dir /path/to/persistent/checkpoints \
  --device cuda --max-time 3h
```

The script reads config from the checkpoint, validates bin sizes, preserves the
context/batch/precision and resumes the original remaining step budget. It stops
and saves when its runtime budget is reached. If the previous run already
finished, explicitly supply `--extra-iters N` to authorize further steps; this
changes the schedule horizon and may raise LR again, so it is a new continuation
experiment rather than a bit-identical continuation of the completed run.

Cross-GPU execution is not guaranteed bit-identical. This recipe is for full-text
pretraining; do not use it to resume assistant-only SFT checkpoints. Do not run
multiple writers against the same output directory. Neither this script nor the
documentation uploads artifacts or provisions cloud runtimes automatically.
