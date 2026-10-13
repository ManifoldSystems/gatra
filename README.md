# gatra

Indonesia-first small language model, trained from first principles.

```text
Gatra-1M -> Gatra-10M -> Gatra-30M -> Gatra-100M
```

Architecture: RoPE, RMSNorm, SwiGLU, weight tying, byte tokenizer.

## Setup

```bash
uv sync --extra dev
```

Device order for `auto`: NVIDIA CUDA, then ROCm/HIP, then MPS, then CPU.

```bash
uv run gatra-train --config configs/gatra-smoke.toml --device cpu
uv run gatra-train --config configs/gatra-1m.toml --device cuda
uv run gatra-train --config configs/gatra-1m.toml --device rocm
uv run gatra-train --config configs/gatra-1m.toml --device mps
```

Install the matching PyTorch wheel first: CUDA from `pytorch-cu128`, ROCm from `pytorch-rocm`, MPS/CPU from the default macOS/CPU build.

## Colab T4

Use `notebooks/colab_train.ipynb`. Runtime → GPU (T4). Clone the repo, `uv sync --extra data`, write checkpoints to Drive:

```bash
uv run gatra-train --config configs/gatra-10m.toml --device cuda --out-dir /content/drive/MyDrive/gatra/checkpoints --max-time 50m
uv run gatra-continue --config configs/gatra-10m.toml --device cuda --out-dir /content/drive/MyDrive/gatra/checkpoints --extra-iters 2000
```

Colab sessions drop. Keep `--max-time` under the session limit and resume from Drive. T4 is the right box for Gatra-10M / 100M study runs, not for debugging the trainer.

## Data

Indonesian sources, all streamed (never download the full split):

| `--source` | Dataset | Config | Notes |
|---|---|---|---|
| `fineweb2-hq` | `epfml/FineWeb2-HQ` | `ind_Latn` | top-quality slice of FineWeb-2; closest to FineWeb-Edu for Indonesian |
| `fineweb2` | `HuggingFaceFW/fineweb-2` | `ind_Latn` | larger, cleaner than CulturaX |
| `wiki` | `wikimedia/wikipedia` | `20231101.id` | small, formal, fact-dense |
| `culturax` | `uonlp/CulturaX` | `id` | gated, noisier (mC4 + OSCAR) |
| `cendol-wikihow` | `indonlp/cendol_collection_v2` | `subset_name == wikihow` | how-to prose, prompt + answer |
| `cendol-puisi` | `indonlp/cendol_collection_v2` | `subset_name == indo_puisi` | poetry; keep the budget small |
| `aya` | `CohereLabs/aya_dataset` | `language_code == ind` | ~800 human-written instructions |

FineWeb-Edu itself is English only.

```bash
uv sync --extra data
huggingface-cli login
uv run gatra-prepare --source wiki --max-bytes 40M
uv run gatra-prepare --source fineweb2-hq --max-bytes 160M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source cendol-wikihow --max-bytes 30M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source cendol-puisi --max-bytes 10M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source aya --dedup-against "data/*.jsonl"
```

Instruction sets are flattened to `prompt\n\nanswer`. Cendol's wikipedia subsets duplicate `wiki`, and its identity/safety prompts teach "I am Cendol", so only wikihow and poetry are pulled.

Not usable as-is: OpenAssistant (12 Indonesian messages), OLMo/Dolma (English), Indo4B (download link returns 403), SEACrowd loaders (need dataset scripts, dropped in `datasets` 4+).

### Identity

`data/gatra-identity.jsonl` is checked in: hard-coded chats in the style of `allenai/olmo-2-hard-coded` (Gatra, built by Braincore RnD, Jakarta). Each row has `messages` for a later SFT stage and `text` for the pretraining mix. Edit `src/gatra/identity.py`, then:

```bash
uv run gatra-identity
```

At ~10KB it is noise next to 200MB of web text; identity really sticks during SFT.

Byte tokenizer: 1 byte = 1 token, so `--max-bytes` is a token budget. Chinchilla wants ~20 tokens per parameter: Gatra-10M ≈ 200M, Gatra-100M ≈ 2B. `--limit N` caps documents instead (default 5000 when `--max-bytes` is unset).

Dedup runs twice: `gatra-prepare` skips docs already seen in the stream (and in `--dedup-against` files), and the trainer drops cross-file duplicates when loading (`dedup = true`, whitespace-normalized exact match). CulturaX and FineWeb-2 both come from CommonCrawl, so expect overlap.

Mix jsonl files like puzzle pieces:

```toml
[data]
paths = [
    "data/fineweb2hq-id.jsonl",
    "data/wiki-id.jsonl",
]
shuffle_docs = true
dedup = true
```

Globs work too: `paths = ["data/*.jsonl"]`. Documents are deduped, shuffled, then split train/val.

## Train

```bash
uv run gatra-train --config configs/gatra-smoke.toml
uv run gatra-train --config configs/gatra-1m.toml
uv run gatra-continue --config configs/gatra-1m.toml --extra-iters 1000
uv run gatra-continue --config configs/gatra-1m.toml --which best --max-time 30m
```

`gatra-continue` loads `checkpoints/<name>/latest.pt` by default, restores model, optimizer, step, tokens seen, and RNG, then keeps training. Use `--extra-iters` to add steps after a finished run; `--max-iters` sets an absolute target. Checkpoints also write `best.pt` when val loss improves.

`gatra-smoke` is a 20-step pipeline check. Real runs use `gatra-1m` / `gatra-10m`.

Eval logs print `train` loss, `val` loss, `ppl` (e^val), and `tok/s`. Samples print at even quarters of `max_iters`, e.g. 500/1000/1500/2000.

Modes: `smoke` minutes, `study` 30-60m, `run` overnight. Time budget stops training cleanly with a checkpoint.

## Generate

```bash
uv run gatra-generate --checkpoint checkpoints/gatra-smoke/latest.pt --prompt "Pagi itu"
```

## Layout

```text
configs/     model size and train recipes
data/        jsonl corpus, one {"text": ...} per line
src/gatra/   model, data, train, generate
tests/       tokenizer, shapes, config
```
