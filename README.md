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

Pipeline: `gatra-prepare` streams each source to `data/*.jsonl` → `gatra-build` dedups, splits, weights, and packs them into token bins → `gatra-train` memory-maps the bins.

### Sources

All streamed; never download a full split.

| `--source` | Dataset | Slice | Notes |
|---|---|---|---|
| `fineweb2-hq` | `epfml/FineWeb2-HQ` | `ind_Latn` | top-quality slice of FineWeb-2; closest to FineWeb-Edu for Indonesian |
| `fineweb2` | `HuggingFaceFW/fineweb-2` | `ind_Latn` | larger, cleaner than CulturaX |
| `wiki` | `wikimedia/wikipedia` | `20231101.id` | small, formal, fact-dense; streamed shuffled |
| `culturax` | `uonlp/CulturaX` | `id` | gated, noisier (mC4 + OSCAR) |
| `kopi-news` | `munggok/KoPI-CC_News` | 2022 → 2016 | SEACrowd `kopi_cc_news`; CC-News, CC0 |
| `newspapers-2018` | `indonesian-nlp/id_newspapers_2018` | title + content | SEACrowd `id_newspaper_2018`; ~500k articles, CC-BY-4.0 |
| `cendol-wikihow` | `indonlp/cendol_collection_v2` | `subset_name == wikihow` | how-to prose, prompt + answer |
| `cendol-puisi` | `indonlp/cendol_collection_v2` | `subset_name == indo_puisi` | poetry; keep the budget small |
| `aya` | `CohereLabs/aya_dataset` | `language_code == ind` | ~800 human-written instructions |

```bash
uv sync --extra data
huggingface-cli login
uv run gatra-prepare --source wiki --max-bytes 40M
uv run gatra-prepare --source fineweb2-hq --max-bytes 160M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source cendol-wikihow --max-bytes 30M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source cendol-puisi --max-bytes 10M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source aya --dedup-against "data/*.jsonl"
uv run gatra-prepare --source newspapers-2018 --sample-rate 0.25 --max-bytes 50M --dedup-against "data/*.jsonl"
uv run gatra-prepare --source kopi-news --sample-rate 0.1 --max-bytes 50M --dedup-against "data/*.jsonl"
```

- Byte tokenizer: 1 byte = 1 token, so `--max-bytes` is a token budget. Chinchilla wants ~20 tokens per parameter: Gatra-10M ≈ 200M, Gatra-100M ≈ 2B. `--limit N` caps documents instead (default 5000 when `--max-bytes` is unset).
- `--sample-rate` keeps a hash-chosen fraction of docs. Dumps sorted by site or date (`newspapers-2018`, `kopi-news`) otherwise spend the whole budget on the first site or year.
- Instruction sets are flattened to `prompt\n\nanswer`. Cendol's wikipedia subsets duplicate `wiki`, and its identity/safety prompts teach "I am Cendol", so only wikihow and poetry are pulled.
- SEACrowd loaders are dataset scripts, which `datasets` 4+ no longer runs. Presets with raw `files` read the upstream `.jsonl.zst`, `.jsonl.gz`, or `.tgz` directly.
- Not usable as-is: OpenAssistant (12 Indonesian messages), OLMo/Dolma (English), Indo4B (link returns 403; release is lowercased), Liputan6 (pre-tokenized summarization data).

### Identity

`data/gatra-identity.jsonl` is checked in: hard-coded chats in the style of `allenai/olmo-2-hard-coded` (Gatra, built by Braincore RnD, Jakarta). Each row has `messages` for a later SFT stage and `text` for the pretraining mix. Edit `src/gatra/identity.py`, then:

```bash
uv run gatra-identity
```

At ~10KB it is noise next to 200MB of web text even at weight 20; identity really sticks during SFT.

### Mix and build

```toml
[data]
bin_dir = "data/bin/gatra-10m"
val_ratio = 0.1
dedup = true

[data.mix]
"data/gatra-identity.jsonl" = 20.0
"data/wiki-id.jsonl" = 1.5
"data/fineweb2hq-id.jsonl" = 1.0
"data/cendol-puisi-id.jsonl" = 0.5
```

```bash
uv run gatra-build --config configs/gatra-10m.toml
```

- Weight = how many times a source's train docs are written: `1.5` writes every doc once and half of them twice, `0.5` keeps half, `0` drops the source.
- Dedup is whitespace-normalized exact match across all sources; earlier `mix` entries win. CulturaX and FineWeb-2 both come from CommonCrawl, so expect overlap.
- Train/val is split per document by hash, so a doc stays on the same side across rebuilds and every source lands in val. Val docs are never repeated.
- Output: `train.bin` / `val.bin` (uint8 bytes) and `meta.json` with per-source docs, duplicates, and bytes. Training warns when `meta.json` was built from a different mix.
- Without `bin_dir`, the trainer builds the same corpus in memory each run (fine for `gatra-smoke` / `gatra-1m`). `paths = [...]` still works and means weight 1.0 for each.

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
