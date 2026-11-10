# Declarative server data plan

## Storage contract

The Mac repository contains scripts, TOML configuration and documentation.
Downloads, caches, raw snapshots, JSONL exports, training bins, checkpoints and
full evaluation outputs belong on the server under `/workspace/gatra`.
Do not download new datasets to the Mac or mirror server artifacts back to it.
Previously downloaded local data are not deleted by this plan.

`configs/server-data.toml` is the source of truth for the initial acquisition
batch: full Indonesian Aya Collection splits and the AITF EOS corpus. The
191-repository catalog is a separate inventory, not this job's download queue.

## Local preview

```bash
.venv/bin/python scripts/run_server_data.py
```

Preview only parses/prints the plan. It does not import Hugging Face clients or
access the network. `--execute` requires the configured server working directory.

## Deploy code and execute remotely

Upload only these code/config files to their corresponding paths on the server:

- `configs/server-data.toml`
- `scripts/run_server_data.py`
- `scripts/acquire_full_id.py`
- `scripts/prepare_local_tasks.py`
- `docs/server-data-plan.md`

Use SSH streaming/tar; this server's SSH configuration has no SFTP subsystem.
The existing server environment provides PyArrow, datasets, huggingface-hub and
python-dotenv. Use `.venv/bin/python` to preserve the V100-compatible Torch build.
HF authentication uses the server's existing cache/environment; do not copy or
print credentials from the Mac.

```bash
cd /workspace/gatra
.venv/bin/python -u scripts/run_server_data.py --execute
```

Run under a detached process with stdout/stderr redirected to a server log.
The runner records commands, exit codes and per-source log paths in
`logs/server-data-v1/runs.jsonl`. Existing output directories are rejected to
avoid silently overwriting a previous or partially completed acquisition.

## Data products and subsequent stages

| Source | Product | Intended use |
| --- | --- | --- |
| Aya Indonesian | Original Parquet plus train/validation/test `.jsonl.gz` | Instruction tuning |
| EOS | Raw Arrow snapshot plus train/validation `.jsonl.gz` | Continued text pretraining |

The runner resolves source revisions through the existing acquisition scripts;
each resulting manifest records the exact revision, row counts and file hashes.
The plan is acquisition provenance, not a promise that upstream `main` never
changes. Reproduce a completed acquisition using its recorded revision.

Inventory on the server before training: export counts, bytes, source/task
composition and length distribution with Gatra's tokenizer. Preserve upstream
validation/test and distinguish rows from documents and tokens. Aya's millions
of records include task/template duplication; do not interpret them as unique
examples. EOS's advertised token count is not a Gatra byte-token count.

Prepare pretraining bins and assistant-masked SFT data as distinct products;
the current SFT trainer accepts at most 512 bytes per complete prompt/answer.
Long examples need an explicit context/tokenizer experiment, not silent answer
truncation. More extensive quality filtering and cross-source near-dedup are
deferred work, not features claimed by this acquisition plan.

Training remains a subsequent stage until the server-side inventory is known.
The existing pretraining checkpoint is the intended control; retain v3 separately.

## Expanded acquisition batch

`configs/server-data-expanded.toml` independently adds IndoSum (fold 1), ITQA,
AMR-NLI, Indo-NER, Indo-Bloom, and STEM/non-STEM textbooks. It writes to separate
source directories from the Aya/EOS batch. Execute on the server with:

```bash
.venv/bin/python -u scripts/run_server_data.py \
  --plan configs/server-data-expanded.toml --execute
```

IndoSum additionally needs `gdown` and `nltk`; Treebank detokenization needs no
downloaded NLTK language models. Failures are logged per source and independent
sources continue; the batch exits unsuccessfully if any source failed. Inspect
partial destinations before retrying. This batch does not restart the bulk catalog.

Known interpretation issues: AMR-NLI labels entailment, Indo-NER labels are
silver, and Indo-Bloom generates questions rather than answering them. Textbooks
are grouped by book and reconstructed as pages. The existing deterministic
non-STEM split may yield empty validation, so that manifest must be checked and
the book split corrected before any evaluation. These acquisition exports are
not claimed to be clean, unique, or ready for the current 512-byte SFT trainer.
