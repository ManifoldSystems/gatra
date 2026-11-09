# Indonesian instruction datasets

Prepared on 2026-11-09 for instruction tuning after pretraining. These files are
not part of the active `gatra-10m-v100-300mb` pretraining mix. The existing trainer
does not yet implement assistant-only loss masking; preserving messages here
does not by itself enable supervised chat fine-tuning.

| Source | Train conversations | Validation conversations |
| --- | ---: | ---: |
| Alpaca GPT-4 Indonesian | 45,007 | 4,961 |
| OpenAssistant OASST1 Indonesian | 3 | 0 |
| OpenAssistant OASST1 multilingual (alternative to Indonesian subset) | 40,754 | 2,092 |

## Files and use

Each source directory contains `train.jsonl.gz`, `validation.jsonl.gz`, and a
manifest. Gzip archives are committed to Git; uncompressed JSONL files are local
generated files and are ignored. The manifest SHA-256 values refer to the
**uncompressed** JSONL bytes.

```bash
python - <<'PY'
from pathlib import Path
import gzip
import shutil
for archive in Path('data/instruction').glob('*/*.jsonl.gz'):
    destination = archive.with_suffix('')
    if destination.exists():
        continue
    with gzip.open(archive, 'rb') as src, destination.open('wb') as dst:
        shutil.copyfileobj(src, dst)
PY
```

Rows contain `messages` (ordered user/assistant messages), `text` (role-labelled
text for inspection or a future explicitly configured text pipeline), `source`,
and `group_id`. OASST rows also retain `message_ids`. Do not concatenate training
and validation files or repartition individual turns.

## Provenance and processing

- **Alpaca:** [Ichsan2895/alpaca-gpt4-indonesian](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian),
  revision `4790943e56ebac4626f0717e00a13e8d956645c2`.
  Reformatted from [FreedomIntelligence/alpaca-gpt4-indonesian](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-indonesian).
  Synthetic instruction/answer data; the selected dataset card declares
  **CC-BY-SA-4.0**. Preserve upstream attribution and applicable terms when
  redistributing. Split approximately 90/10 by hash of normalized input so
  identical prompts stay on one side; one exact normalized conversation duplicate
  was removed. Original input and output become user and assistant messages.
- **OpenAssistant:** [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
  revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, **Apache-2.0**.
  The ready-for-export train split has 12 Indonesian messages; validation has none.
  Complete Indonesian root-to-assistant paths are reconstructed using parent IDs.
  Deleted and negative-review messages are excluded. The best-ranked available
  Indonesian assistant reply per parent is selected, and shorter prefix paths are
  omitted. Original upstream splits are preserved and checked for tree isolation.
  This yields three training conversations: useful examples, not a substantial
  Indonesian conversation corpus or a standalone validation set.

Both sources retain their upstream licenses; the repository's code license does
not replace dataset licenses. See upstream cards for authors, citations, and terms.
No additional semantic quality or factuality filtering was performed.

## Reproduce acquisition

With the `data` extra installed, run from the repository root:

```bash
python scripts/prepare_instruction.py --out-dir data/instruction-rebuilt
```

Source revisions are pinned by default; `--alpaca-revision` and
`--oasst-revision` can override them. Authentication uses the Hugging Face cache
or `HF_TOKEN` environment variable. Credentials are not included in this repo.

The acquisition script refuses to overwrite existing source directories.

## Multilingual OpenAssistant

`oasst1-multilingual/` contains all languages of the pinned OASST1 ready-for-export
release (84,437 raw train messages and 4,401 raw validation messages). The larger
161,443-message all-trees export includes incomplete and rejected trees and is
not the input used here. Prepared conversation counts, byte sizes, language
breakdowns, and checksums are recorded in this directory's manifest.

This pinned ready-for-export release contains 25 distinct message-language tags;
the prepared paths cover 23 root languages. The upstream project's 35-language
description refers to its broader collection. Prepared training text is 60.13 MB
and validation text is 3.14 MB (UTF-8, including repeated branch context).
English and Spanish account for 17,868 and 13,073 training paths respectively.

```bash
python scripts/prepare_oasst_multilingual.py \
  --out-dir data/instruction-rebuilt/oasst1-multilingual
```

All valid reply branches are retained, not only the highest-ranked answer.
Complete root-to-assistant paths are emitted; shorter prefixes already covered
by longer valid paths are omitted. Deleted, negative-review, empty, orphaned,
cyclic, and malformed paths are excluded. Original tree splits are preserved;
exact normalized conversation duplicates are removed with validation priority.
Shared prefixes across sibling branches remain, so serialized text size includes
repeated context. Message IDs, language labels, and ranks are preserved per path.

Use this **instead of** `oasst1-id` in a mixture: the multilingual data already
contains Indonesian and otherwise duplicates those messages. This is a
conversation corpus, not a comprehensive or fact-checked knowledge database.
For an Indonesia-first 10M model, an initial experiment can allocate around 5%
of training tokens to multilingual conversations, then compare Indonesian and
multilingual validation metrics. This is a proposed token ratio, not a configured
file weight or an automatic addition to the ongoing pretraining run.

The multilingual export carries the same upstream Apache-2.0 license and
OpenAssistant attribution as the Indonesian subset.
