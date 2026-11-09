# Community dataset candidates

Recommended by the community on 2026-11-09. These candidates are not part of the
completed initial 10M pretraining run. Acquisition scripts preserve provenance;
dataset availability and terms differ by source.

| Dataset | Role | Acquisition |
| --- | --- | --- |
| [Lyon28/Corpus-Indonesia](https://huggingface.co/datasets/Lyon28/Corpus-Indonesia) | General Indonesian corpus candidate | Bounded 100 MB text sample spread over seven shards |
| [junwatu/indonesian-recipes](https://huggingface.co/datasets/junwatu/indonesian-recipes) | Recipe/procedural language | Acquired after access approval; 66,419 train / 3,496 validation |
| [junwatu/javanese-multilingual-lexicon](https://huggingface.co/datasets/junwatu/javanese-multilingual-lexicon) | Historical Javanese/Kawi vocabulary | Acquired after access approval; 182,241 train / 20,671 validation |

Prepared outputs have been copied to both the GPU server and local checkout and
verified by SHA-256 and row counts. Corpus Indonesia produced 330,650 train /
36,554 validation rows, 100.00 MB text total. Recipes produced 61.78 MB text;
lexicon produced 28.80 MB. Corpus audit flags: 365 gambling-keyword documents,
374 HTML-like documents, 304 replacement-character documents (categories may
overlap). Lexicon has 31 gambling-keyword entries; dictionary mentions alone do
not imply spam. No automatic keyword deletion was applied.

## Corpus Indonesia

Pinned revision: `11ff800114081d81bc9fb4ef49aa30350037e677`.
The card declares Apache-2.0 and 19,483,721 rows, 3,233,596,819 uncompressed dataset
bytes, approximately 1.90 GB downloaded. Its size-category tag disagrees with
the row count; use the detailed metadata rather than the tag. The card does not
document collection provenance or quality filtering.

Initial inspection found an outdated province count and a joke appended to an
encyclopedia-style passage. Treat it as a candidate needing quality review.
The preparation script samples an equal accepted-text byte budget per shard with
a 1,000-row shuffle buffer; this is **not** a uniform sample of all rows. Exact
whitespace-normalized duplicates are checked against existing top-level data
JSONL files and within the new corpus. The manifest records counts and keyword
flags (gambling, HTML-like text, replacement characters); flags are audit signals,
not proof of low quality or comprehensive filtering. Near-duplicate detection and
fact checking are not implemented.

## Indonesian Recipes

Pinned revision: `6c7682e9a96b64be51d454cc991a8c0f1e7f2131`.
Fields: title, ingredient list, ordered steps, and size counters. The script
preserves the upstream train/eval split, formats ingredients and numbered steps,
and retains the source record. Exact duplicates prioritize eval; near-identical
titles/recipes across splits still need auditing.

The card declares `research-use-as-is` and explicitly grants no rights to the
underlying recipes. Record this distinction when deciding downstream use or
redistribution; it is not an Apache/MIT-licensed corpus.

## Javanese Multilingual Lexicon

Pinned revision: `cdd4e83cb890e7fe8d99d48e0002de3cf62c206d`.
202,912 dictionary entries derived from 32 historical sources at Sastra.org,
including Javanese, Kawi, Indonesian, Dutch, English, and French. The `all` config
contains the pair-specific entries, so do not concatenate it with those subsets.
Keep source references, diacritics and historical spellings. Split by normalized
headword so senses of the same headword stay on one side. This is a lexicon, not
aligned modern translation sentences or conversational instruction data.

License: **CC-BY-NC-4.0**, original content © Yayasan Sastra Lestari. Attribute both
Sastra.org and Junwatu. Non-commercial restriction applies.

## Run

With the data extra installed and HF authentication configured:

```bash
python scripts/prepare_community.py --source corpus-indonesia
python scripts/prepare_community.py --source recipes
python scripts/prepare_community.py --source javanese-lexicon
```

Accept the access terms on the two Junwatu dataset pages using the account whose
token is installed. Being able to read a gated dataset card does not imply data
file access. A 403 means acquisition is blocked; do not represent those sources
as downloaded. Scripts refuse existing output directories; inspect partial runs
before retrying to a fresh `--out-root`.

Outputs live under `data/community/`, ignored by Git. This keeps large candidate
corpora and restricted upstream content out of Git history; scripts, source
references, and preparation metadata can be shared independently. Token files
and HF credentials must never be committed.

## Basis Global Emotions: speech candidate

[basis-ai/basis-global-emotions](https://huggingface.co/datasets/basis-ai/basis-global-emotions),
card checked at revision `57f26ff90a6240e54df94a2fabaf0f744df02575`.
This is expressive speech data: 525.17 hours, 137,696 recordings, 54,575 series,
10,420 speakers, 108 locales. Indonesian has 1,824 recordings / 7.47 hours,
Javanese 1,205 / 4.30 hours, Sundanese 1,493 / 5.80 hours.

Main uses are expressive TTS and speech emotion recognition. `given_line` is the
instructed script, not a verified transcript. Multiple readings repeat one script
at different intensities, so text-only extraction would need script/series dedup.
Speech evaluation should use speaker-disjoint splits and keep a series together.
This is not currently acquired or included in Gatra's text training data.

Access requires manual approval. The card describes the custom Basis Data License
as permitting research and commercial model training, while prohibiting
redistribution of recordings and derivatives containing them. Consult the actual
LICENSE before a speech project; do not commit recordings to this repository.
