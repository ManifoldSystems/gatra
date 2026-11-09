# IndoNLU instruction candidates v1

Prepared FacQA, NERGrit and SmSA for the existing 512-byte assistant-only SFT
format. This is a data preparation run; these candidates have not been trained.

## Sources and reproducibility

- HF: `indonlp/indonlu@939bfb4e87cd0f4f717f4222ec19c55cdc302982`.
- Actual data: `IndoNLP/indonlu@ce728f6926a36174b9923dfe49d6a6839b6e9bb7` on GitHub,
  following the HF loader's paths with its mutable `master` replaced by a commit.
- Script: `scripts/prepare_indonlu.py` (Python standard library plus the existing
  `prepare_sft_pilot.make_row` filtering/formatting helper).
- Exports: `data/instruction/indonlu-v1/{train,validation,test}.jsonl`.
- Manifest records every source URL/hash, output hash, filtering count and revision.
- `raw/` retains original data, HF card/loader and upstream license.
- HF card declares MIT; upstream root LICENSE is Apache-2.0. These repository
  labels conflict and do not establish blanket rights for every underlying text.

```bash
.venv/bin/python scripts/prepare_indonlu.py --out data/instruction/indonlu-v1
.venv/bin/python -m pytest tests/test_prepare_indonlu.py -q
```

The output directory must not already exist. Use a new directory to reproduce.
Raw and generated records are mirrored on the Mac and server, not committed;
the script, manifest, tests and this report are committed.

## Retained instruction rows

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| FacQA | 1,196 | 175 | 226 |
| NERGrit | 4,616 | 593 | 582 |
| SmSA | 10,530 | 1,231 | 498 |
| **Total** | **16,342** | **1,999** | **1,306** |

NER generates three typed questions per source sentence before length filtering,
so these counts are not counts of unique sentences. No context or answer is
truncated. Every retained prompt + answer + framing fits 512 UTF-8 bytes.

## Conversion and split contract

- FacQA: reconstruct passage and question using space-joined source tokens;
  require a single valid contiguous BIO span and use its tokens as the answer.
  Reject zero/multiple spans, unaligned tags and invalid BIO transitions.
- NERGrit: decode PERSON, PLACE and ORGANISATION spans, preserving multi-token
  names and separate adjacent B-tags. Ask one question for each type; return
  unique names in occurrence order, separated by `; `, or `Tidak ada`.
- SmSA: preserve input text and map positive/neutral/negative to
  positif/netral/negatif. No label inference by another model.
- Answers are single-line and end with a supervised newline. Source split/index
  and extracted spans or original sentiment labels accompany each example.
- Keep upstream splits. Hash normalized source contexts across **all** raw rows,
  including rows later excluded for length. Test takes priority over validation,
  and validation over train; discard lower-priority conflicts without moving rows.
- This removes 896 FacQA train rows and 83 validation rows with held-out context
  overlap, plus 16 SmSA train rows. Duplicate prompts are also removed.
- Final group/prompt intersections across the three exports are empty. This is
  exact normalized-context protection, not semantic or article-level deduplication.
- Previous Cendol/pretraining exposure has not been exhaustively checked. Reserve
  test for final evaluation; do not call it a contamination-free benchmark.

## Quality audit and sampling considerations

Reviewed five hash-ranked **train** examples per source (SHA-256 of serialized
text; test contents were not used for manual tuning). All 15 have mechanically
consistent conversions, but source meaning/annotation issues remain:

- FacQA indices 2341, 2284, 269, 1148, 2328: dates and names are supported by
  source passages; index 2341 says “Gunung Bukit Krakatau” in the question although
  the context says “Gunung Krakatau”. Tokenization leaves spaces around punctuation.
- NERGrit indices 300, 72, 1042, 1016, 1167: NBC is extracted as an organization;
  several type queries correctly yield no names. Index 1016 labels “Ramadhan”
  as PERSON inside the film title “Ramadhan dan Ramona”, a questionable annotation.
- SmSA indices 6615, 6081, 2312, 9823, 4257: labels are broadly consistent with
  the expressed sentiment; colloquial/noisy text and strongly abusive wording
  occur in inputs. Assistant targets remain short sentiment labels.
- An additional initial raw FacQA inspection found train index 4 asking who
  *will move* a statue while the passage says it *does not need to be moved*.
  Correct span extraction alone does not establish a valid question premise.

NER train contains **2,507/4,616 empty answers (54.3%)**. Report per-type results
and entity-present accuracy separately in any future evaluation. SmSA train
contains 6,051 positive, 3,343 negative and 1,136 neutral examples. The combined
export is dominated by sentiment (64.4%); these are candidate inventories, not
a recommended uncapped training mixture. Select a balanced task mixture before
training rather than treating every exported row as equally informative.

## Verification

Seven parser/span/split tests passed. All 19,647 exported examples were checked
with the SFT batch builder for byte lengths and assistant-label alignment. Output
SHA-256 values and disjoint group/prompt sets were verified locally. Mirror transfer
verifies every copied file's SHA-256 remotely.
