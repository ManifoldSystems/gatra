# Indonesian short-context SFT pilot preparation

## Prepared candidate counts

| Source | Train | Validation |
| --- | ---: | ---: |
| Alpaca ID | 12,000 | 1,200 |
| Cendol Indonesian QQP paraphrases | 12,000 | 1,200 |
| TyDi QA Indonesian | 1,538 | 162 |
| **Total** | **25,538** | **2,562** |

Total formatted train text is 5,871,244 bytes; validation is 600,566 bytes.
These are candidate exports, not a claim of uniformly correct examples or a
finished SFT run. Complete exports and gzip copies live in
`data/instruction/sft-pilot-v1/` on both machines.

### Manual spot-check findings

Five hash-ranked training examples per source were inspected (15 total, not a
representative statistical audit). This found problems that length/dedup checks
cannot catch:

- Cendol can lose specifics: a question comparing Sahara to Chihuahuan becomes
  a comparison against deserts in general. Another output leaves English words
  untranslated. Avoid treating every QQP pair as an exact semantic paraphrase.
- A TyDi example asks Newton's cause of death; its answer says mercury poisoning,
  while the passage only discusses mercury as a possible explanation of his
  eccentricity. Answer-span presence alone does not establish entailment.
- Alpaca includes image-editing requests with placeholder URLs and responses
  about model limitations, plus awkward lexical substitutions.

Before production SFT, perform a separate semantic/content-quality pass and
compare smaller per-source pilots. The equal 12K source caps here are collection
limits, not a recommended final training mixture. In particular, avoid allowing
paraphrase tasks to dominate a pilot intended to improve direct question answering.

Build the candidate dataset with the data dependencies installed:

```bash
python scripts/prepare_sft_pilot.py \
  --cendol-files data/train-00000-of-00029-07539590a64c4a85.parquet \
  --cendol-subsets id_qqp_nusantara_t2t
```

The script refuses to overwrite its output directory. Hub revisions are pinned.
The prepared Alpaca input is the existing `data/instruction/alpaca-id/` export;
its upstream revision and split hashes are copied into the output manifest.

## Sources and selection

- **Alpaca ID:** existing user/assistant pairs, original splits retained.
- **TyDi QA GoldP:** Indonesian IDs from `secondary_task`; original splits
  retained, original answer offsets checked. Full paragraphs only: no cropping
  using gold answers, and no truncation of prompts or answers. Article-title
  groups shared with validation are removed from training among eligible rows.
- **Cendol v1:** only the inspected `id_qqp_nusantara_t2t` subset and
  `paraphrase_1` template in the listed shard, with upstream `dataset_key=train`.
  Upstream validation and test examples are excluded. This subset teaches
  paraphrasing questions, not answering those questions. The separate Paracotta
  subset was inspected but not selected because examples included poor semantic
  preservation. Other languages/tasks/shards are not implicitly included.

Cendol paraphrase sentence pairs form connected groups so reversed pairs and
shared source/target sentences stay together. Hash those groups for a 90/10
pilot split. Exact normalized prompts are deduplicated across all sources; any
train/validation group or prompt conflict is resolved by dropping the training
example. Source caps are deterministic hash-ranked, not first-N selection.

The short examples are filtered for empty fields, replacement characters,
selected HTML/spam indicators and degenerate repeated trigrams. These checks
do not establish correctness or replace manual inspection. Cendol's translated
question pairs can remain awkward or only approximately equivalent.

## Byte format and training contract

```text
User: {prompt}

Assistant: {answer}
```

A trailing newline is included. The complete formatted sequence must be at most
512 UTF-8 bytes; rejected examples are counted in the manifest. No EOS special
token is added to the existing 256-byte vocabulary.

`assistant_start_byte` points to the first answer byte in `text.encode('utf-8')`.
For next-token loss, construct `x = tokens[:-1]`, `y = tokens[1:]`; mask labels
whose original token positions are before this boundary, and mask padding.
Include the answer's final newline as a supervised target. Keep examples
separate rather than concatenating unrelated conversations without boundaries.
The current full-text pretraining loop does not implement this masking; this
export alone is not an SFT training run.

## Evaluation limitations

The model already saw some Alpaca training examples during continued pretraining.
Prepared validation is useful for SFT development, but is not proof of unseen
capability. Other pretraining sources may overlap the QA paragraphs or tasks.
Semantic near-duplicates across sources are not exhaustively checked. Keep the
fixed-prompt suite and upstream test splits separate from training.

Hub cards mark Cendol v1 and TyDi QA as Apache-2.0; these are collection metadata,
not a blanket relicensing of all source texts. The pilot manifest retains source
identity. The Cendol shard is not redistributed in Git. Prepared pilot exports
remain local/server-side; script, manifest and documentation are tracked.
