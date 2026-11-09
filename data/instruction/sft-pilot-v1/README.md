# Short-context Indonesian SFT candidates

Prepared: 25,538 train and 2,562 validation examples from Alpaca ID, Cendol v1
Indonesian QQP paraphrases, and TyDi QA Indonesian GoldP. Full formatted examples
fit 512 UTF-8 bytes without truncation. `assistant_start_byte` identifies the
answer boundary for future assistant-only loss.

See [preparation and quality findings](../../../docs/sft-pilot.md) before use.
These are technically filtered candidates, not human-verified training data.
Manual inspection found semantic errors that require a further quality pass.

`manifest.json` records revisions, counts, filters and checksums. Full JSONL and
gzip exports are stored locally and on the V100 server, outside Git tracking.
No SFT training has been launched with this export.
