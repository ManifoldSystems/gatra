# Single-line SFT pilot v2

Starts independently from the pretraining checkpoint at step 88,138, with fresh
optimizer/scaler. No Cendol paraphrase or Alpaca examples. Three epochs, FP16,
batch 16, context 512 bytes, peak LR 1e-5; 1,749 attempted steps.

| Task | Train | Validation |
| --- | ---: | ---: |
| Integer addition | 1,466 | 172 |
| Exact copying | 1,088 | 112 |
| Extract owner/color/location | 5,262 | 570 |
| Short TyDi QA | 1,511 | 157 |

`scripts/prepare_sft_v2.py` builds rule-generated labels and retains the filtered
TyDi QA selection. Addition operand pairs (including reversed order), copied
sentences and extraction contexts are split as groups. Templates of the same
underlying group cannot cross splits. Synthetic validation uses the same task
templates and vocabulary; this measures narrow compositional generalization,
not general conversational skill. Existing 2+3 diagnostics are excluded from
addition training, and the prior named-cat/copy prompts are not used as examples.
TyDi QA still has the source-quality and prior-pretraining caveats from v1.

Every answer is single-line and supervised through its final newline byte.
`gatra.generate --stop-at-newline` stops at the first newly generated newline,
not at newline bytes already in the prompt. This is an explicit inference option
for single-line answers, not a general EOS token or a multiline chat solution.
Without that option, the generator still runs to its normal token budget.

```bash
python scripts/prepare_sft_v2.py
python scripts/train_sft.py --prepared --epochs 3 --lr 1e-5 \
  --data data/instruction/sft-pilot-v2 \
  --checkpoint checkpoints/gatra-10m-expanded-ctx512-fp16/latest.pt \
  --out checkpoints/gatra-10m-sft-pilot-v2
```

Evaluate before/after using the same greedy decoder and newline stop for both
models: `scripts/evaluate_sft_tasks.py`, 25 hash-ranked validation rows per task,
128 new byte tokens. Record strict stripped-string exact match and whether a
newline was generated. Also compare the original fixed prompts and paired
general-text validation; answer loss alone was misleading in v1.
