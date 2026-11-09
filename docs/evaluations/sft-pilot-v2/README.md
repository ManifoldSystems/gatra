# SFT v2: single-line curriculum

Completed 3 epochs / 1,749 updates in about 90 seconds of training/evaluation,
with zero AMP overflows. Assistant-token validation loss fell from 1.3123 to
0.1548. These values use the v2 validation mix and cannot be compared directly
to v1's aggregate loss. Preparation/training details: [pilot v2](../../sft-pilot-v2.md).

| Task loss | Before | After |
| --- | ---: | ---: |
| Addition | 4.1813 | 1.1859 |
| Copy | 0.2847 | 0.0010 |
| Synthetic extraction | 2.4014 | 0.0009 |
| TyDi QA | 0.7391 | 0.2318 |

Manual greedy single-line probes (`probes.json`) show exact copying of the
previous diagnostic sentence and a newline ending. Other diagnostics still fail:
2+3 produces 4; the two-cat question gives only Miko; Japan's capital produces
Asia Tenggara. A requested story continuation merely repeats “langit mendung.”
All five end in newline under the explicit stop option. Ending an answer is not
evidence that the answer is correct.

Keep the base checkpoint as the general-text control. This is a narrow task
adaptation experiment, not a generally improved chatbot. Do not interpret high
copy/template extraction accuracy as broad reading comprehension or reasoning.

To try:

```bash
.venv/bin/python -m gatra.generate \
  --checkpoint checkpoints/gatra-10m-sft-pilot-v2/best.pt \
  --device cuda --stop-at-newline --top-k 1 --top-p 1 \
  --max-new-tokens 128 \
  --prompt $'User: Tulis ulang persis: Saya suka membaca buku.\n\nAssistant: '
```
