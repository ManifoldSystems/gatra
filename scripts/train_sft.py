"""Small CUDA assistant-only SFT pilot from a Gatra byte-token checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F

from gatra.config import config_from_dict
from gatra.model import Gatra
from gatra.train import load_checkpoint, save_checkpoint


def batchify(rows, device):
    width = max(len(r['text'].encode()) - 1 for r in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        raw = row['text'].encode('utf-8')
        start = row['assistant_start_byte']
        assert 0 < start < len(raw) <= 512
        assert raw[start:] == (row['messages'][1]['content']+'\n').encode()
        x[i, :len(raw)-1] = torch.tensor(list(raw[:-1]))
        y[i, start-1:len(raw)-1] = torch.tensor(list(raw[start:]))
    return x.to(device), y.to(device)


def select_rows(path, caps):
    rows, audit, counts = [], Counter(), Counter()
    candidates = [json.loads(line) for line in path.read_text().splitlines()]
    candidates.sort(key=lambda r: hashlib.sha256(r['text'].encode()).hexdigest())
    for row in candidates:
        text = row['text']
        answer = row['messages'][1]['content']
        if re.search(r'https?://|example\.com|jmpl\||<[^>]+>|\b(?:gambar|foto|video|audio)\b', text, re.I):
            audit['unsupported_media_or_markup'] += 1
            continue
        if re.search(r'\b(?:OpenAI|ChatGPT|Cendol)\b', answer, re.I):
            audit['other_model_identity'] += 1
            continue
        if 'Newton' in text and 'raksa' in text:
            audit['spotcheck_unsupported_answer'] += 1
            continue
        if len(answer.encode()) > 256:
            audit['long_answer'] += 1
            continue
        source = row['source']
        if counts[source] >= caps[source]:
            audit['source_cap'] += 1
            continue
        rows.append(row)
        counts[source] += 1
    return rows, dict(counts), dict(audit)


@torch.inference_mode()
def evaluate(model, rows, device, batch_size):
    model.eval()
    totals, counts = Counter(), Counter()
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        for start in range(0, len(group), batch_size):
            x, y = batchify(group[start:start+batch_size], device)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum')
            totals[source] += loss.item()
            counts[source] += (y != -100).sum().item()
    model.train()
    return dict(loss=sum(totals.values())/sum(counts.values()),
                per_source={s:totals[s]/counts[s] for s in totals}, target_bytes=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/instruction/sft-pilot-v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--prepared', action='store_true', help='Use all rows of an independently prepared dataset')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    device = torch.device('cuda')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = config_from_dict(payload['config'])
    assert config.data.tokenizer == 'byte' and config.model.block_size == 512
    config.name = args.out.name
    config.train.precision = 'fp16'
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    model = Gatra(config.model).to(device)
    load_checkpoint(args.checkpoint, model, None, device)
    # New objective: intentionally reset optimizer/scaler; base weights retained.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    if args.prepared:
        train = [json.loads(line) for line in (args.data/'train.jsonl').read_text().splitlines()]
        val = [json.loads(line) for line in (args.data/'validation.jsonl').read_text().splitlines()]
        train_counts, val_counts = (dict(Counter(r['source'] for r in rows)) for rows in (train, val))
        train_audit = val_audit = {'prepared_input': True}
        for row in train + val:
            batchify([row], torch.device('cpu'))
    else:
        train, train_counts, train_audit = select_rows(args.data/'train.jsonl', {'alpaca-id':4000,'tydiqa-id':2000,'cendol':1000})
        val, val_counts, val_audit = select_rows(args.data/'validation.jsonl', {'alpaca-id':200,'tydiqa-id':200,'cendol':200})
    assert train and val
    for field in ('prompt_hash', 'group_id'):
        assert not {r[field] for r in train} & {r[field] for r in val}
    steps = math.ceil(len(train)/args.batch_size)*args.epochs
    config.train.max_iters = steps
    metadata = dict(source_checkpoint=str(args.checkpoint), source_step=payload['step'],
                    source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    source_tokens_seen=payload['tokens_seen'], epochs=args.epochs, planned_steps=steps,
                    train_counts=train_counts, validation_counts=val_counts,
                    train_filters=train_audit, validation_filters=val_audit,
                    learning_rate=args.lr, optimizer='fresh AdamW', scaler='fresh GradScaler',
                    objective='assistant-only CE; prompt and padding labels -100',
                    step_counter='SFT-local attempted batches', seed=1337,
                    quality='Heuristic and limited spot-check filtering; not fully human verified')
    for split, rows in [('train',train),('validation',val)]:
        raw=''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
        (args.out/f'{split}.jsonl').write_bytes(raw)
        metadata[split+'_sha256']=hashlib.sha256(raw).hexdigest()
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('[SFT]',json.dumps(metadata),flush=True)
    del payload
    start_time=time.perf_counter()
    best=float('inf')
    seen=0
    supervised=0
    skipped=0
    step=0

    def checkpoint_eval(epoch):
        nonlocal best
        result=evaluate(model,val,device,args.batch_size)
        metric=dict(step=step, epoch=epoch, val_loss=result['loss'], per_source=result['per_source'],
                    tokens_seen=seen, supervised_bytes=supervised, skipped_updates=skipped,
                    scaler_scale=scaler.get_scale(), elapsed_s=time.perf_counter()-start_time,
                    base_checkpoint=metadata['source_checkpoint'], base_step=metadata['source_step'])
        print(f"\n[SFT STEP {step:,} / {steps:,}] [epoch: {epoch}]\n[VAL] {json.dumps(result)}\n[AMP] [skipped: {skipped}] [scale: {scaler.get_scale():g}]",flush=True)
        if result['loss'] < best:
            best=result['loss']
            save_checkpoint(args.out/'best.pt',model,optimizer,step,config,metric,seen,best,device,scaler)
        save_checkpoint(args.out/'latest.pt',model,optimizer,step,config,metric,seen,best,device,scaler)
        with (args.out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(metric)+'\n')

    checkpoint_eval(0)
    for epoch in range(1,args.epochs+1):
        order=list(train)
        random.Random(1337+epoch).shuffle(order)
        for offset in range(0,len(order),args.batch_size):
            rows=order[offset:offset+args.batch_size]
            x,y=batchify(rows,device)
            lr=args.lr*min(1.,(step+1)/50)*max(0.1,1-step/max(steps,1))
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits,_=model(x)
                loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite SFT loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped+=int(scaler.get_scale()<scale)
            step+=1
            seen+=sum(len(r['text'].encode())-1 for r in rows)
            supervised+=(y!=-100).sum().item()
            if step%100==0:
                print(f'\n[TRAIN] [SFT step: {step}/{steps}] [loss: {loss.item():.4f}] [lr: {lr:.2e}] [skipped: {skipped}]',flush=True)
        checkpoint_eval(epoch)
    print(f'\n[DONE] [SFT steps: {step}] [best val: {best:.4f}]\n[CHECKPOINT] {args.out}/best.pt',flush=True)


if __name__ == '__main__':
    main()
