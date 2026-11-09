"""Matched greedy single-line task evaluation on a hash-ranked held-out sample."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch

from gatra.config import config_from_dict
from gatra.model import Gatra
from gatra.train import load_checkpoint


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-task',type=int,default=25)
    args=p.parse_args()
    torch.set_num_threads(4)
    rows=[json.loads(x) for x in args.validation.read_text().splitlines()]
    rows.sort(key=lambda r:hashlib.sha256(r['text'].encode()).hexdigest())
    selected=[]; counts=defaultdict(int)
    for row in rows:
        if counts[row['source']]<args.per_task:
            selected.append(row); counts[row['source']]+=1
    results=[]
    for checkpoint in args.checkpoints:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=config_from_dict(payload['config'])
        model=Gatra(config.model).cuda().eval()
        load_checkpoint(checkpoint,model,None,torch.device('cuda'))
        del payload
        samples=[]; correct=defaultdict(int); ended=defaultdict(int)
        for row in selected:
            prompt=row['text'].encode()[:row['assistant_start_byte']]
            idx=torch.tensor([list(prompt)],device='cuda')
            output=model.generate(idx, max_new_tokens=128,temperature=1.,top_k=1,stop_token_id=10)
            generated=output[0,len(prompt):].tolist()
            answer=bytes(generated).decode('utf-8',errors='replace').strip()
            expected=row['messages'][1]['content'].strip()
            ok=answer==expected
            stopped=generated[-1]==10
            correct[row['source']]+=int(ok); ended[row['source']]+=int(stopped)
            samples.append(dict(source=row['source'],prompt=prompt.decode(),expected=expected,
                                answer=answer,exact_match=ok,stopped_on_newline=stopped))
        result=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    counts=dict(counts),correct=dict(correct),stopped=dict(ended),samples=samples)
        results.append(result)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(dict(settings=dict(decoding='FP32 greedy top_k=1',max_new_bytes=128,stop_byte=10,
            validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),per_task=args.per_task),results=results),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
        del model


if __name__=='__main__': main()
