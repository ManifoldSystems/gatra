import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

spec=importlib.util.spec_from_file_location('train_sft',Path(__file__).resolve().parents[1]/'scripts/train_sft.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(prompt, answer):
    prefix=f'User: {prompt}\n\nAssistant: '
    return dict(text=prefix+answer+'\n',assistant_start_byte=len(prefix.encode()),
                messages=[dict(content=prompt),dict(content=answer)])


def test_only_answer_and_newline_receive_supervision():
    rows=[row('Arti “pagi”?','Awal hari.'),row('A','é')]
    x,y=module.batchify(rows,torch.device('cpu'))
    for i,r in enumerate(rows):
        raw=r['text'].encode()
        start=r['assistant_start_byte']
        assert (y[i,:start-1]==-100).all()
        assert y[i,start-1:len(raw)-1].tolist()==list(raw[start:])
        assert (y[i,len(raw)-1:]==-100).all()
        assert x[i,:len(raw)-1].tolist()==list(raw[:-1])
    logits=torch.randn(*y.shape,256,requires_grad=True)
    F.cross_entropy(logits.flatten(0,1),y.flatten()).backward()
    assert (logits.grad[y==-100]==0).all()
    assert logits.grad[y!=-100].abs().sum()>0
