import torch

from gatra.config import ModelConfig
from gatra.model import Gatra


def test_newline_stop_ignores_prompt_newlines_and_stops_on_generated_byte():
    model=Gatra(ModelConfig(n_embd=16,n_head=2,n_layer=1,block_size=32))
    def forward(idx, targets=None):
        logits=torch.full((*idx.shape,256),-1000.)
        logits[:,:,-1]=0
        logits[:,:,10]=1000
        return logits,None
    model.forward=forward
    prompt=torch.tensor([[65,10,66]])
    output=model.generate(prompt, max_new_tokens=5, stop_token_id=10)
    assert output.tolist()==[[65,10,66,10]]
    assert model.generate(prompt,max_new_tokens=5).shape[1]==8
