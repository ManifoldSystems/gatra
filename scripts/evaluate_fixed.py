"""Paired validation windows and fixed-prompt checkpoint comparison (no updates)."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from gatra.config import config_from_dict
from gatra.data import TokenDataset
from gatra.model import Gatra
from gatra.tokenizer import load_tokenizer
from gatra.train import load_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--bin-dir', required=True)
    parser.add_argument('--prompts', type=Path, default=Path('data/eval/fixed-prompts-v1.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    device = torch.device(args.device)
    torch.set_num_threads(4)
    settings = dict(precision='fp32', validation_seed=20261109, validation_batches=100,
                    validation_batch_size=16, validation_context=256,
                    generation_seeds=[1337, 2026], max_new_tokens=192,
                    temperature=0.7, top_k=40, top_p=0.9,
                    torch_version=torch.__version__, device=str(device), bin_dir=args.bin_dir,
                    prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest())
    results = []
    for checkpoint in args.checkpoints:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = config_from_dict(payload['config'])
        tokenizer = load_tokenizer(config.data.tokenizer or 'byte')
        if config.data.tokenizer != 'byte' or config.model.block_size < 256:
            raise ValueError('This comparison requires byte-tokenized checkpoints with context >=256')
        model = Gatra(config.model).to(device)
        load_checkpoint(checkpoint, model, None, device)
        model.eval()
        context = config.model.block_size
        # Shared 256-byte windows permit comparison with the original checkpoint.
        config.model.block_size = 256
        config.train.batch_size = 16
        config.data.bin_dir = args.bin_dir
        dataset = TokenDataset(config)
        torch.manual_seed(settings['validation_seed'])
        losses = []
        for _ in range(settings['validation_batches']):
            x, y = dataset.get_batch('val', device)
            _, loss = model(x, y)
            losses.append(loss.item())
        config.model.block_size = context
        mean = sum(losses) / len(losses)
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      step=payload['step'], tokens_seen=payload.get('tokens_seen'),
                      generation_context=context, val_loss=mean, byte_perplexity=math.exp(mean),
                      val_batch_losses=losses, samples=[])
        del payload, dataset
        for item in prompts:
            for seed in settings['generation_seeds']:
                torch.manual_seed(seed)
                idx = torch.tensor([tokenizer.encode(item['prompt'])], dtype=torch.long, device=device)
                tokens = model.generate(idx, max_new_tokens=settings['max_new_tokens'],
                                        temperature=settings['temperature'], top_k=settings['top_k'],
                                        top_p=settings['top_p'])
                continuation = tokenizer.decode(tokens[0, idx.shape[1]:].tolist())
                result['samples'].append(dict(**item, seed=seed, continuation=continuation))
        results.append(result)
        (args.out/'results.json').write_text(json.dumps(dict(settings=settings, results=results), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'val_batch_losses')}), flush=True)
        del model
    lines = ['# Fixed-prompt evaluation v1', '', 'Exploratory samples, not a standardized capability score.',
             'All checkpoints evaluated in FP32 on identical expanded-validation windows (256 byte tokens).',
             'Generation uses each checkpoint’s native context; prompts are not checked for training contamination.', '']
    for result in results:
        lines.extend([f"## {result['checkpoint']}", '', f"Step {result['step']}; paired validation loss {result['val_loss']:.6f}.", ''])
        for item in result['samples']:
            lines.extend([f"### {item['id']} / seed {item['seed']}", '', 'Prompt:', '```text', item['prompt'], '```',
                          'Continuation:', '```text', item['continuation'], '```', ''])
    (args.out/'samples.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
