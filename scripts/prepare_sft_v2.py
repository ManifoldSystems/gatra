"""Bounded single-line instruction curriculum; deterministic rule-generated labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from prepare_sft_pilot import make_row, digest
from train_sft import select_rows


def main():
    out = Path('data/instruction/sft-pilot-v2')
    out.mkdir(parents=True, exist_ok=False)
    rows = {'train': [], 'validation': []}
    # Hold out complete underlying groups, not alternate templates of a fact.
    def add(prompt, answer, task, group):
        split = 'validation' if int(digest(group)[:8], 16) / 2**32 < .1 else 'train'
        row, reason = make_row(prompt, answer, task, group, {'label_method':'deterministic rule'})
        assert reason is None and '\n' not in answer
        rows[split].append(row)

    # Do not include 2+3 or its commuted pair from the existing diagnostic suite.
    for a in range(40):
        for b in range(a,40):
            if (a,b)==(2,3): continue
            for prompt in [f'Berapa hasil {a} + {b}?', f'Hitung {a} + {b}. Jawab angkanya saja.']:
                add(prompt,str(a+b),'addition',f'add:{a}:{b}')
    subjects=['Adi','Nina','Sari','Doni','Lina','Rudi','Tono','Wati','Dedi','Maya']
    verbs=['membaca','membawa','membeli','menyimpan','meminjam','menemukan']
    objects=['buku','pensil','tas','payung','sepatu','kertas','jaket','topi','botol','penggaris']
    for who in subjects:
        for verb in verbs:
            for obj in objects:
                sentence=f'{who} {verb} {obj}.'
                for prefix in ['Tulis ulang persis: ', 'Salin kalimat ini tanpa perubahan: ']:
                    add(prefix+sentence,sentence,'copy','copy:'+digest(sentence))
    rng=random.Random(20261109)
    pairs=set()
    for _ in range(2500):
        who=rng.choice(subjects)
        obj=rng.choice(objects)
        color=rng.choice(['merah','biru','kuning','hijau','putih','hitam','ungu','cokelat'])
        place=rng.choice(['meja','lemari','kursi','rak','lantai','kotak'])
        key=(who,obj,color,place)
        if key in pairs: continue
        pairs.add(key)
        context=f'{who} memiliki {obj} berwarna {color}. {obj.capitalize()} itu disimpan di {place}.'
        group='extract:'+digest(context)
        for question,answer in [(f'Apa warna {obj} milik {who}?',color),
                                (f'Di mana {obj} itu disimpan?',place),
                                (f'Siapa pemilik {obj} itu?',who)]:
            add(f'Baca teks berikut:\n{context}\n\nPertanyaan: {question}',answer,'extraction',group)
    # Keep actual QA alongside the narrow, mechanically checkable curriculum.
    for split in rows:
        selected,_,_=select_rows(Path('data/instruction/sft-pilot-v1')/f'{split}.jsonl',
                                  {'alpaca-id':0,'cendol':0,'tydiqa-id':2000})
        for row in selected:
            if '\n' not in row['messages'][1]['content']:
                rows[split].append(row)
    seen=set()
    for split in ('validation','train'):
        unique=[]
        for row in rows[split]:
            if row['prompt_hash'] not in seen:
                seen.add(row['prompt_hash']); unique.append(row)
        rows[split]=unique
    assert not {r['group_id'] for r in rows['train']} & {r['group_id'] for r in rows['validation']}
    manifest=dict(version=2, seed=20261109, max_bytes=512, ending='newline byte 10',
                  counts={s:dict(Counter(r['source'] for r in rr)) for s,rr in rows.items()},
                  limitations=['Synthetic held-out results measure template/task generalization, not broad chat ability.',
                               'TyDi QA remains heuristically filtered, not fully human-verified.',
                               'No paraphrase or Alpaca examples; narrow experimental curriculum.'])
    for split,rr in rows.items():
        raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr).encode()
        (out/f'{split}.jsonl').write_bytes(raw)
        manifest[split+'_sha256']=hashlib.sha256(raw).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
