import gzip
import json
from pathlib import Path
import subprocess
import sys


def test_new_sources_keep_heldout_groups_even_when_validation_is_too_long(tmp_path):
    root = tmp_path/'data'
    for name in ('indoqa', 'prdect', 'commerce'):
        folder = root/'server-expanded-v1/tasks'/name
        folder.mkdir(parents=True)
        train = []
        for group in ('shared', 'unique'):
            train.append(dict(messages=[dict(role='user', content=name+' '+group),
                                        dict(role='assistant', content='positive' if name == 'commerce' else 'jawaban')],
                              group_id=group, provenance={}))
        heldout = dict(messages=[dict(role='user', content='x'*600),
                                 dict(role='assistant', content='positive' if name == 'commerce' else 'jawaban')],
                       group_id='shared', provenance={})
        if name == 'indoqa':
            for row in train + [heldout]:
                row['messages'][0]['content'] = ('Jawab berdasarkan konteks berikut. Jika informasi tidak tersedia, '
                    'nyatakan bahwa jawaban tidak tersedia dalam konteks.\n\nKonteks:\n' + row['messages'][0]['content'])
        for split, rows in [('train',train), ('validation',[heldout])]:
            with gzip.open(folder/f'{split}.jsonl.gz','wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
    out = tmp_path/'mix'
    script = Path(__file__).resolve().parents[1]/'scripts/prepare_server_mix.py'
    subprocess.run([sys.executable,str(script),'--root',str(root),'--out',str(out)],check=True,capture_output=True)
    rows = [json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert all(r['group_id'].endswith(':unique') for r in rows)
    assert next(r for r in rows if r['source']=='commerce')['messages'][1]['content']=='positif'
    for row in rows:
        assert row['text'].encode()[row['assistant_start_byte']:] == (row['messages'][1]['content']+'\n').encode()
