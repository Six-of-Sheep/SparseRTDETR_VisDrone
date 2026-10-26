"""Render existing diagnostic counts, with SHA verification. No scoring or models."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def render(summary, expected, output):
    raw=summary.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('summary SHA differs')
    d=json.loads(gzip.decompress(raw))
    if d['status']!='PASS' or d['images']!=548 or not d['native_primary_original_count_match']:
        raise ValueError('not the complete fixed real diagnostic')
    output.mkdir(parents=True,exist_ok=False)
    def table(name, records):
        fields=list(records[0])
        with (output/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records)
    rows=[]
    for key,v in d['stage_counts'].items():
        arm,scale,t,k=key.split('/')
        rows.append(dict(arm=arm,size=scale,IoU=t,budget=k,**v,
                         native_primary_final_match=d['native_primary_final'][key]['matched']))
    table('DIAGNOSTIC_COUNTS.csv',rows)
    rows=[]
    for key,v in d['breakdowns'].items():
        arm,scale,k,kind,member=key.split('/')
        rows.append(dict(arm=arm,size=scale,IoU='.75',budget=k,group=kind,member=member,**v))
    table('DIAGNOSTIC_BREAKDOWNS.csv',rows)
    rows=[]
    for key,count in d['transitions'].items():
        pair,scale,t,k,stage,state=key.split('/')
        rows.append(dict(pair=pair,size=scale,IoU=t,budget=k,stage=stage,N_state=state[0],other_state=state[1],count=count))
    table('DIAGNOSTIC_TRANSITIONS.csv',rows)
    index=[]
    for p in sorted(output.iterdir()):
        b=p.read_bytes();index.append(dict(path=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    (output/'OUTPUT_INDEX.json').write_text(json.dumps(dict(input_sha256=expected,files=index,
                  scope='existing diagnostic counts only; no AP/AR'),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED_EXISTING_COUNTS',tables=3,rows=[len(d[x]) for x in ('stage_counts','breakdowns','transitions')],AP_AR_computed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.summary,a.sha256,a.output)
