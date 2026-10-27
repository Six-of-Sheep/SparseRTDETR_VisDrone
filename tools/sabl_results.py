"""SHA-bound existing scoring JSON to tables; never computes AP/AR."""
import argparse,json,hashlib,math
from pathlib import Path
COCO=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','AR_small','AR_medium','AR_large','AP75_small')
PRIMARY=('AP','AP50','AP75','AR1','AR10','AR100','AR500')
ARMS=('N_original','N24','A24','S24')

def metrics(score):
    c=score['COCO_bridge']['metrics'];p=score['official']
    result={**{'COCO_'+k:float(c[k])*100 if c[k]>=0 else None for k in COCO},**{'primary_'+k:float(p[k]) if p[k]>=0 else None for k in PRIMARY}}
    if any(v is not None and not math.isfinite(v) for v in result.values()):raise ValueError('nonfinite score')
    return result

def render(rows):
    if set(rows)!=set(ARMS):raise ValueError('missing original-N or contemporaneous control')
    for a in ARMS:
        if set(rows[a])!={'600','300'}:raise ValueError('two budgets required')
    differences={};table=['|budget|arm|'+ '|'.join('COCO_'+k for k in COCO)+'|'+'|'.join('primary_'+k for k in PRIMARY)+'|',
                       '|---|---|'+'---:|'*(len(COCO)+len(PRIMARY))]
    for budget in ('600','300'):
        for a in ARMS:table.append('|'+budget+'|'+a+'|'+'|'.join('UNKNOWN' if v is None else '%.8f'%v for v in rows[a][budget].values())+'|')
    for label,left,right in [('S-N24','S24','N24'),('S-A24','S24','A24'),('S-originalN','S24','N_original'),('N24-originalN','N24','N_original'),('A24-N24','A24','N24')]:
        differences[label]={}
        for budget in ('600','300'):
            differences[label][budget]={k:None if rows[left][budget][k] is None or rows[right][budget][k] is None else rows[left][budget][k]-rows[right][budget][k] for k in rows[left][budget]}
    checks={}
    thresholds={'600':{'COCO_AP75_small':.30,'COCO_APS':.20,'COCO_AP':0,'COCO_APM':-.20,'COCO_APL':-.20,'COCO_AR_large':-.20,'primary_AP':-.20},
                '300':{'COCO_AP75_small':0,'COCO_AP':-.20,'COCO_APS':-.20,'primary_AP':-.20}}
    for comparison in ('S-N24','S-A24'):
        checks[comparison]={budget:{k:None if differences[comparison][budget][k] is None else differences[comparison][budget][k]>=threshold for k,threshold in limits.items()} for budget,limits in thresholds.items()}
    controlled=all(v is True for pair in checks.values() for budget in pair.values() for v in budget.values())
    original_changes=[differences['S-originalN'][b]['COCO_'+k] for b in ('600','300') for k in ('AP','APS','AP75_small')]
    below_original=any(v is not None and v<0 for v in original_changes)
    interpretation='相对控制改善，尚未形成现有基线净收益' if controlled and below_original else ('同期质量条件通过；原N净收益须逐项报告' if controlled else '同期质量条件未通过或存在UNKNOWN')
    return {'absolute_percent':rows,'differences_pp':differences,'original_quality_gates':checks,'quality_controls_pass':controlled,'interpretation':interpretation,'new_gate_added':False,'cost':'UNKNOWN: attach existing measured cost verdict separately'},'\n'.join(table)+'\n'

def main(args):
    source=json.loads(Path(args.inputs).read_text(encoding="utf-8"));expected='SYNTHETIC_NOT_SCIENTIFIC' if args.synthetic else 'FORMAL_EXISTING_SCORES'
    if source['scope']!=expected:raise ValueError('synthetic/formal scope')
    if source['N_checkpoint_sha256']!='be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83':raise ValueError('original N checkpoint identity')
    rows={}
    for a,entries in source['scores'].items():
        rows[a]={}
        for budget,ref in entries.items():
            raw=Path(ref['path']).read_bytes()
            if not args.synthetic and a=='N_original' and ref['sha256']!={'600':'86be4fcc1478442bd4bb6f3b6e5b0c421e29f854e6977c54ba906814e79dfc7e','300':'5a829d1ed3c68022053605ab5f143d3dd988eb277960918b898ffcaacd7e3129'}[budget]:raise ValueError('uncertified original-N score')
            if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('score SHA')
            score=json.loads(raw)
            if not args.synthetic and (score['official']['evaluated_image_count']!=548 or score['COCO_bridge']['maxDets']!=[1,10,100]):raise ValueError('evaluation protocol conflict')
            rows[a][budget]=metrics(score)
    result,table=render(rows);result['scope']=expected
    out=Path(args.out);out.mkdir();(out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'TABLE.md').write_text(table)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--synthetic',action='store_true');main(p.parse_args())
