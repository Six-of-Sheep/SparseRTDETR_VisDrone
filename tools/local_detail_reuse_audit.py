"""Local-only B0/N provenance audit. Reads receipts/source, never tensors or images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def audit(delivery):
    from sparse_rtdetr.protocol_r18.model import Config
    base=Path(delivery);receipts=base/'receipts'
    index=json.loads((base/'SLIM_RETURN_VERIFIED.json').read_text(encoding='utf-8'))
    entries={x['relative']:x for x in index['files']}
    needed=['formal-execution-003/COMPLETED_B0_AUTHENTICATED.json',
        'formal-execution-001/contract.json','resource-age-repair-bundle-001/RUN_SOURCE_MANIFEST.json',
        'attempt-formal-001/train-B0/binding.json','attempt-formal-001/train-B0/endpoint.json',
        'attempt-formal-001/train-B0/RESULT.json','attempt-formal-001/train-B0/worker-exit.json',
        'attempt-formal-001/STOP.json','attempt-formal-001/admission/initialization.json',
        'attempt-formal-003/score/B0-600.json','attempt-formal-003/score/B0-300.json',
        'attempt-formal-003/infer-B0/RESULT.json','attempt-formal-003/score/CLOSED.json']
    checked=[];values={}
    for name in needed:
        raw=(receipts/name).read_bytes(); row=entries[name]
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('local receipt differs: '+name)
        checked.append(dict(relative=name,bytes=len(raw),sha256=row['sha256']))
        values[name]=json.loads(raw)
    auth=values[needed[0]];binding=values[needed[3]];endpoint=values[needed[4]]
    if auth['status']!='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE' or auth['old_closure_created']:
        raise ValueError('old closure attribution differs')
    if binding['config']!=Config().binding():raise ValueError('B0/N common config differs')
    if endpoint['updates']!=36480 or endpoint['epoch']!=120 or endpoint['EMA'] is not True or endpoint['used_acceptance_state'] is not False:
        raise ValueError('not the fixed fresh-training e120 EMA')
    if endpoint['checkpoint']['sha256']!='be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83':
        raise ValueError('unexpected B0 endpoint')
    source=values[needed[2]]['files']; rows=[];excluded=[]
    for row in source:
        p=row['path']
        selected=p.startswith(('vendor/','src/sparse_rtdetr/baseline/training_v2b',
                               'src/sparse_rtdetr/protocol_r18/','src/sparse_rtdetr/r0/'))
        if not selected or '/span_' in p:continue
        if p.endswith('/cuda_identity.py'):
            excluded.append(dict(path=p,reason='admission identity helper; not model/data/loss/update arithmetic'))
            continue
        raw=(ROOT/p).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('N scientific source differs: '+p)
        rows.append(row)
    data=values[needed[1]]['data']
    if data['train_core']['count']!=4869 or data['development']['count']!=548:raise ValueError('data counts differ')
    return dict(schema='H2-B0-as-N-audit-v1',status='PASS_LOCAL_REFERENCE_AUDIT',
        decision='REUSE_B0_AS_HISTORICAL_N_ENDPOINT_NO_N_RETRAINING',
        not_same_round_N_training=True,primary_routing_comparison='D-U',
        source_files_equal=rows,excluded_source_checks=excluded,receipts_reread=checked,
        config=Config().binding(),data=data,checkpoint_reference=endpoint['checkpoint'],
        initial_model_sha256=endpoint['initial_model_sha256'],old_stage_status=auth['status'],
        old_STOP_preserved=True,old_closure_created=False,server_connected=False,
        physical_checkpoint_rehashed_now=False,physical_images_or_weights_read=False,
        conditions_before_GPU_use=['fresh byte/SHA and old binding verification of original EMA on server',
            'same data members, preprocessing, evaluator, budgets and raw-ignore identity',
            'new same-session N/U/D cost measurement; old timing cannot substitute',
            'reuse only final N inference; never old optimizer/RNG or acceptance state for U/D training'],
        limitations=['D-N is a matched-protocol historical comparison, not a contemporaneous retraining comparison',
            'single seed and repeatedly used development; no independent generalization',
            'missing N coarse-center mechanism field stays UNKNOWN; no extra N endpoint inference'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.delivery)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],decision=result['decision'],source_files=len(result['source_files_equal']),
                         receipts=len(result['receipts_reread']))))


if __name__=='__main__':main()
