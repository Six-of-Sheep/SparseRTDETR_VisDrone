"""Read three existing raw batches only; no model, pixels, scoring or launcher."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

PARENT_SHA='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28'

def run(parent,out):
    started=time.monotonic();raw=parent.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('certified parent contract SHA differs')
    c=json.loads(raw);source=Path(c['source_root'])
    for row in c['source_files']:
        b=(source/row['path']).read_bytes()
        if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('parent source differs')
    sys.path.insert(0,str(source));import softnms_job as old
    c=old.load_contract(str(parent));torch=old.cpu_only();before=torch.get_rng_state().clone()
    specs,rows,_=old.context(c)
    from softnms_reuse import validate_reuse
    certified=validate_reuse(c)
    old.require(old.digest(specs)==certified['context_sha256'] and old.digest(rows)==certified['rows_sha256'],'authenticated context differs')
    own=Path(__file__).resolve().parent
    old.require(old.sha(own/'slice_core.py')==old.sha(source/'slice_core.py'),'original NMS source differs')
    sys.path.insert(0,str(own));from sixth_view_core import merge_six
    checks=[]
    for s in specs:
        ref=s['full']['raw_view_refs'][0]
        raw=torch.load(io.BytesIO(old.exact(ref,contents=True)),map_location='cpu',weights_only=True)
        pools=old.decode_raw(raw,rows[:4],s['identity'],s['seed'])
        for row,pool in zip(rows[:4],pools):
            original=old.merge_candidates(pool);actual=merge_six(pool)
            old.require(original==actual,'disabled original C changed')
            for budget in (600,300):
                old.require([old.prediction(row['stable_image_id'],p) for p in original[f'top{budget}']]==
                            [old.prediction(row['stable_image_id'],p) for p in actual[f'top{budget}']], 'serialized C changed')
        checks.append({'seed':s['seed'],'raw_ref':ref,'images':[r['stable_image_id'] for r in rows[:4]],
                       'identity':s['identity'],'disabled_C_two_budgets_exact':True})
    old.require(not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state()),'CPU RNG/CUDA changed')
    old.publish(out,{'status':'PASS','actual_raw_batches_read':3,'images':12,'checks':checks,
          'all_411_prior_rebuilds':'certification reused; not rerun','parent_reuse_certificate':c['reuse_certificate'],
          'scope':'disabled path equality on each seed first four images; no new variant prediction',
          'GPU_calls':0,'model_constructions':0,'science_scores':0,'elapsed_seconds':time.monotonic()-started,
          'own_core_sha256':old.sha(own/'sixth_view_core.py'),'old_NMS_sha256':old.sha(source/'slice_core.py')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only requires hidden CUDA')
    run(a.parent,a.output)
