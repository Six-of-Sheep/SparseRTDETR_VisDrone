"""Single-use CPU cache entry: admission -> exact rebuild -> six scores -> closure/export."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TASK='sahi-two-of-four-20261019-001'
PARENT_SHA='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_bytes(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def publish(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))

def require(v,s):
    if not v:raise ValueError(s)

def dependency(parent):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    os.environ['MKL_THREADING_LAYER']='GNU'
    require(sha(parent)==PARENT_SHA,'parent contract SHA')
    c=read(parent);root=Path(c['source_root'])
    for r in c['source_files']:
        p=Path(r['path'])
        require(not p.is_absolute() and '..' not in p.parts and sha(root/p)==r['sha256'],'dependency source SHA')
    require(sha(root/'slice_core.py')==sha(ROOT/'slice_core.py'),'merge implementation differs')
    sys.path.insert(0,str(root))
    import softnms_job as j
    torch=j.cpu_only();c=j.load_contract(parent)
    specs,rows,data=j.context(c)
    spec=next(s for s in specs if s['seed']==0)
    require(len(rows)==548 and len(spec['full']['raw_view_refs'])==137,'scope')
    parsed=j.parsed_gt(rows,data)
    return j,torch,spec,rows,parsed

def validate_plan(plan):
    from two_of_four_core import PAIRS,PROTECTION
    require(plan['task_id']==TASK and plan['seed']==0 and plan['images']==548 and plan['budget']==600,'plan scope')
    require(plan['pairs']==[list(p) for p in PAIRS] and plan['gain_retention']==.8
            and plan['protection_pp']==PROTECTION and plan['protection_references']==['C','A'],'screen rules')
    require(plan['forward_calls']==plan['GPU_calls']==plan['training_updates']==0,'forbidden work')

def verify_contract(path):
    c=read(path);m=read(ROOT/'RUN_SOURCE.json')
    require(c['task_id']==TASK and c['parent_sha256']==PARENT_SHA and c['commit']==m['commit'],'contract identity')
    require(c['own_files']==m['source_files'],'run source list differs')
    for r in m['source_files']:
        p=Path(r['path'])
        require(not p.is_absolute() and '..' not in p.parts and sha(ROOT/p)==r['sha256'],'own source SHA')
    require(sha(ROOT/'configs/sahi_two_of_four.json')==c['plan_sha256'],'plan SHA')
    gate=read(c['CPU_GATE']['path'])
    require(sha(c['CPU_GATE']['path'])==c['CPU_GATE']['sha256'] and gate['status']=='PASS'
            and gate['commit']==c['commit'] and gate['skips']==0,'CPU gate')
    return c

def score(rows,parsed,predictions,repo):
    from sparse_rtdetr.r0.evaluation import score_budget
    return score_budget(rows,parsed,predictions,repo)

def fixture(out,fail):
    # Same entry, core, score wrapper, real dual evaluators and terminal/export code.
    sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    from slice_core import Candidate
    from two_of_four_core import merge_pair
    from sparse_rtdetr.localization.data import parse_raw
    rows=[dict(stable_image_id='fixture'+str(i),width=100,height=100) for i in range(5)]
    parsed=[parse_raw(b'10,10,10,10,1,1,0,0\n',r['stable_image_id'],expected_rows=1) for r in rows]
    pool=[Candidate((10,10,20,20),.9,v,v,1,1) for v in range(5)]
    preds=[dict(image_id=r['stable_image_id'],category_id=p.category_id,score=p.score,bbox_xyxy=list(p.box_xyxy))
           for r in rows for p in merge_pair(pool,(1,2))]
    publish(out/'WORKER_STARTED.json',dict(pid=os.getpid(),fixture=True))
    if fail:raise RuntimeError('injected worker failure')
    record=score(rows,parsed,preds,ROOT)
    publish(out/'FIXTURE_METRICS.json',record)
    require(record['official']['evaluated_image_count']==5,'fixture real evaluator members')
    return dict(status='CPU_FIXTURE_COMPLETE',dual_evaluators=True,images=5)

def close(out,result):
    ref=publish(out/'RESULT.json',result)
    publish(out/'CLOSED.json',dict(result=ref,returncode=0))
    names=['RESULT.json','CLOSED.json','INPUT_PROOFS.json','REBUILD.json','SUMMARY.json','RESOURCE.json',
           'SEQUENCES.json','COST.json','ADMISSION.json','FIXTURE_METRICS.json','WORKER_STARTED.json']
    files=[dict(path=str(out/n),bytes=(out/n).stat().st_size,sha256=sha(out/n)) for n in names if (out/n).is_file()]
    require(sum(r['bytes'] for r in files)<8*1024**2,'slim size cap')
    publish(out/'SLIM_EXPORT.json',dict(status='READY',files=files,large_predictions='SERVER_ONLY'))

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('check','run','fixture'))
    p.add_argument('--parent',type=Path);p.add_argument('--contract',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inject-failure',action='store_true')
    a=p.parse_args(argv)
    if a.mode!='fixture' and (a.parent is None or a.inject_failure or TASK not in a.output.resolve().parts):p.error('production scope')
    if a.mode=='run' and a.contract is None:p.error('contract required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();publish(out/'START.json',dict(pid=os.getpid(),time=time.time(),mode=a.mode))
    try:
        plan=read(ROOT/'configs/sahi_two_of_four.json');validate_plan(plan)
        if a.mode=='fixture':result=fixture(out,a.inject_failure)
        elif a.mode=='check':
            j,torch,spec,rows,parsed=dependency(a.parent)
            resource=j.resources(out,plan)
            for r in spec['full']['raw_view_refs']:
                q=Path(r['path']);require(q.is_file() and q.stat().st_size==r.get('bytes',r.get('size_bytes')),'raw presence/bytes')
            result=dict(status='ADMISSION_PASS',source_files=len(read(a.parent)['source_files']),seed=0,
                        images=548,annotation_SHA_files=len(parsed),resource=resource,
                        raw_batches=137,raw_SHA_verification='production before decode',checkpoint=spec['identity']['checkpoint_sha256'])
        else:
            c=verify_contract(a.contract)
            # Exclusive token is consumed before any scientific dataflow; never replayed.
            token=Path(c['token']);require(read(token)['task_id']==TASK,'token task')
            publish(token.with_name(token.name+'.consumed'),dict(contract_sha256=sha(a.contract),output=str(out),pid=os.getpid()))
            from two_of_four_run import run
            result=run(a.parent,c,plan,out)
            result['execution']=dict(commit=c['commit'],contract_sha256=sha(a.contract),
                run_source_sha256=sha(ROOT/'RUN_SOURCE.json'),plan_sha256=c['plan_sha256'])
        result['elapsed_seconds']=time.monotonic()-start
        close(out,result)
        print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
        return 0
    except Exception:
        publish(out/'STOP.json',dict(traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,automatic_retry=False))
        traceback.print_exc();return 2

if __name__=='__main__':sys.exit(main())
