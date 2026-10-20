"""Independent subprocess acceptance of final preparation CLI; no PYTHONPATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    runs=[]
    def run(name,out,extra=(),fault=False):
        cmd=[sys.executable,'-B',str(ROOT/'r18_distill_prepare.py'),'cpu-check','--output',str(out),*extra]
        actual_env=dict(env)
        if fault:actual_env['R18_DISTILL_CPU_INJECT_FAILURE']='1'
        start=time.monotonic()
        with (args.output/(name+'.log')).open('x',encoding='utf-8') as log:
            p=subprocess.run(cmd,cwd=ROOT,env=actual_env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        runs.append(dict(name=name,command=cmd,returncode=p.returncode,seconds=time.monotonic()-start,cwd=str(ROOT),extra_PYTHONPATH=False))
        return p.returncode
    normal=args.output/'normal'
    assert run('normal',normal)==0
    proof=json.loads((normal/'RESULT.json').read_text(encoding='utf-8'))
    assert proof['CUDA_initialized'] is False and proof['metadata']['fixture']['CPU_synthetic_updates_total']==9
    before={str(p.relative_to(normal)):hashlib.sha256(p.read_bytes()).hexdigest() for p in normal.rglob('*') if p.is_file()}
    assert run('duplicate',normal)!=0
    after={str(p.relative_to(normal)):hashlib.sha256(p.read_bytes()).hexdigest() for p in normal.rglob('*') if p.is_file()}
    assert before==after
    bad_plan=args.output/'wrong-plan.json'
    plan=json.loads((ROOT/'configs/research/r18_multiview_distill_v1.json').read_text());plan['seed']=1
    bad_plan.write_text(json.dumps(plan),encoding='utf-8')
    assert run('bad-plan',args.output/'bad-plan',('--plan',str(bad_plan)))!=0
    assert (args.output/'bad-plan/STOP.json').is_file()
    assert run('worker-exception',args.output/'worker-error',fault=True)!=0
    assert (args.output/'worker-error/STOP.json').is_file()
    assert not (args.output/'worker-error/CLOSED.json').exists()
    value=dict(status='CPU_ENTRY_PASS',runs=runs,actual_CPU_synthetic_updates=9,
               actual_teacher_synthetic_forward_views=5,CUDA_initialized=False,
               worker_error_stopped=True,duplicate_unchanged=True,bad_plan_refused=True,
               GPU_authorized=False,scientific_updates=0,science_scoring_groups=0,
               production_GPU_entry='NOT_IMPLEMENTED_NOT_AUTHORIZED',no_skips=True)
    (args.output/'GATE.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
