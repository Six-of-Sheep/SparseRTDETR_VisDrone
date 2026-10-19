"""Exercise the deployed preparation CLI with real allowed development inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import confirmation_prepare as p


def tree(path):
    return {str(f.relative_to(path)):p.reference(f) for f in path.rglob('*') if f.is_file()}


def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--input-bindings',required=True)
    a.add_argument('--legacy-contract',required=True);a.add_argument('--output',required=True);v=a.parse_args()
    p.cpu_only();base=Path(v.output);base.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    old=p.read(v.input_bindings);plan=p.read(p.ROOT/'configs/sahi_independent_confirmation.json')
    parents=[{k:s[k] for k in ('seed','identity','checkpoint_sha256')} for s in old['seeds']]
    p.require([s['checkpoint_sha256'] for s in parents]==plan['checkpoint_sha256'],'original parents differ from frozen plan')
    legacy=p.reference(v.legacy_contract)
    p.require(legacy['sha256']=='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28','historical verifier contract changed')
    c=dict(task_id=p.TASK,source_root=str(p.ROOT),source_manifest=p.reference(v.manifest),
        plan=p.reference(p.ROOT/'configs/sahi_independent_confirmation.json'),parents=parents,legacy_contract=legacy,
        scope='development_CPU_inputs_only',GPU_authorized=False,science_authorized=False,sealed_access_authorized=False)
    results=[]
    for failure in (False,True):
        name='worker-error' if failure else 'real-success';contract=base/(name+'-contract.json');d=copy.deepcopy(c)
        runtime=Path('/home/lyy/.cache')/p.TASK/base.name/name;runtime.parent.mkdir(parents=True,exist_ok=True)
        d.update(output=str(base/name),runtime=str(runtime),inject_worker_failure=failure)
        d['contract_sha256']=p.digest(d);p.publish(contract,d)
        cmd=[sys.executable,'-B',str(p.ROOT/'confirmation_prepare.py'),'start','--contract',str(contract)]
        with (base/(name+'.log')).open('xb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ))
        root=base/name
        p.require(r.returncode==(2 if failure else 0),'real CLI returned unexpected code: '+name)
        p.require((root/('STOP.json' if failure else 'CLOSED.json')).is_file(),'expected terminal absent')
        p.require(p.read(root/'SLIM_EXPORT.json')['status']=='READY_FOR_SLIM_RETURN' and p.read(root/'EXIT.json')['returncode']==r.returncode,'export/exit absent')
        if not failure:
            result=p.read(root/'RESULT.json');p.require(result['real_input_groups']==18 and result['scientific_evaluations']==0,'real input coverage')
            for seed in (0,1,2):
                s=p.read(root/f'seed{seed}/RESULT.json');p.require(s['seed']==seed and s['AP_calls']==s['forward_calls']==0,'real seed scope')
        before=tree(root)
        for action in ('start','run'):
            with (base/(name+'-duplicate-'+action+'.log')).open('xb') as f:
                again=subprocess.run([sys.executable,'-B',str(p.ROOT/'confirmation_prepare.py'),action,'--contract',str(contract)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ))
            p.require(again.returncode!=0 and tree(root)==before,'duplicate launch mutated attempt')
        results.append({'case':name,'returncode':r.returncode,'contract':p.reference(contract),'terminal':p.reference(root/('STOP.json' if failure else 'CLOSED.json')),'duplicate_start_and_run_rejected':True,'export':p.reference(root/'SLIM_EXPORT.json'),'exit':p.reference(root/'EXIT.json')})
    p.publish(base/'CPU_GATE.json',dict(status='PASS',source=p.reference(v.manifest),records=results,
        entry='confirmation_prepare start -> run -> original guard.supervise -> true-seed worker -> actual dual-evaluator INPUTS -> RESULT/CLOSED -> export -> exit',
        real_input_groups=18,AP_calls=0,model_constructions=0,forward_calls=0,training_updates=0,
        sealed_payload_opened=False,CUDA_initialized=False,elapsed_seconds=time.monotonic()-started,
        independent_data_qualification='BLOCKED_PENDING_ACCESS_HISTORY',does_not_certify_future_GPU_or_unsealing=True))
    print(json.dumps(p.read(base/'CPU_GATE.json')))


if __name__=='__main__':main()
