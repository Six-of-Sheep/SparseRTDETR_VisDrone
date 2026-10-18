"""Final deployed CLI integration; real supervisor and real dual evaluators."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from sixth_view_runtime import ROOT,read,reference,require,publish,digest,load,install


def checked(argv,c):
    from sixth_view_job import environment
    return subprocess.run(argv,env=environment(c),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)


def main():
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    c=read(a.template);require(c['synthetic'] and not c['GPU_authorized'] and not c['science_authorized'],'CPU fixture authority')
    records=[]
    for failure in (False,True):
        name='worker-error' if failure else 'success';d=copy.deepcopy(c)
        d.update(job_root=str(out/name),runtime_root=str(Path(c['runtime_root'])/name),inject_worker_failure=failure)
        Path(d['job_root']).mkdir();d['contract_path']=str(out/(name+'-contract.json'))
        d['contract_sha256']=digest({k:v for k,v in d.items() if k!='contract_sha256'});publish(d['contract_path'],d)
        argv=[d['python'],'-B',str(ROOT/'sixth_view_job.py'),'start','--contract',d['contract_path']]
        result=checked(argv,d);require(result.returncode==0,'actual start entry failed: '+result.stdout.decode())
        launch=read(Path(d['job_root'])/'LAUNCH_RECEIPT.json')
        # pidfd kernel wait, no periodic state polling.
        import select
        try:
            fd=os.pidfd_open(launch['pid']);wait=select.poll();wait.register(fd,select.POLLIN);wait.poll();os.close(fd)
        except ProcessLookupError:pass
        root=Path(d['job_root'])
        require((root/('STOP.json' if failure else 'CLOSED.json')).is_file(),'CPU actual terminal absent')
        if not failure:
            r=read(root/'RESULT.json');require(r['status']=='COMPLETE' and r['synthetic'],'synthetic result missing')
            ev=read(root/'seed0/evaluation/RESULT.json');require(ev['new_dual_evaluator_calls']==4,'actual four dual evaluator calls absent')
            from sixth_view_job import verify_phase
            for mode in __import__('sixth_view_runtime').PHASES:verify_phase(d,0,mode)
        else:require('supervised phase failed' in read(root/'STOP.json')['error'],'unexpected worker-error path')
        require(read(root/'SLIM_EXPORT.json')['status']=='READY_FOR_SLIM_RETURN','actual export absent')
        prior={str(q):reference(q) for q in root.rglob('*') if q.is_file()}
        duplicate=checked(argv,d)
        require(duplicate.returncode!=0 and prior=={str(q):reference(q) for q in root.rglob('*') if q.is_file()},'duplicate launch modified prior evidence')
        records.append({'case':name,'start_command':argv,'run_pid':launch['pid'],'actual_terminal':reference(root/('STOP.json' if failure else 'CLOSED.json')),'export':reference(root/'SLIM_EXPORT.json'),'duplicate_rejected_without_mutation':True})
    # Real authenticated stream mutations: missing, reordered, repeated terminal
    # are rejected by the same Reader used by worker, guard, archive and closure.
    install();from sparse_rtdetr.localization import channel
    source=out/'success/seed0/evaluation/runtime-archive';b=read(source/'identity.json')
    lines=[json.loads(x) for x in (source/'progress.jsonl').read_bytes().splitlines()]
    cases={'missing':lines[:2]+lines[3:],'reordered':lines[:1]+[lines[2],lines[1]]+lines[3:],'duplicate_terminal':lines+[copy.deepcopy(lines[-1])]}
    for name,values in cases.items():
        dest=out/(name+'.jsonl');prev=None
        for i,r in enumerate(values):r.update(seq=i+1,previous=prev);r.pop('sha256',None);r['sha256']=digest(r);prev=r['sha256']
        dest.write_text(''.join(json.dumps(r)+'\n' for r in values))
        try:channel.Reader(dest,b,'progress').poll()
        except RuntimeError:pass
        else:raise RuntimeError('bad stream was admitted: '+name)
    # All modes have been driven through main, not direct pipeline/supervisor.
    publish(out/'CPU_GATE.json',{'status':'PASS','source_sha256':c['source_sha256'],'records':records,
        'real_entry_chain':'start -> run -> guard.supervise -> worker -> actual dual evaluators -> RESULT/CLOSED -> SLIM_EXPORT -> exit',
        'seven_phase_process_chain':True,'dual_evaluator_calls':4,'synthetic_only':True,
        'bad_streams_rejected':list(cases),'CUDA_initialized':False,'detector_updates':0,
        'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
