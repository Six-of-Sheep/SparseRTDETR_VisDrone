"""One supervisor/one worker, exclusive output, no restart or continuation."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from r18_support_diagnostic import load, require, publish, reference, exact


def bounds(elapsed, size, forwards, allocated=0):
    require(0 <= elapsed < 600, '600s GPU stage cap')
    require(size <= 20*1024**2, '20MiB output cap')
    require(forwards <=16, '16 forward cap')
    require(allocated <=18*1024**3, '18GiB allocated cap')


def output_bytes(out):
    return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())


def resource(c,old,entry,baseline=None):
    sys.path.insert(0,str(Path(c['frozen_release'])/'src'))
    from sparse_rtdetr.protocol_r18.span_runtime import snapshot
    check=copy.deepcopy(old)
    check.update(job_root=c['output'],attempt=c['id'],task_id=c['id'])
    sample=snapshot(check,entry=entry,gpu=True,baseline=baseline)
    if entry:
        require(not sample['compute_apps'],'GPU has compute processes')
    return sample


def run(c,old):
    out=Path(c['output'])
    require(not out.exists(),'output already exists: no retry')
    baseline=resource(c,old,True)
    out.mkdir(parents=False,exist_ok=False)
    publish(out/'CONSUMED.json',dict(id=c['id'],time=time.time(),no_retry=True,contract=c,
        parent_environment=old['environment'],entry_resource=baseline,supervisor_pid=os.getpid()))
    started=time.monotonic();child=None;error=None;rc=None;samples=0;last=baseline
    try:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=c['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        argv=[old['python'],'-B',str(Path(__file__).with_name('r18_support_diagnostic.py')),
              '--mode','worker','--contract',c['contract_path']]
        publish(out/'ARGV.json',dict(argv=argv,env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','CUDA_DEVICE_ORDER')}))
        with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            while child.poll() is None:
                forwards=len(list(out.glob('FORWARD-*.json')))
                bounds(time.monotonic()-started,output_bytes(out),forwards)
                last=resource(c,old,False,baseline)
                allowed={child.pid}
                require(all(int(v['pid']) in allowed for v in last['compute_apps']), 'foreign GPU compute PID')
                samples+=1
                time.sleep(2)
            rc=child.returncode
        require(rc==0 and (out/'WORKER_RESULT.json').is_file(),'worker did not PASS')
        require(not (out/'WORKER_STOP.json').exists(),'worker STOP')
        bounds(time.monotonic()-started,output_bytes(out),len(list(out.glob('FORWARD-*.json'))))
        r=json.loads((out/'WORKER_RESULT.json').read_text())
        require(r['status']=='PASS' and r['forward_calls']==16 and r['images_per_arm']==32,'worker incomplete')
        exact(r['result']);exact(r['inventory'])
        publish(out/'CLOSED.json',dict(status='PASS',worker=reference(out/'WORKER_RESULT.json')))
    except BaseException:
        error=traceback.format_exc()
        if child is not None and child.poll() is None:
            # Necessary fail-stop of OUR own isolated child; never a foreign job.
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        rc=None if child is None else child.returncode
        publish(out/'STOP.json',dict(status='STOP_NO_RETRY',error=error,worker_returncode=rc))
    finally:
        publish(out/'EXIT.json',dict(status='PASS' if error is None else 'STOP_NO_RETRY',error=error,
            worker_pid=None if child is None else child.pid,returncode=rc,forward_reserved=len(list(out.glob('FORWARD-*.json'))),
            GPU_stage_seconds=time.monotonic()-started,resource_samples=samples,last_resource=last,
            output_bytes=output_bytes(out),remaining_live=[] if child is None or child.poll() is not None else [child.pid]))
    return 0 if error is None else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',required=True)
    p.add_argument('--mode',required=True,choices=['preflight','visible-precheck','run'])
    a=p.parse_args();c,old,_=load(a.contract)
    require(not Path(c['output']).exists(),'attempt output already exists')
    if a.mode=='preflight':
        sample=resource(c,old,True)
        print(json.dumps(dict(status='PASS',source_files=len(exact(c['frozen_source'])['files']),
                              selected_images=32,resource=sample)))
    elif a.mode=='visible-precheck':
        sys.path.insert(0,str(Path(c['frozen_release'])/'src'))
        import torch
        from sparse_rtdetr.protocol_r18.cuda_identity import inspect_cuda_identity
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'CUDA availability')
        identity=inspect_cuda_identity(c['gpu_uuid'],Path(c['precheck_output']),torch_module=torch)
        print(json.dumps(dict(status='PASS',identity=identity)))
    else:
        return run(c,old)
    return 0


if __name__=='__main__':
    sys.exit(main())
