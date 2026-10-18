"""Drive final continuation CLI through real supervisor/evaluators/export/exit."""
import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import time
import sixth_view_runtime as rt
from sixth_view_runtime import ROOT,read,require,reference,publish,digest


def wait_exit(pid):
    try:
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
        if fd<0:
            code=ctypes.get_errno()
            if code==3:return
            raise OSError(code,os.strerror(code))
        poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    except ProcessLookupError:pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();c=read(a.template);ref=reference(a.manifest)
    c.update(source_root=str(ROOT),source_manifest=a.manifest,source_reference=ref,source_sha256=ref['sha256'],source_commit=read(a.manifest)['commit'],
        synthetic=True,GPU_authorized=False,science_authorized=False,entry_kind='seed0_cpu_then_frozen_conditional_seed12',no_seed0_GPU=True)
    publish(out/'FIXTURE_ONLY_GATE.json',{'status':'PASS','source_manifest':ref,'synthetic_fixture_identity_only':True})
    c['CPU_gate']=reference(out/'FIXTURE_ONLY_GATE.json');records=[]
    from sixth_view_job import environment,verify_phase
    for fail in (False,True):
        name='worker-error' if fail else 'success';d=copy.deepcopy(c);root=out/name;root.mkdir()
        d.update(job_root=str(root),runtime_root=str(Path(c['runtime_root']).parent/out.name/name),inject_worker_failure=fail,
            contract_path=str(out/(name+'-contract.json')))
        d['contract_sha256']=digest({k:v for k,v in d.items() if k!='contract_sha256'});publish(d['contract_path'],d)
        argv=[d['python'],'-B',str(ROOT/'sixth_view_score.py'),'start','--contract',d['contract_path']]
        p=subprocess.run(argv,env=environment(d),capture_output=True);require(p.returncode==0,'final scoring start failed: '+p.stderr.decode())
        launch=read(root/'LAUNCH_RECEIPT.json');wait_exit(launch['pid'])
        require((root/('STOP.json' if fail else 'CLOSED.json')).is_file(),'continuation terminal absent')
        if not fail:
            result=verify_phase(d,0,'evaluation');require(result['new_dual_evaluator_calls']==4,'four actual dual evaluator calls')
            require(read(root/'RESULT.json')['seed0_forward_calls']==0,'CPU forward scope')
        else:require('supervised phase failed' in read(root/'STOP.json')['error'],'worker-error refusal')
        require(read(root/'SLIM_EXPORT.json')['status']=='READY_FOR_SLIM_RETURN','actual export absent')
        before={str(f):reference(f) for f in root.rglob('*') if f.is_file()}
        duplicate=subprocess.run(argv,env=environment(d),capture_output=True)
        require(duplicate.returncode!=0 and before=={str(f):reference(f) for f in root.rglob('*') if f.is_file()},'duplicate start altered evidence')
        # Direct run replay also rejected by immutable CONSUMED, before scoring.
        dup_run=subprocess.run([d['python'],'-B',str(ROOT/'sixth_view_score.py'),'run','--contract',d['contract_path']],env=environment(d),capture_output=True)
        require(dup_run.returncode!=0 and before=={str(f):reference(f) for f in root.rglob('*') if f.is_file()},'duplicate run altered evidence')
        records.append({'case':name,'argv':argv,'chain_pid':launch['pid'],'terminal':reference(root/('STOP.json' if fail else 'CLOSED.json')),'duplicate_start_and_run_rejected':True})
    publish(out/'CPU_GATE.json',{'status':'PASS','source_manifest':ref,'records':records,'synthetic_only':True,
        'real_entry':'sixth_view_score start -> run -> original guard.supervise -> real worker -> four dual evaluators -> RESULT/CLOSED -> export -> exit',
        'does_not_replace_real_548_inputs':True,'CUDA_initialized':False,'forward_calls':0,'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
