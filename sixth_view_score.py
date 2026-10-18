"""New single-use continuation entry: seed0 CPU evaluation, conditional seed1/2."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import sixth_view_runtime as rt
from sixth_view_runtime import ROOT,read,require,reference,publish,digest


def cpu_only():
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'seed0 CPU continuation CUDA boundary')
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs):raise RuntimeError('CPU continuation forbids CUDA/model/backward')
    torch.cuda._lazy_init=forbidden;torch.nn.Module.__init__=forbidden;torch.autograd.backward=forbidden


def build(a):
    cpu_only()
    from sixth_view_score_reuse import exact
    certref=reference(a.certificate);cert=exact(certref);old=exact(cert['parent_contract'])
    require(cert['status']=='PASS' and cert['source_manifest']==reference(a.manifest),'real reuse CPU gate source differs')
    gate=exact(reference(a.cpu_gate));require(gate['status']=='PASS' and gate['source_manifest']==reference(a.manifest),'final scoring entry CPU gate source differs')
    c=copy.deepcopy(old);job=Path(a.job);job.mkdir(parents=True,exist_ok=False)
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity
    rt.install();sample=guard.snapshot(job);guard.screen(sample,gpu=False,entry=True,disk_bytes=40*1024**3)
    c.update(source_root=str(ROOT),source_manifest=a.manifest,source_reference=reference(a.manifest),source_sha256=reference(a.manifest)['sha256'],
        source_commit=read(a.manifest)['commit'],job_root=str(job),runtime_root=a.runtime,contract_path=a.output,
        synthetic=False,CPU_gate=reference(a.cpu_gate),seed0_evaluation_reuse=certref,
        entry_kind='seed0_cpu_then_frozen_conditional_seed12',boot_id=sample['boot_id'],storage_identity=identity(sample['storage']),
        no_seed0_GPU=True,total_wallclock_cap=None)
    c['contract_sha256']=digest({k:v for k,v in c.items() if k!='contract_sha256'});publish(a.output,c)
    publish(job/'PREPARATION.json',{'status':'PASS','contract':reference(a.output),'reuse':certref,'CPU_gate':reference(a.cpu_gate),'resource':sample,'CUDA_initialized':False})
    print(json.dumps({'status':'PASS','contract':reference(a.output)}))


def run(c):
    from sixth_view_job import phase,export
    root=Path(c['job_root']);publish(root/'CONSUMED.json',{'contract_sha256':c['contract_sha256'],'pid':os.getpid(),'epoch':time.time(),'no_retry':True})
    start=time.monotonic();completed=[]
    try:
        gate=read(c['CPU_gate']['path']);require(reference(c['CPU_gate']['path'])==c['CPU_gate'] and gate['status']=='PASS' and gate['source_manifest']==c['source_reference'],'CPU entry gate binding')
        require(c['entry_kind']=='seed0_cpu_then_frozen_conditional_seed12' and c['no_seed0_GPU'] is True,'continuation scope')
        if not c['synthetic']:
            cpu_only()
            from sixth_view_score_reuse import exact
            cert=exact(c['seed0_evaluation_reuse']);cost=exact(cert['cost'])
            require(cost['status']=='PASS' and all(cost['ratios_over_A'][v]<=6 for v in ('center','flip')),'reused cost gate')
            publish(root/'seed0-COST-REUSE.json',{'parent_cost':cert['cost'],'cost':cost,'original_binding_preserved':True})
        phase(c,0,'evaluation')
        decision=read(root/'seed0/evaluation/COMPARISON.json')
        permits=False if c['synthetic'] else decision['quality_permits_next_seeds']
        completed.append({'seed':0,'cost_passed':True,'quality':decision,'evaluation':reference(root/'seed0/evaluation/RESULT.json'),'GPU_forward_calls':0})
        if permits:
            for seed in (1,2):
                # Each downstream worker is independent; same frozen schedule.
                results={mode:phase(c,seed,mode) for mode in rt.PHASES[:4]}
                costs={v:results['accept-'+v]['e2e_seconds']/results['accept-A']['e2e_seconds'] for v in ('C','center','flip')}
                passed=all(costs[v]<=6 for v in ('center','flip'))
                publish(root/f'seed{seed}/COST.json',{'status':'PASS' if passed else 'FAIL_CLOSED','ratios_over_A':costs,'limit':6.,'all_real_views_included':True})
                require(passed,'new seed resource/cost admission failed; STOP')
                for mode in rt.PHASES[4:]:phase(c,seed,mode)
                d=read(root/f'seed{seed}/evaluation/COMPARISON.json')
                completed.append({'seed':seed,'cost_passed':True,'quality':d,'evaluation':reference(root/f'seed{seed}/evaluation/RESULT.json')})
        success=len(completed)==3 and all(x['quality']['quality_permits_next_seeds'] for x in completed)
        publish(root/'RESULT.json',{'status':'COMPLETE','completed':completed,'synthetic':c['synthetic'],
            'fixed_candidate_status':'SYNTHETIC_ONLY' if c['synthetic'] else ('PASS' if success else 'FAIL_CLOSED'),
            'seed0_forward_calls':0,'detector_updates':0,'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
        publish(root/'CLOSED.json',{'status':'PASS','result':reference(root/'RESULT.json')});export(c);return 0
    except BaseException as e:
        publish(root/'STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),'completed':completed,'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
        export(c);return 2


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','start','run']);p.add_argument('--contract');
    for name in ('certificate','manifest','cpu-gate','output','job','runtime'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='build':return build(a)
    c=rt.load(a.contract)
    if a.action=='run':return run(c)
    root=Path(c['job_root']);require(not any((root/n).exists() for n in ('LAUNCH.json','CONSUMED.json','STOP.json','CLOSED.json')),'continuation launch already used')
    argv=[c['python'],'-B',str(ROOT/'sixth_view_score.py'),'run','--contract',a.contract]
    publish(root/'LAUNCH.json',{'argv':argv,'contract':reference(a.contract),'epoch':time.time()})
    from sixth_view_job import environment
    with (root/'chain.log').open('xb') as log:
        child=subprocess.Popen(argv,env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    from sparse_rtdetr.localization.guard import identity
    publish(root/'LAUNCH_RECEIPT.json',{'pid':child.pid,'identity':identity(child.pid),'contract_sha256':c['contract_sha256'],'epoch':time.time()})
    print(json.dumps(read(root/'LAUNCH_RECEIPT.json')));return 0


if __name__=='__main__':raise SystemExit(main())
