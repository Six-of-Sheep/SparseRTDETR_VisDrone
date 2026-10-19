"""CPU-only production authoring after the FINAL extension entry gate passes."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import supplemental_runtime as rt
from supplemental_runtime import exact, read, reference, require, publish, digest, ROOT


def main():
    p=argparse.ArgumentParser()
    for name in ('source-manifest','cpu-gate','data-context','parent-contract','parent-sha256','output','job-root','runtime-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();rt.install()
    import torch,torchvision
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity as sid
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.data_protocol.supplemental_zero_gt import specification, VERSION
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU authoring')
    torch.set_num_threads(1);start=time.monotonic()
    out=Path(a.output);out.mkdir(exist_ok=False)
    src=reference(a.source_manifest);source=exact(src)
    require(all(reference(ROOT/x['path'])==dict(path=str(ROOT/x['path']),bytes=x['bytes'],sha256=x['sha256']) for x in source['files']),'final source byte mismatch')
    gate_ref=reference(a.cpu_gate);gate=exact(gate_ref)
    require(gate['status']=='PASS' and gate['entry']=='supplemental_job.py' and gate['source_sha256']==src['sha256'],'final CPU gate/source')
    require(gate['real_restore_pre_CUDA_seed_paths']==[0,1,2] and gate['actual_partial_batches']==[4,2],'CPU entry coverage')
    extension_proof=exact(gate['real_input_extension_proof'])
    require(extension_proof['status']=='PASS' and extension_proof['real1602_proof']['members']==1602 and extension_proof['real1602_proof']['admitted_zero_rows']==2,'real1602 CPU evidence')
    parent_ref=reference(a.parent_contract);require(parent_ref['sha256']==a.parent_sha256,'certified parent metadata')
    parent=exact(parent_ref);context=read(a.data_context)
    ext_ref=reference(ROOT/'configs/supplemental_zero_gt_extension_v1.json');require(exact(ext_ref)==specification(),'protocol specification')
    require(context['evaluation_input_extension']==ext_ref,'reader/context protocol differs')
    authority=exact(context['authorization'])
    require(authority['evaluation_input_protocol']==VERSION and authority['manifest_sha256']==rt.SEALED_SHA and authority['authorized'] is True,'direct new authority')
    parents=parent['parents'];proof=[]
    for item in parents:
        for r in (item['original_contract'],item['endpoint'],item['binding'],item['checkpoint_sidecar']):exact(r)
        b=exact(item['binding']);cp=item['checkpoint'];state=checkpoint.read(cp['path'],cp['sha256'],b)
        require(b['config']==Config(seed=item['seed']).binding() and b['scope']==f"formal_R18_seed{item['seed']}_e120",'own original seed binding')
        require(state['engine']['completed_updates']==state['ema']['updates']==36480 and state['engine']['phase']==0 and set(('model','optimizer','ema','engine','rng')).issubset(state),'original e120 complete state')
        proof.append(dict(seed=item['seed'],checkpoint=cp,binding=item['binding'],full_SHA_and_state_verified=True));del state
    actual=dict(torch=str(torch.__version__),torchvision=torchvision.__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    require(actual==parent['environment'],'environment changed')
    job,runtime=Path(a.job_root),Path(a.runtime_root)
    require(not job.exists() and not runtime.exists(),'new production paths must be absent')
    sample=guard.snapshot(out)
    guard.screen(sample,gpu=True,gpu_uuid=parent['gpu_uuid'],entry=True,disk_bytes=40*1024**3)
    require(not sample['compute_apps'],'other GPU process present')
    token=out/'LAUNCH_TOKEN.json'
    publish(token,dict(task_id=rt.TASK,purpose=rt.PURPOSE,source_sha256=src['sha256'],job_root=str(job),evaluation_input_protocol=VERSION,single_use=True))
    contract=out/'execution-contract.json'
    c=dict(task_id=rt.TASK,purpose=rt.PURPOSE,source_root=str(ROOT),source_reference=src,source_sha256=src['sha256'],source_commit=source['commit'],
           plan=reference(ROOT/'configs/sahi_supplemental_evaluation.json'),parent_metadata=parent_ref,parents=parents,seeds=[0,1,2],
           synthetic=False,GPU_authorized=True,science_authorized=True,sealed_access_authorized=True,authorization=context['authorization'],CPU_gate=gate_ref,
           launch_token=reference(token),training_updates=0,forward_budget=6423,python=sys.executable,boot_id=sample['boot_id'],
           gpu_uuid=parent['gpu_uuid'],storage_identity=sid(sample['storage']),environment=actual,job_root=str(job),runtime_root=str(runtime),contract_path=str(contract),
           evaluation_data=context['evaluation_data'],evaluation_input_extension=ext_ref,total_wallclock_cap=None,
           sealed_original_metadata_unchanged=True,independent_qualification='BLOCKED_PENDING_ACCESS_HISTORY')
    c['contract_sha256']=digest(c);publish(contract,c)
    require(rt.load(str(contract))==c and not torch.cuda.is_initialized() and not job.exists() and not runtime.exists(),'hidden CUDA contract check')
    publish(out/'PRELAUNCH.json',dict(status='PASS',contract=reference(contract),source=src,CPU_gate=gate_ref,parents=proof,actual_environment=actual,
                                    resources=sample,evaluation_input_extension=ext_ref,CUDA_initialized=False,elapsed_seconds=time.monotonic()-start))
    print(__import__('json').dumps(dict(status='PRELAUNCH_PASS',contract=reference(contract),elapsed_seconds=time.monotonic()-start)))


if __name__=='__main__':main()
