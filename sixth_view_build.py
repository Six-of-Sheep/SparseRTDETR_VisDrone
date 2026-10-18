"""CPU-only parent lineage and execution contract authoring, no launcher."""
import argparse
import sys
from pathlib import Path
import sixth_view_runtime as rt
from sixth_view_runtime import ROOT,read,publish,reference,require,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True);p.add_argument('--materials',required=True);p.add_argument('--output',required=True);p.add_argument('--job',required=True);p.add_argument('--runtime',required=True);p.add_argument('--cpu-gate');a=p.parse_args()
    import os,torch,torchvision
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'contract construction CPU only')
    torch.set_num_threads(1);rt.install()
    materials=Path(a.materials)
    plan_json=reference(materials/'closeout-001/PLAN.json');plan_md=reference(materials/'closeout-001/PLAN.md')
    parent=materials.parent/'sahi-softnms-fixed-20261017-001/revision-007/execution-contract.json'
    require(reference(parent)['sha256']=='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28','fixed parent contract SHA')
    input_ref=reference(materials/'INPUT_BINDINGS.json');require(input_ref['sha256']=='3f6be7afa4f0eb7c02fe1c2c9fd36c86edc562eab45fdb7cf6e3e2077690df6a','original input index differs')
    pc=read(parent);sys.path.insert(0,pc['source_root']);import softnms_job as old
    pc=old.load_contract(str(parent));ss,rows,parsed=old.context(pc)
    from softnms_reuse import validate_reuse
    cert=validate_reuse(pc);require(old.digest(ss)==cert['context_sha256'] and old.digest(rows)==cert['rows_sha256'],'certified cached evidence changed')
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.model import Config
    inputs=read(input_ref['path']);parents=[]
    for s,known in zip(ss,inputs['seeds']):
        seed=s['seed'];original=s['contract'];require(s['identity']==known['identity'],'authenticated parent identity differs')
        if seed==0:
            endref=original['endpoint'];binding=original['parent_binding'];construction=read(original['parent_contract']['path']);araw=original['raw_cache']
        else:
            if seed==1:
                endref=original['parent_reuse']['endpoint'];ar=read(original['parent_reuse']['A_result']['path'])
            else:
                endref=reference(Path(s['root'])/'formal/endpoint-lock.json');ar=read(Path(s['root'])/'infer-A/RESULT.json')
            end=read(endref['path']);binding=end['binding'];construction=original;araw=ar['raw']
        end=read(endref['path']);cp=end['checkpoint'];side=reference(Path(cp['path']).with_suffix('.pth.complete.json'))
        require(cp['sha256']==known['checkpoint_sha256'] and cp['completed_updates']==36480,'own e120 checkpoint differs')
        b=read(binding['path']);state=checkpoint.read(cp['path'],cp['sha256'],b)
        require(state['binding']['config']==Config(seed=seed).binding() and state['binding']['scope']==f'formal_R18_seed{seed}_e120','CPU full parent scope/config')
        require(state['engine']['completed_updates']==state['ema']['updates']==36480 and state['engine']['phase']==0,'CPU terminal clocks')
        del state
        parents.append(dict(seed=seed,identity=s['identity'],original_contract=known['original_contract'],
            endpoint=endref,binding=binding,checkpoint=cp,checkpoint_sidecar=side,fixed64=known['fixed64_ids'],
            construction={k:construction[k] for k in ('config','data','pretrained','data_sha256','environment')},
            A_raw=araw,accept_C_result=reference(Path(s['root'])/'accept-C/RESULT.json')))
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity
    job=Path(a.job);job.mkdir(parents=True,exist_ok=False)
    sample=guard.snapshot(job);guard.screen(sample,gpu=False,entry=True,disk_bytes=40*1024**3)
    c={'task_id':rt.TASK,'source_root':str(ROOT),'source_manifest':a.source_manifest,'source_reference':reference(a.source_manifest),
        'source_sha256':reference(a.source_manifest)['sha256'],'source_commit':read(a.source_manifest)['commit'],
        'plan_json':plan_json,'plan_md':plan_md,'input_binding':input_ref,
        'parent_contract':reference(parent),'parents':parents,'seeds':[0,1,2],'synthetic':not bool(a.cpu_gate),
        'GPU_authorized':bool(a.cpu_gate),'science_authorized':bool(a.cpu_gate),'training_updates':0,'holdout':False,
        'job_root':str(job),'runtime_root':a.runtime,'python':sys.executable,'contract_path':a.output,
        'boot_id':sample['boot_id'],'gpu_uuid':sample['snapshot']['gpu']['devices'][0]['uuid']['value'],
        'storage_identity':identity(sample['storage']),'environment':parents[0]['construction']['environment'],
        'parent_CPU_full_payloads_checked':True,'historical_sources_unchanged':True,'total_wallclock_cap':None}
    actual={'torch':str(torch.__version__),'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
    require(actual==c['environment'],'runtime environment differs')
    if a.cpu_gate:
        c['CPU_gate']=reference(a.cpu_gate);g=read(a.cpu_gate);require(g['status']=='PASS' and g['source_sha256']==c['source_sha256'],'source-bound final CPU gate')
    c['contract_sha256']=digest(c);publish(a.output,c)
    publish(job/'PREPARATION.json',{'status':'PASS','source':c['source_reference'],'contract':reference(a.output),'actual_environment':actual,
        'parents_checked':[{'seed':p['seed'],'checkpoint':p['checkpoint'],'binding':p['binding']} for p in parents],
        'resource_snapshot':sample,'CUDA_initialized':False,'model_constructions':0})
    print(__import__('json').dumps({'status':'PASS','contract':reference(a.output)}))


if __name__=='__main__':main()
