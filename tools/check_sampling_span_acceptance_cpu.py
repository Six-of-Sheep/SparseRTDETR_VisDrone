"""Actual acceptance-only CLI lifecycle on tiny fixed synthetic CPU materials.

No lab, CUDA, checkpoint download, scientific scoring or model quality claim.
Every child starts in the release cwd without inherited PYTHONPATH.
"""
import argparse, copy, json, os, sys, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from sparse_rtdetr.protocol_r18 import span_runtime as rt, span_acceptance as ac, span_acceptance_budget as budget
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    import torch
    import numpy as np
    from PIL import Image
    rt.require(torch.version.cuda is None and not torch.cuda.is_initialized(),'CPU-only torch required')
    source=rt.source_receipt(out/'source.json') if args.source is None else rt.reference(args.source.resolve())
    rt.verify_source(source);checks=[];processes=[]
    def check(name,value):
        rt.require(value,name);checks.append(name)
    def rejected(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('not rejected: '+name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    def cli(name,argv,okay=True,reason=None):
        cmd=[sys.executable,'-B',*map(str,argv)]
        r=recorded_subprocess(out/(name+'-process'),cmd,cwd=ROOT,env=env)
        check(name+' exit',(r.returncode==0)==okay)
        if reason:check(name+' rejection reason',reason in r.stderr.decode(errors='replace'))
        processes.append(dict(name=name,argv=cmd,returncode=r.returncode));return r
    data={}
    for role,split in (('train_core','train'),('development','val')):
        folder=out/role;(folder/'images').mkdir(parents=True);(folder/'annotations').mkdir()
        rows=[]
        for i in range(6):
            pixels=np.full((128+i%2,144+i,3),20+i*20,dtype=np.uint8)
            ip=folder/'images'/f'cpu-{i}.png';Image.fromarray(pixels).save(ip)
            text='' if i==0 else '5,5,12,14,1,1,0,0\n20,18,40,40,1,2,0,0\n1,1,90,90,1,3,0,0\n'
            ap=folder/'annotations'/f'cpu-{i}.txt';ap.write_text(text,encoding='utf-8')
            rows.append(dict(stable_image_id=stable_image_id(split,f'{split}/images/{ip.name}'),split=split,
                relative_path=f'{split}/images/{ip.name}',annotation_relative_path=f'{split}/annotations/{ap.name}',
                height=pixels.shape[0],width=pixels.shape[1],image_sha256=rt.sha(ip),image_size_bytes=ip.stat().st_size,
                annotation_sha256=rt.sha(ap),annotation_size_bytes=ap.stat().st_size,raw_annotation_rows=len(text.splitlines())))
        manifest=folder/'manifest.json';rt.publish(manifest,dict(role=role,profile='CPU_FIXTURE_ONLY',records=rows))
        data[role]=dict(role=role,count=6,manifest=rt.reference(manifest),roots=dict(role=role,images=str(folder/'images'),annotations=str(folder/'annotations')))
    from sparse_rtdetr.protocol_r18.span_worker import save_tensor
    pretrained=save_tensor(out/'SYNTHETIC-pretrained.pth',dict(profile='CPU_FIXTURE_ONLY'))
    rt.publish(out/'PRIOR_COST.json',dict(task_id=rt.TASK,profile='CPU_FIXTURE_ONLY',GPU_seconds=0,forward_calls=0))
    sample=rt.fixture_sample();config=rt.acceptance_config()
    c=dict(task_id=rt.TASK,protocol=rt.ACCEPTANCE_PROTOCOL,execution_scope=config['scope'],acceptance_limits=config['limits'],
        acceptance_sha256=rt.digest(config),profile='CPU_FIXTURE_ONLY',attempt='attempt-acceptance-cpu-success',nonce=uuid.uuid4().hex,
        job_root=str(out/'attempt-acceptance-cpu-success'),source_root=str(ROOT),python=sys.executable,source=source,commit=rt.exact(source)['commit'],
        science_sha256=rt.digest(rt.recipe()),execution_sha256=rt.digest(rt.execution()),stages=config['stages'],
        resources=rt.execution()['resources'],GPU_authorized=False,science_authorized=False,authorization=None,
        boot_id=sample['boot_id'],gpu_uuid='CPU_FIXTURE_NO_GPU',storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')},
        data=data,pretrained=pretrained,seed=0,prior_cost=rt.reference(out/'PRIOR_COST.json'),
        fixed_ids=[r['stable_image_id'] for r in rt.exact(data['development']['manifest'])['records']],
        baselines=rt.reference(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json'))
    spec=out/'raw-spec.json';rt.publish(spec,c);contract=out/'contract.json';rt.seal_contract(c,contract)
    entry=ROOT/'r18_sampling_acceptance.py';full=ROOT/'r18_sampling_span.py'
    cli('prepare',[entry,'prepare','--spec',spec,'--source',source['path'],'--output',out/'prepare'])
    check('prepare zero forward and CUDA',rt.read(out/'prepare/PREPARE_PASS.json')['model_forward_calls']==0 and not torch.cuda.is_initialized())
    cli('success',[entry,'run','--contract',contract])
    final=ac.load(contract);root=Path(c['job_root']);result=ac.verify(final)
    check('exact 9 updates and 18 actual forwards',result['actual_updates']==9 and result['forward_calls']==18)
    check('only four authorized stages',result['stages']==config['stages'] and not any((root/s).exists() for s in rt.STAGES[4:]))
    check('no scientific quality PASS',result['scientific_quality_evaluated'] is False and 'quality_passed' not in result)
    check('CPU result not GPU validation',result['execution_kind']=='CPU_SYNTHETIC_ONLY_NOT_GPU_VALIDATED')
    for arm in budget.ARMS:
        r=rt.verify_stage(final,'accept-'+arm)
        check(arm+' exact restore and no CUDA',r['actual_updates']==3 and r['full_state_RNG_exact'] and not r['state_usable_for_formal'] and not r['CUDA_initialized'])
    check('automatic independent slim export',(root/'slim-export/SHA_INDEX.json').exists())
    cli('verify',[entry,'verify','--contract',contract])
    cli('independent-export',[entry,'export','--contract',contract,'--output',out/'independent-export'])
    exported=rt.read(out/'independent-export/SHA_INDEX.json')
    check('no state tensors exported',exported['large_states_not_exported'] and not list((out/'independent-export').glob('*.pth')))
    check('three-arm ordinary DN summaries',set(rt.read(out/'independent-export/mechanism-summary.json'))=={'accept-'+a for a in budget.ARMS})
    closed=rt.reference(root/'CLOSED.json')
    cli('repeat',[entry,'run','--contract',contract],False)
    check('repeat leaves closure unchanged',closed==rt.reference(root/'CLOSED.json'))
    cli('export-repeat',[entry,'export','--contract',contract,'--output',out/'independent-export'],False)
    for stage in rt.STAGES[4:]:
        cli('forbidden-'+stage,[entry,'worker','--contract',contract,'--stage',stage],False,'invalid choice')
        rejected('runtime rejects '+stage,lambda s=stage:rt.stage_path(final,s))
    cli('full-entry-acceptance-contract',[full,'run','--contract',contract],False,'foreign task/protocol')
    cli('full-worker-acceptance-contract',[full,'worker','--contract',contract,'--stage','train-B0'],False,'foreign task/protocol')
    cli('full-prepare-acceptance-spec',[full,'prepare','--spec',spec,'--source',source['path'],'--output',out/'forbidden-full-prepare'],False,'acceptance spec refused')
    cli('full-bind-acceptance-spec',[full,'bind','--spec',spec,'--source',source['path'],'--cpu-gate',out/'PRIOR_COST.json','--authorization',out/'PRIOR_COST.json','--output',out/'forbidden-full-bind.json'],False,'acceptance spec refused')
    full_c=copy.deepcopy(c)
    for k in ('execution_scope','acceptance_limits','acceptance_sha256'):full_c.pop(k)
    full_c['protocol']=rt.PROTOCOL;full_c['stages']=list(rt.STAGES)
    path=out/'full-contract.json';rt.seal_contract(full_c,path)
    cli('acceptance-entry-full-contract',[entry,'run','--contract',path],False,'foreign task/protocol')
    for field,value in (('actual_updates_total',10),('batch_forward_calls_total',19),('actual_updates_per_arm',4)):
        bad=copy.deepcopy(c);bad['acceptance_limits'][field]=value;path=out/(field+'-contract.json');rt.seal_contract(bad,path)
        cli('limit-'+field,[entry,'run','--contract',path],False,'acceptance scope/limits differ')
    bad=copy.deepcopy(c);bad['science_authorized']=True;path=out/'science-true.json';rt.seal_contract(bad,path)
    cli('science-authority',[entry,'run','--contract',path],False,'acceptance cannot authorize')
    bad=copy.deepcopy(c);bad.update(attempt='attempt-acceptance-cpu-fault',job_root=str(out/'attempt-acceptance-cpu-fault'),fixture_fault_stage='admission')
    path=out/'fault-contract.json';rt.seal_contract(bad,path)
    cli('worker-fault',[entry,'run','--contract',path],False,'acceptance supervised stage stopped')
    check('fault has STOP and no closure',(Path(bad['job_root'])/'STOP.json').exists() and not (Path(bad['job_root'])/'CLOSED.json').exists())
    cli('fault-repeat',[entry,'run','--contract',path],False)
    # Exercise independent pre-operation update and microbatch-forward counters.
    unit=copy.deepcopy(final);unit['job_root']=str(out/'counter-unit');Path(unit['job_root']).mkdir();budget.start(unit)
    for arm in budget.ARMS:
        for _ in range(3):
            budget.reserve_update(unit,arm)
            for _ in range(2):budget.reserve_forward(unit,arm);budget.complete_forward(unit,arm,'torch.float32')
            budget.complete_update(unit,arm)
    budget.verify_complete(unit);before=rt.reference(budget.path(unit))
    rejected('tenth update rejected before execution',lambda:budget.reserve_update(unit,'P'))
    check('denied update leaves count bytes unchanged',before==rt.reference(budget.path(unit)))
    rejected('nineteenth forward rejected before execution',lambda:budget.reserve_forward(unit,'P'))
    check('denied forward leaves count bytes unchanged',before==rt.reference(budget.path(unit)))
    # Exercise the production metadata validator without CUDA or lab authority.
    label='ACCEPTANCE_CPU_ENTRY_PASS'
    direct=dict(status=label,source_sha256=source['sha256'],source=source,skips=0)
    check('same-source acceptance gate',rt.verify_cpu_gate(direct,source,label)==direct)
    parent=out/'CPU_FIXTURE_PARENT_GATE.json';rt.publish(parent,direct)
    rebound=dict(direct,byte_identical_parent_gate=rt.reference(parent),byte_identical_parent_source=source)
    check('acceptance evidence byte-identical rebind',rt.verify_cpu_gate(rebound,source,label)==rebound)
    rejected('acceptance gate cannot serve full protocol',lambda:rt.verify_cpu_gate(rebound,source,'PRODUCTION_CPU_ENTRY_PASS'))
    legacy=dict(direct,status='PRODUCTION_CPU_ENTRY_PASS');p=out/'CPU_FIXTURE_FULL_PARENT_GATE.json';rt.publish(p,legacy)
    bad=dict(rebound,byte_identical_parent_gate=rt.reference(p))
    rejected('full parent cannot serve acceptance protocol',lambda:rt.verify_cpu_gate(bad,source,label))
    old=dict(legacy,byte_identical_parent_gate=rt.reference(p),byte_identical_parent_source=source)
    check('old full protocol rebind retained',rt.verify_cpu_gate(old,source,'PRODUCTION_CPU_ENTRY_PASS')==old)
    changed=copy.deepcopy(rt.exact(source));changed['files'][0]['sha256']='0'*64
    p=out/'CPU_FIXTURE_CHANGED_SOURCE.json';rt.publish(p,changed)
    prior=dict(direct,source=rt.reference(p));p=out/'CPU_FIXTURE_CHANGED_PARENT_GATE.json';rt.publish(p,prior)
    bad=dict(rebound,byte_identical_parent_gate=rt.reference(p),byte_identical_parent_source=prior['source'])
    rejected('changed source cannot rebind evidence',lambda:rt.verify_cpu_gate(bad,source,label))
    bad=dict(rebound,byte_identical_parent_source=dict(source,bytes=source['bytes']+1))
    rejected('parent reference conflict cannot rebind evidence',lambda:rt.verify_cpu_gate(bad,source,label))
    rt.publish(out/'CPU_GATE.json',dict(status='ACCEPTANCE_CPU_ENTRY_PASS',source=source,source_sha256=source['sha256'],
        checks=checks,checks_run=len(checks),skips=0,commands=processes,synthetic_updates=9,synthetic_model_forwards=18,
        production_GPU_forward_calls=0,scientific_updates=0,scientific_AP_calls=0,CUDA_initialized=False))
    # Only this successful tool's newly generated synthetic tensors, not old states.
    removed=[]
    for pth in [Path(pretrained['path']),*root.rglob('*.pth')]:removed.append(rt.reference(pth));pth.unlink()
    rt.publish(out/'OWN_SUCCESSFUL_SYNTHETIC_STATES_REMOVED.json',dict(files=removed,old_D_28_untouched=True))
    print(json.dumps(dict(status='ACCEPTANCE_CPU_ENTRY_PASS',checks_run=len(checks),skips=0)))


if __name__=='__main__':main()
