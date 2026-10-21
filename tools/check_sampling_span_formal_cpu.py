"""Actual CLI acceptance -> external reference -> formal/score/export on CPU.

Synthetic CPU evidence is explicit; never claims production data/GPU quality.
"""
import argparse,copy,json,os,sys,uuid,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    from PIL import Image
    from sparse_rtdetr.protocol_r18 import span_runtime as rt,span_worker as w,span_formal as f,span_budget as budget
    from sparse_rtdetr.protocol_r18.span_external_acceptance import descriptor
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    rt.require(torch.version.cuda is None and not torch.cuda.is_initialized(),'CPU-only environment required')
    source=rt.source_receipt(out/'source.json') if a.source is None else rt.reference(a.source.resolve());rt.verify_source(source)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('PYTHONPATH',None)
    checks=[];commands=[]
    def check(name,ok):rt.require(ok,name);checks.append(name)
    def cli(name,entry,argv,okay=True,reason=None):
        r=recorded_subprocess(out/(name+'-process'),[sys.executable,'-B',str(ROOT/entry),*map(str,argv)],cwd=ROOT,env=env)
        check(name+' exit',(r.returncode==0)==okay)
        if reason:check(name+' rejection',reason in r.stderr.decode(errors='replace'))
        commands.append(dict(name=name,returncode=r.returncode));return r
    data={}
    for role,split in (('train_core','train'),('development','val')):
        folder=out/role;(folder/'images').mkdir(parents=True);(folder/'annotations').mkdir();rows=[]
        for i in range(6):
            ip=folder/'images'/f'{i}.png';pixels=np.random.default_rng(i).integers(0,256,(129+i,161+i,3),dtype=np.uint8);Image.fromarray(pixels).save(ip)
            lines=[] if i==0 else [f'{5+j*2},{5+j},12,14,1,{j+1},0,0' for j in range(10)]+['20,20,40,40,1,1,0,0','1,1,110,100,1,2,0,0','130,110,20,15,0,0,0,0']
            ap=folder/'annotations'/f'{i}.txt';ap.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
            rows.append(dict(stable_image_id=stable_image_id(split,f'{split}/images/{ip.name}'),split=split,
                relative_path=f'{split}/images/{ip.name}',annotation_relative_path=f'{split}/annotations/{ap.name}',height=129+i,width=161+i,
                image_sha256=rt.sha(ip),image_size_bytes=ip.stat().st_size,annotation_sha256=rt.sha(ap),annotation_size_bytes=ap.stat().st_size,raw_annotation_rows=len(lines)))
        m=folder/'manifest.json';rt.publish(m,dict(role=role,profile='CPU_FIXTURE_ONLY',records=rows));data[role]=dict(role=role,count=6,manifest=rt.reference(m),roots=dict(role=role,images=str(folder/'images'),annotations=str(folder/'annotations')))
    pre=w.save_tensor(out/'SYNTHETIC-pretrained.pth',dict(profile='CPU_FIXTURE_ONLY'))
    rt.publish(out/'zero-cost.json',dict(task_id=rt.TASK,profile='CPU_FIXTURE_ONLY',GPU_seconds=0,forward_calls=0))
    sample=rt.fixture_sample();cfg=rt.acceptance_config();oldroot=out/'attempt-old-cpu-acceptance'
    old=dict(task_id=rt.TASK,protocol=rt.ACCEPTANCE_PROTOCOL,profile='CPU_FIXTURE_ONLY',execution_scope='GPU_ACCEPTANCE_ONLY',
        acceptance_limits=cfg['limits'],acceptance_sha256=rt.digest(cfg),attempt=oldroot.name,job_root=str(oldroot),nonce=uuid.uuid4().hex,
        source_root=str(ROOT),python=sys.executable,commit=rt.exact(source)['commit'],source=source,science_sha256=rt.digest(rt.recipe()),execution_sha256=rt.digest(rt.execution()),
        stages=cfg['stages'],resources=rt.execution()['resources'],GPU_authorized=False,science_authorized=False,authorization=None,
        boot_id=sample['boot_id'],gpu_uuid='CPU_FIXTURE_NO_GPU',storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')},
        data=data,pretrained=pre,seed=0,prior_cost=rt.reference(out/'zero-cost.json'),
        fixed_ids=[r['stable_image_id'] for r in rt.exact(data['development']['manifest'])['records']],
        baselines=rt.reference(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json'))
    parent=out/'parent';parent.mkdir();oldpath=parent/'contract.json';rt.seal_contract(old,oldpath)
    cli('old-acceptance','r18_sampling_acceptance.py',['run','--contract',oldpath])
    # CPU acceptance has zero allocated GPU time. Explicit fixture receipt uses
    # actual finished counts; it is never accepted by LAB production identity.
    rt.publish(oldroot/'COST_LEDGER.json',dict(profile='CPU_FIXTURE_ONLY',active=None,forward_calls=18,completed_GPU_seconds=0,
        counts=rt.reference(oldroot/'ACCEPTANCE_COUNTS.json'),simulated_cost_only=True))
    rt.publish(parent/'EXECUTION_EXIT.json',dict(status='PASS',returncode=0,contract=rt.reference(oldpath),scientific_quality_evaluated=False))
    desc=out/'external.json';rt.publish(desc,descriptor(oldpath))
    rt.publish(out/'prior-cost.json',dict(task_id=rt.TASK,profile='CPU_FIXTURE_ONLY',GPU_seconds=0,forward_calls=18))
    raw={k:v for k,v in old.items() if k not in ('acceptance_limits','acceptance_sha256','stages','protocol','resources','source','commit','python','source_root')}
    raw.update(execution_scope='GPU_FORMAL_E120_ONCE',external_acceptance=rt.reference(desc),prior_cost=rt.reference(out/'prior-cost.json'),
        attempt='attempt-formal-cpu',job_root=str(out/'attempt-formal-cpu'),nonce=uuid.uuid4().hex)
    spec=out/'raw-spec.json';rt.publish(spec,raw)
    cli('prepare','r18_sampling_formal.py',['prepare','--spec',spec,'--source',source['path'],'--output',out/'prepare'])
    check('prepare real construction zero CUDA/forward',not rt.read(out/'prepare/PREPARE_PASS.json')['CUDA_initialized'] and rt.read(out/'prepare/PREPARE_PASS.json')['model_forward_calls']==0)
    c=f.base(spec,source);c.update(GPU_authorized=False,science_authorized=False,authorization=None)
    contract=out/'formal-contract.json';rt.seal_contract(c,contract)
    oldclose=rt.reference(oldroot/'CLOSED.json');oldcounts=rt.reference(oldroot/'ACCEPTANCE_COUNTS.json')
    cli('formal','r18_sampling_formal.py',['run','--contract',contract])
    final=f.load(contract);result=f.verify(final);root=Path(c['job_root'])
    check('eleven formal stages',len(result['stages'])==11)
    check('inherited18 new66 cumulative84',result['inherited_forward_calls']==18 and result['new_forward_calls']==66 and result['forward_calls_total']==84)
    check('no local acceptance directories',not any((root/('accept-'+arm)).exists() for arm in ('B0','B1','P')))
    check('original acceptance closure/count unchanged',oldclose==rt.reference(oldroot/'CLOSED.json') and oldcounts==rt.reference(oldroot/'ACCEPTANCE_COUNTS.json'))
    for stage in rt.permitted_stages(final):
        check(stage+' actual closure',rt.verify_stage(final,stage)['status']=='PASS')
        sample=rt.read(root/stage/'resource-entry.json')
        age=sample['age_check']
        check(stage+' fresh resource timing and stage',0<=age['actual_age_seconds']<25 and
            age['sampling_started'] is not None and age['sampling_completed'] is not None and age['phase_identity']['stage']==stage)
    for arm in ('B0','B1','P'):
        r=rt.verify_stage(final,'train-'+arm)
        check(arm+' independent initialization',not r['used_acceptance_state'] and r['initial_model_sha256']==rt.verify_stage(final,'accept-'+arm)['initial_model_sha256'])
        state=w.torch_ref(rt.reference(root/('train-'+arm)/'checkpoint-e000.pth'))
        check(arm+' empty optimizer and zero EMA/updates',not state['optimizer']['state'] and state['engine']['completed_updates']==state['ema']['updates']==0)
    check('slim export includes original closure bytes',rt.sha(root/'export/accept-B0-CLOSED.json')==rt.sha(oldroot/'accept-B0/CLOSED.json'))
    check('dual evaluator six synthetic groups only',rt.verify_stage(final,'score')['groups']==6 and rt.verify_stage(final,'score')['evaluator_calls']==12)
    check('CPU comparison is not scientific pass',rt.verify_stage(final,'compare')['quality_passed'] is None and rt.verify_stage(final,'compare')['cost_passed'] is None)
    export=rt.exact(rt.verify_stage(final,'export')['index'])
    check('whitelisted final export exact SHA',all(rt.reference(r['path'])==r for r in export['files']) and not any(r['path'].endswith('.pth') for r in export['files']))
    cli('verify','r18_sampling_formal.py',['verify','--contract',contract])
    cli('repeat','r18_sampling_formal.py',['run','--contract',contract],False)
    cli('forbidden-accept','r18_sampling_formal.py',['worker','--contract',contract,'--stage','accept-B0'],False,'invalid choice')
    cli('old-entry-rejects-formal','r18_sampling_span.py',['run','--contract',contract],False,'foreign task/protocol')
    def bad(name,descriptor_value=None,prior=None):
        cc=copy.deepcopy(c)
        if descriptor_value is not None:
            p=out/(name+'-evidence.json');rt.publish(p,descriptor_value);cc['external_acceptance']=rt.reference(p)
        if prior is not None:cc['prior_cost']=prior
        p=out/(name+'-contract.json');rt.seal_contract(cc,p)
        cli(name,'r18_sampling_formal.py',['prepare','--contract',p,'--output',out/name],False)
    d=rt.read(desc);wrong=copy.deepcopy(d);wrong['contract']['sha256']='0'*64;bad('external-SHA',wrong)
    wrong=copy.deepcopy(d);wrong['source']['sha256']='0'*64;bad('external-source-identity',wrong)
    wrong=copy.deepcopy(d);wrong['closed']['path']=str(out/'unclosed.json');bad('external-unclosed',wrong)
    changed=rt.read(oldpath);changed['task_id']='foreign';p=out/'foreign-parent.json';rt.seal_contract(changed,p);wrong=dict(d,contract=rt.reference(p));bad('external-task-identity',wrong)
    rt.publish(out/'reset-cost.json',dict(task_id=rt.TASK,profile='CPU_FIXTURE_ONLY',GPU_seconds=0,forward_calls=0));bad('cost-reset',prior=rt.reference(out/'reset-cost.json'))
    # Same cumulative budget implementation, simulated LAB metadata, zero CUDA.
    unit=copy.deepcopy(final);unit.update(profile='LAB_GPU_ONCE',job_root=str(out/'budget-unit'));Path(unit['job_root']).mkdir()
    rt.publish(out/'SIMULATED-prior.json',dict(task_id=rt.TASK,profile='LAB_GPU_ONCE',GPU_seconds=79.265548953088,forward_calls=18))
    unit['prior_cost']=rt.reference(out/'SIMULATED-prior.json');budget.start(unit,'train-B0');budget.guard(unit,forward_calls=2)
    ledger_path=Path(unit['job_root'])/'COST_LEDGER.json';ledger=rt.read(ledger_path)
    check('budget inherits79seconds and18forward before reservation',ledger['completed_GPU_seconds']==79.265548953088 and ledger['forward_calls']==20)
    ledger['forward_calls']=219693;rt.atomic(ledger_path,ledger)
    try:budget.guard(unit,forward_calls=1)
    except RuntimeError:checks.append('total219693 rejects extra forward')
    else:raise RuntimeError('forward budget not enforced')
    ledger=rt.read(ledger_path);ledger['completed_GPU_seconds']=60*3600;rt.atomic(ledger_path,ledger)
    try:budget.guard(unit)
    except RuntimeError:checks.append('cumulative60hour rejects without reset')
    else:raise RuntimeError('GPU time cap not enforced')
    check('CPU has no CUDA',not torch.cuda.is_initialized())
    rt.publish(out/'CPU_GATE.json',dict(status='FORMAL_CPU_ENTRY_PASS',source=source,source_sha256=source['sha256'],checks=checks,checks_run=len(checks),skips=0,
        CUDA_initialized=False,production_GPU_forward_calls=0,synthetic_parent_forward_calls=18,synthetic_formal_forward_calls=66,scientific_AP_calls=0,
        actual_CLI_processes=commands,contract=rt.reference(contract),external_acceptance=rt.reference(desc)))
    removed=[]
    for p in [Path(pre['path']),*oldroot.rglob('*.pth'),*root.rglob('*.pth')]:removed.append(rt.reference(p));p.unlink()
    rt.publish(out/'OWN_SUCCESSFUL_SYNTHETIC_STATES_REMOVED.json',dict(files=removed,old_D28_untouched=True))
    print(json.dumps(dict(status='FORMAL_CPU_ENTRY_PASS',checks_run=len(checks),skips=0)))
if __name__=='__main__':main()
