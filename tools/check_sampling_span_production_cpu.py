"""Final production CLI integration on explicit synthetic CPU-only materials.

Never reads lab paths/weights/data, creates GPU authority, or initializes CUDA.
"""
import argparse,copy,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path,help='pre-frozen actual-byte manifest for WSL/exported release without Git metadata')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if os.name!='posix':raise RuntimeError('use Linux/WSL CPU-only environment; original evaluator POSIX reader required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import numpy as np
    import torch
    from PIL import Image
    from sparse_rtdetr.protocol_r18 import span_runtime as rt,span_worker as w,span_checkpoint as checkpoint
    from sparse_rtdetr.protocol_r18.model import Config,build
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    torch.set_num_threads(1)
    if torch.cuda.is_initialized() or torch.version.cuda is not None:raise RuntimeError('CPU-only torch with CUDA uninitialized required')
    checks=[];calls=[]
    def check(name,condition):
        if not condition:raise RuntimeError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError,FileNotFoundError,TypeError,AssertionError):checks.append(name)
        else:raise RuntimeError('not rejected: '+name)
    def cli(name,c,action='run',stage=None,okay=True,extra=()):
        contract=out/(name+'-contract.json');rt.seal_contract(c,contract)
        cmd=[sys.executable,'-B',str(ROOT/'r18_sampling_span.py'),action,'--contract',str(contract)]
        if stage:cmd+=['--stage',stage]
        cmd+=list(extra)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU');env.pop('PYTHONPATH',None)
        start=time.monotonic()
        from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
        result=recorded_subprocess(out/(name+'-process'),cmd,cwd=ROOT,env=env)
        p=result
        calls.append(dict(name=name,command=cmd,returncode=p.returncode,seconds=time.monotonic()-start))
        check(name+' exit', (p.returncode==0)==okay)
        # Expected rejection contracts are deliberately invalid. Do not load
        # them again after the real entry has correctly rejected them.
        return rt.load(contract) if okay else None
    datasets={}
    for role,split in (('train_core','train'),('development','val')):
        directory=out/role;(directory/'images').mkdir(parents=True);(directory/'annotations').mkdir();rows=[]
        for i in range(6):
            h,wi=129+i,161+i;name=f'CPU_{role}_{i}'
            pixels=np.random.default_rng(i).integers(0,256,(h,wi,3),dtype=np.uint8)
            ip=directory/'images'/(name+'.png');Image.fromarray(pixels).save(ip)
            # Empty GT, ignores, all ten categories and native area bins.
            lines=[] if i==0 else [f'{5+j*2},{5+j},12,14,1,{j+1},0,0' for j in range(10)]
            if i>0:lines+=['20,20,40,40,1,1,0,0','1,1,110,100,1,2,0,0','130,110,20,15,0,0,0,0']
            ap=directory/'annotations'/(name+'.txt');ap.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
            rows.append(dict(stable_image_id=stable_image_id(split,f'{split}/images/{ip.name}'),split=split,
                relative_path=f'{split}/images/{ip.name}',annotation_relative_path=f'{split}/annotations/{ap.name}',
                height=h,width=wi,image_sha256=rt.sha(ip),image_size_bytes=ip.stat().st_size,
                annotation_sha256=rt.sha(ap),annotation_size_bytes=ap.stat().st_size,raw_annotation_rows=len(lines)))
        path=directory/'manifest.json';rt.publish(path,dict(role=role,profile='CPU_FIXTURE_ONLY',records=rows))
        datasets[role]=dict(role=role,count=6,manifest=rt.reference(path),roots=dict(role=role,images=str(directory/'images'),annotations=str(directory/'annotations')))
    cfg=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32');model,_,_=build(cfg,synthetic=True)
    pretrained=w.save_tensor(out/'SYNTHETIC-pretrained.pth',dict(profile='CPU_FIXTURE_ONLY'));del model
    source=rt.source_receipt(out/'source.json') if args.source is None else rt.reference(args.source.resolve())
    rt.verify_source(source);sample=rt.fixture_sample()
    rt.publish(out/'PRIOR_COST.json',dict(task_id=rt.TASK,GPU_seconds=0,forward_calls=0,scope='CPU_FIXTURE_ONLY'))
    c=dict(task_id=rt.TASK,protocol=rt.PROTOCOL,profile='CPU_FIXTURE_ONLY',attempt='attempt-cpu-success',nonce=uuid.uuid4().hex,
        job_root=str(out/'attempt-cpu-success'),source_root=str(ROOT),python=sys.executable,commit=rt.exact(source)['commit'],source=source,
        science_sha256=rt.digest(rt.recipe()),execution_sha256=rt.digest(rt.execution()),stages=list(rt.STAGES),resources=rt.execution()['resources'],
        GPU_authorized=False,science_authorized=False,authorization=None,boot_id=sample['boot_id'],gpu_uuid='CPU_FIXTURE_NO_GPU',
        storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')},data=datasets,pretrained=pretrained,seed=0,prior_cost=rt.reference(out/'PRIOR_COST.json'),
        fixed_ids=[r['stable_image_id'] for r in rt.exact(datasets['development']['manifest'])['records']],
        baselines=rt.reference(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json'))
    cli('prepare-success',c,action='prepare',extra=('--output',str(out/'prepare-success')))
    check('prepare zero forward',rt.read(out/'prepare-success/PREPARE_PASS.json')['model_forward_calls']==0)
    rt.publish(out/'prepare-spec.json',c)
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    result=recorded_subprocess(out/'prepare-spec-process',[sys.executable,'-B',str(ROOT/'r18_sampling_span.py'),'prepare','--spec',str(out/'prepare-spec.json'),'--source',source['path'],'--output',str(out/'prepare-via-spec')],cwd=ROOT,env=env)
    check('prepare spec real CLI exit',result.returncode==0)
    final=cli('success',c)
    check('14 stage closure',len(rt.read(Path(c['job_root'])/'RESULT.json')['stages'])==14)
    for stage in rt.STAGES:
        value=rt.verify_stage(final,stage);check(stage+' CPU no CUDA',not value['CUDA_initialized'] and value['fixture_only'])
    check('inference tail2',rt.verify_events(final,'infer-P',rt.stage_path(final,'infer-P')/'events.jsonl')[2]['fields']['actual_batch']==2)
    check('dual evaluators called',rt.verify_stage(final,'score')['evaluator_calls']==12)
    for arm in ('B0','B1','P'):
        check(arm+' independent formal parent',not rt.verify_stage(final,'train-'+arm)['used_acceptance_state'])
        check(arm+' cross epoch',rt.exact(rt.verify_stage(final,'train-'+arm)['endpoint'])['epoch']==2)
    check('not scientific quality/cost',rt.verify_stage(final,'compare')['quality_passed'] is None and rt.verify_stage(final,'compare')['cost_passed'] is None)
    before=rt.reference(Path(c['job_root'])/'CLOSED.json');cli('repeat',c,okay=False)
    check('repeat leaves closure unchanged',before==rt.reference(Path(c['job_root'])/'CLOSED.json'))
    bad=copy.deepcopy(c);bad.update(attempt='attempt-cpu-fault',job_root=str(out/'attempt-cpu-fault'),fixture_fault_stage='admission')
    cli('worker-fault',bad,okay=False)
    check('worker STOP no closure',(Path(bad['job_root'])/'admission/STOP.json').exists() and not (Path(bad['job_root'])/'CLOSED.json').exists())
    bad=copy.deepcopy(c);bad['source']['sha256']='0'*64;cli('source-reject',bad,okay=False)
    bad=copy.deepcopy(c);bad['seed']=1;cli('seed-reject',bad,okay=False)
    bad=copy.deepcopy(c);bad['pretrained']['sha256']='0'*64
    cli('prepare-initial-SHA',bad,action='prepare',okay=False,extra=('--output',str(out/'prepare-initial-SHA')))
    bad=copy.deepcopy(c);bad['data']['train_core']['role']='development';cli('role-reject',bad,okay=False)
    bad=copy.deepcopy(c);bad.update(attempt='attempt-cpu-missing-material',job_root=str(out/'attempt-cpu-missing-material'))
    bad['data']['train_core']['roots']['images']=str(out/'NOT_PRESENT')
    cli('prepare-missing',bad,action='prepare',okay=False,extra=('--output',str(out/'prepare-missing')))
    cli('missing-material',bad,okay=False)
    check('missing material fails before GPU acceptance',not (Path(bad['job_root'])/'accept-B0').exists())
    reject('JSON nonfinite rejected',lambda:rt.publish(out/'invalid-json.json',dict(v=float('nan'))))
    reject('JSON foreign integer rejected',lambda:rt.canonical(dict(v=np.int64(1))))
    value=dict(native_count=int(np.int64(1)),zero=0,empty=[]);rt.publish(out/'roundtrip.json',value)
    check('JSON native roundtrip',rt.read(out/'roundtrip.json')==value and type(rt.read(out/'roundtrip.json')['native_count']) is int)
    for name,alter in [('RAM',lambda s:s['snapshot']['resources'].update(ram_available=0)),('stale',lambda s:s.update(sampled_epoch=0)),
        ('boot',lambda s:s.update(boot_id='other')),('temperature',lambda s:s['snapshot'].update(cpu_temperature_max=90)),
        ('OOM',lambda s:s['memory_events']['vmstat'].update(oom_kill=1))]:
        sample=rt.fixture_sample();alter(sample)
        reject('resource '+name,lambda:rt.screen(sample,c,entry=True,gpu=False,baseline=rt.fixture_sample()))
    check('GPU resource schema simulated only',rt.screen(rt.fixture_sample(),c,entry=True,gpu=True)['fixture_only'])
    sample=rt.fixture_sample();sample['snapshot']['gpu']['devices'][0]['memory.free']['value']=19999
    reject('GPU resource memory rejection simulated only',lambda:rt.screen(sample,c,entry=True,gpu=True))
    state_ref=rt.verify_stage(final,'accept-P')['checkpoint']
    engine,_=w.engine(final,'P','accept-P',torch.device('cpu'))
    wrong=dict(engine.binding,arm='r18-sampling-span-v1/B0')
    reject('restore cross arm binding',lambda:checkpoint.restore(engine,wrong,state_ref['path'],state_ref['sha256']))
    check('rejected restore poisons engine',engine.failed)
    del engine
    events=rt.stage_path(final,'admission')/'events.jsonl';original=events.read_bytes();badpath=out/'bad-events.jsonl'
    lines=original.splitlines(keepends=True)
    for label,raw in [('missing',b''.join(lines[:-1])),('reorder',b''.join([lines[1],lines[0],*lines[2:]])),('duplicate',original+lines[-1]),('partial',original+b'{')]:
        badpath.write_bytes(raw);reject('terminal '+label,lambda:rt.verify_events(final,'admission',badpath))
    badpath.write_bytes(original+b'{');check('live partial tail safe',len(rt.verify_events(final,'admission',badpath,complete=False))==len(lines))
    from sparse_rtdetr.protocol_r18.span_acceptance_checks import check_interfaces
    checks+=check_interfaces(final,out)
    receipt=dict(status='PRODUCTION_CPU_ENTRY_PASS',profile='CPU_FIXTURE_ONLY',source_sha256=source['sha256'],source=source,
        checks=checks,checks_run=len(checks),skips=0,commands=calls,contract=rt.reference(Path(final['_path'])),
        production_GPU_calls=0,production_teacher_caches=0,scientific_training_updates=0,scientific_AP_calls=0,
        synthetic_forward_calls=rt.verify_stage(final,'compare')['total_forward_calls'],CUDA_initialized=torch.cuda.is_initialized())
    receipt['environment']=dict(python=sys.version,torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,numpy=np.__version__,platform=sys.platform)
    rt.publish(out/'CPU_GATE.json',receipt)
    # Only successfully closed, generated synthetic tensors are removed locally.
    removed=[]
    for p in [Path(pretrained['path']),*Path(c['job_root']).rglob('*.pth')]:
        removed.append(rt.reference(p));p.unlink()
    rt.publish(out/'SYNTHETIC_TENSORS_NOT_RETAINED.json',dict(files=removed,reason='own successful synthetic CPU tensors only; all small audit/closure/SHA retained'))
    print(json.dumps(dict(status=receipt['status'],checks_run=len(checks),skips=0)))


if __name__=='__main__':main()
