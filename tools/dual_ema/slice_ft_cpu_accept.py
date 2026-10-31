"""Drive the final dual-EMA executable through real supervisors and dual evaluators (CPU gate)."""
import copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def main():
    root=Path(sys.argv[1]).resolve();source=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source/'src'))
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as execution
    from sparse_rtdetr.localization.runio import read,publish,reference,digest
    from sparse_rtdetr.localization import guard,channel
    execution.install()
    source_ref=json.loads(Path(sys.argv[2]).read_text());manifest=read(source_ref['path'])
    assert reference(source_ref['path'])==source_ref
    assert all((source/r['path']).stat().st_size==r['bytes'] and hashlib.sha256((source/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['source_files'])
    snap=guard.snapshot(root);guard.screen(snap,gpu=False,entry=True,disk_bytes=100*1024**3)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MKL_THREADING_LAYER':'GNU','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    results=[]
    ids=[hashlib.sha256(('CPU_FT_'+str(i)).encode()).hexdigest() for i in range(8)]
    def create(name,fault=None):
        d=root/name;d.mkdir()
        path=d/'contract.json'
        c={'task_id':execution.TASK+'-CPU-'+name,'scope':'CPU_FIXTURE','contract_path':str(path),
           'source_root':str(source),'source_manifest':source_ref['path'],'source_sha256':source_ref['sha256'],'source_commit':manifest['commit'],
           'plan':str(source/'configs/research/r18_dual_ema_v1.json'),'plan_sha256':execution.PLAN_SHA,
           'job_root':str(d/'job'),'runtime_root':str(Path('/home/lyy/.cache')/(execution.TASK+'-CPU-'+name+'-'+hashlib.sha256(str(root).encode()).hexdigest()[:16])),
           'epochs':2,'updates_per_epoch':1,'total_updates':2,'development_ids':ids,'fixed64':ids,
           'GPU_authorized':False,'formal_authorized':False,'python':sys.executable,
           'gpu_uuid':snap['snapshot']['gpu']['devices'][0]['uuid']['value'],'boot_id':snap['boot_id'],
           'data_sha256':hashlib.sha256(b'CPU_FT_fixture_data_v1').hexdigest(),
           'parent_checkpoint':{'scope':'CPU_FIXTURE_ONLY','sha256':'a'*64},'launch_nonce':hashlib.sha256(name.encode()).hexdigest(),
           'fixture_worker_error':fault}
        c['contract_sha256']=digest(c);publish(path,c)
        token=d/'token.json';publish(token,{'task_id':c['task_id'],'contract_sha256':c['contract_sha256'],'nonce':c['launch_nonce']})
        return c,token
    # Targeted dual-EMA checks must pass before the fixture chain is driven.
    check_out=root/'dual-ema-check'
    checker=subprocess.run([sys.executable,'-B',str(source/'tools/dual_ema_check.py'),str(check_out)],env=env,cwd=source,capture_output=True)
    (root/'dual-ema-check.log').write_bytes(checker.stdout+checker.stderr)
    assert checker.returncode==0,(checker.stdout+checker.stderr).decode()[-2000:]
    dual_check=read(check_out/'DUAL_EMA_CHECK.json');assert dual_check['status']=='PASS'
    success,token=create('normal')
    def run(c,token,label):
        cmd=[sys.executable,'-B',str(source/'slice_ft_run.py'),'launch','--contract',c['contract_path'],'--token',str(token)]
        start=time.monotonic()
        with (root/(label+'.log')).open('x') as f:p=subprocess.run(cmd,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT)
        row={'label':label,'returncode':p.returncode,'elapsed_seconds':time.monotonic()-start,'command':cmd};results.append(row)
        return p.returncode
    assert run(success,token,'success')==0,'actual success entry failed'
    result=read(Path(success['job_root'])/'RESULT.json');assert result['status']=='COMPLETE'
    for phase in execution.PHASES:
        r=execution.complete(success,phase)
        if phase=='evaluation':assert r['score_budget_calls']==4
        q=Path(success['job_root'])/phase/'runtime-archive/supervisor-exit.json';e=read(q)
        assert e['status']=='PASS' and not e['remaining_live'] and not e['unknown_pids']
    job=Path(success['job_root'])
    # Declared checkpoint cadence, dual-EMA payload and real fixture sizes.
    formal=job/'formal-D0'
    present=sorted(p.name for p in formal.glob('checkpoint-FT*.pth'))
    assert present==['checkpoint-FT000.pth','checkpoint-FT002.pth'],present
    accept_result=read(job/'accept-D0'/'RESULT.json')
    endpoint=read(formal/'RESULT.json')
    assert endpoint['checkpoint_passes']==[0,2],endpoint['checkpoint_passes']
    assert endpoint['dual_ema']['ema_r_updates']==2 and endpoint['dual_ema']['ema_c_updates']==2+36480
    assert endpoint['dual_ema']['rng_neutral_alt_ema_updates']==2
    assert accept_result['dual_ema']['ema_c_updates']==2+36480 and accept_result['dual_ema']['rng_neutral_alt_ema_updates']==2
    payload=__import__('torch').load(str(formal/'checkpoint-FT002.pth'),map_location='cpu',weights_only=True)
    assert payload['ema']['updates']==2 and payload['ema_alt']['updates']==2+36480,'dual-EMA payload counters differ'
    assert set(payload['ema_alt']['module'])==set(payload['ema']['module']),'EMA payload module keys differ'
    sizes=sorted(({'path':str(p),'bytes':p.stat().st_size} for p in job.rglob('*.pth')),key=lambda r:r['path'])
    fixture_bytes=sum(r['bytes'] for r in sizes)
    del payload
    before={str(p):reference(p) for p in job.glob('*.json')}
    assert run(success,token,'duplicate-launch')!=0
    assert before=={str(p):reference(p) for p in job.glob('*.json')},'duplicate changed closed outputs'
    fail,token2=create('worker-failure',fault='formal-D0')
    assert run(fail,token2,'worker-exception')!=0
    assert read(Path(fail['job_root'])/'STOP.json')['status']=='STOP'
    assert (Path(fail['job_root'])/'SLIM_EXPORT.json').is_file()
    assert not (Path(fail['job_root'])/'infer-D0-A-emaR').exists(),'worker failure continued chain'
    # Mutate REAL generated wire records, rebuild wire hashes, require semantic
    # rejection. This is not a hand-authored successful event stream.
    rejected=[]
    for label,phase in [('missing_terminal','formal-D0'),('duplicate_terminal','formal-D0'),('epoch_skip','formal-D0'),('missing_eval','evaluation')]:
        stage=Path(success['job_root'])/phase
        identity=read(stage/'runtime-archive/identity.json')
        records=[json.loads(s) for s in (stage/'runtime-archive/progress.jsonl').read_text().splitlines()]
        if label=='missing_terminal':records.pop()
        elif label=='duplicate_terminal':records.append(copy.deepcopy(records[-1]))
        elif label=='epoch_skip':
            for r in records:
                if r['payload']['event']=='epoch_begin' and r['payload']['epoch']==2:r['payload']['epoch']=3
        else:
            ix=next(i for i,r in enumerate(records) if r['payload']['event']=='evaluation_budget_complete');records.pop(ix)
        prev=None
        for i,r in enumerate(records,1):
            r.pop('sha256',None);r['seq']=i;r['previous']=prev;r['sha256']=digest(r);prev=r['sha256']
        q=root/(label+'.jsonl');q.write_text(''.join(json.dumps(r)+'\n' for r in records))
        try:
            reader=channel.Reader(q,identity,'progress');last,_=reader.poll()
            if not reader.clock.get('_ft_terminal'):raise ValueError('terminal missing')
        except (ValueError,RuntimeError):rejected.append(label)
        else:raise AssertionError('bad lifecycle accepted: '+label)
    # Entry import is safe in spawned children; no dispatch/torch import.
    code="import runpy,sys;runpy.run_path('slice_ft_run.py',run_name='spawn_fixture');assert 'torch' not in sys.modules"
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=source,env=env,capture_output=True);assert p.returncode==0
    gate={'status':'PASS','source_sha256':source_ref['sha256'],'source_commit':manifest['commit'],
          'actual_entry':str(source/'slice_ft_run.py'),'scope':'CPU synthetic native128, 2epochs x1 update, single arm D0 with dual EMA records; original dual evaluators on 8 fake images, not science',
          'real_supervisor':True,'actual_success_closed':True,'worker_exception_stopped':True,'duplicate_launch_refused_no_changes':True,
          'cross_epoch_actual_formal':True,'A_two_budgets_actual_dual_evaluator_calls':4,'rejections':rejected,
          'declared_checkpoint_passes':endpoint['checkpoint_passes'],'checkpoint_files_written':present,
          'dual_ema_payload_counters':{'accept_C2_direct_r':accept_result['restored']['ema_r_updates'],
                                       'accept_C2_direct_c':accept_result['restored']['ema_c_updates'],
                                       'formal_endpoint_r':endpoint['dual_ema']['ema_r_updates'],
                                       'formal_endpoint_c':endpoint['dual_ema']['ema_c_updates']},
          'rng_neutral_alt_ema_updates':{'accept':accept_result['dual_ema']['rng_neutral_alt_ema_updates'],
                                         'formal':endpoint['dual_ema']['rng_neutral_alt_ema_updates']},
          'targeted_dual_ema_checks':reference(str(check_out/'DUAL_EMA_CHECK.json')),
          'fixture_checkpoint_bytes':fixture_bytes,'fixture_checkpoint_sizes':sizes,
          'export_actual':reference(Path(success['job_root'])/'SLIM_EXPORT.json'),'runs':results,'GPU_calls':0,'CUDA_initialized':False}
    publish(root/'FINAL_ENTRY_CPU_GATE.json',gate)
    print(json.dumps(gate))


if __name__=='__main__':main()
