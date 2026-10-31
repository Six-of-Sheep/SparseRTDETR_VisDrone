"""Drive the final three-arm detail-path executable through real supervisors and dual evaluators."""
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
           'plan':str(source/'configs/research/r18_detail_arms_v1.json'),'plan_sha256':execution.PLAN_SHA,
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
        if phase=='evaluation':assert r['score_budget_calls']==6
        q=Path(success['job_root'])/phase/'runtime-archive/supervisor-exit.json';e=read(q)
        assert e['status']=='PASS' and not e['remaining_live'] and not e['unknown_pids']
    before={str(p):reference(p) for p in Path(success['job_root']).glob('*.json')}
    assert run(success,token,'duplicate-launch')!=0
    assert before=={str(p):reference(p) for p in Path(success['job_root']).glob('*.json')},'duplicate changed closed outputs'
    fail,token2=create('worker-failure',fault='accept-G1')
    assert run(fail,token2,'worker-exception')!=0
    assert read(Path(fail['job_root'])/'STOP.json')['status']=='STOP'
    assert (Path(fail['job_root'])/'SLIM_EXPORT.json').is_file()
    assert not (Path(fail['job_root'])/'accept-G2').exists(),'worker failure continued chain'
    # Mutate REAL generated wire records, rebuild wire hashes, require semantic
    # rejection. This is not a hand-authored successful event stream.
    rejected=[]
    for label,phase in [('missing_terminal','formal-G0'),('duplicate_terminal','formal-G0'),('epoch_skip','formal-G0'),('missing_eval','evaluation')]:
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
          'actual_entry':str(source/'slice_ft_run.py'),'scope':'CPU synthetic native128, 2epochs x1 update, arms G0/G1/G2; original dual evaluators on8 fake images, not science',
          'real_supervisor':True,'actual_success_closed':True,'worker_exception_stopped':True,'duplicate_launch_refused_no_changes':True,
          'cross_epoch_actual_formal':True,'A_two_budgets_actual_dual_evaluator_calls':6,'rejections':rejected,
          'export_actual':reference(Path(success['job_root'])/'SLIM_EXPORT.json'),'runs':results,'GPU_calls':0,'CUDA_initialized':False}
    publish(root/'FINAL_ENTRY_CPU_GATE.json',gate)
    print(json.dumps(gate))


if __name__=='__main__':main()
