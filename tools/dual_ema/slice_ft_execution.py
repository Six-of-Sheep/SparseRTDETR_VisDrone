"""Frozen single-use dual-EMA recording execution, guard/supervisor and slim closeout."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from ..localization.runio import require,read,publish,reference,sha,digest,event
from ..localization import channel,guard
from . import slice_ft_progress as progress

TASK='r18-dual-ema-20261031-001'
ROOT=Path(__file__).resolve().parents[3]
PLAN_SHA='d0a3593d14878bf0e0d400e68e35fdec40c33dd742aceae0ab7dd78c4714c11c'
PHASES=('accept-D0','formal-D0','infer-D0-A-emaR','infer-D0-A-emaC','evaluation')


def install():
    # The native install must precede the FT Reader installation in EVERY process.
    from .control import install_resources
    install_resources();progress.install()


def verified(ref):
    require(reference(ref['path'])==ref,'FT bound reference changed: '+str(ref['path']))
    return read(ref['path'])


def load(path):
    install();c=read(path)
    require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'FT contract digest')
    require(c['task_id'].startswith(TASK) and c['contract_path']==str(Path(path).resolve()),'FT task/path identity')
    require(c['source_root']==str(ROOT) and sha(c['source_manifest'])==c['source_sha256'],'FT source root/manifest')
    manifest=read(c['source_manifest'])
    require(manifest['commit']==c['source_commit'],'FT Git/source binding')
    for row in manifest['source_files']:
        q=Path(row['path']);require(not q.is_absolute() and '..' not in q.parts,'FT source path escaped')
        target=ROOT/q
        require(not target.is_symlink() and target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],'FT source changed: '+str(q))
    require(sha(c['plan'])==c['plan_sha256']==PLAN_SHA,'FT frozen science plan changed')
    for key in ('job_root','runtime_root'):
        p=Path(c[key]);require(p.is_absolute() and not p.is_symlink(),'FT absolute root required')
    require(c['scope'] in ('CPU_FIXTURE','PRODUCTION'),'FT scope absent')
    if c['scope']=='CPU_FIXTURE':
        require(c['GPU_authorized'] is False and c['formal_authorized'] is False and c['epochs']==2 and c['updates_per_epoch']==1 and c['total_updates']==2,'CPU fixture authority expanded')
        require(len(c['development_ids'])==len(c['fixed64'])==8,'CPU fixture membership changed')
    else:
        require(c['GPU_authorized'] is True and c['formal_authorized'] is True and c['epochs']==12 and c['updates_per_epoch']==304 and c['total_updates']==3648,'FT production authority/budget')
        require(len(c['development_ids'])==len(set(c['development_ids']))==548 and len(c['fixed64'])==len(set(c['fixed64']))==64 and set(c['fixed64'])<=set(c['development_ids']),'FT development/fixed64 identity')
        g=verified(c['CPU_gate']);require(g['status']=='PASS' and g['source_sha256']==c['source_sha256'],'FT final entry CPU source gate')
        for r in c['materials']:verified(r)
        for name in ('original_G0','parent'):
            for b in ('600','300'):verified(c['reference_metrics'][name]['A'][b])
        require(c['budget_limits']['gpu_hours']==2.0 and c['budget_limits']['wall_hours']==24 and c['budget_limits']['server_new_bytes']==8589934592,'FT round budget declaration changed')
    return c


def phase_arm(phase):
    return 'paired' if phase=='evaluation' else phase.split('-')[1]


def complete(c,phase):
    root=Path(c['job_root'])/phase;closed=read(root/'CLOSED.json')
    require(closed['status']=='PASS' and closed['phase']==phase and closed['contract_sha256']==c['contract_sha256'] and closed['source_sha256']==c['source_sha256'],'FT phase closure identity')
    result=verified(closed['result']);verified(closed['runtime_index'])
    return result


def prerequisites(c,phase):
    require(phase in PHASES,'unknown FT phase')
    for prev in PHASES[:PHASES.index(phase)]:
        result=complete(c,prev)
        if prev.startswith('accept-'):
            require(result['actual_updates']==3 and result['sequence']==[1,2,2] and result['full_state_RNG_exact'] and result['input_DN_output_loss_preclip_gradient_exact'] and result['state_usable_for_formal'] is False,'FT acceptance gate not exact')


def identity(c,phase):
    return {**{k:c[k] for k in ('task_id','source_sha256','contract_sha256','plan_sha256','epochs','updates_per_epoch','total_updates','development_ids','fixed64')},
            'mode':phase,'phase':phase,'arm':phase_arm(phase),'progress_protocol':progress.PROTOCOL,
            'output_root':str(Path(c['job_root'])/phase),'evaluation_images':len(c['development_ids']),
            'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']}


def supervise(c,phase):
    install();prerequisites(c,phase)
    out=Path(c['job_root'])/phase;runtime=Path(c['runtime_root'])/phase
    gpu=c['scope']=='PRODUCTION' and phase!='evaluation'
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2',
         'MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU',
         'CUBLAS_WORKSPACE_CONFIG':':4096:8','CUDA_VISIBLE_DEVICES':c['gpu_uuid'] if gpu else ''}
    cmd=[c['python'],'-B',str(ROOT/'slice_ft_run.py'),'worker','--contract',c['contract_path'],'--phase',phase]
    code=guard.supervise(cmd,out,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=gpu,
          checkpoint_bytes=512*1024**2 if phase.startswith(('accept-','formal-')) else 0,
          formal=phase.startswith('formal-'),runtime=runtime,run_identity=identity(c,phase))
    b=read(runtime/'identity.json');channel.verify_archive(runtime,out,b)
    exit_record=read(runtime/'supervisor-exit.json')
    require(code==0 and exit_record['status']=='PASS' and not exit_record['remaining_live'] and not exit_record['unknown_pids'] and not exit_record['cleanup_errors'],'FT worker/supervisor exit not clean')
    r=channel.Reader(out/'runtime-archive/progress.jsonl',b,'progress');last,_=r.poll()
    require(last is not None and last['payload']['event']=='phase_complete' and r.clock.get('_ft_terminal'),'FT terminal operation absent')
    publish(out/'CLOSED.json',{'status':'PASS','phase':phase,'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],
         'result':reference(out/'RESULT.json'),'runtime_index':reference(out/'runtime-archive-index.json'),
         'supervisor_exit':reference(runtime/'supervisor-exit.json')})


def worker(c,phase):
    import torch
    install();prerequisites(c,phase)
    b=channel.binding();out=Path(c['job_root'])/phase
    require(b is not None and b['phase']==phase and b['arm']==phase_arm(phase) and all(b[k]==c[k] for k in ('task_id','source_sha256','contract_sha256')),'FT worker identity absent')
    require(read(Path(b['runtime_root'])/'identity.json')==b and os.getppid()==b['supervisor_pid'] and guard.identity(os.getppid())['start']==b['supervisor_start'],'FT worker parent identity differs')
    publish(out/'execution-binding.json',b);holder={}
    try:
        if c['scope']=='CPU_FIXTURE':require(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized(),'CPU fixture initialized CUDA')
        require(c.get('fixture_worker_error')!=phase,'injected CPU worker exception')
        from . import slice_ft_job as job
        if phase.startswith('accept-'):job.acceptance(c,out,phase_arm(phase),holder)
        elif phase.startswith('formal-'):job.formal(c,out,phase_arm(phase),holder)
        elif phase.startswith('infer-'):job.inference(c,out,phase,holder)
        else:job.evaluation(c,out)
        event(out,'phase_complete',result=reference(out/'RESULT.json'))
        publish(channel.receipt_root(out)/'worker-exit.json',{'success':True,'status':'PASS','pid':os.getpid()})
    except BaseException as error:
        evidence={'status':'STOP','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False}
        try:
            from .failure import preserve
            evidence['forensic']=preserve(holder,out)
        except BaseException as e:evidence['forensic_error']=repr(e)
        publish(out/'STOP.json',evidence)
        publish(channel.receipt_root(out)/'failure.json',{'status':'STOP','phase':phase,'error':repr(error)})
        publish(channel.receipt_root(out)/'worker-exit.json',{'success':False,'status':'STOP','pid':os.getpid()})
        raise


def export(c):
    root=Path(c['job_root']);paths=[]
    for name in ('CONSUMED.json','RESULT.json','STOP.json','CHAIN_EXIT.json','PREFLIGHT.json','LEDGER.json'):
        if (root/name).is_file():paths.append(root/name)
    # Small reports/receipts only; full runtime/audits/predictions/weights stay remote.
    for phase in PHASES:
        q=root/phase
        for name in ('RESULT.json','CLOSED.json','STOP.json','endpoint-lock.json','binding.json','COMPARISON.json','TABLE.csv'):
            if (q/name).is_file():paths.append(q/name)
        for q2 in sorted(q.glob('*.pth.complete.json')):paths.append(q2)
        if phase=='evaluation':
            for q2 in sorted(q.glob('*-metrics.json')):paths.append(q2)
        for name in ('supervisor-exit.json','worker-exit.json','supervisor-start.json'):
            q2=q/'runtime-archive'/name
            if q2.is_file():paths.append(q2)
    records=[reference(q) for q in paths]
    require(all(r['bytes']<2*1024**2 and not r['path'].endswith(('.pth','.npz')) for r in records),'FT export exceeded slim whitelist')
    publish(root/'SLIM_EXPORT.json',{'schema':'FT-slim-v1','records':records,'bytes':sum(r['bytes'] for r in records),
           'large_state_location':'only remote; index is not a backup','source_sha256':c['source_sha256']})


def launch(c,token_path):
    root=Path(c['job_root'])
    require(not root.exists() and not Path(c['runtime_root']).exists(),'FT duplicate output/runtime refused')
    token=read(token_path)
    require(token=={'task_id':c['task_id'],'contract_sha256':c['contract_sha256'],'nonce':c['launch_nonce']},'FT launch credential differs')
    root.mkdir(parents=True,exist_ok=False)
    publish(root/'CONSUMED.json',{'task_id':c['task_id'],'contract_sha256':c['contract_sha256'],'nonce':c['launch_nonce'],'pid':os.getpid(),'epoch':time.time(),'automatic_retry':False})
    started=time.monotonic();code=0
    try:
        s=guard.snapshot(root);guard.screen(s,gpu_uuid=c['gpu_uuid'],entry=True,gpu=c['scope']=='PRODUCTION',disk_bytes=100*1024**3)
        require(s['boot_id']==c['boot_id'],'FT new boot before launch')
        publish(root/'PREFLIGHT.json',s)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MKL_THREADING_LAYER':'GNU','PYTHONDONTWRITEBYTECODE':'1'}
        for phase in PHASES:
            prerequisites(c,phase)
            cmd=[c['python'],'-B',str(ROOT/'slice_ft_run.py'),'supervise','--contract',c['contract_path'],'--phase',phase]
            with (root/(phase+'-supervise.log')).open('x') as log:r=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            require(r.returncode==0,'FT phase failed: '+phase)
            complete(c,phase)
        result=complete(c,'evaluation')
        publish(root/'RESULT.json',{'status':'COMPLETE','quality':result['quality'],'cost_passed':result['cost_passed'],'evaluation':reference(root/'evaluation/RESULT.json'),'elapsed_seconds':time.monotonic()-started,'GPU_acceptance_actual_updates':0 if c['scope']=='CPU_FIXTURE' else 3,'formal_updates':0 if c['scope']=='CPU_FIXTURE' else c['total_updates'],'CPU_fixture':c['scope']=='CPU_FIXTURE','no_auto_next':True})
    except BaseException as e:
        code=2
        publish(root/'STOP.json',{'status':'STOP','error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False,'elapsed_seconds':time.monotonic()-started})
    publish(root/'CHAIN_EXIT.json',{'returncode':code,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-started,'no_retry':True})
    export(c)
    return code
