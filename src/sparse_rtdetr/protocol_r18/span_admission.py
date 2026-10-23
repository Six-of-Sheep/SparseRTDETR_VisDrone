"""One-attempt admission from raw spec, explicit cost, CI and frozen source.

No GPU operations in construction check; lab prepares, preflights and binds.
Only the acceptance CLI is launchable. No retry, formal dispatch or overrides.
"""
import argparse,json,os,shlex,subprocess,sys,traceback,uuid
from pathlib import Path
from . import span_runtime as rt, span_acceptance as ac, span_job as job
from .cuda_identity import recorded_subprocess


def checked(spec,source,output,observation=None):
    c=ac.preparation_contract(spec,source,output)
    rt.require(c['resources']==rt.execution()['resources'],'constructed resources differ')
    if observation is not None:
        rt.require(c['profile']=='CPU_FIXTURE_ONLY','resource fixture forbidden for production')
        sample=rt.screen(rt.read(observation),c,entry=True,gpu=False)
    else:sample=rt.snapshot(c,entry=True,gpu=False)
    rt.publish(Path(output).with_name(Path(output).name+'-construction.json'),dict(
        status='CONSTRUCTION_RESOURCE_PASS',source=source,contract=rt.reference(c['_path']),
        resources_sha256=rt.digest(c['resources']),resource_sample=sample,
        construction_before_screen=True,CUDA_initialized=False,model_forward_calls=0))
    return c


def check(spec,source,output,observation=None):
    rt.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','construction check requires hidden CUDA')
    c=checked(spec,source,output,observation)
    import torch
    rt.require(not torch.cuda.is_initialized(),'construction check initialized CUDA')
    print(json.dumps(dict(status='CONSTRUCTION_RESOURCE_PASS',profile=c['profile'],CUDA_initialized=False)))
    return 0


def absent(c,session):
    rt.require(not Path(c['job_root']).exists(),'attempt already exists; no retry')
    rt.require(subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==1,'tmux exists/unavailable')
    ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args=']).decode()
    related=[s for s in ps.splitlines() if str(Path(c['job_root']).parent) in s and
             any(n in s for n in ('r18_sampling_acceptance.py','r18_sampling_span.py'))]
    rt.require(not related,'related sampling process already exists')


def lab(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        rt.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','lab admission requires hidden CUDA')
        source=rt.reference(args.source.resolve());manifest=rt.verify_source(source)
        ci=rt.read(args.ci)
        rt.require(ci['head']==manifest['commit'] and ci['state']=='open' and not ci['merged'] and
            len(ci['checks'])==4 and all(x['head_sha']==manifest['commit'] and x['status']=='completed' and x['conclusion']=='success' for x in ci['checks']),
            'exact head four successful CI checks required')
        parent=rt.reference(args.parent_spec.resolve())
        rt.require(parent['sha256']==args.parent_sha,'parent raw spec SHA conflict')
        c=rt.exact(parent)
        rt.require(c['profile']=='LAB_GPU_ONCE','lab cannot accept fixture')
        task=out.parent
        rt.require(task.name==rt.TASK and task==Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution')/rt.TASK,'lab metadata root differs')
        # All prior task GPU attempts must be absent for this first-GPU binder.
        # A later task with consumed GPU cost needs explicit reviewed accounting.
        prior_attempts=list(task.glob('attempt-gpu-*'))
        rt.require(not prior_attempts,'previous GPU attempt exists: cost reconciliation required')
        prior=rt.exact(c['prior_cost'])
        rt.require(prior['task_id']==rt.TASK and prior['GPU_seconds']==0 and prior['forward_calls']==0,'prior GPU cost must be explicitly reconciled')
        stops=[]
        for p in args.prior_stop:
            ref=rt.reference(p.resolve());row=rt.exact(ref)
            rt.require(row['status']=='STOP_NO_RETRY','prior STOP identity/status differs')
            stops.append(ref)
        rt.require(stops,'explicit previous STOP evidence required')
        rt.publish(out/'PRIOR_COST.json',dict(task_id=rt.TASK,profile='LAB_GPU_ONCE',GPU_seconds=0,forward_calls=0,
            previous_receipt=c['prior_cost'],previous_STOP=stops,previous_GPU_attempts=[],
            CPU_costs_preserved_separately=True,reason='no prior task GPU attempt/allocation; preserved STOPs before GPU'))
        from ..localization.thermal_resume import snapshot
        sample=snapshot(str(out));rt.publish(out/'resource-observation.json',sample)
        c.update(protocol=rt.ACCEPTANCE_PROTOCOL,execution_scope='GPU_ACCEPTANCE_ONLY',
            attempt=args.attempt,job_root=str(task/args.attempt),nonce=uuid.uuid4().hex,
            boot_id=sample['boot_id'],storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')},
            prior_cost=rt.reference(out/'PRIOR_COST.json'))
        rt.publish(out/'raw-spec.json',c)
        # Same constructor as prepare, no field-filled fixture or alternate dict.
        constructed=checked(out/'raw-spec.json',source,out/'construction')
        absent(constructed,args.session)
        rt.snapshot(constructed,entry=True,gpu=True)
        env=job.environment(constructed)
        cmd=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_acceptance.py'),'prepare','--spec',str(out/'raw-spec.json'),'--source',source['path'],'--output',str(out/'prepare')]
        p=recorded_subprocess(out/'prepare-process',cmd,cwd=rt.ROOT,env=env)
        rt.require(p.returncode==0,'real preparation stopped; no retry')
        c['prepare_gate']=rt.reference(out/'prepare/PREPARE_PASS.json')
        cmd=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_span.py'),'cuda-preflight','--gpu-uuid',c['gpu_uuid'],'--output',str(out/'cuda-preflight')]
        p=recorded_subprocess(out/'cuda-preflight-process',cmd,cwd=rt.ROOT,env=job.environment(constructed,True))
        rt.require(p.returncode==0,'CUDA preflight stopped; no retry')
        rt.snapshot(constructed,entry=True,gpu=True)
        rt.publish(out/'admitted-spec.json',c)
        built=ac.base(out/'admitted-spec.json',source)
        approval=dict(scope='GPU_ACCEPTANCE_ONLY',task_id=rt.TASK,attempt=c['attempt'],commit=built['commit'],
            source_sha256=source['sha256'],science_sha256=built['science_sha256'],execution_sha256=built['execution_sha256'],
            acceptance_sha256=built['acceptance_sha256'],limits=built['acceptance_limits'],nonce=c['nonce'])
        rt.publish(out/'authorization.json',approval)
        cmd=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_acceptance.py'),'bind','--spec',str(out/'admitted-spec.json'),'--source',source['path'],'--cpu-gate',str(args.cpu_gate.resolve()),'--authorization',str(out/'authorization.json'),'--output',str(out/'contract.json')]
        p=recorded_subprocess(out/'bind-process',cmd,cwd=rt.ROOT,env=env)
        rt.require(p.returncode==0,'acceptance bind stopped; no retry')
        rt.publish(out/'ADMISSION_PASS.json',dict(status='PASS',contract=rt.reference(out/'contract.json'),source=source,
            parent_spec=parent,CI=rt.reference(args.ci.resolve()),session=args.session,model_forward_calls=0,
            scope='GPU_ACCEPTANCE_ONLY',prior_cost=rt.reference(out/'PRIOR_COST.json')))
        print(json.dumps(rt.read(out/'ADMISSION_PASS.json')));return 0
    except BaseException as exc:
        rt.publish(out/'ADMISSION_STOP.json',dict(status='STOP_NO_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise


def launch(contract,session):
    c=ac.load(contract);rt.require(c['profile']=='LAB_GPU_ONCE','launch cannot accept fixture')
    out=Path(contract).parent;gate=rt.read(out/'ADMISSION_PASS.json')
    rt.require(gate['contract']==rt.reference(contract) and gate['session']==session,'launch admission binding differs')
    absent(c,session);rt.snapshot(c,entry=True,gpu=True)
    command=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_admission.py'),'execute','--contract',str(contract),'--session',session]
    rt.publish(out/'LAUNCH_CONSUMED.json',dict(contract=rt.reference(contract),command=command,session=session,no_retry=True))
    shell='exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join(command)
    p=recorded_subprocess(out/'tmux-start-process',['tmux','new-session','-d','-s',session,'-c',str(rt.ROOT),shell],cwd=rt.ROOT,env=job.environment(c))
    rt.require(p.returncode==0,'tmux launch stopped; no retry')
    print(json.dumps(dict(status='STARTED',session=session,contract=rt.reference(contract))));return 0


def execute(contract,session):
    c=ac.load(contract);out=Path(contract).parent;launch=rt.read(out/'LAUNCH_CONSUMED.json')
    rt.require(launch['contract']==rt.reference(contract) and launch['session']==session,'execute launch binding differs')
    code=2
    try:
        p=recorded_subprocess(out/'acceptance-run-process',[sys.executable,'-B',str(rt.ROOT/'r18_sampling_acceptance.py'),'run','--contract',str(contract)],cwd=rt.ROOT,env=job.environment(c))
        code=p.returncode
        if code==0:ac.verify(c)
        return code
    except BaseException:
        code=2
        raise
    finally:
        rt.publish(out/'EXECUTION_EXIT.json',dict(returncode=code,status='PASS' if code==0 else 'STOP_NO_RETRY',contract=rt.reference(contract),scientific_quality_evaluated=False))
        subprocess.run(['tmux','wait-for','-S',session+'-exit'],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','screen','lab','launch','execute'))
    for name in ('spec','source','output','ci','parent-spec','cpu-gate','contract','resource-observation'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--parent-sha');p.add_argument('--attempt');p.add_argument('--session')
    p.add_argument('--prior-stop',type=Path,action='append')
    a=p.parse_args()
    needed={'check':('spec','source','output'),'screen':('contract',),'lab':('parent_spec','parent_sha','source','cpu_gate','ci','attempt','session','output','prior_stop'),
            'launch':('contract','session'),'execute':('contract','session')}[a.action]
    rt.require(all(getattr(a,n) is not None for n in needed),'explicit admission arguments missing')
    if a.action=='check':return check(a.spec,rt.reference(a.source.resolve()),a.output,a.resource_observation)
    if a.action=='screen':
        rt.require('contract_sha256' in rt.read(a.contract),'resource screen requires constructed contract')
        c=ac.load(a.contract,preparation=True);rt.snapshot(c,entry=True,gpu=False);return 0
    if a.action=='lab':return lab(a)
    return (launch if a.action=='launch' else execute)(a.contract.resolve(),a.session)
