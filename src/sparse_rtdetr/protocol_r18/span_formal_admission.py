"""Tracked formal admission and single launch; inherits certified actual costs."""
from pathlib import Path
import json,os,shlex,subprocess,sys,traceback,uuid
from . import span_runtime as rt,span_job as job,span_formal as formal
from .span_external_acceptance import descriptor
from .cuda_identity import recorded_subprocess


def absent(c,session):
    rt.require(not Path(c['job_root']).exists(),'formal attempt exists; no retry')
    rt.require(subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==1,'formal tmux exists/unavailable')
    ps=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args=']).decode()
    rows=[s for s in ps.splitlines() if str(Path(c['job_root']).parent) in s and
        any(n in s for n in ('r18_sampling_formal.py run','r18_sampling_formal.py supervise','r18_sampling_formal.py worker',
                            'r18_sampling_acceptance.py run','r18_sampling_acceptance.py supervise','r18_sampling_acceptance.py worker'))]
    rt.require(not rows,'related sampling process active')


def lab(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        rt.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','formal admission must hide CUDA')
        source=rt.reference(a.source.resolve());m=rt.verify_source(source);ci=rt.read(a.ci)
        rt.require(ci['head']==m['commit'] and ci['state']=='open' and not ci['merged'] and len(ci['checks'])==4 and
            all(x['head_sha']==m['commit'] and x['status']=='completed' and x['conclusion']=='success' for x in ci['checks']), 'formal exact head CI incomplete')
        pref=rt.reference(a.parent_spec.resolve());rt.require(pref['sha256']==a.parent_sha,'formal parent raw spec differs')
        c=rt.exact(pref);rt.require(c['profile']=='LAB_GPU_ONCE','formal lab only real profile')
        task=out.parent
        rt.require(task==Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution')/rt.TASK,'formal task root differs')
        rt.require(not list(task.glob('attempt-formal-*')),'previous formal attempt exists; cost reconciliation required')
        d=descriptor(a.acceptance_contract.resolve());rt.publish(out/'external-acceptance.json',d)
        ledger=rt.exact(d['cost_ledger'])
        rt.publish(out/'PRIOR_COST.json',dict(task_id=rt.TASK,profile='LAB_GPU_ONCE',GPU_seconds=ledger['completed_GPU_seconds'],
            forward_calls=ledger['forward_calls'],acceptance_ledger=d['cost_ledger'],external_acceptance=rt.reference(out/'external-acceptance.json'),
            previous_CPU_STOPs_preserved=True,no_refund=True))
        from ..localization.thermal_resume import snapshot
        sample=snapshot(str(out));rt.publish(out/'resource-observation.json',sample)
        c.update(protocol=rt.FORMAL_PROTOCOL,execution_scope=formal.policy()['scope'],external_acceptance=rt.reference(out/'external-acceptance.json'),
            prior_cost=rt.reference(out/'PRIOR_COST.json'),attempt=a.attempt,job_root=str(task/a.attempt),nonce=uuid.uuid4().hex,
            boot_id=sample['boot_id'],storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')})
        rt.publish(out/'raw-spec.json',c)
        built=formal.preparation_contract(out/'raw-spec.json',source,out/'construction')
        rt.publish(out/'SOURCE_REUSE_PROOF.json',{k:v for k,v in built['_verified_external'].items() if k!='parent'})
        absent(built,a.session);rt.snapshot(built,entry=True,gpu=True)
        env=job.environment(built)
        cmd=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_formal.py'),'prepare','--spec',str(out/'raw-spec.json'),'--source',source['path'],'--output',str(out/'prepare')]
        proc=recorded_subprocess(out/'prepare-process',cmd,cwd=rt.ROOT,env=env)
        rt.require(proc.returncode==0,'formal preparation STOP; no retry')
        c['prepare_gate']=rt.reference(out/'prepare/PREPARE_PASS.json')
        proc=recorded_subprocess(out/'cuda-preflight-process',[sys.executable,'-B',str(rt.ROOT/'r18_sampling_span.py'),'cuda-preflight','--gpu-uuid',c['gpu_uuid'],'--output',str(out/'cuda-preflight')],cwd=rt.ROOT,env=job.environment(built,True))
        rt.require(proc.returncode==0,'formal CUDA preflight STOP; no retry')
        rt.snapshot(built,entry=True,gpu=True);rt.publish(out/'admitted-spec.json',c)
        b=formal.base(out/'admitted-spec.json',source)
        auth=dict(scope=formal.policy()['scope'],task_id=rt.TASK,attempt=c['attempt'],commit=b['commit'],source_sha256=source['sha256'],
            science_sha256=b['science_sha256'],execution_sha256=b['execution_sha256'],nonce=c['nonce'],formal_sha256=b['formal_sha256'],
            external_acceptance_sha256=c['external_acceptance']['sha256'],remaining_forward_calls=219675)
        rt.publish(out/'authorization.json',auth)
        proc=recorded_subprocess(out/'bind-process',[sys.executable,'-B',str(rt.ROOT/'r18_sampling_formal.py'),'bind','--spec',str(out/'admitted-spec.json'),
            '--source',source['path'],'--cpu-gate',str(a.cpu_gate.resolve()),'--authorization',str(out/'authorization.json'),'--output',str(out/'contract.json')],cwd=rt.ROOT,env=env)
        rt.require(proc.returncode==0,'formal bind STOP; no retry')
        rt.publish(out/'ADMISSION_PASS.json',dict(status='PASS',contract=rt.reference(out/'contract.json'),source=source,session=a.session,
            external_acceptance=c['external_acceptance'],prior_cost=c['prior_cost'],CPU_forward_calls=0,remaining_forward_calls=219675))
        print(json.dumps(rt.read(out/'ADMISSION_PASS.json')));return 0
    except BaseException as exc:
        rt.publish(out/'ADMISSION_STOP.json',dict(status='STOP_NO_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise


def launch(contract,session):
    c=formal.load(contract);out=Path(contract).parent;g=rt.read(out/'ADMISSION_PASS.json')
    rt.require(c['profile']=='LAB_GPU_ONCE' and g['contract']==rt.reference(contract) and g['session']==session,'formal launch admission differs')
    absent(c,session);rt.snapshot(c,entry=True,gpu=True)
    cmd=[sys.executable,'-B',str(rt.ROOT/'r18_sampling_formal.py'),'execute','--contract',str(contract),'--session',session]
    rt.publish(out/'LAUNCH_CONSUMED.json',dict(contract=rt.reference(contract),command=cmd,session=session,no_retry=True))
    p=recorded_subprocess(out/'tmux-start-process',['tmux','new-session','-d','-s',session,'-c',str(rt.ROOT),
        'exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.join(cmd)],cwd=rt.ROOT,env=job.environment(c))
    rt.require(p.returncode==0,'formal tmux launch STOP; no retry')
    print(json.dumps(dict(status='STARTED',session=session,contract=rt.reference(contract))));return 0


def execute(contract,session):
    c=formal.load(contract);out=Path(contract).parent;l=rt.read(out/'LAUNCH_CONSUMED.json')
    rt.require(l['contract']==rt.reference(contract) and l['session']==session,'formal execute token differs')
    code=2
    try:
        p=recorded_subprocess(out/'formal-run-process',[sys.executable,'-B',str(rt.ROOT/'r18_sampling_formal.py'),'run','--contract',str(contract)],cwd=rt.ROOT,env=job.environment(c))
        code=p.returncode
        if code==0:formal.verify(c)
        return code
    except BaseException:code=2;raise
    finally:
        rt.publish(out/'EXECUTION_EXIT.json',dict(status='PASS' if code==0 else 'STOP_NO_RETRY',returncode=code,contract=rt.reference(contract)))
        subprocess.run(['tmux','wait-for','-S',session+'-exit'],check=True)


def dispatch(a):
    needed=('parent_spec','parent_sha','source','cpu_gate','ci','attempt','session','output','acceptance_contract') if a.action=='lab' else ('contract','session')
    rt.require(all(getattr(a,n) is not None for n in needed),'formal admission arguments missing')
    if a.action=='lab':return lab(a)
    return (launch if a.action=='launch' else execute)(a.contract.resolve(),a.session)
