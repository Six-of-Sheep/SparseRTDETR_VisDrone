"""Formal orchestration only; accepted worker computations stay byte-identical."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from . import span_runtime as rt,span_job as job,span_budget
from .span_external_acceptance import verify as verify_external


def policy():return rt.read(rt.ROOT/rt.FORMAL_CONFIG)
def load(path,*,preparation=False):return rt.load(path,preparation=preparation,formal=True)


def base(spec,source):
    c=rt.read(spec)
    required={'profile','seed','attempt','job_root','nonce','boot_id','gpu_uuid','storage_identity','data',
        'pretrained','prior_cost','baselines','execution_scope','external_acceptance'}
    if c.get('profile')=='LAB_GPU_ONCE':required|={'environment','historical_A','selection','input_lists'}
    rt.require(not required-set(c),'raw formal spec missing fields: '+','.join(sorted(required-set(c))))
    rt.require(c['execution_scope']==policy()['scope'] and c.get('protocol',rt.FORMAL_PROTOCOL)==rt.FORMAL_PROTOCOL,
        'formal entry refuses non-formal scope/protocol')
    rt.require('acceptance_limits' not in c and 'acceptance_sha256' not in c,'formal cannot authorize acceptance')
    bound=dict(task_id=rt.TASK,protocol=rt.FORMAL_PROTOCOL,source_root=str(rt.ROOT),python=sys.executable,
        source=source,commit=rt.verify_source(source)['commit'],science_sha256=rt.digest(rt.recipe()),
        execution_sha256=rt.digest(rt.execution()),formal_sha256=rt.digest(policy()),
        stages=policy()['stages'],resources=rt.execution()['resources'])
    for k,v in bound.items():rt.require(k not in c or c[k]==v,'raw formal spec conflicts: '+k)
    c.update(bound)
    from .span_inputs import adapt
    adapt(c);return c


def preparation_contract(spec,source,output):
    c=base(spec,source);c.update(GPU_authorized=False,science_authorized=False,authorization=None)
    path=Path(output).with_name(Path(output).name+'-contract.json');rt.seal_contract(c,path)
    return load(path,preparation=True)


def prepare(spec,source,output):return job.prepare(preparation_contract(spec,source,output),output)


def bind(spec,source,cpu_gate,authorization,output):
    c=base(spec,source);rt.require(c['profile']=='LAB_GPU_ONCE','formal bind only LAB')
    c.update(GPU_authorized=True,science_authorized=True,authorization=authorization,CPU_gate=cpu_gate)
    p=rt.exact(c['prepare_gate'])
    rt.require(p['status']=='PREPARE_PASS' and p['contract_protocol']==rt.FORMAL_PROTOCOL and p['source_sha256']==source['sha256']
        and p['data_sha256']==rt.digest(c['data']),'formal real preparation differs')
    candidate=Path(output).with_suffix('.candidate.json');rt.seal_contract(c,candidate)
    valid=load(candidate);rt.snapshot(valid,entry=True,gpu=True)
    rt.publish(output,rt.read(candidate));return rt.reference(output)


def executed_stages(c):
    return [s for s in rt.permitted_stages(c) if not(s=='train-B0' and rt.reused_B0(c))]


def reject_reexecution(c,stage):
    rt.require(not(stage=='train-B0' and rt.reused_B0(c)),'reused B0 cannot execute under new contract')


def verify(c):
    root=Path(c['job_root']);rt.require(not(root/'STOP.json').exists(),'formal root has STOP')
    closed=rt.read(root/'CLOSED.json');result=rt.exact(closed['result'])
    rt.require(closed['status']==result['status']=='PASS' and result['contract_sha256']==c['contract_sha256'],'formal root closure differs')
    rt.require(closed['external_acceptance']==c['external_acceptance'] and
        closed['stage_closures']==[rt.reference(rt.stage_path(c,s)/'CLOSED.json') for s in executed_stages(c)],'formal closure references differ')
    rt.require(closed.get('external_B0')==c.get('external_B0') and result.get('external_B0')==c.get('external_B0')
        and result['stages']==executed_stages(c),'formal completed-worker reference differs')
    rt.require(not any((root/('accept-'+a)).exists() for a in ('B0','B1','P')),'local acceptance stage forbidden')
    verify_external(c)
    for stage in rt.permitted_stages(c):rt.verify_stage(c,stage)
    inherited=rt.exact(c['prior_cost'])['forward_calls']
    rt.require(result['forward_calls_total']==(219693 if c['profile']=='LAB_GPU_ONCE' else 84) and result['inherited_forward_calls']==inherited
        and result['new_forward_calls']==result['forward_calls_total']-inherited,'formal cumulative forward result differs')
    if rt.reused_B0(c):rt.require(not(root/'train-B0').exists(),'reused B0 cannot have a new local stage')
    if c['profile']=='LAB_GPU_ONCE':
        ledger=rt.read(root/'COST_LEDGER.json')
        rt.require(ledger['active'] is None and ledger['forward_calls']==219693,'formal cumulative ledger incomplete')
    return result


def run(c):
    rt.require(c['protocol']==rt.FORMAL_PROTOCOL,'formal run protocol differs')
    root=Path(c['job_root']);root.mkdir(parents=True,exist_ok=False)
    rt.publish(root/'RUN-CONSUMED.json',dict(identity=rt.identity(c,'admission'),nonce=c['nonce'],no_retry=True))
    proof=verify_external(c)
    rt.publish(root/'EXTERNAL_ACCEPTANCE_REUSED.json',{k:v for k,v in proof.items() if k!='parent'})
    inherited=rt.exact(c['prior_cost'])['forward_calls']
    if rt.reused_B0(c):
        from .span_B0_reuse import verify as verify_B0
        verify_B0(c)
        rt.publish(root/'COMPLETED_B0_REFERENCED.json',dict(evidence=c['external_B0'],original_closure_absent=True,
            no_training_reexecution=True,no_checkpoint_copy=True))
    start=time.monotonic();done=[]
    try:
        from .cuda_identity import recorded_subprocess
        for stage in executed_stages(c):
            gpu=job.gpu_stage(c,stage)
            if gpu:span_budget.start(c,stage);span_budget.guard(c)
            rt.snapshot(c,entry=True,gpu=gpu,stage=stage)
            rt.publish(root/(stage+'-CONSUMED.json'),dict(identity=rt.identity(c,stage),no_retry=True))
            p=recorded_subprocess(root/(stage+'-supervisor-process'),job.command(c,'supervise',stage),cwd=rt.ROOT,env=job.environment(c))
            if gpu:span_budget.finish(c)
            rt.require(p.returncode==0,'formal supervised stage stopped: '+stage)
            rt.verify_stage(c,stage);done.append(stage)
        comparison=rt.verify_stage(c,'compare');calls=comparison['total_forward_calls']
        rt.publish(root/'RESULT.json',dict(status='PASS',task_id=rt.TASK,contract_sha256=c['contract_sha256'],
            source_sha256=c['source']['sha256'],profile=c['profile'],stages=done,
            external_acceptance=c['external_acceptance'],external_B0=c.get('external_B0'),inherited_forward_calls=inherited,new_forward_calls=calls-inherited,
            forward_calls_total=calls,quality_passed=comparison['quality_passed'],cost_passed=comparison['cost_passed'],
            elapsed_seconds=time.monotonic()-start,no_retry=True))
        rt.publish(root/'CLOSED.json',dict(status='PASS',result=rt.reference(root/'RESULT.json'),
            external_acceptance=c['external_acceptance'],
            external_B0=c.get('external_B0'),stage_closures=[rt.reference(rt.stage_path(c,s)/'CLOSED.json') for s in executed_stages(c)]))
        verify(c);print(json.dumps(dict(status='PASS',stages=len(done),forward_calls_total=calls,new_forward_calls=calls-inherited)));return 0
    except BaseException as exc:
        ledger=root/'COST_LEDGER.json'
        if ledger.exists() and rt.read(ledger)['active'] is not None:span_budget.finish(c)
        rt.publish(root/'STOP.json',dict(status='STOP_NO_RETRY',completed_stages=done,error=repr(exc),traceback=traceback.format_exc(),
            contract_sha256=c['contract_sha256'],elapsed_seconds=time.monotonic()-start));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','bind','run','supervise','worker','verify','lab','launch','execute','reconcile-B0'))
    for name in ('spec','source','output','cpu-gate','authorization','contract','ci','parent-spec','acceptance-contract','completed-B0-contract'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--parent-sha');p.add_argument('--attempt');p.add_argument('--session')
    p.add_argument('--stage',choices=tuple(policy()['stages']));a=p.parse_args()
    if a.action=='reconcile-B0':
        rt.require(a.completed_B0_contract is not None and a.source is not None and a.output is not None,'B0 reconciliation arguments missing')
        rt.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','B0 reconciliation must hide CUDA')
        from .span_B0_reuse import build
        build(a.completed_B0_contract,rt.reference(a.source.resolve()),a.output);return 0
    if a.action in ('lab','launch','execute'):
        from . import span_formal_admission as admission
        return admission.dispatch(a)
    if a.action=='prepare' and a.spec is not None:
        rt.require(a.source is not None and a.output is not None,'formal prepare references required')
        return prepare(a.spec,rt.reference(a.source.resolve()),a.output)
    if a.action=='bind':
        rt.require(all(x is not None for x in (a.spec,a.source,a.cpu_gate,a.authorization,a.output)),'formal bind references required')
        bind(a.spec,rt.reference(a.source.resolve()),rt.reference(a.cpu_gate.resolve()),rt.reference(a.authorization.resolve()),a.output);return 0
    rt.require(a.contract is not None,'explicit formal contract required')
    c=load(a.contract,preparation=a.action=='prepare')
    if a.action=='prepare':return job.prepare(c,a.output)
    if a.action=='run':return run(c)
    if a.action=='verify':verify(c);return 0
    rt.require(a.stage in rt.permitted_stages(c),'formal stage outside scope')
    reject_reexecution(c,a.stage)
    if a.action=='supervise':return job.supervise(c,a.stage)
    from .span_worker import worker
    return worker(c,a.stage)
