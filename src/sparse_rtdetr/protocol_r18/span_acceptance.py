"""Independent engineering acceptance contract and four-stage CLI lifecycle.

Shares frozen acceptance computation/supervision, never the full run/bind/export.
No formal training, scientific scoring, or complete timing dispatch exists here.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
from . import span_runtime as rt, span_job as job, span_budget, span_acceptance_budget as counts


def load(path, *, preparation=False):
    return rt.load(path, preparation=preparation, acceptance=True)


def base(spec, source):
    c = rt.read(spec); config = rt.acceptance_config()
    required = {'profile','seed','attempt','job_root','nonce','boot_id','gpu_uuid',
                'storage_identity','data','pretrained','prior_cost','baselines','execution_scope'}
    if c.get('profile') == 'LAB_GPU_ONCE':
        required |= {'environment','historical_A','selection','input_lists'}
    rt.require(not required-set(c), 'raw acceptance spec missing fields: '+','.join(sorted(required-set(c))))
    rt.require(c.get('execution_scope') == config['scope'] and
        c.get('protocol', rt.ACCEPTANCE_PROTOCOL) == rt.ACCEPTANCE_PROTOCOL,
        'acceptance entry refuses full-chain spec/contract')
    rt.require('acceptance_limits' not in c or c['acceptance_limits'] == config['limits'],
               'acceptance limits cannot be changed')
    bound = dict(task_id=rt.TASK, protocol=rt.ACCEPTANCE_PROTOCOL, source_root=str(rt.ROOT),
        python=sys.executable, source=source, commit=rt.verify_source(source)['commit'],
        science_sha256=rt.digest(rt.recipe()), execution_sha256=rt.digest(rt.execution()),
        acceptance_sha256=rt.digest(config), acceptance_limits=config['limits'],
        stages=config['stages'], resources=rt.execution()['resources'])
    for key, value in bound.items():
        rt.require(key not in c or c[key] == value, 'raw acceptance spec conflicts with frozen field: '+key)
    c.update(bound)
    from .span_inputs import adapt
    adapt(c)
    return c


def prepare(spec, source, output):
    c = preparation_contract(spec, source, output)
    return job.prepare(c, output)


def preparation_contract(spec, source, output):
    """Only construction path used by prepare and admission before screening."""
    c = base(spec, source)
    c.update(GPU_authorized=False, science_authorized=False, authorization=None)
    path = Path(output).resolve().with_name(Path(output).name+'-contract.json')
    rt.seal_contract(c, path)
    return load(path, preparation=True)


def bind(spec, source, gate, approval, output):
    c = base(spec, source)
    rt.require(c['profile'] == 'LAB_GPU_ONCE', 'GPU acceptance bind requires admitted lab profile')
    c.update(GPU_authorized=True, science_authorized=False, authorization=approval, CPU_gate=gate)
    prepared = rt.exact(c['prepare_gate'])
    rt.require(prepared['status'] == 'PREPARE_PASS' and prepared['profile'] == 'LAB_GPU_ONCE' and
        prepared['contract_protocol'] == rt.ACCEPTANCE_PROTOCOL and
        prepared['source_sha256'] == source['sha256'] and prepared['data_sha256'] == rt.digest(c['data']),
        'acceptance CPU preparation binding absent/conflicting')
    candidate = Path(output).with_suffix('.candidate.json')
    rt.seal_contract(c, candidate)
    valid = load(candidate)
    rt.snapshot(valid, entry=True, gpu=True)
    rt.publish(output, rt.read(candidate))
    return rt.reference(output)


def verify(c, *, pending_export=False):
    root = Path(c['job_root'])
    rt.require(not (root/'STOP.json').exists(), 'acceptance root has STOP')
    closed = rt.read(root/('ENGINEERING_CLOSED.json' if pending_export else 'CLOSED.json'))
    rt.require(closed['status'] == 'PASS' and closed['identity'] == rt.identity(c, 'admission'), 'acceptance root closure differs')
    result = rt.exact(closed['result'])
    rt.require(result['status'] == 'PASS' and result['execution_scope'] == 'GPU_ACCEPTANCE_ONLY' and
        result['scientific_quality_evaluated'] is False and result['actual_updates'] == 9 and
        result['forward_calls'] == 18 and result['contract_sha256'] == c['contract_sha256'],
        'acceptance terminal result differs')
    rt.require(closed['stage_closures'] == [rt.reference(rt.stage_path(c,s)/'CLOSED.json') for s in rt.permitted_stages(c)],
        'acceptance stage closure references differ')
    for stage in rt.permitted_stages(c): rt.verify_stage(c, stage)
    counts.verify_complete(c)
    rt.require(result['counts'] == rt.reference(counts.path(c)), 'acceptance final count bytes differ')
    rt.exact(result['cost'])
    if not pending_export:
        rt.exact(closed['export_index'])
    rt.require(not any((root/s).exists() for s in rt.STAGES[4:]), 'unauthorized stage directory exists')
    return result


def export(c, output, *, pending_export=False):
    result = verify(c,pending_export=pending_export); target = Path(output).resolve(); target.mkdir(parents=True, exist_ok=False)
    root = Path(c['job_root'])
    sources = [(n,root/n) for n in ('RESULT.json','ENGINEERING_CLOSED.json','ENGINEERING_COST.json','ACCEPTANCE_COUNTS.json')]
    if not pending_export:sources.append(('CLOSED.json',root/'CLOSED.json'))
    sources += [('contract.json',Path(c['_path'])),('source-manifest.json',Path(c['source']['path'])),
        ('acceptance-policy.json',rt.ROOT/rt.ACCEPTANCE_CONFIG)]
    observations = {}; checkpoints = {}
    for stage in rt.permitted_stages(c):
        for name in ('RESULT.json','CLOSED.json','supervisor-exit.json'):
            sources.append((stage+'-'+name,rt.stage_path(c,stage)/name))
        if stage.startswith('accept-'):
            stage_result = rt.verify_stage(c,stage); checkpoints[stage] = stage_result['checkpoint']
            events = rt.verify_events(c,stage,rt.stage_path(c,stage)/'events.jsonl')
            observations[stage] = [dict(event=e['event'],observations=e['fields']['result']['observations'])
                for e in events if e['event'] in ('update-1','update-2')]
            rt.require(all({r['group'] for r in row['observations']} == {'ordinary','DN'} for row in observations[stage]),
                       'ordinary/DN mechanism summary missing')
    files = []
    for name, source in sources:
        p = target/name; p.write_bytes(source.read_bytes())
        rt.require(rt.sha(p) == rt.sha(source), 'acceptance export SHA differs'); files.append(rt.reference(p))
    rt.publish(target/'mechanism-summary.json', observations)
    files.append(rt.reference(target/'mechanism-summary.json'))
    rt.publish(target/'SHA_INDEX.json',dict(files=files,checkpoints_server_only=checkpoints,
        result_kind=result['result_kind'],scientific_quality_evaluated=False,large_states_not_exported=True))
    return rt.reference(target/'SHA_INDEX.json')


def run(c):
    rt.require(rt.acceptance_only(c), 'acceptance run refuses full-chain contract')
    root = Path(c['job_root']); root.mkdir(parents=True, exist_ok=False)
    rt.publish(root/'RUN-CONSUMED.json',dict(identity=rt.identity(c,'admission'),nonce=c['nonce'],no_retry=True))
    counts.start(c); start = time.monotonic(); done = []; before = span_budget.storage_bytes(c)
    try:
        from .cuda_identity import recorded_subprocess
        for stage in rt.permitted_stages(c):
            gpu = job.gpu_stage(c,stage)
            if gpu: span_budget.start(c,stage); span_budget.guard(c)
            rt.snapshot(c,entry=True,gpu=gpu)
            rt.publish(root/(stage+'-CONSUMED.json'),dict(identity=rt.identity(c,stage),no_retry=True))
            proc = recorded_subprocess(root/(stage+'-supervisor-process'),job.command(c,'supervise',stage),cwd=rt.ROOT,env=job.environment(c))
            if gpu: span_budget.finish(c)
            rt.require(proc.returncode == 0, 'acceptance supervised stage stopped: '+stage)
            rt.verify_stage(c,stage); done.append(stage)
        counters = counts.verify_complete(c); results = {a:rt.verify_stage(c,'accept-'+a) for a in counts.ARMS}
        rt.require(all(r['actual_updates']==3 and r['forward_calls']==6 and r['full_state_RNG_exact']
            and r['gradient_finite'] and not r['state_usable_for_formal'] for r in results.values()), 'three-arm engineering acceptance incomplete')
        cost = dict(profile=c['profile'],scope='GPU_ENGINEERING_ACCEPTANCE',
            GPU_cost_measured=c['profile']=='LAB_GPU_ONCE',
            per_arm={a:{k:r[k] for k in ('elapsed_seconds','complete_update_seconds','peak_allocated_bytes','peak_reserved_bytes')} for a,r in results.items()},
            storage_before_bytes=before,storage_after_bytes=span_budget.storage_bytes(c),
            cumulative_ledger=rt.reference(root/'COST_LEDGER.json') if c['profile']=='LAB_GPU_ONCE' else None,
            scientific_quality_evaluated=False,short_speed_not_full_training_projection=True)
        cost['storage_growth_bytes']=cost['storage_after_bytes']-before
        rt.publish(root/'ENGINEERING_COST.json',cost)
        rt.publish(root/'RESULT.json',dict(status='PASS',result_kind='GPU_ENGINEERING_ACCEPTANCE',
            execution_kind='REAL_GPU' if c['profile']=='LAB_GPU_ONCE' else 'CPU_SYNTHETIC_ONLY_NOT_GPU_VALIDATED',
            execution_scope='GPU_ACCEPTANCE_ONLY',contract_sha256=c['contract_sha256'],source_sha256=c['source']['sha256'],
            stages=done,actual_updates=9,forward_calls=18,counts=rt.reference(counts.path(c)),
            cost=rt.reference(root/'ENGINEERING_COST.json'),scientific_quality_evaluated=False,
            acceptance_state_usable_for_formal=False,elapsed_seconds=time.monotonic()-start))
        closure=dict(status='PASS',identity=rt.identity(c,'admission'),result=rt.reference(root/'RESULT.json'),
            stage_closures=[rt.reference(rt.stage_path(c,s)/'CLOSED.json') for s in rt.permitted_stages(c)])
        rt.publish(root/'ENGINEERING_CLOSED.json',closure)
        index=export(c,root/'slim-export',pending_export=True)
        rt.publish(root/'EXPORT_COMPLETE.json',dict(index=index,process_will_exit=True))
        rt.publish(root/'CLOSED.json',dict(closure,export_index=index))
        verify(c)
        print(json.dumps(dict(status='PASS',scope='GPU_ACCEPTANCE_ONLY',actual_updates=9,forward_calls=18)))
        return 0
    except BaseException as exc:
        ledger=root/'COST_LEDGER.json'
        if ledger.exists() and rt.read(ledger)['active'] is not None: span_budget.finish(c)
        rt.publish(root/'STOP.json',dict(status='STOP_NO_RETRY',completed_stages=done,error=repr(exc),
            traceback=traceback.format_exc(),contract_sha256=c['contract_sha256'],elapsed_seconds=time.monotonic()-start))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','bind','run','supervise','worker','verify','export'))
    p.add_argument('--contract',type=Path);p.add_argument('--spec',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--stage',choices=tuple(rt.acceptance_config()['stages']))
    p.add_argument('--output',type=Path);p.add_argument('--cpu-gate',type=Path);p.add_argument('--authorization',type=Path)
    a=p.parse_args()
    if a.action=='prepare' and a.spec is not None:
        rt.require(a.source is not None and a.output is not None,'prepare source/output required')
        return prepare(a.spec,rt.reference(a.source),a.output)
    if a.action=='bind':
        rt.require(all(x is not None for x in (a.spec,a.source,a.cpu_gate,a.authorization,a.output)),'all acceptance bind references required')
        bind(a.spec,rt.reference(a.source),rt.reference(a.cpu_gate),rt.reference(a.authorization),a.output);return 0
    rt.require(a.contract is not None,'explicit acceptance contract required')
    c=load(a.contract,preparation=a.action=='prepare')
    if a.action=='prepare':
        rt.require(a.output is not None,'fresh preparation output required');return job.prepare(c,a.output)
    if a.action=='run':return run(c)
    if a.action=='verify':verify(c);return 0
    if a.action=='export':
        rt.require(a.output is not None,'fresh independent export directory required');export(c,a.output);return 0
    rt.require(a.stage is not None,'explicit acceptance stage required')
    if a.action=='supervise':return job.supervise(c,a.stage)
    from .span_worker import worker
    return worker(c,a.stage)
