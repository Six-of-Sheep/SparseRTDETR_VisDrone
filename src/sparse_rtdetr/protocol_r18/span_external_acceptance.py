"""Read-only reuse of authentic external acceptance, never a local stage copy."""
from pathlib import Path,PurePosixPath
from . import span_runtime as rt


def descriptor(contract):
    old=rt.read(contract);root=Path(old['job_root'])
    return dict(schema='sampling-span-external-acceptance-v1',contract=rt.reference(contract),source=old['source'],
        closed=rt.reference(root/'CLOSED.json'),execution_exit=rt.reference(Path(contract).parent/'EXECUTION_EXIT.json'),
        cost_ledger=rt.reference(root/'COST_LEDGER.json'))


def parent(c):
    rt.require(c['protocol']==rt.FORMAL_PROTOCOL,'external resolver only for formal protocol')
    if '_verified_external' not in c:verify(c)
    return c['_verified_external']['parent']


def verify(c):
    policy=rt.read(rt.ROOT/rt.FORMAL_CONFIG);d=rt.exact(c['external_acceptance'])
    rt.require(d['schema']=='sampling-span-external-acceptance-v1','external evidence schema differs')
    old=rt.exact(d['contract'])
    rt.require(old['task_id']==rt.TASK and old['protocol']==rt.ACCEPTANCE_PROTOCOL and old['profile']==c['profile'], 'external acceptance identity/profile differs')
    rt.require(old['contract_sha256']==rt.digest({k:v for k,v in old.items() if k!='contract_sha256'}),'external contract digest differs')
    rt.require(old['execution_scope']=='GPU_ACCEPTANCE_ONLY' and old['science_authorized'] is False and
        old['GPU_authorized'] is (c['profile']=='LAB_GPU_ONCE') and old['stages']==rt.acceptance_config()['stages'] and
        old['acceptance_limits']==rt.acceptance_config()['limits'] and old['acceptance_sha256']==rt.digest(rt.acceptance_config()),
        'external acceptance scope/authority differs')
    rt.require(old['science_sha256']==c['science_sha256'] and old['execution_sha256']==c['execution_sha256'] and
        old['pretrained']==c['pretrained'] and old['data']==c['data'] and old['seed']==c['seed'], 'external science/data/initialization differs')
    rt.require(d['source']==old['source'],'external source reference differs')
    source=rt.exact(d['source']);new=rt.exact(c['source'])
    if c['profile']=='LAB_GPU_ONCE':
        rt.require(source['commit']==policy['accepted_commit'] and d['source']['sha256']==policy['accepted_source_sha256'] and
            d['contract']['sha256']==policy['accepted_contract_sha256'],'external accepted release/contract identity differs')
    root=Path(old['source_root']);names=[]
    for row in source['files']:
        name=PurePosixPath(row['path']);p=root/row['path']
        rt.require(not name.is_absolute() and '..' not in name.parts and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'unsafe external source member')
        rt.require(p.stat().st_size==row['bytes'] and rt.sha(p)==row['sha256'],'external source bytes changed: '+row['path'])
        names.append(row['path'])
    rt.require(len(names)==len(set(names)),'duplicate external source member')
    before={r['path']:r for r in source['files']};after={r['path']:r for r in new['files']}
    rt.require(set(before)<=set(after),'new source dropped authenticated files')
    changed=[n for n in before if before[n]!=after[n]]
    rt.require(set(changed)<=set(policy['existing_file_changes_allowed']),'accepted computation changed: '+','.join(sorted(set(changed)-set(policy['existing_file_changes_allowed']))))
    worker='src/sparse_rtdetr/protocol_r18/span_worker.py'
    if worker in changed:
        from .span_B0_reuse import worker_binding_only
        worker_binding_only((root/worker).read_bytes(),(rt.ROOT/worker).read_bytes())
    # Includes all model/training/checkpoint/criterion/observation/evaluation
    # workers, unchanged byte-for-byte, not an assertion of full-source equality.
    rt.require('src/sparse_rtdetr/protocol_r18/span_worker.py' in before and rt.SCIENCE in before and rt.EXECUTION in before,'external source incomplete')
    rt.require(d['closed']==rt.reference(Path(old['job_root'])/'CLOSED.json'),'external closure path/bytes differ')
    from .span_acceptance import verify as verify_acceptance
    result=verify_acceptance(old)
    rt.require(result['execution_kind']==('REAL_GPU' if c['profile']=='LAB_GPU_ONCE' else 'CPU_SYNTHETIC_ONLY_NOT_GPU_VALIDATED'), 'external result execution kind differs')
    ex=rt.exact(d['execution_exit'])
    rt.require(ex['status']=='PASS' and ex['returncode']==0 and ex['contract']==d['contract'] and ex['scientific_quality_evaluated'] is False,'external execution exit differs')
    for arm in ('B0','B1','P'):
        r=rt.verify_stage(old,'accept-'+arm);state=r['checkpoint'];p=Path(state['path'])
        rt.require(r['actual_updates']==3 and r['forward_calls']==6 and r['full_state_RNG_exact'] and r['gradient_finite'] and not r['state_usable_for_formal'],'external engineering result incomplete')
        rt.require(p.stat().st_size==state['size_bytes'] and rt.sha(p)==state['sha256'] and rt.read(p.with_suffix('.pth.complete.json'))==state,'external checkpoint size/SHA/sidecar conflict')
    ledger=rt.exact(d['cost_ledger']);prior=rt.exact(c.get('acceptance_prior_cost',c['prior_cost']))
    rt.require(ledger['active'] is None and ledger['forward_calls']==18 and prior['forward_calls']==18 and
        prior['task_id']==rt.TASK and prior['profile']==c['profile'] and prior['GPU_seconds']==ledger['completed_GPU_seconds'], 'external cumulative cost differs/reset')
    if c['profile']=='LAB_GPU_ONCE':
        rt.require(rt.exact(result['cost'])['cumulative_ledger']==d['cost_ledger'] and
            ledger['identity']['contract_sha256']==old['contract_sha256'],'external ledger binding differs')
        rt.require(abs(ledger['completed_GPU_seconds']-policy['prior_GPU_seconds_rounded6'])<0.000001,'external GPU seconds identity differs')
    proof=dict(parent=old,descriptor=c['external_acceptance'],prior_cost=c['prior_cost'],changed_files=changed,
        added_files=sorted(set(after)-set(before)),unchanged_accepted_computation=True,
        accepted_source=d['source'],new_source=c['source'],no_checkpoint_loaded_for_training=True)
    c['_verified_external']=proof
    return proof
