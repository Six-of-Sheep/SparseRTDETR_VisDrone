"""Explicit old-endpoint lineage for a new protocol-only inference attempt.

Never changes a checkpoint binding or calls a training/launch entry.
"""
from pathlib import Path


def exact(ref):
    from sparse_rtdetr.localization.runio import reference, require
    actual = reference(ref['path'])
    require(actual['sha256'] == ref['sha256'] and
            actual['bytes'] == ref.get('bytes', ref.get('size_bytes')), 'resume reference bytes differ: ' + ref['path'])
    return actual


def parent_contract(c):
    from sparse_rtdetr.localization.runio import read, require, digest
    reuse = c['parent_reuse']
    require(c['seed'] == 1 and reuse['kind'] == 'certified_closed_seed1_e120_and_A548', 'only original seed1 endpoint is reusable')
    exact(reuse['contract']); parent = read(reuse['contract']['path'])
    require(digest({k:v for k,v in parent.items() if k != 'contract_sha256'}) == parent['contract_sha256'], 'parent contract digest changed')
    require(parent['seed'] == c['seed'] and parent['task_id'] == c['task_id'] and parent['config'] == c['config']
            and parent['data_sha256'] == c['data_sha256'] and parent['environment'] == c['environment']
            and parent['gpu_uuid'] == c['gpu_uuid'], 'resume scientific/runtime parent differs')
    for key in ('data','pretrained','development_ids','fixed64','evaluation_protocol','evaluation_protocol_sha256','materials','plan'):
        require(parent.get(key)==c.get(key), 'resume parent scientific inputs differ: '+key)
    proof = reuse['certification']; exact(proof); certified = read(proof['path'])
    require(certified['status'] == 'PASS' and certified['old_contract'] == reuse['contract']
            and certified['new_source_sha256'] == c['source_sha256']
            and certified['old_source_sha256'] == parent['source_sha256']
            and certified['protocol_only_source_delta'] is True, 'resume certification differs')
    require(certified['reuse_references']=={k:v for k,v in reuse.items() if isinstance(v,dict) and k!='certification'},
            'resume certification reference set differs')
    return parent


def endpoint(c):
    from sparse_rtdetr.localization.runio import read, require, reference
    parent = parent_contract(c) if c.get('parent_reuse') else c
    root = Path(parent['job_root']); lock = read(root/'formal/endpoint-lock.json')
    require(lock['status'] == 'R18_COMMON_E120_EMA_LOCKED' and lock['seed'] == c['seed']
            and lock['epoch'] == 120 and lock['updates'] == lock['actual_updates'] == 36480
            and lock['source_sha256'] == parent['source_sha256'] and lock['contract_sha256'] == parent['contract_sha256']
            and not lock['used_acceptance_state'], 'endpoint parent/source/cursor differs')
    require(lock['binding'] == reference(root/'formal/binding.json'), 'original checkpoint binding changed')
    if c.get('parent_reuse'):
        reuse=c['parent_reuse'];exact(reuse['endpoint']);exact(reuse['formal_closed'])
        require(reuse['endpoint']['path'] == str(root/'formal/endpoint-lock.json'), 'endpoint path differs')
        closed=read(root/'formal/CLOSED.json')
        require(reuse['formal_closed']['path']==str(root/'formal/CLOSED.json') and closed['status']=='PASS'
                and closed['seed']==1 and closed['mode']=='formal' and closed['result']==reference(root/'formal/RESULT.json')
                and closed['runtime_index']==reference(root/'formal/runtime-archive-index.json'), 'original formal closure changed')
    return lock


def original_A_contract(c):
    if not c.get('parent_reuse'):
        return c
    from sparse_rtdetr.localization.runio import read, require, reference
    parent=parent_contract(c);reuse=c['parent_reuse'];endpoint(c)
    for key in ('A_result','A_closed','A_archive_index'):
        exact(reuse[key])
    out=Path(parent['job_root'])/'infer-A';closed=read(out/'CLOSED.json')
    require(reuse['A_result']['path'] == str(out/'RESULT.json') and closed['status']=='PASS'
            and reuse['A_closed']['path']==str(out/'CLOSED.json') and reuse['A_archive_index']['path']==str(out/'runtime-archive-index.json')
            and closed['seed']==1 and closed['mode']=='infer-A'
            and closed['result']==reference(out/'RESULT.json') and closed['runtime_index']==reference(out/'runtime-archive-index.json'),
            'original A closure changed')
    return {**parent, 'checkpoint':endpoint(c)['checkpoint']}


def prerequisite(c, mode):
    from sparse_rtdetr.localization.runio import require
    require(mode in ('accept-A','accept-B','accept-C','full-C','evaluation'), 'seed1 resume cannot train/replay A548 or acceptance state')
    original_A_contract(c)


def restore_contract(c):
    """Use the *unaltered* old binding for exact full-state restore."""
    return parent_contract(c) if c.get('parent_reuse') else c
