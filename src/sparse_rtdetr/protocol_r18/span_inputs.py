"""Production list adapter: historical train IDs are audit only, never timing.

GPU acceptance still uses epoch-1's first two original logical16 windows.
"""
from . import span_runtime as rt

PARENT_SHA = '6dbe86d56d5dd4053bcbc373276ad0c137cc4e892e48035bb41ee2173d1ddba7'
TRAIN_PURPOSE = 'historical_training_acceptance_2x16_audit_only'
TIMING_PURPOSE = 'fixed64_development_cost'


def adapt(c):
    if c['profile'] != 'LAB_GPU_ONCE':
        rt.require(c['profile'] == 'CPU_FIXTURE_ONLY', 'unknown input-list profile')
        return c
    from ..localization.material_paths import MANIFESTS, SELECTION_SHA
    lists = c.get('input_lists')
    rt.require(isinstance(lists, dict) and set(lists) == {'training_acceptance', 'timing'},
               'explicit training_acceptance/timing input lists required')
    required = {'purpose', 'role', 'source', 'ids'}
    train, timing = lists['training_acceptance'], lists['timing']
    for value in (train, timing):
        rt.require(isinstance(value, dict) and set(value) == required, 'input-list fields missing/extra')
    rt.require(train['purpose'] == TRAIN_PURPOSE and train['role'] == 'train_core',
               'training acceptance purpose/role differs')
    rt.require(timing['purpose'] == TIMING_PURPOSE and timing['role'] == 'development',
               'timing purpose/role differs')
    rt.require(train['source']['sha256'] == PARENT_SHA, 'training acceptance source identity differs')
    rt.require(timing['source']['sha256'] == SELECTION_SHA, 'timing source identity differs')
    parent, selection = rt.exact(train['source']), rt.exact(timing['source'])
    historical = parent['fixed_ids']
    rt.require(isinstance(historical, list) and len(historical) == 2 and
               all(isinstance(g, list) and len(g) == 16 for g in historical),
               'historical training acceptance must be two groups of sixteen')
    train_ids = [v for group in historical for v in group]
    timing_ids = [r['stable_image_id'] for r in selection['selected']]
    rt.require(all(type(v) is str for v in train_ids + timing_ids) and
               len(set(train_ids)) == 32 and len(timing_ids) == len(set(timing_ids)) == 64,
               'input-list quantity/uniqueness differs')
    rt.require(train['ids'] == historical, 'historical training acceptance member/order differs')
    rt.require(timing['ids'] == timing_ids, 'timing member/order differs')
    members = {}
    for role, count in (('train_core', 4869), ('development', 548)):
        d = c['data'][role]
        rt.require(d['role'] == role and d['count'] == count and
                   d['manifest']['sha256'] == MANIFESTS[role][0], 'input-list dataset binding differs')
        rows = rt.exact(d['manifest'])['records']
        members[role] = {r['stable_image_id'] for r in rows}
        rt.require(len(rows) == len(members[role]) == count, 'input-list manifest members differ')
    rt.require(not members['train_core'] & members['development'], 'input-list role overlap')
    rt.require(set(train_ids) <= members['train_core'] and set(timing_ids) <= members['development'],
               'input-list member belongs to wrong role')
    rt.require(c['selection'] == timing['source'], 'selection and timing source conflict')
    rt.require(selection['source_manifest_sha256'] == c['data']['development']['manifest']['sha256'],
               'timing selection manifest conflict')
    rt.require('fixed_ids' not in c or c['fixed_ids'] == timing_ids, 'legacy fixed_ids conflicts with timing purpose')
    c['fixed_ids'] = timing_ids
    proof = dict(version='sampling-span-input-list-binding-v1',
        training_acceptance=dict(purpose=TRAIN_PURPOSE, role='train_core', source=train['source'],
            groups=2, per_group=16, ids_sha256=rt.digest(historical), used_for_new_GPU_acceptance=False),
        timing=dict(purpose=TIMING_PURPOSE, role='development', source=timing['source'],
            count=64, ids_sha256=rt.digest(timing_ids)),
        new_GPU_acceptance='original sampler epoch1 first two logical16 windows; unchanged')
    rt.require('input_list_proof' not in c or c['input_list_proof'] == proof, 'input-list proof conflict')
    c['input_list_proof'] = proof
    return c
