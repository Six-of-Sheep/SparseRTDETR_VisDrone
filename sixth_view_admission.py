"""Explicit new-contract to unchanged legacy pre-CUDA resource interface."""
from copy import deepcopy


def adapt_resources(contract):
    """Never infer a missing identity or replace a conflicting legacy field."""
    def need(ok, message):
        if not ok:
            raise RuntimeError('sixth resource contract: ' + message)

    for key in ('runtime_root', 'boot_id', 'gpu_uuid', 'storage_identity', 'environment'):
        need(key in contract, 'missing ' + key)
    for key in ('runtime_root', 'boot_id', 'gpu_uuid'):
        need(isinstance(contract[key], str) and bool(contract[key]), 'invalid ' + key)
    identity = contract['storage_identity']
    keys = {'canonical_mount', 'platform_entry', 'mount', 'st_dev', 'planning_cap_bytes'}
    need(isinstance(identity, dict) and set(identity) == keys, 'storage identity schema')
    for key in ('canonical_mount', 'platform_entry'):
        need(isinstance(identity[key], str) and identity[key].startswith('/'), 'storage ' + key)
    need(isinstance(identity['mount'], dict) and bool(identity['mount']), 'storage mount')
    for key in ('st_dev', 'planning_cap_bytes'):
        need(type(identity[key]) is int and identity[key] >= 0, 'storage ' + key)
    env = contract['environment']
    need(isinstance(env, dict), 'environment schema')
    for key in ('torch', 'torchvision', 'cuda_build', 'cudnn'):
        need(key in env and env[key] is not None, 'missing environment.' + key)
    if 'storage' in contract:
        need(isinstance(contract['storage'], dict) and 'identity' in contract['storage'], 'legacy storage identity missing')
        need(contract['storage']['identity'] == identity, 'conflicting storage identity')
    result = deepcopy(contract)
    if 'storage' not in result:
        result['storage'] = {'identity': deepcopy(identity)}
    return result
