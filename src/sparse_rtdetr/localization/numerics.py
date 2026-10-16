"""Explicit immutable identity for the one authorized decoding arithmetic."""
from pathlib import Path
from .runio import sha,require
from .distribution import NUMERICAL_IMPLEMENTATION

def identity():
    return {'name':NUMERICAL_IMPLEMENTATION,'implementation_sha256':sha(Path(__file__).with_name('distribution.py')),
            'softmax':'FP32 unchanged','mass_and_displacement':'FP64 per-side numerator / mass; both differentiable',
            'center':'reference_center + average signed relative displacements','size':'difference of signed relative displacements',
            'output':'FP32','threshold':'consumer_geometry_contract_v1_terminal_dn_aux_v1','admission':'terminal valid DN- decoded finite auxiliary only; all other original typed consumer/positive/reference/support rules retained',
            'clamp_epsilon_query_skip':False}

def validate_binding(binding,contract=None):
    expected=identity()
    from .consumer_geometry import identity as geometry_identity
    require(binding.get('consumer_geometry_contract')==geometry_identity(),'active checkpoint consumer contract missing or different')
    require(binding.get('numerical_implementation')==expected,'checkpoint numerical implementation identity missing or different')
    if contract is not None:
        require(contract['numerical_implementation']==expected,'contract numerical implementation differs')
        require(binding['source_sha256']==contract['source_sha256'] and binding['contract_sha256']==contract['contract_sha256'],'active numerical checkpoint source/contract differs')
    return True

def restore_active(engine,binding,path,expected_sha,ledger_path=None,contract=None):
    try:
        validate_binding(binding,contract)
        from .cache_adapter import strict_restore
        return strict_restore(engine,binding,path,expected_sha,ledger_path)
    except BaseException:
        engine.failed=True
        raise
