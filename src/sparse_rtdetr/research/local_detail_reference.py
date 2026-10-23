"""Explicit read-only historical N EMA import for future cost inference only.

Never restores an old optimizer, RNG, engine or a U/D model. Caller must use the
SHA/binding from the admitted reference index, not obtain them from the tensor.
"""
import hashlib

import torch

from ..protocol_r18.model import Config
from .local_detail_model import validate_model
from .local_detail_engine import assert_healthy
from .local_detail_checkpoint import file_sha, canonical


def load_N_for_cost(model, path, *, expected_sha, expected_binding):
    validate_model(model,'N')
    if model.training or list(model.decoder.eval_spatial_size) != [768,1344]:
        raise ValueError('historical N import is eval-only on the fixed production canvas')
    if file_sha(path)!=expected_sha:raise ValueError('historical N SHA differs')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (payload.get('format')!='sparse_rtdetr.sampling_span_fullstate_v1'
        or payload.get('binding')!=expected_binding
        or payload.get('binding_sha256')!=hashlib.sha256(canonical(expected_binding)).hexdigest()
        or expected_binding.get('arm')!='r18-sampling-span-v1/B0'
        or expected_binding.get('scope')!='span_formal_e120'
        or expected_binding.get('config')!=Config().binding()
        or payload.get('engine',{}).get('completed_updates')!=36480
        or payload.get('ema',{}).get('updates')!=36480):
        raise ValueError('not the admitted fixed B0 e120 EMA reference')
    if expected_binding.get('pretrained_sha256')!='911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720':
        raise ValueError('historical N initialization differs')
    state=payload['ema']['module'];current=model.state_dict()
    if set(state)!=set(current):raise ValueError('historical N keys differ')
    for name,tensor in current.items():
        value=state[name]
        if value.shape!=tensor.shape or value.dtype!=tensor.dtype:raise ValueError('historical N layout differs: '+name)
        if name.rsplit('.',1)[-1] in {'anchors','valid_mask','num_points_scale'} and not torch.equal(value,tensor.cpu()):
            raise ValueError('historical N immutable cache differs: '+name)
    # Reject all mutable nonfinite state before applying any weights.
    for name,value in state.items():
        if value.is_floating_point():
            if name=='decoder.anchors':
                mask=state['decoder.valid_mask'].expand_as(value)
                ok=torch.where(mask,value.isfinite(),value.isposinf()).all()
            else:ok=value.isfinite().all()
            if not bool(ok):raise ValueError('historical N nonfinite state: '+name)
            if name.endswith('running_var') and bool((value<0).any()):raise ValueError('historical N negative BN variance')
    model.load_state_dict(state,strict=True)
    assert_healthy(model)
    return dict(scope='historical_N_same_session_cost_only',checkpoint_sha256=expected_sha,
                loaded='EMA_model_only',old_optimizer_or_RNG_restored=False,old_CLOSED_created=False)
