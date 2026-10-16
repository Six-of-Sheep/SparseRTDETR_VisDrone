"""Best-effort forensic state. Never a resumable checkpoint."""
import os
from pathlib import Path
import torch
from ..baseline.training_v2b_checkpoint import _safe
from ..localization.runio import fsync_directory,reference
from .checkpoint import capture_rng

def preserve(holder,out):
    e=holder.get('engine')
    if e is None:return {'engine_initialized':False,'resumable':False}
    value={'format':'native-r18-common-failure-forensic-v1','resumable':False,
        'completed_updates':e.completed_updates,'phase':e.phase,'failed':e.failed,
        'model':e.model.state_dict(),'optimizer':e.optimizer.state_dict(),'EMA':e.ema.state_dict(),
        'raw_gradients':{n:p.grad for n,p in e.model.named_parameters()},'RNG':capture_rng(e.device),
        'current_input':holder.get('input'),'binding':holder.get('binding')}
    path=Path(out)/'failure-forensic.pth';partial=path.with_suffix('.pth.partial')
    if path.exists() or partial.exists():raise FileExistsError('forensic state already exists')
    with partial.open('xb') as f:torch.save(_safe(value),f);f.flush();os.fsync(f.fileno())
    os.link(partial,path);fsync_directory(path.parent);partial.unlink();fsync_directory(path.parent)
    return {'engine_initialized':True,'resumable':False,'file':reference(path)}
