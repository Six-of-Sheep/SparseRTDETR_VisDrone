"""One fixed three-arm detail-path fine-tune preparation; no launch or GPU admission.

The native model, criterion, optimizer grouping and full-update math are reused.
Only deterministic whole-image view selection, schedule, new state identity and the
optional detail-path installation are added (see detail_path.py).  G1/G2 share the
identical data stream of G0: installation consumes no global RNG draws, and the
branch input is stop-gradient in both installed arms.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from slice_core import tile_windows
from ..baseline import training_v2b_checkpoint as layout
from ..baseline.training_v2b import _optimizer
from ..localization.data import RawAugmentation, RawRoleDataset, parse_raw
from .engine import Engine, assert_healthy, tensor_digest
from . import checkpoint as native_state
from .model import CommonEMA

VERSION = 'r18-detail-arms-v1'
ARMS = ('G0', 'G1', 'G2')
ARM_INSTALL = {'G0': None, 'G1': 'control', 'G2': 'detail'}
UPDATES = 3648
PASSES = 12
PARENT_SHA = '13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_plan(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    if p['version'] != VERSION or p['arms'] != list(ARMS) or p['updates_per_arm'] != UPDATES:
        raise ValueError('different fixed fine-tune plan')
    if p['GPU_authorized'] is not False or p['parent']['checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('preparation authority or parent differs')
    if p.get('arm_install') != {a: ARM_INSTALL[a] for a in ARMS}:
        raise ValueError('arm install map differs')
    return p


def clock(epoch):
    if type(epoch) is not int or not 1 <= epoch <= PASSES:
        raise ValueError('fine-tune pass outside 1..12')
    return 108 + epoch


def view_id(arm, position):
    if arm not in ARMS or type(position) is not int or not 0 <= position < 4864:
        raise ValueError('arm/position outside fixed full-window pass')
    # every arm consumes only original whole-image views; no draws, no tiles
    return 0


def crop_raw(image, raw, view):
    """Clip every intersecting raw box; no GT-dependent tile or empty resampling.

    Stable GT IDs/raw area/physical row metadata stay those of the original.
    category0 and score0 keep their original roles, with identical crop geometry.
    All positive intersections survive here; baseline sanitize(min_size=1) and
    any subsequent native RandomIoUCrop act without changed rules.
    """
    if type(view) is not int or not 0 <= view <= 4:
        raise ValueError('invalid view')
    if view == 0:
        return image, raw, {'view_id': 0, 'crop_box': [0, 0, image.width, image.height]}
    tile = tile_windows(image.height, image.width)[view-1]
    result = copy.deepcopy(raw)
    for key in ('gt', 'spatial_ignore', 'class_ignore'):
        kept = []
        for row in raw[key]:
            x0, y0, x1, y1 = row['box']
            if not all(math.isfinite(float(v)) for v in row['box']):
                raise ValueError('nonfinite raw geometry')
            b = [max(0., x0-tile.x0), max(0., y0-tile.y0),
                 min(float(tile.width), x1-tile.x0), min(float(tile.height), y1-tile.y0)]
            if b[2] > b[0] and b[3] > b[1]:
                kept.append(dict(row, box=b))
        result[key] = kept
    return image.crop(tile.crop_box), result, tile.as_dict()


class Transform:
    def __init__(self, arm, input_hw=(768, 1344)):
        if arm not in ARMS:
            raise ValueError('unknown fine-tune arm')
        self.arm, self.base = arm, RawAugmentation(input_hw)

    def __call__(self, image, raw, *, epoch, position, image_id):
        view = view_id(self.arm, position)
        original_hw = [image.height, image.width]
        image, target, selection = crop_raw(image, raw, view)
        pixels, target, trace = self.base(image, target, epoch=clock(epoch), seed=0,
                                        position=position, image_id=image_id, training=True)
        trace.update(finetune_pass=epoch, augmentation_epoch=clock(epoch), arm=self.arm,
                     fixed_view=selection, original_source_hw=original_hw,
                     crop_gt_ids=[r['id'] for r in target_rows(raw, selection)],
                     source_gt_count=len(raw['gt']))
        return pixels, target, trace


def target_rows(raw, selection):
    x0,y0,x1,y1 = selection['crop_box']
    return [r for r in raw['gt'] if min(x1,r['box'][2])>max(x0,r['box'][0])
            and min(y1,r['box'][3])>max(y0,r['box'][1])]


class Dataset(RawRoleDataset):
    def __init__(self, arm, **kwargs):
        super().__init__(role='train_core', seed=0, **kwargs)
        self.ft_transform = Transform(arm)

    def __getitem__(self, key):
        # Existing sampler emits the shared baseline epoch109..120.
        source_epoch, position, index = key
        epoch = source_epoch-108
        clock(epoch)
        row = self.records[index]
        image_bytes, annotation = self.bound_bytes(row)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert('RGB')
        if [image.width,image.height] != [row['width'],row['height']]:
            raise ValueError('raw image size disagrees with member manifest')
        raw = parse_raw(annotation,row['stable_image_id'],expected_rows=row['raw_annotation_rows'])
        x,t,trace = self.ft_transform(image,raw,epoch=epoch,position=position,image_id=row['stable_image_id'])
        trace.update(image_sha256=row['image_sha256'],annotation_sha256=row['annotation_sha256'],
                     spawn_observer={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()})
        return x,t,trace


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= UPDATES:
        raise ValueError('fine-tune LR clock outside frozen budget')
    if update <= 200:
        return 1e-5*update/200
    return 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*(update-200)/(UPDATES-200)))


class FineTuneEngine(Engine):
    def __init__(self, model, criterion, optimizer, ema, config, *, arm, plan_sha256, parent_sha256):
        if arm not in ARMS or any(len(v)!=64 for v in (plan_sha256,parent_sha256)):
            raise ValueError('fine-tune identity missing')
        self.arm, self.plan_sha256, self.parent_sha256 = arm,plan_sha256,parent_sha256
        super().__init__(model,criterion,optimizer,ema,config)
        for g in optimizer.param_groups:
            g['lr'] = learning_rate(1)

    def next_learning_rate(self):
        return learning_rate(self.completed_updates+1)

    def state_dict(self):
        s = super().state_dict()
        s.update(schema=VERSION,arm=self.arm,plan_sha256=self.plan_sha256,parent_sha256=self.parent_sha256,
                 total_updates=UPDATES,augmentation_clock='109..120; paired seed0 original sampler',
                 endpoint='FT012_EMA; no intermediate evaluation/selection')
        return s

    def validate_state_dict(self, s):
        u=s.get('completed_updates')
        if type(u) is not int or not 0<=u<=UPDATES:
            raise ValueError('invalid fine-tune update')
        expected=self.state_dict()
        expected.update(completed_updates=u,microsteps=2*u,completed_epochs=u//304,
                        next_epoch=u//304+1,next_window=u%304)
        counts=s.get('parameter_optimizer_steps')
        if not isinstance(counts,dict) or set(counts)!=set(self.parameter_steps) or any(type(v) is not int or not 0<=v<=u for v in counts.values()):
            raise ValueError('fine-tune parameter clock mismatch')
        expected['parameter_optimizer_steps']=counts
        if s!=expected:
            raise ValueError('fine-tune state/arm/parent/plan mismatch')

    def train_window(self,*args,**kwargs):
        if self.completed_updates>=UPDATES:
            raise RuntimeError('fixed fine-tune endpoint already reached')
        return super().train_window(*args,**kwargs)


def initialize_from_parent(model, payload, *, expected_binding):
    """Explicit EMA-only extraction; no old optimizer/EMA clock/RNG continuation.

    Caller must first read full bytes against certified external checkpoint SHA.
    This pure function checks authenticated payload binding and full native schema.
    """
    if payload.get('format')!=native_state.FORMAT or payload.get('binding')!=expected_binding:
        raise ValueError('wrong native parent identity')
    if payload.get('binding_sha256')!=digest(expected_binding):
        raise ValueError('parent binding digest mismatch')
    if payload['engine']['completed_updates']!=36480 or payload['ema']['updates']!=36480:
        raise ValueError('parent is not certified e120')
    if payload['ema']['decay']!=.9999 or payload['ema']['warmups']!=2000:
        raise ValueError('parent EMA recipe differs')
    layout._check_model_state(payload['ema']['module'],layout._model_layout(model),'parent EMA')
    model.load_state_dict(payload['ema']['module'],strict=True)
    return tensor_digest(model.state_dict().items())


def make_optimizer_ema(model, config):
    opt=_optimizer(model,config)
    for g in opt.param_groups:g['lr']=learning_rate(1)
    ema=CommonEMA(model)
    if opt.state or ema.updates!=0 or tensor_digest(model.state_dict().items())!=tensor_digest(ema.module.state_dict().items()):
        raise RuntimeError('not an independent fresh optimizer/EMA')
    return opt,ema


def capture(engine):
    if engine.phase or engine.failed or any(p.grad is not None for p in engine.model.parameters()):
        raise ValueError('not a complete healthy fine-tune boundary')
    assert_healthy(engine.model,engine.optimizer,engine.ema)
    behavior=native_state.behavior_identity(engine)
    value={'format':VERSION,'engine':engine.state_dict(),'behavior':behavior,
           'execution_binding':getattr(engine,'execution_binding',None),
           'torch_version':str(torch.__version__),'device':str(engine.device),
           'model':engine.model.state_dict(),'model_layout':layout._model_layout(engine.model),
           'model_training':{n:m.training for n,m in engine.model.named_modules()},
           'optimizer':engine.optimizer.state_dict(),'optimizer_layout':layout._optimizer_layout(engine.model,engine.optimizer),
           'ema':engine.ema.state_dict(),'ema_layout':layout._model_layout(engine.ema.module),
           'rng':native_state.capture_rng(engine.device)}
    return layout._safe(value)


def save(engine,path):
    path=Path(path)
    value=capture(engine)
    # New output path required; no old checkpoint mutation or overwrite.
    partial=path.with_suffix(path.suffix+'.partial')
    if path.exists() or partial.exists():raise FileExistsError(path)
    with partial.open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    sha=native_state.file_sha(partial)
    os.link(partial,path);partial.unlink()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha,'completed_updates':engine.completed_updates}


def restore(engine,reference):
    try:
        raw=Path(reference['path']).read_bytes()
        if len(raw)!=reference['bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('fine-tune checkpoint SHA/bytes mismatch')
        p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        current=capture(engine)
        if set(p)!=set(current) or any(p[k]!=current[k] for k in ('format','torch_version','device','behavior','model_layout','optimizer_layout','ema_layout','execution_binding')):
            raise ValueError('fine-tune runtime/behavior/layout mismatch')
        engine.validate_state_dict(p['engine'])
        for k,m in [('model',engine.model),('ema',engine.ema.module)]:
            layout._check_model_state(p[k] if k=='model' else p['ema']['module'],layout._model_layout(m),k)
        u=p['engine']['completed_updates']
        expected_lr=learning_rate(max(u,1))
        if any(g['lr']!=expected_lr for g in p['optimizer']['param_groups']):
            raise ValueError('fine-tune schedule mismatch')
        layout._check_optimizer(p['optimizer'],p['optimizer_layout'],engine.model,u)
        for group in p['optimizer_layout']['groups']:
            for n,i in zip(group['names'],group['ids']):
                if int(p['optimizer']['state'].get(i,{}).get('step',0))!=p['engine']['parameter_optimizer_steps'][n]:
                    raise ValueError('fine-tune exact optimizer step differs')
        if p['ema']['updates']!=u or p['ema']['decay']!=.9999 or p['ema']['warmups']!=2000:
            raise ValueError('fine-tune EMA clock/recipe mismatch')
        modes=p['model_training']
        if set(modes)!={n for n,_ in engine.model.named_modules()} or any(type(v) is not bool for v in modes.values()):
            raise ValueError('fine-tune training modes differ')
        native_state.validate_rng(p['rng'],engine.device)
        engine.model.load_state_dict(p['model'],strict=True)
        for n,m in engine.model.named_modules():m.training=modes[n]
        engine.optimizer.load_state_dict(p['optimizer']);engine.ema.load_state_dict(p['ema'])
        engine.load_state_dict(p['engine']);native_state.restore_rng(p['rng'],engine.device)
        if not layout._equal(p,capture(engine)):
            raise RuntimeError('fine-tune exact restored full state/RNG differs')
        return {'full_state_RNG_exact':True,'completed_updates':u,'sha256':reference['sha256']}
    except BaseException:
        engine.failed=True
        raise
