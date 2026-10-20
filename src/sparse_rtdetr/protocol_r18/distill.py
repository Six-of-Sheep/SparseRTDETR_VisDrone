"""Fixed positive-prediction KD preparation; never opens development supervision.

Not a reproduction of full DETRDistill. Native Hungarian matching is reused;
query IDs identify cache rows, never teacher/student correspondence.
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
from torch.nn import functional as F
from torchvision.ops import box_convert, generalized_box_iou

from slice_core import (Candidate, classwise_nms, map_crop_xyxy,
                        select_view_candidates, tile_windows)
from ..baseline import training_v2b_checkpoint as layout
from ..baseline.training_v2b import _optimizer
from ..localization.data import RawAugmentation, RawRoleDataset, parse_raw
from ..localization.ignore import negative_keep_mask
from ..localization.material_paths import MANIFESTS
from . import checkpoint as state
from .engine import tensor_digest
from .model import CommonEMA

VERSION = 'r18-multiview-positive-kd-v1'
ARMS = ('T0', 'T1', 'T2')
TRAIN_SHA = MANIFESTS['train_core'][0]
PARENT_SHA = '13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757'
UPDATES = 3648


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_plan(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    frozen = Path(__file__).resolve().parents[3] / 'configs/research/r18_multiview_distill_v1.json'
    if value != json.loads(frozen.read_text(encoding='utf-8')):
        raise ValueError('plan differs from this source release')
    if value['GPU_authorized'] is not False or value['training_authorized'] is not False:
        raise ValueError('preparation is not GPU authority')
    return value


def train_members(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA:
        raise ValueError('not the certified train_core manifest')
    rows = json.loads(raw)['records']
    if len(rows) != 4869 or len({r['stable_image_id'] for r in rows}) != 4869:
        raise ValueError('train_core members missing or repeated')
    from ..data_protocol.schema import stable_image_id
    for row in rows:
        if (row['split'] != 'train' or
                row['stable_image_id'] != stable_image_id('train', row['relative_path']) or
                Path(row['relative_path']).parts[:2] != ('train', 'images') or
                Path(row['annotation_relative_path']).parts[:2] != ('train', 'annotations')):
            raise ValueError('training member role/identity conflict')
    return rows


def cache_binding(*, source_manifest_sha256, contract_sha256):
    for sha in (source_manifest_sha256, contract_sha256):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('missing certified source/contract SHA')
    return dict(protocol=VERSION, seed=0, role='train_core',
                manifest_sha256=TRAIN_SHA, checkpoint_sha256=PARENT_SHA,
                source_manifest_sha256=source_manifest_sha256,
                contract_sha256=contract_sha256, EMA=True, epoch=120,
                input_hw=[768, 1344], inference_dtype='FP32')


def verify_record(record, row, expected_binding):
    if record.get('binding') != expected_binding:
        raise ValueError('teacher cache binding differs')
    if expected_binding.get('profile') == 'CPU_FIXTURE_ONLY':
        valid = fixture_cache_binding(source_manifest_sha256=expected_binding['source_manifest_sha256'],
            contract_sha256=expected_binding['contract_sha256'], manifest_sha256=expected_binding['manifest_sha256'],
            checkpoint_sha256=expected_binding['checkpoint_sha256'])
    else:
        valid = cache_binding(
            source_manifest_sha256=expected_binding['source_manifest_sha256'],
            contract_sha256=expected_binding['contract_sha256'])
    if expected_binding != valid:
        raise ValueError('teacher cache is not seed0 training-only parent EMA')
    if record.get('member') != row:
        raise ValueError('teacher cache image/annotation/member identity conflict')
    views = record.get('views')
    if not isinstance(views, list) or [r.get('view_id') for r in views] != list(range(5)):
        raise ValueError('teacher cache requires exact ordered views0..4')
    for view in views:
        candidates = [Candidate.from_dict(r) for r in view['candidates']]
        if len(candidates) != 600 or {r.view_id for r in candidates} != {view['view_id']}:
            raise ValueError('teacher per-view top600 missing or mixed')
        if len({r.identity for r in candidates}) != 600:
            raise ValueError('duplicate teacher candidate identity')
        if candidates != sorted(candidates, key=lambda r: r.sort_key):
            raise ValueError('teacher candidate ordering differs')
        if [r.view_candidate_rank for r in candidates] != list(range(1, 601)):
            raise ValueError('teacher ranks differ')
    return record


def fixture_cache_binding(*, source_manifest_sha256, contract_sha256, manifest_sha256, checkpoint_sha256):
    """Isolated synthetic provenance; never declares real train/e120 identity."""
    for value in (source_manifest_sha256, contract_sha256, manifest_sha256, checkpoint_sha256):
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('fixture reference SHA required')
    return dict(protocol=VERSION, profile='CPU_FIXTURE_ONLY', role='synthetic_train',
        manifest_sha256=manifest_sha256, checkpoint_sha256=checkpoint_sha256,
        source_manifest_sha256=source_manifest_sha256, contract_sha256=contract_sha256,
        EMA=False, epoch=None, input_hw=[128,128], inference_dtype='FP32')


def from_raw_views(row, raw_views, binding):
    """Convert ONLY admitted training raw FP32 outputs; no model/GT lookup here.

    Whole decode follows original FP32 arithmetic, tile mapping follows frozen C.
    Complete raw outputs and their file SHA must remain server-side as provenance.
    """
    if set(raw_views) != set(range(5)):
        raise ValueError('raw view coverage differs')
    tiles = tile_windows(row['height'], row['width'])
    views = []
    for vid in range(5):
        raw = raw_views[vid]
        logits, boxes = raw['pred_logits'], raw['pred_boxes']
        if (logits.shape != (300, 10) or boxes.shape != (300, 4) or
                logits.dtype != torch.float32 or boxes.dtype != torch.float32 or
                logits.requires_grad or boxes.requires_grad or
                logits.device.type != 'cpu' or boxes.device.type != 'cpu'):
            raise ValueError('teacher raw outputs must be detached CPU FP32 Q300x10')
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
            raise ValueError('invalid teacher raw prediction geometry')
        hw = (row['height'], row['width']) if vid == 0 else (tiles[vid-1].height, tiles[vid-1].width)
        xy = box_convert(boxes, 'cxcywh', 'xyxy') * torch.tensor([hw[1], hw[0]]*2, dtype=torch.float32)
        xy = xy.tolist() if vid == 0 else map_crop_xyxy(xy.tolist(), tiles[vid-1])
        chosen = select_view_candidates(xy, logits.sigmoid().numpy(), view_id=vid)
        views.append(dict(view_id=vid, candidates=[r.as_dict() for r in chosen]))
    record = dict(binding=binding, member=row, views=views)
    return verify_record(record, row, binding)


def select_teacher(record, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if arm == 'T0':
        return []
    allowed = (0,) if arm == 'T1' else tuple(range(5))
    pool = [Candidate.from_dict(r) for v in record['views'] if v['view_id'] in allowed for r in v['candidates']]
    # Identical postprocess for T1/T2; NMS on original coordinates, before score gate.
    ordered = classwise_nms(pool)[:600]
    return [r for r in ordered if r.score >= .5][:300]


class CacheReader:
    """Read an explicitly authenticated training-only index, never infer paths.

    Each compact record retains one file reference containing all five complete
    raw outputs. Source/contract and all raw files are audited at cache admission;
    per-sample reads recheck the compact record's SHA and exact member identity.
    """
    def __init__(self, index_reference, rows, binding):
        from ..localization.data import read_bound
        raw = read_bound(index_reference['path'], index_reference['sha256'], index_reference['bytes'])
        index = json.loads(raw)
        if index.get('binding') != binding or index.get('closed') is not True:
            raise ValueError('teacher cache index binding/closure differs')
        references = index['records']
        if [r['image_id'] for r in references] != [r['stable_image_id'] for r in rows]:
            raise ValueError('teacher cache missing/repeated/reordered/cross-role members')
        self.by_id = {r['image_id']: r['record'] for r in references}
        self.rows = {r['stable_image_id']: r for r in rows}
        self.binding = binding
        # Full one-time admission, with exact reconstruction from raw outputs.
        # Subsequent training reads use the already certified compact SHA.
        for row in rows:
            compact = self(row)
            ref = compact['raw_reference']
            raw = read_bound(ref['path'], ref['sha256'], ref['bytes'])
            payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if payload.get('binding') != binding or payload.get('member') != row:
                raise ValueError('raw teacher source/member differs')
            rebuilt = from_raw_views(row, payload['raw_views'], binding)
            if any(compact[k] != rebuilt[k] for k in ('binding','member','views')):
                raise ValueError('compact teacher cache differs from raw reconstruction')

    def __call__(self, row):
        from ..localization.data import read_bound
        if self.rows.get(row.get('stable_image_id')) != row:
            raise ValueError('teacher read is outside bound training members')
        ref = self.by_id[row['stable_image_id']]
        value = json.loads(read_bound(ref['path'], ref['sha256'], ref['bytes']))
        return verify_record(value, row, self.binding)


@torch.inference_mode()
def teacher_batch(model, originals, *, input_hw=(768, 1344), synthetic=False):
    """Real frozen teacher forward, five views, no GT input. Never dispatched here.

    Caller must admit model/data/runtime and authenticate parent before invoking.
    Production batches are <=4 with a possible final1; input canvas is fixed.
    """
    from torchvision.transforms.v2 import functional as TF
    from torchvision.transforms.v2 import Resize
    if not 1 <= len(originals) <= 4 or (not synthetic and tuple(input_hw)!=(768,1344)):
        raise ValueError('teacher batch/canvas differs')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('teacher must be frozen and eval')
    device=next(model.parameters()).device
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError('teacher inference must be FP32')
    resize=Resize(list(input_hw));result=[{} for _ in originals]
    geometry=[tile_windows(im.height,im.width) for im in originals]
    for vid in range(5):
        pixels=[]
        for im,tiles in zip(originals,geometry):
            view=im if vid==0 else im.crop(tiles[vid-1].crop_box)
            pixels.append(TF.pil_to_tensor(resize(view)).float()/255.)
            if vid: view.close()
        outputs=model(torch.stack(pixels).to(device))
        for i in range(len(originals)):
            result[i][vid]={k:outputs[k][i].float().detach().cpu().clone() for k in ('pred_logits','pred_boxes')}
    return result


def write_cache_member(directory, row, raw_views, binding):
    """Write to a NEW member directory; never overwrites an old cache."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    value=from_raw_views(row,raw_views,binding)
    raw_path=directory/'raw.pt'
    with raw_path.open('xb') as stream:
        torch.save(dict(binding=binding,member=row,raw_views=raw_views),stream)
        stream.flush();os.fsync(stream.fileno())
    value['raw_reference']=dict(path=str(raw_path),bytes=raw_path.stat().st_size,sha256=state.file_sha(raw_path))
    path=directory/'teacher.json'
    with path.open('xb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    return dict(image_id=row['stable_image_id'],record=dict(path=str(path),bytes=path.stat().st_size,sha256=state.file_sha(path)))


def quality_gates(metrics, plan):
    """Already-computed 0..1 COCO metrics only; no scoring or rounding here."""
    rows = []
    for comparison, limits in plan['quality_gates_pp'].items():
        left, right = comparison.split('-')
        for key, minimum in limits.items():
            budget, name = key.split('_', 1)
            a, b = metrics[left][budget][name], metrics[right][budget][name]
            if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in (a,b)):
                raise ValueError('quality input unavailable/nonfinite/outside valid0..1')
            delta = (a-b)*100
            rows.append(dict(comparison=comparison, metric=key, delta_pp=delta,
                             minimum_pp=minimum, passed=delta >= minimum))
    return dict(passed=all(r['passed'] for r in rows), conditions=rows)


def align_teacher(candidates, original_hw, trace, target, input_hw=(768, 1344)):
    """Replay ACTUAL image geometry, not a second random crop invocation.

    Original GT-driven crop remains untouched. Pseudo boxes follow its center
    admission, clipping and min_size1. No teacher influences augmentation draws.
    """
    if trace.get('augmentation_strong') is not False:
        raise ValueError('fixed weak plan rejects unproved strong augmentation geometry')
    if trace.get('image_id') != target['image_id']:
        raise ValueError('teacher/student augmentation image identity differs')
    if tuple(target['original_hw'].tolist()) != tuple(original_hw):
        raise ValueError('teacher/student original dimensions differ')
    h, w = original_hw
    rows = []
    for candidate in candidates:
        b = np.asarray(candidate.box_xyxy, dtype=np.float64).copy()
        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
        if np.any(b[2:] <= b[:2]):
            continue
        hh, ww = h, w
        keep = True
        for step in trace['geometry_steps']:
            op = step['operation']
            if op == 'zoom':
                left, top, right, bottom = step['padding']
                b += np.array([left, top]*2)
                hh += top+bottom; ww += left+right
            elif op == 'crop':
                top, left, height, width = (step[k] for k in ('top', 'left', 'height', 'width'))
                cx, cy = (b[:2]+b[2:])/2
                # torchvision RandomIoUCrop uses strict center-inside inequalities.
                if not (left < cx < left+width and top < cy < top+height):
                    keep = False; break
                b -= np.array([left, top]*2)
                b[[0, 2]] = np.clip(b[[0, 2]], 0, width)
                b[[1, 3]] = np.clip(b[[1, 3]], 0, height)
                hh, ww = height, width
            elif op == 'horizontal_flip':
                b[[0, 2]] = ww-b[[2, 0]]
            else:
                raise ValueError('unrecognized actual augmentation step')
        if not keep or np.any(b[2:]-b[:2] < 1):
            continue
        b *= np.array([input_hw[1]/ww, input_hw[0]/hh]*2)
        if np.any(b[2:]-b[:2] < 1):
            continue
        b /= np.array(input_hw[::-1]*2)
        rows.append((candidate, b))
    corners = torch.tensor(np.asarray([b for _, b in rows]).reshape(-1, 4), dtype=torch.float32)
    boxes = box_convert(corners, 'xyxy', 'cxcywh')
    labels = torch.tensor([c.category_id-1 for c, _ in rows], dtype=torch.long)
    scores = torch.tensor([c.score for c, _ in rows], dtype=torch.float32)
    if rows:
        empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        keep = negative_keep_mask(boxes[None], empty, [target])[0, torch.arange(len(rows)), labels]
    else:
        keep = torch.empty(0, dtype=torch.bool)
    return dict(boxes=boxes[keep], labels=labels[keep], scores=scores[keep],
                identities=[list(c.identity) for (c, _), yes in zip(rows, keep.tolist()) if yes])


class Dataset(RawRoleDataset):
    """Original whole-image sampler/augmentation plus same-image cached teachers."""
    def __init__(self, *, arm, cache_reader=None, cache_identity=None, **kwargs):
        if arm not in ARMS or (arm != 'T0' and (cache_reader is None or cache_identity is None)):
            raise ValueError('arm or authenticated cache reader missing')
        super().__init__(role='train_core', seed=0, **kwargs)
        self.arm, self.cache_reader, self.cache_identity = arm, cache_reader, cache_identity

    def __getitem__(self, key):
        epoch, position, index = key
        if not 109 <= epoch <= 120:
            raise ValueError('outside fixed local12 pass data clock')
        from PIL import Image
        row = self.records[index]
        image_bytes, annotation = self.bound_bytes(row)
        with Image.open(io.BytesIO(image_bytes)) as source:
            original = source.convert('RGB')
        if (original.width, original.height) != (row['width'], row['height']):
            raise ValueError('training decoded dimensions differ')
        raw = parse_raw(annotation, row['stable_image_id'], expected_rows=row['raw_annotation_rows'])
        image, target, trace = self.transform(original, raw, epoch=epoch+9, seed=0, position=position,
                                              image_id=row['stable_image_id'], training=True)
        original.close()
        trace.update(sampler_epoch=epoch, image_sha256=row['image_sha256'], annotation_sha256=row['annotation_sha256'])
        if self.arm != 'T0':
            record = verify_record(self.cache_reader(row), row, self.cache_identity)
            target['teacher'] = align_teacher(select_teacher(record, self.arm),
                                             (row['height'], row['width']), trace, target,
                                             self.transform.input_hw)
        return image, target, trace


def kd_loss(outputs, teachers, matcher):
    """Final normal head only; confidence-weighted positive selected-class KL.

    No negative teacher boxes, features, query priors or GT assignment changes.
    Native matcher costs 2*focal +5*L1 -2*GIoU, alpha=.25/gamma=2.
    """
    logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
    if len(teachers) != len(boxes) or logits.shape != (*boxes.shape[:2], 10):
        raise ValueError('KD batch/shape mismatch')
    clean = []
    for target in teachers:
        t = {k: target[k].detach().to(boxes.device) for k in ('boxes', 'labels', 'scores')}
        n = len(t['scores'])
        if (n > 300 or t['boxes'].shape != (n, 4) or t['labels'].shape != (n,) or
                t['labels'].dtype != torch.long or t['scores'].dtype != torch.float32 or
                t['boxes'].dtype != torch.float32 or
                not torch.isfinite(t['boxes']).all() or not torch.isfinite(t['scores']).all() or
                (t['boxes'][:, 2:] <= 0).any() or
                ((t['scores'] < .5) | (t['scores'] > 1)).any() or
                ((t['labels'] < 0) | (t['labels'] >= 10)).any()):
            raise ValueError('invalid aligned teacher predictions')
        clean.append(t)
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
        raise ValueError('invalid student prediction geometry')
    indices = matcher(dict(pred_logits=logits, pred_boxes=boxes), clean)['indices']
    total = (logits.sum()+boxes.sum())*0
    denominator = sum((t['scores'].sum() for t in clean), boxes.new_zeros(())).clamp_min(1)
    pairs = []
    for batch, (src, dst) in enumerate(indices):
        src, dst = src.to(boxes.device), dst.to(boxes.device)
        t = clean[batch]
        if len(dst) != len(t['scores']):
            raise ValueError('teacher coverage exceeds student matching capacity')
        if not len(src):
            pairs.append([]); continue
        q = t['scores'][dst]
        z = logits[batch, src, t['labels'][dst]]
        # BCE minus Bernoulli entropy = KL; xlogy defines 0*log(0)=0 at q=1.
        kl = F.binary_cross_entropy_with_logits(z, q, reduction='none') + torch.xlogy(q, q) + torch.xlogy(1-q, 1-q)
        sb, tb = boxes[batch, src], t['boxes'][dst]
        l1 = (sb-tb).abs().sum(-1)
        giou = generalized_box_iou(box_convert(sb, 'cxcywh', 'xyxy'), box_convert(tb, 'cxcywh', 'xyxy')).diag()
        total = total+(q*(kl+5*l1+2*(1-giou))).sum()
        pairs.append(list(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist())))
    loss = total/denominator
    if not torch.isfinite(loss):
        raise ValueError('nonfinite KD objective')
    return loss, pairs


class Criterion(torch.nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown KD arm')
        self.base, self.arm = base, arm
        self.last_diagnostics = []

    def recipe(self):
        return dict(supervised=self.base.recipe(), distillation=VERSION, arm=self.arm)

    def forward(self, outputs, targets, **kwargs):
        supervised = [{k: v for k, v in t.items() if k != 'teacher'} for t in targets]
        losses = self.base(outputs, supervised, **kwargs)
        self.last_diagnostics = copy.deepcopy(self.base.last_diagnostics)
        if self.arm != 'T0':
            loss, pairs = kd_loss(outputs, [t['teacher'] for t in targets], self.base.matcher)
            losses['loss_prediction_kd'] = loss
            self.last_diagnostics.append(dict(head='KD_final', pairs=pairs, normalization='max1_sum_teacher_confidence_per_physical_batch'))
        elif any('teacher' in t for t in targets):
            raise ValueError('T0 must not consume teacher supervision')
        return losses


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= UPDATES:
        raise ValueError('KD schedule outside fixed3648')
    if update <= 200:
        return 1e-5*update/200
    return 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*(update-200)/(UPDATES-200)))


def initialize_parent(model, payload, expected_binding):
    if (payload.get('format') != state.FORMAT or payload.get('binding') != expected_binding or
            payload.get('binding_sha256') != digest(expected_binding) or
            expected_binding.get('scope') != 'formal_R18_seed0_e120' or
            payload['engine']['completed_updates'] != 36480 or payload['ema']['updates'] != 36480 or
            payload['ema']['decay'] != .9999 or payload['ema']['warmups'] != 2000):
        raise ValueError('not authenticated seed0 e120 parent EMA')
    layout._check_model_state(payload['ema']['module'], layout._model_layout(model), 'parent EMA')
    model.load_state_dict(payload['ema']['module'], strict=True)
    return tensor_digest(model.state_dict().items())


def load_parent(model, checkpoint_path, binding_path):
    """Explicit production bytes binding before EMA-only extraction; no RNG load."""
    from ..localization.data import read_bound
    binding_raw=read_bound(binding_path,'93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938',15150)
    binding=json.loads(binding_raw)
    raw=read_bound(checkpoint_path,PARENT_SHA,323460353)
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    return initialize_parent(model,payload,binding)


def optimizer_ema(model, config):
    optimizer, ema = _optimizer(model, config), CommonEMA(model)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(1)
    if optimizer.state or ema.updates != 0:
        raise ValueError('fresh state required')
    return optimizer, ema


def save(engine, path):
    from .distill_engine import capture
    path = Path(path)
    if path.exists() or path.with_suffix('.partial').exists():
        raise FileExistsError(path)
    payload = capture(engine)
    partial = path.with_suffix('.partial')
    with partial.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    sha = state.file_sha(partial)
    os.link(partial, path); partial.unlink()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha)


def restore(engine, reference):
    from .distill_engine import capture
    try:
        raw = Path(reference['path']).read_bytes()
        if len(raw) != reference['bytes'] or hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('KD state SHA/size mismatch')
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        current = capture(engine)
        if set(payload) != set(current) or any(payload[k] != current[k] for k in
                ('protocol', 'binding', 'torch_version', 'device', 'behavior', 'model_layout', 'optimizer_layout')):
            raise ValueError('KD runtime/arm/source/cache/plan binding mismatch')
        engine.validate_state_dict(payload['engine'])
        u = payload['engine']['completed_updates']
        for name, module, values in (('model', engine.model, payload['model']), ('EMA', engine.ema.module, payload['ema']['module'])):
            layout._check_model_state(values, layout._model_layout(module), name)
        if payload['ema']['updates'] != u or payload['ema']['decay'] != .9999 or payload['ema']['warmups'] != 2000:
            raise ValueError('KD EMA clock differs')
        layout._check_optimizer(payload['optimizer'], current['optimizer_layout'], engine.model, u)
        if any(g['lr'] != learning_rate(max(1, u)) for g in payload['optimizer']['param_groups']):
            raise ValueError('KD LR differs')
        for group in current['optimizer_layout']['groups']:
            for name, param_id in zip(group['names'], group['ids']):
                if int(payload['optimizer']['state'].get(param_id, {}).get('step', 0)) != payload['engine']['parameter_optimizer_steps'][name]:
                    raise ValueError('KD exact parameter clock differs')
        state.validate_rng(payload['rng'], engine.device)
        modes = payload['model_training']
        if set(modes) != {n for n, _ in engine.model.named_modules()} or any(type(v) is not bool for v in modes.values()):
            raise ValueError('KD module modes differ')
        engine.model.load_state_dict(payload['model'], strict=True)
        for name, module in engine.model.named_modules():
            module.training = modes[name]
        engine.optimizer.load_state_dict(payload['optimizer']); engine.ema.load_state_dict(payload['ema'])
        engine.load_state_dict(payload['engine']); state.restore_rng(payload['rng'], engine.device)
        if not layout._equal(payload, capture(engine)):
            raise ValueError('KD full restored state/RNG differs')
        return dict(full_state_RNG_exact=True, completed_updates=u)
    except BaseException:
        engine.failed = True
        raise
