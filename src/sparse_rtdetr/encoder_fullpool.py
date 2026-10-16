"""Instance-only A3b encoder auxiliary replacement; frozen classes stay untouched.

Install after constructing model/criterion AND any EMA/deep copies, and before
wrapping in DDP. Installed instances must not subsequently be deep-copied: the
adapter deliberately binds only those training instances. No parameters,
checkpoint keys, decoder queries, DN paths or evaluation behavior are changed.
The per-image matcher reuses the frozen cost and Hungarian implementation;
packed valid losses retain the frozen criterion's GT-count normalization.
VFL diagnostics partition matched-class entries versus all other class entries;
their sum equals weighted encoder VFL up to floating-point reduction order.
"""
from __future__ import annotations

import importlib
import time
import types

import torch
import torch.nn.functional as F


_MASK = '_fullpool_valid_mask'


class EncoderNumericalError(RuntimeError):
    """A legal anchor failed validation; its evidence must not become a mask."""
    def __init__(self, reason, evidence):
        super().__init__('legal encoder anchor numerical anomaly: ' + reason)
        self.evidence = evidence


def _pack(outputs, targets, indices):
    """Remove invalid entries, retaining all valid negatives and GT normalization."""
    mask = outputs[_MASK]
    logits = outputs['pred_logits'][mask].unsqueeze(0)
    boxes = outputs['pred_boxes'][mask].unsqueeze(0)
    src, dst, po, to = [], [], 0, 0
    for valid, target, (i, j) in zip(mask, targets, indices):
        mapping = valid.long().cumsum(0) - 1
        src.append(mapping[i.to(valid.device)] + po)
        dst.append(j.to(valid.device) + to)
        po += int(valid.sum())
        to += len(target['labels'])
    packed_targets = [{k: torch.cat([t[k] for t in targets]) for k in ('labels', 'boxes')}]
    packed_indices = [(torch.cat(src), torch.cat(dst))]
    return {'pred_logits': logits, 'pred_boxes': boxes}, packed_targets, packed_indices


class EncoderFullPoolAdapter:
    def __init__(self, model, criterion, mode='full_valid', observe=True):
        if mode not in ('original', 'full_valid'):
            raise ValueError('mode must be original or full_valid')
        candidates = [m for m in model.modules() if hasattr(m, '_get_decoder_input') and hasattr(m, '_select_topk')]
        if len(candidates) != 1:
            raise ValueError('expected exactly one RTDETR decoder')
        decoder = candidates[0]
        if getattr(decoder, '_fullpool_adapter_installed', False):
            raise ValueError('adapter already installed')
        if decoder.query_select_method != 'default' or not decoder.aux_loss:
            raise ValueError('fixed A3b default selection with aux_loss required')
        if set(criterion.losses) != {'vfl', 'boxes'} or criterion.boxes_weight_format is not None:
            raise ValueError('fixed A3b VFL + boxes criterion required')
        self.stats = {}
        self._restores = []
        self._hooks = []
        self._active = set()
        self._capture = None
        self.mode = mode
        self.observe = observe
        self.criterion = criterion
        self.decoder = decoder
        decoder._fullpool_adapter_installed = True
        self._box_ops = importlib.import_module(criterion.__class__.__module__.rsplit('.', 1)[0] + '.box_ops')

        def patch(obj, name, callback):
            had = name in obj.__dict__
            previous = obj.__dict__.get(name)
            original = getattr(obj, name)
            self._restores.append((obj, name, had, previous))
            setattr(obj, name, types.MethodType(lambda instance, *a, **kw: callback(original, *a, **kw), obj))

        if mode == 'full_valid':
            def anchors(original, *a, **kw):
                result = original(*a, **kw)
                if self._capture is not None:
                    self._capture['anchor_mask'] = result[1]
                return result

            def select(original, memory, logits, coords, topk):
                if self._capture is not None:
                    valid = self._capture['anchor_mask'].squeeze(-1).expand(logits.shape[:2])
                    self._validate(valid, logits, coords=coords)
                    self._capture.update(logits=logits, coords=coords)
                return original(memory, logits, coords, topk)

            def inputs(original, *a, **kw):
                if not decoder.training:
                    return original(*a, **kw)
                capture = {}
                self._capture = capture
                try:
                    result = original(*a, **kw)
                finally:
                    self._capture = None
                logits, coords = capture['logits'], capture['coords']
                # Only the model's predefined anchor mask defines eligibility.
                # Legal-position anomalies already failed before top-k selection.
                valid = capture['anchor_mask'].squeeze(-1).expand(logits.shape[:2])
                safe_coords = torch.where(valid[..., None], coords, torch.zeros_like(coords))
                boxes = safe_coords.sigmoid()
                self._pending_mask = valid
                return result[0], result[1], [boxes], [logits]

            def output_hook(module, args, output):
                if module.training:
                    if len(output.get('enc_aux_outputs', [])) != 1:
                        raise RuntimeError('expected exactly one encoder auxiliary output')
                    output['enc_aux_outputs'][0][_MASK] = self._pending_mask
                    del self._pending_mask
                return output

            patch(decoder, '_generate_anchors', anchors)
            patch(decoder, '_select_topk', select)
            patch(decoder, '_get_decoder_input', inputs)
            self._hooks.append(decoder.register_forward_hook(output_hook))

        def forward(original, outputs, targets, **kw):
            self.stats.clear()
            for enc in outputs.get('enc_aux_outputs', []):
                if _MASK in enc:
                    self._validate(enc[_MASK], enc['pred_logits'], boxes=enc['pred_boxes'])
            self._active = {id(o) for o in outputs.get('enc_aux_outputs', [])}
            try:
                return original(outputs, targets, **kw)
            finally:
                self._active.clear()

        def matcher(original, outputs, targets, **kw):
            if id(outputs) not in self._active:
                return original(outputs, targets, **kw)
            start = time.perf_counter()
            if _MASK not in outputs:
                result = original(outputs, targets, **kw)
                counts = [outputs['pred_logits'].shape[1]] * len(targets)
            else:
                indices, counts = [], []
                for b, target in enumerate(targets):
                    valid = outputs[_MASK][b].nonzero(as_tuple=True)[0]
                    counts.append(len(valid))
                    if len(valid) == 0 or len(target['labels']) == 0:
                        i = j = torch.empty(0, dtype=torch.int64)
                    else:
                        compact = {k: outputs[k][b:b+1, valid] for k in ('pred_logits', 'pred_boxes')}
                        i, j = original(compact, [target], **kw)['indices'][0]
                        i = valid[i.to(valid.device)].cpu()
                    indices.append((i, j))
                result = {'indices': indices}
            self.stats['encoder_match_seconds'] = time.perf_counter() - start
            self.stats['encoder_candidates_per_image'] = counts
            self.stats['encoder_matches_per_image'] = [len(i) for i, _ in result['indices']]
            # Actual assignment used by the losses, in ORIGINAL token coordinates.
            # CPU int64 pairs (prediction index, per-image target index), no graph.
            self.stats['encoder_match_indices'] = [(i.detach().cpu(), j.detach().cpu()) for i, j in result['indices']]
            return result

        def meta(original, loss, outputs, targets, indices):
            if _MASK in outputs:
                outputs, targets, indices = _pack(outputs, targets, indices)
            return original(loss, outputs, targets, indices)

        def loss(original, name, outputs, targets, indices, num_boxes, **kw):
            encoder = id(outputs) in self._active
            if _MASK in outputs:
                outputs, targets, indices = _pack(outputs, targets, indices)
            if name == 'vfl' and outputs['pred_logits'].shape[1] == 0:
                result = {'loss_vfl': outputs['pred_logits'].sum() * 0}
            else:
                result = original(name, outputs, targets, indices, num_boxes, **kw)
            if encoder and name == 'vfl' and self.observe:
                self._observe_vfl(outputs, targets, indices, num_boxes)
            return result

        patch(criterion, 'forward', forward)
        patch(criterion.matcher, 'forward', matcher)
        patch(criterion, 'get_loss_meta_info', meta)
        patch(criterion, 'get_loss', loss)

    @torch.no_grad()
    def _validate(self, valid, logits, *, coords=None, boxes=None):
        if valid.dtype != torch.bool or valid.shape != logits.shape[:2]:
            raise RuntimeError('invalid predefined encoder anchor mask')
        if boxes is None:
            # Match the loss path's sigmoid dtype/shape, including underflow.
            boxes = torch.where(valid[..., None], coords, torch.zeros_like(coords)).sigmoid()
        bad_logits = valid & ~torch.isfinite(logits).all(-1)
        bad_coords = (valid & ~torch.isfinite(coords).all(-1)) if coords is not None else torch.zeros_like(valid)
        bad_boxes = valid & (~torch.isfinite(boxes).all(-1) | (boxes[..., 2:] <= 0).any(-1)
                             | (boxes[..., 2:] > 1).any(-1))
        if bool((bad_logits | bad_coords | bad_boxes).any()):
            evidence = {name: value.detach().cpu().clone() for name, value in {
                'predefined_valid_mask': valid, 'logits': logits, 'boxes': boxes,
                'bad_logits': bad_logits, 'bad_unactivated_coords': bad_coords,
                'bad_boxes': bad_boxes, **({'unactivated_coords': coords} if coords is not None else {}),
            }.items()}
            reasons = [name for name in ('bad_logits', 'bad_unactivated_coords', 'bad_boxes')
                       if bool(evidence[name].any())]
            raise EncoderNumericalError(','.join(reasons), evidence)

    @torch.no_grad()
    def _observe_vfl(self, outputs, targets, indices, num_boxes):
        logits = outputs['pred_logits'].detach()
        idx = self.criterion._get_src_permutation_idx(indices)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        positive = torch.zeros_like(logits, dtype=torch.bool)
        positive[idx[0], idx[1], labels] = True
        source = outputs['pred_boxes'][idx].detach()
        target = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])
        ops = self._box_ops
        iou = ops.box_iou(ops.box_cxcywh_to_xyxy(source), ops.box_cxcywh_to_xyxy(target))[0].diag()
        score = torch.zeros_like(logits)
        score[idx[0], idx[1], labels] = iou.to(score.dtype)
        weight = self.criterion.alpha * logits.sigmoid().pow(self.criterion.gamma) * (~positive) + score
        terms = F.binary_cross_entropy_with_logits(logits, score, weight=weight, reduction='none')
        scale = self.criterion.weight_dict['loss_vfl'] / num_boxes
        self.stats['encoder_vfl_positive'] = terms[positive].sum() * scale
        self.stats['encoder_vfl_negative'] = terms[~positive].sum() * scale

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        for obj, name, had, previous in reversed(self._restores):
            if had:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self._hooks.clear()
        self._restores.clear()
        self.decoder._fullpool_adapter_installed = False
        self._active.clear()


def install(model, criterion, mode='full_valid', observe=True):
    """Return a removable adapter with detached last-call encoder diagnostics."""
    return EncoderFullPoolAdapter(model, criterion, mode, observe)
