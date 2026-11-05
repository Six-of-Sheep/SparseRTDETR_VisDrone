"""Task B of TASK_CACHED_FOLLOWUP_002: one pre-registered horizontal shift of closed cached lists.

CPU only.  Never constructs a detector, never touches CUDA, never reads holdout or the 1602 set,
never rewrites a prediction or GT file.  Existing outputs are immutable.

The single perturbation was fixed by the task before any of these scores existed: every box of every
listed top600 list, all classes and sizes, moves by dx = -0.25 original pixels (both x edges of
bbox_xyxy), dy = dw = dh = 0.  Scores, classes, query ids, order and budget are kept.  The derived
list is built before the certified bridge region filter and then scored once by the unchanged
certified path (raw-ignore filter, class score 0 as crowd, COCOeval maxDets 100 per image per class).
No clip, rounding, NMS or refill is added.  Nothing here searches dx.

The six original A/C lists are replayed through COCO.evaluate only, without accumulate or summarize,
to read the certified one-to-one pairs for the offset table. Their prior metrics are reused from
SHA-bound run-001 files, and per-GT TP flags must equal those arrays. Six match-only calls and their
cost are counted separately from the eight complete baseline/derived score calls.

usage: python -B -I followup_shift.py --config CFG --out DIR --commit SHA --phases coords lists r23 aggregate
"""
from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import core as C
import run_channels as RC

DX = -0.25  # fixed by TASK_CACHED_FOLLOWUP_002 section 3; never tuned, never swept
SHIFT = {'dx_px': DX, 'dy_px': 0.0, 'dw_px': 0.0, 'dh_px': 0.0, 'frame': 'original image pixels',
         'applies_to': 'every box of the list, all classes and sizes, before the bridge region filter',
         'kept': 'score, category, query id, view fields, order and budget'}
HISTORY_TAGS = ('A-seed0', 'C-seed0', 'A-seed1', 'C-seed1', 'A-seed2', 'C-seed2')
REPORT_KEYS = ('AP', 'APS', 'AP75_small', 'APM', 'APL', 'AP50', 'AP75')
USED_KEYS = (('all', C.AREA_ALL, 0.5), ('all', C.AREA_ALL, 0.75), ('small', C.AREA_SMALL, 0.5), ('small', C.AREA_SMALL, 0.75))
OFFSET_KEYS = ('cx', 'cy', 'w', 'h')
require = RC.require
sha = RC.sha
ref_of = RC.ref_of
publish = RC.publish


def ukey(area, t):
    return f"{area}_{str(t).replace('.', '')}"


def shift_predictions(preds, dx=DX):
    """Independent derived list; x edges move by dx, everything else is copied unchanged."""
    out, inexact = [], 0
    for p in preds:
        require('bbox_xyxy' in p and 'bbox' not in p and len(p['bbox_xyxy']) == 4, 'cached box is not bbox_xyxy')
        b = p['bbox_xyxy']
        x0, x1 = b[0] + dx, b[2] + dx
        if x0 - dx != b[0] or x1 - dx != b[2] or (x1 - x0) != (b[2] - b[0]):
            inexact += 1
        q = dict(p)
        q['bbox_xyxy'] = [x0, b[1], x1, b[3]]
        out.append(q)
    return out, inexact


def box_stats(preds, size_of):
    """Prediction coordinates outside the image: evidence that the certified path never clips."""
    n = {'boxes': len(preds), 'x0_below_0': 0, 'x1_above_width': 0, 'y0_below_0': 0, 'y1_above_height': 0}
    for p in preds:
        w, h = size_of[p['image_id']]
        b = p['bbox_xyxy']
        n['x0_below_0'] += b[0] < 0
        n['x1_above_width'] += b[2] > w
        n['y0_below_0'] += b[1] < 0
        n['y1_above_height'] += b[3] > h
    return {k: int(v) for k, v in n.items()}


def class_ap(e):
    """Per-class AP on the same precision array the certified metrics use (maxDets 100)."""
    P = e.eval['precision']

    def per(t_slice, area):
        out = []
        for k in range(P.shape[2]):
            cell = P[t_slice, :, k, area, 2]
            cell = cell[cell > -1]
            out.append(float(cell.mean()) if len(cell) else None)
        return out
    return {'AP': per(slice(None), 0), 'APS': per(slice(None), 1), 'AP75_small': per(5, 1),
            'APM': per(slice(None), 2), 'APL': per(slice(None), 3)}


def class_gt(e, area_index):
    p = e.params
    K, I, A = len(p.catIds), len(p.imgIds), len(p.areaRng)
    out = []
    for k in range(K):
        n = 0
        for i in range(I):
            x = e.evalImgs[k * A * I + area_index * I + i]
            if x is not None:
                n += int(np.count_nonzero(x['gtIgnore'] == 0))
        out.append(n)
    return out


def residuals(rt, gt, dt):
    """Certified one-to-one pair of each matched in-range GT: detection minus GT (cx, cy, w, h)."""
    out = np.full((gt.n, 4), np.nan)
    match = rt.gt_match[gt.id]
    ok = np.logical_and.reduce((rt.gt_used[gt.id], rt.gt_nonig[gt.id], match > 0))
    d = dt.bbox[match[ok] - 1]
    g = gt.bbox[ok]
    out[ok] = np.stack((d[:, 0] + d[:, 2] / 2 - g[:, 0] - g[:, 2] / 2, d[:, 1] + d[:, 3] / 2 - g[:, 1] - g[:, 3] / 2,
                        d[:, 2] - g[:, 2], d[:, 3] - g[:, 3]), 1)
    return out


def evaluate_list(ctx, preds, macro_fn=None, previous=None):
    """Certified bridge and matcher, keeping what the follow-up needs; no oracle is computed."""
    dataset, det, counts = ctx.ev.bridge_inputs(ctx.rows, ctx.parsed, preds)
    kept = RC.map_det_to_pred(preds, det, ctx.rows)
    if previous is None:
        e, values = C.evaluate_certified(dataset, det)
    else:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO()
            coco.dataset = dataset
            coco.createIndex()
            cd = coco.loadRes(det)
            e = COCOeval(coco, cd, 'bbox')
            e.params.imgIds = sorted(r['id'] for r in dataset['images'])
            e.params.catIds = list(range(1, 11))
            e.params.maxDets = [1, 10, 100]
            e.evaluate()
        require(not e.eval, 'match-only replay unexpectedly accumulated metrics')
        values = previous['certified_metrics_reproduced']
    gt, dt = C.gt_table(dataset), C.det_table(det)
    runs = {(a, t): C.collect_run(e, a, t, gt.n) for a in (C.AREA_ALL, C.AREA_SMALL) for t in (0.5, 0.75)}
    if previous is None:
        require(C.check_port(e, runs) == 0.0, 'accumulate port differs')
    used = {ukey(name, t): runs[(a, t)].gt_used[gt.id].copy() for name, a, t in USED_KEYS}
    nonig = {name: runs[(a, 0.5)].gt_nonig[gt.id].copy() for name, a in (('all', C.AREA_ALL), ('small', C.AREA_SMALL))}
    result = {'values': values, 'counts': counts, 'kept': kept, 'gt': gt, 'used': used, 'nonig': nonig,
              'res_all': residuals(runs[(C.AREA_ALL, 0.5)], gt, dt), 'res_small': residuals(runs[(C.AREA_SMALL, 0.5)], gt, dt),
              'groups': C.group_metrics(e, ctx.groups) if previous is None else previous['groups'],
              'class_ap': class_ap(e) if previous is None else {},
              'class_gt': {'all': class_gt(e, 0), 'small': class_gt(e, 1), 'medium': class_gt(e, 2), 'large': class_gt(e, 3)}}
    if macro_fn is not None:
        result['macro'] = macro_fn(e.eval)
    for key in ('AP', 'AP50', 'AP75', 'APS', 'AP75_small'):
        require(result['groups']['ALL'][key] == values[key], 'group ALL differs on ' + key)
    del e, runs, dataset, det, dt
    gc.collect()
    return result


def block(values):
    values = values[~np.isnan(values[:, 0])]
    n = len(values)
    if n == 0:
        return {'n': 0}
    out = {'n': n}
    for k, key in enumerate(OFFSET_KEYS):
        out[key] = {'mean': round(float(values[:, k].mean()), 5), 'median': round(float(np.median(values[:, k])), 5)}
    return out


def offset_summary(res_all, res_small, gt, size, seq):
    nc = np.logical_not(gt.crowd)
    small = np.logical_and(nc, gt.area <= 1024)
    bands = {'small': small, 'medium': np.logical_and.reduce((nc, gt.area > 1024, gt.area <= 9216)),
             'large': np.logical_and(nc, gt.area > 9216)}
    out = {'pair_rule': 'certified one-to-one COCO match at IoU .5 (maxDets 100); small rows use the small-area run, medium/large the all-area run',
           'by_area_band': {'small': block(res_small[small]), 'medium': block(res_all[bands['medium']]),
                            'large': block(res_all[bands['large']]),
                            'small_from_all_area_run': block(res_all[small])},
           'small_by_image_size': {name: block(res_small[np.logical_and(small, size == name)]) for name in sorted(set(size.tolist()))},
           'small_by_class': {str(c): block(res_small[np.logical_and(small, gt.cat == c)]) for c in range(1, 11)}}
    valid = np.logical_and(small, ~np.isnan(res_small[:, 0]))
    rows = []
    for name in sorted(set(seq[valid].tolist())):
        sel = np.logical_and(valid, seq == name)
        rows.append((name, int(sel.sum()), float(res_small[sel, 0].mean()), float(res_small[sel, 1].mean())))
    big = [r for r in rows if r[1] >= 20]
    mx = np.array([r[2] for r in big])
    my = np.array([r[3] for r in big])

    def stat(fn, v):
        return round(float(fn(v)), 5) if len(v) else None
    out['small_sequences'] = {
        'with_at_least_20_pairs': len(big), 'cx_positive': int((mx > 0).sum()), 'cy_positive': int((my > 0).sum()),
        'cx_mean_of_sequence_means': stat(np.mean, mx), 'cx_median_of_sequence_means': stat(np.median, mx),
        'cy_mean_of_sequence_means': stat(np.mean, my), 'cy_median_of_sequence_means': stat(np.median, my),
        'per_sequence': [{'sequence': r[0], 'pairs': r[1], 'cx_mean': round(r[2], 4), 'cy_mean': round(r[3], 4)} for r in rows]}
    return out


def flips(a, b, mask):
    return {'gained': int(np.logical_and.reduce((mask, b, np.logical_not(a))).sum()),
            'lost': int(np.logical_and.reduce((mask, a, np.logical_not(b))).sum()),
            'both': int(np.logical_and.reduce((mask, a, b)).sum()),
            'neither': int(np.logical_and.reduce((mask, np.logical_not(a), np.logical_not(b))).sum())}


def compare(orig, shifted, ctx, gt_meta):
    gt = orig['gt']
    require(np.array_equal(gt.id, shifted['gt'].id) and np.array_equal(gt.bbox, shifted['gt'].bbox), 'GT table differs')
    small = orig['nonig']['small']
    require(np.array_equal(small, shifted['nonig']['small']), 'small GT differs')
    allnon = orig['nonig']['all']
    bins = np.full(gt.n, -1)
    bins[small] = C.min_side_bin(gt.bbox[small, 2], gt.bbox[small, 3])
    out = {'delta_pp': {k: (shifted['values'][k] - orig['values'][k]) * 100. for k in orig['values']},
           'tp_flips': {}, 'filter': {}}
    for name, a, t in USED_KEYS:
        k = ukey(name, t)
        mask = small if name == 'small' else allnon
        out['tp_flips'][k] = flips(orig['used'][k], shifted['used'][k], mask)
        if name == 'small':
            out['tp_flips'][k + '_by_min_side'] = {C.BIN_LABELS[b]: flips(orig['used'][k], shifted['used'][k], np.logical_and(small, bins == b))
                                                    for b in range(4)}
            out['tp_flips'][k + '_by_image_size'] = {s: flips(orig['used'][k], shifted['used'][k], np.logical_and(small, gt_meta['size'] == s))
                                                     for s in sorted(set(gt_meta['size'].tolist()))}
    a, b = set(orig['kept'].tolist()), set(shifted['kept'].tolist())
    out['filter'] = {'detections_after_original': len(a), 'detections_after_shifted': len(b),
                     'dropped_by_shift': len(a - b), 'admitted_by_shift': len(b - a)}
    out['group_delta_pp'] = {g: {k: (shifted['groups'][g][k] - orig['groups'][g][k]) * 100. for k in ('AP', 'APS', 'AP75_small', 'AP50', 'AP75')}
                             for g in orig['groups']}
    out['class_delta_pp'] = {k: [None if (x is None or y is None) else (y - x) * 100. for x, y in zip(orig['class_ap'][k], shifted['class_ap'][k])]
                             for k in orig['class_ap']}
    return out


def gt_meta_of(ctx, gt):
    size = np.array([f"{ctx.rows[i - 1]['width']}x{ctx.rows[i - 1]['height']}" for i in gt.img])
    seq = np.array([ctx.sequence[i - 1] for i in gt.img])
    return {'size': size, 'seq': seq}


def list_record(tag, ref, ev, ctx, gt_meta, role):
    return {'tag': tag, 'role': role, 'prediction': ref, 'values': ev['values'], 'spatial_filter_counts': ev['counts'],
            'groups': ev['groups'], 'class_ap': ev['class_ap'], 'class_gt': ev['class_gt'],
            'offsets_one_to_one': offset_summary(ev['res_all'], ev['res_small'], ev['gt'], gt_meta['size'], gt_meta['seq'])}


def save_arrays(path, orig, shifted):
    arrays = {'gt_id': orig['gt'].id, 'small': orig['nonig']['small'], 'res_small_orig': orig['res_small'], 'res_small_shift': shifted['res_small'],
              'res_all_orig': orig['res_all'], 'res_all_shift': shifted['res_all'],
              'kept_orig': orig['kept'], 'kept_shift': shifted['kept']}
    for k in orig['used']:
        arrays['used_orig_' + k] = orig['used'][k]
        arrays['used_shift_' + k] = shifted['used'][k]
    RC.save_npz(path, **arrays)


def process_history(ctx, cfg, out, commit, tag):
    jpath = Path(out) / f'shift-{tag}.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing output has another commit: ' + tag)
        return
    started, cpu = time.perf_counter(), time.process_time()
    resources = ctx.guard(out)
    arm, seed = tag.split('-seed')
    spec = ctx.spec(int(seed))
    ref = spec['evaluation']['predictions'][arm]['600']
    certified = spec['metrics'][arm]['600']['COCO_bridge']['metrics']
    base = cfg['baseline_run']
    old_json = Path(base['dir']) / f'list-{tag}.json'
    old_npz = Path(base['dir']) / f'list-{tag}.npz'
    require(sha(old_json) == base['sha256'][old_json.name] and sha(old_npz) == base['sha256'][old_npz.name], 'run-001 output differs: ' + tag)
    old = json.loads(old_json.read_bytes())
    require(old['prediction']['sha256'] == ref['sha256'], 'run-001 used another prediction file: ' + tag)
    preds = json.loads(ctx.j.exact(ref, contents=True))
    size_of = {r['stable_image_id']: (r['width'], r['height']) for r in ctx.rows}
    stats_orig = box_stats(preds, size_of)
    orig = evaluate_list(ctx, preds, previous=old)
    diffs = {k: abs(orig['values'][k] - certified[k]) for k in certified}
    require(set(orig['values']) == set(certified) and max(diffs.values()) == 0.0, f'{tag}: certified replay differs {diffs}')
    require(orig['values'] == old['certified_metrics_reproduced'], f'{tag}: replay differs from run-001')
    with np.load(old_npz) as z:
        for k in orig['used']:
            require(np.array_equal(orig['used'][k], z['used_' + k]), f'{tag}: per-GT TP flags differ from run-001 on {k}')
        require(np.array_equal(orig['gt'].id, z['gt_id']), f'{tag}: GT ids differ from run-001')
    shifted, inexact = shift_predictions(preds)
    stats_shift = box_stats(shifted, size_of)
    unshift, _ = shift_predictions(shifted, -DX)
    round_trip = sum(1 for p, q in zip(preds, unshift) if p != q)
    same_fields = all(set(p) == set(q) and all(p[k] == q[k] for k in p if k != 'bbox_xyxy')
                      and p['bbox_xyxy'][1] == q['bbox_xyxy'][1] and p['bbox_xyxy'][3] == q['bbox_xyxy'][3]
                      for p, q in zip(preds, shifted))
    require(same_fields and len(shifted) == len(preds), f'{tag}: shift changed a non-x field')
    del preds, unshift
    gc.collect()
    derived_path = Path(out) / ('derived-' + tag + '-top600.json')
    publish(derived_path, shifted)
    derived_ref = ref_of(derived_path)
    der = evaluate_list(ctx, shifted)
    del shifted
    gc.collect()
    gt_meta = gt_meta_of(ctx, orig['gt'])
    record = {'tag': tag, 'seed': int(seed), 'arm': arm, 'commit': commit, 'shift': SHIFT,
              'checkpoint_sha256': spec['identity']['checkpoint_sha256'],
              'original': list_record(tag, ref, orig, ctx, gt_meta, 'COCO evaluate match-only replay; prior metrics reused from bound run-001, no accumulate/summarize'),
              'shifted': list_record(tag + '-shift', derived_ref, der, ctx, gt_meta, 'derived list, one certified score'),
              'reused_prior_metrics_equal_certified': True, 'baseline_metrics_recomputed': False,
              'per_GT_TP_flags_equal_run001': True,
              'shift_exactness': {'boxes_with_inexact_x_shift_or_width': inexact, 'round_trip_rows_not_bitwise_equal': round_trip,
                                  'non_x_fields_identical': same_fields},
              'box_bounds': {'original': stats_orig, 'shifted': stats_shift},
              'comparison': compare(orig, der, ctx, gt_meta),
              'COCOeval_full_calls': {'derived_score': 1}, 'COCOeval_match_only_calls': 1,
              'resource_at_start': resources, 'cost': RC.cost(started, cpu),
              'cuda_initialized': ctx.torch.cuda.is_initialized()}
    save_arrays(Path(out) / f'shift-{tag}.npz', orig, der)
    publish(jpath, record)
    print(json.dumps({'tag': tag, 'delta_APS_pp': record['comparison']['delta_pp']['APS'],
                      'delta_AP75_small_pp': record['comparison']['delta_pp']['AP75_small'], 'cost': record['cost']}), flush=True)
    del orig, der
    gc.collect()


def load_r23_module(cfg):
    r = cfg['r23']
    manifest = Path(r['release_manifest'])
    require(sha(manifest) == r['release_manifest_sha256'], 'r23 release manifest differs')
    rows = json.loads(manifest.read_bytes())['source_files']
    want = next(x['sha256'] for x in rows if x['path'] == 'tools/flipview/flipview.py')
    return RC.load_by_path('frozen_flipview', Path(r['release_source']) / 'tools/flipview/flipview.py', want), want


def phase_r23(ctx, cfg, out, commit):
    jpath = Path(out) / 'shift-R23-seed0.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing r23 output has another commit')
        return
    started, cpu = time.perf_counter(), time.process_time()
    resources = ctx.guard(out)
    r = cfg['r23']
    flipview, flipview_sha = load_r23_module(cfg)
    comparison = json.loads(Path(r['comparison']).read_bytes())
    require(sha(r['comparison']) == r['comparison_sha256'], 'r23 comparison differs')
    records = json.loads(Path(r['image_records']).read_bytes())
    require(sha(r['image_records']) == r['image_records_sha256'], 'r23 image records differ')
    flip_result = json.loads(Path(r['flip_result']).read_bytes())
    require(sha(r['flip_result']) == r['flip_result_sha256'], 'r23 FLIP result differs')
    spec0 = ctx.spec(0)
    certified_ref = spec0['evaluation']['predictions']['A']['600']
    # 1. identity of r23's A0 against the certified A-seed0 list
    a0 = json.loads(ctx.j.exact(r['A0'], contents=True))
    certified = json.loads(ctx.j.exact(certified_ref, contents=True))
    fields = ('image_id', 'query_id', 'category_id', 'score', 'bbox_xyxy')
    a0_identity = {'rows_r23_A0': len(a0), 'rows_certified_A_seed0': len(certified),
                   'key_sets_r23_A0': sorted({tuple(sorted(p)) for p in a0}), 'key_sets_certified': sorted({tuple(sorted(p)) for p in certified}),
                   'rows_fieldwise_identical': sum(1 for p, q in zip(a0, certified) if all(p.get(k) == q.get(k) for k in fields)),
                   'r23_plan_parent_checkpoint_note': 'r23 COMPARISON.anchor_reproduction names the same certified file (SHA 95eb697c...)'}
    a0_identity['identical'] = (len(a0) == len(certified) and a0_identity['rows_fieldwise_identical'] == len(a0)
                                and a0_identity['key_sets_r23_A0'] == a0_identity['key_sets_certified'])
    require(a0_identity['identical'], 'r23 A0 does not match certified A-seed0 identity')
    require(comparison['anchor_reproduction']['certified_predictions']['sha256'] == certified_ref['sha256'],
            'r23 anchor names another certified file')
    del certified
    gc.collect()
    # 2. R23 list: same names and scores as r23 A0, boxes changed only where the run says
    r23 = json.loads(ctx.j.exact(r['R23'], contents=True))
    require(len(r23) == len(a0), 'R23 length differs from A0')
    names_scores = all(p['image_id'] == q['image_id'] and p['query_id'] == q['query_id'] and p['category_id'] == q['category_id']
                       and p['score'] == q['score'] for p, q in zip(r23, a0))
    require(names_scores, 'R23 names or scores differ from A0')
    changed = defaultdict(int)
    for p, q in zip(r23, a0):
        if p['bbox_xyxy'] != q['bbox_xyxy']:
            changed[p['image_id']] += 1
    per_image_ok = all(changed.get(x['image_id'], 0) == x['changed_entries_at_600'] for x in records)
    total = sum(changed.values())
    require(per_image_ok and total == flip_result['checks']['changed_entries_at_600_total'], 'R23 changed rows differ from the r23 run records')
    del a0
    gc.collect()
    # 3. one reproduction of the original R23 list (new score 1 of 8), then its shifted copy
    size_of = {x['stable_image_id']: (x['width'], x['height']) for x in ctx.rows}
    stats_orig = box_stats(r23, size_of)
    orig = evaluate_list(ctx, r23, macro_fn=flipview.per_class_macro_metrics)
    macro = orig['macro']
    published = comparison['scores']['R23']['600']
    macro_diff = {k: abs(macro[k] - published[k]) for k in published}
    require(max(macro_diff.values()) == 0.0, f'R23 class-equal-weight reproduction differs {macro_diff}')
    require(orig['counts'] == comparison['spatial_filter_counts']['R23']['600'], 'R23 filter counts differ')
    stats_form = comparison['scores_coco_stats_form']['R23']['600']['AP75_small']
    shifted, inexact = shift_predictions(r23)
    stats_shift = box_stats(shifted, size_of)
    del r23
    gc.collect()
    derived_path = Path(out) / 'derived-R23-seed0-top600.json'
    publish(derived_path, shifted)
    derived_ref = ref_of(derived_path)
    der = evaluate_list(ctx, shifted)
    del shifted
    gc.collect()
    gt_meta = gt_meta_of(ctx, orig['gt'])
    record = {'tag': 'R23-seed0', 'commit': commit, 'shift': SHIFT, 'checkpoint_sha256': spec0['identity']['checkpoint_sha256'],
              'r23_A0_identity': a0_identity, 'R23_rows_changed_vs_A0': total, 'R23_changed_rows_match_run_records': True,
              'R23_prediction_sha_note': 'no earlier receipt names this SHA (RESULT-FLIP keeps only the last list written); bound by row identity with A0, per-image changed counts and exact reproduction of the published scores',
              'flipview_py_sha256': flipview_sha,
              'reproduction': {'class_equal_weight_form_max_abs_difference': 0.0, 'published_class_equal_weight': published,
                               'coco_stats_form_this_batch': orig['values'],
                               'coco_stats_form_AP75_small_published': stats_form,
                               'coco_stats_form_AP75_small_abs_difference': abs(orig['values']['AP75_small'] - stats_form),
                               'tolerance_note': 'class-equal-weight and COCO-stats forms average the same precision cells in a different order; only float summation order separates them'},
              'original': list_record('R23-seed0', r['R23'], orig, ctx, gt_meta, 'reproduction of the published R23 list (new score)'),
              'shifted': list_record('R23-seed0-shift', derived_ref, der, ctx, gt_meta, 'derived list, one certified score'),
              'shift_exactness': {'boxes_with_inexact_x_shift_or_width': inexact},
              'box_bounds': {'original': stats_orig, 'shifted': stats_shift},
              'comparison': compare(orig, der, ctx, gt_meta),
              'COCOeval_full_calls': {'reproduction_score': 1, 'derived_score': 1},
              'resource_at_start': resources, 'cost': RC.cost(started, cpu), 'cuda_initialized': ctx.torch.cuda.is_initialized()}
    save_arrays(Path(out) / 'shift-R23-seed0.npz', orig, der)
    publish(jpath, record)
    print(json.dumps({'tag': 'R23-seed0', 'a0_identical': a0_identity['identical'], 'cost': record['cost']}), flush=True)


# ---------------------------------------------------------------------------------------------
# coordinate cases: synthetic, model-free, deterministic
# ---------------------------------------------------------------------------------------------

def phase_coords(ctx, cfg, out, commit):
    jpath = Path(out) / 'COORDS.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing coords output has another commit')
        return
    started, cpu = time.perf_counter(), time.process_time()
    import torch
    from pycocotools import mask as mask_utils
    from sparse_rtdetr.localization.data import parse_raw
    from sparse_rtdetr.data_protocol.parser import parse_annotation_bytes
    flipview, flipview_sha = load_r23_module(cfg)
    res = {'commit': commit, 'frozen_source': str(ctx.source), 'flipview_py_sha256': flipview_sha, 'cases': {}}
    cases = res['cases']
    # GT parsing: (left, top, width, height) -> half-open xyxy, area w*h, bridge xywh
    parsed = parse_annotation_bytes(b'10,20,5,7,1,4,0,0\n')
    raw = parse_raw(b'10,20,5,7,1,4,0,0\n60,0,40,50,0,0,0,0\n', 'img', expected_rows=2)
    rows = [{'stable_image_id': 'img', 'width': 100, 'height': 50}]
    preds = [{'image_id': 'img', 'query_id': 0, 'category_id': 4, 'score': .9, 'bbox_xyxy': [10.25, 20.125, 15.5, 27.0625]},
             {'image_id': 'img', 'query_id': 1, 'category_id': 4, 'score': .8, 'bbox_xyxy': [-3.5, 41.0, 4.0, 52.5]},
             {'image_id': 'img', 'query_id': 2, 'category_id': 4, 'score': .7, 'bbox_xyxy': [70.0, 10.0, 80.0, 20.0]}]
    dataset, det, counts = ctx.ev.bridge_inputs(rows, [raw], preds)
    cases['gt_and_bridge'] = {
        'parser_bbox_xyxy': list(parsed[0].fields.bbox_xyxy), 'parser_area': parsed[0].fields.area,
        'parse_raw_box': raw['rows'][0]['box'], 'bridge_gt_bbox_xywh': dataset['annotations'][0]['bbox'],
        'bridge_gt_area': dataset['annotations'][0]['area'],
        'bridge_det_bbox_xywh': [d['bbox'] for d in det], 'det_kept': len(det), 'counts': counts,
        'expect': 'GT [10,20,15,27] half-open, xywh [10,20,5,7], area 35; detections pass with exact fractional and out-of-image coordinates (no clip, no rounding); the box inside the category-0 region is filtered'}
    ok = (list(parsed[0].fields.bbox_xyxy) == [10., 20., 15., 27.] and dataset['annotations'][0]['bbox'] == [10., 20., 5., 7.]
          and det[0]['bbox'] == [10.25, 20.125, 5.25, 6.9375] and det[1]['bbox'] == [-3.5, 41.0, 7.5, 11.5] and len(det) == 2)
    cases['gt_and_bridge']['pass'] = bool(ok)
    # decode: normalized cxcywh -> xyxy x [W, H, W, H]; no clip
    logits = torch.full((300, 10), -5., dtype=torch.float32)
    logits[0, 3], logits[1, 3] = 4., 3.
    boxes = torch.full((300, 4), .5, dtype=torch.float32)
    boxes[:, 2:] = .01
    boxes[0] = torch.tensor([.5, .25, .1, .2])
    boxes[1] = torch.tensor([.001, .5, .01, .02])
    row = {'stable_image_id': 'img', 'width': 1360, 'height': 765}
    try:
        ranked = ctx.ev.rank_candidates(logits, boxes, row, 600)
        q = {r['query_id']: r['bbox_xyxy'] for r in ranked if r['category_id'] == 4}
        b0 = torch.cat((boxes[0, :2] - boxes[0, 2:] / 2, boxes[0, :2] + boxes[0, 2:] / 2)) * torch.tensor([1360., 765.] * 2)
        b1 = torch.cat((boxes[1, :2] - boxes[1, 2:] / 2, boxes[1, :2] + boxes[1, 2:] / 2)) * torch.tensor([1360., 765.] * 2)
        cases['decode'] = {'query0': q[0], 'query1': q[1], 'pass': bool(q[0] == [float(v) for v in b0.tolist()] and q[1] == [float(v) for v in b1.tolist()] and q[1][0] < 0),
                           'expect': 'FP32 corners times original size, python floats, negative x0 kept'}
    except Exception as exc:  # recorded, not hidden
        cases['decode'] = {'error': repr(exc), 'pass': False}
    # r23 mirror: canvas tensor flip and cx -> 1 - cx
    gen = torch.Generator().manual_seed(23)
    rnd = torch.rand((4096, 4), generator=gen, dtype=torch.float32)
    twice = flipview.mirror_boxes(flipview.mirror_boxes(rnd))
    canvas = torch.zeros((1, 1344))
    canvas[0, 300:317] = 1.
    m = canvas.flip(-1)
    c0 = float(((torch.arange(1344) + .5) * canvas[0]).sum() / canvas[0].sum()) / 1344
    c1 = float(((torch.arange(1344) + .5) * m[0]).sum() / m[0].sum()) / 1344
    mapped = flipview.mirror_boxes(torch.tensor([[c1, .5, .01, .01]]))[0, 0].item()
    cases['r23_mirror'] = {'double_mirror_rows_not_bitwise': int((twice != rnd).any(1).sum()), 'double_mirror_max_abs': float((twice - rnd).abs().max()),
                           'canvas_flip_centroid_mapped_back_minus_original': mapped - c0,
                           'note': 'x.flip(-1) maps canvas column j to 1343-j; cx -> 1-cx is the matching half-open map; FP32 1-(1-cx) can differ from cx in the last bit'}
    cases['r23_mirror']['pass'] = abs(cases['r23_mirror']['canvas_flip_centroid_mapped_back_minus_original']) < 1e-6
    # pycocotools IoU on continuous shifted boxes
    iou = float(mask_utils.iou(np.array([[9.75, 20., 5., 7.]]), np.array([[10., 20., 5., 7.]]), [0])[0, 0])
    cases['coco_iou_continuous'] = {'iou_shift_0p25_on_5x7': iou, 'analytic': (4.75 * 7) / (35 + 35 - 4.75 * 7),
                                    'pass': abs(iou - (4.75 * 7) / (35 + 35 - 4.75 * 7)) < 1e-12}
    # annotation convention on the 548 development images (descriptive; the true convention is not known from code)
    conv = defaultdict(int)
    for r, p in zip(ctx.rows, ctx.parsed):
        W, H = r['width'], r['height']
        for obj in p['rows']:
            if obj['category'] not in range(1, 11) or obj['score'] != 1:
                continue
            x0, y0, x1, y1 = obj['box']
            conv['valid_GT'] += 1
            conv['left_eq_0'] += x0 == 0
            conv['top_eq_0'] += y0 == 0
            conv['right_eq_width'] += x1 == W
            conv['right_gt_width'] += x1 > W
            conv['bottom_eq_height'] += y1 == H
            conv['bottom_gt_height'] += y1 > H
            conv['non_integer_field'] += any(v != int(v) for v in (x0, y0, x1, y1))
    cases['development_GT_convention_counts'] = {k: int(v) for k, v in conv.items()}
    res['all_pass'] = all(c.get('pass', True) for c in cases.values())
    res['cost'] = RC.cost(started, cpu)
    res['cuda_initialized'] = torch.cuda.is_initialized()
    res['transform_cases'] = 'coord_check.py (separate process: torchvision transforms are nn.Modules, which this process forbids)'
    publish(jpath, res)
    print(json.dumps({'phase': 'coords', 'all_pass': res['all_pass'], 'GT_convention': cases['development_GT_convention_counts']}), flush=True)


# ---------------------------------------------------------------------------------------------

def mean_range(values):
    return {'values': values, 'mean': float(np.mean(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}


def phase_aggregate(cfg, out, commit):
    jpath = Path(out) / 'RESULT_FOLLOWUP_B.json'
    if jpath.exists():
        return
    lists = {}
    for tag in HISTORY_TAGS + ('R23-seed0',):
        p = Path(out) / f'shift-{tag}.json'
        require(p.exists(), 'missing ' + p.name)
        lists[tag] = json.loads(p.read_bytes())
        require(lists[tag]['commit'] == commit, 'mixed commits in ' + p.name)
    table = {tag: {'original': {k: rec['original']['values'][k] for k in REPORT_KEYS},
                   'shifted': {k: rec['shifted']['values'][k] for k in REPORT_KEYS},
                   'delta_pp': {k: rec['comparison']['delta_pp'][k] for k in REPORT_KEYS},
                   'tp_flips_small': {k: rec['comparison']['tp_flips'][k] for k in ('small_05', 'small_075')},
                   'filter': rec['comparison']['filter']} for tag, rec in lists.items()}
    arms = {}
    for arm in ('A', 'C'):
        tags = [f'{arm}-seed{s}' for s in (0, 1, 2)]
        arms[arm] = {k: mean_range([table[t]['delta_pp'][k] for t in tags]) for k in REPORT_KEYS}
    a0, r23 = lists['A-seed0'], lists['R23-seed0']
    four = {'A': a0['original']['values'], 'A_shift': a0['shifted']['values'], 'R23': r23['original']['values'], 'R23_shift': r23['shifted']['values']}
    interaction = {k: ((four['R23_shift'][k] - four['A_shift'][k]) - (four['R23'][k] - four['A'][k])) * 100. for k in REPORT_KEYS}
    mirror = {k: {'original_pp': (four['R23'][k] - four['A'][k]) * 100., 'shifted_pp': (four['R23_shift'][k] - four['A_shift'][k]) * 100.,
                  'difference_pp': interaction[k]} for k in REPORT_KEYS}
    result = {'task': 'TASK_CACHED_FOLLOWUP_002 task B', 'commit': commit, 'shift': SHIFT, 'table': table, 'arm_delta_pp_three_seeds': arms,
              'r23_four_cells': four, 'r23_mirror_increment': mirror,
              'r23_A_reference': 'A-seed0 certified list; r23 A0 identical: ' + str(r23['r23_A0_identity']['identical']),
              'scores': {'derived_lists': 7, 'R23_original_reproduction': 1, 'normal_new_scores': 8,
                         'original_match_only_evaluate_calls': 6, 'original_accumulate_or_summarize_calls': 0},
              'interpretation_limits': ['one fixed perturbation chosen after residuals on the same 548 images were seen; exploratory sensitivity only',
                                        'no independent data; seeds are not an independent validation', 'not a root-cause proof; no inference default changes',
                                        'the r23 difference is an interaction description under one fixed perturbation, not a share of the mirror gain']}
    publish(jpath, result)
    print(json.dumps({'phase': 'aggregate', 'A_APS': arms['A']['APS']['mean'], 'C_APS': arms['C']['APS']['mean']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--phases', nargs='+', choices=('coords', 'lists', 'r23', 'aggregate'), required=True)
    parser.add_argument('--tags', nargs='*', default=list(HISTORY_TAGS))
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    cfg = json.loads(Path(args.config).read_bytes())
    require(sys.executable == cfg['python'], 'interpreter differs from the config')
    require(cfg['shift']['dx_px'] == DX and cfg['shift']['dy_px'] == 0.0, 'config shift differs from the fixed task value')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    if args.phases == ['aggregate']:
        phase_aggregate(cfg, out, args.commit)
        return
    ctx = RC.Context(cfg)
    print(json.dumps({'context': RC.cost(started, cpu)}), flush=True)
    if 'coords' in args.phases:
        phase_coords(ctx, cfg, out, args.commit)
    if 'lists' in args.phases:
        for tag in args.tags:
            process_history(ctx, cfg, out, args.commit, tag)
    if 'r23' in args.phases:
        phase_r23(ctx, cfg, out, args.commit)
    require(not ctx.torch.cuda.is_initialized(), 'CUDA was initialized')
    if 'aggregate' in args.phases:
        phase_aggregate(cfg, out, args.commit)
    print(json.dumps({'done': args.phases, 'total': RC.cost(started, cpu)}), flush=True)


if __name__ == '__main__':
    main()
