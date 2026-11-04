"""Assemble the compact result of the cached error channels from one run directory.

No model and no scoring here: counts, rates and differences over stored per-list arrays, plus
cross-checks against old receipts.  Oracle deltas stay separate; nothing sums them or turns them
into shares of a gain.

usage: python -B -I aggregate.py --config CONFIG --run-dir DIR --commit SHA
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import core as C

SEEDS = (0, 1, 2)
ARMS = ('A', 'C')
THRESHOLDS = (0.5, 0.75)
TKEY = {0.5: '05', 0.75: '075'}
CHANNEL_KEYS = ('Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss')
DELTA_KEYS = CHANNEL_KEYS + ('FP_rank', 'FN_recall')
METRICS = ('AP', 'APS', 'AP75_small', 'AP50', 'AP75', 'AP50_small')
BUCKETS = ('no_raw_geometry', 'view_class_selection', 'hard_nms_loss', 'budget_loss', 'ignore_loss', 'class100_loss', 'available')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rate(a, b):
    return round(a / b, 6) if b else None


def r6(x):
    return None if x is None else round(float(x), 6)


def spread(values):
    values = [float(v) for v in values]
    return {'mean': r6(np.mean(values)), 'min': r6(min(values)), 'max': r6(max(values))}


def load_lists(run_dir):
    lists = {}
    for s in SEEDS:
        for a in ARMS:
            tag = f'{a}-seed{s}'
            lists[tag] = (json.loads((run_dir / f'list-{tag}.json').read_bytes()), np.load(run_dir / f'list-{tag}.npz'))
    return lists


def profile(arr, sel):
    n = int(sel.sum())
    out = {'GT': n}
    for t in THRESHOLDS:
        tp = arr[f'used_{t}'][sel]
        usable = arr[f'usable_{t}'][sel]
        be = arr['best_evaluated'][sel]
        bf = arr['best_filtered'][sel]
        tps = int(tp.sum())
        out[str(t)] = {'TP': tps, 'FN': n - tps, 'TP_rate': rate(tps, n),
                       'FN_with_Loc_or_Cls_target': int(np.logical_and(~tp, usable).sum()),
                       'Miss': int(np.logical_and(~tp, ~usable).sum()),
                       'cover_evaluated': int((be >= t).sum()), 'cover_evaluated_rate': rate(int((be >= t).sum()), n),
                       'cover_filtered': int((bf >= t).sum()), 'cover_filtered_rate': rate(int((bf >= t).sum()), n)}
    be = arr['best_evaluated'][sel]
    out['best_same_class_IoU_evaluated'] = {
        'lt_0.1': int((be < .1).sum()), '0.1_to_0.5': int(np.logical_and(be >= .1, be < .5).sum()),
        '0.5_to_0.75': int(np.logical_and(be >= .5, be < .75).sum()), 'ge_0.75': int((be >= .75).sum())}
    return out


def small_arrays(z, small):
    out = {'best_evaluated': z['best_evaluated'][small], 'best_filtered': z['best_filtered'][small]}
    for t in THRESHOLDS:
        out[f'used_{t}'] = z[f'used_small_{TKEY[t]}'][small]
        out[f'usable_{t}'] = z[f'usable_small_{TKEY[t]}'][small]
    return out


def transition(other_best, t):
    bands = {'other_covered_without_TP': int((other_best >= t).sum())}
    if t > .5:
        bands['other_best_0.5_to_t'] = int(np.logical_and(other_best >= .5, other_best < t).sum())
    bands['other_best_0.1_to_min(0.5,t)'] = int(np.logical_and(other_best >= .1, other_best < min(.5, t)).sum())
    bands['other_best_lt_0.1'] = int((other_best < .1).sum())
    return bands


def pair_table(a, c, sel):
    out = {}
    for t in THRESHOLDS:
        ta, tc = a[f'used_{t}'][sel], c[f'used_{t}'][sel]
        ca, cc = a['best_evaluated'][sel] >= t, c['best_evaluated'][sel] >= t
        c_only, a_only = np.logical_and(tc, ~ta), np.logical_and(ta, ~tc)
        out[str(t)] = {
            'TP': {'C_only': int(c_only.sum()), 'A_only': int(a_only.sum()), 'both': int(np.logical_and(ta, tc).sum()),
                   'neither': int(np.logical_and(~ta, ~tc).sum())},
            'cover_evaluated': {'C_only': int(np.logical_and(cc, ~ca).sum()), 'A_only': int(np.logical_and(ca, ~cc).sum()),
                                'both': int(np.logical_and(ca, cc).sum()), 'neither': int(np.logical_and(~ca, ~cc).sum())},
            'A_state_for_C_only_TP': transition(a['best_evaluated'][sel][c_only], t),
            'C_state_for_A_only_TP': transition(c['best_evaluated'][sel][a_only], t)}
    both = np.logical_and(a['used_0.5'][sel], c['used_0.5'][sel])
    ia, ic = a['best_evaluated'][sel][both], c['best_evaluated'][sel][both]
    out['best_IoU_on_GT_TP_in_both_at_0.5'] = {'n': int(both.sum()), 'A_mean': r6(ia.mean()) if len(ia) else None,
                                               'C_mean': r6(ic.mean()) if len(ic) else None,
                                               'C_higher': int((ic > ia).sum()), 'A_higher': int((ia > ic).sum())}
    return out


def seed_churn(small_lists):
    """Same recipe, different seed: how many small GT flip TP status between independent models."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for x, y in ((0, 1), (0, 2), (1, 2)):
            a, b = small_lists[f'{arm}-seed{x}'], small_lists[f'{arm}-seed{y}']
            out[arm][f'seed{x}_vs_seed{y}'] = {}
            for t in THRESHOLDS:
                ta, tb = a[f'used_{t}'], b[f'used_{t}']
                out[arm][f'seed{x}_vs_seed{y}'][str(t)] = {
                    'x_only': int(np.logical_and(ta, ~tb).sum()), 'y_only': int(np.logical_and(tb, ~ta).sum()),
                    'both': int(np.logical_and(ta, tb).sum())}
    return out


def size_gain(lists, small_lists, size_of):
    out = {}
    names = sorted(set(size_of.tolist()))
    for name in names + ['ALL']:
        sel = size_of == name if name != 'ALL' else np.ones(len(size_of), dtype=bool)
        block = {'small_GT': int(sel.sum())}
        for key in METRICS:
            block[key + '_C_minus_A_pp'] = spread([(lists[f'C-seed{s}'][0]['groups'][name][key] - lists[f'A-seed{s}'][0]['groups'][name][key]) * 100 for s in SEEDS])
        for t in THRESHOLDS:
            block[f'small_TP_C_minus_A_at_{t}'] = spread([int(small_lists[f'C-seed{s}'][f'used_{t}'][sel].sum()) - int(small_lists[f'A-seed{s}'][f'used_{t}'][sel].sum()) for s in SEEDS])
        block['classes_in_APS'] = lists['A-seed0'][0]['groups'][name]['classes_small']
        out[name] = block
    return out


def section_bins(manifest, table, small, bins):
    images = manifest['images']
    cat = table['cat'][small]
    img = table['img'][small] - 1
    seqs = np.array([images[i]['sequence'] for i in img])
    sizes = np.array([f"{images[i]['width']}x{images[i]['height']}" for i in img])
    out = []
    for b, label in enumerate(C.BIN_LABELS):
        sel = bins == b
        counts = Counter(seqs[sel].tolist())
        p = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
        p = p / p.sum() if p.sum() else p
        out.append({'bin_min_side_raw_px': label, 'GT': int(sel.sum()), 'share': rate(int(sel.sum()), len(bins)),
                    'by_class': np.bincount(cat[sel] - 1, minlength=10).tolist(),
                    'by_image_size': {k: int(v) for k, v in sorted(Counter(sizes[sel].tolist()).items())},
                    'images': int(len(set(img[sel].tolist()))), 'sequences': len(counts),
                    'top5_sequence_share': r6(p[:5].sum()), 'effective_sequences': r6(1 / np.sum(p ** 2)) if len(p) else None})
    return out


def cross_check_stage(cfg, manifest, table, small, lists):
    stage = cfg['c_stage']
    images = manifest['images']
    raw_id = table['raw_id'][small]
    img = table['img'][small] - 1
    key_of = {(images[int(i)]['id'], str(r)): k for k, (i, r) in enumerate(zip(img, raw_id))}
    report = {}
    for s in SEEDS:
        z = lists[f'C-seed{s}'][1]
        be, bf = z['best_evaluated'][small], z['best_filtered'][small]
        worst_e, worst_f, flips, seen = 0.0, 0.0, 0, 0
        totals = {t: Counter() for t in ('0.5', '0.75')}
        with (Path(stage['root']) / f'attempt-002/seed{s}-gt.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['budget'] != 600:
                    continue
                k = key_of[(r['image_id'], r['raw_annotation_id'])]
                seen += 1
                worst_e = max(worst_e, abs(r['max_iou']['class100'] - float(be[k])))
                worst_f = max(worst_f, abs(r['max_iou']['raw_ignore'] - float(bf[k])))
                for t, bucket in r['first_unavailable'].items():
                    totals[t][bucket] += 1
                    flips += int((bucket == 'available') != (be[k] >= float(t)))
        report[f'seed{s}'] = {'GT_joined': seen, 'max_abs_IoU_difference_evaluated_vs_class100': worst_e,
                              'max_abs_IoU_difference_filtered_vs_raw_ignore': worst_f,
                              'available_flag_differences': flips,
                              'receipt_available': {t: totals[t]['available'] for t in totals},
                              'this_run_cover_evaluated': {t: int((be >= float(t)).sum()) for t in totals},
                              'receipt_available_plus_class100_loss': {t: totals[t]['available'] + totals[t]['class100_loss'] for t in totals},
                              'this_run_cover_filtered': {t: int((bf >= float(t)).sum()) for t in totals}}
    return report


def section_b(lists):
    runs_out = {}
    for tag, (j, _) in lists.items():
        runs_out[tag] = {}
        for key, r in j['runs'].items():
            runs_out[tag][key] = {k: r[k] for k in ('TP', 'FP', 'GT', 'FN', 'FN_targeted_by_Loc_or_Cls', 'evaluator_ignored_detections', 'counts', 'fix_details')}
            runs_out[tag][key]['AP_pct'] = r6(r['AP'] * 100)
            runs_out[tag][key]['delta_pp'] = {k: r6(v) for k, v in r['delta_pp'].items()}
            if 'oracle_classes_emptied' in r:
                runs_out[tag][key]['oracle_classes_emptied'] = r['oracle_classes_emptied']
    means = {}
    for key in next(iter(lists.values()))[0]['runs']:
        block = {}
        for arm in ARMS:
            rs = [lists[f'{arm}-seed{s}'][0]['runs'][key] for s in SEEDS]
            block[arm] = {'AP_pct': spread([r['AP'] * 100 for r in rs]), 'TP': spread([r['TP'] for r in rs]),
                          'FP': spread([r['FP'] for r in rs]), 'FN': spread([r['FN'] for r in rs]),
                          'FN_targeted_by_Loc_or_Cls': spread([r['FN_targeted_by_Loc_or_Cls'] for r in rs]),
                          'rank_oracle_AP_pct': spread([r['AP'] * 100 + r['delta_pp']['FP_rank'] for r in rs]),
                          'AP_over_rank_oracle_AP': spread([r['AP'] * 100 / (r['AP'] * 100 + r['delta_pp']['FP_rank']) for r in rs]),
                          'counts': {c: spread([r['counts'][c] for r in rs]) for c in CHANNEL_KEYS},
                          'delta_pp': {d: spread([r['delta_pp'][d] for r in rs]) for d in DELTA_KEYS},
                          'fix_details': {name: {k: spread([r['fix_details'][name][k] for r in rs]) for k in rs[0]['fix_details'][name]} for name in ('Cls', 'Loc')}}
        paired = {'AP_pp': [], 'TP': [], 'FP': [], 'rank_oracle_AP_pp': [], 'counts': {c: [] for c in CHANNEL_KEYS}, 'delta_pp': {d: [] for d in DELTA_KEYS}}
        for s in SEEDS:
            a, c = lists[f'A-seed{s}'][0]['runs'][key], lists[f'C-seed{s}'][0]['runs'][key]
            paired['AP_pp'].append((c['AP'] - a['AP']) * 100)
            paired['TP'].append(c['TP'] - a['TP'])
            paired['FP'].append(c['FP'] - a['FP'])
            paired['rank_oracle_AP_pp'].append((c['AP'] - a['AP']) * 100 + c['delta_pp']['FP_rank'] - a['delta_pp']['FP_rank'])
            for x in CHANNEL_KEYS:
                paired['counts'][x].append(c['counts'][x] - a['counts'][x])
            for d in DELTA_KEYS:
                paired['delta_pp'][d].append(c['delta_pp'][d] - a['delta_pp'][d])
        block['C_minus_A_same_seed'] = {'AP_pp': spread(paired['AP_pp']), 'TP': spread(paired['TP']), 'FP': spread(paired['FP']),
                                        'rank_oracle_AP_pp': spread(paired['rank_oracle_AP_pp']),
                                        'counts': {x: spread(v) for x, v in paired['counts'].items()},
                                        'delta_pp': {x: spread(v) for x, v in paired['delta_pp'].items()}}
        means[key] = block
    per_class = {}
    for key in ('small@0.75', 'small@0.5'):
        per_class[key] = {}
        for arm in ARMS:
            rs = [lists[f'{arm}-seed{s}'][0]['runs'][key] for s in SEEDS]
            per_class[key][arm] = {'class_AP_pct_mean': [r6(np.mean([r['class_AP'][k] * 100 for r in rs])) for k in range(10)],
                                   'class_GT': rs[0]['class_GT'],
                                   'class_delta_pp_mean': {d: [r6(np.mean([r['class_delta_pp'][d][k] for r in rs])) for k in range(10)] for d in ('Loc', 'Cls', 'Miss')}}
    return {'by_list': runs_out, 'three_seed_summary': means, 'per_class': per_class}


def section_native(run_dir, lists):
    out = {}
    worst_ap, worst_delta, count_diffs, ignored = 0.0, 0.0, 0, 0
    for tag, (j, _) in lists.items():
        path = run_dir / f'native-{tag}.json'
        n = json.loads(path.read_bytes())
        out[tag] = {}
        for t in ('0.5', '0.75'):
            r = n['runs'][t]
            mine = j['runs'][f'all@{t}']
            ap_diff = abs(r['AP'] - mine['AP'] * 100)
            delta_diff = {k: abs(r['delta_pp_signed'][k] - mine['delta_pp'][k]) for k in CHANNEL_KEYS}
            cdiff = {k: r['counts'][k] - mine['counts'][k] for k in CHANNEL_KEYS}
            worst_ap = max(worst_ap, ap_diff)
            worst_delta = max(worst_delta, max(delta_diff.values()))
            count_diffs += sum(abs(v) for v in cdiff.values())
            ignored += r['errors_on_ignored_predictions']
            out[tag][t] = {'AP_pct': r6(r['AP']), 'counts': r['counts'],
                           'delta_pp_signed': {k: r6(v) for k, v in r['delta_pp_signed'].items()},
                           'delta_pp_clipped_native_rule': {k: r6(v) for k, v in r['delta_pp_clipped_native_rule'].items()},
                           'delta_pp_all_AP_fixing_only_small_qualified_errors': {k: r6(v) for k, v in r['delta_pp_all_AP_fixing_only_small_qualified_errors'].items()},
                           'FP_rank_delta_pp': r6(r['FP_rank_delta_pp']), 'FN_recall_delta_pp': r6(r['FN_recall_delta_pp']),
                           'abs_AP_difference_vs_certified_port': ap_diff, 'count_difference_vs_certified_port': cdiff,
                           'noop_fix_errors_delta_pp': r['noop_fix_errors_delta_pp'],
                           'max_abs_delta_difference_vs_certified_port': max(delta_diff.values()),
                           'seconds': r6(r['seconds'])}
    return {'by_list': out, 'worst_AP_abs_difference_pp': worst_ap, 'worst_delta_abs_difference_pp': worst_delta,
            'total_abs_count_difference': count_diffs, 'errors_on_ignored_predictions': ignored,
            'residual_explanation': 'FP channel counts agree; Miss differs by at most a few GT because COCO keeps the later GT and tidecv the earlier one when a detection has exactly equal IoU with two distinct same-class GT (checked directly on A-seed0@0.75 and C-seed0@0.5: same-size neighbouring boxes offset by a pixel or two). The delta residual follows from those GT and from tie order among equal scores.',
            'protocol_differences': [
                'tidecv caps max_dets per image over all classes; the certified bridge caps 100 per image per class. Input here is the certified capped list and the tidecv cap is opened, so nothing else is truncated.',
                'tidecv ignores a prediction when crowd IoU is strictly above the threshold; COCO uses at-or-above. The 548 development GT after the raster filter contain no class-score-0 crowd rows, so this does not act here.',
                'tidecv scores a class with detections but no GT as 0; COCO omits it. All 10 classes have GT in the full-set all-area evaluation.',
                'tidecv clips negative dAP to 0 in fix_main_errors; this report keeps signed values and stores the clipped value beside them in the JSON.',
                'tidecv has no working AP-small mode (add_qualifiers raises NotImplementedError). Its Qualifier mechanism can restrict which errors get fixed while still reporting all-area AP; that value is stored only as a separate field and is not APS.',
                'Native counts only exist for all-area runs; small runs exist only through the certified port with COCO ignore semantics.'
            ]}


def section_raw(run_dir, small, bins, lists):
    meta = json.loads((run_dir / 'raw-seed0.json').read_bytes())
    z = np.load(run_dir / 'raw-seed0.npz')
    index = z['gt_index']
    small_positions = np.nonzero(small)[0]
    if not np.array_equal(np.sort(index), small_positions):
        raise ValueError('raw table GT differ from the small GT table')
    reorder = np.argsort(index)
    a_raw, c_raw, tile = z['a_raw'][reorder], z['c_raw'][reorder], z['tile_raw'][reorder]
    a_bucket = {'0.5': z['a_bucket_05'][reorder], '0.75': z['a_bucket_075'][reorder]}
    c_bucket = {'0.5': z['c_bucket_05'][reorder], '0.75': z['c_bucket_075'][reorder]}
    tp_a = {t: lists['A-seed0'][1][f'used_small_{TKEY[t]}'][small] for t in THRESHOLDS}
    tp_c = {t: lists['C-seed0'][1][f'used_small_{TKEY[t]}'][small] for t in THRESHOLDS}

    def block(sel):
        n = int(sel.sum())
        out = {'GT': n}
        for t in THRESHOLDS:
            ar, cr = a_raw[sel] >= t, c_raw[sel] >= t
            key = str(t)
            out[key] = {'A_raw300_exists': int(ar.sum()), 'A_raw300_rate': rate(int(ar.sum()), n),
                        'C_raw1500_exists': int(cr.sum()), 'C_raw1500_rate': rate(int(cr.sum()), n),
                        'C_only_from_tile_views': int(np.logical_and(cr, ~ar).sum()),
                        'A_raw_exists_and_A_TP': int(np.logical_and(ar, tp_a[t][sel]).sum()),
                        'C_raw_exists_and_C_TP': int(np.logical_and(cr, tp_c[t][sel]).sum()),
                        'A_buckets': {BUCKETS[i]: int((a_bucket[key][sel] == i).sum()) for i in range(7)},
                        'C_buckets': {BUCKETS[i]: int((c_bucket[key][sel] == i).sum()) for i in range(7)}}
        return out

    table = {'all_small': block(np.ones(len(bins), dtype=bool))}
    for b, label in enumerate(C.BIN_LABELS):
        table[label] = block(bins == b)
    return {'checks': {k: meta[k] for k in ('images', 'small_GT', 'A_raw_rank_reproduces_certified_images',
                                            'C_view0_equals_A_raw_images', 'C_raw_recomputed_differs_from_receipt_count', 'A_raw_nested_in_C_raw')},
            'A_stage_mapping': meta['A_stage_mapping'], 'table': table,
            'units': 'class-agnostic raw existence and first-loss buckets; not AP, recall or a causal split'}


def section_c(run_dir, lists):
    r = json.loads((run_dir / 'r20.json').read_bytes())
    a0 = lists['A-seed0'][0]['groups']
    groups = {}
    for name, g in r['groups'].items():
        m = g['metrics']
        row = {'images': g['images'], 'small_GT': g['small_GT'], 'all_GT_non_crowd': g['all_GT_non_crowd'],
               'classes_in_AP': m['A1']['classes_all'], 'classes_in_APS': m['A1']['classes_small'],
               'class_GT_small': m['A1']['class_GT_small'],
               'scores_pct': {arm: {k: r6(m[arm][k] * 100) for k in METRICS} for arm in ('A1', 'A2', 'A1c')},
               'certified_A0_scores_pct': {k: r6(a0[name][k] * 100) for k in METRICS},
               'differences_pp': {
                   'A2_minus_A1': {k: r6((m['A2'][k] - m['A1'][k]) * 100) for k in METRICS},
                   'A1_minus_A0': {k: r6((m['A1'][k] - a0[name][k]) * 100) for k in METRICS},
                   'A2_minus_A0': {k: r6((m['A2'][k] - a0[name][k]) * 100) for k in METRICS},
                   'A1c_minus_A1': {k: r6((m['A1c'][k] - m['A1'][k]) * 100) for k in METRICS},
                   'A2_minus_A1c': {k: r6((m['A2'][k] - m['A1c'][k]) * 100) for k in METRICS}},
               'small_GT_matching': g['pairs'], 'paired_best_IoU': {k: r6(v) for k, v in g['paired_best_IoU_small_TP_at_.5_in_both'].items()},
               'control_registration_raw_px': g['control_registration_raw_px'], 'changed_boxes': g['changed_boxes']}
        groups[name] = row
    regression = {arm: {name: {'x': {k: r6(v) for k, v in g['x'].items()}, 'y': {k: r6(v) for k, v in g['y'].items()},
                               'mean_abs_predicted_error_x': r6(g['mean_abs_predicted_error_x']),
                               'mean_abs_predicted_error_y': r6(g['mean_abs_predicted_error_y'])}
                        for name, g in by.items()} for arm, by in r['registration_regression'].items()}
    render = json.loads((run_dir / 'render-check.json').read_bytes())
    return {'groups': groups, 'registration_regression_error_on_predicted_shift': regression,
            'A1_A2_box_identity_by_group': r['A1_A2_box_identity_by_group'], 'A1c_note': r['A1c'],
            'pixel_comparison': render['groups'], 'pixel_comparison_high_band': render['high_band_cycles_per_canvas_pixel'],
            'frozen_scores_reproduced': r['frozen_scores_reproduced']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads(Path(args.config).read_bytes())
    manifest = json.loads((run_dir / 'INPUT_MANIFEST.json').read_bytes())
    table = np.load(run_dir / 'gt-table.npz')
    lists = load_lists(run_dir)
    z0 = lists['A-seed0'][1]
    small = z0['small']
    for tag, (_, z) in lists.items():
        if not np.array_equal(z['small'], small) or not np.array_equal(z['gt_bbox'], table['bbox']):
            raise ValueError('GT tables differ across lists: ' + tag)
    w, h = table['bbox'][small, 2], table['bbox'][small, 3]
    bins = C.min_side_bin(w, h)
    small_lists = {tag: small_arrays(z, small) for tag, (_, z) in lists.items()}
    per_list = {}
    cats = table['cat'][small]
    images = manifest['images']
    size_of = np.array([f"{images[int(i) - 1]['width']}x{images[int(i) - 1]['height']}" for i in table['img'][small]])
    for tag, arr in small_lists.items():
        per_list[tag] = {'all_small': profile(arr, np.ones(len(bins), dtype=bool))}
        for b, label in enumerate(C.BIN_LABELS):
            per_list[tag][label] = profile(arr, bins == b)
        per_list[tag]['by_class'] = {str(k + 1): profile(arr, cats == k + 1) for k in range(10)}
        per_list[tag]['by_image_size'] = {name: profile(arr, size_of == name) for name in sorted(set(size_of.tolist()))}
    pairs = {}
    for s in SEEDS:
        a, c = small_lists[f'A-seed{s}'], small_lists[f'C-seed{s}']
        pairs[f'seed{s}'] = {'all_small': pair_table(a, c, np.ones(len(bins), dtype=bool))}
        for b, label in enumerate(C.BIN_LABELS):
            pairs[f'seed{s}'][label] = pair_table(a, c, bins == b)
    group_scores = {tag: {name: {k: r6(g[k] * 100) for k in METRICS} for name, g in j['groups'].items()} for tag, (j, _) in lists.items()}
    probe = json.loads(Path(cfg['r20']['probe_002']['path']).read_bytes())['600']['gt_side']
    lt = np.logical_and(np.logical_not(table['crowd']), table['area'] < 1024)
    a0_filtered = z0['best_filtered']
    probe_check = {'receipt_small_gt': probe['small_gt'], 'this_run_small_gt_area_lt_1024': int(lt.sum()),
                   'receipt_reach_0.5': probe['reach_0.5'], 'this_run_cover_filtered_0.5_area_lt_1024': int((a0_filtered[lt] >= .5).sum()),
                   'receipt_reach_0.75': probe['reach_0.75'], 'this_run_cover_filtered_0.75_area_lt_1024': int((a0_filtered[lt] >= .75).sum())}
    denominators = manifest['gt_denominators']
    if denominators['class_score_0_crowd_after_raster'] != 0:
        raise ValueError('native protocol note assumes no crowd GT after the raster filter')
    equal_side = bool(denominators['small_after_raster_area_le_1024'] - denominators['small_after_raster_area_lt_1024'] == denominators['after_raster_area_eq_1024'])
    stage_check = cross_check_stage(cfg, manifest, table, small, lists)
    result = {
        'task_id': cfg['task_id'], 'commit': args.commit, 'time_unix': time.time(),
        'status_note': 'Exploratory decomposition of closed rounds made after their results were known. Oracle deltas are independent, interact and do not sum to a gain.',
        'definitions': {
            'small_GT': 'bridge GT after the raw-ignore raster filter, iscrowd 0, raw area in [0,1024]: the in-range set of the certified small evaluation',
            'TP': 'small GT matched in the certified COCOeval small-area run at that IoU (score order, one to one, maxDets 100 per image per class)',
            'cover_evaluated': 'at least one same-class box at or above that IoU in the list COCOeval evaluates (after raw-ignore, top 100 per image per class); one box may cover several GT, so this is an existence rate, not recall',
            'cover_filtered': 'same, after raw-ignore but before the top 100 cap',
            'raw_exists': 'seed0 only: any of the raw query boxes at or above that IoU, class and score ignored; A has 300 boxes, C has 1500 over five views',
            'bins': 'min(width,height) of the GT box in original image pixels; canvas scale differs by image size and is not mixed in',
            'rank_oracle_AP': 'AP + FP_rank delta: every TP ranked above every FP with the same TP set; AP_over_rank_oracle_AP is descriptive only'},
        'input_manifest': {'sha256': sha(run_dir / 'INPUT_MANIFEST.json'), 'bytes': (run_dir / 'INPUT_MANIFEST.json').stat().st_size},
        'A': {'denominators': denominators, 'denominator_gap_is_area_eq_1024': equal_side,
              'r20_probe_cross_check': probe_check, 'c_stage_cross_check': stage_check,
              'bins': section_bins(manifest, table, small, bins), 'per_list': per_list, 'A_C_pairs': pairs,
              'seed_churn_same_recipe': seed_churn(small_lists), 'C_minus_A_by_image_size': size_gain(lists, small_lists, size_of),
              'scores_by_image_size_pct': group_scores, 'seed0_raw': section_raw(run_dir, small, bins, lists)},
        'B': {**section_b(lists), 'native_tidecv': section_native(run_dir, lists)},
        'C': section_c(run_dir, lists),
        'reproduction': {tag: {'max_abs_difference_vs_certified': j['max_abs_difference_vs_certified'],
                               'accumulate_port_max_abs_difference': j['accumulate_port_max_abs_difference'],
                               'AP': j['certified_metrics_reproduced']['AP'], 'APS': j['certified_metrics_reproduced']['APS'],
                               'AP50': j['certified_metrics_reproduced']['AP50'], 'AP75': j['certified_metrics_reproduced']['AP75'],
                               'AP75_small': j['certified_metrics_reproduced']['AP75_small'],
                               'spatial_filter_counts': j['spatial_filter_counts']} for tag, (j, _) in lists.items()},
        'cost': {'lists': {tag: j['cost'] for tag, (j, _) in lists.items()},
                 'COCOeval_full_calls_on_real_lists': sum(j['COCOeval_full_calls'] for j, _ in lists.values()) + json.loads((run_dir / 'r20.json').read_bytes())['COCOeval_full_calls'],
                 'oracle_evaluations_certified_port': sum(j['oracle_evaluations'] for j, _ in lists.values()),
                 'raw': json.loads((run_dir / 'raw-seed0.json').read_bytes())['cost'],
                 'r20': json.loads((run_dir / 'r20.json').read_bytes())['cost'],
                 'native_tidecv_seconds': {tag: json.loads((run_dir / f'native-{tag}.json').read_bytes())['seconds'] for tag in lists},
                 'render_check': {k: json.loads((run_dir / 'render-check.json').read_bytes())[k] for k in ('wall_seconds', 'cpu_seconds')},
                 'gpu_calls': 0, 'model_constructions': 0}}
    with (run_dir / 'RESULT_CACHED_ERROR_CHANNELS_001.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)
    print(json.dumps({'bytes': (run_dir / 'RESULT_CACHED_ERROR_CHANNELS_001.json').stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
