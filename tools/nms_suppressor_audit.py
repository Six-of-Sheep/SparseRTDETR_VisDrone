"""Seed0-only CPU attribution of NMS-stage support loss on the frozen original-C pool.

Reuses the executed c_stage_audit admission, raw decoding, exact C reconstruction, stage
buckets and original raster/ignore primitives verbatim; adds the mirror NMS bookkeeping
from nms_attribution. Immutable outputs. Any mismatch, identity difference or failed
reconstruction STOPs without retry. No GPU, model construction, detector forward,
training or AP/AR scoring. Seeds 1 and 2 are not reachable from this entry.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c_stage_audit import PARENT_SHA, sha, publish, geometry
from c_stage_core import STAGES, BUCKETS, availability, class_cap, ious
from nms_attribution import (classwise_nms_with_suppressor, attribute_gt, shared_counts,
                             MODULE_VERSION, CATEGORIES, FATES)

TASK = 'r18-nms-suppressor-attribution-20261030-001'
SEED = 0
TAUS = ('0.75', '0.5')
CAPS = {'wall_seconds': 900, 'rss_bytes': 6 * 1024 ** 3, 'output_bytes': 1024 ** 3,
        'threads': 1, 'processes': 1}
OWN = ('nms_suppressor_audit.py', 'nms_attribution.py', 'c_stage_audit.py', 'c_stage_core.py')
# Frozen 2026-10-18 seed0 stage totals, sahi-c-stage-audit-20261018-001/attempt-002/seed0-SUMMARY.json
EXPECTED_TOTALS = {
    '600': {'0.5': {'available': 22885, 'budget_loss': 446, 'class100_loss': 384,
                    'hard_nms_loss': 908, 'ignore_loss': 2, 'no_raw_geometry': 1523,
                    'view_class_selection': 293},
            '0.75': {'available': 11619, 'budget_loss': 121, 'class100_loss': 94,
                     'hard_nms_loss': 4151, 'ignore_loss': 0, 'no_raw_geometry': 10082,
                     'view_class_selection': 374}},
    '300': {'0.5': {'available': 21783, 'budget_loss': 1804, 'class100_loss': 126,
                    'hard_nms_loss': 908, 'ignore_loss': 4, 'no_raw_geometry': 1523,
                    'view_class_selection': 293},
            '0.75': {'available': 11314, 'budget_loss': 487, 'class100_loss': 33,
                     'hard_nms_loss': 4151, 'ignore_loss': 0, 'no_raw_geometry': 10082,
                     'view_class_selection': 374}},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rss_high_water():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise ValueError('VmHWM unavailable')


def caps(start, out):
    elapsed = time.monotonic() - start
    hwm = rss_high_water()
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(elapsed <= CAPS['wall_seconds'], f'wall cap exceeded: {elapsed:.1f}s')
    require(hwm <= CAPS['rss_bytes'], f'RSS cap exceeded: {hwm}')
    require(size <= CAPS['output_bytes'], f'output cap exceeded: {size}')
    return {'elapsed_seconds': elapsed, 'rss_high_water_bytes': hwm, 'output_bytes': size}


def xyxy(gt):
    x, y, w, h = gt['bbox']
    return [x, y, x + w, y + h]


def aggregator():
    return {'boundary_within_eps': 0, 'comparisons': 0,
            'by_tau': {t: {'gt': 0, 'pairs': 0, 'mixed': 0, 'single_suppressor': 0,
                           'pair_category': Counter(), 'pair_same_view': Counter(),
                           'by_class': defaultdict(Counter), 'by_sequence': defaultdict(Counter),
                           'gt_combo': Counter(), 'fate600': Counter(), 'fate300': Counter(),
                           'shared_pairs': 0, 'lost_by_view': Counter(),
                           'suppressor_gt_iou_ge_tau': 0} for t in TAUS},
            'suppressors': {t: {} for t in TAUS}}


def accumulate(agg, tau, image, gt_id, cat, sequence, rec):
    a = agg['by_tau'][tau]
    a['gt'] += 1
    a['pairs'] += rec['lost_count']
    a['mixed'] += int(rec['mixed'])
    a['single_suppressor'] += int(rec['single_suppressor'])
    a['gt_combo']['+'.join(rec['categories'])] += 1
    a['lost_by_view'].update(rec['lost_by_view'])
    for cr in rec['candidates']:
        a['pair_category'][cr['category']] += 1
        a['by_class'][str(cat)][cr['category']] += 1
        a['by_sequence'][sequence][cr['category']] += 1
        a['pair_same_view']['same_view' if cr['same_view'] else 'cross_view'] += 1
        a['shared_pairs'] += int(cr['shared_with_gts'] > 1)
        a['suppressor_gt_iou_ge_tau'] += int(cr['suppressor_gt_iou'] >= float(tau))
        a['fate600'][cr['suppressor_fate']['600']] += 1
        a['fate300'][cr['suppressor_fate']['300']] += 1
        slot = agg['suppressors'][tau].setdefault((image, tuple(cr['suppressor'])),
                                                 {'gts': set(), 'argmax': set(), 'tie': False})
        slot['gts'].add(gt_id)
        slot['argmax'].update(cr['suppressor_argmax_gt'])
        slot['tie'] = slot['tie'] or cr['suppressor_argmax_tie']


def finalize(agg):
    out = {'by_tau': {}, 'boundary_within_eps': agg['boundary_within_eps'],
           'comparisons': agg['comparisons']}
    for tau in TAUS:
        a = agg['by_tau'][tau]
        boxes = agg['suppressors'][tau]
        histogram = Counter(len(v['gts']) for v in boxes.values())
        out['by_tau'][tau] = {
            'gt': a['gt'], 'candidate_pairs': a['pairs'], 'gt_mixed_categories': a['mixed'],
            'gt_single_suppressor': a['single_suppressor'],
            'pair_category': dict(a['pair_category']), 'pair_same_view': dict(a['pair_same_view']),
            'by_class': {k: dict(v) for k, v in sorted(a['by_class'].items())},
            'by_sequence': {k: dict(v) for k, v in sorted(a['by_sequence'].items())},
            'gt_category_combo': dict(a['gt_combo']),
            'suppressor_fate': {'600': dict(a['fate600']), '300': dict(a['fate300'])},
            'shared_candidate_pairs': a['shared_pairs'],
            'suppressor_gt_iou_ge_tau_pairs': a['suppressor_gt_iou_ge_tau'],
            'lost_by_view': dict(a['lost_by_view']),
            'distinct_suppressor_boxes': len(boxes),
            'gts_per_suppressor_histogram': {str(k): v for k, v in sorted(histogram.items())},
            'suppressor_argmax_is_damaged_gt': sum(1 for v in boxes.values() if v['argmax'] & v['gts']),
            'suppressor_argmax_ambiguous': sum(1 for v in boxes.values() if v['tie']),
        }
    return out


def run(parent, out, prepare_only=False):
    start = time.monotonic()
    require(sha(parent) == PARENT_SHA, 'parent contract SHA differs')
    c = json.loads(parent.read_bytes())
    source = Path(c['source_root'])
    for item in c['source_files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or sha(source / rel) != item['sha256']:
            raise ValueError('external frozen source mismatch')
    sys.path.insert(0, str(source))
    import softnms_job as j
    torch = j.cpu_only()
    c = j.load_contract(parent)
    specs, rows, data_spec = j.context(c)
    from slice_core import classwise_nms, NMS_IOU
    require(NMS_IOU == 0.5, 'frozen NMS IoU differs')
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2

    def forbidden(*args, **kwargs):
        raise RuntimeError('attribution forbids scoring')
    ev.score_budget = ev.bridge_metrics = primary.evaluate_primary_v1 = forbidden
    parsed = j.parsed_gt(rows, data_spec)
    plan = j.read(j.PLAN)
    j.resources(out, plan)
    publish(out / 'ADMISSION.json', {
        'parent_contract_sha256': PARENT_SHA, 'external_source': str(source),
        'source_files_verified': len(c['source_files']), 'images': len(rows),
        'seeds_available': [s['seed'] for s in specs], 'seed_processed': SEED,
        'own_source': {n: sha(HERE / n) for n in OWN}, 'module_version': MODULE_VERSION,
        'nms_iou': NMS_IOU, 'taus': list(TAUS), 'caps': CAPS,
        'GPU': 0, 'model_constructions': 0, 'AP_calls': 0,
        'expected_totals_20261018': EXPECTED_TOTALS})
    if prepare_only:
        return {'status': 'ADMISSION_PASS', 'images': len(rows), 'parsed_annotations': len(parsed),
                'cache_batches_processed': 0, 'model_calls': 0, 'AP_calls': 0, 'seed': SEED}
    spec = next((s for s in specs if s['seed'] == SEED), None)
    require(spec is not None, 'seed0 spec absent')
    proofs, expected = [], {}
    for budget in (600, 300):
        preds = json.loads(j.exact(spec['evaluation']['predictions']['C'][str(budget)], contents=True))
        proofs.append({'seed': SEED, **j.evaluation_input_proof(spec, rows, parsed, preds, 'C', budget)})
        expected[budget] = j.group(preds, spec['ids'], budget)
    totals = {b: {t: Counter({k: 0 for k in BUCKETS}) for t in TAUS} for b in (600, 300)}
    agg = aggregator()
    denominator = 0
    image_counts, count_rows = [], []
    batches = spec['full']['raw_view_refs']
    with (out / 'seed0-attribution.jsonl').open('x', encoding='utf-8') as detail:
        for batch, ref in enumerate(batches):
            j.resources(out, plan)
            state = caps(start, out)
            raw = torch.load(io.BytesIO(j.exact(ref, contents=True)), map_location='cpu', weights_only=True)
            part = rows[4 * batch: 4 * batch + 4]
            pools = j.decode_raw(raw, part, spec['identity'], SEED)
            maps = geometry(raw, part, j, torch)
            for local, (row, pool, geomap) in enumerate(zip(part, pools, maps)):
                image = row['stable_image_id']
                ann = parsed[4 * batch + local]
                require(len(geomap) == 1500, 'raw geometry incomplete')
                require(all(tuple(p.box_xyxy) == geomap[(p.view_id, p.query_id)] for p in pool),
                        'raw mapping mismatch')
                kept = classwise_nms(pool)
                for budget in (600, 300):
                    require([j.prediction(image, p) for p in kept[:budget]] == expected[budget][image],
                            'original C exact reconstruction differs')
                mirror, suppressed, stats = classwise_nms_with_suppressor(pool, NMS_IOU)
                require([p.identity for p in mirror] == [p.identity for p in kept],
                        'mirror NMS differs from frozen NMS')
                by_id = {p.identity: p for p in pool}
                for entry in suppressed.values():
                    pivot = by_id[entry['suppressor']]
                    entry['suppressor_box'] = pivot.box_xyxy
                    entry['suppressor_score'] = pivot.score
                agg['boundary_within_eps'] += stats['boundary_within_eps']
                agg['comparisons'] += stats['comparisons']
                im = PrimaryImageV2(image, row['width'], row['height'])
                integral = primary._build_integral(im, [g for g in ev.truth(row, ann) if g.category_id == 0])
                dataset, _, _ = ev.bridge_inputs([row], [ann], [])
                all_gts = [g for g in dataset['annotations'] if g['iscrowd'] == 0]
                small = [g for g in all_gts if 0 < g['area'] <= 1024]
                denominator += len(small)
                sequence = Path(row['annotation_relative_path']).stem.split('_')[0]
                image_counts.append({'image_id': image, 'sequence': sequence, 'small_GT': len(small)})
                membership, stages = {}, {}
                for budget in (600, 300):
                    top = kept[:budget]
                    filtered = [p for p in top if primary._ignored_fraction(integral, im, p.box_xyxy) < .5]
                    cap = class_cap(filtered)
                    _, canonical_filtered, counts = ev.bridge_inputs([row], [ann], expected[budget][image])
                    converted = [{'image_id': 1, 'category_id': p.category_id, 'score': p.score,
                                  'bbox': [p.box_xyxy[0], p.box_xyxy[1],
                                           p.box_xyxy[2] - p.box_xyxy[0], p.box_xyxy[3] - p.box_xyxy[1]]}
                                 for p in filtered]
                    require(converted == canonical_filtered, 'raw-ignore differs from original bridge')
                    count_rows.append({'image_id': image, 'budget': budget, 'pool': len(pool),
                                       'nms': len(kept), 'top': len(top), 'filtered': len(filtered),
                                       'class100': len(cap), **counts})
                    label = str(budget)
                    membership['top' + label] = {p.identity for p in top}
                    membership['filtered' + label] = {p.identity for p in filtered}
                    membership['cap' + label] = {p.identity for p in cap}
                    stages[budget] = (pool, kept, top, filtered, cap)
                stage_by_class = {budget: {cat: {STAGES[0]: list(geomap.values()),
                                                 **{stage: [p.box_xyxy for p in cand if p.category_id == cat]
                                                    for stage, cand in zip(STAGES[1:], stages[budget])}}
                                           for cat in range(1, 11)} for budget in (600, 300)}
                lost_by_gt = {t: {} for t in TAUS}
                pending = []
                for gt in small:
                    cat = gt['category_id']
                    box = xyxy(gt)
                    results = {}
                    for budget in (600, 300):
                        result = availability(box, stage_by_class[budget][cat])
                        results[budget] = result
                        for tau, bucket in result['first_unavailable'].items():
                            totals[budget][tau][bucket] += 1
                    require(all(results[600]['max_iou'][s] == results[300]['max_iou'][s] for s in STAGES[:3]),
                            'pre-budget stage maxima differ between budgets')
                    for tau in TAUS:
                        if results[600]['first_unavailable'][tau] != 'hard_nms_loss':
                            continue
                        threshold = float(tau)
                        lost = [p for p in pool if p.category_id == cat
                                and float(ious(box, [p.box_xyxy])[0]) >= threshold]
                        require(lost, 'NMS bucket without pool support')
                        same = [{'id': g['raw_annotation_id'], 'box': xyxy(g)}
                                for g in all_gts if g['category_id'] == cat]
                        rec = attribute_gt({'id': gt['raw_annotation_id'], 'box': box}, lost, suppressed,
                                           same, threshold, membership, ious)
                        lost_by_gt[tau][gt['raw_annotation_id']] = [p.identity for p in lost]
                        pending.append((tau, gt, cat, results[600], rec))
                shared = {t: shared_counts(lost_by_gt[t]) for t in TAUS}
                for tau, gt, cat, result, rec in pending:
                    for cr in rec['candidates']:
                        cr['shared_with_gts'] = shared[tau][tuple(cr['candidate'])]
                    accumulate(agg, tau, image, gt['raw_annotation_id'], cat, sequence, rec)
                    detail.write(json.dumps({'seed': SEED, 'image_id': image, 'sequence': sequence,
                                             'raw_annotation_id': gt['raw_annotation_id'],
                                             'category_id': cat, 'area': gt['area'], 'tau': tau,
                                             'stage_max_iou': result['max_iou'],
                                             'first_unavailable': result['first_unavailable'],
                                             **rec}, allow_nan=False) + '\n')
            print(json.dumps({'seed': SEED, 'batch': batch + 1, 'batches': len(batches), **state}), flush=True)
    require(len(image_counts) == 548, 'image count differs from 548')
    for budget in (600, 300):
        for tau in TAUS:
            require(sum(totals[budget][tau].values()) == denominator, 'partition incomplete')
            require(dict(totals[budget][tau]) == EXPECTED_TOTALS[str(budget)][tau],
                    f'seed0 totals differ from the 2026-10-18 diagnostic at {budget}/{tau}')
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    final = finalize(agg)
    final['caps'] = caps(start, out)
    final['totals'] = {str(b): {t: dict(totals[b][t]) for t in TAUS} for b in (600, 300)}
    final['expected_totals_20261018'] = EXPECTED_TOTALS
    final['seed0_totals_reproduced_20261018'] = True
    final['mirror_equals_frozen_all_images'] = True
    final['small_GT'] = denominator
    final['units'] = 'descriptive counts; not AP/recall/recoverable TP/method claim'
    summary = publish(out / 'seed0-ATTRIBUTION_SUMMARY.json', final)
    publish(out / 'seed0-COUNTS.json', count_rows)
    publish(out / 'INPUT_PROOFS.json', proofs)
    return {'status': 'COMPLETE', 'task': TASK, 'seed': SEED, 'seeds_processed': [SEED],
            'seeds_1_2_unreachable_from_entry': True, 'automatic_next': False,
            'original_C_exact': True, 'mirror_equals_frozen_all_images': True,
            'seed0_totals_reproduced_20261018': True,
            'AP_calls': 0, 'GPU_calls': 0, 'model_calls': 0,
            'attribution_summary': summary, 'caps': CAPS}


def fixture(out):
    sys.path.insert(0, str(HERE.parent))
    from slice_core import Candidate, classwise_nms, NMS_IOU
    a = Candidate((0., 0., 10., 10.), .9, 0, 0, 1, 1)
    b = Candidate((1., 0., 11., 10.), .8, 0, 1, 1, 2)
    kept, suppressed, stats = classwise_nms_with_suppressor([a, b], NMS_IOU)
    if not suppressed or [c.identity for c in kept] != [c.identity for c in classwise_nms([a, b])]:
        raise ValueError('synthetic mirror closure failed')
    return {'status': 'COMPLETE', 'synthetic': True, 'module_version': MODULE_VERSION,
            'kept': [list(c.identity) for c in kept],
            'suppressed': {str(k): list(v['suppressor']) for k, v in suppressed.items()},
            'stats': stats, 'categories': list(CATEGORIES), 'fates': list(FATES)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'admit', 'fixture'))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inject-failure', action='store_true')
    args = parser.parse_args(argv)
    if args.mode != 'fixture' and (args.parent is None or args.inject_failure):
        parser.error('invalid production options')
    out = args.output.resolve()
    if args.mode != 'fixture' and TASK not in out.parts:
        parser.error('production output is outside task directory')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    publish(out / 'START.json', {'task': TASK, 'mode': args.mode, 'pid': os.getpid(),
                                 'started_unix': time.time(), 'seed': SEED})
    try:
        if args.mode == 'fixture':
            if args.inject_failure:
                raise RuntimeError('expected fixture failure')
            result = fixture(out)
        else:
            result = run(args.parent, out, prepare_only=args.mode == 'admit')
        result['elapsed_seconds'] = time.monotonic() - start
        ref = publish(out / 'RESULT.json', result)
        publish(out / 'CLOSED.json', {'result': ref, 'returncode': 0})
        print(json.dumps({'status': 'COMPLETE', 'result': ref}), flush=True)
        return 0
    except Exception:
        publish(out / 'STOP.json', {'traceback': traceback.format_exc(),
                                    'elapsed_seconds': time.monotonic() - start,
                                    'automatic_retry': False})
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
