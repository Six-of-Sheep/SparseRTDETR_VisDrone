"""Read frozen raw endpoint caches on CPU. No model/optimizer/evaluator calls."""
from collections import Counter
import csv
import gzip
import importlib
import json
import os
from pathlib import Path
import sys

from r18_cache_diag import exact, read, publish, reference, require


def native(c):
    sys.path.insert(0, str(Path(c['native_root']) / 'src'))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA initialized before diagnostic')
    if c['profile'] == 'REAL_NUD_548':
        import torchvision
        require(torch.__version__ == '2.4.1' and torchvision.__version__ == '0.19.1', 'native package version differs')
    from sparse_rtdetr.localization.data import parse_raw, RawAugmentation
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2, PrimaryGroundTruth
    from sparse_rtdetr.baseline.config import _vendor_path
    from sparse_rtdetr.localization.consumer_geometry import matching
    with _vendor_path(Path(c['native_root']) / 'vendor/rtdetrv2_pytorch'):
        Matcher = importlib.import_module('src.zoo.rtdetr.matcher').HungarianMatcher
        matcher = Matcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True, alpha=.25, gamma=2.)
    def forbidden(*args, **kwargs):
        raise RuntimeError('forbidden model/training/AP/AR operation')
    from sparse_rtdetr.r0 import evaluation
    evaluation.score_budget = evaluation.bridge_metrics = forbidden
    primary.evaluate_primary_v1 = forbidden
    primary._voc_ap = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    original_call = torch.nn.Module._call_impl
    def checked_call(self, *args, **kwargs):
        name = type(self).__module__
        require(name.startswith('torchvision.transforms.') or type(self) is Matcher,
                'unexpected NN module call: ' + name + '.' + type(self).__name__)
        return original_call(self, *args, **kwargs)
    torch.nn.Module._call_impl = checked_call
    # Do not permit an optimizer update even if future code accidentally constructs one.
    torch.optim.Optimizer.step = forbidden
    torch.cuda.init = forbidden
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).replace('\\', '/').lower()
            require('/images/' not in name and not name.endswith(('.jpg', '.jpeg', '.png', '.bmp')),
                    'image payload read forbidden')
            require(not any(x in name for x in ('/holdout/', '/testdev/', '/test_dev/', '/confirmatory/')), 'forbidden data access')
    sys.addaudithook(audit)
    return torch, parse_raw, RawAugmentation((768, 1344)), rank_candidates, primary, PrimaryImageV2, PrimaryGroundTruth, matcher, matching


def diagnose(c, rows, out, guard):
    import numpy as np
    from r18_cache_core import (overlap, stages, summarize, cardinality, size_group, add_counts, STAGES)
    torch, parse_raw, transform, rank_candidates, primary, ImageRow, GTRow, matcher, matching = native(c)
    from PIL import Image
    aggregates, groups, transitions, pair_stats, population, matching_checks = {}, {}, {}, {}, {}, {}
    native_final, native_totals = {}, {}
    N_flags = {}
    files, match_calls, records_count = [], 0, 0
    actual_ids = [r['stable_image_id'] for r in rows]
    record_path = out / 'GT_RECORDS.csv.gz'
    with gzip.open(record_path, 'wt', encoding='utf-8', newline='') as records_file:
        writer = csv.writer(records_file)
        writer.writerow(['arm', 'image_id', 'GT_id', 'category', 'raw_area', 'size', 'in_A', 'B',
                         'best_raw_IoU', 'best_query', 'best_true_class_rank', 'matched_query', 'matched_IoU',
                         'highest_true_class_query_IoU', 'flags_50_600', 'flags_50_300', 'flags_75_600', 'flags_75_300',
                         'native_primary_50_600', 'native_primary_50_300', 'native_primary_75_600', 'native_primary_75_300'])
        for arm in c['arms_order']:
            source = read(c['arms'][arm]['infer_result'])
            expected = {int(k): json.loads(exact(ref).read_text()) for k, ref in source['predictions'].items()}
            recorded_counts = {int(k): read(ref)['official']['match_counts_by_class_iou_max_dets']
                               for k, ref in c['arms'][arm]['scores'].items()}
            require(set(expected) == {300, 600} and all(len(expected[k]) == len(rows) * k for k in expected), 'prediction coverage differs')
            seen = []
            a_counts, p_counts = Counter(), Counter()
            native_match_by_gt = {}
            stored_centers = {}
            if arm != 'N' and 'centers' in source:
                summary = read(source['centers'])
                old = read(summary['raw_records'])
                stored_centers = {(r['image_id'], r['GT_id']): r['query'] for r in old['records']}
                require(len(stored_centers) == summary['matched_small'], 'stored center pair identities duplicate')
            for batchno, ref in enumerate(source['raw_batches']):
                guard()
                p = exact(ref)
                # Only authenticated raw-prediction paths reach torch.load. Never a checkpoint.
                cache = torch.load(p, map_location='cpu', weights_only=True)
                wanted = {'ids', 'logits', 'boxes'} | ({'coarse'} if arm != 'N' else set())
                require(set(cache) == wanted, 'cache fields differ from pre-budget endpoint format')
                batch = rows[batchno * 4:batchno * 4 + 4]
                require(cache['ids'] == [r['stable_image_id'] for r in batch], 'cache stable IDs/order differ')
                logits, boxes = cache['logits'], cache['boxes']
                require(logits.shape == (len(batch), 300, 10) and boxes.shape == (len(batch), 300, 4), 'raw query/class shape differs; truncated cache forbidden')
                require(logits.dtype == boxes.dtype == torch.float32 and logits.device.type == boxes.device.type == 'cpu', 'raw CPU FP32 dtype differs')
                require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all()) and bool((boxes[:, :, 2:] > 0).all()), 'invalid raw values')
                if arm != 'N':
                    require(cache['coarse'].shape == boxes.shape and cache['coarse'].dtype == torch.float32 and bool(torch.isfinite(cache['coarse']).all()), 'coarse shape/value differs')
                targets, parsed_batch = [], []
                for pos, row in enumerate(batch):
                    raw = exact(dict(path=str(Path(c['annotation_root']) / Path(row['annotation_relative_path']).name),
                                     bytes=row['annotation_size_bytes'], sha256=row['annotation_sha256'])).read_bytes()
                    parsed = parse_raw(raw, row['stable_image_id'], expected_rows=row['raw_annotation_rows'])
                    require(not parsed['counts'].get('nonpositive_area', 0), 'unsupported nonpositive annotation')
                    dummy = Image.new('RGB', (row['width'], row['height']))
                    try:
                        pixels, target, trace = transform(dummy, parsed, epoch=120, seed=0,
                                                          position=batchno * 4 + pos, image_id=row['stable_image_id'], training=False)
                        require(not trace['geometry_steps'], 'not resize-only geometry')
                        del pixels
                    finally:
                        dummy.close()
                    targets.append(target)
                    parsed_batch.append(parsed)
                with matching(matcher):
                    pairs = matcher({'pred_logits': logits, 'pred_boxes': boxes}, targets)['indices']
                match_calls += 1
                for i, row in enumerate(batch):
                    guard()
                    ident = row['stable_image_id']
                    seen.append(ident)
                    gt = parsed_batch[i]
                    target = targets[i]
                    a_by_id = {g: j for j, g in enumerate(target['gt_ids'])}
                    assignment = {target['gt_ids'][g]: q for q, g in zip(*[x.tolist() for x in pairs[i]])}
                    for g, q in assignment.items():
                        native_match_by_gt[ident, g] = q
                    a_counts.update(raw_score1_positive=len(gt['gt']), A=len(a_by_id), A_matched=len(assignment))
                    for area in target['raw_area'].tolist():
                        a_counts['A_' + size_group(area)] += 1
                    image = ImageRow(ident, int(row['width']), int(row['height']))
                    regions = [GTRow(g['id'], ident, int(g['category']), tuple(g['box']), float(g['raw_area']), True, False)
                               for g in gt['rows'] if g['category'] == 0]
                    integral = primary._build_integral(image, regions)
                    B = [g for g in gt['gt'] if primary._ignored_fraction(integral, image, tuple(g['box'])) < .5]
                    b_ids = {g['id'] for g in B}
                    a_counts.update(B=len(B), A_and_B=len(set(a_by_id) & b_ids),
                                    A_not_B=len(set(a_by_id) - b_ids), B_not_A=len(b_ids - set(a_by_id)))
                    xy = torch.cat((boxes[i, :, :2] - boxes[i, :, 2:] / 2,
                                    boxes[i, :, :2] + boxes[i, :, 2:] / 2), -1)
                    xy *= torch.tensor([row['width'], row['height']] * 2, dtype=torch.float32)
                    xy = xy.numpy()
                    probs = logits[i].sigmoid().numpy()
                    qids = np.repeat(np.arange(300), 10)
                    clsids = np.tile(np.arange(10), 300)
                    order = np.lexsort((clsids, qids, -probs.reshape(-1)))
                    ranks = np.empty(3000, dtype=np.int64)
                    ranks[order] = np.arange(1, 3001)
                    ranks = ranks.reshape(300, 10)
                    predictions = {}
                    for budget in c['budgets']:
                        actual = rank_candidates(logits[i], boxes[i], row, budget)
                        predictions[budget] = actual
                        offset = (batchno * 4 + i) * budget
                        require(actual == expected[budget][offset:offset + budget], 'native ranking/coordinates differ from original predictions')
                        for pred in actual:
                            require(pred['bbox_xyxy'] == xy[pred['query_id']].tolist(), 'coordinate conversion differs')
                    keep = np.array([primary._ignored_fraction(integral, image, tuple(float(x) for x in b)) < .5 for b in xy])
                    truth = np.array([g['box'] for g in B], dtype=float).reshape(-1, 4)
                    quality = overlap(xy, truth).T
                    categories = np.array([g['category'] - 1 for g in B], dtype=int)
                    sizes = np.array([size_group(g['raw_area']) for g in B], dtype=str)
                    flags_by_combo = {}
                    native_flags = {}
                    for threshold in c['thresholds']:
                        for budget in c['budgets']:
                            key = f'{threshold}/{budget}'
                            native_states = {}
                            available = [p for p in predictions[budget] if keep[p['query_id']]][:500]
                            for category in range(1, 11):
                                gt_items = [g for g in gt['rows'] if g['category'] == category and
                                            primary._ignored_fraction(integral, image, tuple(g['box'])) < .5]
                                det_items = [p for p in available if p['category_id'] == category]
                                gs, ds = primary._match(
                                    [(p['score'], primary._official_xywh(tuple(p['bbox_xyxy']))) for p in det_items],
                                    [(primary._gt_xywh(tuple(g['box'])), g['score'] == 0) for g in gt_items], threshold)
                                native_states.update({g['id']: s for g, s in zip(gt_items, gs) if g['score'] == 1})
                                add_counts(native_totals, f'{arm}/{budget}/{threshold}/{category}',
                                           dict(GT=len(gs), TP=sum(s == 1 for s in ds), FP=sum(s == 0 for s in ds), ignored=sum(s == -1 for s in ds)))
                            native_flags[key] = [int(native_states.get(g['id'], 0) == 1) for g in B]
                            flags, caps = stages(quality, categories, probs.argmax(1), ranks, keep, threshold, budget)
                            flags_by_combo[key] = flags
                            for scale in ('all', 'small', 'medium', 'large'):
                                mask = np.ones(len(B), dtype=bool) if scale == 'all' else sizes == scale
                                geom = quality[mask] >= threshold
                                label = geom & (probs.argmax(1)[None, :] == categories[mask, None])
                                ranked = geom & (ranks[:, categories[mask]].T <= budget)
                                filtered = ranked & keep[None, :]
                                subcaps = [cardinality(e) for e in (geom, label, ranked, filtered)]
                                value = summarize(flags[mask], subcaps)
                                add_counts(aggregates, f'{arm}/{scale}/{key}', value)
                                add_counts(native_final, f'{arm}/{scale}/{key}',
                                           dict(denominator=int(mask.sum()), matched=sum(v for v, yes in zip(native_flags[key], mask) if yes)))
                            # Necessary category/sequence breakdown: high-IoU small/large only, both budgets.
                            if threshold == .75:
                                for scale in ('small', 'large'):
                                    for group_type, labels in [('category', sorted(set(categories[sizes == scale].tolist()))),
                                                               ('sequence', [row['sequence_key']])]:
                                        for group in labels:
                                            mask = (sizes == scale) & ((categories == group) if group_type == 'category' else True)
                                            vals = dict(zip(STAGES, (int(x) for x in flags[mask].sum(0))))
                                            vals['denominator'] = int(mask.sum())
                                            add_counts(groups, f'{arm}/{scale}/{budget}/{group_type}/{int(group) + 1 if group_type == "category" else group}', vals)
                            for j, g in enumerate(B):
                                gid = ident + '/' + g['id']
                                if arm == 'N':
                                    N_flags[gid, key] = flags[j].tolist()
                                else:
                                    require((gid, key) in N_flags, 'cross-model GT universe differs')
                                    before = N_flags[gid, key]
                                    for k, stage in enumerate(STAGES):
                                        trans = f'{arm}-N/{sizes[j]}/{key}/{stage}/{int(before[k])}{int(flags[j,k])}'
                                        transitions[trans] = transitions.get(trans, 0) + 1
                    # Native final Hungarian pair quality uses the original normalized FP32 geometry.
                    normA = target['boxes'].numpy()
                    prednorm = boxes[i].numpy()
                    normxy = np.concatenate((prednorm[:, :2] - prednorm[:, 2:] / 2, prednorm[:, :2] + prednorm[:, 2:] / 2), axis=1)
                    truthxy = np.concatenate((normA[:, :2] - normA[:, 2:] / 2, normA[:, :2] + normA[:, 2:] / 2), axis=1)
                    Aq = overlap(normxy, truthxy)
                    pair_rows = []
                    for gid, q in assignment.items():
                        j = a_by_id[gid]
                        cat = int(target['labels'][j])
                        pair_rows.append(dict(gid=gid, q=q, cat=cat, iou=float(Aq[q,j]), rank=int(ranks[q,cat]),
                                              small=float(target['raw_area'][j]) <= 1024, B=gid in b_ids))
                    high = [x for x in pair_rows if x['small'] and x['iou'] >= .75]
                    p_counts.update(A_small=int((target['raw_area'].numpy() <= 1024).sum()),
                                    matched_A_small=sum(x['small'] for x in pair_rows), high_IoU_matched_A_small=len(high),
                                    high_IoU_matched_A_small_B=sum(x['B'] for x in high))
                    for budget in c['budgets']:
                        lost = [x for x in high if x['rank'] > budget]
                        all_high = [x for x in pair_rows if x['iou'] >= .75 and x['rank'] > budget]
                        lows = [x for x in pair_rows if x['iou'] < .75 and x['rank'] <= budget]
                        events = [(x,y) for x in all_high for y in lows if x['cat'] == y['cat']]
                        prefix = str(budget) + '_'
                        p_counts.update({prefix + 'high_small_outside': len(lost),
                                         prefix + 'high_small_outside_B': sum(x['B'] for x in lost),
                                         prefix + 'pairs_all_sizes': len(events),
                                         prefix + 'pairs_high_small': sum(x['small'] for x,y in events),
                                         prefix + 'pairs_both_B': sum(x['B'] and y['B'] for x,y in events),
                                         prefix + 'unique_high_small_GT': len({x['gid'] for x,y in events if x['small']}),
                                         prefix + 'images_with_high_small_event': int(any(x['small'] for x,y in events))})
                        for cat in range(10):
                            subset = [(x,y) for x,y in events if x['cat'] == cat and x['small']]
                            p_counts[prefix + 'category_' + str(cat+1) + '_pairs'] += len(subset)
                    for j, g in enumerate(B):
                        q = int(quality[j].argmax())
                        highest = int(np.argmax(probs[:, categories[j]]))
                        mq = assignment.get(g['id'])
                        ai = a_by_id.get(g['id'])
                        flags_text = [''.join(str(int(x)) for x in flags_by_combo[f'{t}/{k}'][j])
                                      for t in c['thresholds'] for k in c['budgets']]
                        writer.writerow([arm, ident, g['id'], g['category'], g['raw_area'], sizes[j],
                                         int(g['id'] in a_by_id), 1, float(quality[j,q]), q, int(ranks[q,categories[j]]),
                                         '' if mq is None else mq, '' if mq is None else float(Aq[mq,ai]),
                                         float(quality[j,highest]), *flags_text,
                                         *[native_flags[f'{t}/{k}'][j] for t in c['thresholds'] for k in c['budgets']]])
                        records_count += 1
                    del integral
                print(json.dumps(dict(arm=arm, batch=batchno+1, images=len(seen))), flush=True)
            require(seen == actual_ids and len(set(seen)) == len(rows), 'actual image coverage differs')
            population[arm] = dict(a_counts)
            pair_stats[arm] = dict(p_counts)
            for budget in c['budgets']:
                for threshold, ti in ((.5, 0), (.75, 5)):
                    for category in range(1, 11):
                        v = native_totals[f'{arm}/{budget}/{threshold}/{category}']
                        actual_counts = [v[x] for x in ('GT','TP','FP','ignored')]
                        require(actual_counts == recorded_counts[budget][category-1][ti][3],
                                'native final matching counts differ from frozen original score receipt')
            if stored_centers:
                matching_checks[arm] = dict(stored_small_pairs=len(stored_centers),
                                           reconstructed_same=sum(native_match_by_gt.get(k) == q for k,q in stored_centers.items()),
                                           reconstructed_different=sum(native_match_by_gt.get(k) != q for k,q in stored_centers.items()),
                                           note='CPU reconstruction versus original GPU endpoint pairs, not training assignment')
            else:
                matching_checks[arm] = dict(original_pairs='UNKNOWN_NOT_RECORDED', reconstructed_CPU_only=True)
            del expected
    require(not torch.cuda.is_initialized(), 'CUDA initialized during diagnostic')
    data = dict(status='PASS', images=c['images'], populations=population,
                stage_counts={k:dict(v) for k,v in aggregates.items()},
                breakdowns={k:dict(v) for k,v in groups.items()}, transitions=transitions,
                boundary_events=pair_stats, matcher_reconstruction=matching_checks,
                native_primary_final={k:dict(v) for k,v in native_final.items()},
                native_primary_original_count_match=True,
                flags_order=list(STAGES), interpretation='endpoint diagnosis only; no AP/AR, no deployable GT oracle, no training-cause proof',
                classification_support='argmax label is parallel; all ten finite scores are not ten semantic positives',
                final_matching='score_order_match is a diagnostic physical-query-unique graph; native_primary_final separately reuses original _match at maxDets500 and checks original recorded counts, without AP/AR; COCO per-GT match UNKNOWN')
    with gzip.open(out/'SUMMARY.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    publish(out/'OVERVIEW.json', {k:data[k] for k in ('status','images','populations','boundary_events','matcher_reconstruction','flags_order','interpretation','classification_support','final_matching')})
    files.extend(reference(out/n) for n in ('SUMMARY.json.gz','OVERVIEW.json','GT_RECORDS.csv.gz'))
    return dict(images_per_arm=c['images'], arms=['N','U','D'], native_matcher_CPU_calls=match_calls,
                actual_raw_batches_per_arm=(c['images']+3)//4, records=records_count, files=files,
                GPU_calls=0, detector_forward_calls=0, AP_AR_calls=0, training_updates=0,
                annotation_only=True, checkpoint_loaded=False, CUDA_initialized=False,
                synthetic=c['profile'] == 'SYNTHETIC_FIXTURE')
