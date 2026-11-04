"""Batch driver for the cached error channels.  CPU only; reads frozen history, writes one output dir.

usage (certified interpreter, -B -I, CUDA hidden, lowest priority):
  python -B -I run_channels.py --config configs/r25_cached_error_channels.json --out DIR --commit SHA \
      --phases admit lists raw r20 [--tags A-seed0 ...]

Never constructs a detector, never touches CUDA, never reads holdout or the 1602 set, never rewrites
a prediction or GT file.  Existing outputs are immutable: a phase output that exists is reused only
when it records the same commit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import core as C

SEEDS = (0, 1, 2)
ARMS = ('A', 'C')
LIST_TAGS = tuple(f'{a}-seed{s}' for s in SEEDS for a in ARMS)
THRESHOLDS = (0.5, 0.75)


def dumps(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=1) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ref_of(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def publish(path, value):
    with Path(path).open('xb') as f:
        f.write(dumps(value))
    return ref_of(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_by_path(name, path, expected_sha):
    require(sha(path) == expected_sha, f'frozen file differs: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cost(start_wall, start_cpu):
    return {'wall_seconds': time.perf_counter() - start_wall, 'cpu_seconds': time.process_time() - start_cpu,
            'peak_rss_kib_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class Context:
    """Authenticated history, following the frozen C stage entry (contract, then context)."""

    def __init__(self, cfg):
        self.cfg = cfg
        parent = Path(cfg['parent_contract']['path'])
        require(sha(parent) == cfg['parent_contract']['sha256'], 'parent contract SHA differs')
        contract = json.loads(parent.read_bytes())
        source = Path(contract['source_root'])
        for item in contract['source_files']:
            rel = Path(item['path'])
            require(not rel.is_absolute() and '..' not in rel.parts and sha(source / rel) == item['sha256'],
                    'frozen source differs: ' + item['path'])
        self.source = source
        self.source_file_count = len(contract['source_files'])
        sys.path.insert(0, str(source))
        import softnms_job as j
        self.j = j
        self.torch = j.cpu_only()
        self.contract = j.load_contract(parent)
        self.specs, self.rows, self.data_spec = j.context(self.contract)
        from sparse_rtdetr.r0 import evaluation as ev
        from sparse_rtdetr.baseline import primary_evaluator as primary
        from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2
        self.ev = ev
        self.primary = primary
        self.PrimaryImageV2 = PrimaryImageV2
        self.parsed = j.parsed_gt(self.rows, self.data_spec)
        self.plan = j.read(j.PLAN)
        self.n_img = len(self.rows)
        require(self.n_img == 548, 'development membership differs')
        self.groups = {'ALL': list(range(self.n_img))}
        sizes = defaultdict(list)
        for i, r in enumerate(self.rows):
            sizes[f"{r['width']}x{r['height']}"].append(i)
        for name in sorted(sizes, key=lambda s: (int(s.split('x')[0]), int(s.split('x')[1]))):
            self.groups[name] = sizes[name]
        self.sequence = [Path(r['annotation_relative_path']).stem.split('_')[0] for r in self.rows]

    def spec(self, seed):
        return next(s for s in self.specs if s['seed'] == seed)

    def guard(self, out):
        return self.j.resources(out, self.plan)


def gt_identity(ctx):
    dataset, _, counts = ctx.ev.bridge_inputs(ctx.rows, ctx.parsed, [])
    digest = hashlib.sha256(json.dumps(dataset['annotations'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, {k: counts[k] for k in ('GT_before', 'GT_after')}, dataset


def evaluator_identity(ctx):
    import numpy
    import pycocotools
    import pycocotools.coco as coco_mod
    import pycocotools.cocoeval as eval_mod
    from pycocotools import _mask
    src = ctx.source / 'src/sparse_rtdetr'
    return {'python_executable': sys.executable, 'python_version': platform.python_version(),
            'python_sha256': sha(sys.executable), 'numpy': numpy.__version__,
            'r0_evaluation_py': ref_of(src / 'r0/evaluation.py'),
            'primary_evaluator_py': ref_of(src / 'baseline/primary_evaluator.py'),
            'pycocotools_cocoeval_py': ref_of(eval_mod.__file__), 'pycocotools_coco_py': ref_of(coco_mod.__file__),
            'pycocotools_mask_so': ref_of(_mask.__file__), 'pycocotools_dir': os.path.dirname(pycocotools.__file__),
            'matching': 'pycocotools COCOeval on frozen r0.evaluation.bridge_inputs output; maxDets [1,10,100]; 10 classes; macro over classes with in-range GT'}


def phase_admit(ctx, out, commit):
    started, cpu = time.perf_counter(), time.process_time()
    cfg = ctx.cfg
    j = ctx.j
    digest, counts, _ = gt_identity(ctx)
    ids = [r['stable_image_id'] for r in ctx.rows]
    inputs = []
    for seed in SEEDS:
        spec = ctx.spec(seed)
        for arm in ARMS:
            ref = spec['evaluation']['predictions'][arm]['600']
            metric = spec['evaluation']['metrics'][arm]['600']
            j.exact(ref)
            j.exact(metric)
            inputs.append({'role': 'certified final list', 'tag': f'{arm}-seed{seed}', 'seed': seed, 'arm': arm,
                           'budget': 600, 'endpoint': 'e120 EMA', 'checkpoint_sha256': spec['identity']['checkpoint_sha256'],
                           'prediction': ref, 'metric': metric, 'images': 548, 'ordered_members_sha256': j.digest(ids),
                           'GT_bridge_annotations_sha256': digest, 'GT_counts': counts})
    spec0 = ctx.spec(0)
    raw = cfg['seed0_A_raw_cache']
    j.exact(raw)
    inputs.append({'role': 'seed0 A raw query cache (300 queries)', 'reference': raw,
                   'checkpoint_sha256': spec0['identity']['checkpoint_sha256']})
    for k, view in enumerate(spec0['full']['raw_view_refs']):
        j.exact(view)
    inputs.append({'role': 'seed0 C raw views (137 batches, five views, 1500 queries per image)',
                   'files': len(spec0['full']['raw_view_refs']),
                   'references_sha256': j.digest(spec0['full']['raw_view_refs']),
                   'source': str(Path(spec0['full']['raw_view_refs'][0]['path']).parent)})
    stage = cfg['c_stage']
    index = json.loads(Path(stage['return_index']).read_bytes())
    server = {x['relative_path']: x for x in index['server_only'] + index['small_return']}
    stage_refs = {}
    for name in [f'attempt-002/seed{s}-gt.jsonl' for s in SEEDS] + [f'attempt-002/seed{s}-SUMMARY.json' for s in SEEDS]:
        p = Path(stage['root']) / name
        require(sha(p) == server[name]['sha256'] and p.stat().st_size == server[name]['bytes'], 'C stage receipt differs: ' + name)
        stage_refs[name] = ref_of(p)
    inputs.append({'role': 'old C stage receipts, reused unchanged', 'files': stage_refs,
                   'code': {'c_stage_core.py': stage['core_sha256'], 'c_stage_audit.py': stage['audit_sha256']},
                   'bucket_definition': stage['definition']})
    r20 = cfg['r20']
    for arm, ref in r20['arms'].items():
        j.exact(ref)
        inputs.append({'role': 'r20 final list', 'arm': arm, 'run': 'run-003', 'budget': 600, 'prediction': ref,
                       'endpoint': 'e120 EMA seed0 parent, window pass per arm', 'checkpoint_sha256': spec0['identity']['checkpoint_sha256'],
                       'release_manifest': r20['release_manifest_sha256'], 'release_commit': r20['release_commit']})
    r20_code = {name: ref_of(Path(r20['release_source']) / rel) for name, rel in r20['code'].items()}
    require(r20_code['zoom_refine']['sha256'] == r20['zoom_refine_sha256']
            and r20_code['run_zoom_refine']['sha256'] == r20['run_zoom_refine_sha256'], 'r20 frozen rendering code differs')
    tide = {}
    for rel, value in cfg['tide']['files'].items():
        p = Path(cfg['tide']['root']) / rel
        require(sha(p) == value, 'TIDE file differs: ' + rel)
        tide[rel] = value
    sizes = {k: len(v) for k, v in ctx.groups.items() if k != 'ALL'}
    _, _, dataset = gt_identity(ctx)
    anns = dataset['annotations']
    area = np.array([g['area'] for g in anns], dtype=np.float64)
    crowd = np.array([bool(g['iscrowd']) for g in anns], dtype=bool)
    before_small = sum(1 for parsed in ctx.parsed for obj in parsed['rows']
                       if obj['category'] in range(1, 11) and obj['score'] != 0 and 0 < obj['raw_area'] <= 1024)
    before_all = sum(1 for parsed in ctx.parsed for obj in parsed['rows'] if obj['category'] in range(1, 11))
    denominators = {'GT_categories_1_10_before_raster': before_all, 'GT_after_raster': len(anns),
                    'class_score_0_crowd_after_raster': int(crowd.sum()),
                    'small_before_raster_score1_area_le_1024': before_small,
                    'small_after_raster_area_le_1024': int(np.logical_and(~crowd, area <= 1024).sum()),
                    'small_after_raster_area_lt_1024': int(np.logical_and(~crowd, area < 1024).sum()),
                    'after_raster_area_eq_1024': int(np.logical_and(~crowd, area == 1024).sum()),
                    'after_raster_area_nonpositive': int(np.logical_and(~crowd, area <= 0).sum())}
    save_npz(Path(out) / 'gt-table.npz', raw_id=np.array([g['raw_annotation_id'] for g in anns]),
             img=np.array([g['image_id'] for g in anns], dtype=np.int64),
             cat=np.array([g['category_id'] for g in anns], dtype=np.int64),
             bbox=np.array([g['bbox'] for g in anns], dtype=np.float64), area=area, crowd=crowd)
    images = [{'id': r['stable_image_id'], 'width': r['width'], 'height': r['height'], 'sequence': s}
              for r, s in zip(ctx.rows, ctx.sequence)]
    manifest = {'task_id': cfg['task_id'], 'commit': commit, 'time_unix': time.time(), 'inputs': inputs,
                'r20_frozen_code': r20_code, 'evaluator': evaluator_identity(ctx),
                'parent_contract': cfg['parent_contract'], 'frozen_source_files_verified': ctx.source_file_count,
                'tide': {k: cfg['tide'][k] for k in ('repository', 'commit', 'license')}, 'tide_files': tide,
                'image_size_groups': sizes, 'data_manifest_rows_sha256': j.digest(ctx.rows),
                'data_spec': ctx.data_spec, 'excluded': cfg['excluded'], 'images': images,
                'gt_denominators': denominators,
                'history_notice': 'All inputs belong to closed rounds; every number derived here is an exploratory decomposition made after those results were known.'}
    ref = publish(Path(out) / 'INPUT_MANIFEST.json', manifest)
    print(json.dumps({'phase': 'admit', 'manifest': ref['sha256'], **cost(started, cpu)}), flush=True)
    return ref


def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def list_summary(ctx, runs, gt):
    return {f"{C.AREA_NAMES[a]}@{t}": C.run_summary(rt, gt) for (a, t), rt in runs.items()}


def process_list(ctx, out, tag, seed, arm, commit):
    jpath = Path(out) / f'list-{tag}.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing output has another commit: ' + tag)
        return
    started, cpu = time.perf_counter(), time.process_time()
    resources = ctx.guard(out)
    j, ev = ctx.j, ctx.ev
    spec = ctx.spec(seed)
    ref = spec['evaluation']['predictions'][arm]['600']
    preds = json.loads(j.exact(ref, contents=True))
    proof = j.evaluation_input_proof(spec, ctx.rows, ctx.parsed, preds, arm, 600)
    dataset, det, counts = ev.bridge_inputs(ctx.rows, ctx.parsed, preds)
    del preds
    gc.collect()
    e, values = C.evaluate_certified(dataset, det)
    certified = spec['metrics'][arm]['600']['COCO_bridge']['metrics']
    diffs = {k: abs(values[k] - certified[k]) for k in certified}
    require(set(values) == set(certified) and max(diffs.values()) == 0.0, f'{tag}: certified reproduction differs {diffs}')
    gt, dt = C.gt_table(dataset), C.det_table(det)
    runs = {(a, t): C.collect_run(e, a, t, gt.n) for a in (C.AREA_ALL, C.AREA_SMALL) for t in THRESHOLDS}
    port = C.check_port(e, runs)
    require(port == 0.0, f'{tag}: accumulate port differs by {port}')
    best_filtered, best_evaluated = C.analyze_errors(gt, dt, runs, ctx.n_img, classify=True)
    summaries = list_summary(ctx, runs, gt)
    groups = C.group_metrics(e, ctx.groups)
    for key in ('AP', 'AP50', 'AP75', 'APS', 'AP75_small'):
        require(groups['ALL'][key] == certified[key], f'{tag}: group ALL differs on {key}')
    small = runs[(C.AREA_SMALL, 0.5)].gt_nonig[gt.id]
    require(np.array_equal(small, runs[(C.AREA_SMALL, 0.75)].gt_nonig[gt.id]), 'small GT differs across IoU')
    rule = np.logical_and.reduce((np.logical_not(gt.crowd), gt.area >= 0, gt.area <= 1024))
    require(np.array_equal(small, rule), 'small GT rule differs from the COCO range')
    evaluated = np.sort(runs[(C.AREA_ALL, 0.5)].det_id)
    arrays = dict(n_img=np.array(ctx.n_img), gt_id=gt.id, gt_img=gt.img, gt_cat=gt.cat, gt_bbox=gt.bbox,
                  gt_area=gt.area, gt_crowd=gt.crowd, small=small, best_filtered=best_filtered,
                  best_evaluated=best_evaluated, eval_id=evaluated, eval_img=dt.img[evaluated - 1],
                  eval_cat=dt.cat[evaluated - 1], eval_score=dt.score[evaluated - 1], eval_bbox=dt.bbox[evaluated - 1],
                  filtered_count=np.array(len(det)))
    for (a, t), rt in runs.items():
        key = f"{C.AREA_NAMES[a]}_{str(t).replace('.', '')}"
        arrays[f'used_{key}'] = rt.gt_used[gt.id]
        arrays[f'usable_{key}'] = rt.gt_usable[gt.id]
    save_npz(Path(out) / f'list-{tag}.npz', **arrays)
    result = {'tag': tag, 'seed': seed, 'arm': arm, 'commit': commit, 'prediction': ref, 'input_proof': proof,
              'spatial_filter_counts': counts, 'certified_metrics_reproduced': values, 'max_abs_difference_vs_certified': 0.0,
              'accumulate_port_max_abs_difference': 0.0, 'runs': summaries, 'groups': groups,
              'COCOeval_full_calls': 1, 'oracle_evaluations': len(runs) * len(C.ORACLES),
              'resource_at_start': resources, 'cost': cost(started, cpu),
              'cuda_initialized': ctx.torch.cuda.is_initialized()}
    publish(jpath, result)
    print(json.dumps({'tag': tag, 'cost': result['cost'], 'AP75_small': values['AP75_small']}), flush=True)
    del e, runs, dataset, det, gt, dt, arrays
    gc.collect()


class _Item:
    __slots__ = ('category_id', 'box_xyxy')

    def __init__(self, category_id, box_xyxy):
        self.category_id = category_id
        self.box_xyxy = box_xyxy


def phase_raw(ctx, out, commit):
    jpath = Path(out) / 'raw-seed0.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing raw output has another commit')
        return
    started, cpu = time.perf_counter(), time.process_time()
    cfg = ctx.cfg
    j, ev, primary, torch = ctx.j, ctx.ev, ctx.primary, ctx.torch
    stage = cfg['c_stage']
    sys.path.insert(0, str(Path(stage['source'])))
    core_path = Path(stage['source']) / 'c_stage_core.py'
    audit_path = Path(stage['source']) / 'c_stage_audit.py'
    require(sha(core_path) == stage['core_sha256'] and sha(audit_path) == stage['audit_sha256'], 'C stage source differs')
    import c_stage_audit as audit
    import c_stage_core as sc
    spec = ctx.spec(0)
    preds = json.loads(j.exact(spec['evaluation']['predictions']['A']['600'], contents=True))
    grouped = j.group(preds, spec['ids'], 600)
    del preds
    cache = torch.load(io.BytesIO(j.exact(cfg['seed0_A_raw_cache'], contents=True)), map_location='cpu', weights_only=True)
    require(cache['checkpoint_sha256'] == spec['identity']['checkpoint_sha256'], 'A raw cache checkpoint differs')
    logits, boxes = cache['pred_logits'], cache['pred_boxes']
    require(tuple(logits.shape) == (548, 300, 10) and tuple(boxes.shape) == (548, 300, 4), 'A raw cache shape differs')
    dataset, _, _ = ev.bridge_inputs(ctx.rows, ctx.parsed, [])
    anns = dataset['annotations']
    small_by_image = defaultdict(list)
    for index, g in enumerate(anns):
        if g['iscrowd'] == 0 and 0 < g['area'] <= 1024:
            small_by_image[g['image_id'] - 1].append(index)
    receipt = {}
    with (Path(stage['root']) / 'attempt-002/seed0-gt.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['budget'] == 600:
                receipt[(r['image_id'], r['raw_annotation_id'])] = r
    require(len(receipt) == sum(len(v) for v in small_by_image.values()), 'C stage receipt denominator differs')
    bucket_code = {b: i for i, b in enumerate(sc.BUCKETS)}
    gt_index, a_raw, c_raw, tile_raw, rec_raw, a_stage, c_stage = [], [], [], [], [], [], []
    a_bucket = {t: [] for t in ('0.5', '0.75')}
    c_bucket = {t: [] for t in ('0.5', '0.75')}
    rank_identical, view0_identical, images = 0, 0, 0
    refs = spec['full']['raw_view_refs']
    for batch, ref in enumerate(refs):
        raw = torch.load(io.BytesIO(j.exact(ref, contents=True)), map_location='cpu', weights_only=True)
        part = ctx.rows[4 * batch: 4 * batch + 4]
        maps = audit.geometry(raw, part, j, torch)
        for local, (row, geomap) in enumerate(zip(part, maps)):
            i = 4 * batch + local
            image = row['stable_image_id']
            ann = ctx.parsed[i]
            ranked = ev.rank_candidates(logits[i], boxes[i], row, 600)
            require(ranked == grouped[image], 'A raw cache does not reproduce the certified list: ' + image)
            rank_identical += 1
            b = boxes[i]
            xy = torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1)
            xy = (xy * torch.tensor([row['width'], row['height']] * 2, dtype=torch.float32)).tolist()
            a_boxes = [tuple(v) for v in xy]
            require(a_boxes == [geomap[(0, q)] for q in range(300)], 'C view 0 differs from A raw cache: ' + image)
            view0_identical += 1
            all_boxes = list(geomap.values())
            tiles = [box for key, box in geomap.items() if key[0] != 0]
            im = ctx.PrimaryImageV2(image, row['width'], row['height'])
            integral = primary._build_integral(im, [g for g in ev.truth(row, ann) if g.category_id == 0])
            items = [_Item(p['category_id'], tuple(p['bbox_xyxy'])) for p in grouped[image]]
            kept = [p for p in items if primary._ignored_fraction(integral, im, p.box_xyxy) < .5]
            capped = sc.class_cap(kept)
            by_class = {}
            for cat in range(1, 11):
                pool = [p.box_xyxy for p in items if p.category_id == cat]
                by_class[cat] = {sc.STAGES[0]: a_boxes, sc.STAGES[1]: pool, sc.STAGES[2]: pool, sc.STAGES[3]: pool,
                                 sc.STAGES[4]: [p.box_xyxy for p in kept if p.category_id == cat],
                                 sc.STAGES[5]: [p.box_xyxy for p in capped if p.category_id == cat]}
            for index in small_by_image.get(i, []):
                g = anns[index]
                x, y, w, h = g['bbox']
                box = [x, y, x + w, y + h]
                result = sc.availability(box, by_class[g['category_id']])
                r = receipt[(image, g['raw_annotation_id'])]
                gt_index.append(index)
                a_raw.append(result['max_iou'][sc.STAGES[0]])
                a_stage.append([result['max_iou'][s] for s in sc.STAGES])
                c_raw.append(float(sc.ious(box, all_boxes).max(initial=0.)))
                tile_raw.append(float(sc.ious(box, tiles).max(initial=0.)))
                rec_raw.append(r['max_iou']['raw_geometry'])
                c_stage.append([r['max_iou'][s] for s in sc.STAGES])
                for t in a_bucket:
                    a_bucket[t].append(bucket_code[result['first_unavailable'][t]])
                    c_bucket[t].append(bucket_code[r['first_unavailable'][t]])
            images += 1
    a_raw, c_raw, tile_raw, rec_raw = map(np.array, (a_raw, c_raw, tile_raw, rec_raw))
    require(np.all(a_raw <= c_raw), 'A raw geometry must be nested in C raw geometry')
    mismatch = int(np.count_nonzero(c_raw != rec_raw))
    save_npz(Path(out) / 'raw-seed0.npz', gt_index=np.array(gt_index, dtype=np.int64), a_raw=a_raw, c_raw=c_raw,
             tile_raw=tile_raw, receipt_raw=rec_raw, a_stage=np.array(a_stage), c_stage=np.array(c_stage),
             a_bucket_05=np.array(a_bucket['0.5'], dtype=np.int8), a_bucket_075=np.array(a_bucket['0.75'], dtype=np.int8),
             c_bucket_05=np.array(c_bucket['0.5'], dtype=np.int8), c_bucket_075=np.array(c_bucket['0.75'], dtype=np.int8))
    result = {'commit': commit, 'images': images, 'small_GT': len(gt_index), 'stages': list(sc.STAGES), 'buckets': list(sc.BUCKETS),
              'A_raw_rank_reproduces_certified_images': rank_identical, 'C_view0_equals_A_raw_images': view0_identical,
              'C_raw_recomputed_differs_from_receipt_count': mismatch, 'A_raw_nested_in_C_raw': True,
              'A_stage_mapping': 'raw_geometry=300 class-agnostic boxes; view_class_pool=hard_nms=top_budget=certified A top600 same-class (A has no NMS and no second budget cut); raw_ignore and class100 as C stage',
              'cost': cost(started, cpu), 'cuda_initialized': torch.cuda.is_initialized()}
    publish(jpath, result)
    print(json.dumps({'phase': 'raw', 'mismatch': mismatch, 'cost': result['cost']}), flush=True)


def map_det_to_pred(preds, det, rows):
    ids = {r['stable_image_id']: i + 1 for i, r in enumerate(rows)}
    per_image = [[] for _ in rows]
    for pi, p in enumerate(preds):
        per_image[ids[p['image_id']] - 1].append(pi)
    mapping = np.full(len(det), -1, dtype=np.int64)
    di = 0
    for img, indices in enumerate(per_image):
        for pi in indices:
            if di < len(det) and det[di]['image_id'] == img + 1:
                p, d = preds[pi], det[di]
                b = p['bbox_xyxy']
                if d['category_id'] == p['category_id'] and d['score'] == p['score'] \
                        and d['bbox'] == [b[0], b[1], b[2] - b[0], b[3] - b[1]]:
                    mapping[di] = pi
                    di += 1
    require(di == len(det) and mapping.min() >= 0, 'filtered list is not a subsequence of the predictions')
    return mapping


def r20_list(ctx, preds, name, reference=None):
    dataset, det, counts = ctx.ev.bridge_inputs(ctx.rows, ctx.parsed, preds)
    mapping = map_det_to_pred(preds, det, ctx.rows)
    e, values = C.evaluate_certified(dataset, det)
    gt, dt = C.gt_table(dataset), C.det_table(det)
    runs = {(a, t): C.collect_run(e, a, t, gt.n) for a in (C.AREA_ALL, C.AREA_SMALL) for t in THRESHOLDS}
    require(C.check_port(e, runs) == 0.0, name + ': accumulate port differs')
    _, best = C.analyze_errors(gt, dt, runs, ctx.n_img, classify=False)
    groups = C.group_metrics(e, ctx.groups)
    diffs = None
    if reference is not None:
        diffs = {k: abs(values[k] - reference[k]) for k in reference}
        require(max(diffs.values()) == 0.0, f'{name}: frozen r20 score differs {diffs}')
    keep = dict(values=values, groups=groups, counts=counts, reference_diffs=diffs, gt=gt, dt=dt, best=best,
                mapping=mapping, small_tp={t: runs[(C.AREA_SMALL, t)].gt_used[gt.id] for t in THRESHOLDS},
                small=runs[(C.AREA_SMALL, 0.5)].gt_nonig[gt.id],
                small_match=runs[(C.AREA_SMALL, 0.5)].gt_match[gt.id],
                all_tp={t: runs[(C.AREA_ALL, t)].gt_used[gt.id] for t in THRESHOLDS},
                all_nonig=runs[(C.AREA_ALL, 0.5)].gt_nonig[gt.id])
    del e, runs
    gc.collect()
    return keep


def regression_rows(keep, changed, windows, image_of_pred):
    """Matched small GT at IoU .5 whose detection box came from the window pass."""
    gt, dt = keep['gt'], keep['dt']
    rows = []
    for index in np.nonzero(np.logical_and(keep['small'], keep['small_tp'][0.5]))[0]:
        det_id = int(keep['small_match'][index])
        pi = int(keep['mapping'][det_id - 1])
        if not changed[pi]:
            continue
        x, y, w, h = dt.bbox[det_id - 1]
        gx, gy, gw, gh = gt.bbox[index]
        img = int(image_of_pred[pi])
        rows.append((img, x + w / 2, y + h / 2, gx + gw / 2, gy + gh / 2, gw, gh))
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


def phase_r20(ctx, out, commit):
    jpath = Path(out) / 'r20.json'
    if jpath.exists():
        require(json.loads(jpath.read_bytes())['commit'] == commit, 'existing r20 output has another commit')
        return
    started, cpu = time.perf_counter(), time.process_time()
    cfg = ctx.cfg['r20']
    j = ctx.j
    zr = load_by_path('frozen_zoom_refine', Path(cfg['release_source']) / cfg['code']['zoom_refine'], cfg['zoom_refine_sha256'])
    arms = {k: json.loads(j.exact(v, contents=True)) for k, v in cfg['arms'].items()}
    comparison = json.loads(Path(cfg['comparison']).read_bytes())
    require(sha(cfg['comparison']) == cfg['comparison_sha256'], 'r20 comparison differs')
    receipts = {k: json.loads(Path(p).read_bytes()) for k, p in cfg['receipts'].items()}
    for name in ('A1', 'A2'):
        require(receipts[name]['pred-top600_sha256'] == cfg['arms'][name]['sha256'], 'r20 receipt prediction differs: ' + name)
    index_of = {r['stable_image_id']: i for i, r in enumerate(ctx.rows)}
    base = arms['A0r']
    changed, image_of_pred = {}, None
    for name in ('A1', 'A2'):
        preds = arms[name]
        require(len(preds) == len(base), 'r20 list length differs')
        flags = np.zeros(len(preds), dtype=bool)
        for k, (p, q) in enumerate(zip(preds, base)):
            require(p['image_id'] == q['image_id'] and p['query_id'] == q['query_id'] and p['category_id'] == q['category_id']
                    and p['score'] == q['score'], 'r20 whole pass identity differs at row %d' % k)
            flags[k] = p['bbox_xyxy'] != q['bbox_xyxy']
        changed[name] = flags
        require(int(flags.sum()) == receipts[name]['invariants']['changed_boxes'], 'changed box total differs: ' + name)
    image_of_pred = np.array([index_of[p['image_id']] for p in base], dtype=np.int64)
    for name in ('A1', 'A2'):
        per = np.bincount(image_of_pred[changed[name]], minlength=ctx.n_img)
        audit = {r['image_id']: r['repaired'] for r in receipts[name]['substitution_audit']}
        require(all(per[index_of[k]] == v for k, v in audit.items()), 'repaired count per image differs: ' + name)
    plans = {name: {r['image_id']: r['windows'] for r in receipts[name]['plan_audit']} for name in ('A1', 'A2')}
    require(plans['A1'] == plans['A2'], 'window plans differ across arms')
    registration = []
    for i, row in enumerate(ctx.rows):
        windows = plans['A1'][row['stable_image_id']]
        require(len(windows) == 1, 'expected one window per image')
        w = windows[0]
        x0, y0, x1, y1 = w['crop_box']
        window = zr.Window(1, x0, y0, x1, y1, row['width'], row['height'])
        rect = zr.canvas_rect(window)
        g = C.control_registration(x0, y0, x1, y1, row['width'], row['height'])
        require(g['j0'] == max(int(np.floor(rect[0])), 0) and g['i0'] == max(int(np.floor(rect[1])), 0)
                and g['j1'] == min(int(np.ceil(rect[2])), 1344) and g['i1'] == min(int(np.ceil(rect[3])), 768),
                'registration helper differs from frozen canvas_rect')
        registration.append({**g, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    corrected = []
    for k, p in enumerate(arms['A1']):
        if changed['A1'][k]:
            g = registration[image_of_pred[k]]
            ww, wh = g['x1'] - g['x0'], g['y1'] - g['y0']
            b = p['bbox_xyxy']
            q = dict(p)
            q['bbox_xyxy'] = [C.corrected_coordinate(b[0], g['x0'], ww, g['j0'], g['j1'], g['sx']),
                              C.corrected_coordinate(b[1], g['y0'], wh, g['i0'], g['i1'], g['sy']),
                              C.corrected_coordinate(b[2], g['x0'], ww, g['j0'], g['j1'], g['sx']),
                              C.corrected_coordinate(b[3], g['y0'], wh, g['i0'], g['i1'], g['sy'])]
            corrected.append(q)
        else:
            corrected.append(p)
    reference = {name: comparison['scores'][name]['600'] for name in ('A1', 'A2')}
    lists = {}
    for name, preds, ref in (('A1', arms['A1'], reference['A1']), ('A2', arms['A2'], reference['A2']), ('A1c', corrected, None)):
        lists[name] = r20_list(ctx, preds, name, ref)
        print(json.dumps({'r20': name, 'APS': lists[name]['values']['APS'], 'AP75_small': lists[name]['values']['AP75_small']}), flush=True)
    changed['A1c'] = changed['A1']
    gt = lists['A1']['gt']
    small = lists['A1']['small']
    img_of_gt = gt.img - 1
    group_of_image = {}
    for name, images in ctx.groups.items():
        if name != 'ALL':
            for i in images:
                group_of_image[i] = name
    group_tables = {}
    for name in ctx.groups:
        images = np.array(ctx.groups[name], dtype=np.int64)
        in_group = np.isin(img_of_gt, images)
        mask = np.logical_and(small, in_group)
        tables = {'images': len(images), 'small_GT': int(mask.sum()),
                  'all_GT_non_crowd': int(np.logical_and(np.logical_not(gt.crowd), in_group).sum()),
                  'metrics': {a: lists[a]['groups'][name] for a in lists}, 'pairs': {}}
        for t in THRESHOLDS:
            tables['pairs'][str(t)] = {}
            for x, y in (('A1', 'A2'), ('A1c', 'A2'), ('A1', 'A1c')):
                tx, ty = lists[x]['small_tp'][t][mask], lists[y]['small_tp'][t][mask]
                tables['pairs'][str(t)][f'{x}_vs_{y}'] = {
                    f'{x}_only': int(np.logical_and(tx, np.logical_not(ty)).sum()),
                    f'{y}_only': int(np.logical_and(ty, np.logical_not(tx)).sum()),
                    'both': int(np.logical_and(tx, ty).sum()),
                    'neither': int(np.logical_and(np.logical_not(tx), np.logical_not(ty)).sum())}
        both = np.logical_and.reduce((mask, lists['A1']['small_tp'][0.5], lists['A2']['small_tp'][0.5]))
        tables['paired_best_IoU_small_TP_at_.5_in_both'] = {
            'n': int(both.sum()),
            'A1_mean': float(lists['A1']['best'][both].mean()) if both.any() else None,
            'A2_mean': float(lists['A2']['best'][both].mean()) if both.any() else None,
            'A1c_mean': float(lists['A1c']['best'][both].mean()) if both.any() else None}
        reg = [registration[i] for i in images.tolist()]
        tables['control_registration_raw_px'] = {
            key: {'mean_abs': float(np.mean([abs(r[key]) for r in reg])), 'max_abs': float(np.max([abs(r[key]) for r in reg]))}
            for key in ('error_left', 'error_right', 'error_top', 'error_bottom')}
        tables['changed_boxes'] = {a: int(np.isin(image_of_pred[changed[a]], images).sum()) for a in ('A1', 'A2')}
        group_tables[name] = tables
    regression = {}
    for arm in ('A1', 'A2', 'A1c'):
        rows = regression_rows(lists[arm], changed[arm], registration, image_of_pred)
        out_arm = {}
        for name in ctx.groups:
            images = ctx.groups[name]
            sel = rows[np.isin(rows[:, 0].astype(np.int64), images)] if len(rows) else rows
            ex, ey = np.zeros(len(sel)), np.zeros(len(sel))
            for k in range(len(sel)):
                g = registration[int(sel[k, 0])]
                ex[k] = C.registration_error(sel[k, 1], g['x0'], g['x1'] - g['x0'], g['j0'], g['j1'], g['sx'])
                ey[k] = C.registration_error(sel[k, 2], g['y0'], g['y1'] - g['y0'], g['i0'], g['i1'], g['sy'])
            out_arm[name] = {'x': C.ols(ex, sel[:, 1] - sel[:, 3]), 'y': C.ols(ey, sel[:, 2] - sel[:, 4]),
                             'mean_abs_predicted_error_x': float(np.mean(np.abs(ex))) if len(ex) else None,
                             'mean_abs_predicted_error_y': float(np.mean(np.abs(ey))) if len(ey) else None}
        regression[arm] = out_arm
    same_group = {}
    for name in ctx.groups:
        if name == 'ALL':
            continue
        sel = np.isin(image_of_pred, ctx.groups[name])
        a1 = [arms['A1'][k]['bbox_xyxy'] for k in np.nonzero(sel)[0]]
        a2 = [arms['A2'][k]['bbox_xyxy'] for k in np.nonzero(sel)[0]]
        difference = np.abs(np.array(a1) - np.array(a2))
        same_group[name] = {'rows': len(a1), 'rows_with_any_box_difference': int(np.count_nonzero(difference.max(1) > 0)),
                            'max_abs_box_difference_px': float(difference.max()),
                            'rows_with_box_difference_above_1e-3_px': int(np.count_nonzero(difference.max(1) > 1e-3))}
    result = {'commit': commit, 'frozen_scores_reproduced': {k: lists[k]['reference_diffs'] for k in ('A1', 'A2')},
              'values': {k: lists[k]['values'] for k in lists}, 'groups': group_tables,
              'registration_regression': regression, 'A1_A2_box_identity_by_group': same_group,
              'spatial_filter_counts': {k: lists[k]['counts'] for k in lists},
              'A1c': 'exploratory: A1 window boxes remapped through the integer patch that control_pixels actually used; same selected boxes, scores, classes and order; not an r20 run and no change to its verdict',
              'COCOeval_full_calls': len(lists), 'cost': cost(started, cpu), 'cuda_initialized': ctx.torch.cuda.is_initialized()}
    arrays = {'registration_' + k: np.array([r[k] for r in registration], dtype=np.float64) for k in registration[0]}
    arrays.update(gt_small=small)
    for name in lists:
        for t in THRESHOLDS:
            arrays[f"{name}_tp_{str(t).replace('.', '')}"] = lists[name]['small_tp'][t]
        arrays[f'{name}_best'] = lists[name]['best']
    save_npz(Path(out) / 'r20.npz', **arrays)
    publish(jpath, result)
    print(json.dumps({'phase': 'r20', 'cost': result['cost']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--phases', nargs='+', choices=('admit', 'lists', 'raw', 'r20'), required=True)
    parser.add_argument('--tags', nargs='*', default=list(LIST_TAGS))
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    require(sys.executable == json.loads(Path(args.config).read_bytes())['python'], 'interpreter differs from the config')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_bytes())
    started, cpu = time.perf_counter(), time.process_time()
    ctx = Context(cfg)
    print(json.dumps({'context': cost(started, cpu)}), flush=True)
    if 'admit' in args.phases and not (out / 'INPUT_MANIFEST.json').exists():
        phase_admit(ctx, out, args.commit)
    require((out / 'INPUT_MANIFEST.json').exists(), 'write the input manifest first')
    if 'lists' in args.phases:
        for tag in args.tags:
            arm, seed = tag.split('-seed')
            process_list(ctx, out, tag, int(seed), arm, args.commit)
    if 'raw' in args.phases:
        phase_raw(ctx, out, args.commit)
    if 'r20' in args.phases:
        phase_r20(ctx, out, args.commit)
    require(not ctx.torch.cuda.is_initialized(), 'CUDA was initialized')
    print(json.dumps({'done': args.phases, 'total': cost(started, cpu)}), flush=True)


if __name__ == '__main__':
    main()
