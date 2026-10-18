"""Bounded CPU review, NOT a production evaluator or a launch command.

The alternate evaluator is an isolated namespace of the unchanged certified
module. Only GT geometry admission and its shared coordinate converter are
extended to nonnegative extent. Positive prediction admission remains original.
No module is patched globally; no runtime reader/contract is relaxed by import.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
PROPOSAL = 'supplemental-zero-gt-input-extension-review-v1'
TASK = 'sahi-original-c-supplemental-20261018-001'
MANIFEST_SHA = '1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4'
TARGETS = (
    ('9999985_00000_d_0000020.txt', 12, '620332c35ca9ce25489b90f61cd656bbcc26181d7f07aec513f63a8b12932a1e'),
    ('9999999_00590_d_0000267.txt', 89, 'ceb68c4451e61767b24758e1f2bc9a4c5f6167bec0729d492de83b8a7246efe0'),
)


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def ref(path):
    p = Path(path); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def exact(r):
    require(ref(r['path']) == r, 'reference SHA/size/path conflict: ' + r['path'])
    return json.loads(Path(r['path']).read_bytes())


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def fingerprint(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def isolated_adapter(primary, evaluation):
    """Executable proposed delta only, leaving all original modules untouched."""
    from dataclasses import replace
    from sparse_rtdetr.data_protocol.evaluation import PrimaryGroundTruth
    ns = dict(vars(primary))
    for name, value in list(ns.items()):
        if inspect.isfunction(value) and value.__module__ == primary.__name__:
            f = types.FunctionType(value.__code__, ns, value.__name__, value.__defaults__, value.__closure__)
            f.__kwdefaults__ = value.__kwdefaults__
            ns[name] = f

    def gt_xywh(box, path):
        require(type(box) is tuple and len(box) == 4, path + ': exact four-tuple required')
        x1, y1, x2, y2 = [primary._real(v, path) for v in box]
        require(x2 >= x1 and y2 >= y1, path + ': negative GT extent')
        return x1, y1, x2 - x1, y2 - y1

    def coordinate(box, path):
        # 'box' is the unchanged helper's coordinate conversion. Predictions
        # have already passed the original strict validator before reaching it.
        if path == 'box':
            return gt_xywh(box, path)
        return primary._box_xywh(box, path)

    def validate(value):
        # All immutable input/detection rules remain exactly the original ones.
        require(type(value.ground_truth) is tuple, 'exact GT tuple required')
        primary._validate_input(replace(value, ground_truth=()))
        ids = {r.image_id for r in value.images}; seen = set()
        for i, g in enumerate(value.ground_truth):
            require(type(g) is PrimaryGroundTruth, 'exact GT type')
            require(type(g.annotation_id) is str and g.annotation_id and g.annotation_id not in seen, 'GT annotation identity')
            seen.add(g.annotation_id)
            require(type(g.image_id) is str and g.image_id in ids, 'GT member identity')
            require(type(g.category_id) is int and g.category_id in range(12), 'GT category')
            require(type(g.ignore_region) is bool and type(g.ignored) is bool and g.ignore_region is (g.category_id == 0), 'GT flags')
            _, _, w, h = gt_xywh(g.bbox_xyxy, f'ground_truth/{i}')
            area = primary._real(g.area, 'GT area')
            require(area >= 0 and area == w * h, 'GT original area/extent mismatch')
        return value

    ns['_box_xywh'] = coordinate
    ns['_validate_input'] = validate
    adapter = types.SimpleNamespace(**ns)
    bridge_ns = {**evaluation.bridge_inputs.__globals__, 'primary': adapter}
    bridge_fn = types.FunctionType(evaluation.bridge_inputs.__code__, bridge_ns)

    def bridge(rows, parsed, predictions):
        validate(evaluation.primary_inputs(rows, parsed, predictions))
        return bridge_fn(rows, parsed, predictions)

    return adapter, bridge


def official_raster(image, objects):
    """Literal CPU translation of the pinned integer-region MATLAB path.

    This is not execution under MATLAB/Octave. Tested real rows are integer,
    as are all deterministic fixtures. Fractional region indexing is out of scope.
    """
    import numpy as np
    mask = np.zeros((image.height, image.width), dtype=np.float64)
    for g in objects:
        if g.category_id != 0:
            continue
        x1, y1, x2, y2 = g.bbox_xyxy
        xywh = [x1, y1, x2 - x1, y2 - y1]
        require(all(float(v).is_integer() for v in xywh), 'reference fixture region must have integer coordinates')
        x, y, w, h = [max(1, int(v)) for v in xywh]
        mask[y - 1:min(image.height, y + h), x - 1:min(image.width, x + w)] = 1
    return mask, np.cumsum(np.cumsum(mask, axis=0), axis=1)


def official_fraction(integral, image, box):
    def rnd(x):
        return math.floor(x + .5) if x >= 0 else math.ceil(x - .5)
    x1, y1, x2, y2 = box
    x, y, w, h = [max(1, rnd(v)) for v in (x1, y1, x2 - x1, y2 - y1)]
    x = max(1, min(image.width, x)); y = max(1, min(image.height, y))
    xe = min(image.width, x + w); ye = max(1, min(image.height, y + h))
    return float(integral[y - 1, x - 1] + integral[ye - 1, xe - 1] - integral[y - 1, xe - 1] - integral[ye - 1, x - 1]) / (w * h)


def official_matches(gt, detections, threshold):
    """Pinned compOas/evalRes loop, independently translated for count tests."""
    ordered = sorted(enumerate(gt), key=lambda v: (v[1].ignored, v[0]))
    states = [-1 if g.ignored else 0 for _, g in ordered]
    ds = []
    for d in detections:
        best, bi, bm = threshold, -1, 0
        dx, dy, dx2, dy2 = d.bbox_xyxy
        for j, (_, g) in enumerate(ordered):
            if states[j] == 1:
                continue
            if bm != 0 and states[j] == -1:
                break
            gx, gy, gx2, gy2 = g.bbox_xyxy
            w, h = min(dx2, gx2) - max(dx, gx), min(dy2, gy2) - max(dy, gy)
            overlap = 0.
            if w > 0 and h > 0:
                inter = w * h
                denom = (dx2 - dx) * (dy2 - dy)
                if states[j] != -1:
                    denom += (gx2 - gx) * (gy2 - gy) - inter
                overlap = inter / denom
            if overlap < best:
                continue
            best, bi, bm = overlap, j, -1 if states[j] == -1 else 1
        ds.append(bm)
        if bm == 1:
            states[bi] = 1
    original = [0] * len(gt)
    for j, (i, _) in enumerate(ordered):
        original[i] = states[j]
    return tuple(original), tuple(ds)


def coco_counts(dataset, predictions):
    """Actual installed COCO matching only; no accumulate/AP/summarize."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = copy.deepcopy(dataset); coco.createIndex()
        if predictions:
            det = coco.loadRes(copy.deepcopy(predictions))
        else:
            det = COCO(); det.dataset = {k: copy.deepcopy(dataset[k]) for k in ('info', 'images', 'categories')}
            det.dataset['annotations'] = []; det.createIndex()
        ev = COCOeval(coco, det, 'bbox'); ev.params.imgIds = sorted(coco.imgs)
        ev.params.catIds = list(range(1, 11)); ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
    records = []
    labels = {tuple(v): label for v, label in zip(ev.params.areaRng, ev.params.areaRngLbl)}
    for v in ev.evalImgs:
        if v is None:
            continue
        records.append(dict(image_id=v['image_id'], category_id=v['category_id'], area=labels[tuple(v['aRng'])],
            GT_ids=[int(x) for x in v['gtIds']], GT_ignore=[int(x) for x in v['gtIgnore']],
            matching_denominator=int(sum(x == 0 for x in v['gtIgnore'])),
            detection_count=len(v['dtIds']), TP_IoU50=int(sum((v['dtMatches'][0] > 0) & ~v['dtIgnore'][0])),
            FP_IoU50=int(sum((v['dtMatches'][0] == 0) & ~v['dtIgnore'][0])),
            ignored_detections_IoU50=int(sum(v['dtIgnore'][0]))))
    return records


def trace_case(name, rows, parsed, predictions, primary, adapter, bridge, binding):
    from sparse_rtdetr.r0.evaluation import primary_inputs
    value = primary_inputs(rows, parsed, predictions)
    try:
        primary.evaluate_primary_v1(value, binding)
    except primary.PrimaryEvaluatorContractError as error:
        original_status = dict(status='REJECTED', reason=str(error))
    else:
        original_status = dict(status='ACCEPTED')
    adapter._validate_input(value)
    import numpy as np
    stages = []; gt_kept = []; det_kept = []
    for image in value.images:
        gts = [g for g in value.ground_truth if g.image_id == image.image_id]
        mask, integral = official_raster(image, gts)
        actual = adapter._build_integral(image, [g for g in gts if g.category_id == 0])
        require(np.array_equal(actual, integral), 'raster differs from pinned reference')
        gs = []; ds = []
        for g in gts:
            if g.category_id == 0:
                gs.append(dict(id=g.annotation_id, category=0, kept=False, reason='category0 supplies raster only; not scored denominator', raw_area=g.area))
                continue
            f = official_fraction(integral, image, g.bbox_xyxy)
            require(f == adapter._ignored_fraction(actual, image, g.bbox_xyxy), 'GT raw-ignore fraction differs')
            keep = g.category_id in range(1, 11) and f < .5
            if keep:
                gt_kept.append(g)
            gs.append(dict(id=g.annotation_id, category=g.category_id, kept=keep, ignore_fraction=f, raw_area=g.area,
                reason='raw-ignore fraction >= .5' if f >= .5 else 'not scored class' if g.category_id == 11 else 'retained; zero area is not an exclusion rule'))
        for d in [d for d in value.detections if d.image_id == image.image_id]:
            f = official_fraction(integral, image, d.bbox_xyxy)
            require(f == adapter._ignored_fraction(actual, image, d.bbox_xyxy), 'detection raw-ignore fraction differs')
            if f < .5:
                det_kept.append(d)
            ds.append(dict(category=d.category_id, box=list(d.bbox_xyxy), kept=f < .5, ignore_fraction=f,
                reason='retained' if f < .5 else 'raw-ignore fraction >= .5'))
        stages.append(dict(image_id=image.image_id, raster_pixels=int(mask.sum()), GT=gs, detections=ds))
    # Invoke the actual primary algorithm in the isolated proposal namespace.
    # Only count evidence is retained; unit-case AP is not a scientific result.
    result = adapter.evaluate_primary_v1(value, binding)
    categories = []
    for category in range(1, 11):
        counts = [0, 0, 0, 0]
        for image in value.images:
            gs = [g for g in gt_kept if g.image_id == image.image_id and g.category_id == category]
            ds = [d for d in det_kept if d.image_id == image.image_id][:500]
            ds = [d for d in ds if d.category_id == category]
            _, states = official_matches(gs, ds, .5)
            delta = [len(gs), states.count(1), states.count(0), states.count(-1)]
            counts = [a + b for a, b in zip(counts, delta)]
        require(tuple(counts) == result.match_counts_by_class_iou_max_dets[category - 1][0][3], 'primary/reference match counts differ')
        if counts != [0, 0, 0, 0]:
            categories.append(dict(category=category, denominator=counts[0], TP=counts[1], FP=counts[2], ignored=counts[3]))
    ds, ps, spatial = bridge(rows, parsed, predictions)
    require(result.evaluated_ground_truth_count == spatial['GT_after'] and result.evaluated_detection_count == spatial['detections_after'], 'dual evaluator prefilter count difference')
    return dict(name=name, original_primary=original_status, proposed_protocol=PROPOSAL, original_rows=len(value.ground_truth),
        stages=stages, primary_match_counts=categories, bridge_filter_counts=spatial,
        bridge_GT=[{k: g[k] for k in ('id', 'category_id', 'bbox', 'area', 'iscrowd', 'raw_annotation_id')} for g in ds['annotations']],
        actual_COCO_matching=coco_counts(ds, ps), AP_retained=False, model_predictions_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True); parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--parent-contract', required=True); parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--official-root', required=True); parser.add_argument('--output', required=True)
    a = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU visibility required')
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA was initialized')
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise RuntimeError('review forbids model/CUDA/backward/AP accumulation')
    torch.cuda._lazy_init = forbidden; torch.nn.Module.__init__ = forbidden; torch.autograd.backward = forbidden
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.r0 import evaluation
    from pycocotools import cocoeval
    from sparse_rtdetr.data_protocol.evaluation import PrimaryGroundTruth
    from sparse_rtdetr.localization.data import parse_raw, read_bound
    cocoeval.COCOeval.accumulate = forbidden; cocoeval.COCOeval.summarize = forbidden
    out = Path(a.output); out.mkdir(exist_ok=False); start = time.monotonic()
    def save(name, obj):
        with (out/name).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)
    sources=[]
    manifest=json.loads((ROOT/'manifests/visdrone_det_toolkit_005445.json').read_bytes())
    for row in manifest['inventory']['rows']:
        if row['relative_path'] not in ('evalDET.m', 'utils/saveAnnoRes.m', 'utils/dropObjectsInIgr.m', 'utils/createIntImg.m', 'utils/evalRes.m', 'utils/compOas.m', 'utils/calcAccuracy.m', 'utils/VOCap.m'):
            continue
        p=Path(a.official_root)/row['relative_path'];r=ref(p)
        require(r['sha256']==row['sha256'] and r['bytes']==row['size_bytes'], 'official source identity')
        sources.append(r)
    cr=ref(a.context);require(cr['sha256']==a.context_sha256,'CPU context identity')
    context=exact(cr);spec=context['evaluation_data']
    require(context['task_id']==TASK and spec['manifest']['sha256']==MANIFEST_SHA,'bounded supplemental materials')
    rows=exact(spec['manifest'])['records']; mapping=exact(exact(spec['path_binding'])['annotation_map'])
    byname={Path(r['annotation_relative_path']).name:r for r in rows}
    adapter, bridge=isolated_adapter(primary,evaluation)
    binding=primary.primary_evaluator_contract_binding(ROOT)
    cases=[];real=[];fullrows=[];fullparsed=[]
    for name, line, sha in TARGETS:
        r=byname[name];p=Path(spec['annotations'])/name
        require(any(m['restored']==dict(path=str(p),bytes=r['annotation_size_bytes'],sha256=r['annotation_sha256']) for m in mapping),'recovery mapping identity')
        raw=read_bound(p,r['annotation_sha256'],r['annotation_size_bytes'])
        physical=raw.splitlines(keepends=True)[line-1]
        require(hashlib.sha256(physical).hexdigest()==sha,'original physical line identity')
        v=parse_raw(raw,r['stable_image_id'],expected_rows=r['raw_annotation_rows']);obj=v['rows'][line-1]
        require(obj['role']=='nonpositive_area' and obj['physical_line']==line,'target role/line')
        real.append(dict(file=ref(p),physical_line=line,raw_line_sha256=sha,raw_line_utf8=physical.decode('utf-8'),image_id=r['stable_image_id'],width=r['width'],height=r['height']))
        fullrows.append(r);fullparsed.append(v)
        isolated={k:[] for k in ('gt','spatial_ignore','class_ignore','rows')};isolated['rows']=[copy.deepcopy(obj)];isolated['counts']={'nonpositive_area':1}
        x1,y1,x2,y2=obj['box']
        def pred(box):return dict(image_id=r['stable_image_id'],category_id=4,score=.9,bbox_xyxy=box)
        probes=[('empty',[]),('nearby',[pred([x1,y1+2,x2,y1+3])]),('overlap',[pred([x1,y1,x2,y1+1])]),('fraction_boundary',[pred([x1,y1-1,x2,y1+3])])]
        for tag,preds in probes:
            cases.append(trace_case(name+':'+tag,[r],[isolated],preds,primary,adapter,bridge,binding))
        if obj['category']==4:
            add=copy.deepcopy(obj);add.update(id='synthetic-positive-ignore-region',category=0,score=0.,role='spatial_ignore',box=[x1-1,y1-1,x2+5,y1+3],raw_area=(x2-x1+6)*4)
            augmented=copy.deepcopy(isolated);augmented['rows'].append(add);augmented['spatial_ignore'].append(add)
            cases.append(trace_case(name+':synthetic_ignore_overlap',[r],[augmented],[pred([x1,y1,x2,y1+1])],primary,adapter,bridge,binding))
        else:
            add=copy.deepcopy(obj);add.update(id='synthetic-positive-anchor',category=4,score=1.,role='valid_gt',box=[x1,y1,x2,y1+1],raw_area=x2-x1)
            augmented=copy.deepcopy(isolated);augmented['rows'].append(add);augmented['gt'].append(add)
            cases.append(trace_case(name+':positive_anchor_in_ignore',[r],[augmented],[pred([x1,y1,x2,y1+1])],primary,adapter,bridge,binding))
    # Both full source files remain untouched, including all other raw rows.
    cases.append(trace_case('two_original_files:empty',fullrows,fullparsed,[],primary,adapter,bridge,binding))
    nearby=[dict(image_id=r['stable_image_id'],category_id=4,score=.9,bbox_xyxy=[o['box'][0],o['box'][1],o['box'][2],o['box'][1]+1]) for r,v in zip(fullrows,fullparsed) for o in v['rows'] if o['role']=='nonpositive_area']
    cases.append(trace_case('two_original_files:deterministic_overlap',fullrows,fullparsed,nearby,primary,adapter,bridge,binding))
    lookup={c['name']:c for c in cases}
    positive_zero=lookup[TARGETS[0][0]+':overlap']
    require(positive_zero['primary_match_counts']==[dict(category=4,denominator=1,TP=0,FP=1,ignored=0)], 'zero-area positive GT must remain an unmatched denominator')
    ignore_zero=lookup[TARGETS[1][0]+':overlap']
    require(ignore_zero['stages'][0]['raster_pixels']==22 and ignore_zero['bridge_filter_counts']['detections_after']==0, 'zero-height category0 raster must remain 11x2 pixels')
    boundary=lookup[TARGETS[1][0]+':fraction_boundary']['stages'][0]['detections'][0]
    require(boundary['ignore_fraction']==.5 and boundary['kept'] is False, 'inclusive .5 ignore boundary')
    nearby_case=lookup[TARGETS[1][0]+':nearby']['stages'][0]['detections'][0]
    require(nearby_case['ignore_fraction']==0. and nearby_case['kept'] is True, 'nearby prediction must remain')
    zero_coco=[v for v in positive_zero['actual_COCO_matching'] if v['category_id']==4]
    require({v['area']:v['matching_denominator'] for v in zero_coco}==dict(all=1,small=1,medium=0,large=0), 'COCO area0 must count in all/small, ignored in medium/large')
    save('REAL_LINES.json',real);save('CASE_TRACES.json',cases)
    refusals=[]
    from dataclasses import replace
    sample=evaluation.primary_inputs(fullrows,fullparsed,[])
    g=sample.ground_truth[0]
    variants={
        'negative_GT':replace(g,bbox_xyxy=(1.,1.,0.,2.),area=-1.),
        'nonfinite_GT':replace(g,bbox_xyxy=(1.,1.,2.,float('nan')),area=0.),
        'area_conflict_GT':replace(g,bbox_xyxy=(1.,1.,2.,1.),area=1.),
    }
    for name,v in variants.items():
        try:adapter._validate_input(replace(sample,ground_truth=(v,)))
        except (RuntimeError,ValueError):refusals.append(name)
        else:raise RuntimeError('invalid GT admitted: '+name)
    from sparse_rtdetr.data_protocol.evaluation import Detection
    for name,box in [('zero_prediction',(1.,1.,2.,1.)),('negative_prediction',(2.,1.,1.,2.)),('nonfinite_prediction',(1.,1.,2.,float('nan')))]:
        v=replace(sample,detections=(Detection(fullrows[0]['stable_image_id'],4,box,.9),))
        try:adapter._validate_input(v)
        except (RuntimeError,ValueError):refusals.append(name)
        else:raise RuntimeError('invalid prediction admitted: '+name)
    # Development: bind actual original raw inputs, use only fixed synthetic
    # predictions; do not read any detector prediction/cache or recompute AP.
    pr=ref(a.parent_contract);require(pr['sha256']==a.parent_sha256,'parent identity')
    parent=exact(pr);dev=parent['parents'][0]['construction']['data']['development']
    dm=ref(dev['manifest']);require(dm['sha256']=='5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd','development manifest')
    dr=json.loads(Path(dev['manifest']).read_bytes())['records'];require(len(dr)==548,'development count')
    dp=[];synthetic=[]
    for r in dr:
        p=Path(dev['roots']['annotations'])/Path(r['annotation_relative_path']).name
        v=parse_raw(read_bound(p,r['annotation_sha256'],r['annotation_size_bytes']),r['stable_image_id'],expected_rows=r['raw_annotation_rows'])
        require(v['counts'].get('nonpositive_area',0)==0,'unexpected development nonpositive GT')
        dp.append(v)
        synthetic.append(dict(image_id=r['stable_image_id'],category_id=4,score=.9,bbox_xyxy=[1.,1.,2.,2.]))
    original=evaluation.primary_inputs(dr,dp,synthetic);review=adapter._validate_input(original)
    require(primary._validate_input(original)==review,'positive-domain primary input changed')
    primary_sha=fingerprint(primary._input_payload(original))
    old_bridge=evaluation.bridge_inputs(dr,dp,synthetic);new_bridge=bridge(dr,dp,synthetic)
    require(old_bridge==new_bridge,'development bridge inputs changed')
    devproof=dict(status='PASS',members=548,manifest=dm,original_annotation_files_bound=548,raw_rows=sum(len(x['rows']) for x in dp),nonpositive_raw_rows=0,
        primary_input_sha256=primary_sha,bridge_input_sha256=fingerprint(old_bridge),filter_counts=old_bridge[2],
        predictions='one deterministic positive box per image; not detector predictions',budgets=[600,300],old_model_predictions_read=False,original_AP_recomputed=False)
    save('DEVELOPMENT_INPUT_EQUIVALENCE.json',devproof)
    require(not torch.cuda.is_initialized(),'review initialized CUDA')
    for name,line,sha in TARGETS:
        r=byname[name];require(ref(Path(spec['annotations'])/name)['sha256']==r['annotation_sha256'],'source annotation mutated')
    evidence=dict(status='REVIEW_CPU_PASS_PRODUCTION_REMAINS_STOP',proposal=PROPOSAL,
        classification='VERSIONED_EVALUATION_INPUT_PROTOCOL_EXTENSION_REQUIRED',
        recommendation='retain both raw zero-area rows; official GT count/raster/matching; COCO original area0; never exclude/repair',
        official_commit=manifest['commit_oid'],official_source_refs=sources,
        reference_execution='source audit + independent literal CPU translation; MATLAB/Octave unavailable',
        octave=shutil.which('octave'),matlab=shutil.which('matlab'),cases=len(cases),refusals=refusals,
        actual_COCO_matching=True,COCO_AP_accumulation=False,development=devproof,
        evaluated_GTs_in_two_original_files=cases[-1]['bridge_filter_counts'],
        legacy_primary_unchanged=ref(primary.__file__),legacy_bridge_unchanged=ref(evaluation.__file__),
        original_files_unchanged=True,production_adapter_enabled=False,
        GPU_forward_calls=0,model_predictions_read=False,training_updates=0,scientific_groups=0,
        original_independent_qualification='BLOCKED_PENDING_ACCESS_HISTORY',elapsed_seconds=time.monotonic()-start)
    save('REVIEW_CPU_GATE.json',evidence)
    print(json.dumps({k:evidence[k] for k in ('status','classification','cases','refusals','elapsed_seconds','GPU_forward_calls','scientific_groups')}))


if __name__=='__main__':
    main()
