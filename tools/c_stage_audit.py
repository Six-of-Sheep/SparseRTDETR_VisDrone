"""One-shot CPU cache audit. Immutable outputs; no scientific retry or model API."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from c_stage_core import STAGES, BUCKETS, availability, class_cap

PARENT_SHA = 'b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28'
TASK = 'sahi-c-stage-audit-20261018-001'


def encoded(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def publish(p, x):
    with p.open('xb') as f: f.write(encoded(x))
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def geometry(raw, part, j, torch):
    maps = [dict() for _ in part]
    for view in raw['views']:
        v = view['view_id']
        for k, row in enumerate(part):
            t = j.tile_windows(row['height'], row['width'])[v - 1] if v else None
            w, h = (t.width, t.height) if t else (row['width'], row['height'])
            b = view['pred_boxes'][k]
            xy = torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1)
            xy *= torch.tensor([w, h] * 2, dtype=torch.float32)
            mapped = j.map_crop_xyxy(xy.tolist(), t) if t else xy.tolist()
            maps[k].update({(v, q): tuple(box) for q, box in enumerate(mapped)})
    return maps


def run(parent, out):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    if sha(parent) != PARENT_SHA: raise ValueError('parent contract SHA differs')
    c = json.loads(parent.read_bytes())
    source = Path(c['source_root'])
    # Authenticate every external dependency before import. No new old-source edits.
    for item in c['source_files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or sha(source / rel) != item['sha256']:
            raise ValueError('external frozen source mismatch')
    sys.path.insert(0, str(source))
    import softnms_job as j
    torch = j.cpu_only()
    c = j.load_contract(parent)
    specs, rows, roots = j.context(c)
    from slice_core import classwise_nms
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2
    def forbidden(*a, **kw): raise RuntimeError('stage audit forbids scoring')
    ev.score_budget = ev.bridge_metrics = primary.evaluate_primary_v1 = forbidden
    parsed = j.parsed_gt(rows, {'roots': roots})
    j.resources(out, j.read(j.PLAN))
    publish(out / 'ADMISSION.json', {'parent_contract_sha256': PARENT_SHA,
        'external_source': str(source), 'source_files_verified': len(c['source_files']),
        'images': len(rows), 'seeds': [s['seed'] for s in specs],
        'own_source': {n: sha(Path(__file__).parent / n) for n in ('c_stage_audit.py','c_stage_core.py')},
        'GPU': 0, 'model_constructions': 0, 'AP_calls': 0})
    proofs, summaries = [], []
    for spec in specs:
        seed = spec['seed']; expected = {}
        for budget in (600, 300):
            preds = json.loads(j.exact(spec['evaluation']['predictions']['C'][str(budget)], contents=True))
            proofs.append({'seed': seed, **j.evaluation_input_proof(spec, rows, parsed, preds, 'C', budget)})
            expected[budget] = j.group(preds, spec['ids'], budget)
        totals = {b: {t: Counter({k: 0 for k in BUCKETS}) for t in ('0.5','0.75')} for b in (600,300)}
        seqs = {b: defaultdict(lambda: {t: Counter({k: 0 for k in BUCKETS}) for t in ('0.5','0.75')}) for b in (600,300)}
        classes = {b: defaultdict(lambda: {t: Counter({k: 0 for k in BUCKETS}) for t in ('0.5','0.75')}) for b in (600,300)}
        denominator = 0; image_counts = []; count_rows = []
        with (out / f'seed{seed}-gt.jsonl').open('x', encoding='utf-8') as detail:
            for batch, ref in enumerate(spec['full']['raw_view_refs']):
                j.resources(out, j.read(j.PLAN))
                raw = torch.load(io.BytesIO(j.exact(ref, contents=True)), map_location='cpu', weights_only=True)
                part = rows[4 * batch: 4 * batch + 4]
                pools = j.decode_raw(raw, part, spec['identity'], seed)
                maps = geometry(raw, part, j, torch)
                for local, (row, pool, geomap) in enumerate(zip(part, pools, maps)):
                    image = row['stable_image_id']; ann = parsed[4 * batch + local]
                    j.require(len(geomap) == 1500, 'raw geometry incomplete')
                    j.require(all(tuple(p.box_xyxy) == geomap[(p.view_id,p.query_id)] for p in pool), 'raw mapping mismatch')
                    kept = classwise_nms(pool)
                    for budget in (600,300):
                        j.require([j.prediction(image,p) for p in kept[:budget]] == expected[budget][image],
                                  'original C exact reconstruction differs')
                    # Original raster primitive, applied after truncation, with no refill.
                    im = PrimaryImageV2(image, row['width'], row['height'])
                    integral = primary._build_integral(im, [g for g in ev.truth(row,ann) if g.category_id == 0])
                    dataset, _, _ = ev.bridge_inputs([row], [ann], [])
                    small = [g for g in dataset['annotations'] if g['iscrowd'] == 0 and 0 < g['area'] <= 1024]
                    denominator += len(small)
                    # Stable image IDs use original filename; verify sequence prefix, never infer a random split.
                    sequence = Path(row['annotation_relative_path']).stem.split('_')[0]
                    image_counts.append({'image_id':image, 'sequence':sequence, 'small_GT':len(small)})
                    for budget in (600,300):
                        top = kept[:budget]
                        filtered = [p for p in top if primary._ignored_fraction(integral,im,p.box_xyxy) < .5]
                        cap = class_cap(filtered)
                        _, canonical_filtered, counts = ev.bridge_inputs([row],[ann],expected[budget][image])
                        converted = [{'image_id':1,'category_id':p.category_id,'score':p.score,
                            'bbox':[p.box_xyxy[0],p.box_xyxy[1],p.box_xyxy[2]-p.box_xyxy[0],p.box_xyxy[3]-p.box_xyxy[1]]} for p in filtered]
                        j.require(converted == canonical_filtered, 'raw-ignore differs from original bridge')
                        count_rows.append({'image_id':image,'budget':budget,'pool':len(pool),'nms':len(kept),
                                           'top':len(top),'filtered':len(filtered),'class100':len(cap),**counts})
                        stage_candidates = (pool, kept, top, filtered, cap)
                        stage_by_class = {cat: {STAGES[0]:list(geomap.values()), **{
                            stage:[p.box_xyxy for p in cand if p.category_id == cat]
                            for stage,cand in zip(STAGES[1:],stage_candidates)}} for cat in range(1,11)}
                        for gt in small:
                            x,y,w,h = gt['bbox']; cat = gt['category_id']
                            result = availability([x,y,x+w,y+h], stage_by_class[cat])
                            for t, bucket in result['first_unavailable'].items():
                                totals[budget][t][bucket] += 1
                                seqs[budget][sequence][t][bucket] += 1
                                classes[budget][cat][t][bucket] += 1
                            detail.write(json.dumps({'seed':seed,'image_id':image,'sequence':sequence,
                                'raw_annotation_id':gt['raw_annotation_id'],'category_id':cat,
                                'area':gt['area'],'budget':budget,**result},allow_nan=False) + '\n')
                print(json.dumps({'seed':seed,'batch':batch+1,'batches':len(spec['full']['raw_view_refs'])}),flush=True)
        j.require(len(image_counts) == 548 and all(sum(v.values()) == denominator for ts in totals.values() for v in ts.values()), 'partition incomplete')
        sequence_equal = {}
        for b, sequences in seqs.items():
            sequence_equal[b] = {}
            for t in ('0.5','0.75'):
                valid = [v[t] for v in sequences.values() if sum(v[t].values()) > 0]
                sequence_equal[b][t] = {k: sum(v[k]/sum(v.values()) for v in valid)/len(valid) for k in BUCKETS}
        summary = {'seed':seed,'images':548,'small_GT':denominator,'totals':totals,
            'sequence_count_with_small_GT':len(seqs[600]),'sequence_equal_fraction':sequence_equal,
            'by_sequence':dict(seqs),'by_class':dict(classes),'image_counts':image_counts,
            'units':'counts or fractions, NOT AP/recall/causal attribution','raw_batches':len(spec['full']['raw_view_refs'])}
        publish(out / f'seed{seed}-SUMMARY.json',summary)
        publish(out / f'seed{seed}-COUNTS.json',count_rows)
        summaries.append({k:v for k,v in summary.items() if k not in ('by_sequence','by_class','image_counts')})
    publish(out / 'INPUT_PROOFS.json',proofs)
    j.require(not torch.cuda.is_initialized(), 'CUDA initialized')
    return {'status':'COMPLETE','task':TASK,'seeds':summaries,'original_C_exact':True,
            'AP_calls':0,'GPU_calls':0,'model_calls':0,'automatic_next':False}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run','fixture'))
    p.add_argument('--parent',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inject-failure',action='store_true')
    args = p.parse_args(argv)
    if args.mode == 'run' and (args.parent is None or args.inject_failure): p.error('invalid production options')
    out = args.output.resolve()
    if args.mode == 'run' and TASK not in out.parts: p.error('production output is outside task directory')
    out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    publish(out/'START.json',{'task':TASK,'mode':args.mode,'pid':os.getpid(),'started_unix':time.time()})
    try:
        if args.mode == 'fixture':
            if args.inject_failure: raise RuntimeError('expected fixture failure')
            fixture = availability([0,0,10,10],{s:[[0,0,10,10]] for s in STAGES})
            result = {'status':'COMPLETE','synthetic':True,'result':fixture}
        else: result = run(args.parent,out)
        result['elapsed_seconds'] = time.monotonic() - start
        ref = publish(out/'RESULT.json',result)
        publish(out/'CLOSED.json',{'result':ref,'returncode':0})
        print(json.dumps({'status':'COMPLETE','result':ref}),flush=True)
        return 0
    except Exception:
        publish(out/'STOP.json',{'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})
        traceback.print_exc()
        return 2


if __name__ == '__main__': sys.exit(main())
