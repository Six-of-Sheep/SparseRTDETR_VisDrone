"""One bounded CPU fixed-pair N geometry diagnosis; original evidence is read-only."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from r18_cache_diag import (LIMITS, require, sha, exact, read, publish, reference,
                            contract as old_contract, preflight, rss, output_bytes)


def load(path, digest):
    require(sha(path) == digest, 'geometry contract SHA differs')
    c = json.loads(Path(path).read_text())
    require(c['schema'] == 'r18-fixed-pair-geometry-execution-v1', 'unknown geometry execution schema')
    policy = read(c['policy'])
    frozen = json.loads((Path(__file__).parents[1]/'configs/research/r18_geometry_v1.json').read_text())
    require(policy == frozen and policy['limits'] == LIMITS, 'frozen geometry policy differs')
    require(c['profile'] in ('REAL_N_548', 'SYNTHETIC_FIXTURE'), 'unknown geometry profile')
    require(set(c['runtime_files']) == {'r18_geometry_diag.py','r18_geometry_core.py'}, 'geometry source inventory differs')
    for name, digest_ in c['runtime_files'].items():
        require('/' not in name and sha(Path(__file__).with_name(name)) == digest_, 'geometry executed source differs')
    base = old_contract(c['base_contract']['path'], c['base_contract']['sha256'])
    exact(c['base_contract'])
    require((base['profile'] == 'REAL_NUD_548') == (c['profile'] == 'REAL_N_548'), 'base profile mismatch')
    require(Path(c['output']).is_absolute() and Path(c['output']).resolve() == Path(c['output']), 'noncanonical output')
    if c['profile'] == 'REAL_N_548':
        require(c['prior_records']['sha256'] == 'e641e4b2f9d759d6a43f68b34f4bccc15782597fbb5e66171b91f4c119dff60c', 'not authenticated prior population')
    return c, base, policy


def admission(c, base, guard=lambda: None):
    rows, proof = preflight(base, guard)
    closed = read(c['prior_closed'])
    require(closed['status'] == 'PASS' and closed['files']['GT_RECORDS.csv.gz'] == c['prior_records'], 'prior record closure differs')
    prior = {}
    small = 0
    with gzip.open(exact(c['prior_records']), 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['arm'] != 'N' or r['B'] != '1' or r['size'] != 'small':
                continue
            small += 1
            value = float(r['best_raw_IoU'])
            if .5 <= value < .75:
                key = (r['image_id'], r['GT_id'])
                require(key not in prior, 'duplicate prior selected GT')
                prior[key] = (int(r['best_query']), value)
    if c['profile'] == 'REAL_N_548':
        require(small == 26441 and len(prior) == 10503, 'authenticated population count differs')
    require(bool(prior), 'empty diagnostic population')
    guard()
    return rows, proof, prior, small


def diagnose(c, base, policy, rows, prior, small, out, guard):
    import numpy as np
    from r18_cache_worker import native
    from r18_cache_core import overlap
    from r18_geometry_core import analyze, aggregate
    torch, parse, _, _, primary, ImageRow, GTRow, _, _ = native(base)
    source = read(base['arms']['N']['infer_result'])
    records, seen, seen_small = [], [], 0
    observed = set()
    for batchno, ref in enumerate(source['raw_batches']):
        guard()
        cache = torch.load(exact(ref), map_location='cpu', weights_only=True)
        batch = rows[batchno*4:batchno*4+4]
        require(set(cache) == {'ids', 'logits', 'boxes'}, 'N raw cache stage fields differ')
        require(cache['ids'] == [r['stable_image_id'] for r in batch], 'raw IDs/order differ')
        boxes, logits = cache['boxes'], cache['logits']
        require(boxes.shape == (len(batch),300,4) and logits.shape == (len(batch),300,10), 'truncated query cache')
        require(boxes.dtype == logits.dtype == torch.float32 and boxes.device.type == logits.device.type == 'cpu', 'not native CPU FP32')
        require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(logits).all()) and bool((boxes[:,:,2:] > 0).all()), 'invalid raw prediction')
        for i, row in enumerate(batch):
            ident = row['stable_image_id']; seen.append(ident)
            raw = exact(dict(path=str(Path(base['annotation_root'])/Path(row['annotation_relative_path']).name),
                             bytes=row['annotation_size_bytes'],sha256=row['annotation_sha256'])).read_bytes()
            parsed = parse(raw, ident, expected_rows=row['raw_annotation_rows'])
            image = ImageRow(ident, int(row['width']), int(row['height']))
            ignored = [GTRow(g['id'],ident,0,tuple(g['box']),float(g['raw_area']),True,False) for g in parsed['rows'] if g['category'] == 0]
            integral = primary._build_integral(image, ignored)
            eligible = [g for g in parsed['gt'] if 0 < g['raw_area'] <= 1024 and
                        primary._ignored_fraction(integral,image,tuple(g['box'])) < .5]
            seen_small += len(eligible)
            xy = torch.cat((boxes[i,:,:2]-boxes[i,:,2:]/2, boxes[i,:,:2]+boxes[i,:,2:]/2), -1)
            xy *= torch.tensor([row['width'],row['height']]*2, dtype=torch.float32)
            xy = xy.numpy()
            quality = overlap(xy, [g['box'] for g in eligible]).T
            for j, g in enumerate(eligible):
                if not .5 <= quality[j].max() < .75:
                    continue
                key = (ident,g['id'])
                require(key in prior and key not in observed, 'population identity mismatch/duplicate')
                value = analyze(xy,g['box'])
                require(value['query'] == prior[key][0] and value['metrics']['original_IoU'] == prior[key][1], 'fixed original pair/IoU differs')
                area = float(g['raw_area'])
                bin_ = '(0,64]' if area <= 64 else '(64,256]' if area <= 256 else '(256,576]' if area <= 576 else '(576,1024]'
                value.update(image_id=ident,GT_id=g['id'],category=str(g['category']),sequence=row['sequence_key'],area_bin=bin_,raw_area=area)
                records.append(value); observed.add(key)
            guard()
    require(seen == [r['stable_image_id'] for r in rows] and observed == set(prior) and seen_small == small, 'final image/GT population coverage differs')
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    summary = aggregate(records, range(1,11), sorted({r['sequence_key'] for r in rows}))
    summary.update(schema='r18-fixed-pair-geometry-result-v1',synthetic=c['profile'] != 'REAL_N_548',
        runtime_commit=c['runtime_commit'],policy=policy,images=len(seen),selected_images=len({r['image_id'] for r in records}),
        small_B_denominator=small,selected_GT=len(records),raw_batches=len(source['raw_batches']),prior_population_exact=True,
        GT_assisted_pairing=True,actual_detection_matching=False,AP_AR_calls=0,GPU_calls=0,model_forward_calls=0,training_updates=0,
        checkpoint_loaded=False,native_matcher_calls=0,source_inputs=dict(base_contract=c['base_contract'],prior_closed=c['prior_closed'],prior_records=c['prior_records']))
    publish(out/'SUMMARY.json',summary)
    with gzip.open(out/'PAIRS.csv.gz','wt',encoding='utf-8',newline='') as f:
        fields=['image_id','GT_id','category','sequence','raw_area','area_bin','query','exact_ties','group','prediction','truth',*records[0]['metrics']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({k:(json.dumps(r[k],separators=(',',':')) if k in ('prediction','truth') else r[k]) for k in fields if k not in r['metrics']} | r['metrics'])
    return dict(status='PASS',selected_GT=len(records),images=len(seen),small_B_denominator=small,synthetic=summary['synthetic'],
                GPU_calls=0,model_forward_calls=0,training_updates=0,AP_AR_calls=0,native_matcher_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','execute','_worker']);p.add_argument('--contract',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();c,base,policy=load(a.contract,a.sha256);out=Path(c['output'])
    if a.mode == 'check':
        require(not out.exists(),'geometry output already consumed');admission(c,base)
        print(json.dumps(dict(status='CHECK_PASS',output_created=False)));return
    if a.mode == '_worker':
        require(os.environ.get('R18_GEOM_PARENT') == str(os.getppid()) and (out/'CONSUMED.json').is_file(),'worker has no supervisor')
        started=float(os.environ['R18_GEOM_STARTED'])
        def guard():
            require(time.monotonic()-started < LIMITS['wall_seconds'],'wall budget exceeded')
            require(rss(os.getpid()) <= LIMITS['rss_bytes'],'RSS budget exceeded')
            require(output_bytes(out.parent) <= LIMITS['output_bytes'],'storage budget exceeded')
        try:
            rows,proof,prior,small=admission(c,base,guard)
            publish(out/'ADMISSION.json',dict(status='PASS',proof=proof,prior_selected=len(prior),small_B=small))
            result=diagnose(c,base,policy,rows,prior,small,out,guard)
            publish(out/'RESULT.json',result);guard()
        except BaseException as e:
            publish(out/'STOP.json',dict(status='STOP_NO_RETRY',error=repr(e)));raise
        return
    started=time.monotonic();out.mkdir(parents=False,exist_ok=False)
    publish(out/'CONSUMED.json',dict(contract_sha256=a.sha256,no_retry=True,started=started))
    cpus=sorted(os.sched_getaffinity(0))[:2];require(len(cpus)==2,'two CPUs unavailable');os.sched_setaffinity(0,cpus)
    env=dict(os.environ,R18_GEOM_PARENT=str(os.getpid()),R18_GEOM_STARTED=str(started))
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'_worker','--contract',a.contract,'--sha256',a.sha256]
    violation=None;peak=0
    with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
        child=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr)
        while child.poll() is None:
            peak=max(peak,rss(child.pid)+rss(os.getpid()))
            if time.monotonic()-started >= LIMITS['wall_seconds']:violation='wall budget exceeded'
            if peak > LIMITS['rss_bytes']:violation='RSS budget exceeded'
            if output_bytes(out.parent) > LIMITS['output_bytes']:violation='storage budget exceeded'
            if violation:
                child.terminate()
                try:child.wait(5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(.1)
    ok=child.returncode==0 and violation is None and (out/'RESULT.json').is_file() and not (out/'STOP.json').exists()
    if not ok and not (out/'STOP.json').exists():publish(out/'STOP.json',dict(status='STOP_NO_RETRY',error=violation or 'worker failed'))
    publish(out/'EXIT.json',dict(status='PASS' if ok else 'STOP_NO_RETRY',returncode=child.returncode,error=violation,
        elapsed_seconds=time.monotonic()-started,peak_total_rss_bytes=peak,cpu_affinity=cpus,argv=argv))
    if ok:
        files={p.name:reference(p) for p in out.iterdir() if p.is_file()}
        publish(out/'CLOSED.json',dict(status='PASS',result=files['RESULT.json'],exit=files['EXIT.json'],files=files))
    print(json.dumps(dict(status='PASS' if ok else 'STOP_NO_RETRY',output=str(out))))
    if not ok:raise SystemExit(2)


if __name__ == '__main__':main()
