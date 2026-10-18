"""CPU-only authenticated reuse and real evaluator-input proof. Never scores AP."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import sixth_view_runtime as rt
from sixth_view_runtime import ROOT,require,reference,read,publish,digest


def exact(ref):
    require(reference(ref['path'])==ref,'reuse SHA/size/path changed: '+ref['path'])
    return read(ref['path'])


def validate_annotations(rows,parsed,ids):
    require(type(rows) is list and len(rows)==len(ids)==548 and len(set(ids))==548,'548 distinct members required')
    require([r['stable_image_id'] for r in rows]==ids and all(r['split']=='val' for r in rows),'GT member order/role differs')
    require(type(parsed) is list and len(parsed)==548,'parsed annotations required, not data configuration')
    from collections import Counter
    from sparse_rtdetr.data_protocol.schema import stable_annotation_id
    seen=set()
    for row,raw in zip(rows,parsed):
        require(type(raw) is dict and set(raw)=={'gt','spatial_ignore','class_ignore','rows','counts'},'parsed GT field structure')
        require(len(raw['rows'])==row['raw_annotation_rows'],'physical GT count')
        for obj in raw['rows']:
            require(obj['id']==stable_annotation_id(row['stable_image_id'],obj['physical_line'],obj['raw_line_sha256']),'GT/image identity mismatch')
            require(obj['id'] not in seen,'duplicate GT identity');seen.add(obj['id'])
            require(type(obj['category']) is int and type(obj['score']) is int and obj['score'] in (0,1),'GT category/score type')
            box=obj['box'];require(len(box)==4 and all(math.isfinite(x) for x in box) and box[2]>box[0] and box[3]>box[1],'GT geometry')
        require(dict(Counter(o['role'] for o in raw['rows']))==raw['counts'],'raw role counts')
        for key,role in (('gt','valid_gt'),('spatial_ignore','spatial_ignore'),('class_ignore','class_ignore')):
            require(raw[key]==[o for o in raw['rows'] if o['role']==role],'raw-ignore partitions differ')
    return {'images':548,'ordered_members_sha256':digest(ids),'rows_sha256':digest(rows),'parsed_gt_sha256':digest(parsed),
            'annotation_manifest_sha256':digest([{'image_id':r['stable_image_id'],'sha256':r['annotation_sha256'],'bytes':r['annotation_size_bytes']} for r in rows]),
            'physical_annotations':len(seen)}


def inputs(rows,parsed,predictions,budget):
    """Build the actual two evaluator objects, never call either AP evaluator."""
    from sparse_rtdetr.r0.evaluation import primary_inputs,bridge_inputs
    from sparse_rtdetr.baseline.primary_evaluator import _input_payload
    from sparse_rtdetr.data_protocol.schema import canonical_json_bytes
    import hashlib
    ids=[r['stable_image_id'] for r in rows]
    require(list(dict.fromkeys(p['image_id'] for p in predictions))==ids,'prediction image member order')
    counts={i:0 for i in ids}
    for p in predictions:
        counts[p['image_id']]+=1
        require(type(p['category_id']) is int and p['category_id'] in range(1,11) and math.isfinite(p['score']) and 0<=p['score']<=1,'prediction class/score')
        b=p['bbox_xyxy'];require(len(b)==4 and all(math.isfinite(x) for x in b) and b[2]>b[0] and b[3]>b[1],'prediction geometry')
    require(all(0<n<=budget for n in counts.values()),'per-image output budget')
    primary=primary_inputs(rows,parsed,predictions);payload=_input_payload(primary)
    dataset,filtered,spatial=bridge_inputs(rows,parsed,predictions)
    require([im.image_id for im in primary.images]==ids and len(dataset['images'])==548,'actual dual input member identity')
    require(len(primary.ground_truth)==sum(len(p['rows']) for p in parsed),'primary raw GT loss')
    require(spatial['GT_before']==sum(o['category'] in range(1,11) for p in parsed for o in p['rows']),'bridge GT before')
    require(spatial['GT_after']==len(dataset['annotations']) and spatial['detections_after']==len(filtered) and spatial['detections_before']==len(predictions),'bridge after count')
    return {'canonical_input_sha256':hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            'counts':spatial,'primary_GT_including_ignore':len(primary.ground_truth),'members':548,
            'bridge_dataset_sha256':digest(dataset),'bridge_filtered_sha256':digest(filtered),'AP_scored':False,
            'raw_ignore':'original bridge_inputs raster; budget first; no refill'}


def compatibility(oldc,manifest):
    old=exact(oldc['source_reference']);new=exact(manifest)
    a={r['path']:r for r in old['files']};b={r['path']:r for r in new['files']};changed=[]
    require(set(a)<=set(b),'old source member removed')
    for name,ref in a.items():
        exact({'path':str(Path(oldc['source_root'])/name),'bytes':ref['bytes'],'sha256':ref['sha256']}) if name.endswith('.json') else require(reference(Path(oldc['source_root'])/name)=={'path':str(Path(oldc['source_root'])/name),'bytes':ref['bytes'],'sha256':ref['sha256']},'old source bytes changed')
        require(reference(ROOT/name)=={'path':str(ROOT/name),'bytes':b[name]['bytes'],'sha256':b[name]['sha256']},'new source bytes changed')
        if a[name]!=b[name]:
            require(name in {'sixth_view_runtime.py','sixth_view_job.py','manifests/experiments.json'} or name.startswith('docs/research/'),'unexpected scientific source change: '+name)
            changed.append({'path':name,'old_sha256':a[name]['sha256'],'new_sha256':b[name]['sha256']})
    additions=sorted(set(b)-set(a))
    require(all(n in {'sixth_view_score_reuse.py','sixth_view_score.py','sixth_view_score_check.py','tests/test_sixth_annotations.py'} or n.startswith('docs/research/') for n in additions),'unexpected source addition')
    return {'old_source':oldc['source_reference'],'new_source':manifest,'changed':changed,'added':additions,
            'unchanged_science':'geometry, decoding, NMS, dual evaluators, raw-ignore, decision, config, GPU path byte-identical',
            'old_checks_run_against_original_source':True}


def old_verify(cpath,out):
    # Independent process importing ONLY the unchanged old release.
    c=read(cpath);sys.path[:0]=[c['source_root'],str(Path(c['source_root'])/'src')]
    # This independent helper imported the new rt at module import time. Remove
    # only that module before invoking the original strict historical verifier.
    sys.modules.pop('sixth_view_runtime',None)
    import sixth_view_runtime as oldrt
    import sixth_view_job as oldjob
    c=oldrt.load(cpath);require(Path(oldrt.__file__).parent==Path(c['source_root']),'wrong old verifier')
    results={m:oldjob.verify_phase(c,0,m) for m in oldrt.PHASES[:6]}
    root=Path(c['job_root']);require((root/'STOP.json').exists() and not (root/'CLOSED.json').exists(),'old STOP changed')
    refs={m:{'result':reference(root/'seed0'/m/'RESULT.json'),'closed':reference(root/'seed0'/m/'CLOSED.json')} for m in results}
    publish(out,{'status':'PASS','parent_contract':reference(cpath),'phase_refs':refs,
        'raw_refs':{v:results['full-'+v]['raw_refs'] for v in ('center','flip')},
        'cost':reference(root/'seed0/COST.json'),'data_admission':reference(root/'seed0/evaluation/DATA_ADMISSION.json'),
        'old_STOP':reference(root/'STOP.json'),'source':c['source_reference']})


def prepare(request,out):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=False);req=read(request)
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU only')
    oldc=exact(req['parent_contract']);diff=compatibility(oldc,req['source_manifest']);publish(out/'SOURCE_COMPATIBILITY.json',diff)
    with (out/'old-verify.log').open('xb') as log:
        p=subprocess.run([sys.executable,'-B',str(ROOT/'sixth_view_score_reuse.py'),'old-verify','--request',req['parent_contract']['path'],'--output',str(out/'OLD_PHASES.json')],stdout=log,stderr=subprocess.STDOUT)
    require(p.returncode==0,'old strict phase/source/archive verification failed')
    evidence=read(out/'OLD_PHASES.json');audit=exact(req['parent_audit'])
    require(audit['scientific_score_groups_completed']==0 and audit['forward_calls']==602,'parent STOP scientific scope')
    old,specs,rows,parsed=rt.parent_context(oldc);old.cpu_only()
    spec=specs[0];info=validate_annotations(rows,parsed,spec['ids']);publish(out/'GT_INPUT.json',info)
    # Four authenticated A/C input proofs anchor annotation semantics to old scores.
    prior=[]
    for arm in ('A','C'):
        for b in (600,300):
            pred=json.loads(old.exact(spec['evaluation']['predictions'][arm][str(b)],contents=True))
            prior.append(old.evaluation_input_proof(spec,rows,parsed,pred,arm,b))
    publish(out/'ORIGINAL_AC_INPUT_PROOFS.json',{'status':'PASS','proofs':prior,'AP_scored':False})
    rawfiles=0
    for label,variant in (('center','C-center'),('flip','C-flip')):
        refs=evidence['raw_refs'][label];require(len(refs)==137,'274 raw coverage')
        for k,ref in enumerate(refs):
            payload=torch.load(io.BytesIO(old.exact(ref,contents=True)),map_location='cpu',weights_only=True)
            expected=dict(schema='r18-sixth-extra-raw-v1',seed=0,mode='full-'+label,variant=variant,view_id=5,
                source_sha256=oldc['source_sha256'],contract_sha256=oldc['contract_sha256'],
                checkpoint_sha256=oldc['parents'][0]['checkpoint']['sha256'],image_ids=spec['ids'][4*k:4*k+4])
            require(all(payload[key]==value for key,value in expected.items()),'old raw identity mismatch')
            require(len(payload['views'])==1 and payload['views'][0]['view_id']==5,'raw view schema')
            v=payload['views'][0]
            require(v['pred_logits'].shape==(4,300,10) and v['pred_boxes'].shape==(4,300,4),'raw tensor shape')
            require(all(v[key].dtype==torch.float32 and torch.isfinite(v[key]).all() for key in ('pred_logits','pred_boxes')) and (v['pred_boxes'][...,2:]>0).all(),'raw finite FP32 geometry')
            rawfiles+=1
    admission=exact(evidence['data_admission']);require(admission['status']=='PASS' and admission['original_C_two_budgets_exact'] and admission['new_view_SHA_and_identity_exact'] and admission['raw_batches']==137,'old original C rebuild proof')
    cost=exact(evidence['cost']);require(cost==audit['cost'] and cost['status']=='PASS' and cost['all_real_views_included'] and all(cost['ratios_over_A'][v]<=6 for v in ('center','flip')),'old cost proof')
    byname={Path(r['path']).name:r for r in audit['prediction_files']};predrefs={};proofs={}
    for variant in ('C-center','C-flip'):
        predrefs[variant]={};proofs[variant]={};preds={}
        for budget in (600,300):
            ref=byname[f'{variant}-top{budget}-predictions.json'];pred=json.loads(old.exact(ref,contents=True));preds[budget]=pred
            predrefs[variant][str(budget)]=ref;proofs[variant][str(budget)]=inputs(rows,parsed,pred,budget)
        g={i:[] for i in spec['ids']}
        for p in preds[600]:g[p['image_id']].append(p)
        require(preds[300]==[p for i in spec['ids'] for p in g[i][:300]],'600/300 prefix changed')
    publish(out/'REAL_INPUT_PROOFS.json',{'status':'PASS','proofs':proofs,'original_proofs':reference(out/'ORIGINAL_AC_INPUT_PROOFS.json'),'GT_input':reference(out/'GT_INPUT.json'),'no_old_AP_recomputed':True})
    require(not torch.cuda.is_initialized(),'CUDA initialized')
    certificate={'status':'PASS','parent_contract':req['parent_contract'],'parent_audit':req['parent_audit'],
        'old_phases':reference(out/'OLD_PHASES.json'),'source_compatibility':reference(out/'SOURCE_COMPATIBILITY.json'),
        'real_inputs':reference(out/'REAL_INPUT_PROOFS.json'),'GT_input':reference(out/'GT_INPUT.json'),
        'prediction_refs':predrefs,'raw_files_SHA_identity_verified':rawfiles,'original_C_exact_rebuild_reused':True,
        'cost':evidence['cost'],'source_manifest':req['source_manifest'],'CUDA_initialized':False,'forward_calls':0,'score_budget_calls':0,
        'elapsed_seconds':time.monotonic()-start}
    publish(out/'REUSE_CERTIFICATE.json',certificate)
    print(json.dumps({'status':'PASS','raw_files':rawfiles,'elapsed_seconds':certificate['elapsed_seconds']}))


def admitted_predictions(c,rows,parsed):
    cert=exact(c['seed0_evaluation_reuse']);require(cert['status']=='PASS' and cert['source_manifest']==c['source_reference'],'reuse new source binding')
    oldc=exact(cert['parent_contract']);require(oldc['parents']==c['parents'] and oldc['parent_contract']==c['parent_contract'],'reuse parent/source lineage')
    for field in ('source_compatibility','old_phases','real_inputs','GT_input','parent_audit','cost'):exact(cert[field])
    require(validate_annotations(rows,parsed,[r['stable_image_id'] for r in rows])==exact(cert['GT_input']),'actual parsed GT differs from real CPU gate')
    return {v:{b:exact(ref) for b,ref in bs.items()} for v,bs in cert['prediction_refs'].items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','old-verify']);p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else old_verify)(a.request,a.output)
