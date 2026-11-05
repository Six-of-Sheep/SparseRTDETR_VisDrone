"""R25 six closed endpoint adapter; frozen cached-error core, zero model forwards."""
from pathlib import Path
import sys, json, os, time, gc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import core as C
import run_channels as R
import aggregate as G

# Same interpreter/pycocotools/bridge logic as the formal scorer; run-A-002 stopped on A1-online APM
# differing by 1.11e-16 (2 ULP at 0.388).  Rounding-level only: a single TP/FP change moves AP by >1e-6.
METRIC_TOLERANCE = 1e-12
TOLERANCE_SOURCE = ('float64 rounding of the certified COCO_bridge summary; exact equality was required for '
                    'historical lists, formal R25 A1-online APM differed by 2 ULP; every cell and its ULP '
                    'count is recorded; any non-rounding difference still stops the run')

def large_classes(e, gt):
    ix=e.params.areaRngLbl.index('large')
    lo,hi=e.params.areaRng[ix]
    values=[]
    for k,cat in enumerate(e.params.catIds):
        p=e.eval['precision'][:,:,k,ix,-1]
        v=p[p>-1]
        n=int(((gt.cat==cat)&(~gt.crowd)&(gt.area>=lo)&(gt.area<=hi)).sum())
        values.append({'category_id':int(cat), 'GT':n, 'APL':float(v.mean()) if len(v) else None})
    mean=np.mean([v['APL'] for v in values if v['APL'] is not None])
    R.require(abs(mean-e.stats[5])<1e-14, 'per-class APL macro differs')
    return values

def process(ctx, item, out):
    start,cpu=time.perf_counter(),time.process_time()
    tag=item['tag']; target=out/f'list-{tag}.json'
    R.require(not target.exists(), 'endpoint already evaluated: '+tag)
    pred=json.loads(ctx.j.exact(item['prediction'], contents=True))
    ids=[r['stable_image_id'] for r in ctx.rows]
    grouped=ctx.j.group(pred, ids, 600)
    R.require(len(pred)==548*600 and set(grouped)==set(ids), 'membership/count differs')
    dataset,det,counts=ctx.ev.bridge_inputs(ctx.rows, ctx.parsed, pred)
    del pred,grouped
    e,values=C.evaluate_certified(dataset,det)
    ref=json.loads(ctx.j.exact(item['metrics'],contents=True))['COCO_bridge']['metrics']
    R.require(set(ref)==set(values), 'metric keys differ')
    differences={k:abs(values[k]-ref[k]) for k in ref}
    ulps={k:(float(differences[k]/np.spacing(abs(ref[k]))) if ref[k]!=0 else (0. if differences[k]==0 else float('inf'))) for k in ref}
    R.require(max(differences.values())<=METRIC_TOLERANCE, f'certified metrics differ: {tag}: {differences}')
    gt,dt=C.gt_table(dataset),C.det_table(det)
    runs={(a,t):C.collect_run(e,a,t,gt.n) for a in (C.AREA_ALL,C.AREA_SMALL) for t in R.THRESHOLDS}
    R.require(C.check_port(e,runs)==0, 'accumulation port differs')
    bf,be=C.analyze_errors(gt,dt,runs,ctx.n_img,classify=True)
    summaries=R.list_summary(ctx,runs,gt)
    small=runs[(C.AREA_SMALL,.5)].gt_nonig[gt.id]
    R.require(np.array_equal(small,(~gt.crowd)&(gt.area>=0)&(gt.area<=1024)), 'small rule differs')
    arrays=dict(gt_id=gt.id,gt_img=gt.img,gt_cat=gt.cat,gt_bbox=gt.bbox,gt_area=gt.area,gt_crowd=gt.crowd,
                small=small,best_filtered=bf,best_evaluated=be)
    for (a,t),rt in runs.items():
        key=f'{C.AREA_NAMES[a]}_{G.TKEY[t]}'
        arrays[f'used_{key}']=rt.gt_used[gt.id]
        arrays[f'usable_{key}']=rt.gt_usable[gt.id]
    R.save_npz(out/f'list-{tag}.npz',**arrays)
    result={'tag':tag,'binding':item,'metrics':values,'metric_differences':differences,'metric_difference_ulps':ulps,
            'metric_tolerance':{'absolute_0_1_units':METRIC_TOLERANCE,'source':TOLERANCE_SOURCE},
            'runs':summaries,'spatial_filter_counts':counts,'large_by_class':large_classes(e,gt),
            'COCOeval_full_calls':1,'oracle_evaluations':len(runs)*len(C.ORACLES),
            'cuda_initialized':ctx.torch.cuda.is_initialized(),'cost':R.cost(start,cpu)}
    R.publish(target,result)
    print(json.dumps({'tag':tag,'score_difference':max(differences.values()),'cost':result['cost']}),flush=True)
    del e,runs,dataset,det,gt,dt,arrays
    gc.collect()

def arrays_for(z, area):
    a={'best_evaluated':z['best_evaluated'],'best_filtered':z['best_filtered']}
    for t in R.THRESHOLDS:
        for name in ('used','usable'):
            a[f'{name}_{t}']=z[f'{name}_{area}_{G.TKEY[t]}']
    return a

def summary(out, manifest):
    allj={x['tag']:json.loads((out/f"list-{x['tag']}.json").read_bytes()) for x in manifest['inputs']}
    zs={k:np.load(out/f'list-{k}.npz') for k in allj}
    first=next(iter(zs.values()))
    for z in zs.values():
        for key in ('gt_id','gt_img','gt_cat','gt_bbox','gt_area','gt_crowd','small'):
            R.require(np.array_equal(z[key],first[key]),'GT alignment mismatch: '+key)
    small=first['small']; bins=np.full(len(small),-1,dtype=np.int64)
    bins[small]=C.min_side_bin(first['gt_bbox'][small,2],first['gt_bbox'][small,3])
    sels={'all_GT':~first['gt_crowd'],'all_small':small}
    sels.update({label:small&(bins==i) for i,label in enumerate(C.BIN_LABELS)})
    profiles={}
    for tag,z in zs.items():
        profiles[tag]={label:G.profile(arrays_for(z,'all' if label=='all_GT' else 'small'),sel) for label,sel in sels.items()}
    pairs={}
    for ep in ('EMA','online'):
        for newer,base in (('A2','A0'),('A2','A1'),('A1','A0')):
            name=f'{ep}:{newer}-{base}'
            pairs[name]={'base':base+'-'+ep,'new':newer+'-'+ep,'tables':{}}
            for label,sel in sels.items():
                area='all' if label=='all_GT' else 'small'
                pairs[name]['tables'][label]=G.pair_table(arrays_for(zs[base+'-'+ep],area),arrays_for(zs[newer+'-'+ep],area),sel)
    for arm in ('A0','A1','A2'):
        name=arm+':EMA-online'
        pairs[name]={'base':arm+'-online','new':arm+'-EMA','tables':{label:G.pair_table(arrays_for(zs[arm+'-online'],'all' if label=='all_GT' else 'small'),arrays_for(zs[arm+'-EMA'],'all' if label=='all_GT' else 'small'),sel) for label,sel in sels.items()}}
    R.publish(out/'RESULT_R25_CHANNELS_002.json',{
        'status':'COMPLETE','formal_scientific_decision':'closed_fixed_config','input_manifest':R.ref_of(out/'INPUT_MANIFEST.json'),
        'definitions':{'pair_columns':'A=base, C=new in reused pair_table; labels explicitly provided per comparison',
                       'oracle':'fixed-match custom small extension; independent nonadditive changes, not a reachable upper bound',
                       'coverage':'same-class existence in filtered/final class100 lists, not raw model pool',
                       'endpoint':'EMA primary; online descriptive; no coarse/refined counterfactual forward'},
        'profiles':profiles,'pairs':pairs,'endpoints':allj,
        'cost':{'COCOeval_full_calls':sum(x['COCOeval_full_calls'] for x in allj.values()),
                'oracle_evaluations':sum(x['oracle_evaluations'] for x in allj.values()),
                'list_wall_seconds':sum(x['cost']['wall_seconds'] for x in allj.values()),
                'list_cpu_seconds':sum(x['cost']['cpu_seconds'] for x in allj.values()),
                'model_forwards':0,'gpu_calls':0}})

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--summary-only',action='store_true',help='reuse the six published endpoint outputs in --out; no scoring')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_bytes()); out=Path(args.out)
    if args.summary_only:
        R.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden')
        R.require(sys.executable==cfg['python'],'wrong interpreter')
        for name,h in cfg['core_sha256'].items():
            R.require(R.sha(Path(__file__).parent/name)==h,'analysis core changed '+name)
        manifest=json.loads((out/'INPUT_MANIFEST.json').read_bytes())
        R.require(len(manifest['inputs'])==6,'expected six endpoints')
        for x in manifest['inputs']:
            j=json.loads((out/f"list-{x['tag']}.json").read_bytes())
            R.require(j['binding']==x and j['tag']==x['tag'],'endpoint output not bound to manifest: '+x['tag'])
            R.require(max(j['metric_differences'].values())<=METRIC_TOLERANCE,'endpoint reproduction outside tolerance')
            R.require(j['cuda_initialized'] is False,'CUDA initialized in endpoint run')
        summary(out,manifest)
        R.publish(out/'CLOSED.json',{'status':'PASS','mode':'summary-only over published endpoint outputs',
                  'summary_code':R.ref_of(__file__),'result':R.ref_of(out/'RESULT_R25_CHANNELS_002.json'),
                  'model_forwards':0,'gpu_calls':0,'score_calls':6,'oracle_evaluations':192})
        return
    R.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden')
    R.require(sys.executable==cfg['python'],'wrong interpreter')
    for name,h in cfg['core_sha256'].items():
        R.require(R.sha(Path(__file__).parent/name)==h,'analysis core changed '+name)
    out.mkdir(parents=True,exist_ok=False)
    root=Path(cfg['formal_root']);closed=json.loads((root/'CLOSED.json').read_bytes())
    R.require(closed['status']=='PASS' and len(closed['stages'])==5 and all(s['status']=='PASS' for s in closed['stages']),'formal incomplete')
    for s in closed['stages']:
        R.require(R.sha(s['result']['path'])==s['result']['sha256'],'formal receipt differs')
    ctx=R.Context(json.loads(Path(cfg['context_config']).read_bytes()))
    digest,gtcounts,_=R.gt_identity(ctx)
    inputs=[]; checks={}
    for arm in ('A0','A1','A2'):
        for kind in ('EMA','online'):
            tag=f'{arm}-{kind}'; receipt=root/'evaluation'/tag/'RESULT.json';m=root/'evaluation'/f'{tag}-A-top600-metrics.json'
            r=json.loads(receipt.read_bytes())
            R.require(r['status']=='PASS' and r['images']==548 and r['epoch']==120 and r['update']==36480 and r['arm']==arm and r['kind']==kind,'endpoint identity differs')
            R.require(r['contract_sha256']==closed['contract_sha256'],'contract differs')
            ck=r['checkpoint']
            if ck['path'] not in checks: checks[ck['path']]=R.ref_of(ck['path'])
            R.require(checks[ck['path']]['sha256']==ck['sha256'],'checkpoint hash differs')
            ref=r['predictions']['600']; ctx.j.exact(ref)
            inputs.append({'tag':tag,'receipt':R.ref_of(receipt),'checkpoint':ck,'weights_sha256':r['weights_sha256'],
                           'prediction':ref,'metrics':R.ref_of(m),'endpoint':kind,'epoch':120,'view':'A','seed':0})
    # Formal evaluation root records all 12 score artifacts; no new top300 evaluation is performed.
    R.require(len(list((root/'evaluation').glob('*-A-top*-metrics.json')))==12,'expected twelve fixed scores')
    manifest={'task':cfg['task'],'task_sha256':cfg['task_sha256'],'config':R.ref_of(args.config),'code':R.ref_of(__file__),
              'source_commit':closed['source_commit'],'core_sha256':cfg['core_sha256'],'closed':R.ref_of(root/'CLOSED.json'),
              'release_manifest':R.ref_of(root.parent/'release-002/MANIFEST.json'),'comparison':closed['comparison'],
              'GT_bridge_annotations_sha256':digest,'GT_counts':gtcounts,'ordered_members_sha256':ctx.j.digest([r['stable_image_id'] for r in ctx.rows]),
              'data_manifest_rows_sha256':ctx.j.digest(ctx.rows),'evaluator':R.evaluator_identity(ctx),'inputs':inputs}
    R.publish(out/'INPUT_MANIFEST.json',manifest)
    for item in inputs: process(ctx,item,out)
    summary(out,manifest)
    R.require(not ctx.torch.cuda.is_initialized(),'CUDA initialized')
    R.publish(out/'CLOSED.json',{'status':'PASS','result':R.ref_of(out/'RESULT_R25_CHANNELS_002.json'),'model_forwards':0,'gpu_calls':0,'score_calls':6,'oracle_evaluations':192})

if __name__=='__main__': main()
