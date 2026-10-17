"""Actual CLI synthetic integration, including the unchanged two evaluators.

No production path/contract/token accepted here. Four synthetic images are
never combined with or reported as the 548-development scientific result.
"""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


def run(out):
    import softnms_job as j
    from slice_core import box_iou64, tile_windows
    j.require(not out.exists(),'CPU fixture output already exists')
    out.mkdir(parents=True)
    torch=j.cpu_only()
    from sparse_rtdetr.localization.data import parse_raw
    from sparse_rtdetr.r0.evaluation import rank_candidates, bridge_inputs
    from softnms_core import METRICS
    started=time.monotonic()
    rows=[{'stable_image_id':f'cpu-synthetic-{i}','width':1024,'height':768,'split':'val'} for i in range(4)]
    annotation=[]
    for cat in range(1,11):
        for y,size in [(20,16),(200,48),(400,100)]:
            annotation.append(f'{cat*85-65},{y},{size},{size},1,{cat},0,0')
    annotation+=['0,650,30,30,0,0,0,0','50,650,20,20,0,1,0,0','80,650,20,20,1,11,0,0']
    rawbytes=('\n'.join(annotation)+'\n').encode()
    parsed=[parse_raw(rawbytes,r['stable_image_id']) for r in rows]
    checks=[]
    def reject(name, fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError,KeyError):checks.append(name);return
        raise AssertionError('negative accepted: '+name)
    specs=[]
    for seed in range(3):
        base=out/f'fixture-input-seed{seed}';base.mkdir()
        identity={'task_id':'cpu-synthetic','source_sha256':'c'*64,'contract_sha256':'d'*64,'checkpoint_sha256':str(seed)*64}
        if seed:identity.update(seed=seed,scope=f'formal_R18_seed{seed}_e120')
        views=[]
        for view in range(5):
            tile=tile_windows(768,1024)[view-1] if view else None
            width,height=(tile.width,tile.height) if tile else (1024,768)
            xoff,yoff=(tile.x0,tile.y0) if tile else (0,0)
            boxes=torch.zeros(4,300,4,dtype=torch.float32)
            logits=torch.full((4,300,10),-8.,dtype=torch.float32)
            for q in range(300):
                if q<60:
                    g=parsed[0]['gt'][q%30];x0,y0,x1,y1=g['box'];cat=g['category']
                    logits[:,q,cat-1]=4.-q*.01+seed*.001
                else:x0,y0,x1,y1=(q%40)*24,700,(q%40)*24+5,705
                boxes[:,q]=torch.tensor([((x0+x1)/2-xoff)/width,((y0+y1)/2-yoff)/height,
                                         (x1-x0)/width,(y1-y0)/height],dtype=torch.float32)
            logits[:,299]=-1000.
            views.append({'view_id':view,'pred_boxes':boxes,'pred_logits':logits})
        raw={'schema':'sahi-fixed-raw-views-v1' if seed==0 else 'sahi-repeat-raw-views-v1',
             **identity,'image_ids':[r['stable_image_id'] for r in rows],
             'geometry':[[t.as_dict() for t in tile_windows(768,1024)] for r in rows],'views':views}
        rawpath=base/'raw.pth';torch.save(raw,rawpath);rawref=j.reference(rawpath)
        pools=j.decode_raw(raw,rows,identity,seed)
        def scalar_hard(pool):
            pending=sorted(pool,key=lambda c:c.sort_key);kept=[]
            while pending:
                first=pending.pop(0);kept.append(first)
                pending=[p for p in pending if p.category_id!=first.category_id or box_iou64(first.box_xyxy,p.box_xyxy)<=.5]
            return kept
        predictions={a:{str(b):[] for b in (600,300)} for a in ('A','B','C')}
        for k,(row,pool) in enumerate(zip(rows,pools)):
            image=row['stable_image_id'];whole=[v for v in pool if v.view_id==0]
            for b in (600,300):
                predictions['A'][str(b)]+=rank_candidates(views[0]['pred_logits'][k],views[0]['pred_boxes'][k],row,b)
                for arm,pp in [('B',whole),('C',pool)]:
                    predictions[arm][str(b)]+=[j.prediction(image,p) for p in scalar_hard(pp)[:b]]
        refs={a:{b:j.publish(base/f'{a}{b}.json',value) for b,value in budgets.items()} for a,budgets in predictions.items()}
        old={'scores':{a:{str(b):{m:.25 for m in METRICS} for b in (600,300)} for a in ('A','B','C')}}
        oldref=j.publish(base/'original-comparison.json',old)
        spec={'seed':seed,'identity':identity,'ids':raw['image_ids'],'full':{'raw_view_refs':[rawref]},
              'evaluation':{'predictions':refs,'comparison':oldref},'comparison':old}
        specs.append(spec)
        # Decode and authentication rejection paths use actual functions.
        bad=copy.deepcopy(raw);bad['checkpoint_sha256']='bad'
        reject(f'seed{seed}-wrong-checkpoint',lambda:j.decode_raw(bad,rows,identity,seed))
        bad=copy.deepcopy(raw);bad['views']=bad['views'][:-1]
        reject(f'seed{seed}-missing-view',lambda:j.decode_raw(bad,rows,identity,seed))
        bad=copy.deepcopy(raw);bad['geometry'][0][0]['x0']=3
        reject(f'seed{seed}-bad-geometry',lambda:j.decode_raw(bad,rows,identity,seed))
        bad=copy.deepcopy(raw);bad['views'][0]['pred_logits'][0,0,0]=float('nan')
        reject(f'seed{seed}-nonfinite-raw',lambda:j.decode_raw(bad,rows,identity,seed))
        ref={**rawref,'sha256':'f'*64}
        reject(f'seed{seed}-SHA-damage',lambda:j.exact(ref))
        ref={**rawref,'path':str(base/'missing-raw.pth')}
        reject(f'seed{seed}-missing-file',lambda:j.exact(ref))
        ref={**rawref,'bytes':rawref['bytes']+1}
        reject(f'seed{seed}-wrong-size',lambda:j.exact(ref))
    # Verify raw-ignore and zero-score no-refill using unchanged evaluator bridge.
    ignore_pred=[{'image_id':rows[0]['stable_image_id'],'category_id':1,'bbox_xyxy':[0.,650.,20.,670.],'score':0.}]
    dataset,filtered,counts=bridge_inputs(rows,parsed,ignore_pred)
    j.require(not filtered and counts['detections_before']==1 and counts['detections_after']==0,'ignore/no-refill fixture differs')
    j.require(any(x['iscrowd']==1 for x in dataset['annotations']),'class-ignore lost')
    j.require(not torch.cuda.is_initialized(),'fixture CUDA initialized')
    from softnms_input_check import count_checks as check_count_flow
    input_count_gate=check_count_flow()
    work=out/'actual-entry';work.mkdir()
    j.process_inputs(work,specs,rows,'cpu-fixture-not-production',lambda **kw:None,lambda:parsed)
    j.export(work)
    count_checks=[]
    for seed in range(3):
        prepared=j.read(work/f'seed{seed}/PREPARED.json')
        counts=json.loads(j.exact(prepared['counts'],contents=True))
        j.require([v['image_id'] for v in counts]==[r['stable_image_id'] for r in rows],'counts image order differs')
        for row in counts:
            for arm,total in [('E',600),('D',3000)]:
                values=row[arm]
                for key in ('candidate_count','input_zero_scores','underflow_to_zero','output_zero_scores','subnormal_scores'):
                    j.require(type(values[key]) is int and values[key]>=0,'JSON count not native nonnegative integer')
                j.require(values['candidate_count']==total,'candidate count changed in JSON')
                j.require(values['sigma']==.5 and values['score_cutoff'] is None,'scientific JSON values changed')
                for budget in (600,300):
                    vc=values[f'top{budget}_view_counts']
                    j.require(set(vc)==set(map(str,range(5))) and all(type(v) is int and v>=0 for v in vc.values()),'view counts not native')
                    j.require(sum(vc.values())==budget,'view counts do not equal output budget')
        count_checks.append({'seed':seed,'images':len(counts),'native_counts_and_values':True})
    exported=j.read(work/'SLIM_EXPORT.json')
    j.require(exported['status']=='READY','export not ready')
    j.require(exported['total_bytes']==sum(r['bytes'] for r in exported['files']),'export byte total differs')
    for ref in exported['files']:
        j.exact(ref)
        j.require('predictions' not in Path(ref['path']).name and Path(ref['path']).suffix in ('.json','.md','.csv'),'export whitelist leaked large artifact')
    metric_refs=[r for r in exported['files'] if r['path'].endswith('-metrics.json')]
    j.require(len(metric_refs)==12,'export missing double-evaluator results')
    for ref in metric_refs:
        metric=json.loads(j.exact(ref,contents=True))
        j.require('official' in metric and 'COCO_bridge' in metric and 'spatial_filter_counts' in metric,'double evaluator result missing')
        j.require(metric['softnms_binding']['plan_sha256']==j.sha(j.PLAN),'metric plan binding differs')
    original=(work/'events.jsonl').read_bytes()
    lines=original.splitlines(keepends=True)
    for name,edited in [('missing',lines[:-1]),('duplicate',lines+[lines[-1]]),('reordered',lines[:2]+[lines[3],lines[2]]+lines[4:])]:
        path=out/(name+'.jsonl');path.write_bytes(b''.join(edited))
        reject('lifecycle-'+name,lambda p=path:j.Journal.validate(p,'cpu-fixture-not-production',complete=True))
    reject('lifecycle-wrong-binding',lambda:j.Journal.validate(work/'events.jsonl','wrong',complete=True))
    broken=json.loads(lines[0]);broken['fields']['contract_sha256']='damaged'
    damage=out/'damaged.jsonl';damage.write_bytes(j.encoded(broken).replace(b'\n',b'')+b'\n'+b''.join(lines[1:]))
    reject('lifecycle-corrupt-record',lambda:j.Journal.validate(damage,'cpu-fixture-not-production',complete=True))
    before=(work/'CLOSED.json').read_bytes()
    reject('immutable-closure',lambda:j.publish(work/'CLOSED.json',{'status':'replaced'}))
    j.require((work/'CLOSED.json').read_bytes()==before,'rejected replacement changed closure')
    # Negative cases enter the actual CLI, not a manually assembled event stream.
    for name,args in [('used-output',['--output',str(out)]),('unexpected-production-contract',['--output',str(out/'unused'),'--contract','forbidden'])]:
        p=subprocess.run([sys.executable,'-B',str(j.ROOT/'softnms_job.py'),'cpu-check',*args],capture_output=True,text=True)
        j.require(p.returncode!=0 and 'ValueError' in p.stderr,'invalid actual CLI accepted')
        checks.append('CLI-'+name)
    j.require(not (out/'unused').exists() and (work/'events.jsonl').read_bytes()==original,'rejected CLI changed evidence')
    require_result=j.read(work/'RESULT.json')
    j.require(require_result['new_score_budget_calls']==12 and j.read(work/'CLOSED.json')['status']=='PASS','entry did not complete12 evaluator calls')
    # New scoring path consumes already admitted immutable preparation, never rebuilds.
    certified=[j.read(work/f'seed{s}/PREPARED.json') for s in range(3)]
    reused=out/'certified-entry';reused.mkdir()
    def forbidden(*args,**kwargs):raise AssertionError('certified path rebuilt candidates')
    with patch.object(j,'prepare_seed',forbidden):
        j.process_inputs(reused,specs,rows,'cpu-certified-fixture',lambda **kw:None,lambda:parsed,certified)
    j.export(reused)
    j.require([r['scores'] for r in j.read(reused/'RESULT.json')['seed_results']]
              ==[r['scores'] for r in require_result['seed_results']],
              'certified preparation changed synthetic scores')
    j.publish(out/'CPU_RESULT.json',{'status':'PASS','actual_entry':'softnms_job.py cpu-check',
              'same_processing_functions':'prepare_seed/process_inputs/evaluate_seed/Journal/export',
              'synthetic_images':4,'synthetic_seeds':[0,1,2],'real_score_budget_calls':12,
              'real_evaluators':['COCO_bridge','official_style'],'negative_cases':checks,
              'statistics_JSON_roundtrip':count_checks,'slim_export_refs_verified':len(exported['files']),
              'slim_export_bytes_verified':exported['total_bytes'],'closed_events':j.Journal.validate(work/'events.jsonl','cpu-fixture-not-production',complete=True),
              'CUDA_initialized':torch.cuda.is_initialized(),'detector_constructed':False,'production_token_consumed':False,
              'actual_experiment_scoring':False,'elapsed_seconds':time.monotonic()-started})
    j.publish(out/'REAL_FLOW_CPU_GATE.json',{'status':'PASS','count_checks':input_count_gate,
              'synthetic_certified_path':True,'additional_synthetic_score_budget_calls':12,
              'certified_closure':j.Journal.validate(reused/'events.jsonl','cpu-certified-fixture',complete=True),
              'new_full_development_scoring':False,'GPU_calls':0})
    print(json.dumps({'status':'PASS','output':str(out),'synthetic_only':True,'real_evaluator_calls':12}))
    return 0
