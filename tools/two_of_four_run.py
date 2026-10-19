"""Six fixed cache comparisons; all inputs/rebuilds pass before the first AP call."""
from collections import defaultdict
import io
import json
from pathlib import Path
import time
import numpy as np
from two_of_four_job import dependency,publish,require,score,sha
from two_of_four_core import PAIRS,merge_pair,screen,select,coverage

def sequence_evidence(rows,parsed,predictions,arm):
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    dataset,det,_=bridge_inputs(rows,parsed,predictions)
    dg=defaultdict(list);pg=defaultdict(list)
    for g in dataset['annotations']:
        if not g['iscrowd']:dg[g['image_id']].append(g)
    for p in det:pg[p['image_id']].append(p)
    totals=defaultdict(lambda:dict(GT=0,covered=0,detections=0,images=0))
    image_records=[]
    for i,row in enumerate(rows,1):
        seq=Path(row['annotation_relative_path']).stem.split('_')[0]
        gs=dg[i];ps=pg[i]
        xy=lambda xs:[[g['bbox'][0],g['bbox'][1],g['bbox'][0]+g['bbox'][2],g['bbox'][1]+g['bbox'][3]] for g in xs]
        best=coverage(xy(ps),[p['category_id'] for p in ps],xy(gs),[g['category_id'] for g in gs])
        for area,lo,hi in (('small',0,1024),('medium',1024,9216),('large',9216,1e10)):
            eligible=np.array([lo<=g['area']<=hi for g in gs],bool)
            for threshold in (.5,.75):
                hit=eligible&(best>=threshold);key=(seq,area,threshold)
                t=totals[key];t['GT']+=int(eligible.sum());t['covered']+=int(hit.sum())
                t['detections']+=len(ps);t['images']+=1
                image_records.append(dict(image=row['stable_image_id'],sequence=seq,area=area,iou=threshold,
                    GT=int(eligible.sum()),covered_ids=[g['raw_annotation_id'] for k,g in enumerate(gs) if hit[k]]))
    return [dict(arm=arm,sequence=s,area=a,iou=t,**v) for (s,a,t),v in sorted(totals.items())],image_records

def run(parent,c,plan,out):
    start=time.monotonic();j,torch,spec,rows,parsed=dependency(parent)
    resources=[j.resources(out,plan)]
    publish(out/'WORKER_STARTED.json',dict(pid=__import__('os').getpid(),time=time.time(),contract=c['task_id']))
    ad=c['admission'];admission=json.loads(j.exact(ad,contents=True));require(admission['status']=='ADMISSION_PASS','admission reference')
    publish(out/'ADMISSION.json',dict(parent_sha256=sha(parent),admission=ad,checkpoint=spec['identity']['checkpoint_sha256'],resource=resources[0]))
    expected={};proofs=[];original={};baseline={}
    for arm in ('A','B','C'):
        ps=json.loads(j.exact(spec['evaluation']['predictions'][arm]['600'],contents=True))
        proofs.append(j.evaluation_input_proof(spec,rows,parsed,ps,arm,600))
        expected[arm]=j.group(ps,spec['ids'],600);original[arm]=ps
        baseline[arm]=spec['metrics'][arm]['600']['COCO_bridge']['metrics']
    publish(out/'INPUT_PROOFS.json',proofs)
    from slice_core import classwise_nms
    torch_rng=torch.get_rng_state().clone()
    predictions={'P'+''.join(map(str,p)):[] for p in PAIRS}
    counts=defaultdict(int)
    for batch,ref in enumerate(spec['full']['raw_view_refs']):
        resources.append(j.resources(out,plan))
        raw=torch.load(io.BytesIO(j.exact(ref,contents=True)),map_location='cpu',weights_only=True)
        part=rows[batch*4:batch*4+4];pools=j.decode_raw(raw,part,spec['identity'],0)
        for row,pool in zip(part,pools):
            image=row['stable_image_id'];whole=[p for p in pool if p.view_id==0]
            for arm,items in (('A',sorted(whole,key=lambda p:p.sort_key)),('B',classwise_nms(whole)),('C',classwise_nms(pool))):
                rebuilt=[j.prediction(image,p,whole=arm=='A') for p in items[:600]]
                require(rebuilt==expected[arm][image],'original '+arm+' exact rebuild differs')
                counts[arm]+=1
            for pair in PAIRS:
                arm='P'+''.join(map(str,pair))
                predictions[arm].extend(j.prediction(image,p) for p in merge_pair(pool,pair))
        print(json.dumps(dict(event='cache_rebuilt',batch=batch+1,batches=137)),flush=True)
    refs={a:publish(out/(a+'-predictions.json'),ps) for a,ps in predictions.items()}
    publish(out/'REBUILD.json',dict(status='PASS',raw_batches=137,original_exact_images=dict(counts),
        raw_refs=spec['full']['raw_view_refs'],predictions=refs,all_six_prepared_before_scoring=True))
    # No parameter or configuration decisions after this point.
    results={};metrics_refs={};sequence=[];details={}
    for a,ps in original.items():
        t,d=sequence_evidence(rows,parsed,ps,a);sequence+=t;details[a]=d
    for arm,ps in predictions.items():
        resources.append(j.resources(out,plan))
        print(json.dumps(dict(event='score_started',arm=arm,budget=600)),flush=True)
        record=score(rows,parsed,ps,j.ROOT)
        require(record['COCO_bridge']['units']=='0..1 AP, -1 means no GT in an area/class aggregation','metric units')
        record['input_proof']=j.evaluation_count_proof(rows,parsed,ps,record)
        metric_ref=publish(out/(arm+'-metrics.json'),record);metrics_refs[arm]=metric_ref
        metrics=record['COCO_bridge']['metrics']
        results[arm]=dict(metrics=metrics,official=record['official'],spatial_filter_counts=record['spatial_filter_counts'],**screen(metrics,baseline))
        t,d=sequence_evidence(rows,parsed,ps,arm);sequence+=t;details[arm]=d
        print(json.dumps(dict(event='score_complete',arm=arm,metric=metric_ref)),flush=True)
    base={(r['image'],r['area'],r['iou']):set(r['covered_ids']) for r in details['C']}
    transitions=defaultdict(lambda:dict(gained=0,lost=0,retained=0))
    for arm,records in details.items():
        for r in records:
            before=base[(r['image'],r['area'],r['iou'])];after=set(r['covered_ids'])
            t=transitions[(arm,r['sequence'],r['area'],r['iou'])]
            t['gained']+=len(after-before);t['lost']+=len(before-after);t['retained']+=len(before&after)
    for r in sequence:r.update(transitions[(r['arm'],r['sequence'],r['area'],r['iou'])])
    publish(out/'SEQUENCE_DETAILS.json',details)
    publish(out/'SEQUENCES.json',dict(definition=plan['sequence_diagnostic'],rows=sequence))
    chosen=select(results)
    publish(out/'SUMMARY.json',dict(baseline=baseline,results=results,selected=chosen,
        selection_rule=plan['selection'],scope='seed0 development configuration selection only',
        quality_status='FIXED_PAIR_SCREEN_PASS' if chosen else 'ALL_FIXED_PAIRS_FAIL_CLOSED',
        measured_end_to_end_cost='NOT_MEASURED',future_experiment_authorized=False))
    publish(out/'RESOURCE.json',resources)
    require(torch.equal(torch_rng,torch.get_rng_state()) and not torch.cuda.is_initialized(),'RNG/CUDA state changed')
    cost=dict(cpu_wall_seconds=time.monotonic()-start,score_groups=6,evaluator_calls=12,
        training_updates=0,forward_calls=0,GPU_calls=0,raw_batches=137,prior_costs='preserved separately',
        theoretical_view_fraction_vs_C=3/5,actual_end_to_end_speedup='NOT_MEASURED')
    publish(out/'COST.json',cost)
    return dict(status='FIXED_PAIR_SCREEN_PASS' if chosen else 'ALL_FIXED_PAIRS_FAIL_CLOSED',
        selected=chosen,complete_pairs=list(results),metrics=metrics_refs,cost=cost,automatic_next=False)
