"""Native e120 endpoints; unchanged certified R0 readout/evaluators."""
import json,os
from pathlib import Path
import torch
from ..r0.evaluation import rank_candidates,score_budget,POLICY
from ..r0.evaluation_data import Development548
from ..localization.runio import require,read,write,publish,sha,reference,event
from ..localization.consumer_geometry import positive_box
from .model import ROOT,ARM

def verify_cache(raw,endpoint,c,records):
    require(raw['schema']=='native_r18_common_raw_queries_v1' and raw['arm']==ARM and raw['epoch']==120 and raw['EMA'] is True,'native cache state identity differs')
    require(raw['checkpoint_sha256']==endpoint['checkpoint']['sha256'] and raw['source_sha256']==c['source_sha256'] and raw['contract_sha256']==c['contract_sha256'],'native cache source/endpoint differs')
    require(raw['evaluation_protocol_sha256']==c['evaluation_protocol_sha256'] and raw['image_ids']==[r['stable_image_id'] for r in records],'cache input/protocol differs')
    require(raw['pred_logits'].shape==(548,300,10) and raw['pred_boxes'].shape==(548,300,4) and raw['pred_logits'].dtype==raw['pred_boxes'].dtype==torch.float32,'native cache geometry/dtype differs')
    require(torch.equal(raw['query_ids'],torch.arange(300)) and torch.isfinite(raw['pred_logits']).all(),'cache query/logit identity differs')
    positive_box('native/e120/cache',raw['pred_boxes'])

def evaluate(c,out,holder):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'evaluation CUDA initialized')
    endpoint=read(Path(c['job_root'])/'formal/endpoint-lock.json');require(endpoint['status']=='R18_COMMON_E120_EMA_LOCKED','endpoint not locked')
    ref=read(Path(c['job_root'])/'inference/RESULT.json')['raw'];require(reference(ref['path'])==ref,'native cache SHA differs')
    raw=torch.load(ref['path'],map_location='cpu',weights_only=True);data=Development548(c['data']['development'])
    require(read(c['evaluation_protocol'])==POLICY and sha(c['evaluation_protocol'])==c['evaluation_protocol_sha256'],'certified evaluation protocol changed')
    verify_cache(raw,endpoint,c,data.records);parsed=[data.raw(r) for r in data.records];results={}
    for budget in (600,300):
        predictions=[p for i,row in enumerate(data.records) for p in rank_candidates(raw['pred_logits'][i],raw['pred_boxes'][i],row,budget)]
        write(out/f'top{budget}-predictions.json',predictions)
        metrics=score_budget(data.records,parsed,predictions,ROOT);write(out/f'top{budget}-metrics.json',metrics);results[str(budget)]=metrics
        event(out,'evaluation_budget_complete',name=f'top{budget}',budget=budget,images=548,checkpoint_sha256=raw['checkpoint_sha256'])
    publish(out/'RESULT.json',{'status':'PASS','arm':ARM,'epoch':120,'images':548,'budgets':[600,300],'results':results,'checkpoint_sha256':raw['checkpoint_sha256'],
        'model_forward_calls':0,'R0_rescored':False,'bootstrap_executed':False,'holdout_accessed':False})

def compare_metrics(native,baseline):
    require(native['COCO_bridge']['units']==baseline['COCO_bridge']['units'] and native['COCO_bridge']['maxDets']==baseline['COCO_bridge']['maxDets'] and native['COCO_bridge']['iouThrs']==baseline['COCO_bridge']['iouThrs'] and native['COCO_bridge']['areaRng']==baseline['COCO_bridge']['areaRng'],'comparison metric conventions differ')
    a=native['COCO_bridge']['metrics'];b=baseline['COCO_bridge']['metrics'];require(set(a)==set(b),'comparison metric schema differs')
    rows={k:{'native_score':a[k],'R0_score':b[k],'native_percent':a[k]*100 if a[k]>=0 else None,'R0_percent':b[k]*100 if b[k]>=0 else None,
        'native_minus_R0_pp':(a[k]-b[k])*100 if a[k]>=0 and b[k]>=0 else None} for k in a}
    return {'COCO_bridge':rows,'difference_unit':'pp; original score subtraction times100 once','official_native':native['official'],'official_R0':baseline['official'],
        'official_APS':None,'single_seed_development':True,'causal_attribution_to_single_structure_change':False,'paired_interval_not_requested':True}

def comparison(c,out):
    result=read(Path(c['job_root'])/'evaluation/RESULT.json');require(result['status']=='PASS' and result['budgets']==[600,300],'native evaluation not complete')
    rows={}
    for budget in (600,300):
        ref=c['R0_metrics'][str(budget)];require(reference(ref['path'])==ref,'certified R0 metric bytes changed')
        rows[str(budget)]=compare_metrics(result['results'][str(budget)],read(ref['path']))
    publish(out/'RESULT.json',{'status':'PASS','arm':ARM,'comparison':'native common protocol minus R0','budgets':rows,
        'R0_result_reused':True,'R0_model_loaded':False,'R0_reinferred':False,'point_comparison_only':True,'holdout_accessed':False})
