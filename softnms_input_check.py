"""Real CPU input/count proof and cache admission before a scoring attempt exists."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

PARENTS = ('parent_root','parent_execution','parent_execution_digest','parent_launch','seed0_contract')
SOURCES = ('softnms_job.py','softnms_input_check.py','softnms_parent_check.py','softnms_core.py','slice_core.py',
           'src/sparse_rtdetr/r0/evaluation.py','src/sparse_rtdetr/baseline/primary_evaluator.py',
           'src/sparse_rtdetr/data_protocol/evaluation.py','src/sparse_rtdetr/data_protocol/schema.py')


def source_identity():
    import softnms_job as j
    return {name:j.sha(j.ROOT/name) for name in SOURCES}


def count_checks():
    """Hand-counted fixtures use the original filter; old AP is never evaluated."""
    import softnms_job as j
    from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs
    from sparse_rtdetr.baseline.primary_evaluator import _input_payload
    from sparse_rtdetr.data_protocol.schema import canonical_json_bytes
    rows=[{'stable_image_id':'count-fixture','width':100,'height':100}]
    def obj(name,cat,box,score=1):
        return {'id':name,'category':cat,'box':box,'score':score,
                'raw_area':(box[2]-box[0])*(box[3]-box[1])}
    def pred(q,box):
        return {'image_id':'count-fixture','query_id':q,'category_id':1,'score':.9,'bbox_xyxy':box}
    inside=[10.,10.,20.,20.]; outside=[70.,70.,80.,80.]
    target=[obj('inside',1,inside),obj('outside',1,outside,0),obj('excluded',11,outside)]
    predictions=[pred(0,inside),pred(1,outside)]
    cases=[('no-filter',target, predictions,2,2,2,2),
           ('partial-filter',target+[obj('region',0,[0.,0.,50.,50.],0)],predictions,2,1,2,1),
           ('all-filter',target+[obj('region',0,[0.,0.,100.,100.],0)],predictions,2,0,2,0)]
    checks=[];rejections=[]
    def metric_for(parsed,preds,expected):
        payload=_input_payload(primary_inputs(rows,parsed,preds))
        return {'official':{'canonical_input_sha256':hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
                'evaluated_image_count':1,'evaluated_detection_count':expected['detections_after'],
                'evaluated_ground_truth_count':expected['GT_after']},'spatial_filter_counts':expected}
    for name,objects,preds,gb,ga,db,da in cases:
        parsed=[{'rows':objects}]
        expected=dict(GT_before=gb,GT_after=ga,detections_before=db,detections_after=da)
        metric=metric_for(parsed,preds,expected)
        proof=j.evaluation_count_proof(rows,parsed,preds,metric)
        j.require(proof['counts']==expected,'hand-counted fixture differs: '+name)
        checks.append({'name':name,'counts':expected})
        for field in ('GT_before','GT_after','detections_before','detections_after'):
            bad=copy.deepcopy(metric);bad['spatial_filter_counts'][field]+=1
            try:j.evaluation_count_proof(rows,parsed,preds,bad)
            except ValueError:rejections.append(name+'-wrong-'+field)
            else:raise AssertionError('count mismatch accepted')
        for field in ('evaluated_image_count','evaluated_detection_count','evaluated_ground_truth_count'):
            bad=copy.deepcopy(metric);bad['official'][field]+=1
            try:j.evaluation_count_proof(rows,parsed,preds,bad)
            except ValueError:rejections.append(name+'-wrong-'+field)
            else:raise AssertionError('evaluated mismatch accepted')
        bad=copy.deepcopy(metric);bad['official']['canonical_input_sha256']='f'*64
        try:j.evaluation_count_proof(rows,parsed,preds,bad)
        except ValueError:rejections.append(name+'-wrong-canonical-SHA')
        else:raise AssertionError('SHA mismatch accepted')
    for budget in (600,300):
        parsed=[{'rows':target+[obj('region',0,[0.,0.,50.,50.],0)]}]
        pool=[pred(q,inside) for q in range(budget)]+[pred(budget,outside)]
        selected=pool[:budget]
        expected=dict(GT_before=2,GT_after=1,detections_before=budget,detections_after=0)
        proof=j.evaluation_count_proof(rows,parsed,selected,metric_for(parsed,selected,expected))
        _,wrong_order,_=bridge_inputs(rows,parsed,pool)
        j.require(len(wrong_order[:budget])==1 and proof['counts']['detections_after']==0,
                  'budget/no-refill order fixture did not distinguish orders')
        checks.append({'name':f'top{budget}-before-ignore-no-refill','counts':expected,
                       'incorrect_filter_then_truncate_count':1})
    return {'status':'PASS','hand_counted_cases':checks,'rejections':rejections,
            'old_AP_calls':0,'original_filter_reused':True}


def run(request, out):
    import softnms_job as j
    start=time.monotonic()
    try:
        j.require(out.is_dir() and not (out/'STOP.json').exists()
                  and not (out/'REAL_DATA_GATE.json').exists(),'preparation worker already used')
        req=j.read(request)
        j.require(j.read(out/'PREPARATION_CONSUMED.json')['request']==j.reference(request),'preparation request differs')
        torch=j.cpu_only();before=torch.get_rng_state().clone()
        plan=j.read(j.PLAN)
        def progress(**fields):
            j.resources(out,plan)
            tmp=out/'heartbeat.tmp';tmp.write_bytes(j.encoded({'time':time.time(),**fields}));os.replace(tmp,out/'heartbeat.json')
        j.publish(out/'RESOURCE_START.json',j.resources(out,plan))
        gate=json.loads(j.exact(req['parent_schema_cpu_gate'],contents=True))
        j.require(gate['status']=='PASS' and gate['seeds']==[0,1,2]
                  and gate['plan_sha256']==j.sha(j.PLAN),'metadata gate incomplete')
        for n,v in gate['source_files'].items():j.require(j.sha(j.ROOT/n)==v,'metadata gate source changed')
        def forbidden(*a,**kw):raise AssertionError('preparation forbids old/new AP scoring')
        from sparse_rtdetr.r0 import evaluation
        from sparse_rtdetr.baseline import primary_evaluator
        with patch.object(evaluation,'score_budget',forbidden), \
             patch.object(primary_evaluator,'evaluate_primary_v1',forbidden), \
             patch.object(evaluation,'bridge_metrics',forbidden):
            j.publish(out/'COUNT_CPU_GATE.json',count_checks())
            specs,rows,data=j.context(req['parent_bindings'])
            j.publish(out/'METADATA_GATE.json',{'status':'PASS','parent_schema_cpu_gate':req['parent_schema_cpu_gate'],
                        'seeds':[0,1,2],'images':len(rows),'current_context_sha256':j.digest(specs)})
            parsed=j.parsed_gt(rows,data)
            proofs=j.prove_all_evaluation_inputs(specs,rows,parsed,progress)
            j.require(len(proofs['proofs'])==18,'not all original inputs proved')
            proof_ref=j.publish(out/'EVALUATION_INPUT_PROOF.json',proofs)
            prepared=[]
            for spec in specs:
                dest=out/f'seed{spec["seed"]}';dest.mkdir(exist_ok=False)
                p=j.prepare_seed(spec,rows,dest,progress)
                prepared.append(j.publish(dest/'PREPARED.json',p))
            cache_ref=j.publish(out/'CACHE_REBUILD_GATE.json',{'status':'PASS','seeds':[0,1,2],
                        'images_each':len(rows),'all_A_B_C_exact':True,'prepared':prepared,'new_score_budget_calls':0})
        j.require(not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state()),'preparation RNG/CUDA changed')
        final={'status':'PASS','plan_sha256':j.sha(j.PLAN),
                  'source_files':source_identity(),'parent_bindings':req['parent_bindings'],
                  'metadata_gate':j.reference(out/'METADATA_GATE.json'),'input_proof':proof_ref,'cache_gate':cache_ref,
                  'context_sha256':j.digest(specs),'rows_sha256':j.digest(rows),'score_budget_calls':0,
                  'GPU_calls':0,'elapsed_seconds':time.monotonic()-start}
        draft=j.publish(out/'PREPARATION_DRAFT.json',final)
        c={**req['parent_bindings'],'real_data_gate':{'preparation':draft,'input_proof':proof_ref,'cache_gate':cache_ref}}
        admitted_preparation(c,specs,rows)
        j.publish(out/'GATE_CONSUMER_CPU_GATE.json',gate_checks(c))
        final['gate_consumer_check']=j.reference(out/'GATE_CONSUMER_CPU_GATE.json')
        j.publish(out/'REAL_DATA_GATE.json',final)
        return 0
    except BaseException as e:
        if not (out/'STOP.json').exists():
            j.publish(out/'STOP.json',{'status':'STOP_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),
                      'elapsed_seconds':time.monotonic()-start,'stage':'real-input-and-cache-preparation'})
        return 2


def supervise(request, out):
    import softnms_job as j
    j.require(not out.exists(),'preparation attempt already exists')
    out.mkdir(parents=True)
    j.publish(out/'PREPARATION_CONSUMED.json',{'request':j.reference(request),'time':time.time()})
    started=time.monotonic();plan=j.read(j.PLAN);log=(out/'worker.log').open('xb')
    child=subprocess.Popen([sys.executable,'-B',str(j.ROOT/'softnms_job.py'),'input-worker',
                           '--request',str(request),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True)
    try:
        while child.poll() is None:
            time.sleep(10);j.resources(out,plan)
            touched=max([started]+[time.monotonic()-max(0,time.time()-p.stat().st_mtime)
                         for p in [out/'heartbeat.json'] if p.exists()])
            j.require(time.monotonic()-touched<plan['progress_timeout_seconds'],'preparation no progress')
        j.require(child.returncode==0 and j.read(out/'REAL_DATA_GATE.json')['status']=='PASS','preparation failed; no retry')
        return 0
    except BaseException as e:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if not (out/'STOP.json').exists():
            j.publish(out/'STOP.json',{'status':'STOP_NO_RETRY','error':repr(e),'traceback':traceback.format_exc()})
        return 2
    finally:
        log.close()
        j.publish(out/'WORKER_EXIT.json',{'pid':child.pid,'returncode':child.returncode,
                  'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})


def gate_identity(c):
    import softnms_job as j
    r=c['real_data_gate']
    gate=json.loads(j.exact(r['preparation'],contents=True))
    j.require(gate['status']=='PASS' and gate['score_budget_calls']==0
              and gate['plan_sha256']==j.sha(j.PLAN),'real data gate incomplete')
    j.require(gate['source_files']==source_identity(),'real data gate source changed')
    j.require(gate['parent_bindings']=={k:c[k] for k in PARENTS},'real data gate parents changed')
    metadata=json.loads(j.exact(gate['metadata_gate'],contents=True))
    j.require(metadata['status']=='PASS' and metadata['seeds']==[0,1,2]
              and metadata['current_context_sha256']==gate['context_sha256'],'metadata gate binding differs')
    for name in ('input_proof','cache_gate'):
        j.require(r[name]==gate[name],'real data gate reference differs')
        value=json.loads(j.exact(r[name],contents=True))
        j.require(value['status']=='PASS','real gate not PASS')
    proofs=json.loads(j.exact(r['input_proof'],contents=True))['proofs']
    j.require([(p['seed'],p['arm'],p['budget']) for p in proofs]==
              [(s,a,b) for s in range(3) for a in ('A','B','C') for b in (600,300)],'18-input gate missing/reordered')
    return gate


def gate_checks(c):
    """Actual gate consumer, authenticated real refs plus isolated semantic faults."""
    import softnms_job as j
    gate_identity(c)
    original=j.exact;rejected=[]
    for name,target,mutation in [
        ('source',c['real_data_gate']['preparation']['path'],lambda x:x['source_files'].update({'softnms_job.py':'f'*64})),
        ('parent',c['real_data_gate']['preparation']['path'],lambda x:x['parent_bindings'].update(parent_execution_digest='f'*64)),
        ('input-missing',c['real_data_gate']['input_proof']['path'],lambda x:x['proofs'].pop()),
        ('input-duplicate',c['real_data_gate']['input_proof']['path'],lambda x:x['proofs'].__setitem__(-1,x['proofs'][0])),
        ('input-order',c['real_data_gate']['input_proof']['path'],lambda x:x['proofs'].reverse()),
        ('input-fail',c['real_data_gate']['input_proof']['path'],lambda x:x.update(status='FAIL')),
        ('cache-fail',c['real_data_gate']['cache_gate']['path'],lambda x:x.update(status='FAIL'))]:
        def fault(ref,*,contents=False):
            raw=original(ref,contents=contents)
            if contents and ref['path']==target:
                value=json.loads(raw);mutation(value);return j.encoded(value)
            return raw
        with patch.object(j,'exact',fault):
            try:gate_identity(c)
            except ValueError:rejected.append(name)
            else:raise AssertionError('consumer accepted '+name)
    bad=copy.deepcopy(c);bad['real_data_gate']['preparation']['sha256']='f'*64
    try:gate_identity(bad)
    except ValueError:rejected.append('real-reference-SHA')
    else:raise AssertionError('consumer accepted corrupt gate SHA')
    return {'status':'PASS','positive':'actual authenticated gate consumer','rejected':rejected,
            'scientific_scoring_calls':0}


def admitted_preparation(c,specs,rows):
    import softnms_job as j
    gate=gate_identity(c)
    j.require(gate['context_sha256']==j.digest(specs) and gate['rows_sha256']==j.digest(rows),'prepared context changed')
    cache=json.loads(j.exact(gate['cache_gate'],contents=True))
    j.require(cache['seeds']==[0,1,2] and cache['images_each']==548 and cache['all_A_B_C_exact'] is True,
              'cache rebuild gate incomplete')
    j.require(len(cache['prepared'])==3,'prepared seeds missing')
    result=[]
    for seed,ref in enumerate(cache['prepared']):
        value=json.loads(j.exact(ref,contents=True))
        j.require(value['seed']==seed and value['identity']==specs[seed]['identity']
                  and value['images']==len(rows) and value['all_A_B_C_exact'] is True,'prepared seed identity differs')
        j.exact(value['counts'])
        for arm in ('E','D'):
            for b in ('600','300'):j.exact(value['predictions'][arm][b])
        result.append(value)
    return result
