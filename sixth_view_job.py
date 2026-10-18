"""Single-use supervised six-view chain. No detector training dispatch exists."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import sixth_view_runtime as rt
from sixth_view_runtime import ROOT,read,require,reference,digest,publish


def command(c,action,seed=None,mode=None):
    args=[c['python'],'-B',str(ROOT/'sixth_view_job.py'),action,'--contract',c['contract_path']]
    if seed is not None:args+=['--seed',str(seed),'--mode',mode]
    return args


def environment(c,gpu=False):
    return {**os.environ,'CUDA_VISIBLE_DEVICES':c['gpu_uuid'] if gpu else '',
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),
        'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2' if gpu else '1',
        'MKL_NUM_THREADS':'2' if gpu else '1','OPENBLAS_NUM_THREADS':'1','MKL_THREADING_LAYER':'GNU'}


def verify_phase(c,seed,mode):
    rt.install()  # Fresh archive readers must install the same protocol first.
    from sparse_rtdetr.localization import channel
    out=Path(c['job_root'])/f'seed{seed}'/mode;runtime=Path(c['runtime_root'])/f'seed{seed}'/mode
    closure=read(out/'CLOSED.json');b=read(runtime/'identity.json')
    for k in ('task_id','seed','mode','source_sha256','contract_sha256','checkpoint_sha256','steps','output_root'):
        require(b[k]==rt.identity(c,seed,mode)[k],'closed phase identity differs: '+k)
    require(closure['result']==reference(out/'RESULT.json') and closure['archive']==reference(out/'runtime-archive-index.json'),'closed result/archive changed')
    channel.verify_archive(runtime,out,b)
    ex=read(runtime/'supervisor-exit.json')
    require(ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'phase exit/cleanup incomplete')
    reader=channel.Reader(runtime/'progress.jsonl',b,'progress');last,_=reader.poll()
    require(reader.clock.get('_six_next')==len(b['steps']) and last['payload']['label']=='complete','terminal event missing')
    require(read(out/'execution-binding.json')==b,'worker execution binding differs')
    result=read(out/'RESULT.json');require(result['status']=='PASS','phase not PASS')
    return result


def supervise_phase(c,seed,mode):
    from sparse_rtdetr.localization import guard,channel
    gpu=not c['synthetic'] and mode!='evaluation'
    out=Path(c['job_root'])/f'seed{seed}'/mode;runtime=Path(c['runtime_root'])/f'seed{seed}'/mode
    require(read(out.parent/(mode+'-CONSUMED.json'))['contract_sha256']==c['contract_sha256'],'phase credential differs')
    code=guard.supervise(command(c,'worker',seed,mode),out,environment(c,gpu),gpu_uuid=c['gpu_uuid'],
        boot_id=c['boot_id'],gpu=gpu,checkpoint_bytes=0,formal=False,runtime=runtime,run_identity=rt.identity(c,seed,mode))
    if code:return code
    b=read(runtime/'identity.json');channel.verify_archive(runtime,out,b)
    reader=channel.Reader(runtime/'progress.jsonl',b,'progress');record,_=reader.poll()
    require(reader.clock.get('_six_next')==len(b['steps']) and record['payload']['label']=='complete','cannot close missing/duplicate terminal')
    publish(out/'CLOSED.json',{'status':'PASS','seed':seed,'mode':mode,'result':reference(out/'RESULT.json'),
        'archive':reference(out/'runtime-archive-index.json')})
    verify_phase(c,seed,mode)
    return 0


def phase(c,seed,mode):
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity as sid
    root=Path(c['job_root'])/f'seed{seed}';root.mkdir(exist_ok=True)
    require(not (root/mode).exists(),'phase already used; cannot replay')
    sample=guard.snapshot(root);gpu=not c['synthetic'] and mode!='evaluation'
    require(sample['boot_id']==c['boot_id'] and sid(sample['storage'])==c['storage_identity'],'fresh boot/storage differs')
    guard.screen(sample,gpu=gpu,gpu_uuid=c['gpu_uuid'],entry=True,disk_bytes=40*1024**3)
    publish(root/(mode+'-ADMISSION.json'),sample)
    publish(root/(mode+'-CONSUMED.json'),{'contract_sha256':c['contract_sha256'],'seed':seed,'mode':mode,'epoch':time.time(),'no_retry':True})
    started=time.monotonic()
    with (root/(mode+'-supervisor.log')).open('xb') as log:
        p=subprocess.run(command(c,'supervise',seed,mode),env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    publish(root/(mode+'-EXIT.json'),{'returncode':p.returncode,'elapsed_seconds':time.monotonic()-started})
    require(p.returncode==0,'supervised phase failed: '+mode)
    return verify_phase(c,seed,mode)


def score(rows,parsed,preds,path,binding):
    from sparse_rtdetr.r0.evaluation import score_budget
    # Exact frozen dual evaluator; post-budget raw-ignore occurs inside it.
    value=score_budget(rows,parsed,preds,ROOT)
    require(value['spatial_filter_counts']['detections_before']==len(preds),'before-filter count differs')
    require(value['official']['evaluated_detection_count']==value['spatial_filter_counts']['detections_after'],'after-filter count differs')
    value['sixth_binding']=binding
    publish(path,value)
    return value


def synthetic_predictions():
    """Explicit small CPU fixture. Never passed to production parent admission."""
    from sparse_rtdetr.localization.data import parse_raw
    from slice_core import Candidate
    from sixth_view_core import SixthCandidate,merge_six
    from infer_seed import prediction
    row={'stable_image_id':'synthetic-sixth-0','width':200,'height':160,'split':'val'}
    raw=parse_raw(b'10,10,10,10,1,1,0,0\n',row['stable_image_id'])
    base=[Candidate((q*.2,cat*2.,q*.2+10,cat*2.+10),.4,v,q,cat,(cat-1)*300+q+1) for v in range(5) for cat in (1,2) for q in range(300)]
    extra=[SixthCandidate((q*.2+1,cat*2.,q*.2+11,cat*2.+10),.5,5,q,cat,(cat-1)*300+q+1) for cat in (1,2) for q in range(300)]
    m=merge_six(base,extra,variant='C-center')
    require(any(p.view_id==5 for p in m['top600']),'fixture must exercise real view5')
    return [row],[raw],{str(b):[prediction(row['stable_image_id'],p) for p in m[f'top{b}']] for b in (600,300)}


def evaluate(c,seed,out,emit):
    from sixth_view_decision import assess_quality
    if c['synthetic']:
        rows,parsed,preds=synthetic_predictions();prepared={v:preds for v in ('C-center','C-flip')}
        old_scores=None
    else:
        import torch
        from sixth_view_core import SixthCandidate,merge_six
        from sixth_view_adapter import decode_extra
        from infer_seed import prediction
        old,specs,rows,parsed=rt.parent_context(c);spec=specs[seed]
        raw=[]
        for v in ('center','flip'):
            r=verify_phase(c,seed,'full-'+v)
            require(r['forward_calls']==139 and len(r['raw_refs'])==137,'full view workload not closed')
            raw.append(r['raw_refs'])
        prepared={v:{'600':[],'300':[]} for v in ('C-center','C-flip')}
        originals={str(b):json.loads(old.exact(spec['full']['predictions'][str(b)],contents=True)) for b in (600,300)}
        rebuilt={'600':[],'300':[]}
        for k,ref in enumerate(spec['full']['raw_view_refs']):
            rt.live(c,out,False)
            br=rows[k*4:k*4+4]
            pool=old.decode_raw(torch.load(io.BytesIO(old.exact(ref,contents=True)),map_location='cpu',weights_only=True),br,spec['identity'],seed)
            for j,row in enumerate(br):
                literal=merge_six(pool[j])
                for b in ('600','300'):rebuilt[b].extend(prediction(row['stable_image_id'],p) for p in literal['top'+b])
            for vi,v in enumerate(('C-center','C-flip')):
                payload=torch.load(io.BytesIO(old.exact(raw[vi][k],contents=True)),map_location='cpu',weights_only=True)
                for key,val in dict(seed=seed,variant=v,view_id=5,source_sha256=c['source_sha256'],contract_sha256=c['contract_sha256'],checkpoint_sha256=c['parents'][seed]['checkpoint']['sha256'],image_ids=[r['stable_image_id'] for r in br]).items():require(payload[key]==val,'extra raw identity differs: '+key)
                require(len(payload['views'])==1 and payload['views'][0]['view_id']==5,'extra raw view parser')
                p=payload['views'][0]
                for j,row in enumerate(br):
                    extra=decode_extra(p['pred_logits'][j],p['pred_boxes'][j],row,v)
                    merged=merge_six(pool[j],extra,variant=v)
                    for b in ('600','300'):prepared[v][b].extend(prediction(row['stable_image_id'],q) for q in merged['top'+b])
        require(rebuilt==originals,'literal original C cache rebuild differs')
        publish(out/'DATA_ADMISSION.json',{'status':'PASS','seed':seed,'original_C_two_budgets_exact':True,'raw_batches':137,'old_rebuild_certificate':c['parent_contract'],'new_view_SHA_and_identity_exact':True})
        old_scores={a:{b:spec['metrics'][a][b]['COCO_bridge']['metrics'] for b in ('600','300')} for a in ('A','C')}
        for v in prepared:
            for b in prepared[v]:publish(out/f'{v}-top{b}-predictions.json',prepared[v][b])
    scores={};refs=[]
    for variant in ('C-center','C-flip'):
        scores[variant]={}
        for budget in ('600','300'):
            rt.live(c,out,False)
            value=score(rows,parsed,prepared[variant][budget],out/f'{variant}-top{budget}-metrics.json',
                {'task_id':rt.TASK,'seed':seed,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
                 'checkpoint_sha256':c['parents'][seed]['checkpoint']['sha256'],'variant':variant,'budget':int(budget),'synthetic':c['synthetic']})
            scores[variant][budget]=value['COCO_bridge']['metrics'];refs.append(reference(out/f'{variant}-top{budget}-metrics.json'))
            emit.emit(f'score-{variant}-{budget}',metrics=refs[-1])
    if c['synthetic']:
        decision={'synthetic_only':True,'scientific_decision':False,'all_four_actual_dual_evaluator_calls':True}
    else:decision=assess_quality({**old_scores,**scores})
    publish(out/'COMPARISON.json',decision)
    publish(out/'RESULT.json',{'status':'PASS','seed':seed,'new_dual_evaluator_calls':4,'scores':scores,'metrics_refs':refs,
        'comparison':reference(out/'COMPARISON.json'),'detector_updates':0,'forward_calls':0,'synthetic':c['synthetic']})
    emit.emit('complete',result=reference(out/'RESULT.json'))


def worker(c,seed,mode):
    import torch
    from sparse_rtdetr.localization import channel,guard
    out=Path(c['job_root'])/f'seed{seed}'/mode;b=channel.binding()
    require(b is not None and read(Path(b['runtime_root'])/'identity.json')==b,'worker runtime identity absent')
    require(os.getppid()==b['supervisor_pid'] and guard.identity(os.getppid())['start']==b['supervisor_start'],'worker parent identity')
    for k,v in rt.identity(c,seed,mode).items():
        if k not in ('supervisor_pid','supervisor_start'):require(b[k]==v,'worker contract identity: '+k)
    publish(out/'execution-binding.json',b);emit=rt.Emitter(c,seed,mode,out)
    try:
        rt.live(c,out,not c['synthetic'] and mode!='evaluation');emit.emit('begin')
        if c['synthetic']:
            require(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized(),'CPU fixture CUDA initialized')
            if c.get('inject_worker_failure'):raise RuntimeError('EXPECTED_CPU_WORKER_EXCEPTION')
        if mode=='evaluation':evaluate(c,seed,out,emit)
        elif c['synthetic']:
            # Exercise the same final process/channel/closure chain. Pixel and
            # decode mathematics are separately bound to prior real CPU gate.
            rows,parsed,preds=synthetic_predictions()
            for label in rt.steps(c,mode)[1:-1]:emit.emit(label,synthetic_only=True)
            publish(out/'RESULT.json',{'status':'PASS','seed':seed,'synthetic':True,'forward_calls':0,'e2e_seconds':1.,'state_RNG_exact':True,'detector_updates':0})
            emit.emit('complete',result=reference(out/'RESULT.json'))
        else:
            require(c['GPU_authorized'] and c['science_authorized'],'production authority absent')
            from sixth_view_gpu import run
            run(c,seed,mode,out,emit)
        require(not c['synthetic'] or not torch.cuda.is_initialized(),'synthetic CUDA mutated')
        publish(channel.receipt_root(out)/'worker-exit.json',{'status':'PASS','success':True,'pid':os.getpid()})
    except BaseException as error:
        failure={'status':'STOP','error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False,'seed':seed,'mode':mode}
        publish(out/'STOP.json',failure);publish(channel.receipt_root(out)/'failure.json',failure)
        publish(channel.receipt_root(out)/'worker-exit.json',{'status':'STOP','success':False,'pid':os.getpid()})
        raise


def export(c):
    root=Path(c['job_root']);names={'RESULT.json','CLOSED.json','STOP.json','COMPARISON.json','COST.json','RESTORE.json','DATA_ADMISSION.json','execution-binding.json'}
    files=[]
    for p in sorted(root.rglob('*.json')):
        # Full official metric tensors/raw/predictions and runtime remain remote.
        if p.name in names and 'runtime-archive' not in p.parts:files.append(reference(p))
    publish(root/'SLIM_EXPORT.json',{'status':'READY_FOR_SLIM_RETURN','files':files,'full_raw_predictions_logs_checkpoint':'REMOTE_ONLY','index_is_not_backup':True})


def run(c):
    root=Path(c['job_root']);publish(root/'CONSUMED.json',{'contract_sha256':c['contract_sha256'],'pid':os.getpid(),'epoch':time.time(),'no_retry':True})
    start=time.monotonic();completed=[]
    try:
        require(c['synthetic'] or (c['GPU_authorized'] and c['science_authorized']),'run authority')
        if not c['synthetic']:
            gate=json.loads(Path(c['CPU_gate']['path']).read_bytes());require(reference(c['CPU_gate']['path'])==c['CPU_gate'] and gate['status']=='PASS' and gate['source_sha256']==c['source_sha256'],'final real entry CPU gate differs')
            rt.parent_context(c)
        for seed in c['seeds']:
            results={}
            for mode in rt.PHASES[:4]:results[mode]=phase(c,seed,mode)
            costs={v:results['accept-'+v]['e2e_seconds']/results['accept-A']['e2e_seconds'] for v in ('C','center','flip')}
            costpass=all(costs[v]<=6 for v in ('center','flip'))
            publish(root/f'seed{seed}'/'COST.json',{'status':'PASS' if costpass else 'FAIL_CLOSED','ratios_over_A':costs,'limit':6.,'all_real_views_included':not c['synthetic']})
            if not costpass:
                completed.append({'seed':seed,'cost_passed':False,'quality':'NOT_EVALUATED','reason':'fixed_cost_gate_failed'});break
            for mode in rt.PHASES[4:]:results[mode]=phase(c,seed,mode)
            decision=read(root/f'seed{seed}'/'evaluation/COMPARISON.json')
            permits=True if c['synthetic'] else decision['quality_permits_next_seeds']
            completed.append({'seed':seed,'cost_passed':True,'quality':decision,'evaluation':reference(root/f'seed{seed}'/'evaluation/RESULT.json')})
            if seed==0 and not permits:break
            if c['synthetic']:break
        publish(root/'RESULT.json',{'status':'COMPLETE','completed':completed,'synthetic':c['synthetic'],
            'fixed_candidate_status':'SYNTHETIC_ONLY' if c['synthetic'] else ('PASS' if len(completed)==3 and all(x['cost_passed'] and x['quality']['quality_permits_next_seeds'] for x in completed) else 'FAIL_CLOSED'),
            'elapsed_seconds':time.monotonic()-start,'detector_updates':0,'training':False,'automatic_retry':False})
        publish(root/'CLOSED.json',{'status':'PASS','result':reference(root/'RESULT.json')});export(c);return 0
    except BaseException as error:
        publish(root/'STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-start,'completed':completed,'automatic_retry':False})
        export(c);return 2


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','supervise','worker']);p.add_argument('--contract',required=True);p.add_argument('--seed',type=int,choices=[0,1,2]);p.add_argument('--mode',choices=rt.PHASES);a=p.parse_args()
    c=rt.load(a.contract)
    if a.action=='start':
        root=Path(c['job_root']);require(not (root/'LAUNCH.json').exists() and not (root/'CONSUMED.json').exists(),'single launch already used')
        publish(root/'LAUNCH.json',{'command':command(c,'run'),'contract_sha256':c['contract_sha256'],'epoch':time.time()})
        with (root/'chain.log').open('xb') as log:
            child=subprocess.Popen(command(c,'run'),env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        from sparse_rtdetr.localization.guard import identity
        publish(root/'LAUNCH_RECEIPT.json',{'pid':child.pid,'identity':identity(child.pid),'contract_sha256':c['contract_sha256'],'epoch':time.time()})
        print(json.dumps(read(root/'LAUNCH_RECEIPT.json')));return 0
    if a.action=='run':return run(c)
    require(a.seed is not None and a.mode is not None,'seed/mode required')
    if a.action=='supervise':return supervise_phase(c,a.seed,a.mode)
    worker(c,a.seed,a.mode);return 0


if __name__=='__main__':raise SystemExit(main())
