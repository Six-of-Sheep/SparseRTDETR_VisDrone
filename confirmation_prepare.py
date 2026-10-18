"""Preparation-only CLI: real development inputs, never AP or sealed content.

The legacy verifier is an immutable, explicitly referenced dependency. Its
context returns a data SPECIFICATION; only parsed_gt(rows, data_spec) parses GT.
There is intentionally no inference, training, unsealing or score command.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from sparse_rtdetr.localization.runio import read, require, reference, digest, publish, event

TASK = 'sahi-independent-confirmation-20261018-001'
PROTOCOL = 'original-C-confirmation-preparation-v1'
LABELS = ['begin', 'context', 'parsed_gt'] + [f'input-{a}-{b}' for a in 'ABC' for b in (600, 300)] + ['variable_batches', 'complete']


def exact(ref):
    require(reference(ref['path']) == ref, 'immutable reference differs: ' + ref['path'])
    return read(ref['path'])


def validate_parents(parents,expected):
    require([p['seed'] for p in parents]==[0,1,2], 'three real seeds in frozen order')
    require([p['checkpoint_sha256'] for p in parents]==expected and len(set(expected))==3, 'plan checkpoint/seed differs')
    require(all(p['identity']['checkpoint_sha256']==p['checkpoint_sha256'] for p in parents), 'identity checkpoint conflict')


def batches(rows, size=4):
    require(type(size) is int and size > 0, 'invalid batch size')
    return [rows[k:k + size] for k in range(0, len(rows), size)]


def members(rows, ids):
    require(type(rows) is list and type(ids) is list and len(rows) == len(ids) > 0, 'member count/type')
    require(len(set(ids)) == len(ids) and [r['stable_image_id'] for r in rows] == ids, 'missing/duplicate/reordered member')


def validate_parsed(rows, parsed):
    require(type(parsed) is list and len(parsed) == len(rows), 'parsed GT list required, data spec is not GT')
    from sparse_rtdetr.data_protocol.schema import stable_annotation_id
    seen = set()
    for row, raw in zip(rows, parsed):
        require(type(raw) is dict and set(raw) == {'gt','spatial_ignore','class_ignore','rows','counts'}, 'parsed field structure')
        require(len(raw['rows']) == row['raw_annotation_rows'], 'physical GT row mismatch')
        for o in raw['rows']:
            require(o['id'] == stable_annotation_id(row['stable_image_id'], o['physical_line'], o['raw_line_sha256']), 'GT image/line identity')
            require(o['id'] not in seen, 'duplicate GT identity'); seen.add(o['id'])
            require(type(o['category']) is int and type(o['score']) is float and o['score'] in (0.,1.), 'GT category/score')
            require(len(o['box']) == 4 and all(math.isfinite(v) for v in o['box']) and o['box'][2] > o['box'][0] and o['box'][3] > o['box'][1], 'GT geometry')
        for name, role in (('gt','valid_gt'),('spatial_ignore','spatial_ignore'),('class_ignore','class_ignore')):
            require(raw[name] == [o for o in raw['rows'] if o['role'] == role], 'raw-ignore partition')
    return digest(parsed)


def evaluator_inputs(rows, parsed, predictions, budget):
    """Actual certified input constructors, without invoking either evaluator."""
    members(rows, [r['stable_image_id'] for r in rows]); validate_parsed(rows, parsed)
    require(budget in (600,300), 'output budget')
    ids = [r['stable_image_id'] for r in rows]; counts = dict.fromkeys(ids, 0)
    for p in predictions:
        require(p['image_id'] in counts, 'prediction outside members')
        counts[p['image_id']] += 1
        require(type(p['category_id']) is int and p['category_id'] in range(1,11) and math.isfinite(p['score']) and 0 <= p['score'] <= 1, 'prediction value')
        box=p['bbox_xyxy'];require(len(box)==4 and all(math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1], 'prediction geometry')
    require(all(v <= budget for v in counts.values()), 'budget before ignore')
    # Zero detections for a member are valid. Membership comes from dataset rows.
    from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs
    from sparse_rtdetr.baseline.primary_evaluator import _input_payload
    from sparse_rtdetr.data_protocol.schema import canonical_json_bytes
    primary = primary_inputs(rows, parsed, predictions)
    dataset, filtered, spatial = bridge_inputs(rows, parsed, predictions)
    require([i.image_id for i in primary.images] == ids and len(dataset['images']) == len(ids), 'dual input membership')
    require(len(primary.ground_truth) == sum(len(r['rows']) for r in parsed), 'raw GT lost')
    require(spatial['detections_before'] == len(predictions) and spatial['detections_after'] == len(filtered) and spatial['GT_after'] == len(dataset['annotations']), 'before/after stage counts')
    return {'members':len(ids), 'ordered_ids_sha256':digest(ids),
            'canonical_input_sha256':hashlib.sha256(canonical_json_bytes(_input_payload(primary))).hexdigest(),
            'counts':spatial, 'primary_GT_including_ignore':len(primary.ground_truth), 'AP_scored':False}


def cpu_only():
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only boundary')
    torch.set_num_threads(1)
    def forbidden(*a, **kw):raise RuntimeError('preparation forbids model/CUDA/backward/AP')
    torch.cuda._lazy_init = forbidden
    torch.nn.Module.__init__ = forbidden
    torch.autograd.backward = forbidden
    from sparse_rtdetr.r0 import evaluation
    evaluation.score_budget = forbidden
    evaluation.bridge_metrics = forbidden
    evaluation.primary.evaluate_primary_v1 = forbidden


def install():
    from sparse_rtdetr.protocol_r18.control import install_resources
    install_resources()
    from sparse_rtdetr.localization import channel
    if getattr(channel, '_confirmation_preparation_installed', False):return
    native = channel.validate_event; NativeReader = channel.Reader
    def validate(p,b,stream):
        if b.get('progress_protocol') != PROTOCOL:return native(p,b,stream)
        require(b['task_id']==TASK and type(b['seed']) is int and b['seed'] in (0,1,2), 'true seed/task identity')
        require(all(channel._hex64(b[k]) for k in ('source_sha256','contract_sha256','checkpoint_sha256')), 'immutable identity')
        require(type(p.get('monotonic')) in (int,float) and math.isfinite(p['monotonic']), 'event clock')
        if stream != 'progress':return None
        require(p['event']=='confirmation_prepare_step' and type(p.get('step')) is int and 0<=p['step']<len(LABELS), 'event schema')
        require(p['label']==LABELS[p['step']] and p['seed']==b['seed'] and p['checkpoint_sha256']==b['checkpoint_sha256'], 'event identity/order')
        return digest([p['event'],p['step']])
    class Reader(NativeReader):
        def _accept(self,r):
            ours = self.bound_identity.get('progress_protocol')==PROTOCOL and self.stream=='progress'
            if ours and r['seq'] not in self.seen:require(r['payload']['step']==self.clock.get('_confirmation_next',0), 'missing/repeated/out-of-order step')
            changed=super()._accept(r)
            if ours and changed:self.clock['_confirmation_next']=self.clock.get('_confirmation_next',0)+1
            return changed
    channel.validate_event=validate;channel.Reader=Reader;channel._confirmation_preparation_installed=True


def load(path):
    install();c=read(path)
    require(c['task_id']==TASK and c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}), 'contract identity')
    require(c['scope']=='development_CPU_inputs_only' and c['GPU_authorized'] is False and c['science_authorized'] is False and c['sealed_access_authorized'] is False, 'scope not preparation')
    require([p['seed'] for p in c['parents']]==[0,1,2] and len({p['checkpoint_sha256'] for p in c['parents']})==3, 'three distinct original seed parents')
    require(c['source_root']==str(ROOT), 'source root differs')
    for r in exact(c['source_manifest'])['files']:
        p=ROOT/r['path'];require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(), 'source path')
        require(reference(p)==dict(path=str(p),bytes=r['bytes'],sha256=r['sha256']), 'source bytes differ: '+r['path'])
    plan=exact(c['plan']);exact(c['legacy_contract'])
    validate_parents(c['parents'],plan['checkpoint_sha256'])
    return c


def closed_stream(runtime,binding):
    from sparse_rtdetr.localization import channel
    reader=channel.Reader(Path(runtime)/'progress.jsonl',binding,'progress');last,_=reader.poll()
    require(reader.clock.get('_confirmation_next')==len(LABELS) and last is not None and last['payload']['label']=='complete','cannot close incomplete protocol')
    return last


def legacy(c):
    p=exact(c['legacy_contract']);source=Path(p['source_root'])
    for r in p['source_files']:
        require(reference(source/r['path'])==dict(path=str(source/r['path']),bytes=r['bytes'],sha256=r['sha256']), 'certified dependency source differs')
    sys.path.insert(0,str(source));import softnms_job as old
    bound=old.load_contract(c['legacy_contract']['path']);specs,rows,data_spec=old.context(bound)
    sys.path.insert(0,str(ROOT))
    require([s['seed'] for s in specs]==[0,1,2], 'legacy seed coverage')
    for s,p in zip(specs,c['parents']):
        require(s['seed']==p['seed'] and s['identity']==p['identity'] and s['identity']['checkpoint_sha256']==p['checkpoint_sha256'], 'cross-seed/checkpoint conflict')
    return old,specs,rows,data_spec


def worker(c,seed):
    cpu_only()
    from sparse_rtdetr.localization import channel, guard
    runtime=channel.binding();out=Path(c['output'])/f'seed{seed}'
    require(runtime is not None and runtime['seed']==seed and runtime['checkpoint_sha256']==c['parents'][seed]['checkpoint_sha256'], 'worker identity')
    require(os.getppid()==runtime['supervisor_pid'] and guard.identity(os.getppid())['start']==runtime['supervisor_start'], 'worker owner lifecycle')
    publish(out/'execution-binding.json',runtime);n=0
    def emit(label):
        nonlocal n
        require(n<len(LABELS) and LABELS[n]==label,'writer missing/order/repeated terminal')
        event(out,'confirmation_prepare_step',step=n,label=label,seed=seed,checkpoint_sha256=runtime['checkpoint_sha256']);n+=1
    try:
        emit('begin')
        if c.get('inject_worker_failure'):raise RuntimeError('EXPECTED_CPU_WORKER_EXCEPTION')
        old,specs,rows,data_spec=legacy(c);spec=specs[seed];members(rows,spec['ids']);emit('context')
        # Do not instantiate Development548: it also builds torchvision Module
        # transforms, which are unnecessary for an annotation-only CPU gate.
        parsed=old.parsed_gt(rows,data_spec);gt_sha=validate_parsed(rows,parsed);emit('parsed_gt');proofs=[];variable=[]
        for arm in 'ABC':
            for budget in (600,300):
                predictions=exact(spec['evaluation']['predictions'][arm][str(budget)])
                new=evaluator_inputs(rows,parsed,predictions,budget)
                original=old.evaluation_input_proof(spec,rows,parsed,predictions,arm,budget)
                require(new['canonical_input_sha256']==original['canonical_input_sha256'] and new['counts']==original['counts'], 'real canonical input/filter counts differ')
                proofs.append({'arm':arm,'budget':budget,**new,'original_proof':original})
                if arm=='A' and budget==600:
                    for count in (1,3,4,5,547,548):
                        rs=rows[:count];ids={r['stable_image_id'] for r in rs};ps=[p for p in predictions if p['image_id'] in ids]
                        v=evaluator_inputs(rs,parsed[:count],ps,600)
                        chunks=batches(rs);require([r for b in chunks for r in b]==rs and all(0<len(b)<=4 for b in chunks), 'partial final batch')
                        variable.append({'images':count,'batches':len(chunks),'last_batch':len(chunks[-1]),'inputs':v})
                emit(f'input-{arm}-{budget}')
        require(not batches([]) and [len(b) for b in batches(list(range(1602)))][-1]==2,'empty/1602 scheduling')
        publish(out/'REAL_INPUTS.json',{'seed':seed,'parsed_GT_sha256':gt_sha,'proofs':proofs,'variable_counts':variable,'future_1602_schedule_only':{'batches':401,'last_batch':2,'sealed_content_used':False}})
        emit('variable_batches')
        publish(out/'RESULT.json',{'status':'PASS','seed':seed,'proofs':reference(out/'REAL_INPUTS.json'),'AP_calls':0,'forward_calls':0,'model_constructions':0,'training_updates':0,'CUDA_initialized':False,'sealed_content_opened':False})
        emit('complete')
    except BaseException as e:
        publish(channel.receipt_root(out)/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:
        publish(channel.receipt_root(out)/'worker-exit.json',{'seed':seed,'pid':os.getpid(),'epoch':time.time(),'CUDA_initialized':False})


def export(out):
    names=['RESULT.json','CLOSED.json','STOP.json']
    refs=[reference(out/n) for n in names if (out/n).is_file()]
    for seed in (0,1,2):
        for n in ('REAL_INPUTS.json','RESULT.json','CLOSED.json','runtime-archive-index.json'):
            p=out/f'seed{seed}'/n
            if p.is_file():refs.append(reference(p))
    publish(out/'SLIM_EXPORT.json',{'status':'READY_FOR_SLIM_RETURN','files':refs,'large_materials':'only remote','index_is_not_backup':True})


def run(c,path):
    cpu_only();out=Path(c['output']);start=time.monotonic()
    require(not (out/'CONSUMED.json').exists(),'attempt already consumed')
    publish(out/'CONSUMED.json',{'contract_sha256':c['contract_sha256'],'no_retry':True})
    from sparse_rtdetr.localization import guard,channel
    try:
        for seed in (0,1,2):
            parent=guard.identity(os.getpid());root=out/f'seed{seed}';runtime=Path(c['runtime'])/f'seed{seed}'
            identity=dict(task_id=TASK,seed=seed,progress_protocol=PROTOCOL,source_sha256=c['source_manifest']['sha256'],
                contract_sha256=c['contract_sha256'],checkpoint_sha256=c['parents'][seed]['checkpoint_sha256'],
                output_root=str(root),supervisor_pid=os.getpid(),supervisor_start=parent['start'])
            code=guard.supervise([sys.executable,'-B',str(ROOT/'confirmation_prepare.py'),'worker','--contract',path,'--seed',str(seed)],root,dict(os.environ),
                gpu_uuid=None,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gpu=False,checkpoint_bytes=0,formal=False,runtime=runtime,run_identity=identity)
            require(code==0,'supervised CPU worker failed')
            b=read(runtime/'identity.json');channel.verify_archive(runtime,root,b)
            closed_stream(runtime,b)
            publish(root/'CLOSED.json',{'status':'PASS','result':reference(root/'RESULT.json'),'archive':reference(root/'runtime-archive-index.json')})
        publish(out/'RESULT.json',{'status':'CPU_PREPARATION_PASS','seeds':[0,1,2],'real_input_groups':18,'scientific_evaluations':0,'elapsed_seconds':time.monotonic()-start,'qualification':'BLOCKED_PENDING_ACCESS_HISTORY'})
        publish(out/'CLOSED.json',{'status':'PASS','result':reference(out/'RESULT.json')});export(out);return 0
    except BaseException as e:
        publish(out/'STOP.json',{'error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'no_retry':True});export(out);return 2


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','worker']);p.add_argument('--contract',required=True);p.add_argument('--seed',type=int,choices=(0,1,2));a=p.parse_args();c=load(a.contract)
    cpu_only()
    if a.action=='worker':return worker(c,a.seed)
    if a.action=='run':return run(c,a.contract)
    out=Path(c['output']);require(not out.exists(),'output/attempt already exists');out.mkdir(parents=True,exist_ok=False)
    publish(out/'LAUNCH.json',{'contract':reference(a.contract),'pid':os.getpid()})
    code=subprocess.call([sys.executable,'-B',str(ROOT/'confirmation_prepare.py'),'run','--contract',a.contract],env=dict(os.environ))
    publish(out/'EXIT.json',{'returncode':code,'epoch':time.time()});return code


if __name__=='__main__':raise SystemExit(main())
