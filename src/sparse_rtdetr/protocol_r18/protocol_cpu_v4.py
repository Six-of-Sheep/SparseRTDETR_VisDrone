"""Real CLI/Writer/Reader lifecycle fixtures, no model or CUDA computation."""
import os,time,copy,json,hashlib
from pathlib import Path
from ..localization import channel
from ..localization.runio import event,require,read,write,publish,reference,sha,canonical,digest
from . import transport

def produce(c,out,holder):
    import torch
    require(not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']=='','CPU protocol CUDA scope')
    b=channel.binding();spec=c['CPU_protocol'];checks=[]
    require(b['CPU_protocol_only'] and b['protocol_mode']==spec['mode'],'fixture identity missing')
    channel.resource()
    def reject(label,kind,**fields):
        stream=Path(b['runtime_root'])/'progress.jsonl';before=stream.read_bytes();side=Path(b['runtime_root'])/'progress-writer.json';before_side=side.read_bytes() if side.exists() else None
        try:event(out,kind,**fields)
        except (RuntimeError,ValueError,KeyError):pass
        else:raise RuntimeError('fault accepted: '+label)
        require(stream.read_bytes()==before and (side.read_bytes() if side.exists() else None)==before_side,'Writer refusal mutated stream/sidecar: '+label);checks.append(label)
    if spec['mode'] in ('formal','gpu-recovery'):
        start=spec['starting_cursor'];ep=start//304+1
        reject('wrong_parent','resume_parent_exact',completed_updates=start,completed_epochs=start//304,checkpoint_sha256='0'*64,full_state_exact=True)
        event(out,'resume_parent_exact',completed_updates=start,completed_epochs=start//304,checkpoint_sha256=spec['parent_sha256'],full_state_exact=True)
        dropped=[f'CPU_PROTOCOL/dropped/{i}' for i in range(5)]
        reject('inconsistent_epoch_cursor','epoch_begin',epoch=ep+1,completed_updates=start,completed_epochs=ep,dropped_ids=dropped)
        event(out,'epoch_begin',epoch=ep,completed_updates=start,completed_epochs=ep-1,dropped_ids=dropped)
        reject('real_repeated_operation_conflict','epoch_begin',epoch=ep,completed_updates=start,completed_epochs=ep-1,dropped_ids=[f'conflict/{i}' for i in range(5)])
        # Exact duplicate operation cannot increment semantic completion.
        r=channel.Reader(Path(b['runtime_root'])/'progress.jsonl',b,'progress');r.poll();semantic=r.semantic
        event(out,'epoch_begin',epoch=ep,completed_updates=start,completed_epochs=ep-1,dropped_ids=dropped);r.poll();require(r.semantic==semantic,'duplicate advanced semantic cursor');checks.append('duplicate_no_progress')
        stream=Path(b['runtime_root'])/'progress.jsonl';line=stream.read_bytes().splitlines(keepends=True)[0]
        with stream.open('ab') as f:f.write(line[:len(line)//2]);f.flush();os.fsync(f.fileno())
        before=r.offset;_,advanced=r.poll();require(not advanced and r.semantic==semantic and r.offset==before,'half line advanced cursor')
        with stream.open('ab') as f:f.write(line[len(line)//2:]);f.flush();os.fsync(f.fileno())
        _,advanced=r.poll();require(not advanced and r.semantic==semantic,'duplicate completed line advanced cursor');checks.append('half_line_and_repeated_record_no_progress')
        def audit(u,role='direct'):
            result={'update':u,'input_sha256':digest(['CPU_PROTOCOL_FIXTURE',u]),'loss':0.,'finite':True,'CPU_protocol_fixture':True,'not_a_model_result':True}
            value={**{k:b[k] for k in ('task_id','source_sha256','contract_sha256')},'update':u,'role':role,'result':result}
            path=out/'audits'/f'update-{u:05d}-{role}.json';write(path,value)
            return {'update':u,'audit_role':role,'input_sha256':result['input_sha256'],'audit':reference(path),'loss':0.,'finite':True,'completed_epochs':u//304}
        def checkpoint(u,name):
            event(out,'checkpoint_begin',name=name,update=u,completed_epochs=u//304)
            path=out/name;write(path,{'CPU_PROTOCOL_FIXTURE_ONLY':True,'completed_updates':u,'not_checkpoint_state':True})
            ref=reference(path);cp={'path':str(path),'sha256':ref['sha256'],'size_bytes':ref['bytes'],'completed_updates':u}
            event(out,'checkpoint_complete',name=name,update=u,completed_epochs=u//304,checkpoint=cp)
        gap=audit(start+2);reject('missing_window','update_complete',**gap)
        if spec['mode']=='gpu-recovery':
            event(out,'update_complete',**audit(start+1));checkpoint(start+1,'fixture-C26753.pth')
            # audit306-direct already exists from the gap-refusal fixture; byte-identical reused scientific fixture.
            event(out,'update_complete',**gap);checkpoint(start+2,'fixture-C26754-direct.pth')
            event(out,'update_complete',**audit(start+2,'replay'));checkpoint(start+2,'fixture-C26754-replay.pth')
        else:
            end=spec.get('ending_cursor',36480)
            for u in range(start+1,end+1):
                if u==start+2:p=gap
                else:p=audit(u)
                event(out,'update_complete',**p)
                if u%304==0:
                    checkpoint(u,f'fixture-epoch-{u//304:03d}.pth')
                    if u<end:
                        ep=u//304+1;event(out,'epoch_begin',epoch=ep,completed_updates=u,completed_epochs=ep-1,dropped_ids=dropped)
            reject('epoch_cursor_regression','epoch_begin',epoch=1,completed_updates=0,completed_epochs=0,dropped_ids=dropped)
        reject('unknown_event','unrecognized_native_event')
        reject('wrong_clock','epoch_begin',epoch=120,completed_updates=36176,completed_epochs=119,dropped_ids=dropped,monotonic=-1.)
        # Corrupt bytes and bound identity are independently rejected by actual Reader.
        broken=out/'broken-stream.jsonl';raw=stream.read_bytes();rec=json.loads(raw.splitlines()[0]);rec['sha256']='0'*64;broken.write_bytes(canonical(rec)+b'\n')
        try:channel.Reader(broken,b,'progress').poll()
        except RuntimeError:checks.append('SHA_chain_corruption')
        else:raise RuntimeError('corrupt chain accepted')
        wrong=dict(b);wrong['source_sha256']='0'*64
        try:channel.Reader(stream,wrong,'progress').poll()
        except RuntimeError:checks.append('wrong_run_identity')
        else:raise RuntimeError('wrong identity accepted')
        r.poll();require(r.clock['_v3_cursor']==(start+2 if spec['mode']=='gpu-recovery' else spec.get('ending_cursor',36480)),'CPU final cursor differs')
        checks.append('cross_epoch_checkpoint_and_cursor' if start==0 else 'fixed_recovery_sequence' if spec['mode']=='gpu-recovery' else 'training_endpoint_cursor')
    elif spec['mode']=='inference':
        require(spec['starting_cursor']==36480,'inference before endpoint')
        reject('missing_image','evaluation_image_saved',evaluation_image_number=2,image_id=c['development_ids'][1],checkpoint_sha256=spec['parent_sha256'])
        for n,image in enumerate(c['development_ids'],1):event(out,'evaluation_image_saved',evaluation_image_number=n,image_id=image,checkpoint_sha256=spec['parent_sha256'])
        checks.append('548_endpoint_image_identities')
    elif spec['mode']=='evaluation':
        reject('budget_order','evaluation_budget_complete',budget=300,name='top300',images=548,checkpoint_sha256=spec['parent_sha256'])
        for budget in (600,300):event(out,'evaluation_budget_complete',budget=budget,name='top'+str(budget),images=548,checkpoint_sha256=spec['parent_sha256'])
        checks.append('endpoint_two_budget_transition')
    elif spec['mode']=='comparison':
        from .recovery_e088 import comparison
        comparison(c,out)
        r=read(out/'RESULT.json');require(r['status']=='PASS' and all(row['native_minus_R0_pp']==0. for item in r['budgets'].values() for row in item['COCO_bridge'].values() if row['native_minus_R0_pp'] is not None),'comparison fixture unit differs')
        checks.append('actual_comparison_results_to_progress')
        publish(out/'CPU_PROTOCOL_RESULT.json',{'status':'PASS','CPU_PROTOCOL_FIXTURE_ONLY':True,'source_sha256':c['source_sha256'],'checks':checks,'count':1,'model_constructed':False,'forward_calls':0,'CUDA_initialized':False,'optimizer_updates':0,'formal_launch_token_consumed':False})
        return
    else:raise RuntimeError('unknown CPU fixture phase')
    publish(out/'RESULT.json',{'status':'PASS','CPU_PROTOCOL_FIXTURE_ONLY':True,'source_sha256':c['source_sha256'],'checks':checks,'count':len(checks),
        'model_constructed':False,'forward_calls':0,'CUDA_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'formal_launch_token_consumed':False})
