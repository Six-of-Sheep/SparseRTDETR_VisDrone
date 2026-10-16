"""Original e001 strict restore, then explicit new execution provenance."""
import gc,time
from pathlib import Path
import torch
from . import checkpoint,transport
from .worker import make_engine,save_state,execute_window,full_clock
from .data import dataset,training_loader,input_record
from .engine import tensor_digest,learning_rate
from ..localization import worker as base,channel
from ..localization.runio import require,read,write,publish,reference,digest,sha,event

ALLOWED_CHANGES={'src/sparse_rtdetr/protocol_r18/control.py','src/sparse_rtdetr/protocol_r18/lifecycle.py','src/sparse_rtdetr/protocol_r18/transport.py','src/sparse_rtdetr/protocol_r18/progress_v3.py','src/sparse_rtdetr/protocol_r18/recovery.py','src/sparse_rtdetr/protocol_r18/protocol_cpu.py'}

def parent(c):
    p=c['resume_parent']
    for key in ('binding','contract','manifest','sidecar','STOP'):
        require(reference(p[key]['path'])==p[key],'immutable parent evidence changed: '+key)
    cp=p['checkpoint'];require(reference(cp['path'])=={'path':cp['path'],'bytes':cp['size_bytes'],'sha256':cp['sha256']},'original e001 bytes changed')
    require(read(p['sidecar']['path'])==cp and cp['completed_updates']==304 and cp['scope']=='formal_R18_seed0_e120','original sidecar/cursor differs')
    b=read(p['binding']['path']);old=read(p['contract']['path'])
    require(old['contract_sha256']==digest({k:v for k,v in old.items() if k!='contract_sha256'}),'old contract corruption')
    require(b['source_sha256']==p['manifest']['sha256']==old['source_sha256'] and b['contract_sha256']==old['contract_sha256'] and b['config']==c['config'],'parent source/contract/recipe differs')
    require(b['data_sha256']==old['data_sha256'] and b['pretrained_sha256']==__import__('sparse_rtdetr.protocol_r18.model',fromlist=['PRETRAINED_SHA']).PRETRAINED_SHA and sha(c['pretrained']['path'])==b['pretrained_sha256'],'parent data/pretrain binding differs')
    oldrows={r['path']:r for r in read(p['manifest']['path'])['files']};newrows={r['path']:r for r in read(c['source_manifest'])['files']}
    require(set(oldrows)<=set(newrows),'old source member removed')
    changes={k for k in newrows if k not in oldrows or newrows[k]!=oldrows[k]}
    require(changes<=ALLOWED_CHANGES,'recovery modifies scientific core: '+str(sorted(changes-ALLOWED_CHANGES)))
    for k,r in oldrows.items():require(reference(Path(old['source_root'])/k)=={'path':str(Path(old['source_root'])/k),'bytes':r['bytes'],'sha256':r['sha256']},'parent frozen source changed: '+k)
    return b,cp

def new_binding(c,fresh,role):
    b=dict(fresh);b['recovery']={'original_parent':c['resume_parent']['checkpoint'],'original_binding':c['resume_parent']['binding'],
        'original_cursor':304,'new_execution_role':role,'restore':'unchanged strict checkpoint.restore with exact original binding; rebind only after full_state_exact'}
    return b

def restore_parent(c,runtime,out,holder,role):
    old,cp=parent(c);e,fresh,initial=make_engine(c,runtime,'formal_R18_seed0_e120')
    holder.update(engine=e,binding=old)
    for key in ('arm','scope','run_id','config','model','runtime','pretrained_sha256'):
        require(fresh[key]==old[key],'new construction differs from original: '+key)
    result=checkpoint.restore(e,old,cp['path'],cp['sha256'])
    require(e.completed_updates==e.ema.updates==304 and e.phase==0 and result['full_state_exact'],'original complete restore not exact')
    receipt={**result,'original_binding':c['resume_parent']['binding'],'original_checkpoint':cp,'new_source_sha256':c['source_sha256'],
        'new_contract_sha256':c['contract_sha256'],'original_checkpoint_unchanged':True,'state_usable_for_formal':role=='formal'}
    write(out/'ORIGINAL_E001_STRICT_RESTORE.json',receipt)
    b=new_binding(c,fresh,role);holder['binding']=b;write(out/'binding.json',b)
    # All state bytes except explicit provenance must remain exact after rebinding.
    before=checkpoint.read(cp['path'],cp['sha256'],old);after=checkpoint.capture(e,b)
    for key in before:
        if key not in ('binding','binding_sha256'):require(checkpoint.exact_equal(before[key],after[key]),'rebinding changed full state: '+key)
    event(out,'resume_parent_exact',completed_updates=304,completed_epochs=1,checkpoint_sha256=cp['sha256'],full_state_exact=True)
    return e,b,initial,receipt

def begin_epoch(c,data,e,out,epoch):
    from .control import load
    load(c['contract_path']);require(e.completed_updates==(epoch-1)*304,'restored epoch cursor differs')
    plan=base.plan_epoch(data,epoch)
    if epoch==2:require(plan==read(c['original_epoch2_plan']['path']) and reference(c['original_epoch2_plan']['path'])==c['original_epoch2_plan'],'original epoch2 order changed')
    write(out/f'input-plan-epoch-{epoch:03d}.json',plan)
    event(out,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,completed_updates=e.completed_updates,dropped_ids=plan['ids'][4864:])
    return plan

def accepted_input(images,targets,traces,plan,epoch,win):
    inp=input_record(images,targets,traces);observers=inp.pop('spawn_workers')
    require(inp['ids']==plan['ids'][win*16:(win+1)*16] and all(t['epoch']==epoch and t['position']==win*16+i for i,t in enumerate(traces)),'recovery data order/GT identity differs')
    return inp,observers

def acceptance(c,out,holder):
    require(c['recovery_workload']=={'sequence':[305,306,306],'actual_updates':3,'unique_updates':2,'start':304},'recovery workload expanded')
    runtime=base.prepare_cuda(c,out);data=dataset(c);e,b,initial,restored=restore_parent(c,runtime,out,holder,'acceptance')
    plan=begin_epoch(c,data,e,out,2);rows=[];cp=None;start=time.monotonic()
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=2,window_limit=2)):
        inp,obs=accepted_input(images,targets,traces,plan,2,win)
        r=execute_window(e,images,targets,inp,out,holder,c,capture=True);full_clock(e,r,305+win);rows.append(r);write(out/f'spawn-window-{win}.json',obs)
        if win==0:cp=save_state(e,b,out,'C305-recovery.pth')
    require(len(rows)==2,'recovery unique workload incomplete')
    direct=save_state(e,b,out,'C306-direct.pth');expected=checkpoint.read(direct['path'],direct['sha256'],b)
    holder.pop('engine');del e;gc.collect();torch.cuda.empty_cache()
    clone,fresh,initial2=make_engine(c,runtime,'formal_R18_seed0_e120');new=new_binding(c,fresh,'acceptance');require(new==b and initial2==initial,'replay independent reconstruction differs')
    holder.update(engine=clone,binding=b);result=checkpoint.restore(clone,b,cp['path'],cp['sha256']);write(out/'RESTORE_C305.json',result);n=0
    for images,targets,traces in training_loader(data,epoch=2,completed_windows=1,window_limit=1):
        inp,obs=accepted_input(images,targets,traces,plan,2,1);holder['input']={'images':images,'targets':targets,'metadata':inp}
        base.live_admission(out,c);r=clone.train_window(images,targets,capture=True);full_clock(clone,r,306);r['input']=inp;r['input_sha256']=digest(inp);n+=1
    require(n==1,'replay scope expanded')
    require({k:v for k,v in rows[-1].items() if k!='elapsed_seconds'}=={k:v for k,v in r.items() if k!='elapsed_seconds'},'restored input/DN/output/loss/raw-grad result differs')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,b)),'complete optimizer/EMA/RNG/state differs')
    transport.save_audit(out,channel.binding(),r,'replay');last=save_state(clone,b,out,'C306-replay.pth')
    publish(out/'RESULT.json',{'status':'PASS','sequence':[305,306,306],'actual_updates':3,'unique_updates':2,'start':304,'original_parent':c['resume_parent']['checkpoint'],
        'original_full_restore':restored,'same_next_result_exact':True,'complete_state_RNG_exact':True,'state_usable_for_formal':False,
        'checkpoint_receipts':[cp,direct,last],'elapsed_seconds':time.monotonic()-start,'complete_update_seconds':[x['elapsed_seconds'] for x in rows]+[r['elapsed_seconds']],
        'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved()})

def formal(c,out,holder):
    short=read(Path(c['job_root'])/'gpu-recovery/RESULT.json')
    require(short['status']=='PASS' and short['sequence']==[305,306,306] and short['actual_updates']==3 and short['same_next_result_exact'] and short['complete_state_RNG_exact'] and not short['state_usable_for_formal'] and short['original_parent']==c['resume_parent']['checkpoint'],'new recovery acceptance absent')
    runtime=base.prepare_cuda(c,out);e,b,initial,restored=restore_parent(c,runtime,out,holder,'formal')
    data=dataset(c);start=time.monotonic();times=[];last=None;actual=0
    for epoch in range(2,121):
        plan=begin_epoch(c,data,e,out,epoch);t=time.monotonic();count=0
        for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
            inp,obs=accepted_input(images,targets,traces,plan,epoch,win)
            r=execute_window(e,images,targets,inp,out,holder,c);full_clock(e,r,(epoch-1)*304+win+1);count+=1;actual+=1
        require(count==304 and e.completed_updates==epoch*304,'resumed epoch incomplete')
        last=save_state(e,b,out,f'checkpoint-epoch-{epoch:03d}.pth');elapsed=time.monotonic()-t;times.append(elapsed)
        publish(out/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':e.completed_updates,'checkpoint':last,'epoch_wall_seconds':elapsed,'strong_augmentation':epoch<118})
    require(actual==36176 and e.completed_updates==e.ema.updates==36480 and checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],b),checkpoint.capture(e,b)),'restored e120 endpoint not exact')
    endpoint={'status':'R18_COMMON_E120_EMA_LOCKED','arm':b['arm'],'seed':0,'epoch':120,'updates':36480,'actual_updates':36176,
        'checkpoint':last,'original_parent':c['resume_parent']['checkpoint'],'original_completed_updates':304,'strict_original_restore':restored,
        'initial_model_sha256':initial,'binding':reference(out/'binding.json'),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
        'used_acceptance_state':False,'used_R0_or_v2a_state':False,'training_wall_seconds':time.monotonic()-start,'epoch_wall_seconds':times,'evaluation_before_endpoint':False,'holdout_accessed':False}
    publish(out/'endpoint-lock.json',endpoint);publish(out/'RESULT.json',endpoint)

def cpu_audit(c,out):
    """Production object construction on CPU, no forward and no CUDA init."""
    from .model import Config,build,CommonEMA
    from .engine import Engine
    from ..baseline.training_v2b import _optimizer
    require(not torch.cuda.is_initialized(),'CPU parent audit initialized CUDA')
    b,cp=parent(c);s=checkpoint.read(cp['path'],cp['sha256'],b);cfg=Config();model,criterion,info=build(cfg,pretrained_path=c['pretrained']['path'])
    e=Engine(model,criterion,_optimizer(model,cfg),CommonEMA(model),cfg);old=checkpoint.old
    require(info==b['model'] and s['device']=='cuda:0' and s['torch_version']==str(torch.__version__),'parent production/runtime differs')
    e.validate_state_dict(s['engine']);require(s['engine']['completed_updates']==304 and s['engine']['next_epoch']==2 and s['engine']['next_window']==0,'parent cursor not e002 first window')
    require(checkpoint.behavior_identity(e)==s['behavior'],'nonserialized mathematical behavior changed')
    for label,m,layout,state in [('model',model,s['model_layout'],s['model']),('EMA',e.ema.module,s['ema_layout'],s['ema']['module'])]:
        require(old._model_layout(m)==layout,'parent '+label+' layout differs');old._check_model_state(state,layout,label)
        for k,v in m.state_dict().items():
            if k.rsplit('.',1)[-1] in {'anchors','valid_mask','num_points_scale'}:require(old._equal(state[k],v),'parent immutable cache differs: '+k)
        m.load_state_dict(state,strict=True);require(old._equal(old._safe(m.state_dict()),state),'CPU import bytes differ: '+label)
    layout=old._optimizer_layout(model,e.optimizer);require(layout==s['optimizer_layout'],'optimizer layout differs');old._check_optimizer(s['optimizer'],layout,model,304)
    for g in layout['groups']:
        for n,i in zip(g['names'],g['ids']):require(int(s['optimizer']['state'].get(i,{}).get('step',0))==s['engine']['parameter_optimizer_steps'][n],'optimizer step differs')
    require(s['ema']['updates']==304 and s['ema']['decay']==.9999 and s['ema']['warmups']==2000 and all(g['lr']==learning_rate(304) for g in s['optimizer']['param_groups']),'EMA/LR schedule differs')
    modes=s['model_training'];require(set(modes)=={n for n,_ in model.named_modules()} and all(type(v) is bool for v in modes.values()),'training mode schema differs')
    for n,m in model.named_modules():m.training=modes[n]
    e.optimizer.load_state_dict(s['optimizer']);e.ema.load_state_dict(s['ema']);e.load_state_dict(s['engine'])
    for key,actual in [('optimizer',old._safe(e.optimizer.state_dict())),('ema',old._safe(e.ema.state_dict())),('engine',old._safe(e.state_dict()))]:require(old._equal(actual,s[key]),'CPU import full field differs: '+key)
    rng=s['rng'];cpu=dict(rng);cpu['cuda']=None;checkpoint.validate_rng(cpu,torch.device('cpu'))
    require(isinstance(rng['cuda'],torch.Tensor) and rng['cuda'].dtype==torch.uint8 and tuple(rng['cuda'].shape)==(16,),'CUDA Philox byte structure differs; GPU exact validation still required')
    from .engine import assert_healthy
    assert_healthy(e.model,e.optimizer,e.ema)
    data=dataset(c);plan=base.plan_epoch(data,2);require(plan==read(c['original_epoch2_plan']['path']),'e002 original planned data order differs')
    report={'status':'PASS','scope':'CPU_NO_FORWARD_COMPLETE_E001_STRUCTURE_AND_IMPORT','source_sha256':c['source_sha256'],'parent':cp,'binding':c['resume_parent']['binding'],
        'model_EMA_optimizer_engine_import_exact':True,'cursor':s['engine'],'saved_lr':s['optimizer']['param_groups'][0]['lr'],'model_tensor_count':len(s['model']),
        'all_tensor_count':sum(1 for _ in checkpoint._tensors(s)),'RNG_byte_sha256':{k:__import__('hashlib').sha256(rng[k].numpy().tobytes()).hexdigest() for k in ('torch_cpu','cuda')},
        'CPU_RNG_valid':True,'CUDA_RNG_shape_only':True,'strict_CUDA_full_restore_not_yet_certified':True,'original_epoch2_plan':c['original_epoch2_plan'],
        'forward_calls':0,'CUDA_initialized':torch.cuda.is_initialized(),'parent_bytes_unchanged':reference(cp['path'])['sha256']==cp['sha256']}
    publish(out,report);return report

def inference(c,out,holder):
    # Only explicit provenance differs; unchanged native inference restores the new formal endpoint.
    from . import worker as native
    original=native.make_engine
    def construct(*args,**kwargs):
        e,b,initial=original(*args,**kwargs)
        return e,new_binding(c,b,'formal'),initial
    native.make_engine=construct
    try:return native.inference(c,out,holder)
    finally:native.make_engine=original

def comparison(c,out):
    from . import endpoint
    # Same frozen arithmetic; explicit execution provenance and completion event only.
    result=read(Path(c['job_root'])/'evaluation/RESULT.json');require(result['status']=='PASS' and result['budgets']==[600,300],'evaluation incomplete')
    rows={}
    for budget in (600,300):
        ref=c['R0_metrics'][str(budget)];require(reference(ref['path'])==ref,'R0 metrics changed')
        rows[str(budget)]=endpoint.compare_metrics(result['results'][str(budget)],read(ref['path']))
    value={'status':'PASS','arm':c['arm'],'comparison':'native common protocol minus R0','budgets':rows,'R0_result_reused':True,
        'R0_model_loaded':False,'R0_reinferred':False,'point_comparison_only':True,'holdout_accessed':False,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256']}
    publish(out/'RESULT.json',value)
    cp=read(Path(c['job_root'])/'formal/endpoint-lock.json')['checkpoint']['sha256']
    event(out,'comparison_complete',checkpoint_sha256=cp,budgets=[600,300],result=reference(out/'RESULT.json'))
