"""Future GPU acceptance only. Formal training is intentionally not exposed."""
import gc,random,time
from pathlib import Path
import numpy as np,torch
from ..baseline.training_v2b import _optimizer
from ..localization import worker as base,channel
from ..localization.runio import require,write,publish,event,digest,read
from .model import Config,build,CommonEMA,ARM,PRETRAINED_SHA
from .engine import Engine,tensor_digest
from .data import dataset,training_loader,input_record
from . import checkpoint,transport

def make_engine(c,runtime,scope='gpu_R18_acceptance'):
    conf=Config(seed=c['seed']);conf.validate();require(type(c['config']['seed']) is int and conf.binding()==c['config'],'new native recipe differs')
    model,criterion,info=build(conf,pretrained_path=c['pretrained']['path'])
    initial=tensor_digest(model.state_dict().items())
    model.to('cuda:0');criterion.to('cuda:0')
    engine=Engine(model,criterion,_optimizer(model,conf),CommonEMA(model),conf)
    binding={'arm':ARM,'scope':scope,'run_id':c['task_id'],'config':conf.binding(),'model':info,'runtime':runtime,
        **{k:c[k] for k in ('source_sha256','contract_sha256','data_sha256')},'pretrained_sha256':PRETRAINED_SHA}
    random.seed(conf.seed);np.random.seed(conf.seed);torch.random.default_generator.manual_seed(conf.seed);torch.cuda.default_generators[0].manual_seed(conf.seed)
    require(not engine.optimizer.state and engine.completed_updates==engine.ema.updates==0,'GPU acceptance did not initialize fresh')
    return engine,binding,initial

def save_state(e,b,out,name):
    event(out,'checkpoint_begin',name=name,update=e.completed_updates,completed_epochs=e.completed_updates//304)
    cp=checkpoint.save(e,b,out/name)
    event(out,'checkpoint_complete',name=name,update=e.completed_updates,completed_epochs=e.completed_updates//304,checkpoint=cp)
    return cp

def acceptance(c,out,holder):
    require(c['GPU_execution_authorized'] is True and c['workload']=={'unique_updates':2,'actual_updates':3,'sequence':[1,2,2]},'GPU authority/workload differs')
    # The new output/runtime directory must be exclusive and cannot be relaunched.
    write(out/'single-launch-consumed.json',{'task_id':c['task_id'],'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256']})
    runtime=base.prepare_cuda(c,out);data=dataset(c);e,b,initial=make_engine(c,runtime);write(out/'binding.json',b)
    holder.update(engine=e,binding=b)
    results=[];start=time.monotonic();cp=None
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=1,window_limit=2)):
        inp=input_record(images,targets,traces);observers=inp.pop('spawn_workers')
        holder['input']={'images':images,'targets':targets,'trace':inp}
        require(inp['ids']==c['fixed_ids'][win],'GPU frozen input order differs')
        base.live_admission(out,c);r=e.train_window(images,targets,capture=True);r['input']=inp;r['input_sha256']=digest(inp)
        require(r['update']==win+1 and e.completed_updates==e.ema.updates==win+1 and e.phase==0,'native full update clock differs')
        results.append(r);transport.save_audit(out,channel.binding(),r,'direct')
        write(out/f'spawn-window-{win:02d}.json',observers)
        if win==0:cp=save_state(e,b,out,'C1-native-R18.pth')
    require(len(results)==2 and e.completed_updates==2,'fixed workload incomplete')
    direct=save_state(e,b,out,'C2-native-R18-direct.pth');expected=checkpoint.read(direct['path'],direct['sha256'],b)
    holder.pop('engine');del e;gc.collect();torch.cuda.empty_cache()
    clone,new,initial2=make_engine(c,runtime);require(new==b and initial==initial2,'independent reconstruction differs')
    holder['engine']=clone
    restored=checkpoint.restore(clone,b,cp['path'],cp['sha256']);write(out/'RESTORE_C1.json',restored);count=0
    for images,targets,traces in training_loader(data,epoch=1,completed_windows=1,window_limit=1):
        inp=input_record(images,targets,traces);observers=inp.pop('spawn_workers')
        holder['input']={'images':images,'targets':targets,'trace':inp}
        require(inp['ids']==c['fixed_ids'][1],'GPU replay input order differs')
        base.live_admission(out,c);replay=clone.train_window(images,targets,capture=True);replay['input']=inp;replay['input_sha256']=digest(inp);count+=1
    require(count==1,'GPU replay scope differs')
    require({k:v for k,v in results[-1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'GPU input/DN/output/loss/raw-gradient exact replay failed')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,b)),'GPU complete state/optimizer/EMA/RNG exact replay failed')
    transport.save_audit(out,channel.binding(),replay,'replay');last=save_state(clone,b,out,'C2-native-R18-replay.pth')
    publish(out/'RESULT.json',{'status':'PASS','arm':ARM,'actual_updates':3,'unique_updates':2,'sequence':[1,2,2],
        'initial_model_sha256':initial,'restore':restored,'same_next_result_exact':True,'complete_state_RNG_exact':True,
        'checkpoint_receipts':[cp,direct,last],'state_usable_for_formal':False,'elapsed_seconds':time.monotonic()-start,
        'complete_update_seconds':[x['elapsed_seconds'] for x in results]+[replay['elapsed_seconds']],
        'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved(),
        'formal_training_started':False})

def cpu_state(c,out,holder):
    """Actual native CPU execution to the same audit/channel/checkpoint consumers."""
    require(not torch.cuda.is_initialized(),'CPU lifecycle initialized CUDA')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    conf=Config(seed=c['seed'],input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    def construct():
        model,criterion,info=build(conf,pretrained_path=c['pretrained']['path'],synthetic=True)
        engine=Engine(model,criterion,_optimizer(model,conf),CommonEMA(model),conf)
        b={'arm':ARM,'scope':'cpu_R18_synthetic','run_id':c['task_id'],'config':conf.binding(),'model':info,
            **{k:c[k] for k in ('source_sha256','contract_sha256','data_sha256')}}
        random.seed(conf.seed);np.random.seed(conf.seed);torch.random.default_generator.manual_seed(conf.seed)
        return engine,b
    e,b=construct();write(out/'binding.json',b)
    holder.update(engine=e,binding=b)
    gen=torch.Generator().manual_seed(713);inputs=[]
    for win in range(2):
        images=torch.rand(2,3,128,128,generator=gen)
        targets=[{'boxes':torch.tensor([[.4,.4,.15,.2]]),'labels':torch.tensor([1]),'gt_ids':[f'CPUfixture/w{win}/raw0'],
                   'ignore_union':torch.empty(0,4,dtype=torch.float64),'ignore_class_boxes':torch.empty(0,4,dtype=torch.float64),'ignore_class_labels':torch.empty(0,dtype=torch.int64)},
                 {'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.int64),'gt_ids':[],
                   'ignore_union':torch.empty(0,4,dtype=torch.float64),'ignore_class_boxes':torch.empty(0,4,dtype=torch.float64),'ignore_class_labels':torch.empty(0,dtype=torch.int64)}]
        inputs.append((images,targets))
    from .engine import output_digest
    rows=[];started=time.monotonic()
    for win,(images,targets) in enumerate(inputs):
        holder['input']={'images':images,'targets':targets}
        r=e.train_window(images,targets,capture=True);r['input_sha256']=output_digest({'images':images,'targets':targets});rows.append(r)
        transport.save_audit(out,channel.binding(),r,'direct')
        if win==0:cp=save_state(e,b,out,'CPU-C1.pth')
    expected=checkpoint.capture(e,b);direct=save_state(e,b,out,'CPU-C2-direct.pth');clone,new=construct();require(new==b,'CPU independent constructor binding differs')
    holder['engine']=clone
    restored=checkpoint.restore(clone,b,cp['path'],cp['sha256']);write(out/'RESTORE_C1.json',restored)
    images,targets=inputs[1];replay=clone.train_window(images,targets,capture=True);replay['input_sha256']=output_digest({'images':images,'targets':targets})
    require({k:v for k,v in rows[1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'CPU audit exact same next result differs')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,b)),'CPU lifecycle complete state/RNG differs')
    transport.save_audit(out,channel.binding(),replay,'replay');last=save_state(clone,b,out,'CPU-C2-replay.pth')
    publish(out/'RESULT.json',{'status':'PASS','scope':'CPU_synthetic_native_R18','actual_updates':3,'sequence':[1,2,2],
        'full_state_RNG_exact':True,'result_input_DN_gradient_exact':True,'checkpoints':[cp,direct,last],
        'elapsed_seconds':time.monotonic()-started,'CUDA_initialized':False,'state_for_formal':False})

def full_clock(e,r,u):
    require(e.completed_updates==e.ema.updates==r['update']==u and e.phase==0 and r['optimizer_steps']==r['ema_updates']==1 and r['finite'],'native full clock differs')
    for losses,diag in zip(r['microbatch_losses'],r['normalizers']):
        require(len(losses) in (12,21) and len(diag)==len(losses)//3 and all(d['denominator']>=1 for d in diag),'native loss family/denominator differs')

def execute_window(e,images,targets,inp,out,holder,c,*,capture=False,admit=True):
    holder['input']={'images':images,'targets':targets,'metadata':inp}
    if admit:base.live_admission(out,c)
    r=e.train_window(images,targets,capture=capture);full_clock(e,r,e.completed_updates)
    r['input']=inp;r['input_sha256']=digest(inp)
    transport.save_audit(out,channel.binding(),r,'direct');return r

def formal(c,out,holder):
    require(c['formal_training_authorized'] is True,'formal authority absent')
    short=read(Path(c['job_root'])/'gpu-acceptance/RESULT.json')
    require(short['status']=='PASS' and short['actual_updates']==3 and short['sequence']==[1,2,2] and short['same_next_result_exact'] and short['complete_state_RNG_exact'] and short['state_usable_for_formal'] is False,'native acceptance gate not exact')
    runtime=base.prepare_cuda(c,out);e,b,initial=make_engine(c,runtime,'formal_R18_seed0_e120');holder.update(engine=e,binding=b)
    require(initial==short['initial_model_sha256'] and e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'formal independent untrained native initial state differs')
    write(out/'binding.json',b);zero=save_state(e,b,out,'checkpoint-epoch-000.pth')
    data=dataset(c);started=time.monotonic();times=[];last=None
    for epoch in range(1,121):
        from .control import load
        load(c['contract_path']);require(e.completed_updates==(epoch-1)*304,'native epoch start clock')
        plan=base.plan_epoch(data,epoch);write(out/f'input-plan-epoch-{epoch:03d}.json',plan)
        event(out,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,completed_updates=e.completed_updates,dropped_ids=plan['ids'][4864:])
        t=time.monotonic();count=0
        for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
            inp=input_record(images,targets,traces);observers=inp.pop('spawn_workers')
            require(inp['ids']==plan['ids'][win*16:(win+1)*16] and all(x['epoch']==epoch and x['position']==win*16+i for i,x in enumerate(traces)),'formal data order/cursor differs')
            r=execute_window(e,images,targets,inp,out,holder,c);full_clock(e,r,(epoch-1)*304+win+1);count+=1
        require(count==304 and e.completed_updates==epoch*304,'native epoch fixed work incomplete')
        last=save_state(e,b,out,f'checkpoint-epoch-{epoch:03d}.pth');elapsed=time.monotonic()-t;times.append(elapsed)
        publish(out/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':e.completed_updates,'checkpoint':last,'epoch_wall_seconds':elapsed,'strong_augmentation':epoch<118})
    require(e.completed_updates==e.ema.updates==36480 and checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],b),checkpoint.capture(e,b)),'native full e120 endpoint differs')
    endpoint={'status':'R18_COMMON_E120_EMA_LOCKED','arm':ARM,'seed':0,'epoch':120,'updates':36480,'actual_updates':36480,'checkpoint':last,
        'initial_checkpoint':zero,'initial_model_sha256':initial,'binding':__import__('sparse_rtdetr.localization.runio',fromlist=['reference']).reference(out/'binding.json'),
        'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'used_acceptance_state':False,'used_R0_or_v2a_state':False,
        'training_wall_seconds':time.monotonic()-started,'epoch_wall_seconds':times,'evaluation_before_endpoint':False,'holdout_accessed':False}
    publish(out/'endpoint-lock.json',endpoint);publish(out/'RESULT.json',endpoint)

def inference(c,out,holder):
    from ..r0.evaluation_data import Development548
    from ..localization.runio import reference
    from ..localization.consumer_geometry import positive_box
    from .geometry import forward_guards
    endpoint=read(Path(c['job_root'])/'formal/endpoint-lock.json');require(endpoint['status']=='R18_COMMON_E120_EMA_LOCKED','native endpoint not locked')
    binding=read(Path(c['job_root'])/'formal/binding.json');require(reference(Path(c['job_root'])/'formal/binding.json')==endpoint['binding'],'endpoint binding SHA differs')
    runtime=base.prepare_cuda(c,out);e,b,_=make_engine(c,runtime,'formal_R18_seed0_e120');holder.update(engine=e,binding=b)
    require(b==binding,'native endpoint reconstruction differs');cp=endpoint['checkpoint']
    restored=checkpoint.restore(e,b,cp['path'],cp['sha256']);require(e.completed_updates==e.ema.updates==36480,'not fixed e120')
    model=e.ema.module;model.eval();before=checkpoint.capture(e,b);data=Development548(c['data']['development']);logits=[];boxes=[];ids=[];calls=0
    with torch.inference_mode():
        for start in range(0,548,4):
            rows=data.records[start:start+4];pixels=[data.item(row,pos)[0] for pos,row in enumerate(rows,start)]
            holder['input']=[{'image_id':r['stable_image_id'],'image_sha256':r['image_sha256'],'annotation_sha256':r['annotation_sha256']} for r in rows]
            base.live_admission(out,c)
            with torch.autocast('cuda',enabled=False),forward_guards(model):result=model(torch.stack(pixels).to('cuda'))
            require(set(result)=={'pred_logits','pred_boxes'} and result['pred_logits'].dtype==result['pred_boxes'].dtype==torch.float32 and result['pred_logits'].shape==(4,300,10) and result['pred_boxes'].shape==(4,300,4),'native endpoint output schema')
            require(torch.isfinite(result['pred_logits']).all(),'endpoint logits nonfinite');positive_box('native/e120/pred_boxes',result['pred_boxes'])
            calls+=1;logits.append(result['pred_logits'].cpu());boxes.append(result['pred_boxes'].cpu())
            for pos,row in enumerate(rows,start):
                ids.append(row['stable_image_id']);event(out,'evaluation_image_saved',evaluation_image_number=pos+1,image_id=row['stable_image_id'],checkpoint_sha256=cp['sha256'])
    require(calls==137 and len(ids)==len(set(ids))==548 and checkpoint.exact_equal(before,checkpoint.capture(e,b)),'native inference count or complete state/RNG changed')
    raw={'schema':'native_r18_common_raw_queries_v1','arm':ARM,'epoch':120,'EMA':True,'image_ids':ids,'pred_logits':torch.cat(logits),'pred_boxes':torch.cat(boxes),'query_ids':torch.arange(300),
        'checkpoint_sha256':cp['sha256'],'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'evaluation_protocol_sha256':c['evaluation_protocol_sha256']}
    path=out/'raw-queries-native-R18-e120.pth'
    with path.open('xb') as f:torch.save(raw,f);f.flush();__import__('os').fsync(f.fileno())
    publish(out/'RESULT.json',{'status':'PASS','images':548,'model_batch_forwards':137,'batch_size':4,'FP32':True,'complete_parent_EMA_optimizer_RNG_unchanged':True,
        'checkpoint_sha256':cp['sha256'],'raw':reference(path),'restore':restored,'holdout_accessed':False})

def cpu_execution(c,out,holder):
    """Actual new formal-window/audit plumbing on synthetic CPU scope, never e120."""
    require(not torch.cuda.is_initialized(),'CPU execution initialized CUDA');torch.set_num_threads(2)
    conf=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    model,criterion,info=build(conf,pretrained_path=c['pretrained']['path'],synthetic=True)
    e=Engine(model,criterion,_optimizer(model,conf),CommonEMA(model),conf)
    b={'arm':ARM,'scope':'cpu_R18_synthetic','config':conf.binding(),'model':info,**{k:c[k] for k in ('source_sha256','contract_sha256','data_sha256')}}
    holder.update(engine=e,binding=b);write(out/'binding.json',b);initial=save_state(e,b,out,'CPU-E000.pth');gen=torch.Generator().manual_seed(891)
    targets=[{'boxes':torch.tensor([[.4,.4,.15,.2]]),'labels':torch.tensor([1]),'gt_ids':['synthetic/native/formal-interface/row0']},
        {'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.int64),'gt_ids':[]}]
    for win in range(2):
        images=torch.rand(2,3,128,128,generator=gen);inp={'scope':'CPU_SYNTHETIC_ONLY','epoch':1,'window':win,'GT_ids':[t['gt_ids'] for t in targets],'pixel_sha256':tensor_digest([('input',images)])}
        execute_window(e,images,targets,inp,out,holder,c,capture=True,admit=True)
    last=save_state(e,b,out,'CPU-C2.pth');require(checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],b),checkpoint.capture(e,b)),'CPU formal-interface complete state differs')
    publish(out/'RESULT.json',{'status':'PASS','scope':'CPU_SYNTHETIC_ONLY','updates':2,'EMA':2,'window_audit_and_exact_disk_state':True,'initial':initial,'final':last,'GPU_calls':0,'e120_lock_created':False})
