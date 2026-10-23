"""Fixed H2 U/D training and endpoint chain using the accepted scientific code.

Endpoint/benchmark/scoring follow the certified span_worker algorithms from
a97e1eaf750c69c4ee72af232d07632b1aced931; H2 state and arms remain explicit.
"""
import gc
import hashlib
import math
from pathlib import Path
import random
import statistics
import time
from ..localization.runio import require,read,publish,atomic,reference,digest,sha
from . import h2_formal_contract as ct,h2_worker as short

def setup(c,out,gpu):
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if gpu:
        from .h2_cuda_identity import inspect_cuda_identity
        actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        require(actual==c['environment'],'formal runtime differs')
        proof=inspect_cuda_identity(c['gpu_uuid'],out/'cuda-identity',torch_module=torch)
        require(proof['BF16_supported'],'BF16 unsupported')
        torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory,device=0)
        return torch.device('cuda:0')
    require(not torch.cuda.is_initialized(),'CPU stage initialized CUDA');return torch.device('cpu')

def dataset(c):return short.training_data(c)

def windows(c,data,epoch,completed=0):
    if c['profile']=='LAB_GPU_ONCE':
        from ..localization.data import training_loader
        from ..protocol_r18.data import input_record
        for im,t,tr in training_loader(data,epoch=epoch,completed_windows=completed):
            ident=input_record(im,t,tr);ident.pop('spawn_workers');yield im,t,ident
    else:
        for im,t,ident in short.windows(c,data,completed=epoch-1+completed,limit=ct.profile(c)['windows']-completed):
            yield im,t,dict(ident,epoch=epoch)

def development(c):
    if c['profile']=='LAB_GPU_ONCE':
        from ..r0.evaluation_data import Development548
        d=c['data']['development'];reader=Development548(dict(manifest=d['manifest']['path'],roots=d['roots']))
        ct.exact(d['manifest']);return reader.records,reader
    import torch
    from PIL import Image
    from ..localization.data import parse_raw,RawAugmentation
    rows=[dict(stable_image_id=f'CONSTRUCTED/dev/{i:03d}',width=320,height=256) for i in range(8)]
    class Fixture:
        def raw(self,row):
            raw=b'30,40,12,16,1,1,0,0\n80,80,40,40,1,2,0,0\n120,100,100,100,1,3,0,0\n'
            return parse_raw(raw,row['stable_image_id'],expected_rows=3)
        def item(self,row,position):
            im=Image.new('RGB',(320,256),(31+position,50,71))
            try:return RawAugmentation((128,160))(im,self.raw(row),epoch=120,seed=0,position=position,image_id=row['stable_image_id'],training=False)[:2]
            finally:im.close()
    return rows,Fixture()

def prepare(c,out):
    ct.acceptance(c);ct.historical(c,physical=True)
    gate=_prepare_training(c,out)
    rows,r=development(c)
    for i,row in enumerate(rows):r.item(row,i)
    ids=ct.fixed_ids(c)
    require(len(ids)==ct.profile(c)['fixed_count'] and len(set(ids))==len(ids) and set(ids)<={r['stable_image_id'] for r in rows},'fixed64 development members differ')
    publish(Path(out)/'FORMAL_MATERIAL_PASS.json',dict(status='PASS',development_members=len(rows),fixed_ids=ids,N_status=c['historical_N']['status'],external_acceptance_reused=True,model_forward_calls=0))
    return gate

def binding(c,arm,info,conf):
    return dict(arm=arm,scope='formal_H2_seed0_e120' if c['profile']=='LAB_GPU_ONCE' else 'cpu_H2_synthetic',config=conf.binding(),model=info,
        source_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],data_sha256=digest(c['data']),pretrained_sha256=c['pretrained']['sha256'],runtime=dict(gpu_uuid=c['gpu_uuid']))

def fresh(c,arm,device):
    import torch,numpy as np
    from ..baseline.training_v2b import _optimizer
    from ..protocol_r18.model import CommonEMA
    from ..protocol_r18.engine import tensor_digest
    from .local_detail_engine import Engine
    m,crit,info=short.construct(c,arm);initial=tensor_digest(m.state_dict().items())
    require(initial==ct.exact(c['prepare_gate'])['initializations'][arm]['model_sha256'],'formal fresh initial differs')
    conf=short.configuration(c);b=binding(c,arm,info,conf)
    m=m.to(device);crit=crit.to(device);e=Engine(m,crit,_optimizer(m,conf),CommonEMA(m),conf)
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    if device.type=='cuda':torch.cuda.manual_seed_all(0)
    require(e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'formal initial engine not empty')
    return e,b,initial

def train(c,stage,out,device,emit,forward,update,guard):
    import torch
    from . import local_detail_checkpoint as cp
    arm=stage[-1];e,b,initial=fresh(c,arm,device);publish(out/'binding.json',b)
    counters=dict(fw=0,bf16=0)
    def before(m,a):guard();forward();counters['fw']+=1
    def after(m,a,value):
        require(value['pred_logits'].dtype==torch.bfloat16,'formal forward is not BF16');counters['bf16']+=1;guard()
    hooks=[e.model.register_forward_pre_hook(before),e.model.register_forward_hook(after)]
    emit('initial_loaded',initial_model_sha256=initial,optimizer_entries=0,EMA_updates=0,used_acceptance_state=False,used_old_endpoint=False)
    initial_ref=cp.save(e,b,out/'checkpoint-e000.pth');data=dataset(c);clock=ct.profile(c);inputs=hashlib.sha256();epoch_seconds=[]
    for epoch in range(1,clock['epochs']+1):
        require(e.completed_updates==(epoch-1)*clock['windows'],'epoch start cursor differs')
        started=time.monotonic();n=0
        emit('epoch-begin',epoch=epoch,augmentation='strong' if epoch<118 else 'weak')
        for win,(im,t,ident) in enumerate(windows(c,data,epoch)):
            if epoch==1 and win==0:
                expected=ct.exact(c['prepare_gate'])['windows'][0]
                require(ident==expected if c['profile']=='LAB_GPU_ONCE' else ident['sha256']==expected['sha256'],'first actual training window differs')
            inputs.update(digest(ident).encode());update();r=e.train_window(im,t,capture=False);update(done=True);n+=1;guard()
            require(e.completed_updates==(epoch-1)*clock['windows']+win+1,'formal update clock differs')
            emit('update-'+str(e.completed_updates),input=ident,result=r)
            atomic(out/'progress.json',dict(event='update-'+str(e.completed_updates),epoch=epoch,window=win,monotonic=time.monotonic()))
        require(n==clock['windows'],'incomplete epoch');epoch_seconds.append(time.monotonic()-started)
        if epoch in (1,30,60,90,117,clock['epochs']):
            guard();last=cp.save(e,b,out/f'checkpoint-e{epoch:03d}.pth')
            emit('checkpoint',epoch=epoch,checkpoint=last)
            if c['profile']=='CPU_FIXTURE_ONLY' and epoch==1:
                restored=cp.restore(e,b,last['path'],last['sha256']);require(restored['full_state_exact'],'cross-epoch restoration differs')
                publish(out/'CROSS_EPOCH_RESTORE.json',restored)
    require(e.completed_updates==e.ema.updates==clock['epochs']*clock['windows'],'fixed e120 incomplete')
    # Full state read-back, bounded by one checkpoint, never by the full event log.
    saved=cp.read(last['path'],last['sha256'],b);require(cp.exact_equal(saved,cp.capture(e,b)),'endpoint state read-back differs');del saved
    require(counters['fw']==counters['bf16']==ct.stage_budget(c,stage),'formal BF16/count differs')
    ep=dict(checkpoint=last,binding=reference(out/'binding.json'),epoch=clock['epochs'],updates=e.completed_updates,EMA=True,
        initial_model_sha256=initial,used_acceptance_state=False,used_old_endpoint=False)
    publish(out/'endpoint.json',ep);emit('endpoint_locked',endpoint=reference(out/'endpoint.json'))
    for h in hooks:h.remove()
    return dict(status='PASS',stage=stage,endpoint=reference(out/'endpoint.json'),initial_checkpoint=initial_ref,
        actual_updates=e.completed_updates,forward_calls=counters['fw'],BF16_forward_calls=counters['bf16'],student_inputs_sha256=inputs.hexdigest(),epoch_seconds=epoch_seconds)

def endpoint(c,arm,device):
    import torch
    from . import local_detail_checkpoint as cp
    from .local_detail_model import validate_model
    from .local_detail_engine import assert_healthy
    m,crit,info=short.construct(c,arm)
    if arm=='N':
        if c['profile']=='LAB_GPU_ONCE':
            from .local_detail_reference import load_N_for_cost
            ep,b=ct.historical(c)
            load_N_for_cost(m.eval(),ep['checkpoint']['path'],expected_sha=ep['checkpoint']['sha256'],expected_binding=b)
    else:
        ep=ct.exact(read(Path(c['job_root'])/('train-'+arm)/'RESULT.json')['endpoint']);b=ct.exact(ep['binding'])
        require(b==binding(c,arm,info,short.configuration(c)) and ep['EMA'] and not ep['used_acceptance_state'] and not ep['used_old_endpoint'],'endpoint weight provenance differs')
        state=cp.read(ep['checkpoint']['path'],ep['checkpoint']['sha256'],b)
        require(state['engine']['completed_updates']==state['ema']['updates']==ep['updates']==ct.profile(c)['epochs']*ct.profile(c)['windows'] and ep['epoch']==ct.profile(c)['epochs'],'fixed endpoint clocks differ')
        m.load_state_dict(state['ema']['module'],strict=True);del state
    m=m.to(device).eval().requires_grad_(False);crit=crit.to(device)
    validate_model(m,arm);assert_healthy(m)
    return m,crit

def image_batch(reader,rows,start):
    import torch
    return torch.stack([reader.item(r,start+i)[0] for i,r in enumerate(rows)])

def infer(c,stage,out,device,emit,forward,guard):
    import torch
    from ..r0.evaluation import rank_candidates
    from ..localization.data import RawAugmentation
    from ..protocol_r18.geometry import forward_guards
    from ..localization.consumer_geometry import matching
    arm=stage[-1];model,crit=endpoint(c,arm,device);rows,reader=development(c);pred={600:[],300:[]};rawrefs=[];centers=[];coarse=[]
    # Read-only hook at the already executed layer3 input: detached normalized second-layer box.
    handle=model.decoder.decoder.layers[2].register_forward_pre_hook(lambda m,args:coarse.append(args[1].detach().float().cpu().squeeze(2)))
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),4):
                batch=rows[start:start+4];images=image_batch(reader,batch,start).to(device);forward()
                with forward_guards(model):value=model(images)
                guard();require(len(coarse)==1,'coarse-center hook coverage differs');before=coarse.pop()
                logits=value['pred_logits'].detach().float().cpu();boxes=value['pred_boxes'].detach().float().cpu()
                path=out/f'raw-{start//4:03d}.pth';torch.save(dict(ids=[r['stable_image_id'] for r in batch],logits=logits,boxes=boxes,coarse=before),path);rawrefs.append(reference(path))
                targets=[]
                for i,row in enumerate(batch):
                    raw=reader.raw(row);_,target,_=RawAugmentation(short.configuration(c).input_hw)(__import__('PIL.Image',fromlist=['Image']).new('RGB',(row['width'],row['height'])),raw,epoch=120,seed=0,position=start+i,image_id=row['stable_image_id'],training=False)
                    targets.append({k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in target.items()})
                    for budget in pred:pred[budget]+=rank_candidates(logits[i],boxes[i],row,budget)
                with matching(crit.matcher):pairs=crit.matcher(value,targets)['indices']
                for i,(qi,gi) in enumerate(pairs):
                    row=batch[i];t=targets[i]
                    for q,g in zip(qi.tolist(),gi.tolist()):
                        area=float(t['raw_area'][g])
                        if 0<area<=1024:
                            scale=torch.tensor([row['width'],row['height']]);truth=t['boxes'][g,:2].float().cpu()
                            eb=float(torch.linalg.vector_norm((before[i,q,:2]-truth)*scale)/math.sqrt(area));ef=float(torch.linalg.vector_norm((boxes[i,q,:2]-truth)*scale)/math.sqrt(area))
                            centers.append(dict(image_id=row['stable_image_id'],query=q,GT_id=t['gt_ids'][g],raw_area=area,coarse_center=before[i,q,:2].tolist(),final_center=boxes[i,q,:2].tolist(),GT_center=truth.tolist(),coarse_error=eb,final_error=ef,delta=ef-eb))
                emit('batch-'+str(start//4+1),images=len(batch),raw=rawrefs[-1],matched_small=len(centers))
    finally:handle.remove()
    refs={}
    for b in pred:
        path=out/f'predictions-{b}.json';publish(path,pred[b]);refs[str(b)]=reference(path)
    publish(out/'CENTER_RECORDS.json',dict(records=centers,denominator='own final300 native Hungarian pairs, raw training-valid GT, 0<raw_area<=1024, no confidence filter',N='UNKNOWN_NO_EXTRA_INFERENCE'))
    vals=[x['delta'] for x in centers]
    summary=dict(images=len(rows),matched_small=len(centers),N='UNKNOWN',scope='endpoint_local_support_not_training_cause',
        coarse_mean=statistics.mean(x['coarse_error'] for x in centers) if centers else None,coarse_median=statistics.median(x['coarse_error'] for x in centers) if centers else None,
        final_mean=statistics.mean(x['final_error'] for x in centers) if centers else None,final_median=statistics.median(x['final_error'] for x in centers) if centers else None,
        improved=sum(x<0 for x in vals),unchanged=sum(x==0 for x in vals),worsened=sum(x>0 for x in vals),raw_records=reference(out/'CENTER_RECORDS.json'))
    publish(out/'CENTER_SUMMARY.json',summary)
    return dict(status='PASS',stage=stage,predictions=refs,raw_batches=rawrefs,images=len(rows),forward_calls=math.ceil(len(rows)/4),EMA=True,centers=reference(out/'CENTER_SUMMARY.json'))

def score(c,out,emit):
    from ..r0.evaluation import score_budget
    rows,reader=development(c);truth=[reader.raw(r) for r in rows];results={'N':c['historical_N']['scores']}
    for arm in ('U','D'):
        source=read(Path(c['job_root'])/('infer-'+arm)/'RESULT.json');results[arm]={}
        for budget in ('600','300'):
            pred=ct.exact(source['predictions'][budget]);require(len(pred)==len(rows)*int(budget) and [p['image_id'] for p in pred[::int(budget)]]==[r['stable_image_id'] for r in rows],'prediction order/coverage differs')
            value=score_budget(rows,truth,pred,str(ct.ROOT));path=out/f'{arm}-{budget}.json';publish(path,value);results[arm][budget]=reference(path);emit(arm+'-'+budget,result=reference(path),evaluator_calls=2)
    for b,ref in results['N'].items():
        p=out/f'N-{b}.json';p.write_bytes(Path(ref['path']).read_bytes());require(sha(p)==ref['sha256'],'N reused score SHA differs');results['N'][b]=reference(p)
    return dict(status='PASS',stage='score',results=results,new_groups=4,new_evaluator_calls=8,N_recomputed=False,forward_calls=0)

def cost(c,out,device,emit,forward,guard):
    import torch
    from ..r0.evaluation import rank_candidates,bridge_inputs
    from ..protocol_r18.geometry import forward_guards
    rows,reader=development(c);by={r['stable_image_id']:r for r in rows};fixed=[by[x] for x in ct.fixed_ids(c)];truth=[reader.raw(r) for r in fixed]
    measures={a:[] for a in ct.ARMS};calls=0
    def measure(arm):
        nonlocal calls
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        model,_=endpoint(c,arm,device)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter();pred=[]
        with torch.inference_mode():
            for start in range(0,len(fixed),4):
                im=image_batch(reader,fixed[start:start+4],start).to(device);forward();calls+=1
                with forward_guards(model):value=model(im)
                guard();logits=value['pred_logits'].detach().float().cpu();boxes=value['pred_boxes'].detach().float().cpu()
                for i,row in enumerate(fixed[start:start+4]):pred+=rank_candidates(logits[i],boxes[i],row,600)
            _,_,counts=bridge_inputs(fixed,truth,pred)
        if device.type=='cuda':torch.cuda.synchronize()
        r=dict(seconds=time.perf_counter()-began,images=len(fixed),batch=4,dtype='FP32',filter_counts=counts,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else 0)
        del model;gc.collect();return r
    for a in ct.ARMS:
        for i in range(2):emit(f'warmup-{a}-{i}',measurement=measure(a))
    for i,order in enumerate(ct.POLICY['cost_orders']):
        for a in order:
            r=measure(a);measures[a].append(r);emit(f'measured-{i}-{a}',measurement=r)
    summary={a:dict(mean_seconds=statistics.mean(x['seconds'] for x in rr),sample_SD_seconds=statistics.stdev(x['seconds'] for x in rr),peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rr),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rr)) for a,rr in measures.items()}
    require(calls==ct.stage_budget(c,'cost'),'fixed cost count differs')
    return dict(status='PASS',stage='cost',forward_calls=calls,measurements=measures,summary=summary,orders=ct.POLICY['cost_orders'],boundary='decode_resize_H2D_forward_rank_raw_ignore_synchronize; excludes model load/GT parse',scope='FIXED64_GPU_E2E' if device.type=='cuda' else 'CPU_FIXTURE_NOT_COST_EVIDENCE')

def compare(c,out):
    root=Path(c['job_root']);sc=read(root/'score/RESULT.json');tim=read(root/'cost/RESULT.json');metrics={a:{b:ct.exact(ref) for b,ref in sc['results'][a].items()} for a in ct.ARMS};deltas={};gates=[]
    require(read(root/'train-U/RESULT.json')['student_inputs_sha256']==read(root/'train-D/RESULT.json')['student_inputs_sha256'],'U/D input/GT/augmentation differ')
    limits={600:dict(AP75_small=.3,APS=.2,AP=0,APM=-.2,APL=-.2,AR_large=-.2),300:dict(AP75_small=0,AP=-.2,APS=-.2)}
    for a,b in (('D','N'),('D','U'),('U','N')):
        pair=a+'-'+b;deltas[pair]={}
        for budget in ('600','300'):
            x=metrics[a][budget];y=metrics[b][budget]
            co={k:100*(v-y['COCO_bridge']['metrics'][k]) for k,v in x['COCO_bridge']['metrics'].items()}
            of={k:x['official'][k]-y['official'][k] for k in ('AP','AP50','AP75','AR1','AR10','AR100','AR500')}
            deltas[pair][budget]=dict(COCO_pp=co,primary_pp=of)
            if a=='D':
                for k,limit in limits[int(budget)].items():
                    valid=x['COCO_bridge']['metrics'][k]>=0 and y['COCO_bridge']['metrics'][k]>=0
                    gates.append(dict(pair=pair,budget=int(budget),metric=k,delta_pp=co[k],minimum=limit,defined=valid,passed=valid and co[k]>=limit))
                gates.append(dict(pair=pair,budget=int(budget),metric='primary_AP',delta_pp=of['AP'],minimum=-.2,defined=True,passed=of['AP']>=-.2))
    t=tim['summary'];ratios={a+'/'+b:dict(E2E=t[a]['mean_seconds']/t[b]['mean_seconds'],peak_allocated=t[a]['peak_allocated_bytes']/t[b]['peak_allocated_bytes'] if t[b]['peak_allocated_bytes'] else None) for a,b in (('D','N'),('D','U'),('U','N'))}
    fixture=c['profile']=='CPU_FIXTURE_ONLY';r=ratios['D/N']
    return dict(status='PASS',stage='compare',forward_calls=0,absolute=metrics,deltas_pp=deltas,quality_gates=gates,quality_passed=None if fixture else all(x['passed'] for x in gates),cost_passed=None if fixture else r['E2E']<=1.05 and r['peak_allocated']<=1.05,cost=tim,ratios=ratios,
        scientific_interpretation='single_seed_screen_only; quality_fail_ends_fixed_H2; N_historical_not_fully_contemporaneous; mechanism_not_cause; originality_unconfirmed',fixture_only=fixture)

def execute(c,stage,out,emit,forward,update,guard):
    gpu=c['profile']=='LAB_GPU_ONCE' and stage.startswith(('train-','infer-','cost'))
    device=setup(c,out,gpu)
    if c.get('fixture',{}).get('fault')=='worker_exception':raise RuntimeError('CONSTRUCTED_FORMAL_WORKER_EXCEPTION')
    if stage.startswith('train-'):return train(c,stage,out,device,emit,forward,update,guard)
    if stage.startswith('infer-'):return infer(c,stage,out,device,emit,forward,guard)
    if stage=='score':return score(c,out,emit)
    if stage=='cost':return cost(c,out,device,emit,forward,guard)
    if stage=='compare':return compare(c,out)
    raise ValueError('unregistered formal stage')

def _prepare_training(c,out):
    import torch
    from ..protocol_r18.engine import tensor_digest
    from . import h2_formal_resources as res
    require(not torch.cuda.is_initialized(),'prepare initialized CUDA')
    torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    first=res.snapshot(c,entry=True);publish(out/'resource-entry.json',first)
    for role in ('train_core','development'):ct.exact(c['data'][role]['manifest'])
    if c['profile']=='LAB_GPU_ONCE':
        require(reference(c['pretrained']['path'])==c['pretrained'],'pretrained SHA/bytes differs')
        actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        require(actual==c['environment'],'prepare environment differs')
    data=short.training_data(c);members=0
    if data is not None:
        for row in data.records:
            data.bound_bytes(row);members+=1
            if members%256==0:res.snapshot(c,entry=False,baseline=first)
        require(members==4869,'real train members not complete')
    else:members=0
    expected=[]
    for images,targets,identity in short.windows(c,data):expected.append(identity)
    require(len(expected)==2,'prepare inputs missing')
    initial={};ns=None;us=None
    for arm in ct.ARMS:
        m,criterion,info=short.construct(c,arm);s=m.state_dict()
        if arm=='N':ns={k:v.clone() for k,v in s.items()}
        elif arm=='U':
            for k,v in s.items():
                if 'detail_readout' in k:require(int(v.count_nonzero())==0,'detail head not zero')
                elif 'decoder.decoder.layers.2.cross_attn.sampling_offsets.' in k:
                    keep=torch.arange(192).reshape(8,12,2)[:,4:].flatten();require(torch.equal(v,ns[k][keep]),'coarse offset init differs')
                else:require(torch.equal(v,ns[k]),'unregistered U/N init difference: '+k)
            us={k:v.clone() for k,v in s.items()}
        else:
            for k,v in s.items():
                expected_value=us[k][2:] if k.startswith('decoder.dec_bbox_head.2.layers.2.') else us[k]
                require(torch.equal(v,expected_value),'unregistered D/U init difference: '+k)
        initial[arm]=dict(model_sha256=tensor_digest(s.items()),parameters=info['trainable_parameters'])
        del m,criterion,s;gc.collect()
    if c['profile']=='LAB_GPU_ONCE':require(initial['N']['model_sha256']=='40f9d83dff1acf1918855c17b1d7a000c03a6153394487a3e4b168a4c2bee995','N common ImageNet initial state differs')
    gate=dict(status='PREPARE_PASS',source_sha256=c['source']['sha256'],science=c['science'],data_sha256=digest(c['data']),
        initializations=initial,windows=expected,verified_train_files=members,development_images_read=0,
        model_forward_calls=0,CUDA_initialized=False,profile=c['profile'])
    publish(out/'PREPARE_PASS.json',gate);return gate


