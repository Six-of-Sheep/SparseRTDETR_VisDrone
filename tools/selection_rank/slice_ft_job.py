"""Native single-arm dual-EMA fine-tune work and whole-image A readout; caller owns process.

D0 is the original G0 trajectory (native, no installed module).  Both EMA
records start from the same parent-EMA weights; EMA-R keeps counter 0 and
EMA-C keeps the parent counter 36,480.  Both endpoints are evaluated once at
the fixed FT012 endpoint; nothing is selected, tuned or added.
"""
import copy,gc,io,csv,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from ..localization import channel,worker as base
from ..localization.runio import require,read,publish,reference,digest,event,sha
from ..localization.data import training_loader,collate_window,parse_raw
from . import slice_finetune as ft
from . import slice_ft_progress as progress
from . import slice_ft_execution as execution
from .model import Config,build
from .engine import output_digest,tensor_digest
from . import checkpoint as native_state

EMAS=('R','C')


def admit(c,out):
    if c['scope']=='PRODUCTION':base.live_admission(out,c)
    else:
        s=channel.resource();from ..localization import guard
        guard.screen(s,gpu=False);require(s['boot_id']==c['boot_id'] and 0<=time.monotonic()-s['monotonic']<25,'CPU fixture resource stale')


def construct(c,out,arm,scope,holder,runtime=None):
    fixture=c['scope']=='CPU_FIXTURE'
    if runtime is None:
        if fixture:
            require(not torch.cuda.is_initialized(),'CPU fixture CUDA');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            runtime={'scope':'CPU_SYNTHETIC','torch':str(torch.__version__)}
        else:runtime=base.prepare_cuda(c,out)
    config=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32') if fixture else Config()
    model,criterion,_=build(config,synthetic=True)
    mode=ft.ARM_INSTALL[arm]
    ranking=ft.RANKING_INSTALL.get(arm)
    if ranking is not None:
        from . import selection_rank_arm
        selection_rank_arm.install(model,criterion,ranking)
    if fixture:
        initial=tensor_digest(model.state_dict().items());parent_sha='a'*64
        if mode is not None:
            from . import detail_path
            require(detail_path.install_detail_path(model,mode)==detail_path.DETAIL_PARAMETERS,'FT detail parameter delta differs')
    else:
        binding=execution.verified(c['parent_binding']);r=c['parent_checkpoint']
        require(Path(r['path']).stat().st_size==r['bytes'],'FT parent size')
        payload=native_state.read(r['path'],r['sha256'],binding)
        initial=ft.initialize_from_parent(model,payload,expected_binding=binding);del payload
        require(initial==c['initial_model_sha256'],'FT initial EMA/model bytes differ')
        if mode is not None:
            from . import detail_path
            require(detail_path.install_detail_path(model,mode)==detail_path.DETAIL_PARAMETERS,'FT detail parameter delta differs')
        model.to('cuda:0');criterion.to('cuda:0');parent_sha=r['sha256']
    opt,ema,alt=ft.make_optimizer_ema(model,config)
    ema_digest=tensor_digest(ema.module.state_dict().items())
    alt_digest=tensor_digest(alt.module.state_dict().items())
    require(ema.updates==0 and alt.updates==ft.ALT_OFFSET and ema_digest==alt_digest==initial,'FT dual EMA records do not share the exact initial state')
    e=ft.FineTuneEngine(model,criterion,opt,ema,config,arm=arm,plan_sha256=c['plan_sha256'],parent_sha256=parent_sha,alt_ema=alt)
    e.ranking_mode=ranking
    if ranking is not None:
        from . import selection_rank_arm
        selection_rank_arm.assert_installed(e.model,e.criterion,ranking)
    if mode is not None:
        from . import detail_path
        detail_path.assert_installed(e.model,mode);detail_path.assert_installed(e.ema.module,mode)
    e.execution_binding={**{k:c[k] for k in ('task_id','source_sha256','source_commit','contract_sha256','plan_sha256','data_sha256')},
        'scope':scope,'arm':arm,'install_mode':mode,'ranking_mode':ranking,'parent_sha256':parent_sha,'runtime':runtime,'CPU_fixture':fixture,
        'dual_ema':{'ema_r_updates_start':0,'ema_c_updates_start':ft.ALT_OFFSET,'same_initial_module':True,
                    'initial_module_sha256':initial,'training_feedback':False,
                    'declared_checkpoint_passes':list(ft.CHECKPOINT_PASSES)}}
    holder.update(engine=e,binding=e.execution_binding)
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    if not fixture:torch.cuda.default_generators[0].manual_seed(0)
    require(e.completed_updates==e.ema.updates==0 and e.alt_ema.updates==ft.ALT_OFFSET and not e.optimizer.state,'FT fresh optimizer/EMA clocks')
    return e,runtime,initial


class FixtureTrain(torch.utils.data.Dataset):
    def __init__(self,arm,epoch,start,limit):self.arm,self.epoch,self.start,self.limit=arm,epoch,start,limit
    def __len__(self):return 2*self.limit
    def __getitem__(self,index):
        from PIL import Image
        pos=self.start*2+index
        raw=parse_raw(b'30,30,20,25,1,1,0,0\n90,20,20,20,0,0,0,0\n' if pos%2==0 else b'','cpu-ft-'+str(pos))
        x,t,tr=ft.Transform(self.arm,(128,128))(Image.fromarray(np.full((128,128,3),90+pos,np.uint8)),raw,
                    epoch=self.epoch,position=pos,image_id='cpu-ft-'+str(pos))
        tr['spawn_observer']={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()}
        return x,t,tr


def loader(c,arm,epoch,start=0,limit=None):
    if c['scope']=='CPU_FIXTURE':
        d=FixtureTrain(arm,epoch,start,limit or 1)
        return torch.utils.data.DataLoader(d,batch_size=2,num_workers=2,prefetch_factor=2,multiprocessing_context='spawn',
           collate_fn=collate_window,generator=torch.Generator().manual_seed(0),timeout=120)
    d=c['data']['train_core'];r=d['roots']
    data=ft.Dataset(arm,manifest_path=d['manifest'],images_path=r['images'],annotations_path=r['annotations'],root_binding=r)
    return training_loader(data,epoch=ft.clock(epoch),completed_windows=start,window_limit=limit)


def native_twin(c,out):
    """Parent-loaded native twin used by the installed-arm acceptance probe."""
    fixture=c['scope']=='CPU_FIXTURE'
    config=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32') if fixture else Config()
    twin,criterion,_=build(config,synthetic=True)
    if not fixture:
        binding=execution.verified(c['parent_binding']);r=c['parent_checkpoint']
        payload=native_state.read(r['path'],r['sha256'],binding)
        ft.initialize_from_parent(twin,payload,expected_binding=binding);del payload
        twin.to('cuda:0');criterion.to('cuda:0')
    return twin,criterion


def arm_probe(c,out,arm,mode,e,x,t):
    """Zero-gate identity, stop-gradient and nonzero-gate evidence before the windows.

    Unused by D0 (no installed module); retained unchanged for the shared chain.
    """
    from . import detail_path
    from ..baseline.training_v2b_engine import _move
    bs=e.config.physical_batch_size;pb=max(1,bs//2)
    require(isinstance(x,torch.Tensor) and len(t)>=pb and x.shape[0]==2*bs,'FT probe window shape')
    xp=x[:pb].to(e.device);tp=_move(list(t[:pb]),e.device)
    twin,criterion_t=native_twin(c,out)
    twin.train();criterion_t.train();e.model.train();e.criterion.train()
    with torch.no_grad():
        torch.manual_seed(20261031);a=twin(xp,targets=tp)
        torch.manual_seed(20261031);b=e.model(xp,targets=tp)
    require(all(torch.equal(a[k],b[k]) for k in ('pred_logits','pred_boxes')),'FT probe zero-gate output identity differs')
    info=dict(detail_path.LAST_SOURCE_INFO)
    require(info.get('mode')==mode and info.get('source_requires_grad') is False,'FT probe stop-gradient guard evidence')
    del a,b
    twin.zero_grad(set_to_none=True);e.model.zero_grad(set_to_none=True)
    torch.manual_seed(20261031);out_n=twin(xp,targets=tp)
    torch.manual_seed(20261031);loss_n=sum(criterion_t(out_n,tp).values())
    loss_n_value=loss_n.detach().clone()
    loss_n.backward()
    grads_n={n:(p.grad.detach().clone() if p.grad is not None else None) for n,p in twin.named_parameters()}
    del out_n,loss_n
    twin.zero_grad(set_to_none=True)
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    torch.manual_seed(20261031);out_i=e.model(xp,targets=tp)
    torch.manual_seed(20261031);loss_i=sum(e.criterion(out_i,tp).values())
    require(torch.equal(loss_n_value,loss_i.detach()),'FT probe zero-gate loss identity differs')
    loss_i.backward()
    del out_i,loss_i,loss_n_value
    installed=dict(e.model.named_parameters());shared=0;nonzero=False
    for n,g0 in grads_n.items():
        g1=installed[n].grad
        require((g0 is None)==(g1 is None),'FT probe shared gradient presence differs: '+n)
        if g0 is not None:
            require(torch.equal(g0,g1),'FT probe shared gradient differs: '+n);shared+=1
            if float(g0.abs().sum())>0:nonzero=True
    require(shared>0 and nonzero,'FT probe gradient comparison vacuous')
    gate=e.model.decoder.detail_inject.gate;conv=e.model.decoder.detail_inject.conv.weight
    probe={'mode':mode,'probe_batch':pb,'zero_gate_bitwise_outputs':True,'zero_gate_bitwise_loss':True,
           'shared_parameter_gradients_bitwise_identical':shared,
           'shared_parameter_gradients_nonzero':nonzero,
           'source_requires_grad_before_branch_detach':info.get('source_requires_grad_before_branch_detach'),
           'branch_source_stop_gradient_enforced':True,
           'gate_grad_nonzero_at_zero_gate':bool(gate.grad is not None and gate.grad.abs().sum()>0),
           'conv_grad_zero_at_zero_gate':bool(conv.grad is None or float(conv.grad.abs().max())==0.0)}
    require(probe['gate_grad_nonzero_at_zero_gate'],'FT probe gate received no gradient')
    require(probe['conv_grad_zero_at_zero_gate'],'FT probe conv gradient at zero gate differs')
    e.model.zero_grad(set_to_none=True)
    twin.zero_grad(set_to_none=True)
    del twin,criterion_t,grads_n,xp,tp
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    return probe


def input_record(c,arm,epoch,win,x,t,tr):
    size=2 if c['scope']=='CPU_FIXTURE' else 16
    require(len(x)==len(t)==len(tr)==size,'FT complete input window')
    require(all(not a['spawn_observer']['CUDA_initialized'] and a['spawn_observer']['pid']!=os.getpid() for a in tr),'FT spawn CUDA/ownership')
    require(all(a['epoch']==ft.clock(epoch) and a['position']==win*size+i and a['fixed_view']['view_id']==ft.view_id(arm,win*size+i) for i,a in enumerate(tr)),'FT view/cursor differs')
    # Validate delivered tensor bytes against the real transform audit.
    for image,target,trace in zip(x,t,tr):
        import hashlib
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for k in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):
            h.update(k.encode());h.update(target[k].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode())
        require(h.hexdigest()==trace['augmented_tensor_sha256'],'FT augmented audit differs')
    return {'ids':[a['image_id'] for a in tr],'traces':[{k:v for k,v in a.items() if k!='spawn_observer'} for a in tr],
            'target_sha256':output_digest(t),'image_sha256':tensor_digest([('pixels',x)])}


def update(c,out,arm,epoch,win,e,x,t,tr,holder,role):
    inp=input_record(c,arm,epoch,win,x,t,tr);holder['input']={'images':x,'targets':t,'metadata':inp}
    admit(c,out);r=e.train_window(x,t,capture=True);r['input']=inp;r['input_sha256']=digest(inp)
    require(e.completed_updates==e.ema.updates==r['update'] and e.phase==0 and r['optimizer_steps']==r['ema_updates']==1 and r['finite'],'FT complete update clocks')
    require(e.alt_ema.updates==r['update']+ft.ALT_OFFSET and r.get('ema_alt_updates')==1,'FT complete EMA-C update')
    counts=(13,22) if getattr(e.criterion,'selection_rank',None) is not None else (12,21)
    require(all(len(z) in counts for z in r['microbatch_losses']),'FT native loss count')
    progress.save_audit(out,channel.binding(),r,role)
    return r


def save(e,out,name):
    event(out,'checkpoint_begin',name=name,update=e.completed_updates)
    rec=ft.save(e,Path(out)/name)
    rec.update(execution_binding=e.execution_binding,initial_parent=e.parent_sha256)
    publish(Path(rec['path']+'.complete.json'),rec)
    event(out,'checkpoint_complete',name=name,update=e.completed_updates,checkpoint=rec)
    return rec


def acceptance(c,out,arm,holder):
    e,runtime,initial=construct(c,out,arm,'gpu_acceptance',holder)
    publish(out/'binding.json',e.execution_binding)
    results=[];started=time.monotonic();probe=None
    for win,(x,t,tr) in enumerate(loader(c,arm,1,limit=2)):
        if win==0 and ft.ARM_INSTALL[arm] is not None:
            probe=arm_probe(c,out,arm,ft.ARM_INSTALL[arm],e,x,t)
        results.append(update(c,out,arm,1,win,e,x,t,tr,holder,'direct'))
        if win==0:
            cp=save(e,out,'C1.pth')
            if probe is not None:
                gate_abs=float(e.model.decoder.detail_inject.gate.detach().abs().max())
                probe['gate_absmax_after_update1']=gate_abs
                probe['gate_moved_off_zero']=bool(gate_abs>0)
                require(probe['gate_moved_off_zero'],'FT probe gate did not move off zero')
    require(len(results)==2,'FT fixed acceptance window count')
    if probe is not None:
        ema_abs=float(e.ema.module.decoder.detail_inject.gate.detach().abs().max())
        probe['ema_updates']=int(e.ema.updates)
        probe['ema_gate_absmax']=ema_abs
        probe['ema_gate_nonzero']=bool(ema_abs>0)
        require(probe['ema_updates']==2 and probe['ema_gate_nonzero'],'FT probe EMA did not track the branch')
    require(e.completed_updates==e.ema.updates==2 and e.alt_ema.updates==2+ft.ALT_OFFSET,'FT dual EMA clocks after acceptance differ')
    require(e.alt_rng_neutral_updates==2,'FT EMA-C RNG-check count differs')
    dual={'ema_r_updates':int(e.ema.updates),'ema_c_updates':int(e.alt_ema.updates),'ema_c_offset':ft.ALT_OFFSET,
          'rng_neutral_alt_ema_updates':int(e.alt_rng_neutral_updates),
          'initial_module_sha256':initial,
          'ema_r_module_sha256':tensor_digest(e.ema.module.state_dict().items()),
          'ema_c_module_sha256':tensor_digest(e.alt_ema.module.state_dict().items()),
          'ema_r_equals_initial':tensor_digest(e.ema.module.state_dict().items())==initial,
          'ema_c_equals_initial':tensor_digest(e.alt_ema.module.state_dict().items())==initial,
          'records_recorded_separately':True}
    direct=save(e,out,'C2-direct.pth');expected=ft.capture(e)
    holder.pop('engine');holder.pop('input',None);del e;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    clone,_,initial2=construct(c,out,arm,'gpu_acceptance',holder,runtime=runtime)
    require(initial==initial2,'FT independent acceptance initial bytes')
    restored=ft.restore(clone,cp);n=0
    for x,t,tr in loader(c,arm,1,start=1,limit=1):
        replay=update(c,out,arm,1,1,clone,x,t,tr,holder,'replay');n+=1
    require(n==1 and {k:v for k,v in results[1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'FT GPU input/DN/output/loss/gradient replay mismatch')
    require(native_state.exact_equal(expected,ft.capture(clone)),'FT GPU full state/RNG replay mismatch')
    last=save(clone,out,'C2-replay.pth')
    publish(out/'RESULT.json',{'status':'PASS','arm':arm,'install_mode':ft.ARM_INSTALL[arm],'arm_probe':probe,'dual_ema':dual,'actual_updates':3,'sequence':[1,2,2],
       'input_DN_output_loss_preclip_gradient_exact':True,'full_state_RNG_exact':True,'initial_model_sha256':initial,
       'state_usable_for_formal':False,'restored':restored,'checkpoints':[cp,direct,last],
       'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),
       'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
       'elapsed_seconds':time.monotonic()-started,'CPU_fixture':c['scope']=='CPU_FIXTURE'})


def formal(c,out,arm,holder):
    e,runtime,initial=construct(c,out,arm,'formal',holder)
    accepted=execution.complete(c,execution.ACCEPTANCE_PHASE)
    require(initial==accepted['initial_model_sha256'] and e.completed_updates==0 and not e.optimizer.state,'FT formal did not restart independently from parent')
    publish(out/'binding.json',e.execution_binding)
    plan=ft.read_plan(c['plan']);saves=ft.save_passes(plan,c['epochs'])
    zero=save(e,out,'checkpoint-FT000.pth')
    started=time.monotonic();times=[];written=[zero]
    for epoch in range(1,c['epochs']+1):
        execution.load(c['contract_path'])
        require(e.completed_updates==(epoch-1)*c['updates_per_epoch'],'FT epoch cursor differs')
        event(out,'epoch_begin',epoch=epoch,update=e.completed_updates)
        tick=time.monotonic();n=0
        for win,(x,t,tr) in enumerate(loader(c,arm,epoch)):
            update(c,out,arm,epoch,win,e,x,t,tr,holder,'direct');n+=1
        require(n==c['updates_per_epoch'] and e.completed_updates==epoch*c['updates_per_epoch'],'FT pass workload incomplete')
        times.append(time.monotonic()-tick)
        if epoch in saves:
            last=save(e,out,f'checkpoint-FT{epoch:03d}.pth');written.append(last)
            event(out,'epoch_complete',epoch=epoch,update=e.completed_updates,checkpoint=last)
        else:
            event(out,'epoch_complete',epoch=epoch,update=e.completed_updates,checkpoint=None)
    require(e.completed_updates==e.ema.updates==c['total_updates'],'FT endpoint count mismatch')
    require(e.alt_ema.updates==c['total_updates']+ft.ALT_OFFSET,'FT EMA-C endpoint count mismatch')
    require(e.alt_rng_neutral_updates==c['total_updates'],'FT EMA-C RNG-check count mismatch')
    endpoint={'status':'FT012_EMA_LOCKED' if c['scope']=='PRODUCTION' else 'CPU_SYNTHETIC_ENDPOINT',
       'arm':arm,'install_mode':ft.ARM_INSTALL[arm],'updates':e.completed_updates,'epoch':c['epochs'],'checkpoint':last,'initial_checkpoint':zero,
       'checkpoint_passes':sorted(saves),'checkpoints_written':[q['path'] for q in written],
       'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
       'binding':reference(out/'binding.json'),'initial_model_sha256':initial,'used_acceptance_state':False,
       'dual_ema':{'ema_r_updates':int(e.ema.updates),'ema_c_updates':int(e.alt_ema.updates),'ema_c_offset':ft.ALT_OFFSET,
                   'rng_neutral_alt_ema_updates':int(e.alt_rng_neutral_updates),
                   'ema_r_module_sha256':tensor_digest(e.ema.module.state_dict().items()),
                   'ema_c_module_sha256':tensor_digest(e.alt_ema.module.state_dict().items())},
       'parent_checkpoint':c['parent_checkpoint'],'training_wall_seconds':time.monotonic()-started,
       'epoch_wall_seconds':times,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256']}
    publish(out/'endpoint-lock.json',endpoint);publish(out/'RESULT.json',endpoint)


class FixtureDevelopment:
    def __init__(self,c):
        from PIL import Image
        from ..localization.data import RawAugmentation
        self.records=[{'stable_image_id':i,'width':512,'height':384} for i in c['development_ids']]
        self.by_id={r['stable_image_id']:r for r in self.records};self.transform=RawAugmentation((128,128))
    def raw(self,row):
        return parse_raw(b'20,20,20,20,1,1,0,0\n100,100,50,50,1,2,0,0\n200,180,120,120,1,3,0,0\n0,0,5,5,0,0,0,0\n30,100,15,15,0,4,0,0\n',row['stable_image_id'])


def development(c):
    if c['scope']=='CPU_FIXTURE':return FixtureDevelopment(c)
    from ..r0.evaluation_data import Development548
    d=Development548(c['data']['development'])
    require([r['stable_image_id'] for r in d.records]==c['development_ids'],'FT development member/order differs')
    return d


def view_pixels(c,data,rows,sliced):
    if c['scope']=='PRODUCTION':
        from infer_seed import make_view_pixels
        return make_view_pixels(data,rows,sliced)
    from PIL import Image
    from torchvision.transforms.v2 import functional as TF
    from slice_core import tile_windows
    groups=[[] for _ in range(5 if sliced else 1)]
    for row in rows:
        image=Image.fromarray(np.full((384,512,3),100,np.uint8))
        groups[0].append(TF.pil_to_tensor(TF.resize(image,[128,128])).float()/255.)
        if sliced:
            for tile in tile_windows(384,512):groups[tile.view_id].append(TF.pil_to_tensor(TF.resize(image.crop(tile.crop_box),[128,128])).float()/255.)
    return groups,None


def inference(c,out,phase,holder):
    from slice_core import tile_windows,map_crop_xyxy,select_view_candidates,merge_candidates
    from infer_seed import prediction
    from .geometry import forward_guards
    from ..localization.consumer_geometry import positive_box,corners
    parts=phase.split('-')
    require(len(parts)==4 and parts[0]=='infer','FT inference phase name differs')
    arm,view,emaselect=parts[1],parts[2],parts[3]
    require(view=='A' and emaselect in ('emaR','emaC'),'FT inference phase identity differs')
    select=emaselect[-1]
    e,_,_=construct(c,out,arm,'formal',holder)
    endpoint=execution.complete(c,'formal-'+arm);rec=endpoint['checkpoint']
    ft.restore(e,rec);require(e.completed_updates==c['total_updates'],'FT inference endpoint not locked')
    target=e.ema if select=='R' else e.alt_ema
    expected=rec['completed_updates'] if select=='R' else rec['completed_updates']+ft.ALT_OFFSET
    require(target.updates==expected,'FT selected EMA record is not at the locked endpoint')
    model=target.module;model.eval();before=ft.capture(e);data=development(c);sliced=view=='C'
    fullrows=data.records;ids=[r['stable_image_id'] for r in fullrows]
    fixedrows=[data.by_id[i] for i in c['fixed64']]
    own=None
    if sliced:
        a=execution.complete(c,'infer-'+arm+'-A-ema'+select);raw_ref=a['raw']
        require(reference(raw_ref['path'])==raw_ref,'FT own A cache SHA')
        own=torch.load(raw_ref['path'],map_location='cpu',weights_only=True)
        require(own['image_ids']==ids and own['checkpoint_sha256']==rec['sha256'],'FT own A raw endpoint identity')
    def sync():
        if e.device.type=='cuda':torch.cuda.synchronize()
    calls=0
    def batch(rows):
        nonlocal calls
        sync();start=time.perf_counter();groups,_=view_pixels(c,data,rows,sliced)
        candidates=[[] for _ in rows];raw=None
        for v,items in enumerate(groups):
            admit(c,out)
            with torch.autocast(e.device.type,enabled=False),forward_guards(model):z=model(torch.stack(items).to(e.device))
            calls+=1
            require(set(z)=={'pred_boxes','pred_logits'} and z['pred_boxes'].shape==(4,300,4) and z['pred_logits'].shape==(4,300,10) and z['pred_logits'].dtype==z['pred_boxes'].dtype==torch.float32,'FT inference native outputs')
            positive_box('FT/inference',z['pred_boxes']);require(torch.isfinite(z['pred_logits']).all(),'FT nonfinite logits')
            logits=z['pred_logits'].detach().cpu();boxes=z['pred_boxes'].detach().cpu()
            if v==0:raw={'pred_logits':logits,'pred_boxes':boxes}
            for j,row in enumerate(rows):
                tile=tile_windows(row['height'],row['width'])[v-1] if v else None
                w,h=(tile.width,tile.height) if tile else (row['width'],row['height'])
                xy=torch.cat((boxes[j,:,:2]-boxes[j,:,2:]/2,boxes[j,:,:2]+boxes[j,:,2:]/2),-1)
                xy*=torch.tensor([w,h]*2,dtype=torch.float32);corners('FT/pixel-corners',xy)
                mapped=map_crop_xyxy(xy.tolist(),tile) if tile else xy.tolist()
                candidates[j].extend(select_view_candidates(mapped,logits[j].sigmoid().numpy(),view_id=v))
            del z
        predictions={600:[],300:[]}
        for row,points in zip(rows,candidates):
            merged=merge_candidates(points) if sliced else {'top600':points[:600],'top300':points[:300]}
            for b in (600,300):predictions[b].extend(prediction(row['stable_image_id'],p,whole=not sliced) for p in merged['top'+str(b)])
        if own is not None:
            ix=[ids.index(r['stable_image_id']) for r in rows]
            require(all(torch.equal(raw[k],own[k][ix]) for k in ('pred_boxes','pred_logits')),'FT C whole differs from own A')
        sync();return predictions,raw,time.perf_counter()-start
    preds={600:[],300:[]};rawrows=[];profile=[];started=time.monotonic()
    with torch.inference_mode():
        for k in range(2):batch(fixedrows[k*4:(k+1)*4])
        if e.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for kind,rows in [('profile',fixedrows),('full',fullrows)]:
            for i in range(len(rows)//4):
                current=rows[i*4:(i+1)*4];p,r,elapsed=batch(current)
                if kind=='profile':profile.append(elapsed)
                else:
                    for b in (600,300):preds[b].extend(p[b])
                    if not sliced:rawrows.append(r)
                event(out,'inference_batch',batch_kind=kind,batch=i+1,ids=[z['stable_image_id'] for z in current],elapsed_seconds=elapsed,forward_calls=calls)
    require(native_state.exact_equal(before,ft.capture(e)) and all(p.grad is None for p in e.model.parameters()),'FT inference altered full state/RNG/gradients')
    refs={}
    for b in (600,300):
        q=out/f'top{b}-predictions.json';publish(q,preds[b]);refs[str(b)]=reference(q)
    result={'status':'PASS','arm':arm,'view':view,'ema':select,'ema_updates':int(target.updates),
       'ema_module_sha256':tensor_digest(target.module.state_dict().items()),
       'images':len(ids),'checkpoint_sha256':rec['sha256'],
       'predictions':refs,'profile_e2e_seconds':sum(profile),'profile_batches':profile,'profile_images':len(fixedrows),
       'warmup_batches':2,'batch_size':4,'model_forwards':calls,'detector_updates':0,'full_state_RNG_exact':True,
       'whole_exact':len(ids) if sliced else None,'elapsed_seconds':time.monotonic()-started,
       'peak_allocated_bytes':0 if e.device.type=='cpu' else torch.cuda.max_memory_allocated(),
       'peak_reserved_bytes':0 if e.device.type=='cpu' else torch.cuda.max_memory_reserved()}
    if not sliced:
        q=out/'raw-A.pth'
        with q.open('xb') as f:torch.save({'image_ids':ids,'checkpoint_sha256':rec['sha256'],'ema':select,
             **{key:torch.cat([r[key] for r in rawrows]) for key in ('pred_logits','pred_boxes')}},f);f.flush();os.fsync(f.fileno())
        result['raw']=reference(q)
    publish(out/'RESULT.json',result)


def evaluation(c,out):
    from ..r0.evaluation import score_budget,POLICY
    from infer_seed import bridge_values,_group,_prefix
    data=development(c);rows=data.records;ids=[r['stable_image_id'] for r in rows]
    if c['scope']=='PRODUCTION':require(execution.verified(c['evaluation_policy'])==POLICY,'FT evaluation policy changed')
    parsed=[data.raw(r) for r in rows]
    values={};metricrefs={};ordinal=0
    for name in EMAS:
        values[name]={'A':{}};metricrefs[name]={'A':{}};grouped={}
        result=execution.complete(c,'infer-D0-A-ema'+name)
        for b in (600,300):
            records=execution.verified(result['predictions'][str(b)])
            grouped[b]=_group(records,ids,b,whole=True,exact=True)
            z=score_budget(rows,parsed,records,Path(c['source_root']))
            z['FT_binding']={'endpoint':name,'view':'A','budget':b,'checkpoint_sha256':result['checkpoint_sha256'],
                'ema_module_sha256':result['ema_module_sha256'],'ema_updates':result['ema_updates'],
                'prediction':result['predictions'][str(b)],'CPU_fixture':c['scope']=='CPU_FIXTURE'}
            q=out/f'D0-ema{name}-A-top{b}-metrics.json';publish(q,z)
            values[name]['A'][str(b)]=bridge_values(z);metricrefs[name]['A'][str(b)]=reference(q);ordinal+=1
            event(out,'evaluation_budget_complete',ordinal=ordinal,arm_name=name,view='A',budget=b,images=len(ids))
        _prefix(grouped[600],grouped[300],ids,whole=True)
    if c['scope']=='PRODUCTION':
        parent={str(b):bridge_values(execution.verified(c['parent_metrics']['A'][str(b)])) for b in (600,300)}
        g0={str(b):bridge_values(execution.verified(c['reference_metrics']['original_G0']['A'][str(b)])) for b in (600,300)}
    else:
        parent=copy.deepcopy(values['R']['A']);g0=copy.deepcopy(values['R']['A'])
    refs={'original_G0':g0,'parent':parent}
    rhs={'R':values['R']['A'],'C':values['C']['A'],'G0REF':g0,'PARENT':parent}
    pairs=[('A_C_minus_R','C','R'),('A_R_minus_originalG0','R','G0REF'),
           ('A_C_minus_originalG0','C','G0REF'),('A_R_minus_parent','R','PARENT'),('A_C_minus_parent','C','PARENT')]
    comparisons={}
    for name,lhs_name,rhs_name in pairs:
        lhs=values[lhs_name]['A'];rhs_side=rhs[rhs_name]
        comparisons[name]={str(b):{m:100*(lhs[str(b)][m]-rhs_side[str(b)][m]) for m in lhs[str(b)]} for b in (600,300)}
    acc=execution.complete(c,'accept-D0');endp=execution.complete(c,'formal-D0')
    inf={name:execution.complete(c,'infer-D0-A-ema'+name) for name in EMAS}
    costs={'accept_elapsed_seconds':acc['elapsed_seconds'],'accept_updates':acc['actual_updates'],
           'accept_peak_reserved_bytes':acc['peak_reserved_bytes'],
           'formal_updates':endp['updates'],'formal_training_wall_seconds':endp['training_wall_seconds'],
           'formal_epoch_wall_seconds':endp['epoch_wall_seconds'],'formal_peak_reserved_bytes':endp['peak_reserved_bytes'],
           'checkpoint_passes':endp['checkpoint_passes'],
           'endpoint_inference':{name:{'elapsed_seconds':inf[name]['elapsed_seconds'],'model_forwards':inf[name]['model_forwards'],
                                       'ema_updates':inf[name]['ema_updates'],'peak_reserved_bytes':inf[name]['peak_reserved_bytes']} for name in EMAS}}
    gpu_total=costs['accept_elapsed_seconds']+costs['formal_training_wall_seconds']+sum(v['elapsed_seconds'] for v in costs['endpoint_inference'].values())
    costs['gpu_stage_wall_seconds_total']=gpu_total
    limit=c['budget_limits']['gpu_hours'] if 'budget_limits' in c else 2.0
    cost_passed=gpu_total<=limit*3600 if c['scope']=='PRODUCTION' else True
    comparison={'scores':values,'reference_scores':refs,'differences_pp':comparisons,
       'quality':'RECORD_ONLY','cost':costs,'cost_passed':cost_passed,
       'units':'point_scores0..1; deltas pp converted once',
       'descriptive_note':('Recording experiment only. A_R_minus_originalG0 is the EMA-counter-0 reproduction distance on this trajectory '
                           '(the reproduction itself is asserted bitwise by dual_ema_verify); A_C_minus_R is how much the continued counter '
                           'moves the fixed-endpoint evaluation. No gate, no selection, no attribution: this cannot answer the optimizer-reset '
                           'causal contribution and cannot prove any module effective.'),
       'official_metrics_separate':metricrefs,'scope':'CPU_SYNTHETIC' if c['scope']=='CPU_FIXTURE' else 'seed0_development_exploratory'}
    publish(out/'COMPARISON.json',comparison)
    table=io.StringIO();writer=csv.writer(table);writer.writerow(['comparison','budget','metric','delta_pp'])
    for name,_,_ in pairs:
        for b in (600,300):
            for m in sorted(comparisons[name][str(b)]):writer.writerow([name,b,m,comparisons[name][str(b)][m]])
    with (out/'TABLE.csv').open('x') as f:f.write(table.getvalue())
    publish(out/'RESULT.json',{'status':'PASS','quality':comparison['quality'],'cost_passed':comparison['cost_passed'],
       'comparison':reference(out/'COMPARISON.json'),'metrics':metricrefs,'score_budget_calls':len(EMAS)*2,'no_bootstrap':True,'CUDA_initialized':torch.cuda.is_initialized()})


def accept_ranking(c,out,holder):
    """r17 engineering acceptance: exactly 3 optimizer updates covering A1, A2 and empty supervision.

    Each case is an independent single-update sub-run built from the certified parent through
    the frozen ``construct`` (which installs the declared ranking arm before the model moves and
    builds the two EMA records as deep copies), so the acceptance state can never enter the
    formal training, which restarts from the parent as well.  Per case: the production batch and
    precision configuration, every loss finite, the declared loss-key count (13/22 without/with
    the denoising branch), a nonzero finite gradient norm, the term's own owner/index checks run
    inside the criterion, and a device probe that compares the cached top-k indices against the
    decoder's own gathered enc_aux tensors bitwise on the engine device.  The A1 case also saves
    a chain checkpoint and restores it into a fresh engine: the full state must be exact and the
    A1 generator state (counter and stream) must continue identically.
    """
    started=time.monotonic();cases={};runtime=None;peaks={}
    state0=holder.get('state')

    def probe(e,batch,targets,label):
        """Device/index probe: no-grad forward on the same batch, after the case's update."""
        model=e.model
        with torch.no_grad():
            captured=getattr(e.criterion,'selection_rank',None)
            cache_before=None if captured is None else dict(captured._cache or {})
            model.train()
            outputs=model(batch,targets=targets)
            cache=dict(captured._cache or {}) if captured is not None else {}
        require(captured is not None and set(cache)=={'pool_logits','pool_coord_unact','topk_indices'},
                'FT acceptance ranking capture missing: '+label)
        pool=cache['pool_logits'];coords=cache['pool_coord_unact'];topk=cache['topk_indices']
        require(pool.device.type==e.device.type and coords.device.type==e.device.type and topk.device.type==e.device.type,
                'FT acceptance pool tensors are not on the engine device: '+label)
        require(int(pool.shape[0])==len(targets) and int(topk.shape[1])==int(model.decoder.num_queries),
                'FT acceptance pool/topk shape differs: '+label)
        reference=None
        for item in outputs.get('enc_aux_outputs',()):
            reference=item;break
        if reference is not None:
            classes=pool.shape[-1]
            gathered_logits=pool.gather(1,topk.unsqueeze(-1).expand(-1,-1,classes))
            gathered_boxes=torch.sigmoid(coords.gather(1,topk.unsqueeze(-1).expand(-1,-1,4)))
            require(torch.equal(gathered_logits,reference['pred_logits']) and torch.equal(gathered_boxes,reference['pred_boxes']),
                    'FT acceptance cached top-k indices do not reproduce the decoder enc_aux tensors: '+label)
            index_exact=True
        else:
            index_exact=None
        term=e.criterion.selection_rank
        stats=getattr(term,'last_stats',{}) or {}
        require(stats.get('instances',0)>=0,'FT acceptance term statistics missing: '+label)
        return {'device':str(pool.device),'pool_tokens':int(pool.shape[1]),'selected':int(topk.shape[1]),
                'index_gather_exact':index_exact,'owner_checks_recorded':bool(stats)}

    def one_case(case,arm,supervised,role):
        e,runtime0,initial=construct(c,out,arm,'gpu_acceptance',holder) if runtime is None else \
            construct(c,out,arm,'gpu_acceptance',holder,runtime=runtime)
        require(getattr(e.model.decoder,'_select_topk',None) is not None,'FT acceptance wrapper absent')
        windows=list(loader(c,arm,1,limit=1));require(len(windows)==1,'FT acceptance window count differs')
        x,t,tr=windows[0]
        targets=t;empty_audit=None
        if supervised:
            r=update(c,out,arm,1,0,e,x,targets,tr,holder,role)
        else:
            term=e.criterion.selection_rank
            draws_before=int(term.state_dict()['draws'])
            before=[len(row['gt_ids']) for row in t]
            targets=[{'boxes':row['boxes'][:0],'labels':row['labels'][:0],'raw_area':row['raw_area'][:0],
                      'gt_ids':[],**{k:v for k,v in row.items()
                                     if k not in ('boxes','labels','raw_area','gt_ids')}} for row in t]
            admit(c,out)
            emptied={'ids':[a['image_id'] for a in tr],
                     'traces':[{k:v for k,v in a.items() if k!='spawn_observer'} for a in tr],
                     'target_sha256':output_digest(t),'image_sha256':tensor_digest([('pixels',x)]),
                     'supervision_emptied':{'per_image_gt_before':before,
                                            'per_image_gt_after':[0 for _ in before],
                                            'audit_applied':False,
                                            'frozen_augmentation_trace_audit':'not applied: the delivered targets are emptied after augmentation, so the frozen input_record comparison cannot hold for this case'}}
            holder['input']={'images':x,'targets':targets,'metadata':emptied}
            r=e.train_window(x,targets,capture=True);r['input']=emptied;r['input_sha256']=digest(emptied)
            require(e.completed_updates==e.ema.updates==r['update'] and e.phase==0 and r['optimizer_steps']==r['ema_updates']==1 and r['finite'],'FT empty-case update clocks')
            require(e.alt_ema.updates==r['update']+ft.ALT_OFFSET and r.get('ema_alt_updates')==1,'FT empty-case EMA-C update')
            progress.save_audit(out,channel.binding(),r,role)
            draws_after=int(term.state_dict()['draws'])
            require(draws_after==draws_before,'FT empty supervision consumed an A1 draw')
            empty_audit={'delivered':'targets emptied after augmentation',
                         'frozen_input_record_audit_applied':False,
                         'per_image_gt_before':before,'per_image_gt_after':[0 for _ in before],
                         'term_draws_before':draws_before,'term_draws_after':draws_after,
                         'target_sha256':emptied['target_sha256'],'image_sha256':emptied['image_sha256']}
        counts=[len(row) for row in r['microbatch_losses']]
        if supervised:
            require(sorted(set(counts))==[13,22],'FT acceptance loss-key pair differs: %s'%counts)
        else:
            require(all(count in (13,22) for count in counts),'FT empty-case loss-key count differs: %s'%counts)
        require(all('loss_rank' in row for row in r['microbatch_losses']),'FT acceptance loss_rank missing')
        require(r['finite'] is True and r['gradient_norm_before_clip']>0.0,'FT acceptance gradient/finiteness differs')
        if not supervised:
            require(all(abs(row['loss_rank'])==0.0 for row in r['microbatch_losses']),
                    'FT acceptance empty supervision produced a ranking loss')
        probe_result=probe(e,x,targets,case)
        checks={'case':case,'arm':arm,'supervised':supervised,'loss_key_counts':counts,
                'empty_audit':empty_audit,
                'loss_rank':[row['loss_rank'] for row in r['microbatch_losses']],
                'gradient_norm_before_clip':r['gradient_norm_before_clip'],
                'ranking_mode':getattr(e,'ranking_mode',None),'probe':probe_result,
                'parameter_device':str(next(e.model.parameters()).device)}
        return e,initial,checks

    # case A1: supervised update, then the chain save/restore of the term state
    e1,initial,checks_a1=one_case('A1','A1',True,'A1')
    cp=save(e1,out,'C1-A1.pth')
    expected_state=ft.capture(e1)
    term_before=e1.criterion.selection_rank.state_dict()
    scratch=torch.Generator();scratch.set_state(term_before['a1_generator'])
    next_draws=[int(torch.randint(2**31,(1,),generator=scratch).item()) for _ in range(2)]
    holder.pop('engine');holder.pop('input',None);del e1;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    clone,_,initial2=construct(c,out,'A1','gpu_acceptance',holder,runtime=runtime)
    require(initial==initial2,'FT acceptance independent initial bytes differ')
    restored=ft.restore(clone,cp)
    require(native_state.exact_equal(expected_state,ft.capture(clone)),'FT acceptance full state/RNG restore mismatch')
    term_after=clone.criterion.selection_rank
    require(term_after is not None and int(term_after.state_dict()['draws'])==int(term_before['draws']),
            'FT acceptance A1 draw counter differs after restore')
    require(torch.equal(term_after.state_dict()['a1_generator'],term_before['a1_generator']),
            'FT acceptance A1 generator state differs after restore')
    continued=[int(torch.randint(2**31,(1,),generator=term_after.generator).item()) for _ in range(2)]
    require(continued==next_draws,('FT acceptance A1 stream does not continue identically',continued,next_draws))
    checks_a1['restore']={'full_state_RNG_exact':bool(restored.get('full_state_RNG_exact')),
                          'a1_state_exact':True,'draws_continue_identically':True,
                          'draws':int(term_before['draws'])}
    holder.pop('engine');holder.pop('input',None);del clone;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    # case A2 and the empty-supervision case
    e2,_,checks_a2=one_case('A2','A2',True,'A2')
    holder.pop('engine');holder.pop('input',None);del e2;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    e3,_,checks_empty=one_case('empty','A1',False,'empty')
    checks_empty['note']='empty supervision runs under the A1 arm; the ranking term must contribute exactly 0.0'
    holder.pop('engine');holder.pop('input',None);del e3;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    publish(out/'RESULT.json',{'status':'PASS','actual_updates':3,'sequence':[1,1,1],
        'roles':['A1','A2','empty'],'cases':{'A1':checks_a1,'A2':checks_a2,'empty':checks_empty},
        'initial_model_sha256':initial,'full_state_RNG_exact':True,'a1_state_restore_exact':True,
        'single_arm_acceptance_state_usable_for_formal':False,'state_usable_for_formal':False,
        'ranking_install':{a:ft.RANKING_INSTALL[a] for a in ft.ARMS},
        'device':str(getattr(clone,'device',None)) if False else str(torch.cuda.current_device() if torch.cuda.is_initialized() else 'cpu'),
        'cuda_initialized':bool(torch.cuda.is_initialized()),
        'gradient_checked':True,'loss_key_counts_expected':{'pair':[13,22],'denoising_on_one_microbatch_of_each_update':True},
        'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
        'CPU_fixture':c['scope']=='CPU_FIXTURE','elapsed_seconds':time.monotonic()-started})


def evaluation_selection_rank(c,out):
    """r17 fixed-endpoint evaluation: EMA-C only, three arms, the plan's pre-registered gates.

    Same scoring policy and budgets as the frozen chain (view A, batch 4, 548 development
    images, top600 primary and top300 secondary, FP32, no selection and no tuning).  Reports
    the A2-A0 net-benefit gate and the A2-A1 best-partner increment gate declared in the plan,
    plus A1-A0, every arm against the certified parent and the retrained native A0 against the
    original G0 readout of the same recipe, then the verdict from the plan's combination rule.
    """
    from ..r0.evaluation import score_budget,POLICY
    from infer_seed import bridge_values,_group,_prefix
    plan=ft.read_plan(c['plan']);arms=list(ft.ARMS);gates=plan['gates']
    data=development(c);rows=data.records;ids=[r['stable_image_id'] for r in rows]
    if c['scope']=='PRODUCTION':
        require(execution.verified(c['evaluation_policy'])==POLICY,'FT evaluation policy changed')
    parsed=[data.raw(r) for r in rows]
    values={};metricrefs={};ordinal=0
    for arm in arms:
        result=execution.complete(c,'infer-%s-A-emaC'%arm)
        grouped={};values[arm]={'A':{}};metricrefs[arm]={'A':{}}
        for b in (600,300):
            records=execution.verified(result['predictions'][str(b)])
            grouped[b]=_group(records,ids,b,whole=True,exact=True)
            z=score_budget(rows,parsed,records,Path(c['source_root']))
            z['FT_binding']={'endpoint':'C','view':'A','budget':b,'checkpoint_sha256':result['checkpoint_sha256'],
                'ema_module_sha256':result['ema_module_sha256'],'ema_updates':result['ema_updates'],
                'prediction':result['predictions'][str(b)],'CPU_fixture':c['scope']=='CPU_FIXTURE','arm':arm}
            q=out/f'{arm}-A-emaC-top{b}-metrics.json';publish(q,z)
            values[arm]['A'][str(b)]=bridge_values(z);metricrefs[arm]['A'][str(b)]=reference(q);ordinal+=1
            event(out,'evaluation_budget_complete',ordinal=ordinal,arm_name=arm,view='A',budget=b,images=len(ids))
        _prefix(grouped[600],grouped[300],ids,whole=True)
    if c['scope']=='PRODUCTION':
        parent={str(b):bridge_values(execution.verified(c['parent_metrics']['A'][str(b)])) for b in (600,300)}
        g0={str(b):bridge_values(execution.verified(c['reference_metrics']['original_G0']['A'][str(b)])) for b in (600,300)}
    else:
        parent=copy.deepcopy(values[arms[0]]['A']);g0=copy.deepcopy(values[arms[0]]['A'])
    refs={'original_G0':g0,'parent':parent}
    side={arm:values[arm]['A'] for arm in arms};side['parent']=parent;side['G0REF']=g0
    pairs=[('A2_minus_A0','A2','A0'),('A2_minus_A1','A2','A1'),('A1_minus_A0','A1','A0'),
           ('A0_minus_parent','A0','parent'),('A1_minus_parent','A1','parent'),('A2_minus_parent','A2','parent'),
           ('A0_minus_original_G0','A0','G0REF')]
    comparisons={}
    for name,lhs,rhs in pairs:
        comparisons[name]={str(b):{m:100*(side[lhs][str(b)][m]-side[rhs][str(b)][m])
                                   for m in side[lhs][str(b)]} for b in (600,300)}
    gate_rows={}
    for name,spec in sorted(gates.items()):
        if not isinstance(spec,dict) or 'conditions' not in spec:
            continue
        key='%s_minus_%s'%(spec['lhs'],spec['rhs']);detail=[];passed=True
        for condition in spec['conditions']:
            metric,budget=condition['metric'],str(condition['budget'])
            require(metric in comparisons[key][budget],('FT gate metric missing',name,metric))
            value=comparisons[key][budget][metric]
            ok=value>=condition['threshold'] if condition['op']=='>=' else value<=condition['threshold']
            detail.append({'metric':metric,'budget':int(budget),'op':condition['op'],
                           'threshold':condition['threshold'],'value':float(value),'passed':bool(ok)})
            passed=passed and bool(ok)
        gate_rows[name]={'lhs':spec['lhs'],'rhs':spec['rhs'],'meaning':spec.get('meaning'),
                         'passed':bool(passed),'detail':detail}
    net=bool(gate_rows.get('A2_minus_A0',{}).get('passed',False))
    inc=bool(gate_rows.get('A2_minus_A1',{}).get('passed',False))
    combination=gates.get('combination') if isinstance(gates.get('combination'),dict) else {}
    verdict=(combination.get('both','initial_positive') if (net and inc) else
             (combination.get('net_only','ranking_pressure_only') if net else
              combination.get('net_fail','closed_fixed_config')))
    acc=execution.complete(c,execution.ACCEPTANCE_PHASE)
    endp={arm:execution.complete(c,'formal-%s'%arm) for arm in arms}
    inf={arm:execution.complete(c,'infer-%s-A-emaC'%arm) for arm in arms}
    costs={'acceptance':{'updates':acc['actual_updates'],'roles':acc.get('roles'),'elapsed_seconds':acc['elapsed_seconds']},
           'formal':{arm:{'updates':endp[arm]['updates'],'training_wall_seconds':endp[arm]['training_wall_seconds'],
                          'epoch_wall_seconds':endp[arm]['epoch_wall_seconds'],
                          'peak_reserved_bytes':endp[arm]['peak_reserved_bytes']} for arm in arms},
           'endpoint_inference':{arm:{'elapsed_seconds':inf[arm]['elapsed_seconds'],
                                      'model_forwards':inf[arm]['model_forwards'],'images':inf[arm]['images'],
                                      'peak_reserved_bytes':inf[arm]['peak_reserved_bytes']} for arm in arms}}
    gpu_total=(costs['acceptance']['elapsed_seconds']
               + sum(row['training_wall_seconds'] for row in costs['formal'].values())
               + sum(row['elapsed_seconds'] for row in costs['endpoint_inference'].values()))
    costs['gpu_stage_wall_seconds_total']=gpu_total
    limit=float(c['budget_limits']['gpu_hours']) if 'budget_limits' in c else 12.0
    cost_passed=gpu_total<=limit*3600 if c['scope']=='PRODUCTION' else True
    comparison={'scores':values,'reference_scores':refs,'differences_pp':comparisons,'gates':gate_rows,
       'verdict':verdict,'verdict_meaning':combination,'quality':verdict,'cost':costs,'cost_passed':cost_passed,
       'units':'point_scores0..1; deltas pp converted once','official_metrics_separate':metricrefs,
       'scope':'CPU_SYNTHETIC' if c['scope']=='CPU_FIXTURE' else 'seed0_development_exploratory',
       'descriptive_note':('Three fixed arms from the same parent: A0 native, A1 random eligible partner, A2 best '
                           'eligible partner. Reporting only: A2-A0 is the overall net benefit, A2-A1 the increment '
                           'from selecting the best eligible partner, A1-A0 the effect of random ranking pressure, and '
                           'every arm is also compared against the certified parent; the retrained native A0 is '
                           'compared against the original G0 readout of the same recipe. No selection, no tuning, no '
                           'attribution beyond the pre-registered gates.')}
    publish(out/'COMPARISON.json',comparison)
    table=io.StringIO();writer=csv.writer(table);writer.writerow(['comparison','budget','metric','delta_pp'])
    for name,_,_ in pairs:
        for b in (600,300):
            for m in sorted(comparisons[name][str(b)]):
                writer.writerow([name,b,m,comparisons[name][str(b)][m]])
    for name,row in sorted(gate_rows.items()):
        writer.writerow(['gate_%s'%name,'-','passed' if row['passed'] else 'failed',''])
    with (out/'TABLE.csv').open('x') as f:f.write(table.getvalue())
    publish(out/'RESULT.json',{'status':'PASS','quality':verdict,
       'gates':{k:v['passed'] for k,v in gate_rows.items()},'cost_passed':cost_passed,
       'comparison':reference(out/'COMPARISON.json'),'metrics':metricrefs,
       'score_budget_calls':len(arms)*2,'no_bootstrap':True,'CUDA_initialized':torch.cuda.is_initialized()})
