"""Native single-arm dual-EMA fine-tune work and whole-image A readout; caller owns process.

r24 round.  D0 is the original G0 trajectory (native, no installed module).  Both EMA records
start from the same parent-EMA weights; EMA-R keeps counter 0 and EMA-C keeps the parent counter
36,480.  Every arm is checkpointed with both records and only FT012 EMA-C is scored.

The installed item of this round is the band ROI-resolution box distillation term: A1 feeds the
whole-image view (the canvas the student already saw, so the control carries no new information)
and A2 feeds the four raw-pixel tiles.  The r19 tiny-positives item is RETIRED: its install map is
all-None, so no arm installs it, while its modules remain present and inert.  The native arm
records the ROI band statistics with roi_selected = 0.
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
from . import tiny_positives_arm
from . import roi_ld
from . import roi_ld_arm

EMAS=('C',)                     # scored endpoints: FT012 EMA-C only; EMA-R is state only
ALT_OFFSET=ft.ALT_OFFSET
# CPU-fixture scaffolding only: the group-0 candidate of an even band row is offset by a quarter
# of its own width (overlap 0.6, above the 0.3 bar) and of an odd band row by a full width
# (overlap exactly 0), so both arms select the same rows while the control and the method select
# different boxes and the two ledgers mirror each other.
FIXTURE_SHIFT=0.25
FIXTURE_FAR=1.0
# CPU-fixture geometry.  The fixture runs on the PRODUCTION canvas so its band rows are real band
# rows under the frozen band definition (canvas area <= 1024 px^2 on 768x1344): at 128x128 the same
# rows sit on the band boundary and the strong augmentation can push them out, so the fixture would
# prove only the plumbing.  The raw fixture image is 540x960 with 10 px boxes, which the resize
# turns into roughly 14 px, i.e. about 196 px^2 - comfortably inside the band and comfortably above
# the 1 px drop floor, so a moderate zoom/crop variation cannot move a row out of the band.
FIXTURE_INPUT_HW=(768,1344)
FIXTURE_RAW_HW=(540,960)
FIXTURE_ANNOTATION=b'300,300,200,250,1,1,0,0\n100,100,10,10,1,2,0,0\n400,120,10,10,1,3,0,0\n700,200,200,200,0,0,0,0\n'


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
    config=Config(input_hw=FIXTURE_INPUT_HW,physical_batch_size=1,amp_dtype='float32') if fixture else Config()
    model,criterion,_=build(config,synthetic=True)
    mode=ft.ARM_INSTALL[arm]
    pos=ft.POS_INSTALL.get(arm)
    criterion.tiny_positives_declaration=tiny_positives_arm.declaration(pos)
    if pos is not None:
        tiny_positives_arm.install(model,criterion,pos)
    roi_mode=ft.ROI_INSTALL[arm]
    criterion.roi_ld_declaration=roi_ld_arm.declaration(roi_mode)
    roi_term=roi_ld_arm.install(model,criterion,roi_mode) if roi_mode is not None else None
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
    e.pos_mode=pos;e.roi_mode=roi_mode;e.roi_term=roi_term
    if pos is not None:
        tiny_positives_arm.assert_installed(e.model,e.criterion,pos)
    if roi_mode is not None:
        roi_ld_arm.assert_installed(e.model,e.criterion,roi_mode,term=roi_term)
    if mode is not None:
        from . import detail_path
        detail_path.assert_installed(e.model,mode);detail_path.assert_installed(e.ema.module,mode)
    e.execution_binding={**{k:c[k] for k in ('task_id','source_sha256','source_commit','contract_sha256','plan_sha256','data_sha256')},
        'scope':scope,'arm':arm,'install_mode':mode,'pos_mode':pos,'roi_mode':roi_mode,'parent_sha256':parent_sha,'runtime':runtime,'CPU_fixture':fixture,
        'dual_ema':{'ema_r_updates_start':0,'ema_c_updates_start':ft.ALT_OFFSET,'same_initial_module':True,
                    'initial_module_sha256':initial,'training_feedback':False,
                    'declared_checkpoint_passes':list(ft.CHECKPOINT_PASSES)}}
    holder.update(engine=e,binding=e.execution_binding)
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    if not fixture:torch.cuda.default_generators[0].manual_seed(0)
    require(e.completed_updates==e.ema.updates==0 and e.alt_ema.updates==ft.ALT_OFFSET and not e.optimizer.state,'FT fresh optimizer/EMA clocks')
    return e,runtime,initial


def fixture_roi_target(mode,target):
    """CPU-fixture-only ROI scaffolding; no teacher cache is reachable for synthetic members.

    Every band row of the augmented target receives two same-class candidate boxes tagged with
    their view group: group 0 (whole-image view) and group 1 (raw-pixel tiles).  The even-indexed
    band row gets a group-0 box offset by a quarter of its own width (overlap 0.6 >= tau) plus an
    exact group-1 box, so both armed arms select the row and select DIFFERENT boxes; the
    odd-indexed row gets a group-0 box offset by a full width (overlap exactly 0) plus an exact
    group-1 box, so the control arm records a below-bar row and the method arm a one-sided row.
    Only the four roi_* keys are added, so the audited augmentation identity of the delivered
    tensors is untouched.  The production path uses roi_ld_data.attach and the frozen cache.
    """
    boxes=target['boxes'];band=roi_ld.band_mask(boxes)
    rows=torch.nonzero(band,as_tuple=False).flatten().tolist()
    cand=[];crows=[];clabels=[];cgroups=[]
    for k,row in enumerate(rows):
        b=boxes[row].detach().clone();label=int(target['labels'][row])
        first=b.clone();first[0]=first[0]+(FIXTURE_SHIFT if k%2==0 else FIXTURE_FAR)*b[2]
        cand.append(first);crows.append(row);clabels.append(label);cgroups.append(0)
        cand.append(b);crows.append(row);clabels.append(label);cgroups.append(1)
    target['roi_boxes']=torch.stack(cand) if cand else torch.zeros((0,4),dtype=torch.float32)
    target['roi_rows']=torch.tensor(crows,dtype=torch.long)
    target['roi_labels']=torch.tensor(clabels,dtype=torch.long)
    target['roi_group']=torch.tensor(cgroups,dtype=torch.long)
    target['roi_mode']=mode
    target['roi_group_declared']=roi_ld.GROUP[mode]
    return target


class FixtureTrain(torch.utils.data.Dataset):
    def __init__(self,arm,epoch,start,limit):
        self.arm,self.epoch,self.start,self.limit=arm,epoch,start,limit
        self.roi_mode=ft.ROI_INSTALL[arm]
    def __len__(self):return 2*self.limit
    def __getitem__(self,index):
        from PIL import Image
        pos=self.start*2+index
        raw=parse_raw(FIXTURE_ANNOTATION if pos%2==0 else b'','cpu-ft-'+str(pos))
        x,t,tr=ft.Transform(self.arm,FIXTURE_INPUT_HW)(Image.fromarray(np.full(FIXTURE_RAW_HW,90+pos,np.uint8)),raw,
                    epoch=self.epoch,position=pos,image_id='cpu-ft-'+str(pos))
        if self.roi_mode is not None:fixture_roi_target(self.roi_mode,t)
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
    config=Config(input_hw=FIXTURE_INPUT_HW,physical_batch_size=1,amp_dtype='float32') if fixture else Config()
    twin,criterion,_=build(config,synthetic=True)
    if not fixture:
        binding=execution.verified(c['parent_binding']);r=c['parent_checkpoint']
        payload=native_state.read(r['path'],r['sha256'],binding)
        ft.initialize_from_parent(twin,payload,expected_binding=binding);del payload
        twin.to('cuda:0');criterion.to('cuda:0')
    return twin,criterion


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
    require(all(len(z) in (12,21) for z in r['microbatch_losses']),'FT native loss count')
    progress.save_audit(out,channel.binding(),r,role)
    return r


def save(e,out,name):
    event(out,'checkpoint_begin',name=name,update=e.completed_updates)
    rec=ft.save(e,Path(out)/name)
    rec.update(execution_binding=e.execution_binding,initial_parent=e.parent_sha256)
    publish(Path(rec['path']+'.complete.json'),rec)
    event(out,'checkpoint_complete',name=name,update=e.completed_updates,checkpoint=rec)
    return rec


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
       'arm':arm,'install_mode':ft.ARM_INSTALL[arm],'roi_mode':ft.ROI_INSTALL[arm],'updates':e.completed_updates,'epoch':c['epochs'],'checkpoint':last,'initial_checkpoint':zero,
       'checkpoint_passes':sorted(saves),'checkpoints_written':[q['path'] for q in written],
       'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
       'binding':reference(out/'binding.json'),'initial_model_sha256':initial,'used_acceptance_state':False,
       'dual_ema':{'ema_r_updates':int(e.ema.updates),'ema_c_updates':int(e.alt_ema.updates),'ema_c_offset':ft.ALT_OFFSET,
                   'rng_neutral_alt_ema_updates':int(e.alt_rng_neutral_updates),
                   'ema_r_module_sha256':tensor_digest(e.ema.module.state_dict().items()),
                   'ema_c_module_sha256':tensor_digest(e.alt_ema.module.state_dict().items())},
       'roi_state':roi_ld_arm.state(e),
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
        groups[0].append(TF.pil_to_tensor(TF.resize(image,list(FIXTURE_INPUT_HW))).float()/255.)
        if sliced:
            for tile in tile_windows(384,512):groups[tile.view_id].append(TF.pil_to_tensor(TF.resize(image.crop(tile.crop_box),list(FIXTURE_INPUT_HW))).float()/255.)
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


def roi_ledger(r):
    """Aggregate the ROI observations of one update; the ledger identity must close."""
    rows=[row for diag in r['normalizers'] for row in diag if isinstance(row,dict) and 'roi_band_gt' in row]
    require(rows,'FT acceptance roi observations missing')
    keys=('roi_band_gt','roi_eligible','roi_selected','roi_no_candidate','roi_no_class','roi_below_iou','roi_one_sided','roi_unmatched_band_gt')
    require(all(all(isinstance(row[k],int) and row[k]>=0 for k in keys) for row in rows),'FT acceptance roi counts differ')
    totals={k:sum(int(row[k]) for row in rows) for k in keys}
    totals['rows']=len(rows)
    totals['roi_target_iou_sum']=sum(float(row['roi_target_iou_sum']) for row in rows)
    totals['roi_other_group_iou_sum']=sum(float(row['roi_other_group_iou_sum']) for row in rows)
    totals['set_digests']=[row['roi_set_digest'] for row in rows]
    totals['modes']=sorted({row.get('roi_mode') for row in rows},key=str)
    totals['identity']=(totals['roi_selected']+totals['roi_below_iou']+totals['roi_no_candidate']
                        +totals['roi_no_class']+totals['roi_one_sided']+totals['roi_unmatched_band_gt'])
    require(totals['identity']==totals['roi_band_gt'],'FT acceptance roi ledger does not account every band row')
    return totals


def accept_roi(c,out,holder):
    """r24 engineering acceptance: exactly 4 optimizer updates covering A0, A1, A2 and empty.

    One launch, four unique optimizer updates in total.  Each case is an independent single-update
    sub-run built from the certified parent through the frozen ``construct``, so the acceptance
    state can never enter the formal training.  Per case: the production batch and precision
    configuration, every loss finite, the native loss-key count and key set, a nonzero finite
    gradient norm, and the ROI ledger recorded in the frozen diagnostics.  A0 is native (no term
    installed, roi_selected = 0 with the band statistics still recorded).  A1 and A2 are the two
    armed modes; they must supervise the SAME number of rows (common support) while selecting
    DIFFERENT boxes, and their ledgers must mirror each other on the fixture's fixed band rows.
    The A2 case also saves a chain checkpoint whose restore into a fresh engine must reproduce the
    full state and the ROI term state exactly.
    """
    started=time.monotonic();runtime=None

    def one_case(case,arm,supervised,role):
        nonlocal runtime
        if runtime is None:
            e,runtime,initial=construct(c,out,arm,'gpu_acceptance',holder)
        else:
            e,_,initial=construct(c,out,arm,'gpu_acceptance',holder,runtime=runtime)
        mode=ft.ROI_INSTALL[arm];term=getattr(e.criterion,'roi_ld',None)
        if mode is None:
            require(term is None and getattr(e,'roi_mode',None) is None,'FT acceptance native arm carries a roi term: '+case)
            roi_before=None
        else:
            require(term is not None and term.mode==mode,'FT acceptance roi term missing: '+case)
            roi_before=int(term.state_dict()['targets'])
        windows=list(loader(c,arm,1,limit=1));require(len(windows)==1,'FT acceptance window count differs')
        x,t,tr=windows[0]
        if supervised:
            r=update(c,out,arm,1,0,e,x,t,tr,holder,role)
        else:
            before=[len(row['gt_ids']) for row in t]
            blanked=[{'boxes':row['boxes'][:0],'labels':row['labels'][:0],'raw_area':row['raw_area'][:0],
                      'gt_ids':[],**{k:v for k,v in row.items()
                                     if k not in ('boxes','labels','raw_area','gt_ids')}} for row in t]
            admit(c,out)
            r=e.train_window(x,blanked,capture=True);r['input_sha256']=digest({'emptied':before})
            require(e.completed_updates==e.ema.updates==r['update'] and e.phase==0 and r['optimizer_steps']==r['ema_updates']==1 and r['finite'],'FT empty-case update clocks')
            require(e.alt_ema.updates==r['update']+ft.ALT_OFFSET and r.get('ema_alt_updates')==1,'FT empty-case EMA-C update')
            progress.save_audit(out,channel.binding(),r,role)
        roi_after=None if term is None else int(term.state_dict()['targets'])
        counts=[len(row) for row in r['microbatch_losses']]
        keys=[sorted(row) for row in r['microbatch_losses']]
        require(all(count in (12,21) for count in counts),'FT acceptance loss-key count differs: %s'%counts)
        require(r['finite'] is True and r['gradient_norm_before_clip']>0.0,'FT acceptance gradient/finiteness differs')
        ledger=roi_ledger(r)
        ledger['roi_targets_before']=roi_before;ledger['roi_targets_after']=roi_after
        if supervised:
            require(ledger['roi_band_gt']>0,'FT acceptance fixed window has no ROI-band ground truth: '+case)
            require(ledger['modes']==[mode],'FT acceptance roi mode differs: '+case)
            if mode is None:
                require(ledger['roi_selected']==0 and ledger['roi_below_iou']==0 and ledger['roi_one_sided']==0
                        and all(d is None for d in ledger['set_digests']),'FT acceptance native arm selected an roi target: '+case)
            else:
                require(ledger['roi_selected']>0,'FT acceptance armed arm selected nothing on the fixed window: '+case)
                require(roi_after==roi_before+ledger['roi_selected'],'FT acceptance roi target counter differs: '+case)
        else:
            require(ledger['roi_band_gt']==0 and ledger['roi_selected']==0 and ledger['modes']==[mode],
                    'FT acceptance empty supervision produced roi work')
        return e,initial,{'case':case,'arm':arm,'supervised':supervised,'loss_key_counts':counts,
                          'loss_key_sets':keys,'roi':ledger,'roi_mode':getattr(e,'roi_mode',None),
                          'gradient_norm_before_clip':r['gradient_norm_before_clip'],
                          'parameter_device':str(next(e.model.parameters()).device)}

    # A0: native supervised update (no term object; the ROI band statistics are still recorded)
    e0,initial,checks_a0=one_case('A0','A0',True,'A0')
    holder.pop('engine');holder.pop('input',None);del e0;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    # A1: the information-free control (whole-image-view boxes)
    e1,_,checks_a1=one_case('A1','A1',True,'A1')
    holder.pop('engine');holder.pop('input',None);del e1;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    # A2: the method (raw-pixel-tile boxes), then the chain save/restore of the term state
    e2,_,checks_a2=one_case('A2','A2',True,'A2')
    # Common support fixes the same rows and the same pair count for both armed arms.  The
    # below-bar and one-sided counters are DESIGNED to mirror rather than to agree: the control
    # arm's group carries the box the method arm's group lacks, and vice versa.  They are asserted
    # as a mirror, and the group-agnostic counters as an equality.
    shared_keys=('roi_band_gt','roi_eligible','roi_selected','roi_no_candidate','roi_no_class',
                 'roi_unmatched_band_gt')
    require(all(checks_a1['roi'][key]==checks_a2['roi'][key] for key in shared_keys),
            'FT acceptance A1/A2 common-support ledger differs')
    require(checks_a1['roi']['roi_below_iou']==checks_a2['roi']['roi_one_sided']
            and checks_a1['roi']['roi_one_sided']==checks_a2['roi']['roi_below_iou'],
            'FT acceptance A1/A2 below-bar and one-sided counters are not the expected mirror')
    require(checks_a1['loss_key_sets']==checks_a2['loss_key_sets']==checks_a0['loss_key_sets'],
            'FT acceptance loss-key family differs across arms')
    require(checks_a1['roi']['set_digests']!=checks_a2['roi']['set_digests'],
            'FT acceptance A1/A2 selected identical boxes')
    cp=save(e2,out,'C1-A2.pth')
    expected_state=ft.capture(e2)
    roi_before=e2.criterion.roi_ld.state_dict()
    holder.pop('engine');holder.pop('input',None);del e2;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()
    clone,_,initial2=construct(c,out,'A2','gpu_acceptance',holder,runtime=runtime)
    require(initial==initial2,'FT acceptance independent initial bytes differ')
    restored=ft.restore(clone,cp)
    require(native_state.exact_equal(expected_state,ft.capture(clone)),'FT acceptance full state/RNG restore mismatch')
    term_after=getattr(clone.criterion,'roi_ld',None)
    require(term_after is not None and term_after.mode=='R2','FT acceptance A2 roi term missing after restore')
    require(int(term_after.state_dict()['targets'])==int(roi_before['targets']),'FT acceptance A2 roi counter differs after restore')
    checks_a2['restore']={'full_state_RNG_exact':bool(restored.get('full_state_RNG_exact')),
                          'roi_state_restore_exact':True,'roi_targets':int(roi_before['targets']),
                          'roi_state':roi_ld_arm.state(clone)}
    holder.pop('engine');holder.pop('input',None);del clone;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    # empty: supervision emptied after augmentation, under the A2 arm
    e3,_,checks_empty=one_case('empty','A2',False,'empty')
    checks_empty['note']='empty supervision runs under the A2 arm; no ground truth means no band row and no roi target'
    holder.pop('engine');holder.pop('input',None);del e3;gc.collect()
    if c['scope']=='PRODUCTION':torch.cuda.empty_cache()

    publish(out/'RESULT.json',{'status':'PASS','actual_updates':4,'sequence':[1,1,1,1],
        'roles':['A0','A1','A2','empty'],'cases':{'A0':checks_a0,'A1':checks_a1,'A2':checks_a2,'empty':checks_empty},
        'a1_a2_common_support':True,'a1_a2_selected':checks_a1['roi']['roi_selected'],
        'a1_a2_control_below_bar_rows':checks_a1['roi']['roi_below_iou'],
        'a1_a2_method_one_sided_rows':checks_a2['roi']['roi_one_sided'],
        'a1_a2_ledger_mirrored':True,'a1_a2_selected_boxes_differ':True,'native_records_zero_selected':True,
        'initial_model_sha256':initial,'full_state_RNG_exact':True,'roi_state_restore_exact':True,
        'state_usable_for_formal':False,'single_arm_acceptance_state_usable_for_formal':False,
        'roi_install':{a:ft.ROI_INSTALL[a] for a in ft.ARMS},
        'pos_install':{a:ft.POS_INSTALL[a] for a in ft.ARMS},
        'loss_key_counts_expected':{'per_microbatch_membership':[12,21],
            'note':'the denoising family appears per micro-batch depending on the window content; the extra regression-only pair adds no loss key'},
        'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
        'CPU_fixture':c['scope']=='CPU_FIXTURE','cuda_initialized':bool(torch.cuda.is_initialized()),
        'elapsed_seconds':time.monotonic()-started})


def evaluation_roi(c,out):
    """r24 fixed-endpoint evaluation: six budget scorings, FT012 EMA-C only.

    Three arms x two budgets, view A, batch 4, 548 development images, top600 primary and top300
    secondary, FP32, no selection and no tuning.  Every arm is checkpointed with EMA-R and EMA-C;
    only the FT012 EMA-C endpoint is scored, exactly six times.  The certified parent anchor is
    reported separately from the retrained native A0 and a trained A0 is NOT required to
    reproduce the parent score.  Cost is the measured wall clock of the fixed inference stage; the
    pre-registered threshold is applied to the measured A2/A0 ratio and no inference ratio is
    pre-written anywhere in this round.
    """
    from ..r0.evaluation import score_budget,POLICY
    from infer_seed import bridge_values,_group,_prefix
    plan=ft.read_plan(c['plan']);arms=list(ft.ARMS);gates=plan['gates']
    cost_gate=gates['cost']
    require(cost_gate['metric']=='inference_stage_seconds' and cost_gate['lhs']=='A2' and cost_gate['rhs']=='A0'
            and abs(float(cost_gate['threshold'])-1.05)<1e-12,'FT cost gate changed')
    require(plan['cost_policy']['predeclared_ratio'] is None
            and plan['cost_policy']['ratio_source']=='measured_from_the_inference_stage','FT cost policy predeclares a ratio')
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
            z['FT_binding']={'endpoint':'C','view':'A','budget':b,'arm':arm,
                'checkpoint_sha256':result['checkpoint_sha256'],'ema_module_sha256':result['ema_module_sha256'],
                'ema_updates':result['ema_updates'],'prediction':result['predictions'][str(b)],
                'CPU_fixture':c['scope']=='CPU_FIXTURE'}
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
        if not isinstance(spec,dict) or 'conditions' not in spec:continue
        key='%s_minus_%s'%(spec['lhs'],spec['rhs']);detail=[];passed=True
        for condition in spec['conditions']:
            metric,budget=condition['metric'],str(condition['budget'])
            threshold=condition.get('threshold_pp',condition.get('threshold'))
            require(metric in comparisons[key][budget] and isinstance(threshold,(int,float)),
                    ('FT gate metric/threshold missing',name,metric))
            value=comparisons[key][budget][metric]
            ok=value>=threshold if condition['op']=='>=' else value<=threshold
            detail.append({'metric':metric,'budget':int(budget),'op':condition['op'],
                           'threshold':float(threshold),'value':float(value),'passed':bool(ok)})
            passed=passed and bool(ok)
        gate_rows[name]={'lhs':spec['lhs'],'rhs':spec['rhs'],'meaning':spec.get('meaning'),
                         'passed':bool(passed),'detail':detail}
    net=bool(gate_rows.get('A2_minus_A0',{}).get('passed',False))
    inc=bool(gate_rows.get('A2_minus_A1',{}).get('passed',False))
    combination=gates.get('combination') if isinstance(gates.get('combination'),dict) else {}
    verdict=(combination.get('both','initial_positive_net_and_increment') if (net and inc) else
             (combination.get('net_only','single_seed_preliminary_net_benefit') if net else
              combination.get('net_fail','closed_fixed_config')))
    acc=execution.complete(c,execution.ACCEPTANCE_PHASE)
    endp={arm:execution.complete(c,'formal-%s'%arm) for arm in arms}
    inf={arm:execution.complete(c,'infer-%s-A-emaC'%arm) for arm in arms}
    infer_seconds={arm:float(inf[arm]['elapsed_seconds']) for arm in arms}
    train_seconds={arm:float(endp[arm]['training_wall_seconds']) for arm in arms}
    measured={'inference_stage_seconds':infer_seconds,
              'inference_stage_A2_over_A0':infer_seconds['A2']/infer_seconds['A0'],
              'inference_stage_A1_over_A0':infer_seconds['A1']/infer_seconds['A0'],
              'end_to_end_seconds_including_training':{arm:infer_seconds[arm]+train_seconds[arm] for arm in arms},
              'acceptance_seconds':float(acc['elapsed_seconds']),'predeclared_ratio':None,
              'ratio_source':'measured_from_the_inference_stage',
              'note':'cost is recorded from the measured stage wall clocks of this run; no inference ratio is pre-written'}
    measured['end_to_end_including_training_A2_over_A0']=(measured['end_to_end_seconds_including_training']['A2']
                                                         /measured['end_to_end_seconds_including_training']['A0'])
    cost_passed=bool(measured['inference_stage_A2_over_A0']<=float(cost_gate['threshold'])) if c['scope']=='PRODUCTION' else True
    costs={'acceptance':{'updates':acc['actual_updates'],'roles':acc.get('roles'),'elapsed_seconds':acc['elapsed_seconds']},
           'formal':{arm:{'updates':endp[arm]['updates'],'training_wall_seconds':endp[arm]['training_wall_seconds'],
                          'epoch_wall_seconds':endp[arm]['epoch_wall_seconds'],
                          'peak_reserved_bytes':endp[arm]['peak_reserved_bytes']} for arm in arms},
           'endpoint_inference':{arm:{'elapsed_seconds':inf[arm]['elapsed_seconds'],
                                      'model_forwards':inf[arm]['model_forwards'],'images':inf[arm]['images'],
                                      'peak_reserved_bytes':inf[arm]['peak_reserved_bytes']} for arm in arms},
           'measured_ratios':measured,'gate':cost_gate}
    costs['gpu_stage_wall_seconds_total']=(costs['acceptance']['elapsed_seconds']+sum(train_seconds.values())
                                           +sum(infer_seconds.values()))
    comparison={'scores':values,'reference_scores':refs,'differences_pp':comparisons,'gates':gate_rows,
       'verdict':verdict,'verdict_meaning':combination,'quality':verdict,'cost':costs,'cost_passed':cost_passed,
       'evaluation_definition':{'scorings':ordinal,'ema':'C','ema_r':'state_only',
                                'parent_anchor':'separate_from_trained_A0','trained_A0_required_equal_to_parent':False},
       'units':'point_scores0..1; deltas pp converted once','official_metrics_separate':metricrefs,
       'scope':'CPU_SYNTHETIC' if c['scope']=='CPU_FIXTURE' else 'seed0_development_exploratory',
       'descriptive_note':('Three fixed arms from the same parent: A0 native, A1 the whole-image-view control '
                           '(its target box is re-encoded from the canvas the student already saw, so it carries no '
                           'new information) and A2 the raw-pixel-tile method.  Both armed arms supervise exactly the '
                           'same rows through COMMON SUPPORT, so only the pixel provenance of the target box varies; '
                           'loss VALUES, gradient magnitudes and appended-pair regression quality are NOT claimed equal. '
                           'The six budget scorings evaluate FT012 EMA-C only: EMA-R is maintained and checkpointed by '
                           'every arm and is never scored.  Reporting only: A2-A0 is the single-seed net benefit at this '
                           'starting point and this short schedule; A2-A1 isolates the information increment; A1-A0 is the '
                           'control effect.  The certified parent anchor is carried separately and the retrained native A0 '
                           'is NOT required to reproduce the parent score: the trained A0 minus parent distance is a reading '
                           'of the short-retrain platform movement, and A0 minus original_G0 is the published-readout '
                           'difference.  No gate, no selection, no tuning, no bootstrap, no holdout.')}
    publish(out/'COMPARISON.json',comparison)
    table=io.StringIO();writer=csv.writer(table);writer.writerow(['comparison','budget','metric','delta_pp'])
    for name,_,_ in pairs:
        for b in (600,300):
            for m in sorted(comparisons[name][str(b)]):
                writer.writerow([name,b,m,comparisons[name][str(b)][m]])
    for name,row in sorted(gate_rows.items()):
        writer.writerow(['gate_%s'%name,'-','passed' if row['passed'] else 'failed',''])
    writer.writerow(['gate_cost','-','passed' if cost_passed else 'failed',measured['inference_stage_A2_over_A0']])
    with (out/'TABLE.csv').open('x') as f:f.write(table.getvalue())
    publish(out/'RESULT.json',{'status':'PASS','quality':verdict,
       'gates':{k:v['passed'] for k,v in gate_rows.items()},'cost_passed':cost_passed,
       'comparison':reference(out/'COMPARISON.json'),'metrics':metricrefs,
       'score_budget_calls':len(arms)*2,'scored_endpoint':'FT012_EMA_C','ema_r_scored':False,
       'measured_cost_ratios':measured,'no_bootstrap':True,'CUDA_initialized':torch.cuda.is_initialized()})
