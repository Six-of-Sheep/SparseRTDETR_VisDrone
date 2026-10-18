"""Native fine-tune work and unchanged fixed A/C readout; caller owns process."""
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
    if fixture:initial=tensor_digest(model.state_dict().items());parent_sha='a'*64
    else:
        binding=execution.verified(c['parent_binding']);r=c['parent_checkpoint']
        require(Path(r['path']).stat().st_size==r['bytes'],'FT parent size')
        payload=native_state.read(r['path'],r['sha256'],binding)
        initial=ft.initialize_from_parent(model,payload,expected_binding=binding);del payload
        require(initial==c['initial_model_sha256'],'FT initial EMA/model bytes differ')
        model.to('cuda:0');criterion.to('cuda:0');parent_sha=r['sha256']
    opt,ema=ft.make_optimizer_ema(model,config)
    e=ft.FineTuneEngine(model,criterion,opt,ema,config,arm=arm,plan_sha256=c['plan_sha256'],parent_sha256=parent_sha)
    e.execution_binding={**{k:c[k] for k in ('task_id','source_sha256','source_commit','contract_sha256','plan_sha256','data_sha256')},
        'scope':scope,'arm':arm,'parent_sha256':parent_sha,'runtime':runtime,'CPU_fixture':fixture}
    holder.update(engine=e,binding=e.execution_binding)
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    if not fixture:torch.cuda.default_generators[0].manual_seed(0)
    require(e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'FT fresh optimizer/EMA clock')
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


def acceptance(c,out,arm,holder):
    e,runtime,initial=construct(c,out,arm,'gpu_acceptance',holder)
    publish(out/'binding.json',e.execution_binding)
    results=[];started=time.monotonic()
    for win,(x,t,tr) in enumerate(loader(c,arm,1,limit=2)):
        results.append(update(c,out,arm,1,win,e,x,t,tr,holder,'direct'))
        if win==0:cp=save(e,out,'C1.pth')
    require(len(results)==2,'FT fixed acceptance window count')
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
    publish(out/'RESULT.json',{'status':'PASS','arm':arm,'actual_updates':3,'sequence':[1,2,2],
       'input_DN_output_loss_preclip_gradient_exact':True,'full_state_RNG_exact':True,'initial_model_sha256':initial,
       'state_usable_for_formal':False,'restored':restored,'checkpoints':[cp,direct,last],
       'peak_allocated_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_allocated(),
       'peak_reserved_bytes':0 if c['scope']=='CPU_FIXTURE' else torch.cuda.max_memory_reserved(),
       'elapsed_seconds':time.monotonic()-started,'CPU_fixture':c['scope']=='CPU_FIXTURE'})


def formal(c,out,arm,holder):
    e,runtime,initial=construct(c,out,arm,'formal',holder)
    accepted=execution.complete(c,'accept-'+arm)
    require(initial==accepted['initial_model_sha256'] and e.completed_updates==0 and not e.optimizer.state,'FT formal did not restart independently from parent')
    publish(out/'binding.json',e.execution_binding);zero=save(e,out,'checkpoint-FT000.pth')
    started=time.monotonic();times=[]
    for epoch in range(1,c['epochs']+1):
        execution.load(c['contract_path'])
        require(e.completed_updates==(epoch-1)*c['updates_per_epoch'],'FT epoch cursor differs')
        event(out,'epoch_begin',epoch=epoch,update=e.completed_updates)
        tick=time.monotonic();n=0
        for win,(x,t,tr) in enumerate(loader(c,arm,epoch)):
            update(c,out,arm,epoch,win,e,x,t,tr,holder,'direct');n+=1
        require(n==c['updates_per_epoch'] and e.completed_updates==epoch*c['updates_per_epoch'],'FT pass workload incomplete')
        last=save(e,out,f'checkpoint-FT{epoch:03d}.pth');times.append(time.monotonic()-tick)
        event(out,'epoch_complete',epoch=epoch,update=e.completed_updates,checkpoint=last)
    require(e.completed_updates==e.ema.updates==c['total_updates'],'FT endpoint count mismatch')
    endpoint={'status':'FT012_EMA_LOCKED' if c['scope']=='PRODUCTION' else 'CPU_SYNTHETIC_ENDPOINT',
       'arm':arm,'updates':e.completed_updates,'epoch':c['epochs'],'checkpoint':last,'initial_checkpoint':zero,
       'binding':reference(out/'binding.json'),'initial_model_sha256':initial,'used_acceptance_state':False,
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


def inference(c,out,arm,view,holder):
    from slice_core import tile_windows,map_crop_xyxy,select_view_candidates,merge_candidates
    from infer_seed import prediction
    from .geometry import forward_guards
    from ..localization.consumer_geometry import positive_box,corners
    e,_,_=construct(c,out,arm,'formal',holder)
    endpoint=execution.complete(c,'formal-'+arm);rec=endpoint['checkpoint']
    ft.restore(e,rec);require(e.completed_updates==c['total_updates'],'FT inference endpoint not locked')
    model=e.ema.module;model.eval();before=ft.capture(e);data=development(c);sliced=view=='C'
    fullrows=data.records;ids=[r['stable_image_id'] for r in fullrows]
    fixedrows=[data.by_id[i] for i in c['fixed64']]
    own=None
    if sliced:
        a=execution.complete(c,'infer-'+arm+'-A');raw_ref=a['raw']
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
                event(out,'inference_batch',kind=kind,batch=i+1,ids=[z['stable_image_id'] for z in current],elapsed_seconds=elapsed,forward_calls=calls)
    require(native_state.exact_equal(before,ft.capture(e)) and all(p.grad is None for p in e.model.parameters()),'FT inference altered full state/RNG/gradients')
    refs={}
    for b in (600,300):
        q=out/f'top{b}-predictions.json';publish(q,preds[b]);refs[str(b)]=reference(q)
    result={'status':'PASS','arm':arm,'view':view,'images':len(ids),'checkpoint_sha256':rec['sha256'],
       'predictions':refs,'profile_e2e_seconds':sum(profile),'profile_batches':profile,'profile_images':len(fixedrows),
       'warmup_batches':2,'batch_size':4,'model_forwards':calls,'detector_updates':0,'full_state_RNG_exact':True,
       'whole_exact':len(ids) if sliced else None,'elapsed_seconds':time.monotonic()-started,
       'peak_allocated_bytes':0 if e.device.type=='cpu' else torch.cuda.max_memory_allocated(),
       'peak_reserved_bytes':0 if e.device.type=='cpu' else torch.cuda.max_memory_reserved()}
    if not sliced:
        q=out/'raw-A.pth'
        with q.open('xb') as f:torch.save({'image_ids':ids,'checkpoint_sha256':rec['sha256'],
             **{key:torch.cat([r[key] for r in rawrows]) for key in ('pred_logits','pred_boxes')}},f);f.flush();os.fsync(f.fileno())
        result['raw']=reference(q)
    publish(out/'RESULT.json',result)


def evaluation(c,out):
    from ..r0.evaluation import score_budget,POLICY
    from infer_seed import bridge_values,_group,_prefix
    data=development(c);rows=data.records;ids=[r['stable_image_id'] for r in rows]
    if c['scope']=='PRODUCTION':require(execution.verified(c['evaluation_policy'])==POLICY,'FT evaluation policy changed')
    parsed=[data.raw(r) for r in rows];values={};metricrefs={};ordinal=0
    for arm in ft.ARMS:
        values[arm]={};metricrefs[arm]={}
        for view in ('A','C'):
            result=execution.complete(c,'infer-'+arm+'-'+view);values[arm][view]={};metricrefs[arm][view]={};grouped={}
            for b in (600,300):
                records=execution.verified(result['predictions'][str(b)])
                grouped[b]=_group(records,ids,b,whole=view=='A',exact=view=='A')
                z=score_budget(rows,parsed,records,Path(c['source_root']))
                z['FT_binding']={'arm':arm,'view':view,'budget':b,'checkpoint_sha256':result['checkpoint_sha256'],'prediction':result['predictions'][str(b)],'CPU_fixture':c['scope']=='CPU_FIXTURE'}
                q=out/f'{arm}-{view}-top{b}-metrics.json';publish(q,z)
                values[arm][view][str(b)]=bridge_values(z);metricrefs[arm][view][str(b)]=reference(q);ordinal+=1
                event(out,'evaluation_budget_complete',ordinal=ordinal,arm_name=arm,view=view,budget=b,images=len(ids))
            _prefix(grouped[600],grouped[300],ids,whole=view=='A')
    parents={v:{str(b):bridge_values(execution.verified(c['parent_metrics'][v][str(b)])) for b in (600,300)} for v in ('A','C')} if c['scope']=='PRODUCTION' else copy.deepcopy(values['T-control'])
    plan=read(c['plan']);comparisons={};gates=[]
    for name,lhs,rhs in [('C_T-slice_minus_T-control',values['T-slice']['C'],values['T-control']['C']),
       ('C_T-slice_minus_original_C_seed0',values['T-slice']['C'],parents['C']),
       ('A_T-slice_minus_T-control',values['T-slice']['A'],values['T-control']['A']),
       ('A_T-slice_minus_original_A_seed0',values['T-slice']['A'],parents['A'])]:
        comparisons[name]={str(b):{m:100*(lhs[str(b)][m]-rhs[str(b)][m]) for m in lhs[str(b)]} for b in (600,300)}
        for key,minimum in plan['gates_pp'][name].items():
            budget,metric=key.split('_',1);value=comparisons[name][budget][metric]
            gates.append({'comparison':name,'metric':key,'delta_pp':value,'minimum_pp':minimum,'passed':value>=minimum})
    control_vs_parent={v:{str(b):{m:100*(values['T-control'][v][str(b)][m]-parents[v][str(b)][m]) for m in parents[v][str(b)]} for b in (600,300)} for v in ('A','C')}
    costs={}
    for arm in ft.ARMS:
        a=execution.complete(c,'infer-'+arm+'-A');z=execution.complete(c,'infer-'+arm+'-C');ratio=z['profile_e2e_seconds']/a['profile_e2e_seconds']
        costs[arm]={'A_e2e_seconds':a['profile_e2e_seconds'],'C_e2e_seconds':z['profile_e2e_seconds'],'C_over_A':ratio,'passed':ratio<=6,'A_peak_reserved':a['peak_reserved_bytes'],'C_peak_reserved':z['peak_reserved_bytes']}
    original_C_seconds=c['parent_C_profile_e2e_seconds'] if c['scope']=='PRODUCTION' else costs['T-control']['C_e2e_seconds']
    ratio=costs['T-slice']['C_e2e_seconds']/original_C_seconds
    costs['T-slice_C_over_original_C']={'ratio':ratio,'maximum':1.10,'passed':ratio<=1.10,'original_C_seconds':original_C_seconds,'original_measurement_reused':c['scope']=='PRODUCTION'}
    require(len(gates)==28,'FT frozen gate count differs')
    comparison={'scores':values,'parent_scores':parents,'differences_pp':comparisons,'T-control_minus_parent_pp':control_vs_parent,
       'gates':gates,'quality':'PASS' if all(g['passed'] for g in gates) else 'FAIL','cost':costs,
       'cost_passed':all(x['passed'] for x in costs.values()),'units':'point_scores0..1; deltas pp converted once',
       'official_metrics_separate':metricrefs,'scope':'CPU_SYNTHETIC' if c['scope']=='CPU_FIXTURE' else 'seed0_development_exploratory'}
    publish(out/'COMPARISON.json',comparison)
    table=io.StringIO();writer=csv.writer(table);writer.writerow(['comparison','metric','delta_pp','minimum_pp','passed'])
    for g in gates:writer.writerow([g[k] for k in ('comparison','metric','delta_pp','minimum_pp','passed')])
    with (out/'TABLE.csv').open('x') as f:f.write(table.getvalue())
    publish(out/'RESULT.json',{'status':'PASS','quality':comparison['quality'],'cost_passed':comparison['cost_passed'],
       'comparison':reference(out/'COMPARISON.json'),'metrics':metricrefs,'score_budget_calls':8,'no_bootstrap':True,'CUDA_initialized':torch.cuda.is_initialized()})
