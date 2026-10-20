"""Stage implementations shared by production CLI and explicit CPU fixtures.

The CPU profile uses synthetic pixels/weights and shortened workloads. It is
never a real training-data, BF16, GPU-resource, or scientific-quality gate.
"""
from __future__ import annotations
import copy, gc, io, json, math, os, random, statistics, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import distill as kd, distill_runtime as rt
from .distill_engine import Engine, capture
from .model import Config, native_objects
from .criterion import adapt
from .checkpoint import exact_equal
from .geometry import forward_guards
from ..baseline import training_v2b as native
from ..baseline.training_v2b_checkpoint import _safe
from ..localization.data import RawRoleDataset, RawAugmentation, parse_raw, read_bound, training_loader, collate_window
from ..localization.runio import require


def torch_ref(ref):
    p=Path(ref['path'])
    require(rt.reference(p)==ref,'tensor file SHA/size differs')
    return torch.load(p,map_location='cpu',weights_only=True)


def save_tensor(path,value):
    path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
    require(not path.exists() and not partial.exists(),'tensor output already consumed')
    with partial.open('xb') as f:
        torch.save(_safe(value),f);f.flush();os.fsync(f.fileno())
    os.link(partial,path);partial.unlink()
    return rt.reference(path)


def data(c,role):
    """Return exact rows and the ORIGINAL role reader in production."""
    d=c['data'][role]
    if c['profile']=='LAB_GPU_ONCE':
        if role=='train_core':
            rows=kd.train_members(d['manifest']['path'])
            reader=RawRoleDataset(role=role,manifest_path=d['manifest']['path'],images_path=d['roots']['images'],
                annotations_path=d['roots']['annotations'],root_binding=d['roots'],seed=0)
            require(reader.records==rows,'training constructor/manifest mapping differs')
        else:
            from ..r0.evaluation_data import Development548
            reader=Development548(dict(manifest=d['manifest']['path'],roots=d['roots']))
        require(rt.reference(d['manifest']['path'])==d['manifest'],'dataset manifest reference differs')
        return reader.records,reader
    # Separate six-member fixture, not a fallback for failed production admission.
    require(c['profile']=='CPU_FIXTURE_ONLY','fixture dataset authority missing')
    value=rt.exact(d['manifest']);rows=value['records']
    require(value['role']==role and len(rows)==d['count'] and len({r['stable_image_id'] for r in rows})==len(rows),'fixture membership differs')
    class FixtureReader:
        records=rows
        def bound_bytes(self,row):
            require(row in rows,'fixture cross-member read')
            return (read_bound(Path(d['roots']['images'])/Path(row['relative_path']).name,row['image_sha256'],row['image_size_bytes']),
                read_bound(Path(d['roots']['annotations'])/Path(row['annotation_relative_path']).name,row['annotation_sha256'],row['annotation_size_bytes']))
        def raw(self,row):return parse_raw(self.bound_bytes(row)[1],row['stable_image_id'],expected_rows=row['raw_annotation_rows'])
        def item(self,row,position):
            im=original(self,row)
            try:return RawAugmentation(rt.profile(c)['input_hw'])(im,self.raw(row),epoch=120,seed=0,position=position,image_id=row['stable_image_id'],training=False)[:2]
            finally:im.close()
    return rows,FixtureReader()


def original(reader,row):
    if hasattr(reader,'bound_bytes'):raw=reader.bound_bytes(row)[0]
    else:
        spec=reader.spec
        raw=read_bound(Path(spec['roots']['images'])/Path(row['relative_path']).name,row['image_sha256'],row['image_size_bytes'])
    with Image.open(io.BytesIO(raw)) as source:
        im=source.convert('RGB');im.load()
    require((im.width,im.height)==(row['width'],row['height']),'decoded member dimensions differ')
    return im


def parsed(reader,row):
    if hasattr(reader,'raw'):return reader.raw(row)
    return parse_raw(reader.bound_bytes(row)[1],row['stable_image_id'],expected_rows=row['raw_annotation_rows'])


def admission(c,out,emit,holder):
    counts={};row_ids={}
    for role,event in (('train_core','training_members'),('development','development_members')):
        rows,reader=data(c,role);audits=[]
        for i,row in enumerate(rows):
            im=original(reader,row);im.close();p=parsed(reader,row)
            audits.append(dict(image_id=row['stable_image_id'],image_sha256=row['image_sha256'],
                annotation_sha256=row['annotation_sha256'],physical_rows=len(p['rows'])))
            if i%32==0:live(c,out,gpu=False)
        counts[role]=len(rows);row_ids[role]=[r['stable_image_id'] for r in rows]
        rt.publish(out/(role+'-files.json'),audits);emit.emit(event,images=len(rows),members_sha256=rt.digest(row_ids[role]))
    require(not set(row_ids['train_core'])&set(row_ids['development']),'training/development member conflict')
    if c['profile']=='LAB_GPU_ONCE':
        require(c['parent']['checkpoint']['sha256']==kd.PARENT_SHA and c['parent']['checkpoint']['bytes']==323460353,'wrong parent checkpoint')
        require(rt.reference(c['parent']['checkpoint']['path'])==c['parent']['checkpoint'],'parent bytes conflict')
        binding=rt.exact(c['parent']['binding'])
        require(c['parent']['binding']['sha256']=='93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938','wrong original parent binding')
        require(binding['scope']=='formal_R18_seed0_e120' and binding['config']['seed']==0,'parent identity differs')
        from ..localization.material_paths import SELECTION_SHA
        selection=rt.exact(c['selection'])
        require(c['selection']['sha256']==SELECTION_SHA and [r['stable_image_id'] for r in selection['selected']]==c['fixed_ids'],'fixed64 selection differs')
        # RawRoleDataset authenticates the selection algorithm and source binding.
        d=c['data']['development']
        fixed=RawRoleDataset(role='development',manifest_path=d['manifest']['path'],images_path=d['roots']['images'],
            annotations_path=d['roots']['annotations'],root_binding=d['roots'],selection_path=c['selection']['path'],selection_sha256=SELECTION_SHA)
        require([r['stable_image_id'] for r in fixed.records]==c['fixed_ids'],'fixed64 exact members/order differs')
    baseline=rt.exact(c['baselines'])
    require(c['baselines']['sha256']=='8fb15f6d0580b0351ff36c6fb1e269ba4a3884f596f5d7426542dad0611d1f5c','baseline scalar identity differs')
    require(baseline['schema']=='abc-frozen-scalar-results-v1','baseline scalar schema differs')
    for arm in ('A','C'):
        for budget in (600,300):
            matches=[r for r in baseline['records'] if r['collection']=='development548' and r['seed']==0 and r['arm']==arm and r['budget']==budget]
            require(len(matches)==1 and matches[0]['checkpoint_sha256']==kd.PARENT_SHA,'baseline member/checkpoint differs')
    return dict(members=counts,train_supervision_only=True,baseline_recomputed=False,parent= c['parent']['checkpoint'])


def setup_runtime(c,gpu):
    if gpu:
        require(c['profile']=='LAB_GPU_ONCE' and c['GPU_authorized'] and not torch.cuda.is_initialized(),'unexpected CUDA state/authority')
        require(os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid'] and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','CUDA launch environment differs')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        require(actual==c['environment'],'GPU environment contract differs')
        torch.cuda.init();require(torch.cuda.device_count()==1,'GPU device count differs')
        torch.cuda.set_device(0)
        prop=torch.cuda.get_device_properties(0)
        require(str(prop.uuid).lower()==c['gpu_uuid'].lower(),'CUDA device UUID conflict')
        return torch.device('cuda:0')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized(),'CPU profile initialized CUDA')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return torch.device('cpu')


def native_model(c):
    p=rt.profile(c);cfg=Config(input_hw=p['input_hw'],physical_batch_size=p['physical'],amp_dtype='float32' if c['profile']=='CPU_FIXTURE_ONLY' else 'bfloat16')
    model,base,_,_,_=native_objects(cfg)
    native._configure_model_sampling(model,cfg)
    if c['profile']=='LAB_GPU_ONCE':kd.load_parent(model,c['parent']['checkpoint']['path'],c['parent']['binding']['path'])
    else:
        value=torch_ref(c['parent']['checkpoint'])
        require(value['profile']=='CPU_FIXTURE_ONLY','fixture parent identity differs')
        model.load_state_dict(value['model'],strict=True)
    return model,adapt(base),cfg


def engine(c,arm,stage,device):
    model,base,cfg=native_model(c)
    initial=kd.tensor_digest(model.state_dict().items())
    model.to(device);criterion=kd.Criterion(base,arm).to(device)
    optimizer,ema=kd.optimizer_ema(model,cfg)
    binding=dict(scope=stage,arm=arm,task_id=rt.TASK,attempt=c['attempt'],protocol=kd.VERSION,
        parent_sha256=c['parent']['checkpoint']['sha256'],source_manifest_sha256=c['source']['sha256'],
        contract_sha256=c['contract_sha256'],science_sha256=c['science_sha256'],profile=c['profile'],
        cache_index_sha256=None if arm=='T0' else rt.verify_stage(c,'teacher')['cache_index']['sha256'])
    e=Engine(model,criterion,optimizer,ema,cfg,arm=arm,binding=binding)
    require(e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'new engine did not start from local0')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    if device.type=='cuda':torch.cuda.manual_seed(0)
    return e,initial


def cache_binding(c):
    if c['profile']=='CPU_FIXTURE_ONLY':
        return kd.fixture_cache_binding(source_manifest_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],
            manifest_sha256=c['data']['train_core']['manifest']['sha256'],checkpoint_sha256=c['parent']['checkpoint']['sha256'])
    return kd.cache_binding(source_manifest_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'])


def teacher(c,out,emit,holder):
    rows,reader=data(c,'train_core');model,_,_=native_model(c)
    model=model.to(holder['device']).eval().requires_grad_(False)
    before=kd.tensor_digest(model.state_dict().items());records=[];calls=0;first=[]
    for start in range(0,len(rows),4):
        live(c,out,gpu=holder['device'].type=='cuda');batch=rows[start:start+4]
        originals=[original(reader,row) for row in batch]
        try:raw=kd.teacher_batch(model,originals,input_hw=rt.profile(c)['input_hw'],synthetic=c['profile']=='CPU_FIXTURE_ONLY')
        finally:
            for im in originals:im.close()
        calls+=5
        for row,views in zip(batch,raw):
            rec=kd.write_cache_member(out/row['stable_image_id'],row,views,cache_binding(c));records.append(rec)
            if len(records)<=8:
                v=rt.exact(rec['record']);target=parsed(reader,row)
                disk=torch_ref(v['raw_reference'])
                require(disk['binding']==cache_binding(c) and disk['member']==row,'first-batch raw disk identity differs')
                rebuilt=kd.from_raw_views(row,disk['raw_views'],cache_binding(c))
                require(all(v[k]==rebuilt[k] for k in ('binding','member','views')),'first-batch disk/compact reconstruction differs')
                im=original(reader,row)
                try:_,gt,trace=RawAugmentation(rt.profile(c)['input_hw'])(im,target,epoch=118,seed=0,position=len(records)-1,image_id=row['stable_image_id'])
                finally:im.close()
                aligned={a:len(kd.align_teacher(kd.select_teacher(v,a),(row['height'],row['width']),trace,gt,rt.profile(c)['input_hw'])['boxes']) for a in ('T1','T2')}
                first.append(dict(image_id=row['stable_image_id'],raw_exact=True,targets=aligned))
        # First up-to8 are verified within the original6090 calls, never rerun.
        require(before==kd.tensor_digest(model.state_dict().items()),'teacher state mutated')
        emit.emit('batch-'+str(start//4+1),actual_batch=len(batch),model_forward_calls=5,first_checks=len(first))
    path=out/'index.json';rt.publish(path,dict(binding=cache_binding(c),closed=True,records=records))
    require(len(first)==min(8,len(rows)),'teacher first-check scope differs')
    # Full strict raw/compact reconstruction before cache closure.
    kd.CacheReader(rt.reference(path),rows,cache_binding(c))
    emit.emit('cache_closed',index=rt.reference(path),raw_rebuilt_members=len(rows))
    return dict(cache_index=rt.reference(path),teacher_calls=calls,images=len(rows),first_checks=first,
        expected_calls=5*math.ceil(len(rows)/4),view0_shared=True,teacher_state_exact=True)


def training_dataset(c,arm):
    rows,reader=data(c,'train_core')
    cache=None if arm=='T0' else kd.CacheReader(rt.verify_stage(c,'teacher')['cache_index'],rows,cache_binding(c))
    d=c['data']['train_core']
    if c['profile']=='LAB_GPU_ONCE':
        return kd.Dataset(arm=arm,cache_reader=cache,cache_identity=None if arm=='T0' else cache_binding(c),
            manifest_path=d['manifest']['path'],images_path=d['roots']['images'],annotations_path=d['roots']['annotations'],root_binding=d['roots'])
    # Uses the exact production getitem/augmentation/cache/ignore implementation.
    # Only six-member constructor/loader is explicitly synthetic, not production.
    class FixtureDataset:
        role='train_core';seed=0;records=rows
        def __getitem__(self,key):
            epoch,position,index=key;row=rows[index]
            im=original(reader,row);raw=parsed(reader,row)
            try:image,target,trace=RawAugmentation(rt.profile(c)['input_hw'])(im,raw,epoch=epoch+9,seed=0,position=position,image_id=row['stable_image_id'])
            finally:im.close()
            trace.update(sampler_epoch=epoch,image_sha256=row['image_sha256'],annotation_sha256=row['annotation_sha256'])
            if cache is not None:
                target['teacher']=kd.align_teacher(kd.select_teacher(cache(row),arm),(row['height'],row['width']),trace,target,rt.profile(c)['input_hw'])
            return image,target,trace
    return FixtureDataset()


def windows(c,dataset,*,epoch,completed=0,limit=None):
    if c['profile']=='LAB_GPU_ONCE':
        yield from training_loader(dataset,epoch=epoch,completed_windows=completed,window_limit=limit)
    else:
        # One physical pair per synthetic epoch, or two distinct accept windows.
        for w in range(completed,completed+(limit if limit is not None else 1)):
            yield collate_window([dataset[(epoch,w*2+i,(w*2+i)%6)] for i in range(2)])


def input_identity(c,images,targets,traces,epoch,window):
    p=rt.profile(c);n=p['physical']*2
    require(len(images)==len(targets)==len(traces)==n,'logical student input cardinality differs')
    require(all(t['sampler_epoch']==epoch and t['epoch']==epoch+9 and t['position']==window*n+i and not t['augmentation_strong'] for i,t in enumerate(traces)),'student sampler/weak augmentation clock differs')
    supervised=[{k:v for k,v in t.items() if k!='teacher'} for t in targets]
    return dict(ids=[t['image_id'] for t in traces],pixels_sha256=kd.tensor_digest([('images',images)]),
                supervised_sha256=__import__('sparse_rtdetr.protocol_r18.engine',fromlist=['output_digest']).output_digest(supervised),
                trace_sha256=rt.digest(traces))


def acceptance(c,out,emit,holder,arm):
    e,initial=engine(c,arm,'accept-'+arm,holder['device']);holder['engine']=e
    emit.emit('parent_loaded',initial_model_sha256=initial,local_update=0,optimizer_empty=True,EMA_updates=0)
    dataset=training_dataset(c,arm);results=[];saved=None
    for win,(images,targets,traces) in enumerate(windows(c,dataset,epoch=109,limit=2)):
        live(c,out,gpu=holder['device'].type=='cuda');holder['input']=dict(images=images,targets=targets,traces=traces)
        inp=input_identity(c,images,targets,traces,109,win)
        result=e.train_window(images,targets,capture=True);result['input']=inp;results.append(result)
        emit.emit('update-'+str(win+1),update=e.completed_updates,input=inp,result=result)
        if win==0:
            saved=kd.save(e,out/'C1.pth');emit.emit('checkpoint-1',checkpoint=saved)
    require(len(results)==2,'acceptance two-window workload differs')
    expected=capture(e)
    holder.pop('engine');del e;gc.collect()
    if holder['device'].type=='cuda':torch.cuda.empty_cache()
    clone,again=engine(c,arm,'accept-'+arm,holder['device']);holder['engine']=clone
    require(initial==again,'fresh reconstruction parent differs')
    restored=kd.restore(clone,saved);emit.emit('restore-1',restored=restored)
    replay_count=0
    for images,targets,traces in windows(c,dataset,epoch=109,completed=1,limit=1):
        inp=input_identity(c,images,targets,traces,109,1);replay=clone.train_window(images,targets,capture=True);replay['input']=inp;replay_count+=1
    require(replay_count==1 and {k:v for k,v in results[-1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'acceptance next input/output/loss/DN/gradient differs')
    require(exact_equal(expected,capture(clone)),'acceptance full state/RNG replay differs')
    emit.emit('replay-2',update=2,full_state_RNG_exact=True)
    return dict(actual_updates=3,unique_updates=2,sequence=[1,2,2],initial_model_sha256=initial,
        checkpoint=saved,full_state_RNG_exact=True,state_usable_for_formal=False,forward_calls=6)


def formal(c,out,emit,holder,arm):
    accepted={a:rt.verify_stage(c,'accept-'+a) for a in ('T0','T1','T2')}
    require(all(r['actual_updates']==3 and r['state_usable_for_formal'] is False and r['full_state_RNG_exact'] for r in accepted.values()),'all arms must pass GPU restore before formal')
    e,initial=engine(c,arm,'train-'+arm,holder['device']);holder['engine']=e
    require(initial==accepted[arm]['initial_model_sha256'] and e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'formal did not independently reload parent EMA/local0')
    emit.emit('parent_loaded',initial_model_sha256=initial,local_update=0,optimizer_empty=True,EMA_updates=0,used_acceptance_state=False)
    dataset=training_dataset(c,arm);p=rt.profile(c);last=None;epochs=[];input_digests=[]
    for ep in range(1,p['epochs']+1):
        require(e.completed_updates==(ep-1)*p['windows'],'formal epoch cursor differs')
        emit.emit(f'epoch-{ep}-begin',local_epoch=ep,sampler_epoch=108+ep,augmentation_epoch=117+ep)
        started=time.monotonic();count=0
        for win,(images,targets,traces) in enumerate(windows(c,dataset,epoch=108+ep)):
            live(c,out,gpu=holder['device'].type=='cuda');holder['input']=dict(images=images,targets=targets,traces=traces)
            inp=input_identity(c,images,targets,traces,108+ep,win);input_digests.append(rt.digest(inp))
            r=e.train_window(images,targets,capture=False);count+=1
            require(e.completed_updates==(ep-1)*p['windows']+win+1,'formal local update clock differs')
            emit.emit('update-'+str(e.completed_updates),update=e.completed_updates,input=inp,result=r)
        require(count==p['windows'],'formal fixed windows incomplete')
        last=kd.save(e,out/f'checkpoint-FT{ep:03d}.pth');epochs.append(time.monotonic()-started)
        emit.emit(f'epoch-{ep}-saved',checkpoint=last,completed_updates=e.completed_updates)
    require(e.completed_updates==e.ema.updates==p['epochs']*p['windows'],'formal fixed endpoint incomplete')
    require(exact_equal(torch_ref(last),capture(e)),'saved formal endpoint state differs')
    rt.publish(out/'binding.json',e.binding)
    endpoint=dict(checkpoint=last,binding=rt.reference(out/'binding.json'),updates=e.completed_updates,EMA_updates=e.ema.updates,
        epoch=p['epochs'],EMA=True,initial_model_sha256=initial,used_acceptance_state=False,scope='CPU_FIXTURE_ONLY' if c['profile']=='CPU_FIXTURE_ONLY' else 'FT012_EMA')
    rt.publish(out/'endpoint.json',endpoint);emit.emit('endpoint_locked',endpoint=rt.reference(out/'endpoint.json'))
    return dict(endpoint=rt.reference(out/'endpoint.json'),actual_updates=e.completed_updates,initial_model_sha256=initial,
        used_acceptance_state=False,student_inputs_sha256=rt.digest(input_digests),epoch_wall_seconds=epochs,forward_calls=2*e.completed_updates)


def live(c,out,*,gpu):
    baseline=rt.read(out/'resource-entry.json')
    sample=rt.read(out/'resource-live.json')
    return rt.screen(sample,c,entry=False,gpu=gpu,baseline=baseline)


def endpoint_model(c,arm,device):
    model,_,_=native_model(c)
    if arm!='A':
        value=rt.exact(rt.verify_stage(c,'train-'+arm)['endpoint'])
        state=torch_ref(value['checkpoint'])
        require(state['binding']['arm']==arm and state['binding']['contract_sha256']==c['contract_sha256'], 'endpoint arm/contract differs')
        require(state['engine']['completed_updates']==rt.profile(c)['epochs']*rt.profile(c)['windows'], 'endpoint update differs')
        model.load_state_dict(state['ema']['module'],strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def image_batch(c,reader,rows,position):
    pixels=[]
    for i,row in enumerate(rows):
        image,raw=reader.item(row,position+i)
        require(tuple(image.shape)==(3,*rt.profile(c)['input_hw']), 'evaluation canvas differs')
        pixels.append(image)
    return torch.stack(pixels)


def infer(c,out,emit,holder,arm):
    from ..r0.evaluation import rank_candidates
    rows,reader=data(c,'development');model=endpoint_model(c,arm,holder['device']);pred={600:[],300:[]};refs=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            live(c,out,gpu=holder['device'].type=='cuda');batch=rows[start:start+4]
            images=image_batch(c,reader,batch,start).to(holder['device'])
            with forward_guards(model):value=model(images)
            logits=value['pred_logits'].detach().float().cpu();boxes=value['pred_boxes'].detach().float().cpu()
            refs.append(save_tensor(out/f'raw-{start//4:03d}.pth',dict(ids=[r['stable_image_id'] for r in batch],logits=logits,boxes=boxes)))
            for i,row in enumerate(batch):
                for budget in (600,300):pred[budget]+=rank_candidates(logits[i],boxes[i],row,budget)
            emit.emit('batch-'+str(start//4+1),actual_batch=len(batch),forward_calls=1)
    output={}
    for budget in (600,300):
        path=out/f'predictions-{budget}.json';rt.publish(path,pred[budget]);output[str(budget)]=rt.reference(path)
    return dict(predictions=output,raw_batches=refs,images=len(rows),forward_calls=math.ceil(len(rows)/4),EMA=True,
        inference='FP32 original A; no NMS; truncate then original raster ignore inside both evaluators')


def score(c,out,emit,holder):
    from ..r0.evaluation import primary_inputs,bridge_inputs,score_budget
    rows,reader=data(c,'development');truth=[parsed(reader,r) for r in rows];results={}
    for arm in ('T0','T1','T2'):
        results[arm]={};source=rt.verify_stage(c,'infer-'+arm)
        for budget in (600,300):
            live(c,out,gpu=False);pred=rt.exact(source['predictions'][str(budget)])
            require(len(pred)==len(rows)*budget and [p['image_id'] for p in pred[::budget]]==[r['stable_image_id'] for r in rows], 'prediction coverage/order differs')
            # Inspect the actual objects handed to BOTH certified interfaces.
            primary_inputs(rows,truth,pred);_,_,counts=bridge_inputs(rows,truth,pred)
            value=score_budget(rows,truth,pred,rt.ROOT.as_posix())
            require(counts==value['spatial_filter_counts'],'evaluation filter counts differ')
            path=out/f'{arm}-{budget}.json';rt.publish(path,value);results[arm][str(budget)]=rt.reference(path)
            emit.emit(f'score-{arm}-{budget}',result=rt.reference(path),images=len(rows),counts=counts,evaluator_calls=2)
    return dict(groups=6,evaluator_calls=12,results=results,scope='SYNTHETIC_CPU_SMOKE_NOT_SCIENCE' if c['profile']=='CPU_FIXTURE_ONLY' else 'development548')


def cost(c,out,emit,holder):
    from ..r0.evaluation import rank_candidates,bridge_inputs
    rows,reader=data(c,'development');by_id={r['stable_image_id']:r for r in rows};fixed=[by_id[x] for x in c['fixed_ids']]
    require(len(fixed)==rt.profile(c)['fixed_count'] and len(set(c['fixed_ids']))==len(fixed),'timing fixed members differ')
    raw=[parsed(reader,r) for r in fixed];device=holder['device'];measurements={a:[] for a in ('A','T0','T1','T2')};calls=0
    def measure(arm):
        nonlocal calls
        live(c,out,gpu=device.type=='cuda');gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        model=endpoint_model(c,arm,device)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();pred=[]
        with torch.inference_mode():
            for start in range(0,len(fixed),4):
                images=image_batch(c,reader,fixed[start:start+4],start).to(device)
                with forward_guards(model):value=model(images)
                logits=value['pred_logits'].detach().float().cpu();boxes=value['pred_boxes'].detach().float().cpu()
                for i,row in enumerate(fixed[start:start+4]):pred+=rank_candidates(logits[i],boxes[i],row,600)
                calls+=1
            _,_,counts=bridge_inputs(fixed,raw,pred)
        if device.type=='cuda':torch.cuda.synchronize()
        result=dict(seconds=time.perf_counter()-started,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else None,
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else None,filter_counts=counts,
            batch=4,images=len(fixed),dtype='FP32',includes='decode+resize+H2D+forward+rank+raw-ignore; excludes model load and GT parse')
        del model;gc.collect()
        return result
    for arm in measurements:
        for i in (1,2):emit.emit(f'warmup-{arm}-{i}',measurement=measure(arm))
    for i,order in enumerate(rt.execution()['timing_orders']):
        for arm in order:
            value=measure(arm);measurements[arm].append(value);emit.emit(f'timed-{i+1}-{arm}',measurement=value)
    require(calls==24*math.ceil(len(fixed)/4),'cost forward budget differs')
    summary={}
    for arm,values in measurements.items():
        seconds=[v['seconds'] for v in values]
        summary[arm]=dict(mean_seconds=statistics.mean(seconds),sample_SD_seconds=statistics.stdev(seconds),
            min_seconds=min(seconds),max_seconds=max(seconds),peak_allocated_bytes=max(v['peak_allocated_bytes'] or 0 for v in values),
            peak_reserved_bytes=max(v['peak_reserved_bytes'] or 0 for v in values))
    return dict(measurements=measurements,summary=summary,forward_calls=calls,
        scope='CPU_FIXTURE_NOT_GPU_COST' if c['profile']=='CPU_FIXTURE_ONLY' else 'FIXED64_GPU_E2E')


def compare(c,out,emit,holder):
    scored=rt.verify_stage(c,'score');timing=rt.verify_stage(c,'cost');base=rt.exact(c['baselines']);metrics={}
    for arm in ('T0','T1','T2'):
        metrics[arm]={b:rt.exact(ref)['COCO_bridge']['metrics'] for b,ref in scored['results'][arm].items()}
    for arm in ('A','C'):
        metrics[arm]={str(r['budget']):r['COCO_0_1'] for r in base['records'] if r['collection']=='development548' and r['seed']==0 and r['arm']==arm}
    inputs=[rt.verify_stage(c,'train-'+a)['student_inputs_sha256'] for a in ('T0','T1','T2')]
    require(len(set(inputs))==1,'three-arm student inputs/augmentation differ')
    fixture=c['profile']=='CPU_FIXTURE_ONLY'
    quality=None if fixture else kd.quality_gates(metrics,rt.recipe())
    deltas={}
    for pair in rt.recipe()['evaluation']['comparisons']:
        a,b=pair.split('-');deltas[pair]={budget:{k:(v-metrics[b][budget][k])*100 for k,v in values.items()} for budget,values in metrics[a].items()}
    t=timing['summary'];ratio=t['T2']['mean_seconds']/t['A']['mean_seconds']
    memory=None if fixture else t['T2']['peak_allocated_bytes']/t['A']['peak_allocated_bytes']
    cost_pass=None if fixture else ratio<=1.05 and memory<=1.05
    total_calls=rt.verify_stage(c,'teacher')['teacher_calls']+sum(rt.verify_stage(c,'accept-'+a)['forward_calls']+rt.verify_stage(c,'train-'+a)['forward_calls']+rt.verify_stage(c,'infer-'+a)['forward_calls'] for a in ('T0','T1','T2'))+timing['forward_calls']
    expected=28791 if not fixture else 10+18+12+6+48
    require(total_calls==expected,'total fixed forward budget differs')
    return dict(quality_passed=None if fixture else quality['passed'],cost_passed=cost_pass,quality=quality,
        metrics=metrics,deltas_pp=deltas,cost=dict(T2_over_A_E2E=ratio,T2_over_A_peak_allocated=memory,measurements=timing['summary']),
        total_forward_calls=total_calls,train_updates=sum(rt.verify_stage(c,'train-'+a)['actual_updates'] for a in ('T0','T1','T2')),
        inputs_equal=True,scope='CPU_FIXTURE_ONLY_NO_SCIENTIFIC_CLAIM' if fixture else 'SEED0_PRACTICAL_FEASIBILITY',
        decision='CPU_ENTRY_CHECK_ONLY' if fixture else ('QUALITY_AND_COST_PASS_CLOSE_AT_AUTHORIZED_SCOPE' if quality['passed'] and cost_pass else 'FIXED_RECIPE_CLOSED_NOT_PASSED'))


def export(c,out,emit,holder):
    # Explicit whitelist: never export tensors, predictions, per-update logs.
    files=[]
    sources=[('comparison.json',rt.stage_path(c,'compare')/'RESULT.json'),('science.json',rt.ROOT/rt.SCIENCE),('execution.json',rt.ROOT/rt.EXECUTION)]
    for stage in rt.STAGES[:-1]:sources.append((stage+'-CLOSED.json',rt.stage_path(c,stage)/'CLOSED.json'))
    sources+=[('source-manifest.json',Path(c['source']['path'])),('contract.json',Path(c['_path']))]
    for name,source in sources:
        target=out/name;require(not target.exists(),'export already exists');target.write_bytes(source.read_bytes())
        require(rt.sha(target)==rt.sha(source),'export SHA differs');files.append(rt.reference(target))
    rt.publish(out/'SHA_INDEX.json',dict(files=files,full_artifacts_retained_at=c['job_root'],slim_only=True))
    return dict(index=rt.reference(out/'SHA_INDEX.json'),files=len(files),bytes=sum(r['bytes'] for r in files),no_large_artifacts=True)


def worker(c,stage):
    from .distill_job import predecessors,gpu_stage
    from .failure import preserve
    from ..localization import guard
    out=rt.stage_path(c,stage);owner=rt.read(out/'supervisor-owner.json')
    require(owner['identity']==rt.identity(c,stage) and owner['pid']==os.getppid(),'worker/supervisor ownership differs')
    if os.name=='posix':require(guard.identity(os.getppid())['start']==owner['start'],'supervisor PID reused')
    predecessors(c,stage);emit=rt.Events(c,stage,out);holder={};started=time.monotonic()
    try:
        rt.publish(out/'worker-start.json',dict(identity=rt.identity(c,stage),pid=os.getpid(),parent_pid=os.getppid()))
        deadline=time.monotonic()+20
        while not (out/'supervisor-ack.json').exists():
            require(time.monotonic()<deadline,'supervisor handshake timeout');time.sleep(.02)
        ack=rt.read(out/'supervisor-ack.json')
        require(ack==dict(identity=rt.identity(c,stage),writer_pid=os.getpid(),supervisor_pid=owner['pid']),'supervisor acknowledgement differs')
        holder['device']=setup_runtime(c,gpu_stage(c,stage));live(c,out,gpu=gpu_stage(c,stage))
        emit.emit('begin',CUDA_initialized=torch.cuda.is_initialized(),profile=c['profile'])
        # Explicit fixture-only fault, never available to production contracts.
        if c['profile']=='CPU_FIXTURE_ONLY' and c.get('fixture_fault_stage')==stage:raise RuntimeError('deliberate CPU worker fault')
        if stage=='admission':value=admission(c,out,emit,holder)
        elif stage=='teacher':value=teacher(c,out,emit,holder)
        elif stage.startswith('accept-'):value=acceptance(c,out,emit,holder,stage.split('-')[1])
        elif stage.startswith('train-'):value=formal(c,out,emit,holder,stage.split('-')[1])
        elif stage.startswith('infer-'):value=infer(c,out,emit,holder,stage.split('-')[1])
        elif stage=='score':value=score(c,out,emit,holder)
        elif stage=='cost':value=cost(c,out,emit,holder)
        elif stage=='compare':value=compare(c,out,emit,holder)
        elif stage=='export':value=export(c,out,emit,holder)
        else:raise ValueError('unhandled stage')
        value.update(status='PASS',identity=rt.identity(c,stage),elapsed_seconds=time.monotonic()-started,
                     CUDA_initialized=torch.cuda.is_initialized(),fixture_only=c['profile']=='CPU_FIXTURE_ONLY')
        rt.publish(out/'RESULT.json',value);emit.emit('complete',result=rt.reference(out/'RESULT.json'))
        rt.publish(out/'worker-exit.json',dict(status='PASS',identity=rt.identity(c,stage),pid=os.getpid(),remaining_children=[],profile=c['profile']))
        return 0
    except BaseException as exc:
        try:forensic=preserve(holder,out)
        except BaseException as fail:forensic=dict(error=repr(fail),resumable=False)
        rt.publish(out/'STOP.json',dict(status='STOP_NO_RETRY',identity=rt.identity(c,stage),error=repr(exc),traceback=traceback.format_exc(),forensic=forensic,elapsed_seconds=time.monotonic()-started))
        rt.publish(out/'worker-exit.json',dict(status='STOP',identity=rt.identity(c,stage),error=repr(exc)))
        raise
