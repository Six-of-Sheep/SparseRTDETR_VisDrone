"""Sampling-span stages; explicit synthetic CPU profile is never lab acceptance."""
from __future__ import annotations
import contextlib, copy, gc, io, json, math, os, random, statistics, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import span_runtime as rt, span_checkpoint as checkpoint, span_budget as limits
from .span_model import build, validate_sampling, ARMS, VERSION
from .span_observation import observe
from .model import Config, CommonEMA, PRETRAINED_SHA
from .engine import Engine, tensor_digest, output_digest, assert_healthy
from .checkpoint import exact_equal
from .geometry import forward_guards
from ..baseline import training_v2b as native
from ..baseline.training_v2b_checkpoint import _safe
from ..localization.data import RawRoleDataset, RawAugmentation, parse_raw, read_bound, training_loader, collate_window
from ..localization.runio import require

def torch_ref(ref):
    p=Path(ref['path'])
    expected=dict(path=ref['path'],sha256=ref['sha256'],bytes=ref['size_bytes']) if ref.get('schema')==checkpoint.FORMAT else ref
    require(rt.reference(p)==expected,'tensor file SHA/size differs')
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
            reader=RawRoleDataset(role=role,manifest_path=d['manifest']['path'],images_path=d['roots']['images'],
                annotations_path=d['roots']['annotations'],root_binding=d['roots'],seed=0)
            require(reader.records==rt.exact(d['manifest'])['records'],'training constructor/manifest mapping differs')
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


def setup_runtime(c,gpu,out=None):
    if gpu:
        require(c['profile']=='LAB_GPU_ONCE' and c['GPU_authorized'] and not torch.cuda.is_initialized(),'unexpected CUDA state/authority')
        require(os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid'] and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','CUDA launch environment differs')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        require(actual==c['environment'],'GPU environment contract differs')
        from .cuda_identity import inspect_cuda_identity
        require(out is not None,'GPU identity audit output required')
        identity=inspect_cuda_identity(c['gpu_uuid'],Path(out)/'cuda-identity',torch_module=torch)
        require(identity['BF16_supported'],'frozen BF16 unsupported')
        return torch.device('cuda:0')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized(),'CPU profile initialized CUDA')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return torch.device('cpu')


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
            limits.guard(c,forward_calls=1)
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
    for arm in ARMS:
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
    raw=[parsed(reader,r) for r in fixed];device=holder['device'];measurements={a:[] for a in ('A','B0','B1','P')};calls=0
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
                limits.guard(c,forward_calls=1)
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


def native_model(c, arm):
    p=rt.profile(c)
    cfg=Config(input_hw=p['input_hw'],physical_batch_size=p['physical'],
               amp_dtype='float32' if c['profile']=='CPU_FIXTURE_ONLY' else 'bfloat16')
    model,criterion,info=build(cfg,arm,pretrained=None if c['profile']=='CPU_FIXTURE_ONLY' else c['pretrained']['path'],
                               synthetic=c['profile']=='CPU_FIXTURE_ONLY')
    if c['profile']=='CPU_FIXTURE_ONLY':
        require(torch_ref(c['pretrained'])['profile']=='CPU_FIXTURE_ONLY','fixture initial source differs')
    return model,criterion,cfg,info


def engine(c,arm,stage,device):
    model,criterion,cfg,info=native_model(c,arm)
    initial=tensor_digest(model.state_dict().items())
    # Exclude only the two documented native B1 changes, nothing else.
    shared=tensor_digest((n,v) for n,v in model.state_dict().items()
        if not any(n==f'decoder.decoder.layers.{i}.cross_attn.{key}' for i in range(3)
                   for key in ('sampling_offsets.bias','num_points_scale')))
    expected=rt.read(rt.stage_path(c,'admission')/'initialization.json')
    require(expected[arm]['full']==initial and expected[arm]['shared']==shared,
            'independent arm initialization differs from admitted CPU construction')
    model.to(device);criterion.to(device)
    e=Engine(model,criterion,native._optimizer(model,cfg),CommonEMA(model),cfg);e.arm=arm
    e.binding=dict(arm=VERSION+'/'+arm,scope='span_cpu_fixture' if device.type=='cpu' else
        ('span_GPU_acceptance' if stage.startswith('accept-') else 'span_formal_e120'),
        config=cfg.binding(),model=info,runtime=dict(gpu_uuid=c['gpu_uuid']),
        source_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],
        data_sha256=rt.digest(c['data']),pretrained_sha256=c['pretrained']['sha256'])
    require(e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'arm not independently fresh')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    if device.type=='cuda':torch.cuda.manual_seed(0)
    return e,initial


def save(e,out,name):
    return checkpoint.save(e,e.binding,out/name)


def admission(c,out,emit,holder):
    counts={};ids={}
    for role,event in (('train_core','training_members'),('development','development_members')):
        rows,reader=data(c,role);audits=[]
        for i,row in enumerate(rows):
            image=original(reader,row);image.close();parsed_gt=parsed(reader,row)
            # Training keeps the original parse_raw role/filter semantics;
            # Development548 independently enforces its certified GT geometry.
            audits.append(dict(id=row['stable_image_id'],image_sha256=row['image_sha256'],
                annotation_sha256=row['annotation_sha256'],physical_rows=len(parsed_gt['rows'])))
            if i%32==0:
                if holder.get('preparation'):
                    sample=rt.snapshot(c,entry=False,gpu=False,baseline=rt.read(out/'resource-entry.json'))
                    rt.atomic(out/'resource-live.json',sample)
                live(c,out,gpu=False)
        ids[role]=[r['stable_image_id'] for r in rows];counts[role]=len(rows)
        rt.publish(out/(role+'-files.json'),audits)
        emit.emit(event,images=len(rows),members_sha256=rt.digest(ids[role]))
    require(not set(ids['train_core'])&set(ids['development']),'training/development identity overlap')
    require(rt.reference(c['pretrained']['path'])==c['pretrained'],'initialization source SHA/size conflict')
    if c['profile']=='LAB_GPU_ONCE':
        require(c['pretrained']['sha256']==PRETRAINED_SHA,'wrong ImageNet source')
        from ..localization.material_paths import SELECTION_SHA
        selection=rt.exact(c['selection'])
        require(c['selection']['sha256']==SELECTION_SHA,'fixed64 identity changed')
        require([r['stable_image_id'] for r in selection['selected']]==c['fixed_ids'],'fixed64 order changed')
        d=c['data']['development']
        fixed=RawRoleDataset(role='development',manifest_path=d['manifest']['path'],
            images_path=d['roots']['images'],annotations_path=d['roots']['annotations'],
            root_binding=d['roots'],selection_path=c['selection']['path'],selection_sha256=SELECTION_SHA)
        require([r['stable_image_id'] for r in fixed.records]==c['fixed_ids'],'certified64 algorithm binding differs')
        require(c['historical_A']['checkpoint']['sha256']=='13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757',
                'wrong original A for cost')
        require(rt.reference(c['historical_A']['checkpoint']['path'])==c['historical_A']['checkpoint'],'original A checkpoint conflict')
        binding=rt.exact(c['historical_A']['binding'])
        require(binding['scope']=='formal_R18_seed0_e120' and binding['config']['seed']==0,'original A binding differs')
        historical=torch_ref(c['historical_A']['checkpoint'])
        require(historical['binding']==binding and historical['binding_sha256']==rt.digest(binding)
                and historical['engine']['completed_updates']==36480 and historical['ema']['updates']==36480,
                'historical A complete endpoint/EMA binding differs')
    scalars=rt.exact(c['baselines'])
    require(c['baselines']['sha256']=='8fb15f6d0580b0351ff36c6fb1e269ba4a3884f596f5d7426542dad0611d1f5c'
            and scalars['schema']=='abc-frozen-scalar-results-v1','frozen scalar identity conflict')
    initial={};reference_state=None
    for arm in ARMS:
        model,criterion,cfg,info=native_model(c,arm);state=model.state_dict()
        full=tensor_digest(state.items())
        shared=tensor_digest((n,v) for n,v in state.items() if not any(
            n==f'decoder.decoder.layers.{i}.cross_attn.{k}' for i in range(3)
            for k in ('sampling_offsets.bias','num_points_scale')))
        if reference_state is None:reference_state={n:v.clone() for n,v in state.items()}
        differences=[n for n,v in state.items() if not torch.equal(v,reference_state[n])]
        expected={f'decoder.decoder.layers.{i}.cross_attn.{k}' for i in range(3)
                  for k in ('sampling_offsets.bias','num_points_scale')} if arm=='B1' else set()
        require(set(differences)==expected,'undocumented initial tensor/buffer difference')
        initial[arm]=dict(full=full,shared=shared,differences=differences,sampling=validate_sampling(model,arm))
        del model,criterion;gc.collect()
    require(len({r['shared'] for r in initial.values()})==1,'common initialization mismatch')
    rt.publish(out/'initialization.json',initial)
    return dict(members=counts,initialization=rt.reference(out/'initialization.json'),
                prepare_forward_calls=0,initialization_kind='fresh ImageNet and native seed0; no e120 parent')


def training_dataset(c):
    if c['profile']=='LAB_GPU_ONCE':
        from .data import BoundDataset
        d=c['data']['train_core']
        return BoundDataset(role='train_core',manifest_path=d['manifest']['path'],
            images_path=d['roots']['images'],annotations_path=d['roots']['annotations'],seed=0,root_binding=d['roots'])
    rows,reader=data(c,'train_core')
    class FixtureTraining:
        def __getitem__(self,key):
            epoch,position,index=key;row=rows[index];image=original(reader,row)
            try:
                pixels,target,trace=RawAugmentation(rt.profile(c)['input_hw'])(image,parsed(reader,row),
                    epoch=epoch,seed=0,position=position,image_id=row['stable_image_id'],training=True)
                return pixels,target,trace
            finally:image.close()
    return FixtureTraining()


def windows(c,dataset,*,epoch,completed=0,limit=None):
    if c['profile']=='LAB_GPU_ONCE':
        yield from training_loader(dataset,epoch=epoch,completed_windows=completed,window_limit=limit)
    else:
        for w in range(completed,completed+(limit if limit is not None else 1)):
            yield collate_window([dataset[(epoch,w*2+i,(w*2+i)%6)] for i in range(2)])


def input_identity(c,images,targets,traces,epoch,window):
    if c['profile']=='LAB_GPU_ONCE':
        from .data import input_record
        record=input_record(images,targets,traces);record.pop('spawn_workers')
    else:
        record=dict(ids=[t['image_id'] for t in traces],pixels_sha256=tensor_digest([('pixels',images)]),
                    target_sha256=output_digest(targets),trace_sha256=rt.digest(traces))
    require(all(t['epoch']==epoch and t['position']==window*rt.profile(c)['physical']*2+i
                and t['augmentation_strong']==(epoch<118) for i,t in enumerate(traces)),
            'original augmentation/position clock differs')
    return record


def live(c,out,*,gpu):
    baseline=rt.read(out/'resource-entry.json')
    journal=out/'resources.jsonl'
    if journal.exists():
        # Read the last complete immutable journal record. A DrvFS atomic
        # rename can momentarily hide the live pointer; no retry/substitution.
        with journal.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();offset=max(0,size-65536)
            stream.seek(offset);raw=stream.read()
        if offset:raw=raw[raw.index(b'\n')+1:]
        end=raw.rfind(b'\n');lines=raw[:end].splitlines() if end>=0 else []
        sample=json.loads(lines[-1]) if lines else baseline
    else:
        sample=rt.read(out/'resource-live.json')
    return rt.screen(sample,c,entry=False,gpu=gpu,baseline=baseline)


def update(c,e,images,targets,out,holder,*,capture=False,observed=False):
    live(c,out,gpu=e.device.type=='cuda')
    from . import span_acceptance_budget as acceptance_budget
    scoped=rt.acceptance_only(c)
    if scoped:acceptance_budget.reserve_update(c,e.arm)
    limits.guard(c,forward_calls=2,save=lambda:save(e,out,'budget-stop.pth'))
    holder['input']=dict(images=images,targets=targets)
    rows=[]
    with acceptance_budget.forward_counter(c,e.model,e.arm) if scoped else contextlib.nullcontext(), \
         observe(e.model,e.arm,rows) if observed else contextlib.nullcontext():
        result=e.train_window(images,targets,capture=capture)
    require(e.completed_updates==e.ema.updates==result['update'] and result['finite'] and e.phase==0,
            'complete training update/EMA clock differs')
    if e.device.type=='cuda':
        require(torch.cuda.max_memory_allocated()<=c['resources']['training_peak_allocated_GiB_cap']*rt.GIB,
                'training peak memory exceeds frozen limit')
    if scoped:acceptance_budget.complete_update(c,e.arm)
    return result,rows


def acceptance(c,out,emit,holder,arm):
    e,initial=engine(c,arm,'accept-'+arm,holder['device']);holder['engine']=e;holder['binding']=e.binding
    emit.emit('initial_loaded',initial_model_sha256=initial,update=0,optimizer_empty=True,EMA_updates=0)
    dataset=training_dataset(c);results=[];saved=None
    for win,(images,targets,traces) in enumerate(windows(c,dataset,epoch=1,limit=2)):
        inp=input_identity(c,images,targets,traces,1,win)
        r,obs=update(c,e,images,targets,out,holder,capture=True,observed=True)
        r['input']=inp;r['observations']=obs;results.append(r)
        emit.emit('update-'+str(win+1),update=e.completed_updates,result=r)
        if win==0:
            saved=save(e,out,'C1.pth');emit.emit('checkpoint-1',checkpoint=saved)
    require(len(results)==2,'acceptance workload differs')
    expected=checkpoint.capture(e,e.binding)
    holder.pop('engine');del e;gc.collect()
    if holder['device'].type=='cuda':torch.cuda.empty_cache()
    clone,again=engine(c,arm,'accept-'+arm,holder['device']);holder['engine']=clone;holder['binding']=clone.binding
    require(again==initial,'acceptance fresh reconstruction differs')
    restored=checkpoint.restore(clone,clone.binding,saved['path'],saved['sha256']);emit.emit('restore-1',restored=restored)
    n=0
    for images,targets,traces in windows(c,dataset,epoch=1,completed=1,limit=1):
        replay,obs=update(c,clone,images,targets,out,holder,capture=True,observed=True)
        replay['input']=input_identity(c,images,targets,traces,1,1);replay['observations']=obs;n+=1
    require(n==1 and {k:v for k,v in results[-1].items() if k!='elapsed_seconds'}==
        {k:v for k,v in replay.items() if k!='elapsed_seconds'},'same next input/output/DN/gradient/observation replay differs')
    require(exact_equal(expected,checkpoint.capture(clone,clone.binding)),'full state/RNG replay differs')
    emit.emit('replay-2',full_state_RNG_exact=True)
    return dict(actual_updates=3,unique_updates=2,sequence=[1,2,2],initial_model_sha256=initial,
        checkpoint=saved,full_state_RNG_exact=True,state_usable_for_formal=False,forward_calls=6,
        gradient_finite=True,complete_update_seconds=[r['elapsed_seconds'] for r in results]+[replay['elapsed_seconds']],
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if holder['device'].type=='cuda' else None,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if holder['device'].type=='cuda' else None,
        short_speed_not_full_training_projection=True)


def formal(c,out,emit,holder,arm):
    accepted={a:rt.verify_stage(c,'accept-'+a) for a in ARMS}
    require(all(r['actual_updates']==3 and r['full_state_RNG_exact'] and not r['state_usable_for_formal']
                for r in accepted.values()),'all arms must pass GPU acceptance before training')
    e,initial=engine(c,arm,'train-'+arm,holder['device']);holder['engine']=e;holder['binding']=e.binding
    require(initial==accepted[arm]['initial_model_sha256'] and e.completed_updates==e.ema.updates==0
            and not e.optimizer.state,'formal arm did not start independently at update0')
    emit.emit('initial_loaded',initial_model_sha256=initial,used_acceptance_state=False,update=0)
    save(e,out,'checkpoint-e000.pth');p=rt.profile(c);dataset=training_dataset(c);inputs=[];times=[];last=None
    for epoch in range(1,p['epochs']+1):
        require(e.completed_updates==(epoch-1)*p['windows'],'epoch start cursor differs')
        emit.emit(f'epoch-{epoch}-begin',epoch=epoch,augmentation='strong' if epoch<118 else 'weak')
        start=time.monotonic();count=0
        for win,(images,targets,traces) in enumerate(windows(c,dataset,epoch=epoch)):
            inp=input_identity(c,images,targets,traces,epoch,win);inputs.append(rt.digest(inp))
            observed=win==0 and epoch in rt.execution()['observations']['train_epochs']
            r,observations=update(c,e,images,targets,out,holder,observed=observed);count+=1
            require(e.completed_updates==(epoch-1)*p['windows']+win+1,'update cursor differs')
            emit.emit('update-'+str(e.completed_updates),input=inp,result=r)
            if observed:rt.publish(out/f'sampling-e{epoch:03d}-w000.json',observations)
        require(count==p['windows'],'epoch fixed update count incomplete')
        times.append(time.monotonic()-start)
        if epoch in rt.execution()['checkpoint_epochs'] or epoch==p['epochs']:
            last=save(e,out,f'checkpoint-e{epoch:03d}.pth')
        emit.emit(f'epoch-{epoch}-saved',checkpoint=last,checkpoint_written_this_epoch=epoch in rt.execution()['checkpoint_epochs'] or epoch==p['epochs'],
                  completed_updates=e.completed_updates)
    require(e.completed_updates==e.ema.updates==p['epochs']*p['windows'],'fixed endpoint incomplete')
    require(exact_equal(torch_ref(last),checkpoint.capture(e,e.binding)),'saved endpoint full state differs')
    rt.publish(out/'binding.json',e.binding)
    rt.publish(out/'endpoint.json',dict(checkpoint=last,binding=rt.reference(out/'binding.json'),
        updates=e.completed_updates,epoch=p['epochs'],EMA=True,initial_model_sha256=initial,used_acceptance_state=False))
    emit.emit('endpoint_locked',endpoint=rt.reference(out/'endpoint.json'))
    return dict(endpoint=rt.reference(out/'endpoint.json'),actual_updates=e.completed_updates,
        initial_model_sha256=initial,used_acceptance_state=False,student_inputs_sha256=rt.digest(inputs),
        epoch_wall_seconds=times,forward_calls=e.completed_updates*2)


def endpoint_model(c,arm,device):
    model,_,_,_=native_model(c,'B0' if arm=='A' else arm)
    if arm=='A':
        if c['profile']=='LAB_GPU_ONCE':state=torch_ref(c['historical_A']['checkpoint'])
        else:return model.to(device).eval().requires_grad_(False)
    else:
        ep=rt.exact(rt.verify_stage(c,'train-'+arm)['endpoint']);state=torch_ref(ep['checkpoint'])
        require(state['binding']==rt.exact(ep['binding']) and state['binding']['arm']==VERSION+'/'+arm
                and state['binding']['contract_sha256']==rt.training_contract(c,arm)['contract_sha256'], 'endpoint arm/contract binding differs')
        require(ep['EMA'] and ep['epoch']==rt.profile(c)['epochs'] and state['engine']['completed_updates']==
                rt.profile(c)['epochs']*rt.profile(c)['windows'], 'endpoint clock differs')
    model.load_state_dict(state['ema']['module'],strict=True)
    validate_sampling(model,'B0' if arm=='A' else arm)
    return model.to(device).eval().requires_grad_(False)


def compare(c,out,emit,holder):
    scored=rt.verify_stage(c,'score');timing=rt.verify_stage(c,'cost');metrics={}
    for arm in ARMS:metrics[arm]={b:rt.exact(ref) for b,ref in scored['results'][arm].items()}
    base=rt.exact(c['baselines'])
    for arm in ('A','C'):
        metrics[arm]={str(r['budget']):r for r in base['records'] if
            r['collection']=='development548' and r['seed']==0 and r['arm']==arm}
    inputs=[rt.verify_stage(c,'train-'+a)['student_inputs_sha256'] for a in ARMS]
    require(len(set(inputs))==1,'three-arm training pixels/GT/augmentation differ')
    fixture=c['profile']=='CPU_FIXTURE_ONLY';deltas={};gates=[]
    for baseline in ('B0','B1','A','C'):
        pair='P-'+baseline;deltas[pair]={}
        for b in ('600','300'):
            a=metrics['P'][b]['COCO_bridge']['metrics']
            other=metrics[baseline][b]['COCO_bridge']['metrics'] if baseline in ARMS else metrics[baseline][b]['COCO_0_1']
            if not fixture:require(all(v>=0 and other[k]>=0 for k,v in a.items()),'undefined scientific metric')
            deltas[pair][b]={k:(v-other[k])*100 for k,v in a.items()}
            if baseline in ('B0','B1'):
                for key,limit in rt.recipe()['proposed_gates_pp_for_each_of_P_minus_B0_and_P_minus_B1'].items():
                    prefix='top'+b+'_COCO_'
                    if key.startswith(prefix):
                        name=key[len(prefix):-4];v=deltas[pair][b][name]
                        gates.append(dict(pair=pair,budget=int(b),metric=name,delta_pp=v,minimum=limit,passed=v>=limit))
                def primary_ap(value):
                    return value['official']['AP']
                difference=primary_ap(metrics['P'][b])-primary_ap(metrics[baseline][b])
                limit=rt.recipe()['proposed_gates_pp_for_each_of_P_minus_B0_and_P_minus_B1']['top'+b+'_primary_AP_min']
                gates.append(dict(pair=pair,budget=int(b),metric='primary_AP',delta_pp=difference,minimum=limit,passed=difference>=limit))
    t=timing['summary'];ratio=t['P']['mean_seconds']/t['B0']['mean_seconds']
    memory=None if fixture else t['P']['peak_allocated_bytes']/t['B0']['peak_allocated_bytes']
    calls=sum(rt.verify_stage(c,s)['forward_calls'] for s in rt.STAGES if s.startswith(('accept-','train-','infer-')))+timing['forward_calls']
    expected=219693 if not fixture else 18+12+6+48
    require(calls==expected,'total fixed forward count differs')
    return dict(quality_passed=None if fixture else all(r['passed'] for r in gates),
        cost_passed=None if fixture else ratio<=1.05 and memory<=1.05,quality_gates=gates,deltas_pp=deltas,
        P_over_B0_E2E=ratio,P_over_B0_peak_allocated=memory,cost=timing['summary'],total_forward_calls=calls,
        train_updates=sum(rt.verify_stage(c,'train-'+a)['actual_updates'] for a in ARMS),inputs_equal=True,
        scope='CPU_FIXTURE_ONLY_NOT_SCIENCE' if fixture else 'seed0_development548_exploration',
        single_seed_not_independent_generalization=True,historical_A_C_context_not_causal_controls=True)


def export(c,out,emit,holder):
    # Explicit whitelist: never export tensors, predictions, per-update logs.
    files=[]
    sources=[('comparison.json',rt.stage_path(c,'compare')/'RESULT.json'),('science.json',rt.ROOT/rt.SCIENCE),('execution.json',rt.ROOT/rt.EXECUTION)]
    for stage in rt.STAGES[:-1]:sources.append(rt.export_stage_evidence(c,stage))
    sources+=[('source-manifest.json',Path(c['source']['path'])),('contract.json',Path(c['_path']))]
    for name,source in sources:
        target=out/name;require(not target.exists(),'export already exists');target.write_bytes(source.read_bytes())
        require(rt.sha(target)==rt.sha(source),'export SHA differs');files.append(rt.reference(target))
    rt.publish(out/'SHA_INDEX.json',dict(files=files,full_artifacts_retained_at=c['job_root'],slim_only=True))
    return dict(index=rt.reference(out/'SHA_INDEX.json'),files=len(files),bytes=sum(r['bytes'] for r in files),no_large_artifacts=True)



def worker(c,stage):
    from .span_job import predecessors,gpu_stage
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
        holder['device']=setup_runtime(c,gpu_stage(c,stage),out);live(c,out,gpu=gpu_stage(c,stage))
        emit.emit('begin',CUDA_initialized=torch.cuda.is_initialized(),profile=c['profile'])
        # Explicit fixture-only fault, never available to production contracts.
        if c['profile']=='CPU_FIXTURE_ONLY' and c.get('fixture_fault_stage')==stage:raise RuntimeError('deliberate CPU worker fault')
        if stage=='admission':value=admission(c,out,emit,holder)
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
