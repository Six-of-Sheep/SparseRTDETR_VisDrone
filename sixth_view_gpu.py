"""Fixed admitted GPU workloads; exactly one process per phase, no training."""
import gc
import io
import json
from pathlib import Path
import time
from sixth_view_runtime import require,read,reference,publish,ROOT,live,parent_context


def make_extra_pixels(data,rows,variant):
    from infer_seed import load_original,pixels
    from sixth_view_core import center_window
    from PIL import Image
    images=[]
    for row in rows:
        im=load_original(data,row)
        try:
            x=im.crop(center_window(im.height,im.width)) if variant=='C-center' else im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            try:images.append(pixels(data,x))
            finally:x.close()
        finally:im.close()
    return [images]


def restore(c,seed,out):
    import torch
    from sparse_rtdetr.localization import worker as base
    from sparse_rtdetr.protocol_r18.worker import make_engine
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.model import Config
    p=c['parents'][seed]
    for r in (p['binding'],p['endpoint'],p['checkpoint_sidecar']):require(reference(r['path'])==r,'endpoint/binding/sidecar changed')
    b=read(p['binding']['path']);side=read(p['checkpoint_sidecar']['path']);cp=p['checkpoint']
    require(side['sha256']==cp['sha256'] and side['completed_updates']==36480 and side['scope']==f'formal_R18_seed{seed}_e120','checkpoint endpoint scope')
    require(b['config']==Config(seed=seed).binding() and b['scope']==f'formal_R18_seed{seed}_e120','real seed model recipe differs')
    # Build only the unchanged native model, then strict original full restore.
    # Original binding is preserved; new execution lineage is a separate receipt.
    from sixth_view_admission import adapt_resources
    require(p['construction']['environment']==c['environment'],'parent/resource environment conflict')
    cfg=adapt_resources({**c,**p['construction'],'seed':seed,'runtime_root':c['runtime_root']})
    runtime=base.prepare_cuda(cfg,out)
    engine,constructed,_=make_engine(cfg,runtime,f'formal_R18_seed{seed}_e120')
    require(constructed['config']==b['config'] and constructed['model']==b['model'],'model construction differs from certified endpoint')
    result=checkpoint.restore(engine,b,cp['path'],cp['sha256'])
    require(result['full_state_exact'] and engine.completed_updates==engine.ema.updates==36480,'strict own e120 restore')
    engine.ema.module.eval()
    publish(out/'RESTORE.json',{'status':'PASS','seed':seed,'parent_checkpoint':cp,'parent_binding':p['binding'],
        'new_source_sha256':c['source_sha256'],'new_contract_sha256':c['contract_sha256'],'strict_restore':result,
        'original_binding_not_rewritten':True,'optimizer_updates':0})
    return engine,b


def run(c,seed,mode,out,emit):
    import torch
    from infer_seed import make_view_pixels,prediction,save_tensor
    from slice_core import tile_windows,map_crop_xyxy,select_view_candidates,merge_candidates
    from sixth_view_adapter import make_six_pixels,decode_extra
    from sixth_view_core import merge_six
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.localization.consumer_geometry import corners,positive_box
    from sparse_rtdetr.r0.evaluation_data import Development548
    # prepare_cuda emits its legacy event; route only that event to our bound step.
    from sparse_rtdetr.localization import worker as base
    old_event=base.event
    def cuda_event(path,kind,**kw):
        require(kind=='cuda_initialized','unexpected restore event');emit.emit('cuda_initialized',**kw)
    base.event=cuda_event
    try:e,b=restore(c,seed,out)
    finally:base.event=old_event
    emit.emit('restored',restore=reference(out/'RESTORE.json'))
    old,specs,rows,parsed=parent_context(c);spec=specs[seed]
    parent=c['parents'][seed];data=Development548(parent['construction']['data']['development'])
    require(data.records==rows,'GPU/CPU data member metadata differs')
    full=mode.startswith('full');variant={'accept-center':'C-center','accept-flip':'C-flip','full-center':'C-center','full-flip':'C-flip'}.get(mode)
    ids=[r['stable_image_id'] for r in rows] if full else parent['fixed64']
    work=[data.by_id[i] for i in ids];n=len(work)//4
    require(len(ids)==len(set(ids))==(548 if full else 64),'GPU workload membership')
    model=e.ema.module
    before=checkpoint.capture(e,b)
    guards=[]
    for owner,name in ((torch.autograd,'backward'),(e.optimizer,'step'),(e.ema,'update')):
        guards.append((owner,name,getattr(owner,name)))
        def forbidden(*a,**kw):raise RuntimeError('detector mutation forbidden')
        setattr(owner,name,forbidden)
    ownA=torch.load(io.BytesIO(old.exact(parent['A_raw'],contents=True)),map_location='cpu',weights_only=True)
    require(ownA['checkpoint_sha256']==parent['checkpoint']['sha256'] and ownA['image_ids']==[r['stable_image_id'] for r in rows],'own A raw identity')
    index={r['stable_image_id']:i for i,r in enumerate(rows)}
    # Reference raw from identical original fixed64 batches, not full548 batches.
    originalC=None
    if not full and variant:
        from sixth_view_job import verify_phase
        refs=verify_phase(c,seed,'accept-C')['raw_refs']
        require(len(refs)==16,'new original C benchmark raw incomplete')
        originalC=[torch.load(io.BytesIO(old.exact(r,contents=True)),map_location='cpu',weights_only=True) for r in refs]
    old_accept=read(parent['accept_C_result']['path'])
    require(reference(parent['accept_C_result']['path'])==parent['accept_C_result'],'old accept C metadata changed')
    calls=0;timings=[];raw_refs=[];preds={'600':[],'300':[]};pixel_checks=0

    def batch(br,number,save=False):
        nonlocal calls,pixel_checks
        torch.cuda.synchronize();start=time.perf_counter()
        if full:views=make_extra_pixels(data,br,variant)
        elif variant:views,_=make_six_pixels(data,br,variant)
        else:views,_=make_view_pixels(data,br,mode=='accept-C')
        prep=time.perf_counter();pool=[[] for _ in br];records=[]
        # Pixel checks lie outside the matched e2e measurement below; duplicate
        # preprocessing is explicitly recorded validation overhead, not inference.
        for local,tensors in enumerate(views):
            vid=5 if full else local
            x=torch.stack(tensors).to('cuda:0')
            with torch.autocast('cuda',enabled=False),forward_guards(model):y=model(x)
            torch.cuda.synchronize();calls+=1
            require(set(y)=={'pred_logits','pred_boxes'},'unknown detector output')
            positive_box('sixth/output',y['pred_boxes'])
            logits=y['pred_logits'].detach().cpu();boxes=y['pred_boxes'].detach().cpu()
            require(logits.shape==(4,300,10) and boxes.shape==(4,300,4) and logits.dtype==boxes.dtype==torch.float32 and torch.isfinite(logits).all(),'output schema/numerics')
            records.append({'view_id':vid,'pred_logits':logits,'pred_boxes':boxes})
            for j,row in enumerate(br):
                if vid==5:selected=decode_extra(logits[j],boxes[j],row,variant)
                else:
                    tile=tile_windows(row['height'],row['width'])[vid-1] if vid else None
                    w,h=(tile.width,tile.height) if tile else (row['width'],row['height'])
                    xy=torch.cat((boxes[j,:,:2]-boxes[j,:,2:]/2,boxes[j,:,:2]+boxes[j,:,2:]/2),-1)*torch.tensor([w,h]*2,dtype=torch.float32)
                    corners('sixth/base-output',xy)
                    mapped=map_crop_xyxy(xy.tolist(),tile) if tile else xy.tolist()
                    selected=select_view_candidates(mapped,logits[j].sigmoid().numpy(),view_id=vid)
                pool[j].extend(selected)
            del x,y
        finals=[]
        if not full:
            for j,row in enumerate(br):
                if mode=='accept-A':merged={'top600':pool[j],'top300':pool[j][:300]}
                elif variant:merged=merge_six(pool[j][:3000],pool[j][3000:],variant=variant)
                else:merged=merge_candidates(pool[j])
                finals.append({str(b):[prediction(row['stable_image_id'],p,whole=mode=='accept-A') for p in merged[f'top{b}']] for b in (600,300)})
        torch.cuda.synchronize();end=time.perf_counter()
        timing={'e2e_seconds':end-start,'preprocess_seconds':prep-start}
        if not full:
            for rec in records:
                if rec['view_id']==5 or originalC is None:continue
                expected=originalC[number]['views'][rec['view_id']]
                require(expected['view_id']==rec['view_id'],'original64 view order')
                for key in ('pred_logits','pred_boxes'):require(torch.equal(rec[key],expected[key]),'original fixed64 view output differs')
            a=records[0];ix=[index[r['stable_image_id']] for r in br]
            for key in ('pred_logits','pred_boxes'):require(torch.equal(a[key],ownA[key][ix]),'own original A output differs')
            if variant:
                expected,_=make_view_pixels(data,br,True)
                require(all(torch.equal(views[v][j],expected[v][j]) for v in range(5) for j in range(4)),'original pixels differ')
                pixel_checks+=4
        if save:
            payload={'schema':'r18-sixth-extra-raw-v1','seed':seed,'mode':mode,'variant':variant,'view_id':5,
                'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
                'checkpoint_sha256':parent['checkpoint']['sha256'],'image_ids':[r['stable_image_id'] for r in br],'views':records}
            raw_refs.append(save_tensor(out/f'raw-extra/batch-{number:03d}.pth',payload))
        return finals,timing
    try:
        with torch.inference_mode():
            for k in range(2):live(c,out,True);batch(work[k*4:k*4+4],k)
            emit.emit('warmup',forward_calls=calls)
            torch.cuda.reset_peak_memory_stats()
            for k in range(n):
                live(c,out,True);final,t=batch(work[k*4:k*4+4],k,save=full or mode=='accept-C');timings.append(t)
                for f in final:
                    for budget in ('600','300'):preds[budget].extend(f[budget])
                emit.emit(f'batch-{k+1:03d}',forward_calls=calls,images=(k+1)*4)
        require(checkpoint.exact_equal(before,checkpoint.capture(e,b)),'model/optimizer/EMA/RNG changed during inference')
        require(all(p.grad is None for model_ in (e.model,e.ema.module) for p in model_.parameters()),'nonempty detector gradient')
        emit.emit('state_exact',complete_state_RNG_exact=True)
        if mode=='accept-C':
            for budget in ('600','300'):
                require(preds[budget]==json.loads(old.exact(old_accept['predictions'][budget],contents=True)),'original C64 certified predictions differ')
        expected=(139 if full else (18 if mode=='accept-A' else 90 if mode=='accept-C' else 108))
        require(calls==expected,'fixed forward budget differs')
        result={'status':'PASS','seed':seed,'mode':mode,'variant':variant,'image_ids':ids,'forward_calls':calls,
            'state_RNG_exact':True,'detector_updates':0,'pixels_exact_images':pixel_checks,'raw_refs':raw_refs,
            'e2e_seconds':sum(t['e2e_seconds'] for t in timings),'timings':timings,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'validation_overhead_outside_timing':True,'no_AP_on_fixed64':True}
        publish(out/'RESULT.json',result);emit.emit('complete',result=reference(out/'RESULT.json'))
    finally:
        for owner,name,value in guards:setattr(owner,name,value)
        del e;gc.collect();torch.cuda.empty_cache()
