"""Single authorized endpoint diagnostic. Never trains, scores AP or retries."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def reference(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))


def exact(r):
    require(reference(r['path']) == r, 'reference differs: '+r['path'])
    return json.loads(Path(r['path']).read_text())


def publish(path, value):
    path=Path(path)
    raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    reserve=0 if path.name in ('STOP.json','WORKER_STOP.json','EXIT.json') else 65536
    require(used+len(raw)<=20*1024**2-reserve, 'aggregate output cap')
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())


def load(path):
    c=json.loads(Path(path).read_text())
    require(c['schema']=='r18-endpoint-support-diagnostic-v1', 'contract schema')
    require(c['limits']==dict(images=32,batch=4,forward_calls=16,GPU_seconds=600,
                             allocated_GiB=18,slim_bytes=20*1024**2), 'limits differ')
    require(c['arms']==['P','B0'] and c['policy']=='stable_id_ascending_first32_final_match_v1', 'policy differs')
    require(c['forbidden']==['train','backward','AP','tuning','1602','retry','expand_images','DN'], 'forbidden scope')
    require(len(c['selected_ids'])==32 and c['selected_ids']==sorted(set(c['selected_ids'])), 'selection')
    old=exact(c['parent_contract'])
    require(old['source']==c['frozen_source'] and old['commit']==c['frozen_commit'], 'frozen parent')
    for row in exact(c['frozen_source'])['files']:
        p=Path(c['frozen_release'])/row['path']
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'], 'source changed: '+str(p))
    for row in c['diagnostic_files']:
        exact(row) if row['path'].endswith('.json') else require(reference(row['path'])==row, 'diagnostic source changed')
    require(exact(c['CPU_gate'])['status']=='PASS', 'CPU gate not passed')
    for row in c['historical_receipts']:
        exact(row)
    manifest=exact(old['data']['development']['manifest'])
    selected=sorted(manifest['records'],key=lambda r:r['stable_image_id'])[:32]
    require([r['stable_image_id'] for r in selected]==c['selected_ids'], 'stable selection conflict')
    require(len(manifest['records'])==548 and all(r['split']=='val' for r in selected), 'development role')
    # Only the 32 selected image/annotation payloads are read and SHA checked.
    for row in selected:
        for name,rel,sha,size in [('images','relative_path','image_sha256','image_size_bytes'),
                                  ('annotations','annotation_relative_path','annotation_sha256','annotation_size_bytes')]:
            p=Path(old['data']['development']['roots'][name])/Path(row[rel]).name
            require(p.is_file() and not p.is_symlink() and p.stat().st_size==row[size] and digest(p)==row[sha], 'selected payload changed')
    for arm in c['arms']:
        exact(c['endpoints'][arm])
        cp=c['checkpoints'][arm]
        require(reference(cp['path'])==cp, 'e120 checkpoint SHA changed')
    return c,old,selected


COLUMNS=['arm','image_id','gt_id','query','layer','raw_area','IoU',
         'reference_width_cells','reference_height_cells','triggered','floor_applied',
         'A','actual_Z','native_Z','actual_I','native_I','delta_I',
         'actual_O','native_O','delta_O','actual_V','native_V','delta_V',
         'actual_M','native_M','delta_M','stencil_changed','effective_interpolation_changed',
         'local_core_output_abs_mean']


def worker(c, old, selected, out):
    # Production modules are always imported from the SHA-checked frozen release.
    sys.path.insert(0,str(Path(c['frozen_release'])/'src'))
    import gc
    import random
    import numpy as np
    import torch
    from r18_support_math import capture, support, stencil, delta, effective_change
    from sparse_rtdetr.protocol_r18.span_model import build,validate_sampling
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation_data import Development548
    from sparse_rtdetr.protocol_r18.span_worker import original
    from sparse_rtdetr.localization.data import RawAugmentation
    require(os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid'] and not torch.cuda.is_initialized(), 'CUDA environment')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,
                cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    require(actual==old['environment'], 'runtime versions changed')
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    reader=Development548(dict(manifest=old['data']['development']['manifest']['path'],roots=old['data']['development']['roots']))
    transform=RawAugmentation((768,1344))
    count=0;output=[];inventory=[];images=[];state_checks=[];peaks=[]
    for arm in c['arms']:
        model,criterion,info=build(Config(),arm,pretrained=old['pretrained']['path'])
        recipe=info['criterion']['matcher']
        require(recipe==dict(cost_class=2,cost_bbox=5,cost_giou=2,alpha=.25,gamma=2.,use_focal_loss=True), 'matcher recipe changed')
        ep=exact(c['endpoints'][arm]);state=torch.load(c['checkpoints'][arm]['path'],map_location='cpu',weights_only=True)
        require(state['binding']==exact(ep['binding']) and state['binding']['arm'].endswith('/'+arm), 'state binding')
        require(ep['epoch']==120 and ep['updates']==36480 and ep['EMA'] and not ep['used_acceptance_state'], 'endpoint identity')
        require(state['engine']['completed_updates']==36480 and state['ema']['updates']==36480, 'state endpoint clock')
        model.load_state_dict(state['ema']['module'],strict=True);del state
        validate_sampling(model,arm)
        before=tensor_digest(model.state_dict().items())
        model=model.to(device).eval().requires_grad_(False);criterion=criterion.to(device).eval()
        torch.cuda.reset_peak_memory_stats()
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        matched_ids=set()
        for start in range(0,32,4):
            pixels=[];targets=[];traces=[]
            for pos,row in enumerate(selected[start:start+4],start):
                raw=reader.raw(row);im=original(reader,row)
                try:
                    image,target,trace=transform(im,raw,epoch=120,seed=0,position=pos,image_id=row['stable_image_id'],training=False)
                finally:
                    im.close()
                require(not trace['geometry_steps'] and tuple(image.shape)==(3,768,1344), 'resize-only canvas')
                pixels.append(image);targets.append(target);traces.append(trace)
            require(count <16, 'forward cap')
            count+=1
            publish(out/('FORWARD-%02d.json'%count),dict(arm=arm,batch=start//4,forward_reserved=count,
                    ids=[r['stable_image_id'] for r in selected[start:start+4]]))
            buf=[]
            with torch.inference_mode(),capture(model,arm,buf),forward_guards(model):
                pred=model(torch.stack(pixels).to(device))
            torch.cuda.synchronize()
            require(torch.cuda.max_memory_allocated()<=18*1024**3,'allocated cap')
            with torch.inference_mode():
                gpu_targets=[{k:t[k].to(device) for k in ('labels','boxes')} for t in targets]
                matches=criterion.matcher(pred,gpu_targets)['indices']
            boxes=pred['pred_boxes'].detach().cpu()
            require(boxes.shape==(4,300,4) and torch.isfinite(boxes).all(), 'prediction shape/finite')
            for bi,(row,target,trace,(qi,ti)) in enumerate(zip(selected[start:start+4],targets,traces,matches)):
                qi,ti=qi.tolist(),ti.tolist()
                small=[(q,t) for q,t in zip(qi,ti) if 0<float(target['raw_area'][t])<=1024]
                images.append(dict(arm=arm,id=row['stable_image_id'],pixel_target_sha256=trace['augmented_tensor_sha256'],
                    valid_GT=len(target['labels']),small_GT=int(((target['raw_area']>0)&(target['raw_area']<=1024)).sum()),
                    matched_GT=len(qi),matched_small=len(small),raw_counts=trace['raw_counts']))
                for b in buf:
                    triggers=b['trigger'][bi]
                    inventory.append(dict(arm=arm,image_id=row['stable_image_id'],layer=b['layer'],
                        ordinary=300,below_cell_queries=torch.where(triggers)[0].tolist(),
                        matched_small=len(small),matched_small_triggered=sum(bool(triggers[q]) for q,t in small)))
                    for q,t in small:
                        gt=target['boxes'][t];g=torch.cat((gt[:2]-gt[2:]/2,gt[:2]+gt[2:]/2))
                        pb=boxes[bi,q];p=torch.cat((pb[:2]-pb[2:]/2,pb[:2]+pb[2:]/2))
                        inter=float((torch.minimum(p[2:],g[2:])-torch.maximum(p[:2],g[:2])).clamp(min=0).prod())
                        union=float(pb[2:].prod()+gt[2:].prod())-inter
                        a=support(b['loc'][bi,q],b['weights'][bi,q],b['shape'],g)
                        n=support(b['base'][bi,q],b['weights'][bi,q],b['shape'],g)
                        an,aw,av=stencil(b['loc'][bi,q],b['shape'])
                        nn,nw,nv=stencil(b['base'][bi,q],b['shape'])
                        changed=bool((an!=nn).any())
                        eff=effective_change(b['loc'][bi,q],b['base'][bi,q],b['shape'])
                        gid=target['gt_ids'][t];matched_ids.add((row['stable_image_id'],gid))
                        output.append([arm,row['stable_image_id'],gid,q,b['layer'],float(target['raw_area'][t]),inter/union,
                            float(b['ref'][bi,q,2]*b['shape'][1]),float(b['ref'][bi,q,3]*b['shape'][0]),bool(triggers[q]),
                            arm=='P' and bool(triggers[q]),a['A'],a['Z'],n['Z'],a['I'],n['I'],delta(a['I'],n['I']),
                            a['O'],n['O'],delta(a['O'],n['O']),a['V'],n['V'],delta(a['V'],n['V']),
                            a['M'],n['M'],delta(a['M'],n['M']),changed,eff,float(b['output_abs_mean'][bi,q])])
            del pred,buf,pixels,gpu_targets
            publish(out/('BATCH-%02d.json'%count),dict(arm=arm,images_completed=start+4,forward_completed=count,
                    peak_allocated=torch.cuda.max_memory_allocated()))
        after=tensor_digest(model.state_dict().items())
        np_before=rng[1];np_after=np.random.get_state()
        require(before==after and random.getstate()==rng[0] and np_before[0]==np_after[0]
                and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]
                and torch.equal(rng[2],torch.get_rng_state()) and torch.equal(rng[3],torch.cuda.get_rng_state()), 'model/RNG mutated')
        state_checks.append(dict(arm=arm,model_before=before,model_after=after,RNG_unchanged=True,matcher=recipe))
        peaks.append(dict(arm=arm,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
        del model,criterion
        gc.collect();torch.cuda.empty_cache()
    require(count==16 and len(images)==64, 'incomplete image/forward coverage')
    for i in range(32):
        require(images[i]['pixel_target_sha256']==images[i+32]['pixel_target_sha256'], 'paired inputs differ')
    publish(out/'MEASUREMENTS.json',dict(columns=COLUMNS,rows=output))
    publish(out/'INVENTORY.json',dict(images=images,queries=inventory,state_checks=state_checks,peaks=peaks))
    publish(out/'WORKER_RESULT.json',dict(status='PASS',forward_calls=count,images_per_arm=32,
        result=reference(out/'MEASUREMENTS.json'),inventory=reference(out/'INVENTORY.json'),
        DN='NOT_OBSERVED_EVAL',AP='NOT_COMPUTED',training=False,backward=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',required=True)
    parser.add_argument('--mode',required=True,choices=['contract-check','worker'])
    args=parser.parse_args()
    c,old,selected=load(args.contract)
    if args.mode=='contract-check':
        import torch
        require(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'hidden CUDA')
        print(json.dumps(dict(status='PASS',mode='contract-check',images=32,CUDA_initialized=False)))
        return
    out=Path(c['output'])
    require(out.is_dir() and (out/'CONSUMED.json').is_file(), 'supervisor authority')
    try:
        worker(c,old,selected,out)
    except BaseException:
        import traceback
        publish(out/'WORKER_STOP.json',dict(status='STOP_NO_RETRY',error=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
