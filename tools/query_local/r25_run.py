"""R25 production constructor, CPU acceptance, fixed-input response diagnostic. No AP/formal command."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,hashlib,copy,gc,traceback,random,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np
from sparse_rtdetr.research.query_local import install,roi_boxes,LocalHead,VERSION
from sparse_rtdetr.research.query_local_runtime import create,capture,save,restore,cpu,PARENT_SHA
from sparse_rtdetr.protocol_r18 import checkpoint as ck
from sparse_rtdetr.protocol_r18.model import Config,build,CommonEMA,fp32_outputs
from sparse_rtdetr.protocol_r18.engine import tensor_digest,learning_rate
from sparse_rtdetr.baseline.training_v2b import _optimizer
from sparse_rtdetr.baseline.training_v2b_engine import _move
from sparse_rtdetr.localization.data import RawRoleDataset,collate_window,seed_worker
W=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace')
DEFAULT_PARENT=W/'experiments/localization-distribution/rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-060.pth'
R=W/'experiments/localization-distribution/r25-query-local-regression-20261104-001'
DATAREF=W/'experiments/localization-distribution/r18-tiny-positives-20261102-001/production-contract-002.json'
MILESTONES=(0,1,2,8,32,64,128)

def write(p,v):
    p=Path(p)
    with p.open('x') as f:json.dump(v,f,indent=1,allow_nan=False)

def safety():
    import shutil
    if shutil.disk_usage(R).free<16*1024**3:raise RuntimeError('insufficient disk headroom for next diagnostic stage')

def setup(device):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    if device=='cuda:0':
        if torch.cuda.device_count()!=1:raise ValueError('single configured GPU required')
        torch.cuda.set_device(0)

def clone_rng(e):return ck.capture_rng(e.device)

def original_scale_delta(delta,target,trace):
    h,w=target['original_hw'].tolist();sign=1
    for op in trace['geometry_steps']:
        if op['operation']=='zoom':
            left,top,right,bottom=op['padding'];h+=top+bottom;w+=left+right
        elif op['operation']=='crop':h,w=op['height'],op['width']
        elif op['operation']=='horizontal_flip':sign=-sign
        else:raise ValueError('unknown geometry step')
    return delta*delta.new_tensor([sign*w,h,w,h])

def probe(e,images,targets,traces,out,label):
    rng=clone_rng(e);modes={n:m.training for n,m in e.model.named_modules()}
    e.model.eval();rows=[];pairs=[]
    try:
        with torch.no_grad():
            for i in range(0,len(images),8):
                x=images[i:i+8].to(e.device);t=_move(targets[i:i+8],e.device)
                with torch.autocast(e.device.type,dtype=torch.bfloat16,enabled=e.config.amp_dtype=='bfloat16'):
                    pred=e.model(x)
                obs=e.model.local_observation;coarse=obs['coarse'].float();refined=pred['pred_boxes'].float()
                assoc=e.criterion.matcher({'pred_logits':pred['pred_logits'].float(),'pred_boxes':coarse},t)['indices']
                native=type(e.criterion).__mro__[1].loss_boxes.__get__(e.criterion)
                c=native({'pred_boxes':coarse},t,assoc,max(sum(len(v['boxes']) for v in t),1),None)
                r=native({'pred_boxes':refined},t,assoc,max(sum(len(v['boxes']) for v in t),1),None)
                pix=(refined-coarse)*refined.new_tensor([1344,768,1344,768]);flat=pix.abs().flatten()
                rows.append({'coarse_regression':{k:float(v) for k,v in c.items()},'refined_regression_same_coarse_matches':{k:float(v) for k,v in r.items()},
                             'residual_canvas_abs_px_mean':float(flat.mean()),'residual_canvas_abs_px_quantiles':torch.quantile(flat,flat.new_tensor([0,.5,.9,.99,1])).tolist(),
                             'delta_logit_abs_mean':float(obs['delta'].float().abs().mean()),'F4_shape':obs['F4_shape'],'query_shape':obs['query_shape']})
                # Explicit original-image scale for each image; conversion is linear resize.
                orig=[cpu(original_scale_delta(refined[j]-coarse[j],targets[i+j],traces[i+j])) for j in range(len(x))]
                pairs.append({'coarse_normalized_cxcywh':cpu(coarse),'refined_normalized_cxcywh':cpu(refined),'canvas_delta_cxcywh_px':cpu(pix),
                              'original_inverse_affine_unclipped_delta_cxcywh_px':orig,'GT_association':cpu(assoc),'scores':cpu(pred['pred_logits'].sigmoid()),'reference_roi_canvas_xyxy':cpu(obs['rois'])})
        torch.save(pairs,out/(label+'-pairs.pth'))
    finally:
        for n,m in e.model.named_modules():m.training=modes[n]
        ck.restore_rng(rng,e.device)
    return rows

def param_report(e,initial,grad):
    result={}
    for label in ['shared','new']:
        # Use explicit boolean to avoid chained comparisons in group selection.
        items=[(n,p) for n,p in e.model.named_parameters() if (n.startswith('local_head.')) == (label=='new')]
        num=sum(float((p.detach().float().cpu()-initial[n].float()).square().sum()) for n,p in items)
        den=sum(float(initial[n].float().square().sum()) for n,p in items)
        steps={n:int(e.optimizer.state.get(p,{}).get('step',0)) for n,p in items}
        result[label]={'relative_l2_change':(num/max(den,1e-30))**.5,'absolute_l2_change':num**.5,
                       'step_min':min(steps.values(),default=0),'step_max':max(steps.values(),default=0),
                       'lr':[g['lr'] for g in e.optimizer.param_groups if (g['name']=='r25_new')==(label=='new')]}
    result['gradients']=grad
    return result

class DiagnosticSampler:
    def __init__(self,indices):self.indices=indices
    def __iter__(self):
        for win in range(4):yield [(61,win*16+j,index) for j,index in enumerate(self.indices[win*16:(win+1)*16])]
    def __len__(self):return 4

def materialize(out):
    data=json.loads(DATAREF.read_text())['data']['train_core'];v=json.loads(Path(data['manifest']).read_text())
    records=v['records'];ids=[r['stable_image_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('nonunique native image_id')
    selected=sorted(range(len(ids)),key=lambda i:hashlib.sha256(ids[i].encode()).hexdigest())[:64]
    selection={'rule':'sha256(native image_id utf8) lexicographic first64; native image_id=stable_image_id in RawRoleDataset','ids':[ids[i] for i in selected],
               'manifest':str(data['manifest']),'manifest_sha256':ck.file_sha(data['manifest']),'indices':selected}
    write(out/'SELECTION.json',selection) # Commit selection before any image or annotation access.
    roots=data['roots'];ds=RawRoleDataset(role='train_core',manifest_path=data['manifest'],images_path=roots['images'],annotations_path=roots['annotations'],seed=0,root_binding=roots)
    loader=torch.utils.data.DataLoader(ds,batch_sampler=DiagnosticSampler(selected),num_workers=2,multiprocessing_context='spawn',
                  prefetch_factor=2,collate_fn=collate_window,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(25001))
    start=time.monotonic();windows=[]
    for x,t,tr in loader:
        assert [z['image_id'] for z in tr]==selection['ids'][len(windows)*16:(len(windows)+1)*16]
        windows.append((x,t,tr))
    assert len(windows)==4
    torch.save(windows,out/'fixed-inputs.pth')
    write(out/'INPUTS.json',{'windows':4,'images':64,'epoch_transform':61,'multiprocess_workers':2,'input_processing_seconds':time.monotonic()-start,
             'tensor_sha256':[tensor_digest([('images',x)]) for x,_,_ in windows],
             'augmented_sha256':[[z['augmented_tensor_sha256'] for z in tr] for _,_,tr in windows],
             'file_sha256':ck.file_sha(out/'fixed-inputs.pth'),'bytes':(out/'fixed-inputs.pth').stat().st_size})
    return windows

def train_arm(arm,parent,windows,out):
    safety();torch.cuda.reset_peak_memory_stats();e,bridge=create(parent,arm,'cuda:0');write(out/'BRIDGE.json',bridge)
    initial={n:p.detach().cpu().clone() for n,p in e.model.named_parameters()}
    save(e,out/'initial.pth');responses={};timings=[];grad={};captures=[]
    # Capture actual complete-window clipped gradients immediately before optimizer update.
    def pre_step(opt,args,kwargs):
        for group in opt.param_groups:
            values=[p.grad.detach().float().norm() for p in group['params'] if p.grad is not None]
            grad[group['name']+'_clipped_norm']=float(torch.stack(values).norm()) if values else 0.
        grad['new_parameter_clipped_norms']={n:float(p.grad.detach().float().norm()) if p.grad is not None else None
                for n,p in e.model.named_parameters() if n.startswith('local_head.')}
    handle=e.optimizer.register_step_pre_hook(pre_step)
    e.model.capture_local_gradients=True
    # Local path input gradient hooks are attached by model only for this diagnostic.
    def grad_hook(mod,args,out0):
        if not torch.is_grad_enabled():return
        q,f=e.model.local_gradient_tensors
        if q.requires_grad:q.register_hook(lambda g:grad.__setitem__('local_q_gradient_norm',float(g.float().norm())))
        if f.requires_grad:f.register_hook(lambda g:grad.__setitem__('F4_total_gradient_norm',float(g.float().norm())))
    h=e.model.register_forward_hook(grad_hook)
    x0=torch.cat([w[0] for w in windows]);t0=[t for w in windows for t in w[1]];tr0=[t for w in windows for t in w[2]]
    responses['0']={'parameters':param_report(e,initial,{}),'probe':probe(e,x0,t0,tr0,out,'u000')}
    start=time.monotonic()
    for u in range(1,129):
        safety();x,t,_=windows[(u-1)%4];grad.clear()
        result=e.train_window(x,t,capture=False)
        if not result['finite'] or e.completed_updates!=18240+u:raise ValueError('R25 update clock/finite')
        timings.append(result['elapsed_seconds'])
        # Global clip coefficient from native return; preclip group norms reconstruct exact scaling.
        factor=min(1.,.1/(result['gradient_norm_before_clip']+1e-6))
        for k,v in list(grad.items()):
            if k.endswith('_clipped_norm'):grad[k.replace('_clipped_norm','_raw_norm')]=v/max(factor,1e-30)
        grad['new_parameter_raw_norms']={n:None if v is None else v/max(factor,1e-30) for n,v in grad['new_parameter_clipped_norms'].items()}
        if u in MILESTONES:
            responses[str(u)]={'parameters':param_report(e,initial,dict(grad)),'native_update':result,'probe':probe(e,x0,t0,tr0,out,f'u{u:03d}')}
            write(out/f'RESPONSE-u{u:03d}.json',responses[str(u)])
        write(out/f'update-{u:03d}.json',{'global_update':e.completed_updates,'local_update':u,'loss':result['loss'],
              'gradient_norm_before_clip':result['gradient_norm_before_clip'],'elapsed_seconds':result['elapsed_seconds'],'finite':True})
        if u in (1,2):save(e,out/f'u{u:03d}.pth')
    handle.remove();h.remove();e.model.capture_local_gradients=False
    if hasattr(e.model,'local_gradient_tensors'):del e.model.local_gradient_tensors
    final=save(e,out/'final.pth')
    write(out/'RESPONSES.json',responses)
    write(out/'TRAIN_RESULT.json',{'status':'PASS','arm':arm,'actual_updates':128,'global_update':e.completed_updates,'EMA_updates':e.ema.updates,
        'update_seconds':timings,'steady_update_mean_seconds':sum(timings[64:])/64,'wall_seconds':time.monotonic()-start,
        'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'endpoint':final,'formal_usable':False})
    del e,initial;gc.collect();torch.cuda.empty_cache()
    # One separately costed replay update, not another effective diagnostic update.
    clone,_=create(parent,arm,'cuda:0')
    one=json.loads((out/'u001.json').read_text());two=json.loads((out/'u002.json').read_text())
    receipt=restore(clone,one['path'],one['sha256'])
    replay=clone.train_window(windows[1][0],windows[1][1],capture=False)
    expected=torch.load(two['path'],map_location='cpu',weights_only=True)
    if not ck.exact_equal(expected,capture(clone)):raise ValueError('CUDA full state replay differs')
    write(out/'CUDA_REPLAY.json',{'status':'PASS','restore':receipt,'full_state_exact':True,
              'actual_additional_updates':1,'forward_calls':2,'elapsed_seconds':replay['elapsed_seconds']})
    del clone,expected;gc.collect();torch.cuda.empty_cache()

def benchmark(parent,windows,out):
    result={}
    # Fixed complete64 input, no certified scorer/postprocessor. output materialization is timed separately.
    for arm in ['A0','A1','A2']:
        torch.cuda.reset_peak_memory_stats();e,bridge=create(parent,arm,'cuda:0');e.model.eval();torch.cuda.synchronize();start=time.monotonic()
        forwards=[];output_seconds=0.;input_seconds=0.
        with torch.no_grad():
            for x,_,_ in windows:
                for off in range(0,16,4):
                    tt=time.monotonic();batch=x[off:off+4].to(e.device);torch.cuda.synchronize();input_seconds+=time.monotonic()-tt
                    tt=time.monotonic()
                    with torch.autocast('cuda',enabled=False):v=e.model(batch)
                    torch.cuda.synchronize();forwards.append(time.monotonic()-tt)
                    tt=time.monotonic();raw={k:cpu(v[k]) for k in ('pred_boxes','pred_logits')}
                    from sparse_rtdetr.r0.evaluation import rank_candidates
                    selection=json.loads((out/'SELECTION.json').read_text());data=json.loads(DATAREF.read_text())['data']['train_core']
                    records=json.loads(Path(data['manifest']).read_text())['records'];byid={r['stable_image_id']:r for r in records}
                    for j in range(4):
                        row=byid[selection['ids'][len(forwards)*4-4+j]]
                        for budget in (600,300):rank_candidates(raw['pred_logits'][j],raw['pred_boxes'][j],row,budget)
                    output_seconds+=time.monotonic()-tt
        # Train same four windows from a fresh full parent, not from an inference-modified optimizer.
        del e;gc.collect();torch.cuda.empty_cache();e,_=create(parent,arm,'cuda:0');train=[]
        for x,t,_ in windows:train.append(e.train_window(x,t)['elapsed_seconds'])
        result[arm]={'inference_forwards':16,'inference_images':64,'batch_size':4,'forward_seconds':forwards,
                    'input_transfer_seconds':input_seconds,'output_tensor_CPU_seconds':output_seconds,'inference_total_seconds':sum(forwards)+input_seconds+output_seconds,
                    'train_updates':4,'train_forwards':8,'train_update_seconds':train,'warmup_policy':'no extra warmup; first measured forward/update reported separately, remaining summarized',
                    'prediction_postprocessing':'FP32 original inference plus native rank_candidates top600/top300; includes CPU materialization, no AP scorer',
                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        del e;gc.collect();torch.cuda.empty_cache()
    write(out/'BENCHMARK.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu','diagnostic']);p.add_argument('--out',required=True);p.add_argument('--parent',default=str(DEFAULT_PARENT));p.add_argument('--manifest');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        if a.manifest:
            mf=json.loads(Path(a.manifest).read_text())
            if mf['task_sha256']!='ec7ef314720179855ca8c6a74be4e37f65495891d5f0227f5736e78aabe71241':raise ValueError('task definition binding')
            for row in mf['source_files']:
                pp=ROOT/row['path']
                if pp.stat().st_size!=row['bytes'] or ck.file_sha(pp)!=row['sha256']:raise ValueError('frozen source changed '+row['path'])
            write(out/'SOURCE_BINDING.json',{'manifest':a.manifest,'sha256':ck.file_sha(a.manifest),'commit':mf['commit'],'files':len(mf['source_files'])})
        elif a.action=='diagnostic':raise ValueError('diagnostic requires frozen source manifest')
        if a.action=='cpu':
            setup('cpu');from r25_tests import run;run(out,a.parent)
        else:
            setup('cuda:0');safety();windows=materialize(out)
            for arm in ['A1','A2']:
                d=out/arm;d.mkdir();train_arm(arm,a.parent,windows,d)
            benchmark(a.parent,windows,out)
        write(out/'CLOSED.json',{'status':'PASS','action':a.action,'elapsed_seconds':time.monotonic()-start,'no_formal_training':True,'development_scores':0})
    except BaseException as exc:
        write(out/'STOP.json',{'status':'STOP','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,
              'peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
              'peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
        raise
if __name__=='__main__':main()
