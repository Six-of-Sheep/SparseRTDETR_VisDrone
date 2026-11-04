"""Actual R25 CLI CPU acceptance, independent geometry and complete replay."""
import copy,json,gc,time
from pathlib import Path
import torch
from r25_run import write,create,save,restore,capture,cpu,install,LocalHead,roi_boxes,Config,build,CommonEMA,_optimizer,ck,tensor_digest
from sparse_rtdetr.protocol_r18.engine import Engine
from sparse_rtdetr.research.query_local import VERSION

def run(out,parent):
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    from r25_run import original_scale_delta
    delta=original_scale_delta(torch.tensor([.1,.2,.03,.04]),{'original_hw':torch.tensor([100,200])},
       {'geometry_steps':[{'operation':'zoom','padding':[10,20,30,40]},{'operation':'crop','height':80,'width':120},{'operation':'horizontal_flip'}]})
    check('inverse_affine_original_pixel_units',torch.allclose(delta,torch.tensor([-12.,16.,3.6,3.2])))
    from torchvision.ops import roi_align
    # Independent pixel geometry, two distinct images, non-square, clipped and tiny ROIs.
    b=torch.tensor([[[.5,.5,.2,.4],[.01,.99,.01,.01]],[[.6,.3,.02,.01],[.9,.1,.2,.2]]])
    rois=roi_boxes(b,(128,256));expected=[]
    for batch in range(2):
        for cx,cy,w,h in b[batch].tolist():expected.append([batch,max(0,cx-.75*w)*256,max(0,cy-.75*h)*128,min(1,cx+.75*w)*256,min(1,cy+.75*h)*128])
    check('independent_roi_xyxy_batch_clip',torch.allclose(rois,torch.tensor(expected),atol=2e-5))
    y,x=torch.meshgrid(torch.arange(32.),torch.arange(64.),indexing='ij')
    f=torch.stack([x+3*y,1000+x+3*y])[:,None]
    got=roi_align(f,rois[:1],(7,7),spatial_scale=.25,aligned=True,sampling_ratio=2)[0,0]
    r=expected[0];xx=torch.arange(7.)+.5;yy=torch.arange(7.)+.5
    expected_grid=(r[1]/4-.5+xx[None,:]*(r[3]-r[1])/4/7)+3*(r[2]/4-.5+yy[:,None]*(r[4]-r[2])/4/7)
    check('independent_aligned_linear_ramp',torch.allclose(got,expected_grid,atol=2e-5))
    const=torch.stack([torch.ones(1,32,64)*7,torch.ones(1,32,64)*19]);g=roi_align(const,rois,(7,7),.25,2,True)
    check('batch_separation',torch.allclose(g[:2],torch.full_like(g[:2],7)) and torch.allclose(g[2:],torch.full_like(g[2:],19)))
    # Real e060 named bridge on CPU, no fake reset of existing optimizer/EMA history.
    refs=[]
    for arm in ['A0','A1','A2']:
        e,bridge=create(parent,arm,'cpu');write(out/(arm+'-e060-bridge.json'),bridge)
        check(arm+'.parent_clock',e.completed_updates==e.ema.updates==18240)
        if arm!='A0':refs.append(tensor_digest(e.model.local_head.state_dict().items()))
        del e;gc.collect()
    check('A1_A2_exact_initial_new_parameters',refs[0]==refs[1])
    # Synthetic input, actual complete native model/criterion/engine, smaller spatial cache only.
    cfg=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    m,c,info=build(cfg,synthetic=True);base=Engine(m,c,_optimizer(m,cfg),CommonEMA(m),cfg)
    binding={'scope':'cpu_R18_synthetic','arm':'RT-DETRv2-R18-common-protocol-v1','config':cfg.binding(),'model':{'config':cfg.binding()}}
    sp=out/'synthetic-native.pth';torch.save(ck.capture(base,binding),sp)
    x=torch.rand(2,3,128,128)
    targets=[{'boxes':torch.tensor([[.5,.5,.15,.2]]),'labels':torch.tensor([1]),'gt_ids':['synthetic-gt'],
              'ignore_spatial_boxes':torch.zeros(0,4),'ignore_union':torch.zeros(0,4),
              'ignore_class_boxes':torch.zeros(0,4),'ignore_class_labels':torch.zeros(0,dtype=torch.long)} for _ in range(2)]
    m.eval()
    with torch.no_grad():native=m(x[:1])
    native_state=copy.deepcopy(m.state_dict())
    for arm in ['A1','A2']:
        q=copy.deepcopy(m);install(q,arm);q.eval()
        with torch.no_grad():v=q(x[:1])
        check(arm+'.zero_residual_logits_equal',torch.equal(v['pred_logits'],native['pred_logits']))
        check(arm+'.zero_residual_boxes_tolerance',float((v['pred_boxes']-native['pred_boxes']).abs().max())<=1e-6)
        check(arm+'.coarse_identity',torch.equal(q.local_observation['coarse'],native['pred_boxes']))
        check(arm+'.all_Q300_F4',q.local_observation['query_shape']==[1,300,256] and q.local_observation['F4_shape']==[1,64,32,32])
        del q
        e,bridge=create(sp,arm,'cpu',config=cfg,synthetic=True)
        e.model.capture_local_gradients=True
        grads={}
        handles=[p.register_hook(lambda g,n=n:grads.__setitem__(n,float(g.norm()))) for n,p in e.model.local_head.named_parameters()]
        r1=e.train_window(x,targets,capture=True)
        check(arm+'.first_out_gradient',grads['out.weight']>0 and grads['out.bias']>0)
        check(arm+'.first_internal_zero_permitted',grads['wk.weight']==0)
        one=save(e,out/(arm+'-u1.pth'));grads.clear()
        r2=e.train_window(x,targets,capture=True)
        check(arm+'.second_internal_gradient',grads['wk.weight']>0 and grads['wq.weight']>0 and grads['wv.weight']>0)
        qg,fg=e.model.local_gradient_tensors
        check(arm+'.q_F4_gradient',qg.grad is not None and fg.grad is not None and bool(torch.isfinite(qg.grad).all()) and float(qg.grad.abs().sum())>0 and bool(torch.isfinite(fg.grad).all()) and float(fg.grad.abs().sum())>0)
        for h in handles:h.remove()
        del e.model.local_gradient_tensors;e.model.capture_local_gradients=False
        expected=capture(e)
        other='A2' if arm=='A1' else 'A1';bad,_=create(sp,other,'cpu',config=cfg,synthetic=True)
        try:restore(bad,one['path'],one['sha256'])
        except ValueError:rejected=True
        else:rejected=False
        check(arm+'.cross_arm_rejected',rejected);del bad
        clone,_=create(sp,arm,'cpu',config=cfg,synthetic=True)
        rr=restore(clone,one['path'],one['sha256']);replay=clone.train_window(x,targets,capture=True)
        check(arm+'.save_restore_replay_full_exact',ck.exact_equal(expected,capture(clone)))
        check(arm+'.replay_loss_grad_DN_exact',{k:v for k,v in r2.items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'})
        check(arm+'.aux_DN_loss_count',len(r1['microbatch_losses'][0])==21)
        write(out/(arm+'-replay.json'),rr)
        del clone,e;gc.collect()
    # Inference uses no GT and no extra branch. A1 averages ordinary queries only by contract.
    q=torch.rand(1,300,256);b=torch.rand(1,300,4)*.5+.2;f=torch.rand(1,64,32,32)
    h1=LocalHead('A1');h2=LocalHead('A2');h2.load_state_dict(h1.state_dict())
    with torch.no_grad():h1.out.weight.fill_(.01);h2.out.weight.copy_(h1.out.weight)
    _,_,_,a1=h1(q,b,f,(128,128));_,_,_,a2=h2(q,b,f,(128,128))
    check('conditional_attention_nontrivial_difference',not torch.equal(a1,a2))
    check('fixed_position_not_EMA_averaged','relative_xy' not in h1.state_dict())
    check('key_and_position_no_bias',h1.wk.bias is None and h1.wp.bias is None)
    write(out/'TESTS.json',{'status':'PASS','checks':checks,'count':len(checks),'CUDA_initialized':torch.cuda.is_initialized(),
          'scope':'CPU actual CLI, real full model/criterion/engine synthetic save/replay plus real e060 named import; CUDA still unverified'})
    print('CPU_PASS',len(checks),flush=True)
