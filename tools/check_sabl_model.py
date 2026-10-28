"""Native-model constructed CPU checks; not real-weight or scientific evidence."""
import os,sys,json,time,resource,signal,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU'}.items():os.environ[k]=v
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.alarm(300)
import torch
from sparse_rtdetr.research.sabl_control import *
from sparse_rtdetr.research.sabl_evaluation import rank_refined
from sparse_rtdetr.protocol_r18.model import dn_capture
from sparse_rtdetr.protocol_r18.engine import output_digest,tensor_digest
from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
from sparse_rtdetr.localization.data import WindowSampler
from sparse_rtdetr.protocol_r18.runio import canonical

def main(dest):
    started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    checks=[];forwards=0
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError):checks.append(name);return
        raise AssertionError('accepted '+name)
    cfg=common.Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    models={};criteria={}
    for arm in ('N','S','A'):models[arm],criteria[arm]=build(arm,cfg)
    check('shared_native_parameters_BN_caches',all(_equal(models['N'].native.state_dict(),models[a].native.state_dict()) for a in ('S','A')))
    check('same_S_A_head_initial',_equal(models['S'].boundary.state_dict(),models['A'].boundary.state_dict()))
    check('parameter_inventory',all(sum(p.numel() for p in m.parameters())==PARAMETERS[a] for a,m in models.items()))
    images=torch.rand(2,3,128,128)
    n=models['N'].eval();before=tensor_digest(n.state_dict().items())
    with torch.no_grad():
        original=n.native(images);forwards+=1
        wrapper=n(images);forwards+=1
    check('N_eval_bypass_bitwise',_equal(original,wrapper) and before==tensor_digest(n.state_dict().items()))
    targets=[]
    for i in range(2):
        targets.append({'boxes':torch.tensor([[.4,.45,.2,.18],[.7,.7,.15,.2]]),'labels':torch.tensor([0,1]),'gt_ids':['s%d-a'%i,'s%d-b'%i],
        'raw_area':torch.tensor([400.,1200.]),'ignore_spatial_boxes':torch.empty(0,4),'ignore_union':torch.empty(0,4),
        'ignore_class_boxes':torch.empty(0,4),'ignore_class_labels':torch.empty(0,dtype=torch.long),'ignore_ids':[]})
    outputs={};losses={};noise={};f8=[]
    h=models['S'].encoder.register_forward_hook(lambda m,a,o:(o[0].retain_grad(),f8.append(o[0])) and None)
    for arm in models:
        models[arm].train();torch.manual_seed(17)
        with dn_capture(models[arm]) as dn:outputs[arm]=models[arm](images,targets)
        forwards+=1;noise[arm]=output_digest(dn)
        losses[arm]=criteria[arm](common.fp32_outputs(outputs[arm]),targets)
    h.remove()
    check('same_DN_noise_and_masks',len(set(noise.values()))==1)
    check('native_training_outputs_aux_DN_bitwise',all(_equal(outputs['N'],{k:v for k,v in outputs[a].items() if k!='sabl'}) for a in ('S','A')))
    check('native_21_losses_unchanged',len(losses['N'])==21 and all(all(torch.equal(v,losses[a][k]) for k,v in losses['N'].items()) for a in ('S','A')))
    check('native_matcher_recipe',criteria['S'].matcher.cost_bbox==5 and criteria['S'].matcher.cost_giou==2 and criteria['S'].matcher.cost_class==2)
    check('ordinary_Q300_only',outputs['S']['sabl']['logits'].shape==(2,300,4,7) and len(outputs['S']['dn_aux_outputs'])==3)
    check('reference_detached',not outputs['S']['sabl']['reference'].requires_grad)
    coarse=outputs['S']['pred_boxes'];coarse.retain_grad()
    extra=losses['S']['loss_bucket']+losses['S']['loss_offset'];extra.backward()
    check('new_gradient_reaches_head',sum(float(p.grad.abs().sum()) for p in models['S'].boundary.parameters() if p.grad is not None)>0)
    check('new_gradient_reaches_F8',len(f8)==1 and f8[0].grad is not None and bool((f8[0].grad!=0).any()))
    check('new_gradient_reaches_backbone',any(p.grad is not None and bool((p.grad!=0).any()) for p in models['S'].native.backbone.parameters()))
    check('no_new_gradient_through_reference_or_decoder',coarse.grad is None and all(p.grad is None for p in models['S'].decoder.parameters()))
    check('positive_match_denominators',criteria['S'].last_diagnostics[-1]=={'head':'SABL_ordinary_only','matches':4,'bucket_denominator':16,'offset_denominator':4})
    # Scalar arithmetic, unequal width/height and the pinned side ordering.
    r=torch.tensor([[.5,.5,.2,.4]],requires_grad=True);x,scale=buckets(r)
    check('four_side_directions',bool((x[0,0].diff()>0).all() and (x[0,1].diff()<0).all() and (x[0,2].diff()>0).all() and (x[0,3].diff()<0).all()))
    check('unequal_bucket_step',torch.allclose(scale[0],torch.tensor([.34/14,.34/14,.68/14,.68/14])))
    gt=torch.tensor([[.51,.48,.21,.38]]);t,y,cw,ow=encode(r,gt)
    check('nearest_bucket_label',torch.equal(y.argmax(-1),t.abs().argmin(-1)) and bool((y.sum(-1)==1).all()))
    check('nearest_plus_neighbor_rule',bool((ow.sum(-1)>=1).all() and (ow.sum(-1)<=2).all()) and bool((cw[y==1]==1).all()))
    z=torch.zeros(1,1,4,7,requires_grad=True);d=torch.zeros_like(z,requires_grad=True)
    pred={'logits':z,'offsets':d,'reference':r.reshape(1,1,4)}
    ls,diag=boundary_loss(pred,[{'boxes':gt}],[(torch.tensor([0]),torch.tensor([0]))])
    check('loss_denominators_exact',torch.equal(ls['loss_bucket'],(F.binary_cross_entropy_with_logits(z[0],y,reduction='none')*cw).sum()/4) and torch.equal(ls['loss_offset'],(F.smooth_l1_loss(d[0],t,beta=.1,reduction='none')*ow).sum()))
    empty,ed=boundary_loss(pred,[{'boxes':gt}],[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]);sum(empty.values()).backward()
    check('empty_match_differentiable_zero',all(float(v)==0 for v in empty.values()) and z.grad is not None and d.grad is not None)
    check('GT_reference_no_gradient',r.grad is None and not t.requires_grad)
    # Independently known RoIAlign linear-field coordinate expectation.
    field=(torch.arange(6)[None,:]+10*torch.arange(4)[:,None]).float()[None,None].expand(1,256,4,6).clone().requires_grad_()
    reference=torch.tensor([[[.5,.5,1/6,1/4]]],requires_grad=True)
    roi=roi_features(field,reference,(32,48))
    xx=2+(torch.arange(7)+.5)/7;yy=1+(torch.arange(7)+.5)/7
    expected=xx[None,:]+10*yy[:,None]
    check('RoI_non_square_aligned_coordinate',torch.allclose(roi[0,0],expected,atol=1e-5,rtol=0))
    roi.sum().backward();check('RoI_grad_feature_not_reference',field.grad is not None and reference.grad is None)
    outside=roi_features(field.detach(),torch.tensor([[[3.,3.,.1,.1]]]),(32,48));check('outside_RoI_zero',bool((outside==0).all()))
    border=roi_features(torch.ones_like(field),torch.tensor([[[0.,0.,.1,.1]]]),(32,48));check('boundary_RoI_finite',bool(torch.isfinite(border).all()))
    head=models['S'].boundary
    rr=torch.randn(2,256,7,7);perm=rr.flatten(-2).flip(-1).reshape_as(rr)
    zs,ds=head(rr,'S');za,da=head(rr,'A');zp,dp=head(perm,'A');zsp,dsp=head(perm,'S')
    check('A_spatial_permutation_invariant',torch.allclose(za,zp,atol=1e-7,rtol=1e-5) and torch.allclose(da,dp,atol=1e-7,rtol=1e-5))
    check('S_can_use_spatial_order',not torch.equal(zs,zsp))
    constant=torch.ones(1,256,7,7);check('constant_feature_control_equal',_equal(head(constant,'S'),head(constant,'A')))
    zero=torch.zeros(1,4,7);offset=torch.zeros_like(zero);valid_box,valid=decode(r.detach(),zero,offset)
    check('stable_argmax_tie_lowest',torch.allclose(valid_box[:,0:2],r.detach()[:,0:2]) and bool(valid.all()))
    bad=offset.clone();bad[:,0,:]=-100;bad[:,1,:]=100
    box,mask=decode(r.detach(),zero,bad);check('inverse_box_zero_area_no_swap',not bool(mask.any()) and bool((box[:,2:]==0).all()))
    rejects('nonfinite_reference_refused',lambda:buckets(torch.full((1,4),float('nan'))))
    rejects('nonpositive_reference_refused',lambda:buckets(torch.tensor([[.5,.5,0.,.1]])))
    rejects('nonfinite_offset_refused',lambda:decode(r.detach(),zero,torch.full_like(offset,float('inf'))))
    boxes=torch.tensor([.5,.5,.2,.2]).repeat(300,1);valid=torch.ones(300,dtype=torch.bool);valid[0]=False;boxes[0,2:]=0
    ranked=rank_refined(torch.zeros(300,10),boxes,valid,{'width':48,'height':32,'stable_image_id':'SYNTHETIC'},300)
    check('invalid_removed_AFTER_budget_no_refill',len(ranked)==290 and max(p['query_id'] for p in ranked)==29)
    keys=[k for batch in WindowSampler(4869,seed=0,epoch=1,completed_windows=0,window_limit=2) for k in batch]
    check('real_sampler_two_logical16_windows',len(keys)==32 and [x[1] for x in keys]==list(range(32)) and len({x[2] for x in keys})==32)
    for arm in ('S','A'):
        models[arm].eval()
        with torch.no_grad():out=models[arm](images)
        forwards+=1;check(arm+'_inference_unique_Q300',out['pred_boxes'].shape==(2,300,4) and out['refined_valid'].shape==(2,300))
    empty_targets=[{**t,'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'gt_ids':[],'raw_area':torch.empty(0)} for t in targets]
    models['S'].train();empty_out=models['S'](images,empty_targets);forwards+=1
    empty_losses=criteria['S'](common.fp32_outputs(empty_out),empty_targets)
    check('native_empty_match_full_model',float(empty_losses['loss_bucket'])==0 and float(empty_losses['loss_offset'])==0 and all(bool(torch.isfinite(x)) for x in empty_losses.values()))
    result={'status':'PASS','scope':'CPU_CONSTRUCTED_NO_REAL_WEIGHTS_OR_IMAGES','checks':checks,'CPU_detector_forward':forwards,'CPU_optimizer_updates':0,'real_GPU_consumed':0,'seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'CUDA_initialized':torch.cuda.is_initialized()}
    Path(dest).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
