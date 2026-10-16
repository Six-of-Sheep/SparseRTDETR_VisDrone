import copy,json,math
from pathlib import Path
import pytest,torch
from sparse_rtdetr.protocol_r18.model import Config,build,native_objects,fp32_outputs,dn_capture,CommonEMA,ARM
from sparse_rtdetr.protocol_r18.engine import Engine,learning_rate,tensor_digest,output_digest
from sparse_rtdetr.protocol_r18 import checkpoint
from sparse_rtdetr.protocol_r18.criterion import adapt
from sparse_rtdetr.protocol_r18.geometry import validate_geometry
from sparse_rtdetr.baseline.training_v2b import _optimizer

torch.set_num_threads(2)
def cfg(dtype='float32'):return Config(input_hw=(128,128),physical_batch_size=1,amp_dtype=dtype)
def targets(counts=(2,)):
    result=[]
    for i,n in enumerate(counts):
        result.append({'boxes':torch.tensor([[.4,.4,.15,.2],[.7,.7,.1,.12]][:n],dtype=torch.float32).reshape(n,4),
            'labels':torch.tensor([1,4][:n],dtype=torch.int64),'gt_ids':[f'sample{i}/raw{j}' for j in range(n)],
            'ignore_union':torch.empty(0,4,dtype=torch.float64),'ignore_class_boxes':torch.empty(0,4,dtype=torch.float64),
            'ignore_class_labels':torch.empty(0,dtype=torch.int64)})
    return result
def leaf(v):
    if isinstance(v,torch.Tensor):return v.detach().clone().requires_grad_(v.is_floating_point())
    if isinstance(v,dict):return {k:leaf(x) for k,x in v.items()}
    if isinstance(v,list):return [leaf(x) for x in v]
    return v
def all_leaves(v):
    if isinstance(v,torch.Tensor) and v.requires_grad:yield v
    elif isinstance(v,dict):
        for x in v.values():yield from all_leaves(x)
    elif isinstance(v,list):
        for x in v:yield from all_leaves(x)
def sample_outputs(counts=(2,)):
    model,base,_,_,_=native_objects(cfg());model.train()
    with torch.no_grad():out=model(torch.rand(len(counts),3,128,128),targets(counts))
    return out,base

def test_native_architecture_initialization_and_effective_config():
    model,c,info=build(cfg(),synthetic=True);validate_geometry(model)
    assert info['trainable_parameters']==20094584
    assert type(model.decoder).__name__=='RTDETRTransformerv2' and not hasattr(model.decoder,'pre_bbox_head')
    assert c.share_matched_indices is False and c.losses==['vfl','boxes']
    assert info['effective_config']['eval_spatial_size']==[128,128]
    assert info['config']['pre_head'] is info['config']['feature_history_add'] is info['config']['GO'] is False
    # CPU construction isolates model random draws. No CUDA is initialized.
    before=torch.random.get_rng_state();m2,_,i2=build(cfg(),synthetic=True)
    assert torch.equal(before,torch.random.get_rng_state())
    assert tensor_digest(model.state_dict().items())==tensor_digest(m2.state_dict().items())
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('counts',[(2,),(0,),(2,0)])
def test_all_native_losses_and_gradients_exact_without_ignore(counts):
    out,base=sample_outputs(counts);a,b=leaf(out),leaf(out);common=adapt(copy.deepcopy(base));t=targets(counts)
    x=base(fp32_outputs(a),t);y=common(fp32_outputs(b),t)
    assert len(x)==len(y)==(21 if any(counts) else 12) and x.keys()==y.keys()
    assert all(torch.equal(x[k],y[k]) for k in x)
    sum(x.values()).backward();sum(y.values()).backward()
    assert all((u.grad is None and v.grad is None) or torch.equal(u.grad,v.grad) for u,v in zip(all_leaves(a),all_leaves(b)))
    assert len(common.last_diagnostics)==(7 if any(counts) else 4)
    assert all(r['denominator']==(max(sum(counts),1)*(out['dn_meta']['dn_num_group'] if r['head'].startswith('dn') else 1)) for r in common.last_diagnostics)

def test_native_weighting_once_and_each_head_own_matching():
    out,base=sample_outputs();unit=copy.deepcopy(base);unit.weight_dict={k:1 for k in base.weight_dict}
    a=base(out,targets());b=unit(out,targets())
    for key in a:
        factor=5 if key.startswith('loss_bbox') else 2 if key.startswith('loss_giou') else 1
        assert torch.equal(a[key],b[key]*factor)
    seen=[];h=base.matcher.register_forward_hook(lambda m,args,r:seen.append(r['indices']))
    try:base(out,targets())
    finally:h.remove()
    assert len(seen)==4 # final, aux0, aux1, encoder; no GO union or extra pre

def test_raw_ignore_only_unmatched_negatives_and_native_DN_padding():
    out,base=sample_outputs((2,0));common=adapt(copy.deepcopy(base));t=targets((2,0));normal=common(out,t)
    for row in t:row['ignore_union']=torch.tensor([[0.,0.,1.,1.]],dtype=torch.float64)
    ignored=common(out,t)
    for key in normal:
        if 'vfl' not in key or '_dn_' in key:assert torch.equal(normal[key],ignored[key])
        else:assert ignored[key]<normal[key]
    assert out['dn_meta']['dn_num_split'][0]>0
    a=leaf(out);loss=common(a,t);sum(loss.values()).backward()
    assert a['dn_aux_outputs'][0]['pred_logits'].grad[1].abs().sum()>0 # empty-image padding classification retained
    assert a['dn_aux_outputs'][0]['pred_boxes'].grad[1].abs().sum()==0 # native positive-only regression

def test_ignore_union_class_any_matched_and_full_prediction_denominator():
    _,base=sample_outputs();c=adapt(base)
    boxes=torch.tensor([[[.5,.5,.2,.2],[.5,.5,.2,.2],[.5,.5,.2,.2]]]);logits=torch.zeros(1,3,10,requires_grad=True)
    t=targets((1,));t[0]['boxes']=boxes[0,:1].clone();t[0]['labels']=torch.tensor([1])
    t[0]['ignore_union']=torch.tensor([[.4,.4,.5,.6],[.5,.4,.6,.6]],dtype=torch.float64)
    indices=[(torch.tensor([0]),torch.tensor([0]))]
    c.loss_labels_vfl({'pred_boxes':boxes,'pred_logits':logits},t,indices,1)['loss_vfl'].backward()
    assert logits.grad[0,0].abs().sum()>0 and logits.grad[0,1:].abs().sum()==0
    t[0]['ignore_union']=torch.empty(0,4,dtype=torch.float64)
    # Two same-class rectangles individually below .5; sum would suppress, any must not.
    t[0]['ignore_class_boxes']=torch.tensor([[.4,.4,.48,.6],[.52,.4,.6,.6]],dtype=torch.float64)
    t[0]['ignore_class_labels']=torch.tensor([1,1]);z=torch.zeros_like(logits,requires_grad=True)
    c.loss_labels_vfl({'pred_boxes':boxes,'pred_logits':z},t,indices,1)['loss_vfl'].backward()
    assert z.grad[0,1,1]!=0

def test_transient_DN_observation_output_state_rng_and_gradients_exact():
    model,_,_=build(cfg(),synthetic=True);model.train();images=torch.rand(1,3,128,128);t=targets()
    rng=torch.random.get_rng_state();before=tensor_digest(model.state_dict().items())
    a=model(images,t);rng_after=torch.random.get_rng_state();state_after=copy.deepcopy(model.state_dict())
    model.load_state_dict({k:v for k,v in state_after.items()}) # restore BN below from pre-run copy on independent model
    clone,_,_=build(cfg(),synthetic=True);clone.train();torch.random.set_rng_state(rng)
    with dn_capture(clone) as noise:b=clone(images,t)
    assert output_digest(a)==output_digest(b)
    assert tensor_digest(state_after.items())==tensor_digest(clone.state_dict().items())
    assert torch.equal(rng_after,torch.random.get_rng_state())
    assert noise['embedding_calls']==1 and noise['classes'] is not None and noise['boxes_unact'] is not None and noise['attention_mask'] is not None
    assert '_get_decoder_input' not in clone.decoder.__dict__

@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_complete_save_restore_same_next_window(dtype,tmp_path):
    conf=cfg(dtype);model,c,info=build(conf,synthetic=True)
    e=Engine(model,c,_optimizer(model,conf),CommonEMA(model),conf)
    b={'arm':ARM,'scope':'cpu_R18_synthetic','config':conf.binding(),'model':info,'source_sha256':'a'*64}
    torch.manual_seed(42);images=torch.rand(2,3,128,128);t=targets((2,0))
    first=e.train_window(images,t,capture=True);cp=checkpoint.save(e,b,tmp_path/'C1.pth')
    direct=e.train_window(images,t,capture=True);expected=checkpoint.capture(e,b)
    m2,c2,i2=build(conf,synthetic=True);clone=Engine(m2,c2,_optimizer(m2,conf),CommonEMA(m2),conf)
    restored=checkpoint.restore(clone,b,cp['path'],cp['sha256']);replay=clone.train_window(images,t,capture=True)
    assert restored['completed_updates']==1 and restored['full_state_exact']
    assert {k:v for k,v in direct.items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'}
    assert checkpoint.exact_equal(expected,checkpoint.capture(clone,b))
    assert all(len(row)==(21 if i==0 else 12) for i,row in enumerate(first['microbatch_losses']))
    assert all(p.grad is None for p in clone.model.parameters())
    assert first['update']==1 and first['optimizer_steps']==first['ema_updates']==1
    with pytest.raises(FileExistsError):checkpoint.save(clone,b,cp['path'])

def test_scope_binding_and_immutable_cache_refusals(tmp_path):
    conf=cfg();m,c,info=build(conf,synthetic=True);e=Engine(m,c,_optimizer(m,conf),CommonEMA(m),conf)
    b={'arm':ARM,'scope':'cpu_R18_synthetic','config':conf.binding(),'model':info}
    for bad in ({**b,'arm':'R0'},{**b,'scope':'gpu_R18_acceptance'},{**b,'config':{}}):
        with pytest.raises(ValueError):checkpoint.capture(e,bad)
    m.decoder.valid_mask[0,0]=~m.decoder.valid_mask[0,0]
    with pytest.raises(ValueError,match='anchor/cache'):checkpoint.capture(e,b)
    assert e.completed_updates==0

def test_actual_common_schedule_not_historical_typo():
    assert abs(learning_rate(1)-5e-8)<=math.ulp(5e-8) and learning_rate(2000)==1e-4 and learning_rate(36480)==1e-6
    assert Config().binding()['scheduler']=='update_warmup2000_cosine_to_1e-6'
    for u in (0,36481):
        with pytest.raises(ValueError):learning_rate(u)

def test_native_complete_network_loss_gradient_and_transient_guards_exact():
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    a,c,info=build(cfg(),synthetic=True);b,base,_,_,_=native_objects(cfg())
    from sparse_rtdetr.baseline.training_v2b import _configure_model_sampling
    _configure_model_sampling(b,cfg())
    assert tensor_digest(a.state_dict().items())==tensor_digest(b.state_dict().items())
    images=torch.rand(1,3,128,128);t=targets();rng=torch.random.get_rng_state()
    with dn_capture(a),forward_guards(a):out_a=a(images,t)
    loss_a=c(out_a,t);sum(loss_a.values()).backward()
    torch.random.set_rng_state(rng);out_b=b(images,t);loss_b=base(out_b,t);sum(loss_b.values()).backward()
    assert output_digest(out_a)==output_digest(out_b) and all(torch.equal(loss_a[k],loss_b[k]) for k in loss_a)
    assert tensor_digest((n,p.grad) for n,p in a.named_parameters())==tensor_digest((n,p.grad) for n,p in b.named_parameters())

def test_failure_capture_keeps_raw_nonfinite_scene_and_is_not_resumable(tmp_path):
    from sparse_rtdetr.protocol_r18.failure import preserve
    conf=cfg();m,c,info=build(conf,synthetic=True);e=Engine(m,c,_optimizer(m,conf),CommonEMA(m),conf)
    images=torch.rand(1,3,128,128);out=m(images,targets());sum(c(out,targets()).values()).backward()
    first=next(p for p in m.parameters() if p.grad is not None);first.grad.view(-1)[0]=float('nan');e.failed=True;e.phase=1
    before=tensor_digest((n,p.grad) for n,p in m.named_parameters())
    receipt=preserve({'engine':e,'input':{'images':images,'GT_ids':targets()[0]['gt_ids']},'binding':{'scope':'failure-only'}},tmp_path)
    raw=torch.load(receipt['file']['path'],map_location='cpu',weights_only=True)
    assert raw['resumable'] is False and raw['phase']==1 and raw['completed_updates']==0
    assert any(v is not None and not v.isfinite().all() for v in raw['raw_gradients'].values())
    assert before==tensor_digest((n,p.grad) for n,p in m.named_parameters())
    with pytest.raises(RuntimeError):e.state_dict()
