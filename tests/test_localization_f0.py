"""New/affected F0 interfaces only; synthetic inputs, no development images."""
from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
import torch.nn.functional as F

from sparse_rtdetr.localization.model import F0Config, F0EMA, build_f0
from sparse_rtdetr.localization.distribution import (box_to_corners, corners_to_box, decode,
    edge_order_torch, encode_target, fgl, logits_probabilities, support_torch)
from sparse_rtdetr.localization.criterion import F0Criterion, go_union, ddf_loss
from sparse_rtdetr.localization.ignore import disjoint_union, negative_keep_mask
from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
from sparse_rtdetr.localization.engine import F0Engine, learning_rate
from sparse_rtdetr.localization import checkpoint
from sparse_rtdetr.localization.trace import decision, lexicographic_assignment, official_spatial_filter


@pytest.fixture(scope="module",autouse=True)
def cpu_policy():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def target(n=1, ident="g"):
    return {"boxes":torch.tensor([[.5,.5,.15,.1]]*n,dtype=torch.float32).reshape(-1,4),
            "labels":torch.tensor([i%10 for i in range(n)],dtype=torch.long),
            "gt_ids":[f"{ident}{i}" for i in range(n)]}


def make_components(amp_dtype="float32"):
    from sparse_rtdetr.baseline.training_v2b import _optimizer
    cfg=F0Config(input_hw=(128,192),physical_batch_size=1,amp_dtype=amp_dtype)
    model,criterion,binding=build_f0(cfg,synthetic=True)
    optimizer=_optimizer(model,cfg); ema=F0EMA(model)
    return F0Engine(model,criterion,optimizer,ema,cfg), binding


def test_distribution_geometry_target_and_fp32():
    ref=torch.tensor([[.5,.4,.1,.2]],requires_grad=True)
    z=torch.zeros(1,132,requires_grad=True)
    boxes=decode(z,ref.detach())
    torch.testing.assert_close(boxes,ref,atol=2e-7,rtol=0)
    x=support_torch(ref.detach(),ascending=True)
    assert x.dtype==torch.float64
    assert logits_probabilities(z).dtype==torch.float32
    gt=ref.detach().clone()
    left,right,lw,rw,tail=encode_target(ref.detach(),gt)
    assert not tail.any() and torch.all(lw+rw==1)
    physical=(lw.double()*x.gather(-1, torch.cat((32-left[:,:2],left[:,2:]),-1).unsqueeze(-1)).squeeze(-1)+
              rw.double()*x.gather(-1, torch.cat((32-right[:,:2],right[:,2:]),-1).unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(physical,box_to_corners(gt.double()),atol=1e-8,rtol=0)
    value,_=fgl(z,ref.detach(),gt,torch.ones(1),1)
    torch.testing.assert_close(value,torch.tensor(4*math.log(33)),atol=2e-6,rtol=0)
    weights=torch.arange(1,5,dtype=torch.float32)
    (boxes*weights).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert ref.grad is None
    # Strict physical endpoint classification, distinct from bin index clamping.
    inside=box_to_corners(gt.double()); outside=inside.clone(); outside[:,0]=x[:,0,0]-1e-5
    *_,over=encode_target(ref.detach(),corners_to_box(outside))
    assert over.tolist()==[[True,False,False,False]]


def test_continuous_union_and_negative_mask():
    regions=np.array([[0,0,.3,1],[.2,0,.4,1]])
    union=disjoint_union(regions)
    assert np.sum((union[:,2:]-union[:,:2]).prod(-1))==pytest.approx(.4)
    pred=torch.tensor([[[.5,.5,1.,1.],[.2,.5,.4,1.],[.5,.5,2.,1.]]])
    t={"ignore_union":torch.tensor(union),"ignore_class_boxes":torch.tensor([[0.,0.,.5,1.]],dtype=torch.float64),
       "ignore_class_labels":torch.tensor([2])}
    idx=[(torch.tensor([1]),torch.tensor([0]))]
    keep=negative_keep_mask(pred,idx,[t])
    assert keep[0,0].sum()==9 and not keep[0,0,2] # union .4; one class exact .5
    assert keep[0,1].all() # a matched query retains its entire classification vector
    assert keep[0,2].all() # full off-canvas width2 denominator, not clipped width1
    invalid=copy.deepcopy(t); invalid["ignore_union"][0,0]=float("nan")
    with pytest.raises(RuntimeError): negative_keep_mask(pred,idx,[invalid])
    inverted=torch.tensor([[[.5,.5,-1.,-1.]]])
    with pytest.raises(ValueError): negative_keep_mask(inverted,[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))],[t])


def test_raw_rows_category_zero_priority_and_augmentation():
    raw=(b"20,20,20,20,1,1,0,0\n"*2+b"10,10,50,50,0,0,0,0\n"+
         b"25,25,5,5,0,2,0,0\n"+b"0,0,50,50,1,11,0,0\n")
    parsed=parse_raw(raw,"fixture")
    assert len(parsed["gt"])==2 and parsed["gt"][0]["id"]!=parsed["gt"][1]["id"]
    assert parsed["counts"]=={"valid_gt":2,"spatial_ignore":1,"class_ignore":1,"category11_not_spatial":1}
    img=Image.fromarray(np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3))
    aug=RawAugmentation((128,128))
    rng=torch.get_rng_state().clone()
    one=aug(img.copy(),parsed,epoch=1,seed=0,position=0,image_id="fixture")
    two=aug(img.copy(),parsed,epoch=1,seed=0,position=0,image_id="fixture")
    assert torch.equal(one[0],two[0]) and one[2]==two[2] and torch.equal(rng,torch.get_rng_state())
    last=aug(img.copy(),parsed,epoch=118,seed=0,position=0,image_id="fixture")
    assert not last[2]["augmentation_strong"]
    assert all(s["operation"]=="horizontal_flip" for s in last[2]["geometry_steps"])
    assert len(last[1]["labels"])==2 # valid GT within category0 still trains
    assert len(last[1]["ignore_ids"])==2
    assert all(s in last[1]["gt_ids"] for s in [r["id"] for r in parsed["gt"]])


def test_augment_matches_original_gt_path():
    from sparse_rtdetr.baseline.training_v2b_data import (_transform_config,_vendor_tools,TrainCoreDataConfig,_sample_rng,_sample_seed)
    root=Path(__file__).resolve().parents[1]
    cfg=TrainCoreDataConfig(input_size=128)
    converter,transforms,to_tv,_=_vendor_tools(root,_transform_config(root,cfg))
    img=Image.fromarray(np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3))
    raw=parse_raw(b"20,20,20,20,1,1,0,0\n60,60,30,20,1,2,0,0\n10,10,50,50,0,0,0,0\n","fixture")
    aug=RawAugmentation((128,128))
    class Dummy:
        epoch=0
    for position in (0,7,19):
        ours=aug(img.copy(),raw,epoch=1,seed=0,position=position,image_id="fixture")
        b=torch.tensor([r["box"] for r in raw["gt"]],dtype=torch.float32)
        old_target={"boxes":to_tv(b,key="boxes",box_format="XYXY",spatial_size=(128,128)),"labels":torch.tensor([0,1])}
        with _sample_rng(_sample_seed(0,1,position,"fixture")):
            old_img,old_t,_=transforms(img.copy(),old_target,Dummy())
        assert torch.equal(ours[0],old_img.as_subclass(torch.Tensor))
        assert torch.equal(ours[1]["boxes"],old_t["boxes"].as_subclass(torch.Tensor))
        assert torch.equal(ours[1]["labels"],old_t["labels"])


def test_go_vote_tie_and_family_normalizers():
    ts=[target(2)]; ts[0]["gt_ids"]=["z","a"]
    def v(q,g):return [(torch.tensor(q,dtype=torch.long),torch.tensor(g,dtype=torch.long))]
    votes=[v([0],[0]),v([0],[1]),v([1],[0]),v([2],[1]),v([2],[1])]
    go=go_union(votes,ts)
    assert go[0][0].tolist()==[0,1,2] and go[0][1].tolist()==[1,0,1]
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.i=0
        def forward(self,*args):
            r=votes[self.i];self.i+=1;return {"indices":r}
    def head():return {"pred_boxes":torch.tensor([[[.4,.5,.1,.1],[.5,.5,.1,.1],[.6,.5,.1,.1]]]),
                      "pred_logits":torch.zeros(1,3,10),"pred_corners":torch.zeros(1,3,132),
                      "ref_points":torch.tensor([[[.5,.5,.15,.1]]*3])}
    output=head(); auxiliaries=[head(),head()]
    for h in auxiliaries:h.update(teacher_corners=output["pred_corners"],teacher_logits=output["pred_logits"],teacher_ref_points=output["ref_points"])
    output["aux_outputs"]=auxiliaries
    output["pre_outputs"]={k:v for k,v in head().items() if k in ("pred_boxes","pred_logits")}
    output["enc_aux_outputs"]=[copy.deepcopy(output["pre_outputs"])]
    crit=F0Criterion(Fixed(),alpha=.75)
    losses=crit(output,ts)
    assert len(losses)==20
    diag=crit.last_diagnostics
    assert diag["classification_denominator"]==2 and diag["regression_denominator"]==3
    expected=(output["pred_boxes"][0]-ts[0]["boxes"][[1,0,1]]).abs().sum()/3*5
    assert torch.equal(losses["loss_bbox_enc_0"],expected)


def test_ddf_teacher_detach_and_normal_coefficients():
    g=target(); ref=torch.tensor([[[.5,.5,.15,.1],[.5,.5,.15,.1]]])
    student=torch.arange(264,dtype=torch.float32).reshape(1,2,132)/100;student.requires_grad_()
    teacher=torch.flip(student.detach(),[-1]).requires_grad_()
    teacher_logits=torch.full((1,2,10),-2.,requires_grad=True)
    output={"pred_boxes":ref.clone(),"pred_corners":student,"ref_points":ref,
            "teacher_ref_points":ref,"teacher_corners":teacher,"teacher_logits":teacher_logits}
    indices=[(torch.tensor([0]),torch.tensor([0]))]; coefficients=(2.,7.)
    actual=ddf_loss(output,[g],indices,coefficients)
    q=F.softmax(teacher.detach().reshape(1,2,4,33)/5,-1)
    kl=(q*(q.log()-F.log_softmax(student.reshape(1,2,4,33)/5,-1))).sum(-1)*25
    expected=(2*kl[0,0].mean()+7*torch.sigmoid(torch.tensor(-2.))*kl[0,1].mean())/9
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    actual.backward()
    assert teacher.grad is None and teacher_logits.grad is None and student.grad.abs().sum()>0


def test_model_dn_empty_gradient_and_trace():
    engine,binding=make_components(); model=engine.model
    assert binding["parameter_delta"]==231300
    assert not any("lqe" in n.lower() or "gate" in n.lower() for n,_ in model.decoder.named_modules())
    images=torch.rand(2,3,128,192,generator=torch.Generator().manual_seed(23))
    # Nonzero deterministic regression residual; initialization alone can mask detach errors.
    with torch.no_grad():
        for h in model.decoder.dec_bbox_head:h.layers[-1].weight.normal_(0,.0001)
    output=model(images,targets=[target(1),target(0)])
    assert len(output["aux_outputs"])==2 and len(output["dn_outputs"])==3
    assert output["pred_logits"].shape==(2,300,10)
    assert output["dn_outputs"][0]["query_valid"].sum().item()==200
    assert not output["dn_outputs"][0]["query_valid"][1].any()
    for head in output["aux_outputs"]:
        assert torch.equal(output["ref_points"],head["ref_points"])
    losses=engine.criterion(output,[target(1),target(0)])
    assert len(losses)==38 and all(torch.isfinite(x) for x in losses.values())
    weights=torch.tensor([1.,2.,3.,4.])
    (output["pred_boxes"]*weights).sum().backward()
    assert model.decoder.dec_bbox_head[0].layers[-1].weight.grad.abs().sum()>0
    assert model.decoder.pre_bbox_head.layers[-1].weight.grad is None
    model.zero_grad(set_to_none=True)
    empty=model(images,targets=[target(0),target(0)])
    assert "dn_outputs" not in empty
    zero_losses=engine.criterion(empty,[target(0),target(0)])
    assert len(zero_losses)==20 and all(torch.isfinite(x) for x in zero_losses.values())
    # Trace is eval-only, no DN/BN/RNG mutation and exactly the same final detections.
    model.eval(); rng=torch.get_rng_state().clone(); state=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        ordinary=model(images)
        model.decoder.trace_enabled=True; traced=model(images); model.decoder.trace_enabled=False
    assert len(traced["trace_outputs"])==3 and "dn_outputs" not in traced
    assert torch.equal(ordinary["pred_boxes"],traced["pred_boxes"])
    assert torch.equal(ordinary["pred_logits"],traced["pred_logits"])
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    vd=__import__("src.zoo.rtdetr.rtdetrv2_decoder",fromlist=["get_contrastive_denoising_training_group"])
    many=target(101)
    _,_,_,metadata=vd.get_contrastive_denoising_training_group([many,target(0)],10,300,model.decoder.denoising_class_embed,num_denoising=100)
    assert metadata["dn_num_group"]==1 and metadata["dn_num_split"]==[202,300]


def test_full_checkpoint_and_same_next_window(tmp_path):
    engine,b=make_components()
    binding={"scope":"cpu_synthetic","run_id":"unit_fixture","config":engine.config.binding(),"model":b}
    images=torch.rand(2,3,128,192,generator=torch.Generator().manual_seed(23)); ts=[target(1),target(3)]
    engine.train_window(images,ts,capture=True)
    receipt=checkpoint.save(engine,binding,tmp_path/"C1.pth")
    expected=engine.train_window(images,ts,capture=True)
    expected_state=checkpoint.capture(engine,binding)
    clone,_=make_components()
    restored=checkpoint.restore(clone,binding,tmp_path/"C1.pth",receipt["sha256"])
    assert restored["full_state_exact"]
    actual=clone.train_window(images,ts,capture=True)
    for key in set(expected)-{"elapsed_seconds"}: assert actual[key]==expected[key],key
    assert checkpoint.exact_equal(expected_state,checkpoint.capture(clone,binding))
    clone.model.encoder.pos_embed2[0,0,0]+=1
    with pytest.raises(ValueError,match="position cache"): checkpoint.behavior_identity(clone)
    with pytest.raises(ValueError,match="wrong checkpoint"): checkpoint.read(tmp_path/"C1.pth",receipt["sha256"],{**binding,"scope":"formal_F0_seed0_e120"})


def test_schedule():
    # Analytic 5e-8 is unchanged; one FP64 operation-order rounding ulp is allowed.
    assert abs(learning_rate(1)-5e-8)<=math.ulp(5e-8)
    assert learning_rate(2000)==1e-4
    assert learning_rate(36480)==1e-6
    assert learning_rate(2001)<1e-4
    assert learning_rate(36000)>1e-6
    with pytest.raises(ValueError):learning_rate(36481)


def test_equal_microbatch_gradient_oracle():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.2,.4]))
            self.decoder=torch.nn.Module();self.decoder.trace_enabled=False
        def forward(self,images,targets=None):
            return (images.mean((2,3))[:,:2]*self.weight).sum(-1)
    class Loss(torch.nn.Module):
        def forward(self,output,targets):
            self.last_diagnostics={"gt_count":sum(len(t["labels"]) for t in targets)}
            return {"loss":(output-self.last_diagnostics["gt_count"]).square().mean()}
    model=Toy(); optimizer=torch.optim.SGD(model.parameters(),lr=1e-4); observed=[]
    optimizer.register_step_pre_hook(lambda *args:observed.append(model.weight.grad.detach().clone()))
    cfg=F0Config(input_hw=(128,128),physical_batch_size=1,amp_dtype="float32")
    engine=F0Engine(model,Loss(),optimizer,F0EMA(model),cfg)
    images=torch.zeros(2,3,128,128)
    images[0,0]=.25;images[0,1]=.5;images[1,0]=.75;images[1,1]=.1
    w=model.weight.detach().clone();a=torch.tensor([.25,.5]);b=torch.tensor([.75,.1])
    analytic=(2*a*(a@w-1)+2*b*(b@w-3))/2
    result=engine.train_window(images,[target(1),target(3)])
    norm=torch.linalg.vector_norm(analytic)
    expected=analytic*min(1.,float(.1/(norm+1e-6)))
    torch.testing.assert_close(observed[0],expected,atol=1e-7,rtol=1e-6)
    assert result["gradient_norm_before_clip"]==pytest.approx(float(norm),rel=1e-6)
    assert result["optimizer_steps"]==1 and engine.ema.updates==1


def test_matching_lexicographic_and_stop_gates():
    for weights in (np.array([[.5,.5],[.5,.5]]),np.array([[.9,.8],[.8,.7]]),np.array([[0.,0.],[.4,.5],[.5,.4]])):
        ids=[f"g{i}" for i in range(len(weights))]
        actual=lexicographic_assignment(weights,ids)
        choices=[];n,q=weights.shape
        for plan in itertools.product(range(q+1),repeat=n):
            real=[x for x in plan if x<q]
            if len(real)!=len(set(real)) or any(c<q and weights[r,c]<=0 for r,c in enumerate(plan)):continue
            score=math.fsum(weights[r,c] for r,c in enumerate(plan) if c<q)
            choices.append((-score,plan))
        best=min(choices)[1]
        assert actual.tolist()==[c if c<q else -1 for c in best]
    parsed=parse_raw(b"1,1,10,10,0,0,0,0\n2,2,2,2,1,1,0,0\n20,20,2,2,1,1,0,0\n50,50,2,2,0,0,0,0\n","fixture")
    gt,frac,integ=official_spatial_filter(parsed,32,32)
    assert len(gt)==1 and gt[0]["physical_line"]==3
    assert integ[-1,-1]==121 # off-canvas category0 contributes no boundary pixel
    stop=decision([{"E":[],"E_clip":[],"events":[]}])
    assert stop["recommendation"]=="DO_NOT_INVEST_CURRENT_VERSION" and stop["sample_evidence"]=="INSUFFICIENT"


def test_role_roots_rejected_before_payload(tmp_path, monkeypatch):
    from sparse_rtdetr.localization.data import RawRoleDataset, validate_role_roots
    allowed=tmp_path/"train_core"
    images,annotations=allowed/"images",allowed/"annotations"
    images.mkdir(parents=True);annotations.mkdir()
    bound={"role":"train_core","images":str(images),"annotations":str(annotations)}
    assert validate_role_roots("train_core",images,annotations,bound)==(images,annotations)
    opened=[]
    def forbidden_read(self): opened.append(str(self));raise AssertionError("payload must not be read")
    monkeypatch.setattr(Path,"read_bytes",forbidden_read)
    for bad in (tmp_path/"holdout"/"train_core"/"images",tmp_path/"development"/"images",tmp_path/"elsewhere"/"train_core"/"images"):
        with pytest.raises(ValueError):
            RawRoleDataset(role="train_core",manifest_path="must_not_read",images_path=bad,
                           annotations_path=annotations,root_binding=bound)
    assert not opened


def test_ignore_same_class_is_not_union_and_crop_intersection(monkeypatch):
    from torchvision.transforms.v2 import functional as TF
    pred=torch.tensor([[[.5,.5,1.,1.]]]);empty=torch.empty(0,dtype=torch.long)
    t={"ignore_class_boxes":torch.tensor([[0.,0.,.3,1.],[.3,0.,.6,1.]],dtype=torch.float64),
       "ignore_class_labels":torch.tensor([2,2])}
    assert negative_keep_mask(pred,[(empty,empty)],[t]).all()
    raw=parse_raw(b"45,30,20,20,1,1,0,0\n0,30,70,20,0,0,0,0\n","geometry_fixture")
    aug=RawAugmentation((128,128))
    class Geometry:
        actual_params=None
        def __init__(self,operation):self.operation=operation
        def __call__(self,image,target):
            if self.operation=="zoom":
                self.actual_params={"padding":[10,20,10,20]}
                image=TF.pad(image,[10,20,10,20]);target["boxes"]=TF.pad(target["boxes"],[10,20,10,20])
            else:
                self.actual_params={"top":40,"left":50,"height":60,"width":60}
                image=TF.crop(image,40,50,60,60);target["boxes"]=TF.crop(target["boxes"],40,50,60,60)
            return image,target
    aug.zoom=Geometry("zoom");aug.crop=Geometry("crop");aug.photo=lambda *args:args
    monkeypatch.setattr(torch,"rand",lambda *args,**kwargs:torch.tensor([0.]))
    pixels,t,trace=aug(Image.new("RGB",(100,100)),raw,epoch=1,seed=0,position=0,image_id="geometry_fixture")
    # Original ignore center x=35 maps to x=-5 after crop, but [0,30] intersects.
    # After horizontal flip and resize it is [64,64/3,128,64].
    torch.testing.assert_close(t["ignore_spatial_boxes"],torch.tensor([[.5,1/6,1.,.5]],dtype=torch.float64),atol=1e-12,rtol=0)
    assert [s["operation"] for s in trace["geometry_steps"]]==["zoom","crop","horizontal_flip"]
    only=parse_raw(b"0,0,40,40,0,0,0,0\n","only_ignore")
    _,t,_=RawAugmentation((128,128))(Image.new("RGB",(100,100)),only,epoch=118,seed=0,position=0,image_id="only_ignore")
    assert len(t["labels"])==0 and len(t["ignore_spatial_boxes"])==1


def test_bf16_core_dtypes_and_full_restore(tmp_path):
    engine,b=make_components("bfloat16")
    images=torch.rand(2,3,128,192,generator=torch.Generator().manual_seed(24));ts=[target(1),target(0)]
    with torch.autocast("cpu",dtype=torch.bfloat16):
        output=engine.model(images,targets=ts)
    for h in [output,*output["aux_outputs"],*output["dn_outputs"]]:
        assert h["pred_corners"].dtype==torch.float32
        assert logits_probabilities(h["pred_corners"]).dtype==torch.float32
        assert support_torch(h["ref_points"],valid_mask=h["query_valid"]).dtype==torch.float64
    losses=engine.criterion(output,ts)
    assert len(losses)==38 and all(v.dtype==torch.float32 and torch.isfinite(v) for v in losses.values())
    del output,losses
    binding={"scope":"cpu_synthetic","run_id":"bf16_unit_fixture","config":engine.config.binding(),"model":b}
    engine.train_window(images,ts,capture=True)
    receipt=checkpoint.save(engine,binding,tmp_path/"bf16-C1.pth")
    expected=engine.train_window(images,ts,capture=True);state=checkpoint.capture(engine,binding)
    clone,_=make_components("bfloat16")
    assert checkpoint.restore(clone,binding,tmp_path/"bf16-C1.pth",receipt["sha256"])["full_state_exact"]
    actual=clone.train_window(images,ts,capture=True)
    assert {k:v for k,v in actual.items() if k!="elapsed_seconds"}=={k:v for k,v in expected.items() if k!="elapsed_seconds"}
    assert checkpoint.exact_equal(state,checkpoint.capture(clone,binding))


def test_trace_cache_shape_identity_and_complete_set(tmp_path):
    from sparse_rtdetr.localization.trace_contract import (SCHEMA,validate_cache,validate_collection_identities,save_cache,read_cache)
    row={"stable_image_id":"fixture","image_sha256":"a"*64,"annotation_sha256":"b"*64,
         "height":100,"width":120,"raw_annotation_rows":1}
    z=torch.zeros(3,300,4,33);ref=torch.tensor([.5,.5,.1,.2]).expand(3,300,4).clone()
    cache={"schema":SCHEMA,"arm":"F0","seed":0,"epoch":120,"ema":True,"image_id":"fixture",
           "image_sha256":"a"*64,"annotation_sha256":"b"*64,"original_hw":[100,120],"input_hw":[768,1344],
           "checkpoint_sha256":"c"*64,"selection_sha256":"d"*64,"effective_logits":z.numpy(),
           "references":ref.numpy(),"boxes":decode(z,ref).numpy(),"class_logits":np.zeros((3,300,10),np.float32),
           "probabilities":logits_probabilities(z).numpy(),"teacher_probabilities":logits_probabilities(z[-1],5.).numpy()}
    kwargs={"checkpoint_sha256":"c"*64,"selection_sha256":"d"*64}
    raw=parse_raw(b"1,1,10,10,1,1,0,0\n","fixture")
    assert validate_cache(cache,row,raw_target=raw,**kwargs)["raw_identity"]
    path=tmp_path/"one.npz";save_cache(path,cache)
    assert validate_cache(read_cache(path),row,**kwargs)["decoded_geometry"]
    for name,bad in (("effective_logits",np.full((3,300,4,33),np.nan,np.float32)),
                     ("class_logits",np.zeros((3,299,10),np.float32)),("image_id","other"),
                     ("boxes",cache["boxes"]+np.float32(.01))):
        with pytest.raises(ValueError):validate_cache({**cache,name:bad},row,**kwargs)
    with pytest.raises(ValueError):validate_cache(cache,row,raw_target=parse_raw(b"1,1,10,10,1,1,0,0\n","other"),**kwargs)
    selected=[{"stable_image_id":f"id{i}"} for i in range(64)]
    metadata=[{"image_id":r["stable_image_id"],**kwargs} for r in selected]
    assert validate_collection_identities(metadata,selected,**kwargs)
    for bad in (metadata[:63],[metadata[0]]*64,metadata[::-1]):
        with pytest.raises(ValueError):validate_collection_identities(bad,selected,**kwargs)


def test_checkpoint_scope_metadata_and_failure_quarantine(tmp_path,monkeypatch):
    engine,b=make_components()
    binding={"scope":"cpu_synthetic","run_id":"negative_fixture","config":engine.config.binding(),"model":b}
    incomplete=tmp_path/"incomplete.pth.complete.json";incomplete.write_text('preserved sidecar')
    with pytest.raises(FileExistsError):checkpoint.save(engine,binding,tmp_path/"incomplete.pth")
    assert incomplete.read_text()=='preserved sidecar' and not (tmp_path/"incomplete.pth").exists()
    for invalid in ({**binding,"scope":"formal_F0_seed0_e120"},
                    {**binding,"config":{**binding["config"],"seed":1}}):
        with pytest.raises(ValueError):checkpoint.save(engine,invalid,tmp_path/"forbidden.pth")
        assert not (tmp_path/"forbidden.pth").exists()
    payload=checkpoint.capture(engine,binding)
    def wrong_modes(p):p["model_training"].pop("")
    def wrong_ema(p):p["ema"]["decay"]=.99
    def wrong_rng(p):p["rng"].pop("numpy")
    def wrong_support(p):p["model"]["decoder.support_parameters"][1]=.6
    for mutate in (wrong_modes,wrong_ema,wrong_rng,wrong_support):
        clone,_=make_components();before=copy.deepcopy(clone.model.state_dict())
        damaged=copy.deepcopy(payload);mutate(damaged)
        with monkeypatch.context() as context:
            context.setattr(checkpoint,"read",lambda *args:damaged)
            with pytest.raises((ValueError,RuntimeError)):checkpoint.restore(clone,binding,"mock_path","mock_sha")
        assert clone.failed and checkpoint.exact_equal(before,clone.model.state_dict())
        with pytest.raises(RuntimeError,match="cannot enter"):clone.train_window(None,None)
    clone,_=make_components()
    with monkeypatch.context() as context:
        context.setattr(checkpoint,"read",lambda *args:copy.deepcopy(payload))
        context.setattr(clone.model,"load_state_dict",lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError("injected load failure")))
        with pytest.raises(RuntimeError,match="injected"):checkpoint.restore(clone,binding,"mock_path","mock_sha")
    assert clone.failed
