"""Constructed tensor study; no detector, weights, data reader, or CUDA execution.

Read and authenticate three source files before extracting one pure function and
loading the standalone interpolation core. This is a CPU formula study, not a
production entry, model diagnostic, GPU check, or scientific evaluator.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


DT = torch.float64
SHAPES = [(4, 8), (2, 4), (1, 2)]


def tensor(x, *, grad=False):
    return torch.tensor(x, dtype=DT, device="cpu", requires_grad=grad)


def equal(a, b, atol=1e-10):
    torch.testing.assert_close(a, b, rtol=1e-9, atol=atol)


def scalar(x):
    return float(x.detach())


def authenticate(root, index):
    sources = {}
    for e in index["frozen_sources"]:
        p = root / e["path"]
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            raise ValueError("frozen source SHA mismatch: " + e["path"])
        sources[e["role"]] = (p, raw.decode("utf-8"))
    # Extract only the already authenticated pure extent adapter, never import
    # sampling_span.py (which would import detector classes/registries).
    adapter = ast.parse(sources["adapter"][1])
    fn = next(n for n in adapter.body if isinstance(n, ast.FunctionDef)
              and n.name == "sampling_references")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 str(sources["adapter"][0]), "exec"), namespace)
    spec = importlib.util.spec_from_file_location("frozen_bilinear_leaf", sources["core"][0])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return namespace["sampling_references"], core, sources


def locations(refs, offsets):
    # Native multiplication order; float64 is used only for this CPU study.
    scales = offsets.new_full((12, 1), 0.25)
    delta = offsets * scales * refs[:, :, None, :, 2:] * 0.5
    return refs[:, :, None, :, :2] + delta


def sample(core, values, xy):
    return core.bilinear_grid_sample(values, (2 * xy - 1).reshape(1, 1, -1, 2))


def feature_xy(uv):
    return (uv + tensor([0.5, 0.5])) / tensor([8, 4])


def manual_bilinear(values, u, v):
    total = 0.0
    for x in (math.floor(u), math.floor(u) + 1):
        for y in (math.floor(v), math.floor(v) + 1):
            if 0 <= x < 8 and 0 <= y < 4:
                total += float(values[0, 0, y, x]) * (1-abs(u-x)) * (1-abs(v-y))
    return total


class OffsetGradientScale(torch.autograd.Function):
    """Proposed diagnostic ONLY: bit-identical identity forward, proxy backward."""

    @staticmethod
    def forward(ctx, offsets, factor):
        ctx.save_for_backward(factor)
        return offsets.clone()

    @staticmethod
    def backward(ctx, output_gradient):
        (factor,) = ctx.saved_tensors
        return output_gradient * factor, None


def run(adapter, core, sources):
    cases = []

    def record(name, **values):
        cases.append({"case": name, "status": "PASS", **values})

    # Five rectangles span below, at, above, and asymmetric threshold cases.
    ext = tensor([[.5/8, .5/4], [1/8, 1/4], [1.5/8, 1.5/4],
                  [.2/8, 2/4], [2/8, .2/4]])
    centers = tensor([.48, .43]).expand(5, 2)
    refs = torch.cat([centers, ext], -1).reshape(1, 5, 1, 4)
    off = torch.linspace(-.8, .9, 120, dtype=DT).reshape(1, 5, 1, 12, 2).requires_grad_()
    expanded = adapter(refs, SHAPES)
    p = locations(expanded, off)
    b = locations(refs, off)
    expected = expanded[:, :, None, :, 2:] / 8
    derivative = torch.autograd.grad(p.sum(), off, retain_graph=True)[0]
    equal(derivative, expected)
    cell_derivative = derivative * tensor([8, 4])
    assert bool((cell_derivative[..., :4, :] >= 1/8).all())
    equal(p[..., 4:, :], b[..., 4:, :])
    equal(p[:, 1:3], b[:, 1:3])
    equal(derivative[..., 0], expected[..., 0])
    record("thresholds_asymmetric_extents_and_offset_Jacobian",
           finest_min_cell_Jacobian=scalar(cell_derivative[..., :4, :].min()),
           extent_cases=ext.tolist())

    rho = expanded[..., 2:] / refs[..., 2:]
    rewritten = locations(refs, off * rho[:, :, None])
    equal(rewritten, p)
    record("pointwise_native_offset_rescaling", max_coordinate_error=scalar((rewritten-p).abs().max()),
           condition="positive reference extents; same center, point order and frozen n=4")

    # Real-valued continuity across the max kink, not a production ref gradient.
    eps = 1e-7
    triplet = []
    for w in (1/8-eps, 1/8, 1/8+eps):
        r = tensor([[[[.48, .43, w, .1]]]])
        triplet.append(locations(adapter(r, SHAPES), torch.ones((1, 1, 1, 12, 2), dtype=DT)))
    assert scalar((triplet[1]-triplet[0])[..., :4, :].abs().max()) == 0
    equal((triplet[2]-triplet[1])[..., :4, 0], torch.full((1,1,1,4), eps/8, dtype=DT))
    record("extent_threshold_continuity", note="No derivative through a detached production reference is claimed.")

    z = torch.zeros_like(off, requires_grad=True)
    zero_locations = locations(expanded, z)
    span = (zero_locations[..., :4, :].amax(-2)-zero_locations[..., :4, :].amin(-2)) * tensor([8,4])
    equal(span, torch.zeros_like(span))
    record("zero_offset_counterexample", actual_span_cells=span[0,0,0].tolist(),
           positive_coordinate_Jacobian=True)

    # Native and frozen gather interpolation, input gradients and grid gradients.
    values = torch.arange(32, dtype=DT).reshape(1,1,4,8).requires_grad_()
    uv = tensor([[2.25,1.375], [-.5,1.25], [7.5,2.25], [-1.5,1.25],
                 [2,1], [-1,1.25], [7,3]], grad=True)
    xy = feature_xy(uv)
    ours = sample(core, values, xy)
    native = F.grid_sample(values, (2*xy-1).reshape(1,1,-1,2),
                           mode="bilinear", padding_mode="zeros", align_corners=False)
    equal(ours, native)
    wanted = tensor([manual_bilinear(values.detach(), *pair) for pair in uv.detach().tolist()])
    equal(ours.flatten(), wanted)
    ours_grad = torch.autograd.grad(ours.sum(), (values, uv), retain_graph=True)
    native_grad = torch.autograd.grad(native.sum(), (values, uv))
    for a,bv in zip(ours_grad,native_grad): equal(a,bv)
    record("bilinear_padding_boundary_and_knot_convention", output=ours.detach().flatten().tolist(),
           note="At integer/support knots compare implemented autograd conventions, not a unique mathematical derivative.")

    # Finite differences deliberately avoid interpolation/zero-support knots.
    eps=1e-6
    fd=[]
    for i in range(4):
        row=[]
        for j in range(2):
            hi=uv.detach().clone();lo=hi.clone();hi[i,j]+=eps;lo[i,j]-=eps
            row.append(scalar((sample(core,values.detach(),feature_xy(hi)).sum()
                               -sample(core,values.detach(),feature_xy(lo)).sum())/(2*eps)))
        fd.append(row)
    equal(tensor(fd), ours_grad[1][:4], atol=1e-7)
    record("finite_difference_spatial_gradient_away_from_knots", gradients=fd)

    small_ref=tensor([[[[.4,.4,.5/8,.5/4]]]])
    d=tensor([[[[[.8,.8]]*12]]], grad=True)
    native_xy=locations(small_ref,d)[0,0,0,:4]
    p_xy=locations(adapter(small_ref,SHAPES),d)[0,0,0,:4]
    st0=torch.floor(native_xy*tensor([8,4])-.5)
    st1=torch.floor(p_xy*tensor([8,4])-.5)
    assert torch.equal(st0,st1)
    out0=sample(core,values.detach(),native_xy);out1=sample(core,values.detach(),p_xy)
    assert scalar((out1-out0).abs().max())>0
    record("same_stencil_different_weights_and_output", output_change=scalar((out1-out0).abs().mean()))

    constant = torch.full((1,1,4,8), 3., dtype=DT)
    const_out = sample(core,constant,p_xy)
    const_grad = torch.autograd.grad(const_out.sum(), d, retain_graph=True)[0]
    equal(const_out,torch.full_like(const_out,3.));equal(const_grad,torch.zeros_like(d))
    record("constant_interior_loss_gradient_zero", offset_gradient_max=scalar(const_grad.abs().max()),
           note="Nonzero coordinate Jacobian does not imply nonzero feature derivative.")

    # P can move a point OUT of support even as its coordinate Jacobian grows.
    edge_ref=tensor([[[[.01,.4,.05/8,.5/4]]]])
    edge_d=tensor([[[[[-8.,0.]]*12]]],grad=True)
    edge_b=sample(core,constant,locations(edge_ref,edge_d)[0,0,0,:4]).sum()
    edge_p=sample(core,constant,locations(adapter(edge_ref,SHAPES),edge_d)[0,0,0,:4]).sum()
    gb=torch.autograd.grad(edge_b,edge_d,retain_graph=True)[0]
    gp=torch.autograd.grad(edge_p,edge_d)[0]
    assert scalar(gb.abs().max())>0;equal(gp,torch.zeros_like(gp))
    record("larger_Jacobian_but_smaller_loss_gradient", native_gradient_max=scalar(gb.abs().max()),
           P_gradient_max=scalar(gp.abs().max()), reason="P moves constructed points beyond zero-padding support.")

    varying=sample(core,values.detach(),p_xy)
    stationary=(varying-varying.detach()).square().sum()
    gs=torch.autograd.grad(stationary,d)[0];equal(gs,torch.zeros_like(gs))
    record("stationary_loss_counterexample", gradient_max=scalar(gs.abs().max()))

    theta=tensor(0.,grad=True)
    parabolic=(torch.arange(8,dtype=DT)-3).square().expand(4,8).reshape(1,1,4,8)
    uv_pair=tensor([[2.25,1.25],[3.75,1.25]])+torch.stack([theta/8,theta*0])
    pair=sample(core,parabolic,feature_xy(uv_pair)).flatten()
    g0=torch.autograd.grad(pair[0],theta,retain_graph=True)[0]
    g1=torch.autograd.grad(pair[1],theta,retain_graph=True)[0]
    gsum=torch.autograd.grad(pair.sum(),theta)[0]
    assert scalar(g0)!=0 and scalar(g1)!=0;equal(gsum,tensor(0.))
    record("shared_parameter_gradient_cancellation", individual=[scalar(g0),scalar(g1)], total=scalar(gsum))

    # A toy affine head cannot absorb a general reference-dependent ratio.
    x=tensor([.5,1.,2.]);target=torch.maximum(x,torch.ones_like(x))/x
    slopes=(target[1:]-target[:-1])/(x[1:]-x[:-1])
    assert scalar(slopes[0])!=scalar(slopes[1])
    record("shared_affine_reparameterization_not_guaranteed", inputs=x.tolist(),
           desired_offsets=target.tolist(), adjacent_slopes=slopes.tolist(),
           note="Restricted-head counterexample, not a proof that complete detector function classes differ.")

    # Matched forward coordinates at constant rho do not match an SGD step.
    dp=tensor(.8,grad=True);dn=tensor(1.6,grad=True);r=.05;e=.1;eta=.2;target=.4
    lp=.5*(e*dp/8-target)**2;ln=.5*(r*dn/8-target)**2
    equal(e*dp/8,r*dn/8)
    gdp=torch.autograd.grad(lp,dp)[0];gdn=torch.autograd.grad(ln,dn)[0]
    difference=(dn-eta*gdn)-2*(dp-eta*gdp)
    assert abs(scalar(difference))>1e-8
    record("same_coordinates_not_same_training_step", same_lr_mapped_step_error=scalar(difference),
           note="Constructed scalar SGD arithmetic; no optimizer training or detector backward was run.")

    # Explicitly emulate the two audited graph patterns using only scalars.
    raw=tensor(-1.,grad=True);offset=tensor(.7,grad=True)
    detached=raw.sigmoid().detach()
    sampling=offset*torch.maximum(detached,tensor(.5))/8
    through_ref=torch.autograd.grad(sampling,(raw,offset),allow_unused=True)
    assert through_ref[0] is None and through_ref[1] is not None
    previous_box=raw.sigmoid();box_delta=tensor(.1,grad=True)
    aux_output=(box_delta+torch.logit(previous_box)).sigmoid()
    aux_grad=torch.autograd.grad(aux_output,raw)[0];assert scalar(aux_grad)!=0
    record("detached_sampling_separate_undetached_box_path", sampling_previous_ref_gradient=None,
           aux_previous_ref_gradient=scalar(aux_grad), note="Scalar emulation of source patterns, not a decoder execution.")

    # One proposed future diagnostic, explicitly not added to production.
    d1=tensor([[[[[.8,.8]]*12]]],grad=True);d2=d1.detach().clone().requires_grad_()
    ep=adapter(small_ref,SHAPES);factor=(small_ref[...,2:]/ep[...,2:])[:, :, None]
    x1=locations(ep,d1);x2=locations(ep,OffsetGradientScale.apply(d2,factor))
    assert torch.equal(x1,x2)
    y1=sample(core,values.detach(),x1[0,0,0,:4]);y2=sample(core,values.detach(),x2[0,0,0,:4])
    assert torch.equal(y1,y2)
    gg1=torch.autograd.grad(y1.sum(),d1)[0];gg2=torch.autograd.grad(y2.sum(),d2)[0]
    equal(gg2,gg1*factor)
    equal(torch.autograd.grad(locations(ep,OffsetGradientScale.apply(d2,factor)).sum(),d2)[0],
          small_ref[:, :, None, :, 2:].expand_as(d2)/8)
    record("proposed_P_J0_proxy_only", forward_bit_equal_in_this_FP64_example=True,
           note="Backward restores native reference factor at P locations, not B0 full gradients; not a true forward derivative and not production code.")

    for wh in ((0.,.1),(-.1,.1),(float('nan'),.1)):
        try:adapter(tensor([[[[.4,.4,*wh]]]]),SHAPES)
        except ValueError:pass
        else:raise AssertionError("invalid extent was accepted")
    record("original_extent_domain_rejections", invalid_cases=3)
    assert not torch.cuda.is_initialized()
    assert not any(k.startswith(('sparse_rtdetr','src.zoo')) for k in sys.modules)
    return {"kind":"CONSTRUCTED_CPU_MECHANISM_STUDY_NOT_MODEL_DIAGNOSIS", "status":"PASS",
            "torch":torch.__version__, "dtype":"float64", "device":"cpu",
            "CUDA_initialized":torch.cuda.is_initialized(),"detector_imported":False,
            "checkpoints_or_real_data_read":False,"cases":cases,"skipped":0,
            "frozen_source_roles":list(sources),
            "limitations":["No BF16/GPU/model/production acceptance claim.",
                           "Finite differences avoid mathematical knots; proxy backward is tested against its declared rule, not called an exact derivative.",
                           "No experiment quality or loss-gradient evidence from a trained detector."]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('new output path required; no overwrite/retry')
    torch.set_num_threads(1)
    index_path=Path(__file__).resolve().parents[1]/'docs/research/r18_sampling_mechanism_20261021/SOURCES.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    try:
        adapter,core,sources=authenticate(args.frozen_root,index)
        report=run(adapter,core,sources)
    except Exception as exc:
        import traceback
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            json.dump({'status':'FAILED_CPU_STUDY','kind':'CONSTRUCTED_ONLY',
                       'error_type':type(exc).__name__,'error':str(exc),
                       'traceback':traceback.format_exc()},f,indent=2)
            f.write('\n')
        raise
    report['source_index_sha256']=hashlib.sha256(index_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'checks':len(report['cases']),
                      'skipped':0,'CUDA_initialized':False,'output':str(args.output)}))


if __name__=='__main__':main()
