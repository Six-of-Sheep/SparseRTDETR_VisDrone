"""Bounded, transient observations. No GT, RNG, loss or model-state changes.

Only selected first-microbatch windows are observed. Full-query reference
counts and <=8 queries/group actual coordinates are distinct populations.
Counterfactuals remove the floor with the SAME learned offsets/value/weights;
these are mechanism observations, not another detector or quality result.
"""
import contextlib
import torch
from ..research.sampling_span import deformable_attention_core_func_v2 as core


def summary(t):
    x = t.detach().double().cpu().reshape(-1)
    if not len(x):
        return dict(count=0, mean=None, min=None, p50=None, p90=None, max=None)
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite sampling observation')
    q = torch.quantile(x, torch.tensor([.5, .9], dtype=torch.float64))
    return dict(count=len(x), mean=float(x.mean()), min=float(x.min()),
                p50=float(q[0]), p90=float(q[1]), max=float(x.max()))


def bilinear(loc, shape):
    # EXACT native normalized -> [-1,1] -> align_corners=False operation order.
    h, w = shape
    g = loc * 2 - 1
    xy = ((g + 1) * loc.new_tensor([w, h]) - 1) / 2
    base = xy.floor(); frac = xy-base
    x, y = frac.unbind(-1)
    weights = torch.stack(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y), -1)
    neighbors = base.unsqueeze(-2)+loc.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    valid = ((neighbors >= 0) & (neighbors < loc.new_tensor([w,h]))).all(-1)
    return base, weights * valid, xy


@contextlib.contextmanager
def observe(model, arm, rows, *, max_queries=8):
    handles=[]; originals=[]
    for layer_index, layer in enumerate(model.decoder.decoder.layers):
        m=layer.cross_attn; cache={'calls':0}; original=m.ms_deformable_attn_core
        def before(module,args,kwargs,cache=cache):
            query=args[0] if args else kwargs['query']
            ref=args[1] if len(args)>1 else kwargs['reference_points']
            if cache['calls']:
                return
            with torch.no_grad():
                cache['ref']=ref.detach()
                cache['dtype']=query.dtype
        def offset_output(module,args,result,cache=cache,m=m):
            if not cache['calls']:
                # Observe the actual projection; no extra no_grad Linear in
                # autocast (which could contaminate its cast-weight cache).
                cache['offset']=result.detach().reshape(*cache['ref'].shape[:2],m.num_heads,12,2)
        def wrapped(value,shapes,locations,weights,points,*,cache=cache,old=original,m=m,idx=layer_index):
            output=old(value,shapes,locations,weights,points)
            cache['calls']+=1
            if cache['calls']>1:
                return output
            with torch.no_grad():
                ref=cache['ref']; q=ref.shape[1]
                if q<300:
                    raise ValueError('observation requires known native ordinary-query inventory')
                dn=q-300; scale=m.num_points_scale.to(dtype=cache['dtype']).unsqueeze(-1)
                baseline=ref[:,:,None,:,:2]+cache['offset']*scale*ref[:,:,None,:,2:]*m.offset_scale
                for group,start,stop in (('DN',0,dn),('ordinary',dn,q)):
                    r=ref[:,start:stop,0,2:]; total=r.shape[0]*r.shape[1]
                    threshold=r.new_tensor([1/shapes[0][1],1/shapes[0][0]])
                    triggered=r<threshold
                    chosen=slice(start,min(stop,start+max_queries))
                    loc=locations[:,chosen]; base=baseline[:,chosen]; n=points[0]
                    a,wa,cells=bilinear(loc[:,:,:,:n],shapes[0])
                    b,wb,_=bilinear(base[:,:,:,:n],shapes[0])
                    span=cells.amax(-2)-cells.amin(-2) if stop>start else cells
                    if stop>start:
                        # Same autocast scope as actual core. Detached; no backward.
                        counter=core(value.detach(),shapes,base,weights[:,chosen].detach(),points)
                        delta=output[:,chosen].detach()-counter
                    else:
                        delta=output[:,chosen].detach()
                    rows.append(dict(layer=idx,arm=arm,group=group,queries_per_image=stop-start,
                        reference_instances=total,reference_below_cell_x=int(triggered[...,0].sum()),
                        reference_below_cell_y=int(triggered[...,1].sum()),
                        reference_below_cell_any=int(triggered.any(-1).sum()),
                        reference_floor_trigger_fraction_x=float(triggered[...,0].float().mean()) if total else None,
                        reference_floor_trigger_fraction_y=float(triggered[...,1].float().mean()) if total else None,
                        reference_floor_trigger_fraction_any=float(triggered.any(-1).float().mean()) if total else None,
                        DN_includes_native_padding=group=='DN',
                        floor_applied_instances=int(triggered.any(-1).sum()) if arm=='P' else 0,
                        reference_extent_cells=summary(r*threshold.reciprocal()),
                        sampled_queries_per_image=min(stop-start,max_queries),
                        actual_location_normalized_x=summary(loc[:,:,:,:n,0]),
                        actual_location_normalized_y=summary(loc[:,:,:,:n,1]),
                        actual_offset_cells=summary((loc[:,:,:,:n]-ref[:,chosen,None,:,:2])*threshold.reciprocal()),
                        actual_span_cells_x=summary(span[...,0]),actual_span_cells_y=summary(span[...,1]),outside_image_fraction=float(((loc[:,:,:,:n]<0)|(loc[:,:,:,:n]>1)).any(-1).float().mean()) if stop>start else None,
                        same_stencil_fraction=float((a==b).all(-1).float().mean()) if stop>start else None,
                        bilinear_effective_weight_abs_change=summary((wa-wb).abs()),
                        attention_aggregate_output_abs_change=summary(delta.abs()),
                        counterfactual='same learned offsets, value and attention weights; no reference floor',
                        uses_GT=False))
            return output
        handles.append(m.register_forward_pre_hook(before,with_kwargs=True))
        handles.append(m.sampling_offsets.register_forward_hook(offset_output))
        originals.append((m,original));m.ms_deformable_attn_core=wrapped
    try:
        yield rows
    finally:
        for h in handles:h.remove()
        for m,original in originals:m.ms_deformable_attn_core=original
