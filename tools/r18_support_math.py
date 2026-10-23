"""Endpoint-only support accounting; no detector, data, loss or optimizer imports."""
import contextlib
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stencil(loc, shape):
    """Native zeros/align_corners=False operation order, retaining invalid mass."""
    h, w = shape
    require(torch.isfinite(loc).all(), 'nonfinite coordinates')
    grid = loc * 2 - 1
    xy = ((grid + 1) * loc.new_tensor([w, h]) - 1) / 2
    base = xy.floor()
    x, y = xy.unbind(-1)
    x0, y0 = base.unbind(-1)
    x1, y1 = x0 + 1, y0 + 1
    weight = torch.stack(((x1-x)*(y1-y), (x-x0)*(y1-y),
                          (x1-x)*(y-y0), (x-x0)*(y-y0)), -1)
    neighbors = base.unsqueeze(-2) + loc.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    valid = ((neighbors >= 0) & (neighbors < loc.new_tensor([w,h]))).all(-1)
    return neighbors, weight, valid


def support(loc, attention, shape, gt_xyxy):
    """One query, all heads/finest points. Pool masses BEFORE division.

    I/O/V are fractions of all finest-level attention mass A, including zero
    padding; M=I/(I+O) conditions on valid support only. No denominator epsilon.
    GT corners are normalized, clipped training-valid boxes, half open.
    """
    neighbors, weight, valid = stencil(loc, shape)
    require(loc.ndim == 3 and attention.shape == loc.shape[:-1], 'support shape')
    require(torch.isfinite(attention).all() and (attention >= 0).all(), 'attention')
    centers = (neighbors + .5) / loc.new_tensor([shape[1], shape[0]])
    g = torch.as_tensor(gt_xyxy, dtype=loc.dtype, device=loc.device)
    inside = valid & (centers >= g[:2]).all(-1) & (centers < g[2:]).all(-1)
    weighted = attention.double().unsqueeze(-1) * weight.double()
    a = float(attention.double().sum())
    i = float((weighted * inside).sum())
    o = float((weighted * (valid & ~inside)).sum())
    v = float((weighted * ~valid).sum())
    require(abs(i+o+v-a) <= 2e-6*max(1., a), 'support mass conservation')
    z = i+o
    return dict(A=a, Z=z, I=None if a == 0 else i/a,
                O=None if a == 0 else o/a, V=None if a == 0 else v/a,
                M=None if z == 0 else i/z)


def delta(left, right):
    return None if left is None or right is None else left-right


def effective_change(left, right, shape):
    """Compare coefficients on the UNION of lattice sites, separately per point.

    Moving an entirely invalid stencil does not change effective interpolation.
    """
    ln,lw,lv=stencil(left,shape);rn,rw,rv=stencil(right,shape)
    sites=torch.cat((ln,rn),-2)
    weights=torch.cat((lw*lv,-rw*rv),-1)
    same=(sites.unsqueeze(-2)==sites.unsqueeze(-3)).all(-1)
    coefficients=(same*weights.unsqueeze(-2)).sum(-1)
    return bool((coefficients!=0).any())


def coordinates(reference, offsets, shape, scale, offset_scale=.5):
    """Same association as native attention, with ORIGINAL unexpanded ref."""
    require(reference.shape[-2:] == (1,4), 'reference shape')
    return reference[:,:,None,:,:2] + offsets*scale*reference[:,:,None,:,2:]*offset_scale


@contextlib.contextmanager
def capture(model, arm, buffer):
    """Observe every ordinary query at all three layers; local P core only.

    GT is absent during forward. Hooks detach actual intermediates, never change
    the returned output or recompute a projection/decoder/model. Eval has no DN.
    """
    handles, originals = [], []
    require(arm in ('P','B0') and not model.training, 'eval P/B0 only')
    for index, layer in enumerate(model.decoder.decoder.layers):
        m = layer.cross_attn
        cache = {}
        old = m.ms_deformable_attn_core

        def before(module, args, kwargs, cache=cache):
            require(not cache, 'attention called twice')
            query = args[0] if args else kwargs['query']
            ref = args[1] if len(args)>1 else kwargs['reference_points']
            require(ref.shape[1] == 300 and not torch.is_grad_enabled(), 'ordinary eval inventory')
            cache.update(ref=ref.detach(), dtype=query.dtype)

        def offset(module, args, result, cache=cache, m=m):
            cache['offset'] = result.detach().reshape(*cache['ref'].shape[:2], m.num_heads, 12, 2)

        def wrapped(value, shapes, loc, weights, points, *, cache=cache, old=old, m=m, idx=index):
            result = old(value, shapes, loc, weights, points)
            require(points == [4,4,4] and len(shapes) == 3, 'geometry changed')
            ref = cache['ref']
            base = coordinates(ref, cache['offset'], shapes[0],
                               m.num_points_scale.to(dtype=cache['dtype']).unsqueeze(-1), m.offset_scale)
            trigger = (ref[:,:,0,2:] < ref.new_tensor([1/shapes[0][1],1/shapes[0][0]])).any(-1)
            require(torch.equal(loc[~trigger], base[~trigger]), 'nontrigger changed')
            if arm == 'P':
                counter = old(value.detach(), shapes, base, weights.detach(), points)
                change = (result.detach()-counter).abs().mean(-1)
            else:
                require(torch.equal(loc, base), 'B0 nonnative locations')
                change = torch.zeros_like(trigger, dtype=torch.float32)
            require(torch.isfinite(result).all() and torch.isfinite(change).all(), 'nonfinite local output')
            buffer.append(dict(layer=idx, shape=list(shapes[0]), ref=ref[:,:,0].cpu(),
                               loc=loc[:,:,:,:4].detach().cpu(), base=base[:,:,:,:4].cpu(),
                               weights=weights[:,:,:,:4].detach().cpu(),
                               trigger=trigger.cpu(), output_abs_mean=change.cpu()))
            return result

        handles += [m.register_forward_pre_hook(before, with_kwargs=True),
                    m.sampling_offsets.register_forward_hook(offset)]
        originals.append((m,old))
        m.ms_deformable_attn_core = wrapped
    try:
        yield
        require(len(buffer) == 3 and [r['layer'] for r in buffer] == [0,1,2], 'layer coverage')
    finally:
        for h in handles:
            h.remove()
        for m, old in originals:
            m.ms_deformable_attn_core = old
