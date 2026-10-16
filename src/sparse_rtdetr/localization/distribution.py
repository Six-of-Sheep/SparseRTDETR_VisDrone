"""Torch localization geometry and log-probability operations.

Support geometry and target encoding are FP64; softmax/log/KL are FP32.
No localization quality estimator or trainable moving-support arm is here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .operators import NumericalIntegrityError, weighting_function


def require_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise NumericalIntegrityError("nonfinite " + label)


def validate_reference(reference, valid_mask=None):
    require_finite(reference, "reference")
    checked = reference if valid_mask is None else reference[valid_mask]
    if reference.shape[-1] != 4 or bool((checked[..., 2:] <= 0).any()):
        raise NumericalIntegrityError("reference has nonpositive width/height")


def support_torch(reference, *, up=.5, reg_scale=4., ascending=False, valid_mask=None):
    validate_reference(reference, valid_mask)
    r = reference.double()
    w = torch.as_tensor(weighting_function(up=up, reg_scale=reg_scale), device=r.device)
    sign = r.new_tensor([-1., -1., 1., 1.])
    x = r[..., [0, 1, 0, 1], None] + sign[:, None] * r[..., [2, 3, 2, 3], None] * (.5 + w / reg_scale)
    if ascending:
        x = torch.cat((x[..., :2, :].flip(-1), x[..., 2:, :]), dim=-2)
    require_finite(x, "physical support")
    gap = x.diff(dim=-1).abs()
    tolerance = 64 * torch.finfo(torch.float64).eps * x.abs().amax(-1, keepdim=True).clamp_min(1.)
    unresolved = (gap <= tolerance).any(dim=(-1,-2))
    if bool((unresolved if valid_mask is None else unresolved[valid_mask]).any()):
        raise NumericalIntegrityError("unresolvable physical support")
    require_finite(x, "physical support")
    from .consumer_geometry import tensor
    tensor("support/atoms",x,False)
    return x


def edge_order_torch(p):
    return torch.cat((p[..., :2, :].flip(-1), p[..., 2:, :]), dim=-2)


def logits_probabilities(logits, temperature=1.):
    z = logits.float().reshape(*logits.shape[:-1], 4, 33) if logits.shape[-1] == 132 else logits.float()
    require_finite(z, "effective logits")
    return F.softmax(z / temperature, dim=-1)


def residual_from_prior(prior, residual):
    q = prior.detach().float()
    require_finite(q, "prior")
    require_finite(residual, "residual")
    if bool((q <= 0).any()) or bool(((q.double().sum(-1) - 1.).abs() > 3e-6).any()):
        raise NumericalIntegrityError("invalid FP32 prior for residual update")
    full = torch.log(q) + residual.float()
    return full, F.softmax(full, dim=-1)


def corners_to_box(corners):
    low, high = corners[..., :2], corners[..., 2:]
    return torch.cat(((low + high) / 2., high - low), -1)


def box_to_corners(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2., box[..., :2] + box[..., 2:] / 2.), -1)


NUMERICAL_IMPLEMENTATION = "fp32_softmax_fp64_mass_centered_v1"


def centered_box_from_probabilities(probabilities, reference, valid_mask=None, *, decoded_role=None):
    """Differentiable FP64 mass-normalized relative displacement; no clipping."""
    # Keep the original reference and physical-support resolution admission.
    support_torch(reference, valid_mask=valid_mask)
    require_finite(probabilities, "decode probabilities")
    if probabilities.shape != (*reference.shape[:-1], 4, 33) or bool((probabilities < 0).any()):
        raise NumericalIntegrityError("invalid decode probability layout or sign")
    p64 = probabilities.double()
    mass = p64.sum(-1)
    require_finite(mass, "decode probability mass")
    if bool((mass <= 0).any()):
        raise NumericalIntegrityError("nonpositive decode probability mass")
    # Both numerator and denominator retain their autograd paths.
    atoms = torch.as_tensor(weighting_function(), device=p64.device, dtype=torch.float64)
    relative = (p64 * (.5 + atoms / 4.)).sum(-1) / mass
    r = reference.double()
    signs = r.new_tensor([-1., -1., 1., 1.])
    delta = signs * r[..., [2, 3, 2, 3]] * relative
    box64 = torch.cat((r[..., :2] + (delta[..., :2] + delta[..., 2:]) / 2.,
                       delta[..., 2:] - delta[..., :2]), -1)
    from .consumer_geometry import positive_box
    from .terminal_geometry import decoded_positive_mask
    positive_mask = decoded_positive_mask(reference, valid_mask, decoded_role)
    positive_box("decode/box64",box64,positive_mask)
    box = box64.float()
    positive_box("decode/box32",box,positive_mask)
    require_finite(box, "decoded box")
    checked = box if positive_mask is None else box[positive_mask]
    if bool((checked[..., 2:] <= 0).any()):
        raise NumericalIntegrityError("decoded nonpositive or unresolved box")
    return box


def decode(logits, reference, valid_mask=None, *, decoded_role=None):
    # Softmax stays FP32 exactly as in the frozen training recipe.
    probabilities = logits_probabilities(logits)
    try:
        return centered_box_from_probabilities(probabilities, reference, valid_mask, decoded_role=decoded_role)
    except NumericalIntegrityError as error:
        if str(error) == "decoded nonpositive or unresolved box":
            from .lab_geometry import centered_box_evidence
            try:
                diagnostic = centered_box_evidence(logits, reference, valid_mask, probabilities)
            except Exception as capture_error:
                diagnostic = {"diagnostic_capture_error": repr(capture_error)}
            raise NumericalIntegrityError("decoded nonpositive or unresolved box", **diagnostic) from error
        raise


@torch.no_grad()
def encode_target(reference, target_boxes):
    """Two physical atoms, returned in original W-bin order; strict GT tail."""
    x = support_torch(reference, ascending=True)
    values = box_to_corners(target_boxes.double())
    clipped = values.clamp(min=x[..., 0], max=x[..., -1])
    index = torch.searchsorted(x.contiguous(), clipped.unsqueeze(-1).contiguous(), right=True).squeeze(-1) - 1
    index = index.clamp(0, 31)
    lo = x.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    hi = x.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
    right_weight = (clipped - lo) / (hi - lo)
    left_weight = 1. - right_weight
    from .consumer_geometry import tensor
    tensor("fgl/interpolation_gap",hi-lo,False)
    if bool(((hi-lo)<=0).any()): raise NumericalIntegrityError("nonpositive FGL interpolation gap")
    tensor("fgl/right_weight",right_weight,False);tensor("fgl/left_weight",left_weight,False)
    left_index = torch.cat((32 - index[..., :2], index[..., 2:]), -1)
    right_index = torch.cat((31 - index[..., :2], index[..., 2:] + 1), -1)
    tail = (values < x[..., 0]) | (values > x[..., -1])
    return left_index, right_index, left_weight.float(), right_weight.float(), tail


def fgl(logits, reference, targets, iou_weights, normalizer):
    if logits.numel() == 0:
        return logits.sum() * 0., 0
    lo, hi, lw, rw, tail = encode_target(reference, targets)
    logp = F.log_softmax(logits.float().reshape(-1, 4, 33), dim=-1)
    from .consumer_geometry import tensor
    tensor("fgl/logits",logits);tensor("fgl/logp",logp);tensor("fgl/iou_weight",iou_weights,False)
    ce = -(logp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * lw +
           logp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * rw)
    return (ce * iou_weights.detach()[:, None]).sum() / normalizer, int(tail.sum())
