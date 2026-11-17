"""R28 budgeted separable resampling: demand map, closed-form allocation, monotone maps, tent resampler, demand net.

Definition: CANDIDATE_R28_001.md section 2, with TASK_R28_CODEX_AMENDMENT_002.md section 2 (maps extend linearly
outside [0, 1] with the end slopes; nothing here clips a box edge).  Allocation and map arithmetic are FP64; the
resampled image is FP32.  Every function is batched over images and runs on the device of its inputs.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

H_HW = (1152, 2016)
L_HW = (768, 1344)
GRID = (12, 21)
DEMAND_HW = (384, 672)
EPS = 0.2
LO, HI = 2.0 / 3.0, 1.5
S_LO, S_HI = 32.0, 64.0
TAPS = 4
CONST_RTOL = 1e-12
DEMAND_CHANNELS = (16, 32, 64, 96, 128)
F64 = torch.float64
DEFINITION = {
    "name": "r28_budgeted_separable_resampling_v1", "H_HW": list(H_HW), "L_HW": list(L_HW), "grid": list(GRID),
    "demand_input_hw": list(DEMAND_HW), "eps": EPS, "slope_bounds": [LO, HI], "omega_px": [S_LO, S_HI],
    "blur": "3x3 mean, replicate border", "taps_per_axis": TAPS,
    "allocation": "per axis: marginal of D+eps, linear interpolation to the fine grid (constant beyond outer cell "
                  "centres), m=clip(d/lambda,lo,hi) with mean(m)=1 solved exactly on the sorted breakpoints",
    "maps": "piecewise linear on fine knots; linear extension outside [0,1] with the end slopes; no clipping",
    "resampler": "per output pixel a tent of half-width max(1, footprint/1.5) source pixels centred at the inverse "
                 "map of the pixel centre, normalised weights, replicate border, horizontal then vertical, FP32",
    "demand_net": "5x[conv3x3 s2, BN, ReLU] channels 16/32/64/96/128, conv3x3(128)+ReLU, conv1x1, softplus",
}


# ---------------------------------------------------------------- demand from boxes (labels only)

def blur3(d):
    """3x3 mean with replicated border over the last two dims."""
    shape = d.shape
    p = F.pad(d.reshape(-1, 1, shape[-2], shape[-1]), (1, 1, 1, 1), mode="replicate")
    return F.avg_pool2d(p, 3, 1).reshape(shape)


def demand_from_boxes(xyxy):
    """Normalised xyxy boxes (N, 4) of one image -> blurred demand map (GH, GW), FP64, without eps."""
    gh, gw = GRID
    b = xyxy.to(F64).reshape(-1, 4)
    d = torch.zeros(gh * gw, dtype=F64, device=b.device)
    if b.shape[0]:
        w = (b[:, 2] - b[:, 0]).clamp_min(0) * L_HW[1]
        h = (b[:, 3] - b[:, 1]).clamp_min(0) * L_HW[0]
        omega = ((S_HI - torch.sqrt(w * h)) / (S_HI - S_LO)).clamp(0, 1)
        ci = ((b[:, 1] + b[:, 3]) / 2 * gh).floor().long().clamp(0, gh - 1)
        cj = ((b[:, 0] + b[:, 2]) / 2 * gw).floor().long().clamp(0, gw - 1)
        d.index_add_(0, ci * gw + cj, omega)
    return blur3(d.view(gh, gw))


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


# ---------------------------------------------------------------- allocation

def interp_marginal(marg, fine):
    """(B, n) coarse marginal -> (B, fine): linear between coarse cell centres, constant beyond the outer centres."""
    n = marg.shape[-1]
    pos = (torch.arange(fine, dtype=F64, device=marg.device) + 0.5) / fine * n - 0.5
    i0 = pos.floor().clamp(0, n - 1).long()
    i1 = (i0 + 1).clamp(max=n - 1)
    fr = (pos - i0).clamp(0, 1)
    a, b = marg[:, i0], marg[:, i1]
    return a + fr * (b - a)


def waterfill(d, lo=LO, hi=HI):
    """Exact solution of m = clip(d / lam, lo, hi), mean(m) = 1 for positive d (B, n), FP64.

    Returns (m, constant) where constant marks rows whose d is constant (m is exactly 1 there).
    The multiplier is found on the sorted breakpoints d/hi and d/lo: between two neighbours the clipped sets do not
    change, so lam is the closed form active_sum / (n - n_lo*lo - n_hi*hi).
    """
    d = d.to(F64)
    B, n = d.shape
    dmax, dmin = d.max(1).values, d.min(1).values
    const = (dmax - dmin) <= CONST_RTOL * dmax.clamp_min(1e-30)
    s = d.sort(1).values
    P = F.pad(s.cumsum(1), (1, 0))

    def parts(lam):
        n_lo = torch.searchsorted(s, (lam * lo).contiguous(), right=True)
        n_hi = n - torch.searchsorted(s, (lam * hi).contiguous(), right=False)
        act = P.gather(1, n - n_hi) - P.gather(1, n_lo)
        return n_lo.to(F64), n_hi.to(F64), act   # counts in FP64: an integer tensor times a float is FP32

    cand = torch.cat([s / hi, s / lo], 1).sort(1).values
    n_lo, n_hi, act = parts(cand)
    g = (n_lo * lo + n_hi * hi + act / cand) / n
    j = (g > 1.0).sum(1, keepdim=True).clamp(1, 2 * n - 1)
    mid = torch.sqrt(cand.gather(1, j - 1) * cand.gather(1, j))
    n_lo, n_hi, act = parts(mid)
    den = n - n_lo * lo - n_hi * hi
    ok = (act > 0) & (den > 0)
    lam = torch.where(ok, act / torch.where(ok, den, torch.ones_like(den)), mid)
    m = (d / lam).clamp(lo, hi)
    m = torch.where(const[:, None], torch.ones_like(m), m)
    return m, const


def knots(m):
    k = F.pad(m.cumsum(1) / m.shape[1], (1, 0))
    k[:, -1] = 1.0
    return k


@dataclass
class Plan:
    kx: torch.Tensor          # (B, LW + 1) forward-map knots, horizontal
    ky: torch.Tensor          # (B, LH + 1)
    mx: torch.Tensor          # (B, LW) slopes
    my: torch.Tensor          # (B, LH)
    identity: torch.Tensor    # (B,) bool: both axes exactly uniform

    def select(self, index):
        return Plan(self.kx[index], self.ky[index], self.mx[index], self.my[index], self.identity[index])

    def to(self, device):
        return Plan(self.kx.to(device), self.ky.to(device), self.mx.to(device), self.my.to(device),
                    self.identity.to(device))


def plan_from_demand(demand):
    """(B, GH, GW) demand without eps -> Plan.  A constant demand map gives the identity plan exactly."""
    d = demand.detach().to(F64) + EPS
    my, cy = waterfill(interp_marginal(d.sum(2), L_HW[0]))
    mx, cx = waterfill(interp_marginal(d.sum(1), L_HW[1]))
    return Plan(knots(mx), knots(my), mx, my, cx & cy)


# ---------------------------------------------------------------- maps (differentiable in the coordinate)

def fwd(k, u):
    """Forward map source -> warped.  k (B, n+1), u (B, M) normalised; linear extension outside [0, 1]."""
    n = k.shape[-1] - 1
    t = u.to(F64) * n
    i = t.detach().floor().clamp(0, n - 1).long()
    k0, k1 = k.gather(1, i), k.gather(1, i + 1)
    return k0 + (t - i) * (k1 - k0)


def inv(k, v):
    """Inverse map warped -> source.  k (B, n+1), v (B, M) normalised; linear extension outside [0, 1]."""
    n = k.shape[-1] - 1
    v = v.to(F64)
    i = (torch.searchsorted(k, v.detach().contiguous(), right=True) - 1).clamp(0, n - 1)
    k0, k1 = k.gather(1, i), k.gather(1, i + 1)
    return (i + (v - k0) / (k1 - k0)) / n


def map_boxes(plan, boxes, inverse):
    """cxcywh boxes (B, M, 4) -> cxcywh boxes in the other frame, edge by edge, in the dtype of `boxes`.

    inverse=False: source frame -> warped canvas (targets for the denoising queries).
    inverse=True:  warped canvas -> source frame (every predicted box family at the output boundary).
    """
    f = inv if inverse else fwd
    b = cxcywh_to_xyxy(boxes.to(F64))
    x1, y1, x2, y2 = b.unbind(-1)
    out = torch.stack([f(plan.kx, x1), f(plan.ky, y1), f(plan.kx, x2), f(plan.ky, y2)], -1)
    return xyxy_to_cxcywh(out).to(boxes.dtype)


def magnification(plan, xyxy):
    """Per-box (ox, oy) of normalised xyxy boxes (B, M, 4) with positive width and height."""
    b = xyxy.to(F64)
    x1, y1, x2, y2 = b.unbind(-1)
    ox = (fwd(plan.kx, x2) - fwd(plan.kx, x1)) / (x2 - x1)
    oy = (fwd(plan.ky, y2) - fwd(plan.ky, y1)) / (y2 - y1)
    return ox, oy


# ---------------------------------------------------------------- resampler

def taps(k, n_out, n_src):
    """Source indices (B, n_out, TAPS), FP32 weights, tap span actually needed and footprint (source pixels)."""
    B = k.shape[0]
    j = torch.arange(n_out + 1, dtype=F64, device=k.device)
    edge = inv(k, (j / n_out).expand(B, -1))
    centre = inv(k, ((j[:-1] + 0.5) / n_out).expand(B, -1))
    p = n_src * centre - 0.5
    d = n_src * (edge[:, 1:] - edge[:, :-1])
    h = (d / 1.5).clamp_min(1.0)
    base = (p - h).floor() + 1
    idx = base[..., None] + torch.arange(TAPS, dtype=F64, device=k.device)
    w = (1.0 - (idx - p[..., None]).abs() / h[..., None]).clamp_min(0)
    span = ((p + h).floor() - base + 1).max()
    w = w / w.sum(-1, keepdim=True)
    return idx.long().clamp(0, n_src - 1), w.to(torch.float32), span, d


def resample(images, plan, out_hw=L_HW):
    """(B, C, Hs, Ws) -> (B, C, LH, LW) FP32 under the plan; horizontal pass then vertical pass."""
    x = images.to(torch.float32)
    B, C, Hs, Ws = x.shape
    LH, LW = out_hw
    ix, wx, sx, _ = taps(plan.kx, LW, Ws)
    iy, wy, sy, _ = taps(plan.ky, LH, Hs)
    if max(float(sx), float(sy)) > TAPS:
        raise RuntimeError("R28 resampler needs more than %d taps: %s %s" % (TAPS, float(sx), float(sy)))
    t = None
    for a in range(TAPS):
        g = x.gather(3, ix[:, None, None, :, a].expand(B, C, Hs, LW)) * wx[:, None, None, :, a]
        t = g if t is None else t + g
    out = None
    for a in range(TAPS):
        g = t.gather(2, iy[:, None, :, None, a].expand(B, C, LH, LW)) * wy[:, None, :, None, a]
        out = g if out is None else out + g
    return out


# ---------------------------------------------------------------- demand network

class DemandNet(nn.Module):
    """Image at DEMAND_HW -> non-negative demand map at GRID.  About 0.34 M parameters."""

    def __init__(self):
        super().__init__()
        layers, cin = [], 3
        for c in DEMAND_CHANNELS:
            layers += [nn.Conv2d(cin, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv2d(cin, cin, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(cin, 1, 1))

    def forward(self, x):
        return F.softplus(self.head(self.features(x))).squeeze(1)


def demand_input(images):
    return F.interpolate(images.to(torch.float32), size=DEMAND_HW, mode="bilinear", align_corners=False,
                         antialias=True)


def demand_loss(pred, target):
    return F.smooth_l1_loss(torch.log1p(pred), torch.log1p(target.to(pred.dtype)))
