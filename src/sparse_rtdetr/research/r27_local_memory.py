"""R27: fixed-budget local high-scale memory read by the global Q300 decoder (TASK_R27_001).

The detector receives the S15 H tensor I_H (1152x2016, augmented, [0,1]).  The global semantic path runs at
L = 768x1344 on ``I_L = interpolate(I_H, (768,1344), bilinear, align_corners=False, antialias=False)`` (FP32, on
the model device, inside the model).  H is split into 3x3 non-overlapping cores of 384x672 (row-major ids 0..8,
left-closed / right-open, the right/bottom image edge belongs to the last core).  Each image selects exactly four
cores; each selected core plus a 32 px halo (448x736, zero outside the image) runs an independent shallow
PResNet18 (conv1 + max_pool + res_layers[0:2], stride 8, 128 ch) and a 1x1 Conv+BN projection to 256 ch.  The
three native decoder layers keep self-attention, global cross-attention, FFN, heads and iterative refinement; each
layer adds a local-read residual to the global cross-attention output before the native residual/norm2/FFN.

Routing (the only C/P difference)
  P  the native L encoder top300 (DN excluded): p_i = max_c sigmoid(logit_ic), centre of the selected native box;
     a_j = sum of p_i whose centre lies in core j (one-hot product + sum, no atomics); four largest a_j, ties by the
     smaller id.  Everything detached / no_grad; no routing loss.
  C  independent uniform 4-of-9: tiles ordered by SHA256 of the UTF-8 string
     ``r27-v1|seed0|<scope>|<image_key>|<epoch>|<occurrence>|<tile_id>`` (ascending hex, then tile id), first four.
     Fields must not contain '|' or control characters (refused, never escaped); integers are plain decimal.
Selected tiles are sorted by id and packed in that order.

Local read (per layer, own parameters, d=256, 8 heads x 4 points, one stride-8 level)
  query  = native target_after_norm1 + query_pos_embed;  b = the layer's native detached reference (cx,cy,w,h)
  core j = core of (cx,cy); no local output unless j is one of the four selected cores; only j's memory is read.
  p_global = (cx,cy) + offset(query) * 1/4 * (w,h) * 0.5          (native v2 four-dim reference formula)
  p_patch  = (p_global * (2016,1152) - (core_x-32, core_y-32)) / (736,448)
  A point is valid iff it lies in the current patch [0,736)x[0,448) and in the real image [0,2016)x[0,1152).
  Invalid points are removed before the softmax; a head with no valid point contributes 0; a query whose centre
  is outside the selected cores, or none of whose 32 points is valid, has a local output of exactly 0 (the mask
  is applied after output_proj, so its bias cannot leak).  Coordinates, softmax and sampling are FP32; sampling
  is ``training_v2b_deterministic_sampling.bilinear_grid_sample`` (bilinear / zeros / align_corners=False).
  value_proj, the P3 rows of sampling_offsets/attention_weights are copied from the layer's native cross-attention;
  output_proj weight and bias are zero.  No gate, dropout or extra loss.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "R27-budgeted-local-memory-v1"
ARMS = ("C", "P")
H_HW = (1152, 2016)
L_HW = (768, 1344)
GRID = 3
CORE_HW = (384, 672)
HALO = 32
PATCH_HW = (448, 736)
TILES = 4
STRIDE = 8
FEAT_HW = (56, 92)
HEADS = 8
POINTS = 4
D_MODEL = 256
HEAD_DIM = 32
TOP = 300
NATIVE_POINTS = (4, 4, 4)
ROUTE_PREFIX = "r27-v1"
ROUTE_SEED = "seed0"
SCOPES = ("train", "eval")
LOCAL = "r27_local"
READ = "r27_local_read"
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
PARENT_UPDATE = 18240

IDENTITY = {
    "version": VERSION, "H": list(H_HW), "L": list(L_HW),
    "H_to_L": "F.interpolate(I_H, size=(768,1344), mode='bilinear', align_corners=False, antialias=False), FP32, "
              "autocast disabled, on the model device",
    "grid": [GRID, GRID], "core": list(CORE_HW), "halo": HALO, "patch": list(PATCH_HW), "tiles_per_image": TILES,
    "core_rule": "col=clamp(floor(cx*2016/672),0,2), row=clamp(floor(cy*1152/384),0,2), id=3*row+col (FP32)",
    "patch_padding": "zeros of the [0,1] image tensor outside the image",
    "local_backbone": "independent copy of PResNet18 conv1 + max_pool2d(3,2,1) + res_layers[0:2]; stride 8, 128 ch; "
                      "plus 1x1 Conv+BN to 256 copied from encoder.input_proj[0]; batch = 4 patches per image",
    "local_read": {"heads": HEADS, "points": POINTS, "levels": 1, "d_model": D_MODEL,
                   "formula": "p = (cx,cy) + off*(1/4)*(w,h)*0.5; p_patch=(p*(2016,1152)-(core_x-32,core_y-32))/(736,448)",
                   "valid": "0<=p_patch<1 (both axes) and 0<=p*(2016,1152)<(2016,1152) and centre core selected",
                   "softmax": "FP32 over valid points only; all-invalid head -> 0; query without any valid point -> 0",
                   "sampler": "training_v2b_deterministic_sampling.bilinear_grid_sample, FP32 value and grid",
                   "init": "value_proj full copy; sampling_offsets/attention_weights rows of native level-0 (P3) points; "
                           "output_proj weight=0 bias=0",
                   "mask": "after output_proj"},
    "route_P": "p_i = max_c sigmoid(logit_ic) of native L encoder top300 (topk of native max logit; DN excluded); "
               "centre = native selected box; a_j = sum_i onehot(core_i)_j p_i (FP32, one-hot sum); "
               "top4 by stable descending sort (ties -> smaller id); detached, no_grad",
    "route_C": "sha256(utf8('r27-v1|seed0|scope|image_key|epoch|occurrence|tile')) hex ascending (tie: tile id), first 4; "
               "fields with '|' or control characters refused; ints decimal; train: scope=train, epoch=sampler epoch, "
               "occurrence=0 (native per-epoch permutation, checked unique); eval: scope=eval, epoch=0, occurrence=0; "
               "image_key = manifest relative image path",
    "tile_order": "selected ids ascending",
}
IDENTITY_SHA256 = hashlib.sha256(json.dumps(IDENTITY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ---------------------------------------------------------------- geometry helpers
def to_low(x):
    """I_L from I_H; FP32 with autocast disabled (the same expression is used for the native reference)."""
    require(x.dim() == 4 and tuple(x.shape[-2:]) == H_HW and x.dtype == torch.float32, f"R27 expects FP32 H input, got {tuple(x.shape)} {x.dtype}")
    with torch.autocast(x.device.type, enabled=False):
        return F.interpolate(x, size=L_HW, mode="bilinear", align_corners=False, antialias=False)


def core_of(cx, cy):
    """Core id of normalized centres (FP32); left-closed/right-open, image edge -> last core."""
    xp = cx.float() * float(H_HW[1])
    yp = cy.float() * float(H_HW[0])
    col = torch.floor(xp / float(CORE_HW[1])).clamp(0, GRID - 1).long()
    row = torch.floor(yp / float(CORE_HW[0])).clamp(0, GRID - 1).long()
    return row * GRID + col


def core_origin(core):
    """Top-left pixel of the core (y, x) in the H image; patch origin is this minus the halo."""
    return (core // GRID) * CORE_HW[0], (core % GRID) * CORE_HW[1]


def extract_patches(x, tiles):
    """[B,3,H,W] x [B,4] -> [B*4,3,448,736]; zero padding; pure indexing (no gradient path to the image)."""
    require(tiles.dim() == 2 and tiles.shape[1] == TILES and tiles.shape[0] == x.shape[0], "tile layout")
    b = x.shape[0]
    with torch.no_grad():
        padded = F.pad(x, (HALO, HALO, HALO, HALO))
        win = padded.unfold(2, PATCH_HW[0], CORE_HW[0]).unfold(3, PATCH_HW[1], CORE_HW[1])  # [B,3,3r,3c,448,736]
        require(tuple(win.shape[2:4]) == (GRID, GRID), "R27 window grid")
        win = win.permute(0, 2, 3, 1, 4, 5)
        t = tiles.to(device=x.device, dtype=torch.long)
        bi = torch.arange(b, device=x.device)[:, None].expand(b, TILES)
        patches = win[bi, t // GRID, t % GRID]  # [B,4,3,448,736]
        return patches.reshape(b * TILES, x.shape[1], *PATCH_HW).contiguous()


# ---------------------------------------------------------------- routing
def route_string(scope, image_key, epoch, occurrence, tile):
    require(scope in SCOPES, f"R27 route scope {scope!r}")
    require(isinstance(image_key, str) and image_key and "|" not in image_key
            and all(0x20 <= ord(ch) != 0x7F for ch in image_key), f"R27 route image_key refused: {image_key!r}")
    for v in (epoch, occurrence, tile):
        require(type(v) is int and v >= 0, f"R27 route integer field {v!r}")
    require(tile < GRID * GRID, "R27 route tile id")
    return f"{ROUTE_PREFIX}|{ROUTE_SEED}|{scope}|{image_key}|{epoch}|{occurrence}|{tile}"


def route_c_one(scope, image_key, epoch, occurrence):
    order = sorted(range(GRID * GRID), key=lambda j: (hashlib.sha256(
        route_string(scope, image_key, epoch, occurrence, j).encode("utf-8")).hexdigest(), j))
    return sorted(order[:TILES])


def route_p(enc_logits, top_boxes):
    """P tiles and per-core score mass from the native encoder logits [B,N,C] and the top300 boxes [B,300,4+]."""
    with torch.no_grad():
        logits = enc_logits.detach()
        _, ind = torch.topk(logits.max(-1).values, TOP, dim=-1)
        p = logits.float().sigmoid().max(-1).values.gather(1, ind)
        mass = core_mass(p, top_boxes)
        order = torch.sort(mass, dim=1, descending=True, stable=True).indices[:, :TILES]
        return torch.sort(order, dim=1).values, mass, p


def core_mass(p, boxes):
    cells = core_of(boxes[..., 0].detach(), boxes[..., 1].detach())
    onehot = F.one_hot(cells, GRID * GRID).to(torch.float32)
    return (onehot * p.detach().float()[..., None]).sum(1)


# ---------------------------------------------------------------- shared per-forward context
class Context:
    """Per-model, per-forward state shared by the top-level forward, the score head and the decoder layers.

    It holds no tensors between forwards; deepcopy of a model copies one Context shared by its copies.
    """

    def __init__(self, arm):
        require(arm in ARMS, f"R27 arm {arm}")
        self.arm = arm
        self.local = None
        self.route_keys = collections.deque()
        self.audit = False          # lightweight per-forward routing/coverage record (GPU tensors, detached)
        self.records = []
        self.monitor = None         # list: per-layer residual / read statistics (also needs audit)
        self.events = None          # dict of CUDA event pairs when the instrumented timing pass is active
        self.check = False          # cross-check that the P route uses the native selection (tests only)
        self.counters = collections.Counter()
        self.clear()

    def clear(self):
        self.x = None
        self.tiles = None
        self.memory = None
        self.enc_logits = None
        self.prepared = False
        self.layers = 0
        self.record = None
        self.batch_keys = None

    def stamp(self, name):
        if self.events is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.events.setdefault(name, []).append(ev)

    # ------------------------------------------------------------ forward lifecycle
    def begin(self, x, targets):
        require(self.x is None and not self.prepared, "R27 context re-entered")
        b = x.shape[0]
        keys = None
        if self.route_keys:
            require(len(self.route_keys) >= b, "R27 route keys shorter than the batch")
            keys = [self.route_keys.popleft() for _ in range(b)]
            if targets is not None:   # metadata cross-check only; routing never reads targets
                require([k[0] for k in keys] == [t["image_id"] for t in targets], "R27 route keys / batch images differ")
        require(self.arm != "C" or keys is not None, "R27 C routing requires image keys for every image")
        self.batch_keys = keys
        self.x = x
        if self.arm == "C":
            tiles = [route_c_one(*k[1:]) for k in keys]
            self.tiles = torch.tensor(tiles, dtype=torch.long).to(x.device, non_blocking=True)
        self.record = {} if self.audit else None

    def prepare(self, model_local, reference):
        """Layer-0 hook: route (P), extract patches, run the local shallow backbone + projection."""
        self.stamp("prepare_start")
        ref = reference.detach()[:, :, 0, :]
        top = ref[:, -TOP:, :]
        if self.arm == "P":
            tiles, mass, p = route_p(self.enc_logits, top)
            self.tiles = tiles
            if self.check:
                self.counters["route_checks"] += 1
        elif self.audit:
            _, mass, p = route_p(self.enc_logits, top)
        self.stamp("route_end")
        patches = extract_patches(self.x, self.tiles)
        self.stamp("extract_end")
        memory = model_local(patches)
        require(tuple(memory.shape) == (self.x.shape[0] * TILES, D_MODEL, *FEAT_HW), f"local memory shape {tuple(memory.shape)}")
        self.memory = memory.flatten(2).transpose(1, 2)  # [B*4, 5152, 256]
        self.stamp("local_backbone_end")
        self.counters["local_backbone_calls"] += 1
        self.counters["local_patches"] += int(patches.shape[0])
        self.prepared = True
        if self.record is not None:
            with torch.no_grad():
                cells = core_of(top[..., 0], top[..., 1])
                sel = (cells[..., None] == self.tiles[:, None, :]).any(-1)
                self.record.update(tiles=self.tiles.detach().clone(), mass=mass.detach().clone(),
                                   top300_center_selected=sel.float().mean(1), layers=[])

    def end(self, layers_expected):
        require(self.prepared and self.layers == layers_expected, f"R27 forward read {self.layers} layers")
        if self.record is not None:
            self.records.append(self.record)
        self.clear()


# ---------------------------------------------------------------- local modules
class LocalBackbone(nn.Module):
    def __init__(self, conv1, res0, res1, proj):
        super().__init__()
        self.conv1 = conv1
        self.res_layers = nn.ModuleList([res0, res1])
        self.proj = proj

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for stage in self.res_layers:
            x = stage(x)
        return self.proj(x)


def _level0_rows():
    total = sum(NATIVE_POINTS)
    offsets = [(h * total + p) * 2 + xy for h in range(HEADS) for p in range(POINTS) for xy in range(2)]
    weights = [h * total + p for h in range(HEADS) for p in range(POINTS)]
    return offsets, weights


class LocalRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(D_MODEL, D_MODEL)
        self.sampling_offsets = nn.Linear(D_MODEL, HEADS * POINTS * 2)
        self.attention_weights = nn.Linear(D_MODEL, HEADS * POINTS)
        self.output_proj = nn.Linear(D_MODEL, D_MODEL)

    @classmethod
    def from_native(cls, ca):
        require(ca.num_heads == HEADS and list(ca.num_points_list) == list(NATIVE_POINTS) and ca.embed_dim == D_MODEL
                and ca.offset_scale == 0.5 and ca.method == "default" and ca.sampling_offsets.out_features == 192
                and ca.attention_weights.out_features == 96, "R27 native cross-attention identity")
        with torch.random.fork_rng(devices=[]):
            r = cls()
        off, w = _level0_rows()
        with torch.no_grad():
            r.value_proj.weight.copy_(ca.value_proj.weight)
            r.value_proj.bias.copy_(ca.value_proj.bias)
            r.sampling_offsets.weight.copy_(ca.sampling_offsets.weight[off])
            r.sampling_offsets.bias.copy_(ca.sampling_offsets.bias[off])
            r.attention_weights.weight.copy_(ca.attention_weights.weight[w])
            r.attention_weights.bias.copy_(ca.attention_weights.bias[w])
            r.output_proj.weight.zero_()
            r.output_proj.bias.zero_()
        p = ca.value_proj.weight
        r.to(device=p.device, dtype=p.dtype)
        r.train(ca.training)
        r.requires_grad_(p.requires_grad)
        return r

    def geometry(self, offsets, ref):
        """FP32 patch grid, validity and ownership.  offsets [B,Q,64] (any dtype), ref [B,Q,4] -> dict."""
        b, q = ref.shape[:2]
        r = ref.detach().float()
        off = offsets.float().view(b, q, HEADS, POINTS, 2)
        wh = r[:, :, None, None, 2:]
        loc = r[:, :, None, None, :2] + off * (1. / POINTS) * wh * 0.5
        core = core_of(r[..., 0], r[..., 1])                                  # [B,Q]
        oy, ox = core_origin(core)
        size = loc.new_tensor([float(H_HW[1]), float(H_HW[0])])
        gpx = loc * size                                                       # global pixels (x, y)
        origin = torch.stack([ox - HALO, oy - HALO], -1).float()[:, :, None, None, :]
        ppx = gpx - origin                                                     # patch pixels
        patch = loc.new_tensor([float(PATCH_HW[1]), float(PATCH_HW[0])])
        valid = ((ppx >= 0) & (ppx < patch) & (gpx >= 0) & (gpx < size)).all(-1)   # [B,Q,8,4]
        return {"core": core, "patch_xy": ppx / patch, "valid": valid}

    def forward(self, query, reference, ctx, layer):
        """query [B,Q,256] (native cross-attention input), reference [B,Q,1,4] (native detached)."""
        b, q, _ = query.shape
        off = self.sampling_offsets(query)
        logit = self.attention_weights(query)
        value = self.value_proj(ctx.memory)                                   # [B*4, 5152, 256]
        dev = query.device.type
        with torch.autocast(dev, enabled=False):
            g = self.geometry(off, reference[:, :, 0, :])
            owner = g["core"][..., None] == ctx.tiles[:, None, :]             # [B,Q,4]
            covered = owner.any(-1)                                           # [B,Q]
            valid = g["valid"] & covered[:, :, None, None]
            lg = logit.float().view(b, q, HEADS, POINTS)
            anyv = valid.any(-1, keepdim=True)
            lg = torch.where(valid, lg, torch.full_like(lg, float("-inf")))
            lg = torch.where(anyv, lg, torch.zeros_like(lg))
            w = torch.softmax(lg, -1) * valid.to(lg.dtype)                     # [B,Q,8,4]
            v = value.float().view(b, TILES, FEAT_HW[0] * FEAT_HW[1], HEADS, HEAD_DIM)
            v = v.permute(0, 1, 3, 4, 2).reshape(b * TILES * HEADS, HEAD_DIM, *FEAT_HW)
            grid = (2 * g["patch_xy"] - 1).permute(0, 2, 1, 3, 4)            # [B,8,Q,4,2]
            grid = grid[:, None].expand(b, TILES, HEADS, q, POINTS, 2).reshape(b * TILES * HEADS, q, POINTS, 2)
            from ..baseline.training_v2b_deterministic_sampling import bilinear_grid_sample
            s = bilinear_grid_sample(v, grid).view(b, TILES, HEADS, HEAD_DIM, q, POINTS)
            own = owner.permute(0, 2, 1).to(s.dtype)[:, :, None, None, :, None]
            s = (s * own).sum(1)                                              # [B,8,32,Q,4]
            s = (s * w.permute(0, 2, 1, 3)[:, :, None]).sum(-1)               # [B,8,32,Q]
            s = s.reshape(b, D_MODEL, q).permute(0, 2, 1)
            mask = covered & valid.flatten(2).any(-1)                         # [B,Q]
        out = self.output_proj(s)
        out = out * mask[..., None].to(out.dtype)
        if ctx.record is not None:
            with torch.no_grad():
                top = slice(q - TOP, q)
                rec = {"covered": covered[:, top].float().mean(1), "read": mask[:, top].float().mean(1),
                       "valid_point_fraction": valid[:, top].float().mean((1, 2, 3)),
                       "dn_queries": q - TOP}
                if q > TOP:
                    rec["dn_covered"] = covered[:, :q - TOP].float().mean(1)
                    rec["dn_read"] = mask[:, :q - TOP].float().mean(1)
                if ctx.monitor is not None:
                    rec["residual_norm"] = out.detach().float().norm()
                    rec["residual_norm_top300"] = out[:, top].detach().float().norm()
                    rec["output_proj_input_norm"] = s.detach().float().norm()
                ctx.record["layers"].append(rec)
        return out


# ---------------------------------------------------------------- class swaps (decoder/TransformerDecoder types stay native)
_TYPES = {}


def _swap(obj, name, attrs):
    native = type(obj)
    key = (name, native)
    if key not in _TYPES:
        _TYPES[key] = type(name, (native,), {**attrs, "_r27_native": native})
    obj.__class__ = _TYPES[key]


def _score_head_forward(self, x):
    out = nn.Linear.forward(self, x)
    ctx = self.r27_ctx
    if ctx.x is not None:
        require(ctx.enc_logits is None, "R27 encoder score head called twice in one forward")
        ctx.enc_logits = out.detach()
    return out


def _layer_forward(self, target, reference_points, memory, memory_spatial_shapes, attn_mask=None, memory_mask=None,
                   query_pos_embed=None):
    """Native TransformerDecoderLayer.forward plus the local residual on the global cross-attention output."""
    ctx = self.r27_ctx
    q = k = self.with_pos_embed(target, query_pos_embed)
    target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
    target = target + self.dropout1(target2)
    target = self.norm1(target)
    query = self.with_pos_embed(target, query_pos_embed)
    target2 = self.cross_attn(query, reference_points, memory, memory_spatial_shapes, memory_mask)
    if self.r27_index == 0:
        require(not ctx.prepared, "R27 layer 0 prepared twice")
        ctx.prepare(ctx.local, reference_points)
    require(ctx.prepared and ctx.layers == self.r27_index, "R27 layer order")
    ctx.stamp(f"read{self.r27_index}_start")
    target2 = target2 + getattr(self, READ)(query, reference_points, ctx, self.r27_index)
    ctx.stamp(f"read{self.r27_index}_end")
    ctx.layers += 1
    target = target + self.dropout2(target2)
    target = self.norm2(target)
    target2 = self.forward_ffn(target)
    target = target + self.dropout4(target2)
    target = self.norm3(target)
    return target


def _model_forward(self, x, targets=None):
    ctx = self.r27_ctx
    ctx.begin(x, targets)
    try:
        ctx.stamp("h2l_start")
        xl = to_low(x)
        ctx.stamp("h2l_end")
        feats = self.backbone(xl)
        ctx.stamp("backbone_end")
        feats = self.encoder(feats)
        ctx.stamp("encoder_end")
        out = self.decoder(feats, targets)
        ctx.stamp("decoder_end")
        ctx.end(len(self.decoder.decoder.layers) if self.training else self.decoder.decoder.eval_idx + 1)
        return out
    except BaseException:
        ctx.clear()
        raise


def _native_identity(model):
    bb, enc, dec = model.backbone, model.encoder, model.decoder
    layers = dec.decoder.layers
    p0 = enc.input_proj[0]
    ident = {"backbone": type(bb).__name__, "res_layers": len(bb.res_layers), "return_idx": list(bb.return_idx),
             "conv1": list(bb.conv1._modules), "res1_out": bb.res_layers[1].blocks[-1].branch2b.conv.out_channels,
             "input_proj0": [p0.conv.in_channels, p0.conv.out_channels, list(p0.conv.kernel_size), p0.conv.bias is None,
                             type(p0.norm).__name__],
             "decoder_layers": len(layers), "eval_idx": dec.decoder.eval_idx, "canvas": list(dec.eval_spatial_size),
             "queries": dec.num_queries, "classes": dec.num_classes}
    require(ident == {"backbone": "PResNet", "res_layers": 4, "return_idx": [1, 2, 3],
                      "conv1": ["conv1_1", "conv1_2", "conv1_3"], "res1_out": 128,
                      "input_proj0": [128, 256, [1, 1], True, "BatchNorm2d"], "decoder_layers": 3, "eval_idx": 2,
                      "canvas": list(L_HW), "queries": TOP, "classes": 10}, ("R27 native identity differs", ident))
    return ident


def install(model, arm):
    """Install R27 in place on a native RT-DETRv2-R18 model built at L; local parts copy the model's own weights."""
    require(arm in ARMS, f"R27 arm {arm}")
    require(not hasattr(model, "r27_ctx"), "R27 already installed")
    require(not any(getattr(m, "_s15_checkpointed", False) for m in model.modules()), "install R27 before activation checkpointing")
    ident = _native_identity(model)
    ctx = Context(arm)
    bb = model.backbone
    with torch.random.fork_rng(devices=[]):
        local = LocalBackbone(copy.deepcopy(bb.conv1), copy.deepcopy(bb.res_layers[0]), copy.deepcopy(bb.res_layers[1]),
                              copy.deepcopy(model.encoder.input_proj[0]))
    reads = [LocalRead.from_native(layer.cross_attn) for layer in model.decoder.decoder.layers]
    model.add_module(LOCAL, local)
    ctx.local = local
    for i, (layer, r) in enumerate(zip(model.decoder.decoder.layers, reads)):
        layer.add_module(READ, r)
        layer.r27_ctx = ctx
        layer.r27_index = i
        _swap(layer, "R27DecoderLayer", {"forward": _layer_forward})
    head = model.decoder.enc_score_head
    head.r27_ctx = ctx
    _swap(head, "R27EncScoreHead", {"forward": _score_head_forward})
    model.r27_ctx = ctx
    model.r27_arm = arm
    _swap(model, "R27RTDETR", {"forward": _model_forward})
    return {"arm": arm, "native": ident, "local_parameters": sum(p.numel() for p in local.parameters()),
            "read_parameters": sum(p.numel() for r in reads for p in r.parameters()),
            "new_parameters": sum(p.numel() for p in local.parameters()) + sum(p.numel() for r in reads for p in r.parameters())}


def arm_of(model):
    return getattr(model, "r27_arm", None)


def ctx_of(model):
    return model.r27_ctx


def source_name(name):
    """Native parameter/buffer whose value a new R27 entry was copied from (None for output_proj)."""
    if name.startswith(LOCAL + ".conv1."):
        return "backbone.conv1." + name[len(LOCAL + ".conv1."):]
    if name.startswith(LOCAL + ".res_layers."):
        return "backbone.res_layers." + name[len(LOCAL + ".res_layers."):]
    if name.startswith(LOCAL + ".proj."):
        return "encoder.input_proj.0." + name[len(LOCAL + ".proj."):]
    if f".{READ}." in name:
        layer, rest = name.split(f".{READ}.")
        if rest.startswith("output_proj."):
            return None
        return f"{layer}.cross_attn.{rest}"
    return "native"


def is_new(name):
    return name.startswith(LOCAL + ".") or f".{READ}." in name


def verify_copies(model):
    """Every new entry equals its declared native source (rows for offsets/weights); output_proj is zero."""
    sd = model.state_dict()
    off, w = _level0_rows()
    checked = 0
    for name, value in sd.items():
        if not is_new(name):
            continue
        src = source_name(name)
        if src is None:
            require(not bool(value.abs().sum() != 0), f"R27 output_proj not zero: {name}")
        elif name.endswith("sampling_offsets.weight") or name.endswith("sampling_offsets.bias"):
            require(torch.equal(value, sd[src][off]), f"R27 offset rows differ: {name}")
        elif name.endswith("attention_weights.weight") or name.endswith("attention_weights.bias"):
            require(torch.equal(value, sd[src][w]), f"R27 weight rows differ: {name}")
        else:
            require(torch.equal(value, sd[src]), f"R27 copy differs: {name}")
        require(value.data_ptr() != sd[src].data_ptr() if src is not None else True, f"R27 copy shares storage: {name}")
        checked += 1
    return checked


# ---------------------------------------------------------------- engine construction from the e060 parent
def create(parent, device, arm, *, fixture=False, activation_checkpoint=True):
    """Global L network from the native e060 complete state (no size-cache bridge needed), R27 installed on the
    online and EMA models (each from its own weights), engine at the S15 H recipe (or the CPU H fixture)."""
    from . import scale_baseline as S
    from ..protocol_r18 import checkpoint as native
    from ..protocol_r18.engine import Engine, learning_rate, tensor_digest
    from ..protocol_r18.geometry import validate_geometry
    from ..protocol_r18.model import CommonEMA, Config, build
    from ..baseline.training_v2b import _optimizer
    require(arm in ARMS, f"R27 arm {arm}")
    gcfg = Config(seed=0)
    ecfg = Config(seed=0, input_hw=H_HW, physical_batch_size=1, amp_dtype="float32") if fixture else S.production_config(0)
    if not fixture:
        S.validate_production(ecfg)
    require(native.file_sha(parent) == PARENT_SHA, "R27 parent identity")
    s = torch.load(parent, map_location="cpu", weights_only=True)
    require(s["engine"]["completed_updates"] == PARENT_UPDATE, "R27 parent clock")
    require(s["binding"]["config"] == gcfg.binding() and s["engine"].get("config", s["binding"]["config"]) == gcfg.binding(),
            "R27 parent recipe is not the common 768x1344 seed0 recipe")
    model, criterion, _ = build(gcfg, synthetic=True)
    require(native.old._model_layout(model) == s["model_layout"], "R27 global model layout differs from the parent")
    require(criterion.recipe() == s["behavior"]["criterion"], "R27 criterion behaviour differs")
    validate_geometry(model)
    model.load_state_dict(s["model"], strict=True)
    ema = CommonEMA(model)
    ema.load_state_dict(s["ema"])
    for n, m in model.named_modules():
        m.training = s["model_training"][n]
    info = install(model, arm)
    install(ema.module, arm)
    ema.module.requires_grad_(False)
    copies = {"online": verify_copies(model), "EMA": verify_copies(ema.module)}
    saved = {}
    for g in s["optimizer_layout"]["groups"]:
        for name, i in zip(g["names"], g["ids"]):
            require(name not in saved, "duplicate parent optimizer name")
            saved[name] = s["optimizer"]["state"].get(i, {})
    opt = _optimizer(model, ecfg)
    old_groups = s["optimizer"]["param_groups"]
    require([g["name"] for g in opt.param_groups] == [g["name"] for g in old_groups], "R27 optimizer group order")
    group_of = {n: g["name"] for g in opt.param_groups for n in g["param_names"]}
    old_group_of = {n: g["name"] for g in old_groups for n in g["param_names"]}
    for g, og in zip(opt.param_groups, old_groups):
        for k, v in og.items():
            if k not in ("params", "param_names"):
                g[k] = copy.deepcopy(v)
        for n, p in zip(g["param_names"], g["params"]):
            if n in saved:
                opt.state[p] = copy.deepcopy(saved[n])
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    new = [n for n in names if is_new(n)]
    require(sorted(set(names) - set(new)) == sorted(saved), "R27 old optimizer coverage")
    mapping = {}
    for n in new:
        src = source_name(n)
        src_group = old_group_of[src] if src is not None else old_group_of[n.replace(f".{READ}.", ".cross_attn.")]
        require(group_of[n] == src_group, f"R27 new parameter group differs from its source rule: {n}")
        mapping[n] = {"source": src or n.replace(f".{READ}.", ".cross_attn."), "group": group_of[n]}
    model.to(device)
    criterion.to(device)
    ema.module.to(device)
    for p, st in opt.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and k != "step":
                st[k] = v.to(p.device)
    e = Engine(model, criterion, opt, ema, ecfg)
    e.completed_updates = s["engine"]["completed_updates"]
    old_steps = s["engine"]["parameter_optimizer_steps"]
    e.parameter_steps = {n: (old_steps[n] if not is_new(n) else 0) for n in e.parameter_steps}
    require(set(old_steps) == set(n for n in e.parameter_steps if not is_new(n)), "R27 parameter step coverage")
    e.r25_arm = f"R27-{arm}"
    e.r25_parent = PARENT_SHA
    e.r27_arm = arm
    wrapped = []
    if activation_checkpoint:
        wrapped = S.install_activation_checkpoint(model)
        for i, block in enumerate(model.r27_local.res_layers):
            S._wrap(block)
            wrapped.append(f"{LOCAL}.res_layers.{i}")
    e.s15_activation_checkpoint = bool(activation_checkpoint)
    validate_geometry(model)
    validate_geometry(ema.module)
    named = {n: p for n, p in model.named_parameters()}
    checks = {
        "online_old_exact": all(torch.equal(model.state_dict()[n].cpu(), v) for n, v in s["model"].items()),
        "EMA_old_exact": all(torch.equal(ema.module.state_dict()[n].cpu(), v) for n, v in s["ema"]["module"].items()),
        "optimizer_old_exact": all(native.exact_equal(S._cpu(opt.state[named[n]]), S._cpu(v)) for n, v in saved.items()),
        "new_optimizer_state_empty": all(not opt.state.get(named[n]) for n in new),
        "parameter_steps_old_exact": all(e.parameter_steps[n] == v for n, v in old_steps.items()),
        "new_parameter_steps_zero": all(e.parameter_steps[n] == 0 for n in new),
        "copies_checked": copies, "EMA_updates": ema.updates, "completed_updates": e.completed_updates,
        "lr_next": learning_rate(e.completed_updates + 1),
        "new_parameter_names": new, "new_parameter_group_mapping": mapping,
        "new_parameters": sum(named[n].numel() for n in new),
        "parameters": sum(p.numel() for p in model.parameters()),
        "old_parameters_sha256": tensor_digest((n, p) for n, p in model.named_parameters() if not is_new(n)),
        "new_parameters_sha256": tensor_digest((n, p) for n, p in model.named_parameters() if is_new(n)),
        "activation_checkpoint_segments": wrapped,
        "engine_config": ecfg.binding(), "global_model_config": gcfg.binding(),
        "identity_sha256": IDENTITY_SHA256, **info,
    }
    require(all(checks[k] for k in ("online_old_exact", "EMA_old_exact", "optimizer_old_exact", "new_optimizer_state_empty",
                                    "parameter_steps_old_exact", "new_parameter_steps_zero")), ("R27 bridge mismatch", checks))
    require(ema.updates == e.completed_updates == PARENT_UPDATE, "R27 clocks differ")
    if device != "cpu":
        native.restore_rng(s["rng"], e.device)
    else:
        rng = dict(s["rng"])
        rng["cuda"] = None
        native.restore_rng(rng, e.device)
    return e, checks


def build_eval_model(arm, state, device):
    """FP32 evaluation model of an arm from a saved module state (EMA or online); no activation checkpoint."""
    from ..protocol_r18.engine import tensor_digest
    from ..protocol_r18.geometry import validate_geometry
    from ..protocol_r18.model import Config, build
    model, _, _ = build(Config(seed=0), synthetic=True)
    install(model, arm)
    model.load_state_dict(state, strict=True)
    validate_geometry(model)
    model.eval().requires_grad_(False).to(device)
    require(tensor_digest(model.state_dict().items()) == tensor_digest(state.items()), "R27 evaluation strict load differs")
    return model
