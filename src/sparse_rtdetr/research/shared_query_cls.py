"""SQC-001: shared-query cross-scale classification readout on a frozen detector.

TASK_SHARED_QUERY_CLASSIFICATION_001.  The detector (seed0 S15 e120 EMA) is frozen: eval mode, FP32,
no parameter, BatchNorm, EMA or buffer update.  One native H (1152x2016) forward provides the real
entry of the last (third) decoder layer -- query content q2_H, reference box r2_H, query position
embedding, masks, H memory and H spatial shapes -- together with that layer's native output z_H, the
native logits s_H and the final boxes b_H.

The L (768x1344) rendering of the same original image runs only backbone, encoder and the decoder
memory projection (``_get_encoder_input``) of a second instance holding the same learned tensors and
its own native L canvas caches.  The very same captured entry is then sent through the same last layer
with the L memory and the L spatial shapes, giving z_L|H.  No L query selection, no L decoder query
slots, no cross-model pairing.  Every output box is b_H.

Head (identical structure and initial bytes for both arms):
    delta = Linear(128, C)(GELU(Linear(2d+4, 128)(cat(z_H, z_source, b_H))));  s_new = s_H + delta
    B1: z_source = z_H        B2: z_source = z_L|H

Supervision: the native final VFL classification term on the fixed native H match (computed once from
s_H/b_H), with the native raw-ignore negative mask, alpha/gamma, num_boxes normalisation and weight.
"""
from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = "SQC-001-v1"
H_HW = (1152, 2016)
L_HW = (768, 1344)
STRIDES = (8, 16, 32)
QUERIES = 300
CLASSES = 10
D_MODEL = 256
HIDDEN = 128
LAST_LAYER = 2
TRAIN_CORE_IMAGES = 4869
VALIDATION_IMAGES = 512
STEP_IMAGES = 16
EPOCHS = 8
SPLIT_SALT = "sqc-v1|"
ORDER_SALT = "sqc-order-v1|"
INIT_SEED = 0
OPTIMIZER = {"type": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4,
             "weight_decay_scope": "all head parameters", "warmup": None, "scheduler": None}
CLIP_NORM = 1.0
SIZE_BUCKETS = ("small", "medium", "large")  # COCO area thresholds 32^2 / 96^2 on the raw GT area (descriptive)
HEAD_PARAMETERS = (2 * D_MODEL + 4) * HIDDEN + HIDDEN + HIDDEN * CLASSES + CLASSES


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_tensors(items):
    """sha256 over (name, dtype, shape, raw bytes) in the given order."""
    h = hashlib.sha256()
    for name, value in items:
        x = value.detach().cpu().contiguous()
        h.update(f"{name}:{x.dtype}:{tuple(x.shape)}".encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def level_shapes(hw):
    return [[hw[0] // s, hw[1] // s] for s in STRIDES]


# ---------------------------------------------------------------- split and fixed order (no data read)
def _key(text):
    return hashlib.sha256(text.encode()).digest()


def split_ids(ids):
    """First 512 by sha256('sqc-v1|'+id) bytes (ties by id) are head validation, the rest head training."""
    ids = list(ids)
    require(len(ids) == len(set(ids)) == TRAIN_CORE_IMAGES, "train_core membership is not the fixed 4869")
    keyed = sorted(ids, key=lambda i: (_key(SPLIT_SALT + i), i))
    return {"validation": keyed[:VALIDATION_IMAGES], "train": keyed[VALIDATION_IMAGES:]}


def epoch_order(train, epoch):
    require(type(epoch) is int and 1 <= epoch <= EPOCHS, "epoch outside 1..8")
    return sorted(train, key=lambda i: (_key(f"{ORDER_SALT}{epoch}|{i}"), i))


def schedule(train):
    """[(epoch, step_in_epoch, global_step, ids)]; 16 images per step, last step of an epoch keeps the rest."""
    steps = []
    for epoch in range(1, EPOCHS + 1):
        order = epoch_order(train, epoch)
        for k in range(0, len(order), STEP_IMAGES):
            steps.append((epoch, k // STEP_IMAGES + 1, len(steps) + 1, order[k:k + STEP_IMAGES]))
    return steps


# ---------------------------------------------------------------- head
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2 * D_MODEL + 4, HIDDEN)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(HIDDEN, CLASSES)

    def forward(self, z_h, z_source, boxes):
        return self.fc2(self.act(self.fc1(torch.cat((z_h, z_source, boxes), -1))))


def initial_state():
    """One independent CPU generator (seed 0): Xavier-uniform first weight, zero first bias, zero last layer."""
    g = torch.Generator().manual_seed(INIT_SEED)
    weight = torch.empty(HIDDEN, 2 * D_MODEL + 4, dtype=torch.float32)
    nn.init.xavier_uniform_(weight, generator=g)
    return {"fc1.weight": weight, "fc1.bias": torch.zeros(HIDDEN), "fc2.weight": torch.zeros(CLASSES, HIDDEN),
            "fc2.bias": torch.zeros(CLASSES)}


def new_head(state, device="cpu"):
    with torch.random.fork_rng(devices=[]):  # default Linear init must not touch the caller's RNG
        head = Head()
    head.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
    require(sum(p.numel() for p in head.parameters()) == HEAD_PARAMETERS, "head parameter count")
    return head.to(device)


def new_optimizer(head):
    return torch.optim.AdamW(head.parameters(), lr=OPTIMIZER["lr"], betas=tuple(OPTIMIZER["betas"]),
                             eps=OPTIMIZER["eps"], weight_decay=OPTIMIZER["weight_decay"])


def apply_head(head, s_h, z_h, z_source, b_h):
    """s_new = s_H + delta over all queries and classes; boxes are not an output of this function."""
    n, q, _ = s_h.shape
    delta = head(z_h.reshape(n * q, -1), z_source.reshape(n * q, -1), b_h.reshape(n * q, 4))
    return s_h + delta.reshape(n, q, CLASSES)


def head_digest(head):
    return digest_tensors(head.state_dict().items())


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    items = []
    for pid in sorted(state["state"]):
        for k in sorted(state["state"][pid]):
            v = state["state"][pid][k]
            items.append((f"{pid}.{k}", v if isinstance(v, torch.Tensor) else torch.tensor(v)))
    return digest_tensors(items)


# ---------------------------------------------------------------- last-layer entry capture and readouts
@dataclass
class Entry:
    target: torch.Tensor          # q2_H: query content entering the last layer   [B, 300, d]
    reference: torch.Tensor       # r2_H: sigmoid reference box, unsqueezed      [B, 300, 1, 4]
    memory: torch.Tensor          # H memory                                     [B, Lv, d]
    spatial_shapes: list          # H spatial shapes
    attn_mask: object
    memory_mask: object
    query_pos: torch.Tensor       # query_pos_head(r2_H)                         [B, 300, d]

    def identity(self):
        """Per-image digest of the query identity actually entering the last layer."""
        return [digest_tensors([("q", self.target[i]), ("r", self.reference[i]), ("pos", self.query_pos[i])])
                for i in range(self.target.shape[0])]


@contextlib.contextmanager
def last_layer_capture(model):
    """Transient hooks on the native last decoder layer; no arithmetic or state is changed."""
    layer = model.decoder.decoder.layers[LAST_LAYER]
    record = {"calls": 0, "outputs": 0}

    def pre(module, args, kwargs):
        require(not kwargs and len(args) == 7, "last decoder layer called with an unexpected signature")
        record["calls"] += 1
        record["entry"] = Entry(*args)

    def post(module, args, result):
        record["outputs"] += 1
        record["z"] = result

    h1 = layer.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = layer.register_forward_hook(post)
    try:
        yield record
    finally:
        h1.remove()
        h2.remove()


def h_forward(model, x):
    """Native full forward with the last-layer entry and output captured."""
    require(not model.training and model.decoder.decoder.eval_idx == LAST_LAYER, "eval-mode native decoder required")
    with last_layer_capture(model) as record:
        out = model(x)
    require(record["calls"] == record["outputs"] == 1, "last layer must run exactly once per forward")
    entry, z = record["entry"], record["z"]
    require(entry.attn_mask is None and entry.memory_mask is None, "eval forward has no attention/memory mask")
    return out, entry, z


def last_layer(model, entry, memory=None, spatial_shapes=None):
    """Same last layer, same captured entry; only the memory (and its own spatial shapes) may differ."""
    layer = model.decoder.decoder.layers[LAST_LAYER]
    if memory is None:
        memory, spatial_shapes = entry.memory, entry.spatial_shapes
    require(spatial_shapes is not None and memory.shape[1] == sum(h * w for h, w in spatial_shapes),
            "memory length differs from its spatial shapes")
    return layer(entry.target, entry.reference, memory, spatial_shapes, entry.attn_mask, entry.memory_mask,
                 entry.query_pos)


def native_logits(model, z):
    return model.decoder.dec_score_head[LAST_LAYER](z)


def memory_only(model, x):
    """Backbone, encoder and decoder memory projection of one canvas; no query selection or decoding."""
    feats = model.encoder(model.backbone(x))
    memory, spatial_shapes = model.decoder._get_encoder_input(feats)
    return memory, spatial_shapes


def cross_readout(model_h, model_l, entry, x_l):
    """z_L|H: the H entry through the H model's last layer with the L memory of the same image."""
    memory, shapes = memory_only(model_l, x_l)
    require([list(s) for s in shapes] == level_shapes(tuple(model_l.decoder.eval_spatial_size)),
            "L spatial shapes differ from the L canvas")
    return last_layer(model_h, entry, memory, shapes), memory, shapes


# ---------------------------------------------------------------- labels (offline, after all forwards)
def h_training_target(aug, raw_target, original_hw, image=None):
    """Training-geometry GT/ignore at the H canvas without augmentation draws.

    Mirrors RawAugmentation with training=True at epoch >= 118 (no photometric/zoom/crop) minus the
    random horizontal flip: positive-area clip -> sanitize(original) -> resize -> sanitize(H canvas).
    ``image`` may be omitted; boxes are transformed from their own canvas size (checked equal).
    """
    from PIL import Image
    from torchvision import tv_tensors
    from torchvision.ops import box_convert
    from ..localization.data import _clip, _valid
    from ..localization.ignore import disjoint_union
    hw = aug.input_hw
    h, w = original_hw
    all_gt = raw_target["gt"]
    gt_np = _clip(np.array([r["box"] for r in all_gt], dtype=np.float64).reshape(-1, 4), (h, w))
    kept = np.where(_valid(gt_np))[0]
    boxes = tv_tensors.BoundingBoxes(torch.tensor(gt_np[kept], dtype=torch.float32), format="XYXY", canvas_size=(h, w))
    target = {"boxes": boxes, "labels": torch.tensor([all_gt[i]["category"] - 1 for i in kept], dtype=torch.long),
              "row_indices": torch.tensor(kept, dtype=torch.long)}
    ignored = raw_target["spatial_ignore"] + raw_target["class_ignore"]
    rectangles = _clip(np.array([r["box"] for r in ignored], dtype=np.float64).reshape(-1, 4), (h, w))
    canvas = image if image is not None else Image.new("RGB", (w, h))
    require((canvas.height, canvas.width) == (h, w), "image size differs from the original size")
    canvas, target = aug.sanitize(canvas, target)
    before_hw = (canvas.height, canvas.width)
    canvas, target = aug.resize(canvas, target)
    rectangles *= np.array([hw[1] / before_hw[1], hw[0] / before_hw[0]] * 2)
    rectangles = _clip(rectangles, hw)
    canvas, target = aug.sanitize(canvas, target)
    gt_boxes = box_convert(target["boxes"].as_subclass(torch.Tensor), "xyxy", "cxcywh")
    target["boxes"] = gt_boxes / torch.tensor(hw[::-1] * 2, dtype=torch.float32)
    retained = target.pop("row_indices").tolist()
    target["gt_ids"] = [all_gt[i]["id"] for i in retained]
    target["raw_area"] = torch.tensor([all_gt[i]["raw_area"] for i in retained], dtype=torch.float64)
    valid = _valid(rectangles)
    n0 = len(raw_target["spatial_ignore"])
    norm = np.array(hw[::-1] * 2)
    spatial = rectangles[:n0][valid[:n0]] / norm
    class_boxes = rectangles[n0:][valid[n0:]] / norm
    target["ignore_spatial_boxes"] = torch.tensor(spatial, dtype=torch.float64)
    target["ignore_union"] = torch.tensor(disjoint_union(spatial), dtype=torch.float64)
    target["ignore_class_boxes"] = torch.tensor(class_boxes, dtype=torch.float64)
    target["ignore_class_labels"] = torch.tensor([r["category"] - 1 for r, k in zip(ignored[n0:], valid[n0:]) if k],
                                                 dtype=torch.long)
    target["ignore_ids"] = [r["id"] for r, k in zip(ignored, valid) if k]
    return target


def target_digest(target):
    items = [(k, target[k]) for k in ("boxes", "labels", "raw_area", "ignore_spatial_boxes", "ignore_union",
                                      "ignore_class_boxes", "ignore_class_labels")]
    h = hashlib.sha256(digest_tensors(items).encode())
    h.update(repr((target["gt_ids"], target["ignore_ids"])).encode())
    return h.hexdigest()


def _vendor_module(criterion):
    native = type(criterion).__bases__[0]
    return __import__(native.__module__, fromlist=["box_iou", "box_cxcywh_to_xyxy"])


def size_bucket(area):
    return 0 if area < 32 ** 2 else 1 if area < 96 ** 2 else 2


def query_labels(criterion, logits, boxes, target):
    """Native matcher once on native (s_H, b_H); VFL class/IoU targets and the raw-ignore keep mask."""
    from ..localization import consumer_geometry as geo
    from ..localization.ignore import negative_keep_mask
    require(logits.shape == (QUERIES, CLASSES) and boxes.shape == (QUERIES, 4), "single-image native output")
    module = _vendor_module(criterion)
    outputs = {"pred_logits": logits[None], "pred_boxes": boxes[None]}
    with geo.matching(criterion.matcher):
        src, dst = criterion.matcher(outputs, [{"labels": target["labels"], "boxes": target["boxes"]}])["indices"][0]
    classes = torch.full((QUERIES,), CLASSES, dtype=torch.int64)
    classes[src] = target["labels"][dst]
    quality = torch.zeros(QUERIES, dtype=torch.float32)
    bucket = torch.full((QUERIES,), -1, dtype=torch.int8)
    if len(src):
        iou, _ = module.box_iou(module.box_cxcywh_to_xyxy(boxes[src]), module.box_cxcywh_to_xyxy(target["boxes"][dst]))
        quality[src] = iou.diag().detach().to(torch.float32)
        bucket[src] = torch.tensor([size_bucket(float(target["raw_area"][j])) for j in dst.tolist()], dtype=torch.int8)
    keep = negative_keep_mask(boxes[None], [(src, dst)], [target], num_classes=CLASSES)[0]
    return {"classes": classes, "quality": quality, "keep": keep, "bucket": bucket, "num_gt": len(target["labels"]),
            "matched_queries": src.tolist(), "matched_gt_index": dst.tolist(),
            "matched_gt_ids": [target["gt_ids"][j] for j in dst.tolist()]}


# ---------------------------------------------------------------- native final VFL on new logits
def vfl_terms(logits, classes, quality, keep, alpha, gamma):
    """Per (image, query, class) native VFL terms, raw-ignore masked; logits [N, Q, C]."""
    target = F.one_hot(classes, num_classes=CLASSES + 1)[..., :-1]
    score = quality.unsqueeze(-1) * target
    weight = alpha * logits.sigmoid().detach().pow(gamma) * (1 - target) + score
    return F.binary_cross_entropy_with_logits(logits, score, weight=weight, reduction="none") * keep


def vfl_loss(logits, classes, quality, keep, num_gt, alpha, gamma, loss_weight):
    """Native normalisation: mean over queries, sum, times Q, over max(GT count, 1); times the native weight."""
    num_boxes = max(float(num_gt), 1.0)
    loss = vfl_terms(logits, classes, quality, keep, alpha, gamma)
    return loss.mean(1).sum() * logits.shape[1] / num_boxes * loss_weight


def loss_decomposition(logits, classes, quality, keep, bucket, alpha, gamma):
    """Unnormalised sums of the VFL terms by matched-GT size bucket and background (descriptive)."""
    per_query = vfl_terms(logits, classes, quality, keep, alpha, gamma).sum(-1)
    out = {"background": float(per_query[bucket < 0].sum())}
    for k, name in enumerate(SIZE_BUCKETS):
        out[name] = float(per_query[bucket == k].sum())
        out[name + "_queries"] = int((bucket == k).sum())
    out["background_queries"] = int((bucket < 0).sum())
    return out


# ---------------------------------------------------------------- one head update (shared by checks and fitting)
ARMS = ("B1", "B2")


def source_features(arm, z_h, z_l):
    """The only difference between the arms: the second feature copy."""
    require(arm in ARMS, "unknown arm")
    return z_h if arm == "B1" else z_l


def head_step(head, optimizer, batch, recipe):
    """Native final VFL on s_H + delta for every query/class of the batch; global clip 1.0; one AdamW step."""
    logits = apply_head(head, batch["s_H"], batch["z_H"], batch["z_source"], batch["b_H"])
    loss = vfl_loss(logits, batch["classes"], batch["quality"], batch["keep"], batch["num_gt"],
                    recipe["alpha"], recipe["gamma"], recipe["loss_weight"])
    require(bool(torch.isfinite(loss)), "nonfinite head loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(head.parameters()), CLIP_NORM)
    require(bool(torch.isfinite(norm)), "nonfinite head gradient norm")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm_before_clip": float(norm), "num_gt": int(batch["num_gt"])}


def head_state(head, optimizer, cursor, rng):
    """Complete head training state (CPU copies)."""
    def cpu(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
        if isinstance(v, dict):
            return {k: cpu(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(cpu(x) for x in v)
        return v
    return {"version": VERSION, "head": cpu(head.state_dict()), "optimizer": cpu(optimizer.state_dict()),
            "cursor": dict(cursor), "rng": cpu(rng)}
