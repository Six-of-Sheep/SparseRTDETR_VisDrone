"""SQC-001 unit tests on a native synthetic fixture (CPU; no data, no checkpoint, no CUDA)."""
import copy

import pytest
import torch

from sparse_rtdetr.protocol_r18.model import Config, build
from sparse_rtdetr.protocol_r18.geometry import validate_geometry
from sparse_rtdetr.localization.data import RawAugmentation
from sparse_rtdetr.research import shared_query_cls as Q

torch.set_num_threads(2)
FH, FL = (192, 256), (128, 192)  # fixture "H" and "L" canvases (same aspect handling as production)


def _state_digest(model):
    return Q.digest_tensors(model.state_dict().items())


@pytest.fixture(scope="module")
def models():
    h, criterion, _ = build(Config(input_hw=FH, physical_batch_size=1, amp_dtype="float32"), synthetic=True)
    l, _, _ = build(Config(input_hw=FL, physical_batch_size=1, amp_dtype="float32"), synthetic=True)
    state = h.state_dict()
    own = l.state_dict()
    loaded = {k: (own[k].clone() if k in ("decoder.anchors", "decoder.valid_mask") else v.clone()) for k, v in state.items()}
    l.load_state_dict(loaded, strict=True)
    for m in (h, l):
        m.eval().requires_grad_(False)
        validate_geometry(m)
    return h, l, criterion


def _images(seed, hw, n=2):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3, *hw), generator=g)


def _pair(seed=1, n=2):
    """Same 'original' content rendered at both canvases by one direct resize each."""
    from torchvision.transforms.v2 import functional as TF
    base = _images(seed, (384, 512), n)
    return TF.resize(base, list(FH), antialias=True), TF.resize(base, list(FL), antialias=True)


def test_replay_and_native_heads(models):
    h, _, _ = models
    xh, _ = _pair()
    with torch.inference_mode():
        out, entry, z = Q.h_forward(h, xh)
        z2 = Q.last_layer(h, entry)
        logits = Q.native_logits(h, z)
        ref = entry.reference.squeeze(2)
        boxes = torch.sigmoid(h.decoder.dec_bbox_head[Q.LAST_LAYER](z) + _inverse_sigmoid(ref))
    assert torch.equal(z2, z)
    assert torch.equal(logits, out["pred_logits"])
    assert torch.equal(boxes, out["pred_boxes"])
    assert entry.target.shape == (2, 300, 256) and entry.reference.shape == (2, 300, 1, 4)
    assert [list(s) for s in entry.spatial_shapes] == [[FH[0] // s, FH[1] // s] for s in Q.STRIDES]


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def test_l_shapes_and_no_h_pollution(models):
    h, l, _ = models
    xh, xl = _pair(2)
    dh, dl = _state_digest(h), _state_digest(l)
    with torch.inference_mode():
        out1, entry, z1 = Q.h_forward(h, xh)
        zl, memory, shapes = Q.cross_readout(h, l, entry, xl)
        out2, entry2, z2 = Q.h_forward(h, xh)
    assert [list(s) for s in shapes] == [[FL[0] // s, FL[1] // s] for s in Q.STRIDES]
    assert memory.shape[1] == sum(a * b for a, b in shapes) != entry.memory.shape[1]
    assert zl.shape == z1.shape and not torch.equal(zl, z1)
    assert torch.equal(out1["pred_logits"], out2["pred_logits"]) and torch.equal(out1["pred_boxes"], out2["pred_boxes"])
    assert torch.equal(z1, z2) and entry.identity() == entry2.identity()
    assert _state_digest(h) == dh and _state_digest(l) == dl
    assert list(h.decoder.eval_spatial_size) == list(FH) and list(l.decoder.eval_spatial_size) == list(FL)
    # the H memory can not be interpreted with the L shapes
    with pytest.raises(ValueError):
        Q.last_layer(h, entry, entry.memory, shapes)


def test_initial_bytes_and_zero_head_identity(models):
    rng = torch.get_rng_state()
    a, b = Q.initial_state(), Q.initial_state()
    assert torch.equal(torch.get_rng_state(), rng)
    assert Q.digest_tensors(a.items()) == Q.digest_tensors(b.items())
    h1, h2 = Q.new_head(a), Q.new_head(b)
    assert torch.equal(torch.get_rng_state(), rng)
    assert Q.head_digest(h1) == Q.head_digest(h2)
    assert sum(p.numel() for p in h1.parameters()) == Q.HEAD_PARAMETERS == 67466
    bound = (6 / (516 + 128)) ** .5
    assert a["fc1.weight"].abs().max() <= bound and a["fc1.weight"].abs().max() > .9 * bound
    assert not a["fc1.bias"].any() and not a["fc2.weight"].any() and not a["fc2.bias"].any()
    s = torch.randn(2, 300, 10)
    z = torch.randn(2, 300, 256)
    zl = torch.randn(2, 300, 256)
    bx = torch.rand(2, 300, 4)
    before = bx.clone()
    with torch.no_grad():
        assert torch.equal(Q.apply_head(h1, s, z, z, bx), s)
        assert torch.equal(Q.apply_head(h2, s, z, zl, bx), s)
    assert torch.equal(bx, before)


def test_split_and_schedule():
    ids = [f"train/img{i:05d}" for i in range(Q.TRAIN_CORE_IMAGES)]
    s1 = Q.split_ids(ids)
    s2 = Q.split_ids(list(reversed(ids)))
    assert s1 == s2 and len(s1["validation"]) == 512 and len(s1["train"]) == 4357
    assert not set(s1["validation"]) & set(s1["train"])
    steps = Q.schedule(s1["train"])
    assert len(steps) == 2184 and steps[-1][2] == 2184
    per_epoch = [sum(1 for s in steps if s[0] == e) for e in range(1, 9)]
    assert per_epoch == [273] * 8
    for e in range(1, 9):
        last = [s for s in steps if s[0] == e][-1]
        assert len(last[3]) == 5
        assert sorted(i for s in steps if s[0] == e for i in s[3]) == sorted(s1["train"])
    assert Q.epoch_order(s1["train"], 1) != Q.epoch_order(s1["train"], 2)
    with pytest.raises(ValueError):
        Q.split_ids(ids[:-1])


def _raw_target():
    def row(i, box, cat, score=1):
        return {"id": f"g{i}", "box": box, "category": cat, "score": score, "raw_area": (box[2] - box[0]) * (box[3] - box[1])}
    gt = [row(0, [10, 20, 60, 90], 1), row(1, [100, 40, 102.5, 42.5], 4), row(2, [200, 100, 380, 300], 6),
          row(3, [500, 0, 512, 10], 9)]
    return {"gt": gt, "spatial_ignore": [row(10, [0, 300, 50, 384], 0)], "class_ignore": [row(11, [300, 0, 340, 30], 4, 0)],
            "counts": {}}


def test_targets_resize_only_and_canvas_invariant():
    raw = _raw_target()
    th = Q.h_training_target(RawAugmentation(FH), raw, (384, 512))
    tl = Q.h_training_target(RawAugmentation(FL), raw, (384, 512))
    original = torch.tensor([g["box"] for g in raw["gt"]], dtype=torch.float64)
    expect = torch.stack(((original[:, 0] + original[:, 2]) / 2 / 512, (original[:, 1] + original[:, 3]) / 2 / 384,
                          (original[:, 2] - original[:, 0]) / 512, (original[:, 3] - original[:, 1]) / 384), 1)
    keep = [th["gt_ids"].index(g) for g in th["gt_ids"]]
    rows = [int(g[1:]) for g in th["gt_ids"]]
    assert torch.allclose(th["boxes"].double(), expect[rows], atol=1e-6)
    assert th["labels"].tolist() == [raw["gt"][r]["category"] - 1 for r in rows] and keep == list(range(len(rows)))
    # 2.5 px original box: 1.25 px at the fixture H canvas, < 1 px at L: native sanitize removes it there only
    assert "g1" in th["gt_ids"] and "g1" not in tl["gt_ids"]
    assert th["ignore_union"].shape[0] == 1 and th["ignore_class_labels"].tolist() == [3]


def _labels(criterion, s, b, target):
    return Q.query_labels(criterion, s, b, target)


def _fixture_targets(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = 3 + i
        c = torch.rand((k, 2), generator=g) * .6 + .2
        wh = torch.rand((k, 2), generator=g) * .2 + .05
        out.append({"boxes": torch.cat((c, wh), 1), "labels": torch.randint(0, 10, (k,), generator=g),
                    "gt_ids": [f"x{i}-{j}" for j in range(k)], "ignore_ids": ["u"],
                    "raw_area": torch.tensor([200., 2000., 20000.] + [5.] * (k - 3), dtype=torch.float64),
                    "ignore_spatial_boxes": torch.tensor([[0., 0., .3, .3]], dtype=torch.float64),
                    "ignore_union": torch.tensor([[0., 0., .3, .3]], dtype=torch.float64),
                    "ignore_class_boxes": torch.tensor([[.5, .5, .9, .9]], dtype=torch.float64),
                    "ignore_class_labels": torch.tensor([2])})
    return out


def test_vfl_equals_native_criterion(models):
    h, _, criterion = models
    xh, _ = _pair(4)
    with torch.inference_mode():
        out, _, _ = Q.h_forward(h, xh)
    s, b = out["pred_logits"].clone(), out["pred_boxes"].clone()
    targets = _fixture_targets(2)
    labels = [_labels(criterion, s[i], b[i], targets[i]) for i in range(2)]
    module = Q._vendor_module(criterion)
    from sparse_rtdetr.localization import consumer_geometry as geo
    with geo.matching(criterion.matcher):
        indices = criterion.matcher({"pred_logits": s, "pred_boxes": b}, targets)["indices"]
    assert [x.tolist() for x, _ in indices] == [lab["matched_queries"] for lab in labels]
    new = s + torch.randn(s.shape, generator=torch.Generator().manual_seed(5)) * .3
    native = criterion.loss_labels_vfl({"pred_logits": new, "pred_boxes": b}, targets, indices,
                                       max(sum(len(t["labels"]) for t in targets), 1))["loss_vfl"]
    mine = Q.vfl_loss(new, torch.stack([x["classes"] for x in labels]), torch.stack([x["quality"] for x in labels]),
                      torch.stack([x["keep"] for x in labels]), sum(x["num_gt"] for x in labels),
                      criterion.alpha, criterion.gamma, criterion.weight_dict["loss_vfl"])
    assert torch.allclose(native, mine, rtol=1e-6, atol=0)
    assert any(not x["keep"].all() for x in labels)  # ignore mask is exercised
    assert all((x["classes"] < 10).sum() == len(x["matched_queries"]) for x in labels)
    assert module is not None


def _batch(models, arm, seed):
    h, l, criterion = models
    xh, xl = _pair(seed)
    with torch.inference_mode():
        out, entry, z = Q.h_forward(h, xh)
        zl, _, _ = Q.cross_readout(h, l, entry, xl)
    targets = _fixture_targets(2, seed)
    labels = [Q.query_labels(criterion, out["pred_logits"][i], out["pred_boxes"][i], targets[i]) for i in range(2)]
    z, zl = z.clone(), zl.clone()
    return {"s_H": out["pred_logits"].clone(), "b_H": out["pred_boxes"].clone(), "z_H": z,
            "z_source": Q.source_features(arm, z, zl),
            "classes": torch.stack([x["classes"] for x in labels]), "quality": torch.stack([x["quality"] for x in labels]),
            "keep": torch.stack([x["keep"] for x in labels]), "num_gt": sum(x["num_gt"] for x in labels)}


def test_head_steps_detector_frozen_and_resume(models):
    h, l, criterion = models
    recipe = {"alpha": criterion.alpha, "gamma": criterion.gamma, "loss_weight": criterion.weight_dict["loss_vfl"]}
    dh, dl = _state_digest(h), _state_digest(l)
    init = Q.initial_state()
    for arm in Q.ARMS:
        b1, b2 = _batch(models, arm, 6), _batch(models, arm, 7)
        head = Q.new_head(init)
        opt = Q.new_optimizer(head)
        r1 = Q.head_step(head, opt, b1, recipe)
        saved = copy.deepcopy(Q.head_state(head, opt, {"step": 1}, {"torch_cpu": torch.get_rng_state()}))
        r2 = Q.head_step(head, opt, b2, recipe)
        final = (Q.head_digest(head), Q.optimizer_digest(opt))
        again = Q.new_head(saved["head"])
        opt2 = Q.new_optimizer(again)
        opt2.load_state_dict(saved["optimizer"])
        r2b = Q.head_step(again, opt2, b2, recipe)
        assert (Q.head_digest(again), Q.optimizer_digest(opt2)) == final and r2b == r2
        assert r1["loss"] > 0 and Q.head_digest(head) != Q.digest_tensors(init.items())
        assert all(p.grad is None for p in h.parameters()) and all(p.grad is None for p in l.parameters())
    assert _state_digest(h) == dh and _state_digest(l) == dl
    assert all(not p.requires_grad for p in h.parameters()) and not h.training


def test_arms_differ_only_in_second_feature(models):
    b1, b2 = _batch(models, "B1", 8), _batch(models, "B2", 8)
    for k in ("s_H", "b_H", "z_H", "classes", "quality", "keep"):
        assert torch.equal(b1[k], b2[k])
    assert b1["num_gt"] == b2["num_gt"] and torch.equal(b1["z_source"], b1["z_H"])
    assert not torch.equal(b2["z_source"], b2["z_H"])
