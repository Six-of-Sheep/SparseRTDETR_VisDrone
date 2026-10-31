"""CPU unit tests for the size-relative matching cost (frozen release source).

Run in the frozen interpreter with CUDA hidden, from the release source root:
  CUDA_VISIBLE_DEVICES= <python> -I tools/matcher_scale/test_matcher_scale.py --source-root .
The tests build small synthetic-config models through the real
``protocol_r18.model.build`` (no pretrained weights, no data) and never touch the
frozen checkpoints.  ``--out`` optionally writes a JSON evidence file.
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

matcher_scale = None  # bound by _configure() from the staged release package

RELEASE_SRC = None
EVIDENCE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _configure(source_root):
    global RELEASE_SRC, matcher_scale
    RELEASE_SRC = str(source_root)
    sys.path.insert(0, str(Path(RELEASE_SRC) / "src"))
    import sparse_rtdetr
    resolved = Path(sparse_rtdetr.__file__).resolve()
    require(str(resolved).startswith(str(Path(RELEASE_SRC) / "src")),
            "sparse_rtdetr resolved outside the release: " + str(resolved))
    from sparse_rtdetr.protocol_r18 import matcher_scale as module
    matcher_scale = module
    torch.set_num_threads(2)


def _build(mode=None):
    from sparse_rtdetr.protocol_r18.model import Config, build
    conf = Config(seed=0, input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    model, criterion, info = build(conf, pretrained_path=None, synthetic=True)
    if mode is not None:
        matcher_scale.install_matcher_scale(criterion, mode)
    return model, criterion


def _vendor_matcher(criterion):
    module = importlib.import_module(type(criterion.matcher).__module__)
    matcher_scale.bind_vendor_names(module)
    return module, module.HungarianMatcher({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                                           use_focal_loss=True, alpha=0.25, gamma=2.0)


WEIGHTS = {"cost_bbox": 5.0, "cost_class": 2.0, "cost_giou": 2.0, "alpha": 0.25, "gamma": 2.0}


def _case(seed, queries, gts, labels=None):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, queries, 10, generator=gen)
    boxes = torch.rand(1, queries, 4, generator=gen) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:].clamp_min(1e-3)
    gt_boxes = torch.rand(gts, 4, generator=gen) * 0.5 + 0.25
    gt_boxes[..., 2:] = gt_boxes[..., 2:].clamp_min(1e-3)
    gt_labels = torch.randint(0, 10, (gts,), generator=gen) if labels is None else labels
    targets = [{"labels": gt_labels, "boxes": gt_boxes}]
    return {"pred_logits": logits, "pred_boxes": boxes}, targets


def _index_pairs(indices):
    return [(int(i), int(j)) for i, j in zip(indices[0][0].tolist(), indices[0][1].tolist())]


def test_recipe_and_matcher_fields():
    _, native = _build(None)
    recipe = native.recipe()
    require(recipe["matching"] == matcher_scale.NATIVE_DECLARATION, "native matching declaration differs")
    require(recipe["matcher"] == {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2, "alpha": 0.25,
                                  "gamma": 2.0, "use_focal_loss": True},
            "native matcher fields changed")
    for mode in matcher_scale.MODES:
        _, criterion = _build(mode)
        declared = criterion.recipe()
        require(declared["matching"] == matcher_scale.declaration(mode), "declared matching differs")
        require(declared["matcher"] == recipe["matcher"], "installed matcher fields changed")
        require(declared["weights"] == recipe["weights"], "installed loss weights changed")
        matcher_scale.assert_installed(criterion, mode)
        EVIDENCE.setdefault("declarations", {})[mode] = declared["matching"]
    EVIDENCE["declarations"]["native"] = matcher_scale.NATIVE_DECLARATION


def _transcribed_native_costs(outputs, targets):
    """Independent transcription of the frozen matcher's arithmetic (no helper reuse)."""
    out_prob = torch.sigmoid(outputs["pred_logits"].flatten(0, 1))
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])
    out_prob = out_prob[:, tgt_ids]
    neg = (1 - 0.25) * (out_prob ** 2.0) * (-(1 - out_prob + 1e-8).log())
    pos = 0.25 * ((1 - out_prob) ** 2.0) * (-(out_prob + 1e-8).log())
    cost_class = pos - neg
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    _, criterion = _build(None)
    vendor = importlib.import_module(type(criterion.matcher).__module__)
    cost_giou = -vendor.generalized_box_iou(vendor.box_cxcywh_to_xyxy(out_bbox),
                                            vendor.box_cxcywh_to_xyxy(tgt_bbox))
    C = 5.0 * cost_bbox + 2.0 * cost_class + 2.0 * cost_giou
    return cost_class, cost_bbox, cost_giou, C


def test_off_mode_is_bitwise_the_native_expression():
    _, criterion = _build(None)
    _vendor_matcher(criterion)
    compared = 0
    for seed, queries, gts in ((1, 300, 40), (2, 300, 7), (3, 17, 3), (4, 300, 1)):
        outputs, targets = _case(seed, queries, gts)
        cls_, bbox_, giou_, C = _transcribed_native_costs(outputs, targets)
        got_cls, got_bbox, got_giou, got_C, _ = matcher_scale.cost_terms(outputs, targets, "off", WEIGHTS)
        require(torch.equal(cls_, got_cls) and torch.equal(bbox_, got_bbox)
                and torch.equal(giou_, got_giou) and torch.equal(C, got_C),
                "off-mode cost differs from the transcribed native expression (seed %d)" % seed)
        compared += 1
    EVIDENCE["off_mode_bitwise_cases"] = compared


def test_assignment_matches_the_real_vendor_matcher():
    _, criterion = _build(None)
    _, real = _vendor_matcher(criterion)
    compared = 0
    for seed, queries, gts in ((11, 300, 40), (12, 300, 1), (13, 300, 5), (14, 7, 2), (15, 300, 90)):
        outputs, targets = _case(seed, queries, gts)
        expected = _index_pairs(real(outputs, targets)["indices"])
        got, _, _ = matcher_scale.assignment(outputs, targets, WEIGHTS, "off")
        require(_index_pairs(got) == expected, "off-mode assignment differs from the real matcher (seed %d)" % seed)
        compared += 1
    EVIDENCE["assignment_identity_cases"] = compared


def test_control_reduces_to_method_when_sizes_are_equal():
    gen = torch.Generator().manual_seed(21)
    logits = torch.randn(1, 40, 10, generator=gen)
    boxes = torch.rand(1, 40, 4, generator=gen) * 0.4 + 0.3
    same = torch.tensor([[0.30, 0.30, 0.05, 0.05], [0.70, 0.70, 0.05, 0.05]])
    targets = [{"labels": torch.tensor([1, 1]), "boxes": same}]
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    _, _, _, C_rel, _ = matcher_scale.cost_terms(outputs, targets, "rel", WEIGHTS)
    _, _, _, C_ctrl, _ = matcher_scale.cost_terms(outputs, targets, "ctrl", WEIGHTS)
    require(torch.equal(C_rel, C_ctrl), "control differs from method although all GT sizes are equal")
    EVIDENCE["control_equals_method_on_equal_sizes"] = True


def test_method_changes_ownership_on_a_constructed_case():
    """One small and one large GT, two tokens: absolute L1 is cheapest on the identity
    pairing; the size-relative cost must decide differently."""
    small = [0.30, 0.30, 0.02, 0.02]
    large = [0.31, 0.31, 0.20, 0.20]
    targets = [{"labels": torch.tensor([1, 1]), "boxes": torch.tensor([small, large])}]
    queries = torch.tensor([[[0.30, 0.30, 0.20, 0.20],      # big box centred on the small GT
                             [0.31, 0.31, 0.20, 0.20]]])    # exact box of the large GT
    logits = torch.zeros(1, 2, 10)
    outputs = {"pred_logits": logits, "pred_boxes": queries}
    off, _, _ = matcher_scale.assignment(outputs, targets, WEIGHTS, "off")
    rel, _, info = matcher_scale.assignment(outputs, targets, WEIGHTS, "rel")
    ctrl, _, _ = matcher_scale.assignment(outputs, targets, WEIGHTS, "ctrl")
    EVIDENCE["constructed_case"] = {"off": _index_pairs(off), "rel": _index_pairs(rel),
                                    "ctrl": _index_pairs(ctrl), "rel_scales": [info["scale_norm_min"],
                                                                               info["scale_norm_max"]]}
    require(_index_pairs(off) == [(0, 0), (1, 1)], "native assignment on the constructed case is not the identity")
    require(_index_pairs(rel) != _index_pairs(off),
            "the size-relative cost did not change the constructed-case ownership")


def test_gradients_and_parameter_keys_are_unchanged():
    _, criterion = _build("rel")
    model, _ = _build("rel")
    require([n for n, _ in model.named_parameters()] == [n for n, _ in _build(None)[0].named_parameters()],
            "parameter names changed by installation")
    images = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(31))
    targets = [
        {"boxes": torch.tensor([[.4, .4, .15, .2]]), "labels": torch.tensor([1]), "gt_ids": ["t0"],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "gt_ids": [],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
    ]
    model.train()
    out = model(images, targets)
    losses = criterion(out, targets)
    require(torch.isfinite(sum(losses.values())), "installed loss is not finite")
    sum(losses.values()).backward()
    grads = [(n, p.grad) for n, p in model.named_parameters()]
    require(all(g is not None and torch.isfinite(g).all() for _, g in grads), "some parameter has no finite gradient")
    require(any(float(g.abs().sum()) > 0 for _, g in grads), "all gradients are zero")
    EVIDENCE["gradient_parameters"] = len(grads)
    EVIDENCE["loss_keys"] = sorted(losses)
    require(len(losses) in (12, 21), "native loss count changed: %d" % len(losses))


def test_rng_neutral_and_geometry_consumer_patches_this_module():
    from sparse_rtdetr.localization import consumer_geometry as geo
    _, criterion = _build(None)
    vendor, _ = _vendor_matcher(criterion)
    before = torch.random.get_rng_state()
    _install_no_rng = matcher_scale.install_matcher_scale(criterion, "rel")
    require(torch.equal(before, torch.random.get_rng_state()), "installation consumed an RNG draw")
    require(matcher_scale.VENDOR_MODULE is vendor, "vendor module binding differs")
    original = matcher_scale.generalized_box_iou
    with geo.matching(criterion.matcher):
        patched = matcher_scale.generalized_box_iou
        require(patched is not original, "the geometry consumer did not patch this module's vendor name")
        require(matcher_scale.linear_sum_assignment is not vendor.linear_sum_assignment,
                "the geometry consumer did not patch this module's solver name")
    require(matcher_scale.generalized_box_iou is original, "the geometry consumer did not restore the vendor name")
    EVIDENCE["geometry_consumer_patch"] = True


def test_declarations_are_distinct_and_stable():
    require(matcher_scale.declaration("rel") != matcher_scale.declaration("ctrl"), "mode declarations coincide")
    require(matcher_scale.declaration("rel") != matcher_scale.NATIVE_DECLARATION, "method declaration equals native")
    require(matcher_scale.declaration(None) == matcher_scale.NATIVE_DECLARATION, "default declaration is not native")
    EVIDENCE["declaration_digest"] = matcher_scale.declared_digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    _configure(args.source_root)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.monotonic()
    failures = []
    names = sorted(n for n, f in globals().items() if n.startswith("test_") and callable(f))
    for name in names:
        try:
            globals()[name]()
            print("PASS", name, flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append((name, repr(error)))
            print("FAIL", name, repr(error), flush=True)
    payload = {"status": "PASS" if not failures else "FAIL", "tests": names,
               "failures": failures, "evidence": EVIDENCE,
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "cuda_initialized": bool(torch.cuda.is_initialized())}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": failures, "evidence": EVIDENCE}))
    if failures:
        raise SystemExit("FAILURES: %d" % len(failures))
    print("ALL-OK")


if __name__ == "__main__":
    main()
