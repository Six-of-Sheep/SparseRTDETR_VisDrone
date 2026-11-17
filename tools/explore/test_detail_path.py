"""CPU unit tests for the sub-stride detail path (frozen release source).

Run with the real CLI in the frozen interpreter with CUDA hidden:
  CUDA_VISIBLE_DEVICES= python -I tools/explore/test_detail_path.py --source-root <release>/source
Under pytest the same functions run when the RELEASE source root is provided via
the ``--source-root`` pytest option or the ``DETAIL_RELEASE_SRC`` env variable.

Covered: parameter delta and key names; zero-gate bitwise identity in both
modes plus nonzero-gate output change; deepcopy self-reference; strict
state_dict roundtrip; the real loss path (21 terms incl. DN) with gradient
behaviour at gate 0; installation RNG-neutrality; zero-gate shared-parameter
gradients bitwise identical to the native model; and the nonzero-gate
stop-gradient branch check (source detached, gate+conv train, main path live).
"""
from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import detail_path  # noqa: E402
import frozen_r18  # noqa: E402

RELEASE_SRC = None  # set via --source-root / DETAIL_RELEASE_SRC in _configure().


def _configure(release_src):
    global RELEASE_SRC
    RELEASE_SRC = str(release_src)
    frozen_r18.require_release_source(RELEASE_SRC)
    torch.set_num_threads(2)


def _build(mode):
    from sparse_rtdetr.protocol_r18.model import Config, build
    conf = Config(seed=0, input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    model, criterion, info = build(conf, pretrained_path=None, synthetic=True)
    if mode is not None:
        detail_path.install_detail_path(model, mode)
    return model, criterion


def _probe_input(seed=713):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 128, 128, generator=generator)


def _forward_pair(model):
    model.eval()
    with torch.no_grad():
        out = model(_probe_input())
    return out


def _targets():
    return [
        {"boxes": torch.tensor([[.4, .4, .15, .2]]), "labels": torch.tensor([1]),
         "gt_ids": ["fixture/raw0"],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "gt_ids": [],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
    ]


def _np_state_equal(a):
    b = np.random.get_state()
    return a[0] == b[0] and (a[1] == b[1]).all() and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]


def test_parameter_delta_and_key_names():
    native, _ = _build(None)
    detail, _ = _build("detail")
    native_keys = set(native.state_dict())
    detail_keys = set(detail.state_dict())
    added = detail_keys - native_keys
    assert added == {"decoder.detail_inject.conv.weight", "decoder.detail_inject.gate"}, added
    assert not (native_keys - detail_keys)
    delta = sum(p.numel() for p in detail.parameters()) - sum(p.numel() for p in native.parameters())
    assert delta == detail_path.DETAIL_PARAMETERS
    detail_path.assert_installed(detail, "detail")


def test_zero_gate_identity_bitwise_both_modes():
    native, _ = _build(None)
    native_out = _forward_pair(native)
    for mode in ("detail", "control"):
        model, _ = _build(mode)
        out = _forward_pair(model)
        for key in ("pred_logits", "pred_boxes"):
            assert torch.equal(out[key], native_out[key]), (mode, key)
        # and a nonzero gate must actually change the outputs
        with torch.no_grad():
            model.decoder.detail_inject.gate.fill_(0.01)
        changed = _forward_pair(model)
        diff = max(float((changed[k] - native_out[k]).abs().max()) for k in ("pred_logits", "pred_boxes"))
        assert diff > 0.0, mode
        with torch.no_grad():
            model.decoder.detail_inject.gate.zero_()


def test_deepcopy_self_reference():
    model, _ = _build("detail")
    clone = copy.deepcopy(model)
    detail_path.assert_installed(clone, "detail")
    assert clone.decoder is not model.decoder
    assert clone.backbone is not model.backbone
    # outputs of the clone equal the original while gate is zero
    a = _forward_pair(model)
    b = _forward_pair(clone)
    for key in ("pred_logits", "pred_boxes"):
        assert torch.equal(a[key], b[key])


def test_state_dict_roundtrip_strict():
    model, _ = _build("control")
    with torch.no_grad():
        model.decoder.detail_inject.gate.fill_(0.02)
        model.decoder.detail_inject.conv.weight.mul_(0.5)
    state = model.state_dict()
    fresh, _ = _build("control")
    fresh.load_state_dict(state, strict=True)
    a = _forward_pair(model)
    b = _forward_pair(fresh)
    for key in ("pred_logits", "pred_boxes"):
        assert torch.equal(a[key], b[key])


def test_loss_path_and_gradients():
    from sparse_rtdetr.protocol_r18.model import Config  # noqa: F401 (release import check)
    model, criterion = _build("detail")
    model.train()
    images = _probe_input(11).expand(2, 3, 128, 128).contiguous()
    targets = _targets()
    outputs = model(images, targets)
    losses = criterion(outputs, targets)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    # the transient capture is consumed and cleared by the same forward (layout-invisible)
    assert model.backbone._detail_capture.get("s4") is None
    model.backbone(images)
    held = model.backbone._detail_capture.get("s4")
    assert held is not None and not held.requires_grad and held.grad_fn is None
    gate = model.decoder.detail_inject.gate
    conv = model.decoder.detail_inject.conv.weight
    assert gate.grad is not None and torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0
    # at gate == 0 the conv receives exactly zero gradient (documented behavior)
    assert conv.grad is None or float(conv.grad.abs().max()) == 0.0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        gate.fill_(0.02)
    outputs = model(images, targets)
    total = sum(criterion(outputs, targets).values())
    total.backward()
    assert conv.grad is not None and torch.isfinite(conv.grad).all() and conv.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)


def test_install_rng_neutral():
    model, _ = _build(None)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    s_py = random.getstate()
    s_np = np.random.get_state()
    s_t = torch.get_rng_state()
    detail_path.install_detail_path(model, "control")
    assert random.getstate() == s_py
    assert _np_state_equal(s_np)
    assert torch.equal(torch.get_rng_state(), s_t)
    # a fixed draw sequence is identical before and after installation
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    fresh, _ = _build(None)
    detail_path.install_detail_path(fresh, "detail")
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    after = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert before == after


def test_zero_gate_shared_gradients_bitwise_vs_native():
    images = _probe_input(11).expand(2, 3, 128, 128).contiguous()
    targets = _targets()
    for mode in ("detail", "control"):
        native, crit_n = _build(None)
        model, crit_m = _build(mode)
        native.train()
        model.train()
        torch.manual_seed(99)
        out_n = native(images, targets)
        torch.manual_seed(99)
        out_m = model(images, targets)
        for key in ("pred_logits", "pred_boxes"):
            assert torch.equal(out_n[key], out_m[key]), (mode, key)
        torch.manual_seed(7)
        loss_n = sum(crit_n(out_n, targets).values())
        torch.manual_seed(7)
        loss_m = sum(crit_m(out_m, targets).values())
        assert torch.equal(loss_n, loss_m), mode
        loss_n.backward()
        loss_m.backward()
        grads_n = {n: p.grad for n, p in native.named_parameters()}
        compared = 0
        for n, p in model.named_parameters():
            if n not in grads_n:
                continue
            g = grads_n[n]
            assert (g is None) == (p.grad is None), (mode, n)
            if g is not None:
                assert torch.equal(g, p.grad), (mode, n)
                compared += 1
        assert compared > 300, (mode, compared)
        native.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)


def test_stop_gradient_and_nonzero_gate_paths():
    for mode in ("detail", "control"):
        model, criterion = _build(mode)
        model.train()
        images = _probe_input(11).expand(2, 3, 128, 128).contiguous()
        targets = _targets()
        with torch.no_grad():
            model.decoder.detail_inject.gate.fill_(0.02)
        outputs = model(images, targets)
        info = dict(detail_path.LAST_SOURCE_INFO)
        assert info["mode"] == mode
        assert info["source_requires_grad"] is False, mode
        if mode == "control":
            # the live training forward had a grad-requiring main-path tensor
            assert info["source_requires_grad_before_branch_detach"] is True, mode
        else:
            # the stride-4 capture is value-only by construction (detached at capture)
            assert info["source_requires_grad_before_branch_detach"] is False, mode
        assert info["shape"][-2:] == [16, 16]  # stride-4 grid for 128px input
        total = sum(criterion(outputs, targets).values())
        total.backward()
        gate = model.decoder.detail_inject.gate
        conv = model.decoder.detail_inject.conv.weight
        assert gate.grad is not None and torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0, mode
        assert conv.grad is not None and torch.isfinite(conv.grad).all() and conv.grad.abs().sum() > 0, mode
        shared = [p.grad for n, p in model.named_parameters() if not n.startswith("decoder.detail_inject")]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in shared), mode
        model.zero_grad(set_to_none=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    _configure(args.source_root)
    failures = []
    for name, fn in sorted((n, f) for n, f in globals().items()
                           if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name, flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append((name, repr(error)))
            print("FAIL", name, repr(error), flush=True)
    if failures:
        raise SystemExit("FAILURES: %d" % len(failures))
    print("ALL-OK")


if __name__ == "__main__":
    main()
