"""TASK_R27_L0_CONTROL_002 synthetic unit checks of the L0 wrapper (CPU, randomly initialised models, no data).

Every forward in this file is a synthetic unit-test forward.  None of them belongs to any formal ledger (training
updates, evaluation passes, observations or timing).
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r27_local_memory as R  # noqa: E402
from sparse_rtdetr.research import scale_baseline as S  # noqa: E402
from sparse_rtdetr.protocol_r18.engine import output_digest, tensor_digest  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
PINNED_IDENTITY = "136401a290c57de2f3ed69115e73159910f3a30ec17581f4b6a4019e90b5a0f8"


@pytest.fixture(scope="module")
def base():
    """Native RT-DETRv2-R18 at L with random weights; tests only read it (mutating tests work on deep copies)."""
    model, _, _ = build(Config(seed=0), synthetic=True)
    return model.eval()


def _wrapped(base):
    model = copy.deepcopy(base)
    L0.wrap(model)
    return model


def _image(seed, batch=1):
    return torch.rand(batch, 3, *R.H_HW, generator=torch.Generator().manual_seed(seed))


def _layout(model):
    return {"state": list(model.state_dict()), "parameters": [(n, tuple(p.shape)) for n, p in model.named_parameters()],
            "buffers": [n for n, _ in model.named_buffers()], "modules": [n for n, _ in model.named_modules()],
            "count": sum(p.numel() for p in model.parameters()), "digest": tensor_digest(model.state_dict().items())}


# ---------------------------------------------------------------- identity
def test_identity_is_stable_and_is_not_an_r27_identity():
    canonical = json.dumps(L0.IDENTITY, sort_keys=True, separators=(",", ":")).encode()
    assert L0.IDENTITY_SHA256 == hashlib.sha256(canonical).hexdigest() == PINNED_IDENTITY
    assert L0.IDENTITY_SHA256 != R.IDENTITY_SHA256 and L0.VERSION != R.VERSION
    assert L0.ARM == "L0" and L0.ARM not in R.ARMS and L0.ARM not in ("N", "A0", S.ARM)
    assert L0.IDENTITY["H_to_L"] is R.IDENTITY["H_to_L"] and L0.to_low is R.to_low
    assert L0.IDENTITY["H"] == [1152, 2016] and L0.IDENTITY["L"] == [768, 1344]
    assert L0.IDENTITY["anchor_position_cache_canvas"] == [768, 1344]
    assert all(L0.IDENTITY[k] is None for k in ("local_backbone", "local_projection", "local_read", "selector",
                                                 "route_keys", "local_recompute"))
    assert L0.IDENTITY["new_learnable_parameters"] == 0
    assert L0.IDENTITY["activation_checkpoint_segments"] == list(S.CHECKPOINT_SEGMENTS)
    assert L0.IDENTITY["parent_sha256"] == R.PARENT_SHA == S.PARENT_SHAS[0] and L0.IDENTITY["parent_update"] == 18240
    assert set(L0.IDENTITY["is_not"]) == {"R27_N", "R27_P_branch_off", "A0_direct_L"}


# ---------------------------------------------------------------- wrapper structure
def test_wrap_keeps_native_names_and_adds_nothing(base):
    model = copy.deepcopy(base)
    before = _layout(model)
    assert not L0.is_wrapped(model)
    info = L0.wrap(model)
    assert _layout(model) == before
    assert L0.is_wrapped(model) and isinstance(model, type(base)) and type(model) is not type(base)
    assert info["arm"] == "L0" and info["new_parameters"] == 0
    assert (info["state_keys"], info["parameter_tensors"]) == (len(before["state"]), len(before["parameters"]))
    assert info["native"]["canvas"] == [768, 1344] and info["native"]["queries"] == 300
    assert info["native"]["decoder_layers"] == 3
    assert not any(R.is_new(k) for k in model.state_dict())
    assert not hasattr(model, "r27_ctx") and not hasattr(model, "r27_arm") and R.arm_of(model) is None
    copy.deepcopy(base).load_state_dict(model.state_dict(), strict=True)  # still a native state


def test_double_wrap_is_refused_also_on_a_copy(base):
    model = _wrapped(base)
    with pytest.raises(ValueError, match="already wrapped"):
        L0.wrap(model)
    clone = copy.deepcopy(model)
    assert L0.is_wrapped(clone)
    with pytest.raises(ValueError, match="already wrapped"):
        L0.wrap(clone)


def test_r27_installed_model_is_refused_and_r27_on_top_is_not_l0(base):
    for arm in R.ARMS:
        model = copy.deepcopy(base)
        R.install(model, arm)
        assert not L0.is_wrapped(model)
        with pytest.raises(ValueError, match="R27 installed"):
            L0.wrap(model)
    stacked = _wrapped(base)
    R.install(stacked, "P")
    assert not L0.is_wrapped(stacked)


def test_h_canvas_model_and_state_are_refused(base):
    """The R27 N arm is the native network on the H canvas (S15); it is neither wrappable nor loadable as L0."""
    model, _, _ = build(S.production_config(0), synthetic=True)
    with pytest.raises(ValueError, match="native identity differs"):
        L0.wrap(model)
    assert not L0.is_wrapped(model)
    assert set(model.state_dict()) == set(base.state_dict())
    with pytest.raises(RuntimeError, match="size mismatch"):
        L0.build_eval_model(model.state_dict(), "cpu")


# ---------------------------------------------------------------- forward (synthetic unit-test forwards)
def test_eval_output_is_the_native_forward_of_to_low_bit_exact(base):
    model = _wrapped(base)
    x = _image(1, batch=2)
    with torch.no_grad():
        ref = base(R.to_low(x))
        out = model(x)
    assert set(out) == set(ref) == {"pred_logits", "pred_boxes"}
    assert tuple(out["pred_logits"].shape) == (2, 300, 10) and tuple(out["pred_boxes"].shape) == (2, 300, 4)
    assert all(torch.equal(out[k], ref[k]) for k in ref)
    assert output_digest(out) == output_digest(ref)


def test_train_forward_passes_targets_and_updates_bn_once_like_native(base):
    native, model = copy.deepcopy(base).train(), _wrapped(base).train()
    x = _image(2)
    targets = [{"labels": torch.tensor([0, 3, 9]),
                "boxes": torch.tensor([[.2, .3, .1, .1], [.5, .5, .2, .3], [.8, .7, .05, .08]])}]
    torch.manual_seed(7)
    ref = native(R.to_low(x), copy.deepcopy(targets))
    torch.manual_seed(7)
    out = model(x, copy.deepcopy(targets))
    assert "dn_meta" in ref and "aux_outputs" in ref  # the targets reached the native denoising path
    assert output_digest(out) == output_digest(ref)
    after = tensor_digest(model.state_dict().items())
    assert after == tensor_digest(native.state_dict().items())
    assert after != tensor_digest(base.state_dict().items())  # BN statistics moved, identically in both


def test_non_h_or_non_fp32_input_is_refused(base):
    model = _wrapped(base)
    bad = [torch.zeros(1, 3, *R.L_HW),  # a tensor already at L (the historical direct-L input shape)
           torch.zeros(1, 3, 1152, 2015), torch.zeros(1, 3, 2016, 1152), torch.zeros(3, *R.H_HW),
           torch.zeros(1, 3, *R.H_HW, dtype=torch.float64), torch.zeros(1, 3, *R.H_HW, dtype=torch.float16),
           torch.zeros(1, 3, *R.H_HW, dtype=torch.bfloat16)]
    for x in bad:
        with torch.no_grad(), pytest.raises(ValueError, match="expects FP32 H input"):
            model(x)


# ---------------------------------------------------------------- evaluation model
def test_build_eval_model_accepts_only_the_native_key_set(base):
    state = {k: v.clone() for k, v in base.state_dict().items()}
    model = L0.build_eval_model(state, "cpu")
    assert L0.is_wrapped(model) and not model.training and not any(p.requires_grad for p in model.parameters())
    assert not any(getattr(m, "_s15_checkpointed", False) for m in model.modules())
    assert tensor_digest(model.state_dict().items()) == tensor_digest(base.state_dict().items())
    # Like for like: the evaluation model has requires_grad=False.  On this torch build the native output bits
    # depend on that flag even under no_grad (observed on the unwrapped model), so the reference carries it too.
    ref = copy.deepcopy(base).requires_grad_(False)
    x = _image(3)
    with torch.no_grad():
        assert output_digest(model(x)) == output_digest(ref(R.to_low(x)))
    for arm in R.ARMS:  # an R27-shaped state carries the local keys
        r27 = copy.deepcopy(base)
        R.install(r27, arm)
        extra = r27.state_dict()
        assert set(extra) > set(state) and any(R.is_new(k) for k in extra)
        with pytest.raises(ValueError, match="native key set"):
            L0.build_eval_model(extra, "cpu")
    for key in (next(iter(state)), "decoder.anchors", list(state)[-1]):
        missing = {k: v for k, v in state.items() if k != key}
        assert len(missing) == len(state) - 1
        with pytest.raises(ValueError, match="native key set"):
            L0.build_eval_model(missing, "cpu")
