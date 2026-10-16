"""Atomic complete common R18 state with external SHA and exact post-restore audit.

Reuses the tested model/optimizer layout, safe serialization and RNG adapters.
Old checkpoint formats and cross-scope short-window states are rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from ..baseline import training_v2b_checkpoint as old
from .engine import assert_healthy

FORMAT = "sparse_rtdetr.native_r18_common_fullstate_v1"


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()


def capture_rng(device):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"algorithm": str(state[0]),
            "keys": torch.tensor(state[1].astype(np.int64)), "position": int(state[2]),
            "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None}


def validate_rng(state, device):
    if not isinstance(state,dict) or set(state)!={"python","numpy","torch_cpu","cuda"}:
        raise ValueError("RNG state schema mismatch")
    random.Random().setstate(state["python"])
    numpy_state = old._numpy_state(state["numpy"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    if (state["cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("CPU/CUDA RNG scope mismatch")
    if device.type == "cuda":
        if state["cuda"].dtype != torch.uint8 or state["cuda"].shape != torch.cuda.get_rng_state(device).shape:
            raise ValueError("CUDA RNG state shape mismatch")
        torch.Generator(device=device).set_state(state["cuda"])
    return numpy_state


def restore_rng(state, device):
    numpy_state=validate_rng(state,device)
    random.setstate(state["python"]); np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda": torch.cuda.set_rng_state(state["cuda"], device)


def behavior_identity(engine):
    from ..baseline.training_v2b import validate_model_sampling,_optimizer
    from .geometry import validate_geometry,geometry_identity
    from .model import COMMIT
    identities={}
    for label,model in (("model",engine.model),("ema",engine.ema.module)):
        validate_geometry(model)
        if "_get_decoder_input" in model.decoder.__dict__:
            raise ValueError("active transient DN observer cannot be checkpointed")
        identities[label]={"sampling":validate_model_sampling(model,sampling_backend=engine.config.sampling_backend),
                           "geometry":geometry_identity(model),"official_commit":COMMIT,
                           "decoder_class":type(model.decoder).__module__+"."+type(model.decoder).__name__}
    if identities["model"]!=identities["ema"]:raise ValueError("native model/EMA behavior differs")
    identities["criterion"]=engine.criterion.recipe()
    expected={"weights":{"loss_vfl":1,"loss_bbox":5,"loss_giou":2},"alpha":.75,"gamma":2.,"losses":["vfl","boxes"],
        "share_matched_indices":False,"boxes_weight_format":None,
        "matcher":{"cost_class":2,"cost_bbox":5,"cost_giou":2,"alpha":.25,"gamma":2.,"use_focal_loss":True},
        "ignore":"raw_unmatched_negatives_normal_aux_encoder_only; native_DN_unchanged",
        "normalization":"native_GT_count_per_physical_batch; DN_times_groups; empty_min1",
        "loss_weight_application":"native_criterion_once; engine_sum_all"}
    if identities["criterion"]!=expected:raise ValueError("native criterion recipe differs")
    if old._optimizer_layout(engine.model,engine.optimizer)!=old._optimizer_layout(engine.model,_optimizer(engine.model,engine.config)):
        raise ValueError("native optimizer recipe differs")
    return identities


def capture(engine, binding):
    if binding.get("scope") not in {"cpu_R18_synthetic", "gpu_R18_acceptance", f"formal_R18_seed{engine.config.seed}_e120"}:
        raise ValueError("unknown common R18 state scope")
    if binding.get("arm")!="RT-DETRv2-R18-common-protocol-v1":
        raise ValueError("checkpoint is not the common R18 arm")
    if binding.get("config")!=engine.config.binding() or binding.get("model",{}).get("config")!=engine.config.binding():
        raise ValueError("checkpoint binding/config mismatch")
    engine.config.validate(synthetic=binding["scope"]=="cpu_R18_synthetic")
    if binding["scope"]!="cpu_R18_synthetic":
        if engine.device.type!="cuda" or not binding.get("runtime",{}).get("gpu_uuid"):
            raise ValueError("real common R18 state requires admitted CUDA identity")
        for key in ("source_sha256","contract_sha256","data_sha256","pretrained_sha256"):
            value=binding.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("real common R18 state is missing source/input binding: "+key)
    assert_healthy(engine.model, engine.optimizer, engine.ema)
    if any(p.grad is not None for p in engine.model.parameters()):
        raise ValueError("checkpoint boundary must have cleared gradients")
    state = {"format": FORMAT, "binding": binding, "binding_sha256": hashlib.sha256(canonical(binding)).hexdigest(),
             "torch_version": str(torch.__version__), "device": str(engine.device),
             "behavior": behavior_identity(engine), "model": engine.model.state_dict(),
             "model_layout": old._model_layout(engine.model),
             "model_training": {n: m.training for n,m in engine.model.named_modules()},
             "optimizer": engine.optimizer.state_dict(), "optimizer_layout": old._optimizer_layout(engine.model, engine.optimizer),
             "ema": engine.ema.state_dict(), "ema_layout": old._model_layout(engine.ema.module),
             "engine": engine.state_dict(), "rng": capture_rng(engine.device)}
    return old._safe(state)


def save(engine, binding, path):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix+".complete.json")
    temporary = sidecar.with_suffix(sidecar.suffix+".partial")
    if any(p.exists() for p in (path,path.with_suffix(path.suffix+".partial"),sidecar,temporary)):
        raise FileExistsError("refusing to overwrite common R18 state")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = capture(engine, binding)
    partial = path.with_suffix(path.suffix+".partial")
    with partial.open("xb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    sha = file_sha(partial)
    from .runio import fsync_directory
    os.link(partial,path);fsync_directory(path.parent)
    partial.unlink();fsync_directory(path.parent)
    receipt = {"schema": FORMAT, "path": str(path), "sha256": sha, "size_bytes": path.stat().st_size,
               "binding_sha256": payload["binding_sha256"], "scope": binding["scope"],
               "completed_updates": payload["engine"]["completed_updates"],
               "written_at": datetime.now(timezone.utc).isoformat()}
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary,sidecar);fsync_directory(path.parent)
    temporary.unlink();fsync_directory(path.parent)
    return receipt


def read(path, expected_sha, expected_binding):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("binding") != expected_binding or payload.get("binding_sha256") != hashlib.sha256(canonical(expected_binding)).hexdigest():
        raise ValueError("wrong checkpoint format, arm, scope or binding")
    return payload


def _restore(engine, binding, path, expected_sha):
    payload = read(path, expected_sha, binding)
    expected_keys = set(capture(engine, binding))
    if set(payload) != expected_keys or payload["device"] != str(engine.device) or payload["torch_version"] != str(torch.__version__):
        raise ValueError("complete checkpoint schema/runtime mismatch")
    engine.validate_state_dict(payload["engine"])
    if behavior_identity(engine) != payload["behavior"]:
        raise ValueError("checkpoint non-serialized behavior mismatch")
    model_layout, ema_layout = old._model_layout(engine.model), old._model_layout(engine.ema.module)
    if payload["model_layout"] != model_layout or payload["ema_layout"] != ema_layout:
        raise ValueError("checkpoint model/EMA layout mismatch")
    old._check_model_state(payload["model"], model_layout, "model")
    old._check_model_state(payload["ema"]["module"], ema_layout, "ema")
    modes=payload["model_training"]
    if not isinstance(modes,dict) or set(modes)!={n for n,_ in engine.model.named_modules()} or any(type(v) is not bool for v in modes.values()):
        raise ValueError("saved model training modes mismatch")
    if set(payload["ema"])!={"module","updates","decay","warmups"} or payload["ema"]["decay"]!=engine.ema.decay or payload["ema"]["warmups"]!=engine.ema.warmups or type(payload["ema"]["updates"]) is not int:
        raise ValueError("saved EMA metadata mismatch")
    for name in ("model","ema"):
        state=payload["model"] if name=="model" else payload["ema"]["module"]
        fresh=(engine.model if name=="model" else engine.ema.module).state_dict()
        for key,value in fresh.items():
            if key.rsplit(".",1)[-1] in {"anchors","valid_mask","num_points_scale"} and not old._equal(state[key],value):
                raise ValueError("saved native immutable cache differs: "+key)
    validate_rng(payload["rng"],engine.device)
    current_layout = old._optimizer_layout(engine.model, engine.optimizer)
    if payload["optimizer_layout"] != current_layout:
        raise ValueError("optimizer parameter layout mismatch")
    old._check_optimizer(payload["optimizer"], current_layout, engine.model, payload["engine"]["completed_updates"])
    saved_steps=payload["engine"]["parameter_optimizer_steps"]
    for group in current_layout["groups"]:
        for n,param_id in zip(group["names"],group["ids"]):
            actual=int(payload["optimizer"]["state"].get(param_id,{}).get("step",0))
            if actual!=saved_steps[n]:raise ValueError("common R18 exact optimizer step ledger differs: "+n)
    if payload["ema"]["updates"] != payload["engine"]["completed_updates"]:
        raise ValueError("EMA/optimizer clocks disagree")
    from .engine import learning_rate
    expected_lr=learning_rate(payload["engine"]["completed_updates"]) if payload["engine"]["completed_updates"] else engine.config.learning_rate
    if any(g["lr"]!=expected_lr for g in payload["optimizer"]["param_groups"]):
        raise ValueError("saved LR is not the complete update LR")
    engine.model.load_state_dict(payload["model"], strict=True)
    for name, module in engine.model.named_modules(): module.training = payload["model_training"][name]
    engine.optimizer.load_state_dict(payload["optimizer"])
    engine.ema.load_state_dict(payload["ema"])
    engine.load_state_dict(payload["engine"])
    restore_rng(payload["rng"], engine.device)
    restored = capture(engine, binding)
    if not old._equal(payload, restored):
        raise RuntimeError("full restored state is not exactly equal")
    return {"full_state_exact": True, "tensor_count": sum(1 for _ in _tensors(payload)),
            "completed_updates": engine.completed_updates, "checkpoint_sha256": expected_sha}


def restore(engine, binding, path, expected_sha):
    try:
        return _restore(engine,binding,path,expected_sha)
    except BaseException:
        # A rejected or partially restored engine cannot perform another update.
        engine.failed=True
        raise


def _tensors(value):
    if isinstance(value, torch.Tensor): yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from _tensors(item)
    elif isinstance(value, (list,tuple)):
        for item in value: yield from _tensors(item)


def exact_equal(left, right): return old._equal(left, right)
