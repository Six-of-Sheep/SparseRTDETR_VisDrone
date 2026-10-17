"""Two equal physical losses, one complete update, explicit update LR clock."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import time
import torch

from ..baseline.training_v2b_engine import _batchnorm_backward_layout, _move
from ..localization.distribution import require_finite
from .model import dn_capture,fp32_outputs
from .geometry import validate_geometry,forward_guards


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= 36480:
        raise ValueError("LR update is outside the fixed 120 epoch schedule")
    if update <= 2000:
        return 1e-4 * update / 2000.
    return 1e-6 + .5*(1e-4-1e-6)*(1+math.cos(math.pi*(update-2000)/(36480-2000)))


def tensor_digest(items):
    result = hashlib.sha256()
    for name, value in items:
        if value is None:
            result.update((name+":None").encode()); continue
        x = value.detach().cpu().contiguous()
        result.update((name+":"+str(x.dtype)+":"+str(tuple(x.shape))).encode())
        result.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def output_digest(value):
    """Hash every normal, auxiliary, encoder, pre and actual/padded DN output."""
    result = hashlib.sha256()
    def walk(item, path):
        if isinstance(item, torch.Tensor):
            result.update((path + ':tensor:' + tensor_digest([(path,item)])).encode())
        elif isinstance(item, dict):
            result.update((path + ':dict:' + json.dumps(sorted(item))).encode())
            for key in sorted(item):walk(item[key],path+'/'+key)
        elif isinstance(item, (tuple,list)):
            result.update((path + ':' + type(item).__name__ + ':' + str(len(item))).encode())
            for i, child in enumerate(item):walk(child,path+'/'+str(i))
        elif item is None or type(item) in (bool,int,float,str):
            result.update((path + ':' + type(item).__name__ + ':' + json.dumps(item,allow_nan=False)).encode())
        else:
            raise TypeError('unrecorded model output type: '+path+'/'+type(item).__name__)
    walk(value,'output')
    return result.hexdigest()


def assert_healthy(model, optimizer=None, ema=None, *, gradients=False):
    validate_geometry(model)
    if ema is not None: validate_geometry(ema.module)
    finite = []
    for label, component in (("model", model), ("ema", getattr(ema, "module", None))):
        if component is None: continue
        for name, value in component.state_dict().items():
            if not value.is_floating_point(): continue
            if name.rsplit(".", 1)[-1] == "anchors":
                finite.append(~(value.isnan() | value.isneginf()).any())
            else: finite.append(value.isfinite().all())
            if name.endswith("running_var"): finite.append((value >= 0).all())
    if gradients:
        finite.extend(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
    if optimizer is not None:
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    finite.append(value.isfinite().all())
                    if name in {"exp_avg_sq", "max_exp_avg_sq"}: finite.append((value >= 0).all())
    # Adam step scalars may be on CPU while the model is on CUDA.
    by_device = {}
    for value in finite: by_device.setdefault(str(value.device), []).append(value)
    if any(not bool(torch.stack(values).all()) for values in by_device.values()):
        raise RuntimeError("common R18 numerical/state health check failed")


class Engine:
    def next_learning_rate(self):
        """Override only in an explicitly bound new schedule/engine identity."""
        return learning_rate(self.completed_updates+1)

    def __init__(self, model, criterion, optimizer, ema, config):
        self.model, self.criterion, self.optimizer, self.ema = model, criterion, optimizer, ema
        self.config = config
        self.device = next(model.parameters()).device
        self.completed_updates = 0
        self.phase = 0
        self.failed = False
        self.parameter_steps={n:0 for n,p in model.named_parameters() if p.requires_grad}
        optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
        expected = [id(p) for p in model.parameters() if p.requires_grad]
        if len(optimized) != len(set(optimized)) or set(optimized) != set(expected):
            raise ValueError("optimizer parameter coverage mismatch")

    def state_dict(self):
        if self.phase or self.failed:
            raise RuntimeError("only a healthy complete update boundary is resumable")
        return {"schema": "native-r18-equal-microbatch-engine-v1", "config": self.config.binding(),
                "completed_updates": self.completed_updates, "microsteps": 2*self.completed_updates,
                "completed_epochs": self.completed_updates//304, "next_epoch": self.completed_updates//304+1,
                "next_window": self.completed_updates%304, "phase": 0, "failed": False,
                "sampler": "v2b-order:seed:epoch; acknowledge complete logical16 only",
                "augmentation": "v2b-augment:seed:epoch:position:stable_id",
                "loss_coefficients":[.5,.5],"parameter_optimizer_steps":dict(self.parameter_steps)}

    def validate_state_dict(self, state):
        u = state.get("completed_updates")
        if type(u) is not int or not 0 <= u <= 36480:
            raise ValueError("invalid common R18 update count")
        expected = self.state_dict()
        expected.update(completed_updates=u, microsteps=2*u, completed_epochs=u//304, next_epoch=u//304+1, next_window=u%304)
        counts=state.get("parameter_optimizer_steps")
        if not isinstance(counts,dict) or set(counts)!=set(self.parameter_steps) or any(type(v) is not int or not 0<=v<=u for v in counts.values()):
            raise ValueError("common R18 parameter step ledger mismatch")
        expected["parameter_optimizer_steps"]=counts
        if state != expected:
            raise ValueError("common R18 engine/cursor/schedule configuration mismatch")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.completed_updates = state["completed_updates"]
        self.parameter_steps=dict(state["parameter_optimizer_steps"])

    def train_window(self, images, targets, *, capture=False):
        if self.failed or self.phase or self.completed_updates >= 36480:
            raise RuntimeError("common R18 engine cannot enter another update")
        bs = self.config.physical_batch_size
        if tuple(images.shape) != (bs*2, 3, *self.config.input_hw) or len(targets) != bs*2:
            raise ValueError("not a full fixed input window")
        require_finite(images, "training images")
        if bool(((images < 0) | (images > 1)).any()):
            raise ValueError("image tensor outside [0,1]")
        for target in targets:
            b = target["boxes"]
            require_finite(b, "GT boxes")
            if b.shape != (len(target["labels"]),4) or bool((b[:,2:] <= 0).any()) or len(target["gt_ids"]) != len(b):
                raise ValueError("invalid positive GT / identity alignment")
        self.model.train(); self.criterion.train()
        if getattr(self.model.decoder, "trace_enabled", False):
            raise RuntimeError("training must not enable trace collection")
        lr = self.next_learning_rate()
        for group in self.optimizer.param_groups: group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        started = time.monotonic()
        values, diagnostics, outputs_hash,noise_hash = [], [], [], []
        try:
            with _batchnorm_backward_layout(self.model, self.device):
                for micro in range(2):
                    subset = _move(targets[micro*bs:(micro+1)*bs], self.device)
                    batch = images[micro*bs:(micro+1)*bs].to(self.device)
                    context = torch.autocast(self.device.type, dtype=torch.bfloat16) if self.config.amp_dtype == "bfloat16" else contextlib.nullcontext()
                    with context, dn_capture(self.model) as noise, forward_guards(self.model):
                        out = self.model(batch, targets=subset)
                        # Core probability/log/loss arithmetic explicitly opts out of BF16.
                    with torch.autocast(self.device.type, enabled=False):
                        losses = self.criterion(fp32_outputs(out), subset)
                        objective = sum(losses.values()) * .5
                    require_finite(objective, "equal microbatch objective")
                    objective.backward()
                    self.phase += 1
                    assert_healthy(self.model, gradients=True)
                    values.append({k: float(v.detach()) for k,v in losses.items()})
                    diagnostics.append(copy.deepcopy(self.criterion.last_diagnostics))
                    if capture:
                        outputs_hash.append(output_digest(out))
                        noise_hash.append(output_digest(noise))
                    del out, losses, objective, subset, batch
            before_clip = tensor_digest((name, p.grad) for name,p in self.model.named_parameters()) if capture else None
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_max_norm, error_if_nonfinite=True)
            assert_healthy(self.model, gradients=True)
            used={n for n,p in self.model.named_parameters() if p.grad is not None}
            self.optimizer.step()
            for n,p in self.model.named_parameters():
                if n in used:self.parameter_steps[n]+=1
                actual=int(self.optimizer.state.get(p,{}).get("step",0))
                if actual!=self.parameter_steps[n]:raise RuntimeError("common R18 parameter optimizer clock mismatch: "+n)
            self.ema.update(self.model)
            self.completed_updates += 1
            assert_healthy(self.model, self.optimizer, self.ema)
            self.optimizer.zero_grad(set_to_none=True)
            self.phase = 0
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            result = {"update": self.completed_updates, "epoch": (self.completed_updates-1)//304+1,
                      "window": (self.completed_updates-1)%304, "microsteps": self.completed_updates*2,
                      "lr": lr, "coefficients": [.5,.5], "microbatch_losses": values,
                      "loss": sum(sum(v.values()) for v in values)/2., "normalizers": diagnostics,
                      "gradient_norm_before_clip": float(norm), "gradient_sha256_before_clip": before_clip,
                      "output_sha256":outputs_hash,"dn_noise_sha256":noise_hash, "elapsed_seconds": time.monotonic()-started,
                      "finite": True, "optimizer_steps": 1, "ema_updates": 1}
            return result
        except Exception:
            self.failed = True
            # Do not erase gradients or roll back partially updated state.
            raise
