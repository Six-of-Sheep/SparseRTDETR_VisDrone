"""Isolated fixed-budget KD engine; original common Engine bytes unchanged.

train_window copies the certified Engine update math, with only the endpoint
and LR function changed. Criterion composes final-head KD outside native losses.
"""
from __future__ import annotations
import contextlib, copy, time
from types import SimpleNamespace
import torch
from ..baseline.training_v2b_engine import _batchnorm_backward_layout, _move
from ..localization.distribution import require_finite
from .model import dn_capture, fp32_outputs
from .geometry import forward_guards
from .engine import Engine as CommonEngine, assert_healthy, tensor_digest, output_digest
from .distill import VERSION, ARMS, UPDATES, learning_rate


class Engine(CommonEngine):
    def __init__(self, *args, arm, binding, **kwargs):
        if arm not in ARMS or binding['arm'] != arm:
            raise ValueError('distillation arm binding differs')
        self.arm, self.binding = arm, copy.deepcopy(binding)
        super().__init__(*args, **kwargs)

    def state_dict(self):
        value = super().state_dict()
        value.update(schema=VERSION, arm=self.arm, binding=self.binding, total_updates=UPDATES)
        return value

    def validate_state_dict(self, value):
        u = value.get('completed_updates')
        if type(u) is not int or not 0 <= u <= UPDATES:
            raise ValueError('KD update cursor differs')
        expected = self.state_dict()
        expected.update(completed_updates=u, microsteps=2*u, completed_epochs=u//304,
                        next_epoch=u//304+1, next_window=u%304)
        counts = value.get('parameter_optimizer_steps')
        if not isinstance(counts,dict) or set(counts)!=set(self.parameter_steps) or any(type(v) is not int or not 0<=v<=u for v in counts.values()):
            raise ValueError('KD optimizer cursor differs')
        expected['parameter_optimizer_steps'] = counts
        if value != expected:
            raise ValueError('KD full cursor/arm/plan/source/cache differs')

    def train_window(self, images, targets, *, capture=False):
        if self.failed or self.phase or self.completed_updates >= 3648:
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
        lr = learning_rate(self.completed_updates+1)
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


def capture(engine):
    from . import checkpoint as native
    from ..baseline import training_v2b_checkpoint as layout
    if engine.phase or engine.failed or any(p.grad is not None for p in engine.model.parameters()):
        raise ValueError('not a healthy complete KD update boundary')
    assert_healthy(engine.model, engine.optimizer, engine.ema)
    # Explicit base-criterion view for the existing native behavior validator;
    # no global replacement or mutation of the active engine/criterion.
    proxy = SimpleNamespace(model=engine.model, ema=engine.ema, optimizer=engine.optimizer,
                            criterion=engine.criterion.base, config=engine.config)
    behavior = native.behavior_identity(proxy)
    behavior['KD'] = engine.criterion.recipe()
    value = dict(protocol=VERSION, binding=engine.binding,
                 torch_version=str(torch.__version__), device=str(engine.device),
                 behavior=behavior, engine=engine.state_dict(),
                 model=engine.model.state_dict(), model_layout=layout._model_layout(engine.model),
                 model_training={n:m.training for n,m in engine.model.named_modules()},
                 optimizer=engine.optimizer.state_dict(),
                 optimizer_layout=layout._optimizer_layout(engine.model,engine.optimizer),
                 ema=engine.ema.state_dict(), rng=native.capture_rng(engine.device))
    return layout._safe(value)
