"""Reproducible real RT-DETRv2 CPU verification using generated tensors only.

This tool constructs no Dataset, DataLoader, evaluator, CUDA context, or launcher.
Its append-only output lives in a new artifacts/v2b_cpu_<run_id> directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sparse_rtdetr.baseline.training_v2b import (
    V2BConfig, build_v2b_components, validate_model_geometry,
)
from sparse_rtdetr.baseline.training_v2b_checkpoint import save_checkpoint, restore_checkpoint
from sparse_rtdetr.baseline.training_v2b_evidence import (
    build_run_binding, build_completed_report, collect_native_cpu_observation,
    strict_json_loads, validate_completed_report, write_exclusive_json,
)

CONFIG_PATH = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_engineering.json"


def _no_cuda(*args, **kwargs):
    raise RuntimeError("CUDA initialization is forbidden in this CPU verification")


def _equal_tree(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or actual.device != expected.device
                or actual.layout != expected.layout):
            raise AssertionError("tensor replay metadata mismatch: " + path)
        left = actual.detach().reshape(-1).contiguous().view(torch.uint8)
        right = expected.detach().reshape(-1).contiguous().view(torch.uint8)
        if not torch.equal(left, right):
            raise AssertionError("tensor replay byte mismatch: " + path)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("mapping replay mismatch: " + path)
        for key in expected:
            _equal_tree(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("sequence replay mismatch: " + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            _equal_tree(left, right, path + "." + str(index))
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError("value replay mismatch: " + path)


def _rng_snapshot(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "input_generator": generator.get_state().clone(),
    }


def _bn_backward_diagnostic():
    """Measure the CPU layout issue against analytic frozen-BN gradients."""
    generator = torch.Generator().manual_seed(10)
    x0 = torch.randn(1, 256, 8, 8, dtype=torch.float64, generator=generator)
    gradient = torch.randn(1, 336, 256, dtype=torch.float64, generator=generator)
    gradient = gradient[:, 256:320].permute(0, 2, 1).reshape(1, 256, 8, 8)
    result = {"torch_version": str(torch.__version__),
              "shape": list(x0.shape), "gradient_stride": list(gradient.stride())}
    for label, normalize in (("native_layout", False), ("contiguous_layout", True)):
        x = x0.clone().requires_grad_()
        bn = torch.nn.BatchNorm2d(256).double().eval()
        bn(x).backward(gradient.contiguous() if normalize else gradient)
        result[label] = {
            "bias_gradient_l2_error": float((bn.bias.grad - gradient.sum((0, 2, 3))).norm()),
            "input_gradient_l2_error": float((x.grad - gradient / (1 + bn.eps)**.5).norm()),
        }
    if max(result["contiguous_layout"].values()) > 1e-10:
        raise AssertionError("CPU BN contiguous backward violates analytic gradients")
    return result


def _targets(logical_batch):
    generator = torch.Generator().manual_seed(73)
    result = []
    for index in range(logical_batch):
        count = 0 if index == 0 else 1 + index % 3
        centers = .2 + .6 * torch.rand((count, 2), generator=generator)
        sizes = .05 + .15 * torch.rand((count, 2), generator=generator)
        result.append({"labels": torch.arange(count, dtype=torch.int64) % 10,
                       "boxes": torch.cat((centers, sizes), dim=1)})
    return result


def _input_identity(images, second_images, targets, windows):
    digest = hashlib.sha256()
    for value in [images] + ([second_images] if windows == 2 else []):
        digest.update(str((list(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    for target in targets:
        for key in ("labels", "boxes"):
            value = target[key]
            digest.update(str((key, list(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
    return {"fixture": "v2b_cpu_v1; torch.rand RGB seed=900; boxes seed=73; "
                       "GT_i=0 if i=0 else 1+i%3; normalized cxcywh; no real images",
            "tensor_sha256": digest.hexdigest(), "window_count": windows}


def _code_paths(components):
    paths = {
        row["relative_path"]: ROOT / row["relative_path"]
        for key in ("vendor_sources", "package_sources")
        for row in components.initialization[key]
    }
    # Bind YAML dependencies as well as actual imported Python source bytes.
    for path in sorted((ROOT / "vendor/rtdetrv2_pytorch/configs").rglob("*.yml")):
        paths[str(path.relative_to(ROOT))] = path
    for path in (Path(__file__).resolve(), CONFIG_PATH):
        paths[str(path.relative_to(ROOT))] = path
    return paths


def verify(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU verification must start without a CUDA context")
    torch.cuda._lazy_init = _no_cuda
    torch.set_num_threads(args.threads)
    policy = strict_json_loads(CONFIG_PATH.read_bytes())
    if policy["scope"] != "cpu_synthetic_engineering":
        raise ValueError("wrong engineering configuration scope")
    config = V2BConfig(**{
        **policy["config"], "input_size": args.input_size,
        "physical_batch_size": args.physical_batch,
        "accumulation_steps": args.accumulation,
        "amp_dtype": args.amp_dtype,
    })
    if config.sampling_backend != "deterministic_gather":
        raise ValueError("repaired CPU verification requires deterministic_gather sampling")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", args.run_id):
        raise ValueError("run_id must be a short lowercase safe directory name")
    if args.threads < 1 or args.threads > 4:
        raise ValueError("CPU verification threads must be between 1 and 4")
    artifact_root = ROOT / "artifacts"
    if artifact_root.is_symlink():
        raise ValueError("artifacts root must not be a symlink")
    artifact_root.mkdir(exist_ok=True)
    output = artifact_root / ("v2b_cpu_" + args.run_id)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    bn_diagnostic = _bn_backward_diagnostic()
    authority = policy["pretrained_authority"]
    components = build_v2b_components(
        config, repo_root=ROOT, pretrained_path=args.pretrained_path,
        pretrained_sha256=authority["weight_sha256"],
    )
    generator = torch.Generator().manual_seed(900)
    shape = (config.logical_batch_size, 3, config.input_size, config.input_size)
    images = torch.rand(shape, generator=generator)
    preview = torch.Generator()
    preview.set_state(generator.get_state())
    second_expected = torch.rand(shape, generator=preview)
    targets = _targets(config.logical_batch_size)
    binding = build_run_binding(
        run_id=args.run_id, code_paths=_code_paths(components),
        config={**config.binding_config(),
                "resolved_vendor_config": components.resolved_config,
                "geometry": components.geometry,
                "sampling": components.initialization["sampling"],
                "verification_windows": args.windows},
        initial_parameters=components.initialization["parameters"],
        initial_state=components.initialization["model_state"],
        synthetic_input=_input_identity(images, second_expected, targets, args.windows),
    )
    write_exclusive_json(output / "binding.json", binding)
    write_exclusive_json(output / "initialization.json", components.initialization)
    observed_dtypes = []
    hook = components.model.register_forward_hook(
        lambda module, values, out: observed_dtypes.append(str(out["pred_logits"].dtype))
    )
    components.engine.begin_epoch(1)
    windows = [components.engine.train_window(images, targets)]
    resume_reference = None
    if args.windows == 2:
        resume_reference = save_checkpoint(
            output / "resume_after_window1.pth", model=components.model,
            optimizer=components.optimizer, ema=components.ema, engine=components.engine,
            scheduler=components.scheduler, warmup=components.warmup, binding=binding,
            extra_rng_generators={"inputs": generator},
        )
        next_images = torch.rand(shape, generator=generator)
        _equal_tree(next_images, second_expected, "fixture")
        windows.append(components.engine.train_window(next_images, targets))
    hook.remove()
    components.engine.finish_epoch(expected_optimizer_steps=args.windows)
    expected_rng = _rng_snapshot(generator)
    checkpoint = save_checkpoint(
        output / "completed.pth", model=components.model, optimizer=components.optimizer,
        ema=components.ema, engine=components.engine, scheduler=components.scheduler,
        warmup=components.warmup, binding=binding, extra_rng_generators={"inputs": generator},
    )
    replay = {"performed": False, "bitwise_equal": None}
    if resume_reference is not None:
        restored = build_v2b_components(
            config, repo_root=ROOT, pretrained_path=args.pretrained_path,
            pretrained_sha256=authority["weight_sha256"],
        )
        resumed_generator = torch.Generator().manual_seed(999999)
        restore_checkpoint(
            resume_reference["path"], model=restored.model, optimizer=restored.optimizer,
            ema=restored.ema, engine=restored.engine, scheduler=restored.scheduler,
            warmup=restored.warmup, expected_binding=binding,
            expected_sha256=resume_reference["sha256"],
            extra_rng_generators={"inputs": resumed_generator},
        )
        resumed_images = torch.rand(shape, generator=resumed_generator)
        _equal_tree(resumed_images, second_expected, "resumed_inputs")
        resumed_window = restored.engine.train_window(resumed_images, targets)
        restored.engine.finish_epoch(expected_optimizer_steps=args.windows)
        _equal_tree(resumed_window, windows[-1], "window")
        for name in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
            _equal_tree(getattr(restored, name).state_dict(),
                        getattr(components, name).state_dict(), name)
        _equal_tree(_rng_snapshot(resumed_generator), expected_rng, "RNG")
        _equal_tree(validate_model_geometry(restored.model, config),
                    validate_model_geometry(components.model, config), "model_caches")
        replay = {"performed": True, "bitwise_equal": True,
                  "resume_checkpoint": resume_reference,
                  "scope": ["raw_model", "optimizer", "EMA", "engine", "model_caches",
                            "scheduler", "warmup", "Python_NumPy_torch_CPU_RNG",
                            "explicit_input_generator", "next_window_ledger"]}
    counters = {
        "epoch": components.engine.epoch,
        "optimizer_updates": components.engine.optimizer_updates,
        "microsteps": components.engine.microsteps,
        "samples_seen": components.engine.optimizer_updates * config.logical_batch_size,
    }
    ledger = write_exclusive_json(
        output / "windows.json",
        {"run_binding_sha256": binding["binding_sha256"],
         "windows": windows, "counters": counters},
    )
    observation = collect_native_cpu_observation(binding)
    report = build_completed_report(
        binding, observation, checkpoint_reference=checkpoint,
        counters=counters, window_references=[ledger],
    )
    validate_completed_report(report, binding=binding, observation=observation)
    report_reference = write_exclusive_json(output / "cpu_report.json", report)
    summary = {
        "verification_pass": True, "scope": "cpu_synthetic_engineering",
        "model": type(components.model).__name__,
        "parameters": sum(p.numel() for p in components.model.parameters()),
        "batchnorm_modules": len(components.batchnorm_inventory),
        "bn_backward_layout": config.binding_config()["bn_backward_layout"],
        "bn_backward_diagnostic": bn_diagnostic,
        "input_size": config.input_size, "physical_batch": config.physical_batch_size,
        "accumulation": config.accumulation_steps, "amp_dtype": config.amp_dtype,
        "sampling_backend": config.sampling_backend,
        "sampling": components.initialization["sampling"],
        "observed_logits_dtypes": observed_dtypes, "counters": counters,
        "geometry": components.geometry, "checkpoint_replay": replay,
        "cuda_initialized": torch.cuda.is_initialized(), "real_data_accessed": False,
        "gpu_ready": False, "scientific_certified": False,
        "elapsed_seconds": time.monotonic() - started,
        "binding_sha256": binding["binding_sha256"],
        "report": report_reference, "checkpoint": checkpoint,
    }
    if summary["cuda_initialized"]:
        raise AssertionError("unexpected CUDA context")
    summary_reference = write_exclusive_json(output / "verification.json", summary)
    print(json.dumps({"verification": summary_reference,
                      "input_size": config.input_size, "counters": counters,
                      "replay": replay, "gpu_ready": False}, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--input-size", type=int, choices=(128, 640), default=128)
    parser.add_argument("--physical-batch", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--accumulation", type=int, choices=(1, 2), default=2)
    parser.add_argument("--amp-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--windows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    try:
        return verify(args)
    except Exception as exc:
        print(json.dumps({"verification_pass": False, "error_type": type(exc).__name__,
                          "error": str(exc), "gpu_ready": False}), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
