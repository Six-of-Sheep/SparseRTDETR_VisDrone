"""Seeded RT-DETRv2 R18 foundation; no evaluator or automatic launcher.

The vendor implementation and historical v2a contracts remain immutable.
A full logical batch is prepared before the engine splits physical microbatches.
CUDA placement requires an explicitly prepared, natively admitted runtime.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import io
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .config import (
    _restore_registry,
    _snapshot_registry,
    _vendor_config_path,
    _vendor_path,
    _vendor_root,
)
from .training_v2b_engine import AccumulationEngine
from .training_v2b_device import (
    PreparedRuntime, RuntimeDeviceError, cuda_backend_policy_for_sampling,
    place_module, prepare_runtime, validate_bound_runtime_policy,
    validate_component_placement, validate_prepared_runtime,
)


class V2BConfigurationError(ValueError):
    """An explicit engineering configuration or model identity is invalid."""


@dataclass(frozen=True)
class V2BConfig:
    seed: int = 0
    input_size: int = 640
    physical_batch_size: int = 16
    accumulation_steps: int = 1
    amp_dtype: str = "bfloat16"
    bn_statistics: str = "train"
    pretrained_required: bool = True
    num_denoising: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    warmup_steps: int = 2000
    ema_decay: float = 0.9999
    ema_warmups: int = 2000
    sampling_backend: str = "native"

    def __post_init__(self) -> None:
        for name in ("input_size", "physical_batch_size", "accumulation_steps",
                     "warmup_steps", "ema_warmups"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise V2BConfigurationError(f"{name} must be a positive integer")
        if self.input_size % 32 or not (128 <= self.input_size <= 640 or self.input_size in (896, 1024)):
            raise V2BConfigurationError("CPU foundation supports sizes 128..640 divisible by 32, plus 896 and 1024")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise V2BConfigurationError("seed must be an integer in [0, 2**32)")
        if type(self.num_denoising) is not int or self.num_denoising < 0:
            raise V2BConfigurationError("num_denoising must be a nonnegative integer")
        if type(self.pretrained_required) is not bool:
            raise V2BConfigurationError("pretrained_required must be boolean")
        if self.amp_dtype not in ("float32", "bfloat16"):
            raise V2BConfigurationError("only FP32 or BF16 without GradScaler is supported")
        if self.bn_statistics not in ("train", "frozen"):
            raise V2BConfigurationError("bn_statistics must be train or frozen")
        if type(self.sampling_backend) is not str or self.sampling_backend not in ("native", "deterministic_gather"):
            raise V2BConfigurationError("sampling_backend must be native or deterministic_gather")
        for name in ("learning_rate", "clip_max_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise V2BConfigurationError(f"{name} must be positive and finite")
        if type(self.weight_decay) not in (int, float) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise V2BConfigurationError("weight_decay must be nonnegative and finite")
        if type(self.ema_decay) not in (int, float) or not 0 <= self.ema_decay < 1:
            raise V2BConfigurationError("ema_decay must be in [0, 1)")

    @property
    def logical_batch_size(self) -> int:
        return self.physical_batch_size * self.accumulation_steps

    @property
    def cuda_backend_policy(self) -> dict[str, Any]:
        return cuda_backend_policy_for_sampling(self.sampling_backend)

    def binding_config(self, *, device: str = "cpu",
                       scope: str = "cpu_synthetic_engineering",
                       cuda_gpu_uuid: str | None = None) -> dict[str, Any]:
        if device not in ("cpu", "cuda:0"):
            raise V2BConfigurationError("only CPU or explicit cuda:0 is supported")
        if scope not in ("cpu_synthetic_engineering", "train_core_runtime_engineering"):
            raise V2BConfigurationError("unknown v2b engineering scope")
        if device == "cpu" and cuda_gpu_uuid is not None:
            raise V2BConfigurationError("CPU configuration cannot bind a CUDA UUID")
        if device == "cuda:0" and (scope != "train_core_runtime_engineering"
                                   or not isinstance(cuda_gpu_uuid, str)
                                   or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                                                   cuda_gpu_uuid) is None):
            raise V2BConfigurationError("CUDA configuration requires train_core scope and a GPU UUID")
        result = {
            **asdict(self),
            "device": device,
            "scope": scope,
            "num_classes": 10,
            "num_queries": 300,
            "decoder_layers": 3,
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "logical_batch_size": self.logical_batch_size,
            "loss_normalization": "logical_window_gt_sum_clamped_once",
            "dn_policy": "native_per_physical_microbatch",
            "bn_backward_layout": "cpu_contiguous_cuda_native",
            "amp_scope": "model_and_criterion_only",
            "cuda_backend_policy": self.cuda_backend_policy,
            "grad_scaler": False,
            "ema_float_buffers": "vendor_average",
            "ema_integer_buffers": "vendor_unchanged",
            "scheduler": {"milestones": [1000], "gamma": 0.1,
                          "clock": "epoch_after_warmup"},
            "augmentation_stop_internal_epoch": 117,
            "augmentation_stop_logged_epoch": 118,
            "sampler": "shared_logical_batch_before_physical_split",
            "tail_policy": "drop_incomplete_logical_batch",
            "production_training_authorized": False,
        }
        if cuda_gpu_uuid is not None:
            result["cuda_gpu_uuid"] = cuda_gpu_uuid
        return result


def seed_cpu_sources(seed: int) -> None:
    """Initialize actual CPU random sources without querying/initializing CUDA."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise V2BConfigurationError("invalid seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader has already assigned torch.initial_seed() to this worker."""
    if type(worker_id) is not int or worker_id < 0:
        raise V2BConfigurationError("invalid worker_id")
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def logical_batch_indices(dataset_size: int, *, seed: int, epoch: int,
                          logical_batch_size: int = 16,
                          completed_batches: int = 0) -> Iterator[tuple[int, ...]]:
    """Deterministic logical order; independent of physical batch and DN RNG.

    This is the pure order plan. The train_core loader separately keys each
    sample's augmentation RNG and acknowledges only completed logical windows,
    so prefetched items never advance a saved resume position.
    """
    for name, value, minimum in (
        ("dataset_size", dataset_size, 0), ("seed", seed, 0),
        ("epoch", epoch, 1), ("logical_batch_size", logical_batch_size, 1),
        ("completed_batches", completed_batches, 0),
    ):
        if type(value) is not int or value < minimum:
            raise V2BConfigurationError(f"invalid {name}")
    count = dataset_size // logical_batch_size
    if completed_batches > count:
        raise V2BConfigurationError("completed_batches exceeds the epoch")
    generator = torch.Generator(device="cpu")
    # Stable epoch-specific seed; model/DN/worker draws cannot alter this stream.
    digest = hashlib.sha256(f"v2b-order:{seed}:{epoch}".encode()).digest()
    generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    order = torch.randperm(dataset_size, generator=generator).tolist()
    for index in range(completed_batches, count):
        start = index * logical_batch_size
        yield tuple(order[start:start + logical_batch_size])


def _package_source_identities(repo_root: Path) -> list[dict[str, str]]:
    """Bind the imported package bytes, including reused configuration helpers."""
    expected = repo_root.resolve() / "src" / "sparse_rtdetr"
    identities = []
    for name, module in sorted(tuple(sys.modules.items())):
        if name != "sparse_rtdetr" and not name.startswith("sparse_rtdetr."):
            continue
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise V2BConfigurationError(f"package module has no source identity: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(expected):
            raise V2BConfigurationError(
                f"package module belongs to a different checkout: {name}"
            )
        identities.append({"module": name, "relative_path": str(path.relative_to(repo_root)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return identities


def _vendor_source_identities(repo_root: Path) -> list[dict[str, str]]:
    """Reject cached vendor imports from another checkout before construction."""
    expected = _vendor_root(repo_root).resolve() / "src"
    identities = []
    for name, module in sorted(tuple(sys.modules.items())):
        if name != "src" and not name.startswith("src."):
            continue
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            if name == "src":  # an existing namespace package can be extended
                continue
            raise V2BConfigurationError(f"vendor module has no source identity: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(expected):
            raise V2BConfigurationError(
                f"cached vendor module belongs to a different checkout: {name}"
            )
        identities.append({"module": name, "relative_path": str(path.relative_to(repo_root)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return identities


def _resolved_vendor_config(repo_root: Path, config: V2BConfig) -> dict[str, Any]:
    with _vendor_path(_vendor_root(repo_root)):
        from src.core.yaml_utils import load_config
        resolved = copy.deepcopy(load_config(str(_vendor_config_path(repo_root)), cfg={}))
    resolved.update(
        num_classes=10, remap_mscoco_category=False,
        eval_spatial_size=[config.input_size, config.input_size],
        use_amp=config.amp_dtype == "bfloat16", device="cpu", epoches=120,
    )
    resolved["PResNet"].update(
        depth=18, pretrained=False, freeze_at=-1, freeze_norm=False,
    )
    resolved["RTDETRTransformerv2"].update(
        num_layers=3, num_queries=300, num_denoising=config.num_denoising,
    )
    # These are a resolved configuration snapshot, not constructed loaders.
    for name, batch in (("train_dataloader", config.logical_batch_size),
                        ("val_dataloader", 8)):
        loader = resolved[name]
        loader.pop("batch_size", None)
        loader["total_batch_size"] = batch
        loader["num_workers"] = 4
        loader["drop_last"] = name == "train_dataloader"
        loader["dataset"]["img_folder"] = None
        loader["dataset"]["ann_file"] = None
        transforms = loader["dataset"]["transforms"]
        for op in transforms["ops"]:
            if op["type"] == "Resize":
                op["size"] = [config.input_size, config.input_size]
        if "policy" in transforms:
            transforms["policy"]["epoch"] = 117
        if "collate_fn" in loader:
            loader["collate_fn"]["scales"] = None
            loader["collate_fn"]["stop_epoch"] = 117
    resolved.pop("scaler", None)  # this engine never constructs a GradScaler
    resolved["optimizer"] = {
        "type": "AdamW", "lr": config.learning_rate,
        "betas": [0.9, 0.999], "weight_decay": config.weight_decay,
        "params": [{"params": "^(?=.*(?:norm|bn)).*$", "weight_decay": 0.0}],
    }
    resolved["lr_warmup_scheduler"]["warmup_duration"] = config.warmup_steps
    resolved["ema"].update(decay=config.ema_decay, warmups=config.ema_warmups)
    return resolved


def _build_vendor_objects(repo_root: Path, config: V2BConfig):
    _package_source_identities(repo_root)
    _vendor_source_identities(repo_root)
    with _vendor_path(_vendor_root(repo_root)):
        for module in ("src.solver", "src.data", "src.nn", "src.zoo.rtdetr"):
            importlib.import_module(module)
        from src.core import GLOBAL_CONFIG
        from src.core.workspace import create
        from src.core.yaml_utils import merge_config
        from src.optim.ema import ModelEMA
        from src.optim.warmup import LinearWarmup

        _vendor_source_identities(repo_root)
        snapshot = _snapshot_registry(GLOBAL_CONFIG)
        try:
            resolved = _resolved_vendor_config(repo_root, config)
            # clone_registry handles module/class objects that deepcopy cannot.
            registry = _snapshot_registry(GLOBAL_CONFIG)
            merged = merge_config(resolved, registry, inplace=False, overwrite=False)
            seed_cpu_sources(config.seed)  # immediately before the first model
            model = create(merged["model"], merged).to(device="cpu")
            criterion = create(merged["criterion"], merged).to(device="cpu")
            postprocessor = create(merged["postprocessor"], merged).to(device="cpu")
        finally:
            _restore_registry(GLOBAL_CONFIG, snapshot)
    return model, criterion, postprocessor, resolved, ModelEMA, LinearWarmup


def _load_pretrained(model, path: Path, expected_sha256: str) -> dict[str, Any]:
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise V2BConfigurationError("an explicit pretrained SHA-256 is required")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise V2BConfigurationError("pretrained SHA-256 mismatch")
    # Deserialize precisely the bytes just verified, with no arbitrary pickle.
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise V2BConfigurationError("pretrained object must be a tensor state dict")
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise V2BConfigurationError("nonfinite pretrained parameters")
    result = model.backbone.load_state_dict(state, strict=True)
    return {"path": str(path.resolve()), "sha256": actual, "size_bytes": len(raw),
            "strict_load": True, "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys)}


def _optimizer(model, config: V2BConfig):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = []
    for group_name, norm_group in (("norm_or_bn", True), ("default", False)):
        items = [(name, p) for name, p in named
                 if (("norm" in name or "bn" in name) == norm_group)]
        groups.append({"params": [p for _, p in items],
                       "param_names": [name for name, _ in items],
                       "name": group_name, "lr": config.learning_rate,
                       "weight_decay": 0.0 if norm_group else config.weight_decay})
    ids = [id(p) for group in groups for p in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for _, p in named}:
        raise V2BConfigurationError("optimizer parameter partition mismatch")
    return torch.optim.AdamW(groups, lr=config.learning_rate,
                             betas=(0.9, 0.999), weight_decay=config.weight_decay)


def validate_model_geometry(model, config: V2BConfig) -> dict[str, Any]:
    """Check actual caches; invalid anchor +inf sentinels are intentional."""
    size = config.input_size
    for component in (model.encoder, model.decoder):
        if list(component.eval_spatial_size) != [size, size]:
            raise V2BConfigurationError("model eval_spatial_size mismatch")
    count = sum((size // stride) ** 2 for stride in (8, 16, 32))
    if tuple(model.decoder.anchors.shape) != (1, count, 4):
        raise V2BConfigurationError("decoder anchor cache shape mismatch")
    if tuple(model.decoder.valid_mask.shape) != (1, count, 1):
        raise V2BConfigurationError("decoder valid_mask cache shape mismatch")
    positions = {}
    for index in model.encoder.use_encoder_idx:
        name = f"pos_embed{index}"
        shape = tuple(getattr(model.encoder, name).shape)
        expected = (1, (size // model.encoder.feat_strides[index]) ** 2,
                    model.encoder.hidden_dim)
        if shape != expected:
            raise V2BConfigurationError("encoder positional cache shape mismatch")
        positions[name] = list(shape)
    from .training_v2b_evidence import initial_parameter_reference
    cache_tensors = {"decoder.anchors": model.decoder.anchors,
                     "decoder.valid_mask": model.decoder.valid_mask}
    cache_tensors.update({"encoder." + name: getattr(model.encoder, name)
                          for name in positions})
    return {"input_size": [size, size], "anchors": list(model.decoder.anchors.shape),
            "valid_mask": list(model.decoder.valid_mask.shape),
            "position_caches": positions,
            "cache_identity": initial_parameter_reference({
                name: tensor.detach().cpu() for name, tensor in cache_tensors.items()
            })}


_SAMPLING_MODULE_NAMES = tuple(
    f"decoder.decoder.layers.{index}.cross_attn" for index in range(3)
)


def _sampling_source_identity(value: Any) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    module = sys.modules.get(value.__module__)
    filename = getattr(module, "__file__", None)
    if filename is None:
        raise V2BConfigurationError("sampling implementation has no source identity")
    path = Path(filename).resolve()
    if not path.is_relative_to(root):
        raise V2BConfigurationError("sampling implementation belongs to a different checkout")
    raw = path.read_bytes()
    return {"relative_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def validate_model_sampling(model, *, sampling_backend: str) -> dict[str, Any]:
    """Inspect exact per-model sampling callables, including an EMA model.

    This performs no tensor computation or backend mutation. A state_dict does
    not contain these Python callables, so parameter/checkpoint hashes alone
    cannot establish sampling identity. Candidate workers must not install the
    native diagnostic's temporary core observer while this check is active.
    """
    try:
        cuda_backend_policy_for_sampling(sampling_backend)
    except RuntimeDeviceError as exc:
        raise V2BConfigurationError(str(exc)) from exc
    decoder = importlib.import_module("src.zoo.rtdetr.rtdetrv2_decoder")
    attention_type = decoder.MSDeformableAttention
    expected_core = decoder.deformable_attention_core_func_v2
    if sampling_backend == "deterministic_gather":
        from .training_v2b_deterministic_sampling import deformable_attention_core_func_v2
        expected_core = deformable_attention_core_func_v2
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, attention_type)}
    if tuple(sorted(modules)) != _SAMPLING_MODULE_NAMES:
        raise V2BConfigurationError("actual RT-DETRv2 sampling module inventory differs")
    rows = []
    for name in _SAMPLING_MODULE_NAMES:
        module = modules[name]
        core = getattr(module, "ms_deformable_attn_core", None)
        if (type(module) is not attention_type or type(module.method) is not str or module.method != "default"
                or type(core) is not functools.partial or core.func is not expected_core
                or core.args != () or type(core.keywords.get("method")) is not str
                or core.keywords != {"method": "default"}):
            raise V2BConfigurationError("actual sampling callable differs from selected backend: " + name)
        dimensions = {key: getattr(module, key, None)
                      for key in ("num_heads", "num_levels", "head_dim")}
        if any(type(value) is not int or value <= 0 for value in dimensions.values()):
            raise V2BConfigurationError("sampling dimensions are invalid: " + name)
        points = module.num_points_list
        if (type(points) is not list or len(points) != dimensions["num_levels"]
                or any(type(point) is not int or point <= 0 for point in points)):
            raise V2BConfigurationError("sampling point inventory is invalid: " + name)
        offset_scale = module.offset_scale
        if type(offset_scale) not in (int, float) or not math.isfinite(offset_scale) or offset_scale <= 0:
            raise V2BConfigurationError("sampling offset scale is invalid: " + name)
        rows.append({"name": name, "method": module.method,
                     **dimensions, "num_points_list": list(points), "offset_scale": offset_scale})
    return {
        "schema_version": 1, "backend": sampling_backend,
        "geometry": {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
        "module_type": {"module": attention_type.__module__, "qualname": attention_type.__qualname__,
                        "source": _sampling_source_identity(attention_type)},
        "core": {"module": expected_core.__module__, "qualname": expected_core.__qualname__,
                 "partial_args": [], "partial_keywords": {"method": "default"},
                 "source": _sampling_source_identity(expected_core)},
        "modules": rows,
    }


def _configure_model_sampling(model, config: V2BConfig) -> dict[str, Any]:
    original = validate_model_sampling(model, sampling_backend="native")
    if config.sampling_backend == "native":
        return original
    from .training_v2b_deterministic_sampling import deformable_attention_core_func_v2
    modules = dict(model.named_modules())
    for name in _SAMPLING_MODULE_NAMES:
        modules[name].ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method="default",
        )
    return validate_model_sampling(model, sampling_backend=config.sampling_backend)


@dataclass
class V2BComponents:
    config: V2BConfig
    model: Any
    criterion: Any
    postprocessor: Any
    optimizer: Any
    ema: Any
    scheduler: Any
    warmup: Any
    engine: AccumulationEngine
    resolved_config: dict[str, Any]
    initialization: dict[str, Any]
    geometry: dict[str, Any]
    batchnorm_inventory: list[dict[str, Any]]
    runtime: PreparedRuntime


def build_v2b_components(config: V2BConfig, *, repo_root: str | Path | None = None,
                         pretrained_path: str | Path | None = None,
                         pretrained_sha256: str | None = None,
                         runtime: PreparedRuntime | None = None) -> V2BComponents:
    """Initialize on CPU, then place all state before constructing the optimizer.

    The default is CPU-only. A CUDA runtime must already have passed native
    admission; this factory does not authorize, probe, or launch GPU work.
    """
    if not isinstance(config, V2BConfig):
        raise V2BConfigurationError("config must be V2BConfig")
    if config.pretrained_required and pretrained_path is None:
        raise V2BConfigurationError("the configured pretrained authority is required")
    if (pretrained_path is None) != (pretrained_sha256 is None):
        raise V2BConfigurationError("pretrained path and SHA-256 must be paired")
    runtime = prepare_runtime(seed=config.seed) if runtime is None else runtime
    runtime_identity = validate_prepared_runtime(runtime)
    validate_bound_runtime_policy(runtime_identity, {
        "sampling_backend": config.sampling_backend, "cuda_backend_policy": config.cuda_backend_policy,
    })
    if runtime.seed != config.seed:
        raise V2BConfigurationError("prepared runtime seed differs from model seed")
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    model, criterion, postprocessor, resolved, ema_type, warmup_type = _build_vendor_objects(root, config)
    sampling = _configure_model_sampling(model, config)
    pretrained = None
    if pretrained_path is not None:
        pretrained = _load_pretrained(model, Path(pretrained_path), pretrained_sha256)
    from .training_v2b_evidence import initial_parameter_reference
    initialization = {
        "seed": config.seed,
        "seed_initialized_before_model": True,
        "model_initialization_device": "cpu",
        "pretrained": pretrained,
        "sampling": sampling,
        "parameters": initial_parameter_reference(dict(model.named_parameters())),
        "model_state": initial_parameter_reference(model.state_dict()),
        "vendor_sources": _vendor_source_identities(root),
        "package_sources": _package_source_identities(root),
    }
    geometry = validate_model_geometry(model, config)
    bn_inventory = [
        {"name": name, "type": type(module).__name__, "affine": module.affine,
         "track_running_stats": module.track_running_stats,
         "num_features": module.num_features, "momentum": module.momentum}
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    for component in (model, criterion, postprocessor):
        place_module(component, runtime)
    resolved["device"] = str(runtime.device)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
    warmup = warmup_type(scheduler, warmup_duration=config.warmup_steps)
    ema = ema_type(model, decay=config.ema_decay, warmups=config.ema_warmups)
    if validate_model_sampling(ema.module, sampling_backend=config.sampling_backend) != sampling:
        raise V2BConfigurationError("EMA sampling identity differs from the initialized model")
    engine = AccumulationEngine(
        model, criterion, optimizer, physical_batch_size=config.physical_batch_size,
        accumulation_steps=config.accumulation_steps, device=runtime.device,
        amp_dtype=config.amp_dtype, clip_max_norm=config.clip_max_norm,
        ema=ema, warmup=warmup, scheduler=scheduler,
        bn_statistics=config.bn_statistics, expected_input_size=config.input_size,
    )
    validate_component_placement(model=model, optimizer=optimizer, ema=ema,
                                 criterion=criterion, postprocessor=postprocessor,
                                 runtime=runtime)
    return V2BComponents(config, model, criterion, postprocessor, optimizer, ema,
                          scheduler, warmup, engine, resolved, initialization,
                          geometry, bn_inventory, runtime)
