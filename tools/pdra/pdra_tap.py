"""P3 tap and frozen-engine plumbing for PDRA stage 1 / the GPU engineering check.

The tap is a read-only ``forward_hook`` on the frozen RTDETR ``model.encoder`` (the
``HybridEncoder``).  Its forward returns the three decoder feature levels; index 0 is P3
(stride 8, 256 channels for the frozen R18) with shape ``[B, 256, 96, 168]`` on the frozen
768x1344 canvas.  The hook only *observes* that tensor -- it never writes to the module
output, so detection outputs are unchanged by construction, and no extra forward pass is
introduced (one encoder call per view forward, exactly as in the frozen chain).

Engine restoration mirrors the legacy seed0 entry (``deploy.py`` of
``r18-sahi-fixed-20261016-001``, release-002, SHA ``3c974abb...``): build the engine with
the *training release* source root on ``sys.path`` (the checkpoint was written by that
code), rebuild the live binding, assert it byte-equals the saved ``formal/binding.json``,
then run the audited complete-state ``checkpoint.restore`` against the e120 EMA state.
Imports are guarded so ``sparse_rtdetr`` can never silently resolve to the development
worktree copy (which has drifted from the training release).

CPU-only acceptance uses :func:`load_cpu_ema_model_for_acceptance`, which strict-loads the
saved EMA weights into a freshly built CPU model (no engine, no restore audit, no CUDA);
its scope is the tap mechanics only and it never claims the audited GPU path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

P3_LEVEL_INDEX = 0
P3_CHANNELS = 256
TRAINING_TASK = "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001"
FROZEN_SCOPE = "formal_R18_seed0_e120"


class EncoderTap:
    """Read-only capture of the frozen HybridEncoder output list (P3 at index 0)."""

    def __init__(self, model: torch.nn.Module, *, level_index: int = P3_LEVEL_INDEX) -> None:
        self.level_index = int(level_index)
        self.calls = 0
        self.shapes: list[tuple] = []
        self.maps: list[torch.Tensor] = []
        self._handle = None
        self._model = model

    def _hook(self, module, inputs, output) -> None:
        if not isinstance(output, (list, tuple)) or len(output) <= self.level_index:
            raise ValueError("encoder output is not the expected feature list")
        level = output[self.level_index]
        if not torch.is_tensor(level) or level.dim() != 4:
            raise ValueError("tapped level is not a 4-D tensor")
        if not torch.isfinite(level).all():
            raise ValueError("tapped P3 map is not finite")
        self.calls += 1
        self.shapes.append(tuple(level.shape))
        self.maps.append(level.detach())

    def __enter__(self) -> "EncoderTap":
        # the guard from protocol_r18.geometry also registers transients; ours is additive
        self._handle = self._model.encoder.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clone_maps(self) -> dict[int, torch.Tensor]:
        """Ordinary (non-inference) copies of the captured maps.

        Maps captured under ``torch.inference_mode`` are inference tensors, which cannot
        feed an autograd graph outside inference mode.  The numpy round-trip yields plain
        tensors so downstream descriptor/optimiser code is unrestricted; device and dtype
        are preserved.
        """
        import numpy as np
        out = {}
        for index, level in enumerate(self.maps):
            array = np.ascontiguousarray(level.detach().to("cpu").numpy())
            out[index] = torch.from_numpy(array).to(device=level.device, dtype=level.dtype)
        return out


def _require_training_source(training_source_root: str | Path) -> Path:
    root = Path(training_source_root).resolve()
    src = root / "src"
    if not (src / "sparse_rtdetr" / "protocol_r18" / "model.py").is_file():
        raise ValueError(f"training source root lacks the frozen package: {src}")
    import sys
    sys.path.insert(0, str(src))
    import sparse_rtdetr  # noqa: F401  (import order is the point)
    resolved = Path(sparse_rtdetr.__file__).resolve()
    if not str(resolved).startswith(str(src)):
        raise RuntimeError(f"sparse_rtdetr resolved outside the training release: {resolved}")
    return root


def deterministic_device_setup(gpu_uuid: str, *, threads: int = 2) -> dict:
    """The frozen chain's process flags and the live runtime record it saved."""
    import os
    import random
    import numpy as np
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must be the frozen GPU UUID before CUDA init")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before CUDA init")
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized before deterministic setup")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(0)
    np.random.seed(0)
    torch.random.default_generator.manual_seed(0)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError("CUDA inventory differs from the frozen admission")
    torch.cuda.set_device(0)
    torch.cuda.default_generators[0].manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "uuid", None) is not None:
        if str(props.uuid).lower() != gpu_uuid.lower():
            raise ValueError("CUDA/NVML UUID mismatch")
    return {"gpu_uuid": gpu_uuid, "torch": str(torch.__version__),
            "torchvision": __import__("torchvision").__version__,
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu_name": props.name, "capability": [props.major, props.minor],
            "memory_bytes": props.total_memory, "cpu_threads": threads,
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "cudnn_tf32": False, "matmul_tf32": False,
            "cublas_workspace": ":4096:8", "power_clock_settings_written": False}


def restore_frozen_seed0_engine(*, training_source_root: str | Path,
                                train_contract: Mapping, binding_path: str | Path,
                                checkpoint_path: str | Path, checkpoint_sha256: str,
                                gpu_uuid: str) -> dict:
    """Audited restoration of the frozen seed0 e120 EMA engine (CUDA only)."""
    import json
    _require_training_source(training_source_root)
    from sparse_rtdetr.protocol_r18.worker import make_engine
    from sparse_rtdetr.protocol_r18.recovery_e088 import new_binding
    from sparse_rtdetr.protocol_r18 import checkpoint
    runtime = deterministic_device_setup(gpu_uuid)
    saved_binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    if saved_binding.get("scope") != FROZEN_SCOPE or saved_binding.get("run_id") != train_contract["task_id"]:
        raise ValueError("saved binding is not the frozen seed0 e120 formal binding")
    engine, fresh, _ = make_engine(dict(train_contract), runtime, FROZEN_SCOPE)
    binding = new_binding(dict(train_contract), fresh, "formal")
    if binding != saved_binding:
        raise ValueError("live endpoint construction differs from the saved binding")
    restored = checkpoint.restore(engine, binding, str(checkpoint_path), checkpoint_sha256)
    if not restored.get("full_state_exact") or engine.completed_updates != 36480 \
            or engine.ema.updates != 36480:
        raise ValueError("e120 restore is not exact")
    model = engine.ema.module
    model.eval()
    return {"engine": engine, "binding": binding, "restored": restored, "model": model,
            "runtime": runtime}


def load_cpu_ema_model_for_acceptance(*, training_source_root: str | Path,
                                      train_contract: Mapping,
                                      checkpoint_path: str | Path) -> dict:
    """CPU acceptance-only model: strict EMA weight load; no engine, no restore audit."""
    _require_training_source(training_source_root)
    from sparse_rtdetr.protocol_r18.model import Config, build
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if payload.get("binding", {}).get("scope") != FROZEN_SCOPE:
        raise ValueError("checkpoint is not the frozen seed0 e120 state")
    conf = Config()
    model, _criterion, info = build(conf, pretrained_path=train_contract["pretrained"]["path"])
    model.load_state_dict(payload["ema"]["module"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"model": model, "info": info,
            "ema_updates": int(payload["ema"]["updates"]),
            "scope": payload["binding"]["scope"]}


# --- frozen five-view forward (mirrors deploy.py of the legacy seed0 chain) -----
def load_original(data, row):
    import io
    from PIL import Image
    from sparse_rtdetr.localization.data import read_bound
    raw = read_bound(Path(data.spec["roots"]["images"]) / Path(row["relative_path"]).name,
                     row["image_sha256"], row["image_size_bytes"])
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.load()
    if (image.width, image.height) != (row["width"], row["height"]):
        raise ValueError("original image dimensions differ")
    return image


def pixels(data, image):
    from torchvision.transforms.v2 import functional as TF
    return TF.pil_to_tensor(data.transform.resize(image)).float() / 255.


def make_view_pixels(data, rows, *, sliced: bool = True):
    from slice_core import tile_windows
    views = [[] for _ in range(5 if sliced else 1)]
    geometry = []
    for row in rows:
        image = load_original(data, row)
        try:
            views[0].append(pixels(data, image))
            tiles = tile_windows(image.height, image.width)
            geometry.append([t.as_dict() for t in tiles])
            if sliced:
                for tile in tiles:
                    crop = image.crop(tile.crop_box)
                    try:
                        views[tile.view_id].append(pixels(data, crop))
                    finally:
                        crop.close()
        finally:
            image.close()
    return views, geometry


def five_view_forward(data, rows: Sequence, model: torch.nn.Module, *, tap: EncoderTap | None = None,
                      device: str = "cuda:0") -> dict:
    """One frozen batch (4 rows, five views) with the exact legacy decode path.

    Returns per-view CPU records (for bitwise comparison against frozen raw views), the
    per-row merged candidate pool (``select_view_candidates`` at the frozen budgets) and,
    when a tap is active, the captured P3 maps per view index.
    """
    import time as _time
    from slice_core import tile_windows, map_crop_xyxy, select_view_candidates, merge_candidates
    on_cuda = str(device).startswith("cuda")
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)
    sync()
    started = _time.perf_counter()
    views, geometry = make_view_pixels(data, rows, sliced=True)
    preprocessed = _time.perf_counter()
    candidates = [[] for _ in rows]
    records, forward_seconds, decode_seconds = [], 0.0, 0.0
    calls = 0
    for view_id, image_tensors in enumerate(views):
        x = torch.stack(image_tensors).to(device)
        sync()
        f0 = _time.perf_counter()
        if on_cuda:
            with torch.autocast("cuda", enabled=False), torch.inference_mode():
                output = model(x)
        else:
            with torch.inference_mode():
                output = model(x)
        sync()
        forward_seconds += _time.perf_counter() - f0
        calls += 1
        d0 = _time.perf_counter()
        if set(output) != {"pred_logits", "pred_boxes"}:
            raise ValueError("model output namespace differs")
        logits = output["pred_logits"].detach().cpu()
        boxes = output["pred_boxes"].detach().cpu()
        if logits.dtype != torch.float32 or boxes.dtype != torch.float32:
            raise ValueError("output precision differs")
        if not (logits.shape == (len(rows), 300, 10) and boxes.shape == (len(rows), 300, 4)):
            raise ValueError("output shape differs")
        for j, row in enumerate(rows):
            tile = tile_windows(row["height"], row["width"])[view_id - 1] if view_id else None
            width, height = (tile.width, tile.height) if tile else (row["width"], row["height"])
            xyxy = torch.cat((boxes[j, :, :2] - boxes[j, :, 2:] / 2,
                              boxes[j, :, :2] + boxes[j, :, 2:] / 2), -1)
            xyxy *= torch.tensor([width, height] * 2, dtype=torch.float32)
            mapped = map_crop_xyxy(xyxy.tolist(), tile) if tile else xyxy.tolist()
            candidates[j].extend(select_view_candidates(
                mapped, logits[j].sigmoid().numpy(), view_id=view_id))
        records.append({"view_id": view_id, "pred_logits": logits, "pred_boxes": boxes})
        decode_seconds += _time.perf_counter() - d0
        del x, output
    merged = [merge_candidates(pool) for pool in candidates]
    raw_whole = {"pred_logits": records[0]["pred_logits"], "pred_boxes": records[0]["pred_boxes"]}
    sync()
    end = _time.perf_counter()
    timing = {"e2e_seconds": end - started, "preprocess_seconds": preprocessed - started,
              "forward_seconds": forward_seconds, "decode_rank_seconds": decode_seconds}
    result = {"records": records, "raw_whole": raw_whole, "candidates": candidates,
              "merged": merged, "geometry": geometry, "forward_calls": calls,
              "timing": timing}
    if tap is not None:
        if tap.calls != calls:
            raise ValueError("tap call count differs from the forward count")
        result["p3_maps"] = tap.clone_maps()
    return result
