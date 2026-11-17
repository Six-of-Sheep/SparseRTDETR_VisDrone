"""Frozen common-protocol R18 seed0 e120 access for CPU probes (read-only).

The frozen development carrier of the r18 line is the RT-DETRv2-R18
common-protocol seed0 run (rtdetrv2-r18-common-protocol-seed0-e120-20261013-001);
its e120 EMA endpoint was locked under e088-io-20261014-001/formal and is the
exact state used by every downstream comparison (SAHI C line, slice fine-tune,
PDRA).  This module restores that state for CPU diagnostics only:

  * the training release source root is put on sys.path first and guarded so
    ``sparse_rtdetr`` can only resolve inside the release (same pattern as the
    audited restore in the former tools/pdra/pdra_tap.py and the legacy
    deploy.py chain);
  * the checkpoint file SHA, binding scope and EMA clock are checked before any
    build; the released ``ema.module`` state is strict-loaded into a freshly
    built CPU model, and every parameter is frozen.

Scope: CPU diagnostics only.  No CUDA, no optimizer, no training.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

FROZEN_SCOPE = "formal_R18_seed0_e120"
FROZEN_CHECKPOINT_SHA256 = "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757"
FROZEN_SOURCE_MANIFEST_SHA256 = "ee1204a2c47f768baebfb6f93f2227cdc1ddb59c3173bbc8bfc37363a08175d4"
FROZEN_EMA_UPDATES = 36480
FROZEN_TASK = "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_release_source(training_source_root):
    root = Path(training_source_root).resolve()
    src = root / "src"
    require((src / "sparse_rtdetr" / "protocol_r18" / "model.py").is_file(),
            "training source root lacks the frozen package: " + str(src))
    sys.path.insert(0, str(src))
    import sparse_rtdetr
    resolved = Path(sparse_rtdetr.__file__).resolve()
    require(str(resolved).startswith(str(src)),
            "sparse_rtdetr resolved outside the training release: " + str(resolved))
    return root


def load_frozen_cpu(training_source_root, contract_path, checkpoint_path,
                    source_manifest_path=None):
    """Strict CPU load of the frozen e120 EMA endpoint of the common-protocol seed0 run."""
    root = require_release_source(training_source_root)
    if source_manifest_path is not None:
        manifest_sha = file_sha256(source_manifest_path)
        require(manifest_sha == FROZEN_SOURCE_MANIFEST_SHA256,
                "training release source manifest differs: " + manifest_sha)
    import torch
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the CPU probe")
    checkpoint_sha = file_sha256(checkpoint_path)
    require(checkpoint_sha == FROZEN_CHECKPOINT_SHA256,
            "checkpoint is not the frozen common-protocol seed0 e120 state: " + checkpoint_sha)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    require(contract.get("task_id") == FROZEN_TASK,
            "contract is not the common-protocol seed0 run")
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    require(payload.get("binding", {}).get("scope") == FROZEN_SCOPE,
            "checkpoint binding scope differs")
    require(int(payload["ema"]["updates"]) == FROZEN_EMA_UPDATES,
            "frozen EMA clock differs")
    from sparse_rtdetr.protocol_r18.model import Config, build
    model, criterion, info = build(Config(), pretrained_path=contract["pretrained"]["path"])
    model.load_state_dict(payload["ema"]["module"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"model": model, "criterion": criterion, "info": info, "contract": contract,
            "binding": payload["binding"], "checkpoint_sha256": checkpoint_sha,
            "release_root": str(root)}


def development548(contract):
    from sparse_rtdetr.r0.evaluation_data import Development548
    return Development548(contract["data"]["development"])


def image_pixels_and_target(dev, row, position):
    """The production evaluation transform (resize-only) plus its GT target."""
    from PIL import Image
    from sparse_rtdetr.localization.data import read_bound
    spec = dev.spec
    raw = dev.raw(row)
    image_bytes = read_bound(Path(spec["roots"]["images"]) / Path(row["relative_path"]).name,
                             row["image_sha256"], row["image_size_bytes"])
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        require((image.width, image.height) == (row["width"], row["height"]),
                "image dimensions differ")
        pixels, target, trace = dev.transform(
            image, raw, epoch=120, seed=0, position=position,
            image_id=row["stable_image_id"], training=False)
    require(not trace["geometry_steps"], "evaluation transform must be resize-only")
    return pixels, target
