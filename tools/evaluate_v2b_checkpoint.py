#!/usr/bin/env python3
"""Read-only raw/EMA development evaluation for a verified REV1 contract.

This tool never constructs an optimizer step, writes to the checkout, or
accesses confirmatory/test data.  It evaluates only the checkpoint entries in
the sealed evaluation contract and writes canonical metric summaries to an
external evidence directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
from sparse_rtdetr.baseline.training_v2b_development import (
    _DevelopmentDataset, _coco_metrics, _primary_result,
    build_development_binding,
)
from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime


class EvaluationError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(str(value.dtype).encode("ascii")); h.update(b"\0")
        h.update(canonical(list(value.shape))); h.update(b"\0")
        h.update(value.numpy().tobytes()); h.update(b"\0")
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON object required: {path}")
    return value


def authority_file(authority: dict, relative: str) -> Path:
    root = Path(authority["runtime_locator"]["path"]).resolve(strict=True)
    path = (root / Path(*relative.split("/"))).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise EvaluationError(f"authority file is not a regular bound file: {relative}")
    return path


def authorities(contract_dir: Path) -> dict[str, dict]:
    doc = read_json(contract_dir / "external_authorities.json")
    return {item["authority_id"]: item for item in doc["authorities"]}


def checkpoint_entries(contract_dir: Path) -> list[dict]:
    doc = read_json(contract_dir / "evaluation_contract.json")
    entries = doc.get("checkpoints")
    if not isinstance(entries, list) or len(entries) != 12:
        raise EvaluationError("REV1 evaluation contract must contain 12 entries")
    return entries


def _target_predictions(dataset, model, postprocessor, selected, device):
    from faster_coco_eval import COCO

    evaluator = dataset.evaluator_type(COCO(copy.deepcopy(dataset.coco)), ["bbox"])
    evaluator.world_size = 1
    wrapper = VisDronePostProcessor(vendor_postprocessor=postprocessor)
    predictions = []
    for start in range(0, len(selected), 4):
        rows = selected[start:start + 4]
        items = [dataset.item(image) for image in rows]
        samples = torch.stack([item[0] for item in items]).to(device=device)
        sizes = torch.stack([item[1]["orig_size"] for item in items]).to(device=device)
        outputs = model(samples)
        if (not isinstance(outputs, dict) or list(outputs["pred_logits"].shape) != [len(items), 300, 10]
                or list(outputs["pred_boxes"].shape) != [len(items), 300, 4]):
            raise EvaluationError("model output geometry differs from REV1")
        if any(not bool(torch.isfinite(outputs[name]).all()) for name in ("pred_logits", "pred_boxes")):
            raise EvaluationError("nonfinite model output")
        processed = wrapper(outputs, sizes)
        cpu_predictions = {}
        for image, row in zip(rows, processed):
            boxes = row["boxes"]
            if (not bool(torch.isfinite(boxes).all())
                    or not bool((boxes[:, 2:] > boxes[:, :2]).all())):
                raise EvaluationError("invalid postprocessed boxes")
            cpu_predictions[image["id"]] = {key: value.detach().cpu()
                                             for key, value in row.items()}
        predictions.extend(evaluator.prepare_for_coco_detection(cpu_predictions))
        evaluator.update(cpu_predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    metrics, axes, precision, recall = _coco_metrics(evaluator.coco_eval["bbox"])
    return evaluator, predictions, metrics, axes, precision, recall


def evaluate_entry(repo_root: Path, auth: dict[str, dict], entry: dict,
                   output_dir: Path, device_name: str = "cpu") -> dict:
    checkpoint = entry["checkpoint"]
    selector = entry["weight_selector"]
    checkpoint_id = checkpoint["checkpoint_id"]
    cp_path = Path(entry["runtime_locator"]["path"]).resolve(strict=True)
    raw = cp_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checkpoint["sha256"]:
        raise EvaluationError(f"checkpoint drift: {checkpoint_id}")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise EvaluationError(f"checkpoint payload invalid: {checkpoint_id}")
    if not isinstance(payload.get("ema"), dict) or not isinstance(payload["ema"].get("module"), dict):
        raise EvaluationError(f"checkpoint EMA payload invalid: {checkpoint_id}")

    gt = auth["data:visdrone:development-gt-r3"]
    images = auth["data:visdrone:development-images-r3"]
    weights = auth["weights:presnet18-vd:imagenet-v1"]
    gt_root = Path(gt["runtime_locator"]["path"])
    image_root = Path(images["runtime_locator"]["path"])
    weight_root = Path(weights["runtime_locator"]["path"])
    config = V2BConfig(
        seed=checkpoint["seed"], input_size=checkpoint["resolution"],
        physical_batch_size=8, accumulation_steps=2, amp_dtype="bfloat16",
        bn_statistics="train", pretrained_required=True, num_denoising=100,
        sampling_backend="deterministic_gather",
    )
    runtime = prepare_runtime(seed=checkpoint["seed"])
    components = build_v2b_components(
        config, repo_root=repo_root, runtime=runtime,
        pretrained_path=weight_root / "ResNet18_vd_pretrained_from_paddle.pth",
        pretrained_sha256="911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720",
    )
    if device_name not in {"cpu", "cuda"}:
        raise EvaluationError("device must be cpu or cuda")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationError("CUDA is not available")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        components.model.to(device)
        components.ema.module.to(device)
        components.postprocessor.to(device)
    target = components.model if selector == "raw" else components.ema.module
    state = payload["model"] if selector == "raw" else payload["ema"]["module"]
    target.load_state_dict(state, strict=True)
    target.eval(); components.postprocessor.eval()
    binding = build_development_binding(
        repo_root=repo_root,
        annotation_file=gt_root / "development_coco.json",
        annotation_sha256="3d019173c2af009bbdce3928a25b05d42060b3855c3eea01d6257c3133bb2c05",
        manifest_file=gt_root / "development_manifest.json",
        manifest_sha256="5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd",
        image_root=image_root, input_size=checkpoint["resolution"],
    )
    dataset = _DevelopmentDataset(binding)
    selected = dataset.images
    with torch.inference_mode():
        evaluator, predictions, metrics, axes, precision, recall = _target_predictions(
            dataset, target, components.postprocessor, selected, device)
    primary = _primary_result(dataset, selected, predictions)
    body = {
        "schema_version": 1, "evaluation_contract": "P3-V2B-CONTRACT-REV1",
        "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_epoch": checkpoint["epoch"], "source_campaign": checkpoint["source_campaign"],
        "seed": checkpoint["seed"], "resolution": checkpoint["resolution"],
        "weight_selector": selector, "data_binding_sha256": binding["binding_sha256"],
        "evaluated_image_count": len(selected), "evaluated_ground_truth_count": sum(
            len(dataset.annotations[row["id"]]) for row in selected),
        "model_state_sha256": state_digest(state),
        "primary": {name: primary[name] for name in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")},
        "coco_secondary": {name: value["value"] for name, value in metrics.items()
                            if name in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_small")},
        "coco_axes_sha256": digest(axes),
        "prediction_sha256": digest(predictions),
        "precision_shape": list(precision.shape), "recall_shape": list(recall.shape),
        "device": str(device), "torch_version": str(torch.__version__),
    }
    body["result_sha256"] = digest(body)
    output = output_dir / f"{checkpoint_id}--{selector}.json"
    if output.exists():
        raise EvaluationError(f"refusing to overwrite result: {output}")
    output.write_bytes(canonical(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--selector", choices=("raw", "ema"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve(strict=True)
    contract_dir = args.contract_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=False)
    auth = authorities(contract_dir)
    entries = checkpoint_entries(contract_dir)
    if args.checkpoint_id is not None:
        entries = [entry for entry in entries
                   if entry["checkpoint"]["checkpoint_id"] == args.checkpoint_id]
    if args.selector is not None:
        entries = [entry for entry in entries if entry["weight_selector"] == args.selector]
    if not entries:
        raise SystemExit("no contract entries match the requested filter")
    rows = []
    try:
        for entry in entries:
            rows.append(evaluate_entry(repo_root, auth, entry, output_dir, args.device))
        pairs = {}
        for row in rows:
            pairs.setdefault(row["checkpoint_id"], {})[row["weight_selector"]] = row
        deltas = []
        for checkpoint_id, pair in sorted(pairs.items()):
            if set(pair) != {"raw", "ema"}:
                continue
            raw, ema = pair["raw"], pair["ema"]
            names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_small"]
            deltas.append({"checkpoint_id": checkpoint_id,
                           "ema_minus_raw": {name: ema["coco_secondary"].get(name, ema["primary"].get(name))
                                             - raw["coco_secondary"].get(name, raw["primary"].get(name))
                                             for name in names}})
        summary = {"schema_version": 1, "evaluation_contract": "P3-V2B-CONTRACT-REV1",
                   "entry_count": len(rows), "results": rows, "ema_minus_raw": deltas}
        summary["summary_sha256"] = digest(summary)
        (output_dir / "matrix.json").write_bytes(canonical(summary))
        print(json.dumps({"status": "PASS", "entry_count": len(rows),
                          "summary_sha256": summary["summary_sha256"],
                          "output_dir": str(output_dir)}, sort_keys=True, indent=2))
        return 0
    except Exception:
        # Preserve the failed evidence directory without replacing any prior run.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
