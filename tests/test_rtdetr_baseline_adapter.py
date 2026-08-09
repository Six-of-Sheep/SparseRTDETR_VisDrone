from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")

from sparse_rtdetr.baseline.artifacts import resolve_runtime_paths, verify_r3_binding
from sparse_rtdetr.baseline.categories import (
    coco_category_to_model_label,
    map_coco_tensor_to_model,
    map_model_tensor_to_coco,
    model_label_to_coco_category,
)
from sparse_rtdetr.baseline.config import (
    build_baseline_config,
    build_r18_cpu_model,
    build_upstream_r18_cpu_model,
    canonical_config_bytes,
    load_isolated_vendor_config_dict,
)
from sparse_rtdetr.baseline.contract import (
    BaselineContractError,
    R3_ARTIFACT_INVENTORY_SHA256,
    R3_ARTIFACT_RELATIVE,
    R3_ENTRY_CANONICAL_INVENTORY_SHA256,
)
from sparse_rtdetr.baseline.dataset import VisDroneCocoDetection
from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.data_protocol.evaluation import Detection


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "rtdetrv2_pytorch"
PYTHON = sys.executable
ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": str(ROOT / "src"),
}


def test_contract_schema_and_frozen_sha():
    config = build_baseline_config(ROOT)
    contract = config["baseline_contract"]
    assert config["schema_version"] == 1
    assert contract["baseline_id"] == "rtdetrv2_r18_visdrone_baseline_v1"
    assert contract["num_classes"] == 10
    assert contract["num_queries"] == 300
    assert contract["feature_strides"] == [8, 16, 32]
    assert contract["deformable_sampling_points"] == [4, 4, 4]
    assert contract["pretrained"] is False
    assert contract["checkpoint"] is None
    assert config["r3_binding"]["artifact_inventory_sha256"] == R3_ARTIFACT_INVENTORY_SHA256
    assert contract["upstream_default_num_classes"] == 80
    assert contract["upstream_default_parameter_count"] == 20184464
    assert contract["visdrone_num_classes"] == 10
    assert contract["visdrone_parameter_count"] == 20094584
    assert contract["class_dependent_parameter_delta"] == 89880
    assert contract["parameter_count_scope"] == "random_initialized_no_checkpoint"


def test_category_bijection_is_exhaustive():
    assert [coco_category_to_model_label(value) for value in range(1, 11)] == list(range(10))
    assert [model_label_to_coco_category(value) for value in range(10)] == list(range(1, 11))


@pytest.mark.parametrize("value", [None, True, False, 0, 11, -1, "1", 1.0])
def test_invalid_coco_category_ids_fail_closed(value):
    with pytest.raises(BaselineContractError):
        coco_category_to_model_label(value)


@pytest.mark.parametrize("value", [None, True, False, 10, 11, -1, "0", 1.0])
def test_invalid_model_labels_fail_closed(value):
    with pytest.raises(BaselineContractError):
        model_label_to_coco_category(value)


def test_empty_target_and_tensor_mapping():
    empty = torch.empty((0,), dtype=torch.int64)
    assert map_coco_tensor_to_model(empty).numel() == 0
    assert map_model_tensor_to_coco(empty).numel() == 0
    assert torch.equal(map_coco_tensor_to_model(torch.tensor([1, 10])), torch.tensor([0, 9]))
    assert torch.equal(map_model_tensor_to_coco(torch.tensor([0, 9])), torch.tensor([1, 10]))


@pytest.mark.parametrize("labels", [torch.tensor([0]), torch.tensor([11]), torch.tensor([1.0]), torch.tensor([True])])
def test_invalid_coco_tensor_labels_fail_closed(labels):
    with pytest.raises(BaselineContractError):
        map_coco_tensor_to_model(labels)


@pytest.mark.parametrize("labels", [torch.tensor([-1]), torch.tensor([10]), torch.tensor([1.0]), torch.tensor([True])])
def test_invalid_model_tensor_labels_fail_closed(labels):
    with pytest.raises(BaselineContractError):
        map_model_tensor_to_coco(labels)


class _SyntheticVendorDataset:
    categories = [{"id": index, "name": str(index)} for index in range(1, 11)]

    def __len__(self):
        return 1

    def load_item(self, index):
        return "synthetic-image", {
            "labels": torch.tensor([1, 10], dtype=torch.int64),
            "boxes": torch.zeros((2, 4)),
        }


def test_dataset_maps_before_transform():
    observed = []

    def transform(image, target, dataset):
        observed.append(target["labels"].clone())
        return image, target, None

    dataset = VisDroneCocoDetection(
        "synthetic/images",
        "synthetic/annotations.json",
        transforms=transform,
        vendor_dataset=_SyntheticVendorDataset(),
        role="train_core",
    )
    image, target = dataset[0]
    assert image == "synthetic-image"
    assert torch.equal(observed[0], torch.tensor([0, 9]))
    assert torch.equal(target["labels"], torch.tensor([0, 9]))


@pytest.mark.parametrize("role", [None, "confirmatory", "test", "raw_train", "raw_val", "unknown", True, 1])
def test_dataset_adapter_enforces_role_gate(role):
    with pytest.raises(BaselineContractError):
        VisDroneCocoDetection(
            "synthetic/images",
            "synthetic/annotations.json",
            role=role,
            vendor_dataset=_SyntheticVendorDataset(),
        )


def test_postprocessor_preserves_vendor_math_and_maps_labels():
    torch.manual_seed(7)
    outputs = {
        "pred_logits": torch.randn((1, 300, 10), dtype=torch.float32),
        "pred_boxes": torch.rand((1, 300, 4), dtype=torch.float32),
    }
    sizes = torch.tensor([[640, 640]], dtype=torch.float32)
    from sparse_rtdetr.baseline.config import _vendor_path

    with _vendor_path(VENDOR):
        from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor

    vendor = RTDETRPostProcessor(10, True, 300, False)
    expected = vendor(outputs, sizes)[0]
    actual = VisDronePostProcessor(vendor_root=VENDOR)(outputs, sizes)[0]
    assert torch.equal(actual["labels"], expected["labels"] + 1)
    assert torch.equal(actual["boxes"], expected["boxes"])
    assert torch.equal(actual["scores"], expected["scores"])
    assert actual["labels"].shape == (300,)


def test_postprocessor_detection_schema_and_no_nms():
    class FakePostprocessor(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = 10
            self.num_top_queries = 300
            self.use_focal_loss = True
            self.remap_mscoco_category = False

        def forward(self, outputs, sizes):
            return [{
                "labels": torch.tensor([0, 9]),
                "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                "scores": torch.tensor([0.25, 0.75]),
            }]

    adapter = VisDronePostProcessor(vendor_postprocessor=FakePostprocessor())
    result = adapter({"pred_logits": torch.empty(0), "pred_boxes": torch.empty(0)}, torch.empty(0))
    detections = adapter.to_detections(result, ["stable-001"])
    assert adapter.nms is False
    assert detections == (
        Detection("stable-001", 1, (1.0, 2.0, 3.0, 4.0), 0.25),
        Detection("stable-001", 10, (5.0, 6.0, 7.0, 8.0), 0.75),
    )


@pytest.mark.parametrize("kwargs", [
    {"num_classes": 80},
    {"num_top_queries": 299},
    {"use_focal_loss": False},
    {"remap": True},
])
def test_postprocessor_rejects_identity_drift(kwargs):
    class FakePostprocessor(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = kwargs.get("num_classes", 10)
            self.num_top_queries = kwargs.get("num_top_queries", 300)
            self.use_focal_loss = kwargs.get("use_focal_loss", True)
            self.remap_mscoco_category = kwargs.get("remap", False)

    with pytest.raises(BaselineContractError):
        VisDronePostProcessor(vendor_postprocessor=FakePostprocessor())


def test_detection_gate_rejects_bad_id_labels_and_nonfinite_values():
    class FakePostprocessor(nn.Module):
        num_classes = 10
        num_top_queries = 300
        use_focal_loss = True
        remap_mscoco_category = False

        def forward(self, outputs, sizes):
            return []

    adapter = VisDronePostProcessor(vendor_postprocessor=FakePostprocessor())
    base = {
        "labels": torch.tensor([1]),
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "scores": torch.tensor([0.5]),
    }
    for image_id in ("", 1, None):
        with pytest.raises(BaselineContractError):
            adapter.to_detections([base], [image_id])
    for result in (
        {**base, "labels": torch.tensor([0])},
        {**base, "boxes": torch.tensor([[float("nan"), 2.0, 3.0, 4.0]])},
        {**base, "scores": torch.tensor([float("inf")])},
        {**base, "boxes": torch.zeros((1, 5))},
        {**base, "scores": torch.zeros((2,))},
    ):
        with pytest.raises(BaselineContractError):
            adapter.to_detections([result], ["stable-001"])


def test_runtime_roles_and_path_gate():
    train = resolve_runtime_paths(ROOT, "/portable/visdrone/train_core", "train_core")
    development = resolve_runtime_paths(ROOT, "/portable/visdrone/development", "development")
    assert train.annotation_file.name == "train_core_coco.json"
    assert development.annotation_file.name == "development_coco.json"
    for role in ("confirmatory", "test", "raw_train", "raw_val", "unknown"):
        with pytest.raises(BaselineContractError):
            resolve_runtime_paths(ROOT, "/portable/visdrone/data", role)
    with pytest.raises(BaselineContractError):
        resolve_runtime_paths(ROOT, "/portable/visdrone/test/images", "development")


def test_portable_config_is_deterministic_and_unshared():
    first = build_baseline_config(ROOT)
    second = build_baseline_config(ROOT)
    assert canonical_config_bytes(first) == canonical_config_bytes(second)
    first["model"]["num_classes"] = 999
    assert second["model"]["num_classes"] == 10
    assert "/" + "media" not in canonical_config_bytes(second).decode("utf-8")


def test_config_parameter_contract_rejects_types_and_contradictions():
    from sparse_rtdetr.baseline.config import _validate_contract

    valid = build_baseline_config(ROOT)
    for field, invalid in {
        "visdrone_num_classes": True,
        "visdrone_parameter_count": "20094584",
        "class_dependent_parameter_delta": -1,
        "upstream_default_parameter_count": None,
    }.items():
        candidate = json.loads(json.dumps(valid))
        candidate["baseline_contract"][field] = invalid
        with pytest.raises(BaselineContractError):
            _validate_contract(candidate)
    candidate = json.loads(json.dumps(valid))
    candidate["baseline_contract"]["class_dependent_parameter_delta"] = 1
    with pytest.raises(BaselineContractError):
        _validate_contract(candidate)


def test_r3_binding_and_metadata_drift_fail_closed():
    binding = verify_r3_binding(ROOT)
    assert binding.artifact_root == (ROOT / R3_ARTIFACT_RELATIVE).resolve()
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        artifact_root = temp_root / R3_ARTIFACT_RELATIVE
        artifact_root.mkdir(parents=True)
        source_root = ROOT / R3_ARTIFACT_RELATIVE
        for name in ("completion.json", "artifact_inventory.json", "config.json", "category_contract.json", "source_identity.json"):
            shutil.copy2(source_root / name, artifact_root / name)
        (artifact_root / "config.json").write_bytes((artifact_root / "config.json").read_bytes() + b"\n")
        with pytest.raises(BaselineContractError):
            verify_r3_binding(temp_root)


def test_vendor_config_loads_are_isolated_and_deterministic():
    first = load_isolated_vendor_config_dict(ROOT)
    second = load_isolated_vendor_config_dict(ROOT)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(second, sort_keys=True, separators=(",", ":"))
    first["PResNet"]["isolation_probe"] = True
    assert "isolation_probe" not in second["PResNet"]


def test_baseline_exports_have_no_duplicate_names():
    module = importlib.import_module("sparse_rtdetr.baseline")
    exported = list(module.__all__)
    assert len(exported) == len(set(exported))
    assert len([name for name in dir(module) if name == "map_model_tensor_to_coco"]) == 1


def test_import_has_no_vendor_model_or_cuda_side_effect():
    script = "import sys, torch; import sparse_rtdetr.baseline; print('src' in sys.modules, torch.cuda.is_initialized())"
    result = subprocess.run([PYTHON, "-c", script], env=ENV, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False False"


def test_r18_cpu_model_is_finite_and_has_frozen_parameter_count(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden weight/CUDA operation")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    model = build_r18_cpu_model(ROOT)
    params = list(model.parameters())
    assert sum(parameter.numel() for parameter in params) == 20094584
    assert all(bool(torch.isfinite(parameter).all()) for parameter in params)
    assert all(parameter.device.type == "cpu" for parameter in params)
    assert not hasattr(model, "optimizer")
    assert not hasattr(model, "scheduler")


def test_r18_parameter_contract_reconciliation_by_named_shapes():
    upstream = build_upstream_r18_cpu_model(ROOT)
    visdrone = build_r18_cpu_model(ROOT)
    upstream_named = dict(upstream.named_parameters())
    visdrone_named = dict(visdrone.named_parameters())
    assert sum(parameter.numel() for parameter in upstream.parameters()) == 20184464
    assert sum(parameter.numel() for parameter in visdrone.parameters()) == 20094584
    changed = {
        name for name in upstream_named
        if upstream_named[name].shape != visdrone_named[name].shape
    }
    expected_changed = {
        "decoder.denoising_class_embed.weight",
        "decoder.enc_score_head.weight",
        "decoder.enc_score_head.bias",
        "decoder.dec_score_head.0.weight",
        "decoder.dec_score_head.0.bias",
        "decoder.dec_score_head.1.weight",
        "decoder.dec_score_head.1.bias",
        "decoder.dec_score_head.2.weight",
        "decoder.dec_score_head.2.bias",
    }
    assert changed == expected_changed
    delta = sum(upstream_named[name].numel() - visdrone_named[name].numel() for name in changed)
    assert delta == 89880
    assert delta == 70 * (4 * 257 + 256)
    for name in upstream_named:
        if name not in changed:
            assert upstream_named[name].shape == visdrone_named[name].shape
    assert not any(name.startswith("backbone.") for name in changed)
    assert not any(name.startswith("encoder.") for name in changed)
    assert not any("dec_bbox_head" in name for name in changed)
    assert visdrone.decoder.num_classes == 10
    assert visdrone.decoder.num_layers == 3
    assert visdrone.decoder.num_queries == 300
    assert all(parameter.device.type == "cpu" and bool(torch.isfinite(parameter).all()) for parameter in upstream.parameters())
    assert all(parameter.device.type == "cpu" and bool(torch.isfinite(parameter).all()) for parameter in visdrone.parameters())


def test_repository_checker_rejects_unknown_machine_model_large_and_symlink_files():
    spec = importlib.util.spec_from_file_location("repository_contract_check", ROOT / "tools" / "repository_contract_check.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "rogue.py").write_text("import torch\n", encoding="utf-8")
        machine_path = "/" + "media/example"
        (root / "machine.py").write_text(f"PATH = {machine_path!r}\n", encoding="utf-8")
        (root / "large.bin").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
        (root / "target.txt").write_text("target", encoding="utf-8")
        (root / "link.txt").symlink_to(root / "target.txt")
        files, policy_failures = checker._file_policy(root)
        assert "rogue.py" not in (checker.ALLOWED_FILES | checker.BASELINE_FILES)
        assert any("large file" in failure for failure in policy_failures)
        assert any("symlink" in failure for failure in policy_failures)
        source_failures = checker._source_policy_failures(root, files)
        assert "model import: rogue.py" in source_failures
        assert "machine-specific path: machine.py" in source_failures


def test_repository_contract_checker_passes_after_allowlist_update():
    result = subprocess.run(
        [PYTHON, "tools/repository_contract_check.py"],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
