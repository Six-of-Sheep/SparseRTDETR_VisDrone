"""Synthetic CPU tests for the production converter boundary."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sparse_rtdetr.data_protocol.cli import contract_check, main
from sparse_rtdetr.data_protocol.protocol import protocol_v2_schema
from sparse_rtdetr.data_protocol.converter import (
    ConversionContractError,
    _chunk_jsonl,
    build_conversion_bundle,
    load_protocol_v2_config,
    record_from_bytes,
    _sequence_key,
    validate_protocol_config,
    validate_protocol_v2_config,
    validate_run_arguments,
    write_conversion_artifacts,
    write_failure_evidence,
)
from sparse_rtdetr.data_protocol.process_launcher import _input_protocol_config_ref, _validate_entry
from sparse_rtdetr.data_protocol.split import plan_confirmatory_split_v2


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs" / "visdrone_protocol_v1.json"
CONFIG = ROOT / "configs" / "visdrone_protocol_v2.json"


def _record(path: str, annotation: bytes, image: bytes = b"synthetic-image", split: str = "train"):
    return record_from_bytes(
        split,
        f"{split}/images/{path}",
        image,
        annotation,
        annotation_relative_path=f"{split}/annotations/{Path(path).stem}.txt",
        width=640,
        height=480,
    )


def _bundle():
    complex_annotation = (
        b"1,2,3,4,1,1,0,0,\n"
        b"1,2,3,4,1,1,0,0,\n"
        b"0,0,20,20,1,0,0,0\n"
        b"4,4,4,4,1,11,0,0\n"
        b"4,4,4,4,0,1,0,0\n"
        b"4,4,0,4,1,1,0,0\n"
    )
    formal = b"1,2,40,40,1,1,0,0\n"
    train = (
        _record("0000001_00001_d_0000001.jpg", complex_annotation),
        _record("0000002_00001_d_0000002.jpg", formal),
        _record("0000003_00001_d_0000003.jpg", formal, image=b"another-image"),
        _record("0000004_00001_d_0000004.jpg", formal, image=b"third-image"),
    )
    val = (_record("0000005_00001_d_0000001.jpg", formal, image=b"val-image", split="val"),)
    return build_conversion_bundle(
        train,
        val,
        planner_train_image_count=4,
        planner_target_ratio=0.5,
        planner_tolerance=1.0,
    )


def _v2_bundle():
    formal = b"".join(
        f"1,2,40,40,1,{raw},0,0\n".encode("ascii") for raw in range(1, 11)
    )
    train = tuple(
        _record(
            f"000000{index}_00001_d_000000{index}.jpg",
            formal,
            image=f"v2-image-{index}".encode(),
        )
        for index in range(1, 5)
    )
    val = (_record("0000005_00001_d_0000005.jpg", formal, image=b"v2-val", split="val"),)
    return build_conversion_bundle(
        train,
        val,
        planner_train_image_count=6471,
        planner_target_ratio=0.10,
        planner_tolerance=0.05,
        planner=plan_confirmatory_split_v2,
        protocol_definition=protocol_v2_schema,
        protocol_id="P3-VISDRONE-DATA-PROTOCOL-V2",
        protocol_config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    )


class VisDroneConverterTests(unittest.TestCase):
    def test_official_sequence_key_requires_strict_filename(self):
        self.assertEqual(_sequence_key("train/images/0000001_02999_d_0000005.jpg"), "0000001")
        for name in (
            "000001_02999_d_0000005.jpg",
            "00000001_02999_d_0000005.jpg",
            "000000a_02999_d_0000005.jpg",
            "0000001_02999_x_0000005.jpg",
            "0000001_02999_d_0000005.png",
            "0000001_02999_d.jpg",
        ):
            with self.assertRaises(ConversionContractError):
                _sequence_key(f"train/images/{name}")

    def test_contract_check_is_data_free_and_cuda_hidden(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            result = contract_check()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["real_data_accessed"])
        self.assertFalse(result["output_directory_created"])
        self.assertFalse(result["torch_imported"])
        self.assertFalse(result["dataset_or_dataloader_constructed"])
        self.assertFalse(result["dataset_test_accessed_by_this_process"])

    def test_cli_contract_check_requires_hidden_cuda(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaises(ConversionContractError):
                contract_check()

    def test_v1_v2_config_identity_cannot_be_interchanged(self):
        v2 = load_protocol_v2_config(ROOT / "configs" / "visdrone_protocol_v2.json")
        self.assertEqual(v2["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V2")
        validate_protocol_v2_config(v2)
        with self.assertRaises(ConversionContractError):
            validate_protocol_v2_config(json.loads(V1_CONFIG.read_text(encoding="utf-8")))
        with self.assertRaises(ConversionContractError):
            validate_protocol_config(protocol_v2_schema())

    def test_v2_bundle_binds_protocol_and_split_policy(self):
        formal = b"".join(
            f"1,2,40,40,1,{raw},0,0\n".encode("ascii") for raw in range(1, 11)
        )
        train = tuple(
            _record(f"000000{index}_00001_d_000000{index}.jpg", formal, image=f"image-{index}".encode())
            for index in range(1, 5)
        )
        val = (_record("0000005_00001_d_0000005.jpg", formal, image=b"val", split="val"),)
        bundle = build_conversion_bundle(
            train,
            val,
            planner_train_image_count=4,
            planner_target_ratio=0.5,
            planner_tolerance=0.05,
            planner=plan_confirmatory_split_v2,
            protocol_definition=protocol_v2_schema,
            protocol_id="P3-VISDRONE-DATA-PROTOCOL-V2",
            protocol_config_sha256="a" * 64,
        )
        self.assertEqual(bundle.protocol_id, "P3-VISDRONE-DATA-PROTOCOL-V2")
        self.assertEqual(bundle.selection_policy, "feasibility_first_nearest_hash_prefix_v2")
        self.assertEqual(json.loads(bundle.files["config.json"])["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V2")
        split_plan = json.loads(bundle.files["split_plan.json"])
        self.assertEqual(split_plan["selection_policy"], "feasibility_first_nearest_hash_prefix_v2")
        self.assertGreaterEqual(split_plan["feasible_prefix_count"], 1)
        with tempfile.TemporaryDirectory() as temp:
            completion = write_conversion_artifacts(Path(temp) / "v2", bundle)
            self.assertEqual(completion["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V2")
            self.assertEqual(completion["input_protocol_config_sha256"], "a" * 64)

    def test_real_v2_writer_output_is_accepted_by_launcher_validator(self):
        bundle = _v2_bundle()
        input_before = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "v2"
            completion = write_conversion_artifacts(output, bundle)
            output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(output_config["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V2")
            self.assertTrue(output_config["real_conversion_outputs_generated"])
            self.assertTrue(output_config["production_split_manifest_generated"])
            self.assertFalse(output_config["confirmatory_metrics_accessed"])
            self.assertFalse(output_config["test_access_allowed"])
            before_ref = _input_protocol_config_ref(CONFIG)
            after_ref = _input_protocol_config_ref(CONFIG)
            result = _validate_entry(output, before_ref, after_ref)
            self.assertTrue(result["entry_success_accepted"])
            self.assertEqual(completion["input_protocol_config_sha256"], before_ref["sha256"])
        self.assertEqual(CONFIG.read_bytes(), input_before)

    def test_cli_run_requires_all_arguments_and_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "P3_VISDRONE_PRODUCTION_CONVERSION_AUTHORIZED": ""}, clear=False):
                with self.assertRaises(ConversionContractError):
                    validate_run_arguments(root, root / "output", CONFIG, ("train", "val"))
            self.assertFalse((root / "output").exists())

    def test_preexisting_output_and_test_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            with mock.patch.dict(os.environ, {"P3_VISDRONE_PRODUCTION_CONVERSION_AUTHORIZED": "1"}, clear=False):
                with self.assertRaises(ConversionContractError):
                    validate_run_arguments(root, output, CONFIG, ("train", "val"))
            with self.assertRaises(ConversionContractError):
                validate_run_arguments(root, root / "new", CONFIG, ("test", "val"))

    def test_global_integer_ids_are_contiguous_and_lineage_ids_are_sha(self):
        bundle = _bundle()
        ids = bundle.coco_image_ids
        self.assertEqual(sorted(ids.values()), list(range(1, len(ids) + 1)))
        self.assertEqual(ids[("train", "train/images/0000001_00001_d_0000001.jpg")], 1)
        self.assertEqual(ids[("val", "val/images/0000005_00001_d_0000001.jpg")], 5)
        for record in bundle.train_records + bundle.val_records:
            self.assertEqual(len(record.stable_image_id), 64)
            for item in record.lineage:
                self.assertEqual(len(item.annotation_id), 64)
        self.assertEqual(len(set(ids.values())), len(ids))

    def test_keep_ignore_filter_and_sidecar_conservation(self):
        bundle = _bundle()
        audit = json.loads(bundle.files["conversion_audit.json"])
        self.assertTrue(audit["train_core"]["row_conservation_pass"])
        self.assertTrue(audit["development"]["row_conservation_pass"])
        self.assertTrue(audit["confirmatory"]["row_conservation_pass"])
        self.assertTrue(audit["train_core"]["keep_equals_coco"])
        self.assertTrue(audit["development"]["keep_equals_coco"])
        for lineage_name, sidecar_name in (
            ("train_core_lineage.jsonl", "train_core_ignore_regions.jsonl"),
            ("development_lineage.jsonl", "development_ignore_regions.jsonl"),
        ):
            lineage = [json.loads(line) for line in bundle.files[lineage_name].splitlines()]
            expected = [item for item in lineage if item["ignore_region"] is True]
            actual = [json.loads(line) for line in bundle.files[sidecar_name].splitlines()]
            self.assertEqual(actual, expected)
        train_lineage = [
            json.loads(line)
            for name in ("train_core_lineage.jsonl", "confirmatory_lineage.jsonl")
            for line in bundle.files[name].splitlines()
        ]
        self.assertTrue(any(item["ignore_region"] is True for item in train_lineage))
        coco = json.loads(bundle.files["train_core_coco.json"])
        self.assertTrue(all(item["category_id"] in range(1, 11) for item in coco["annotations"]))
        self.assertTrue(all(item["category_id"] != 0 for item in coco["annotations"]))

    def test_confirmatory_is_sealed_and_not_in_development_coco(self):
        bundle = _bundle()
        sealed = json.loads(bundle.files["confirmatory_sealed_manifest.json"])
        self.assertFalse(sealed["selection_allowed"])
        self.assertFalse(sealed["metrics_access_allowed"])
        self.assertTrue(sealed["single_final_access_only"])
        development = json.loads(bundle.files["development_coco.json"])
        confirm_ids = {item["coco_image_id"] for item in sealed["records"]}
        self.assertTrue(confirm_ids.isdisjoint({item["id"] for item in development["images"]}))
        self.assertNotIn("confirmatory_coco.json", bundle.files)

    def test_chunk_manifest_is_deterministic_and_below_limit(self):
        records = ({"index": index, "payload": "x" * 5} for index in range(20))
        first = _chunk_jsonl(records, "manifest", max_bytes=80)
        second = _chunk_jsonl(({"index": index, "payload": "x" * 5} for index in range(20)), "manifest", max_bytes=80)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(len(payload) <= 80 for payload in first.values()))

    def test_atomic_success_write_inventory_and_completion_binding(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "conversion"
            completion = write_conversion_artifacts(output, bundle)
            self.assertEqual(completion["status"], "COMPLETED")
            inventory_bytes = (output / "artifact_inventory.json").read_bytes()
            self.assertEqual(completion["artifact_inventory_sha256"], hashlib.sha256(inventory_bytes).hexdigest())
            inventory = json.loads(inventory_bytes)
            self.assertNotIn("artifact_inventory.json", {item["relative_path"] for item in inventory["artifacts"]})
            self.assertNotIn("completion.json", {item["relative_path"] for item in inventory["artifacts"]})
            self.assertFalse(completion["completion_self_hash_included"])
            self.assertFalse(completion["dataset_test_accessed_by_this_process"])
            self.assertFalse((output / "error.json").exists())
            with self.assertRaises(ConversionContractError):
                write_conversion_artifacts(output, bundle)

    def test_two_writes_have_identical_artifact_bytes(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            write_conversion_artifacts(first, bundle)
            write_conversion_artifacts(second, bundle)
            first_files = {path.name: path.read_bytes() for path in first.iterdir()}
            second_files = {path.name: path.read_bytes() for path in second.iterdir()}
            self.assertEqual(first_files, second_files)

    def test_failure_evidence_preserves_failure_without_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failed"
            write_failure_evidence(output, RuntimeError("synthetic failure"))
            self.assertTrue((output / "error.json").is_file())
            self.assertTrue((output / "partial_inventory.json").is_file())
            completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
            error = json.loads((output / "error.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "FAILED")
            self.assertEqual(error["exception_type"], "RuntimeError")
            self.assertNotEqual(completion["status"], "COMPLETED")

    def test_converter_sources_have_no_framework_or_model_imports(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/sparse_rtdetr/data_protocol").glob("*.py"))
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+(?:torch|torchvision|ultralytics|rtdetr)\b")
        self.assertNotIn("DataLoader(", source)
        self.assertNotIn("torch.load", source)

    def test_cli_main_contract_check_emits_no_output_directory(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            self.assertEqual(main(["contract-check"]), 0)


if __name__ == "__main__":
    unittest.main()
