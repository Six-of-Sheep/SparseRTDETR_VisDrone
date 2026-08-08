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
from sparse_rtdetr.data_protocol.converter import (
    ConversionContractError,
    _chunk_jsonl,
    build_conversion_bundle,
    record_from_bytes,
    validate_run_arguments,
    write_conversion_artifacts,
    write_failure_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "visdrone_protocol_v1.json"


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
        _record("a_0001.jpg", complex_annotation),
        _record("b_0001.jpg", formal),
        _record("c_0001.jpg", formal, image=b"another-image"),
        _record("d_0001.jpg", formal, image=b"third-image"),
    )
    val = (_record("v_0001.jpg", formal, image=b"val-image", split="val"),)
    return build_conversion_bundle(
        train,
        val,
        planner_train_image_count=4,
        planner_target_ratio=0.5,
        planner_tolerance=1.0,
    )


class VisDroneConverterTests(unittest.TestCase):
    def test_contract_check_is_data_free_and_cuda_hidden(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            result = contract_check()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["real_data_accessed"])
        self.assertFalse(result["output_directory_created"])
        self.assertFalse(result["torch_imported"])
        self.assertFalse(result["dataset_or_dataloader_constructed"])
        self.assertFalse(result["test_accessed"])

    def test_cli_contract_check_requires_hidden_cuda(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaises(ConversionContractError):
                contract_check()

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
        self.assertEqual(ids[("train", "train/images/a_0001.jpg")], 1)
        self.assertEqual(ids[("val", "val/images/v_0001.jpg")], 5)
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
        ignore_lines = bundle.files["train_core_ignore_regions.jsonl"].splitlines()
        self.assertTrue(ignore_lines)
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
            self.assertFalse(completion["test_accessed"])
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
