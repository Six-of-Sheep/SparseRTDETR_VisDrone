"""Synthetic CPU tests for the independent process-evidence launcher."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sparse_rtdetr.data_protocol.process_launcher import (
    ProcessLauncherContractError,
    _split_arg,
    contract_check,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "visdrone_protocol_v2.json"
V1_CONFIG = ROOT / "configs" / "visdrone_protocol_v1.json"
SOURCE = ROOT / "src"


FAKE_CLI = '''
import hashlib
import json
import os
import sys
from pathlib import Path

args = sys.argv

def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _sha(value):
    return hashlib.sha256(value).hexdigest()

def _output():
    args = sys.argv
    return Path(args[args.index("--output-dir") + 1])

def _write_success(
    unlisted=False,
    bad_inventory_hash=False,
    output_protocol="P3-VISDRONE-DATA-PROTOCOL-V2",
    top_level_test=True,
    generated=True,
    bad_input_sha=False,
    bad_split=False,
    bad_planner=False,
    bad_count=False,
    bad_type=False,
    bad_checks=False,
):
    output = _output()
    output.mkdir(parents=True)
    protocol_config = Path(args[args.index("--protocol-config") + 1])
    input_config_sha = _sha(protocol_config.read_bytes())
    config = {
        "schema_version": 1,
        "protocol_id": output_protocol,
        "real_conversion_outputs_generated": generated,
        "production_split_manifest_generated": generated,
        "confirmatory_metrics_accessed": False,
        "split": {
            "test": "disabled",
            "seed": 20260808,
            "salt": "P3-confirmatory-v1",
            "target_confirmatory_images": 647,
            "selection_policy": "wrong-policy" if bad_split else "feasibility_first_nearest_hash_prefix_v2",
            "planner": "wrong-planner" if bad_split or bad_planner else "plan_confirmatory_split_v2",
            "selection_allowed": False,
            "metrics_access_allowed": False,
            "single_final_access_only": True,
        },
    }
    if top_level_test:
        config["test_access_allowed"] = False
    else:
        config["data_identity"] = {"test_access_allowed": False}
    split = {
        "schema_version": 1,
        "seed": 20260808,
        "salt": "P3-confirmatory-v1",
        "target_image_count": 647,
        "selected_image_count": 647,
        "selected_group_list_sha256": "a" * 64,
        "selection_policy": "wrong-policy" if bad_split else "feasibility_first_nearest_hash_prefix_v2",
        "evaluated_prefix_count": True if bad_type else 0 if bad_count else 1,
        "feasible_prefix_count": 1,
        "selected_prefix_length": 1,
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
        "checks": [{"name": "synthetic", "passed": False if bad_split or bad_checks else True}],
    }
    config_bytes = _canonical(config)
    split_bytes = _canonical(split)
    (output / "config.json").write_bytes(config_bytes)
    (output / "split_plan.json").write_bytes(split_bytes)
    rows = [
        {"relative_path": "config.json", "size_bytes": len(config_bytes), "sha256": _sha(config_bytes)},
        {"relative_path": "split_plan.json", "size_bytes": len(split_bytes), "sha256": _sha(split_bytes)},
    ]
    inventory = {
        "schema_version": 1,
        "excluded_from_inventory": ["artifact_inventory.json", "completion.json"],
        "artifacts": rows,
        "canonical_inventory_sha256": _sha(_canonical(rows)),
    }
    inventory_bytes = _canonical(inventory)
    (output / "artifact_inventory.json").write_bytes(inventory_bytes)
    completion = {
        "schema_version": 1,
        "status": "COMPLETED",
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "config_sha256": _sha(config_bytes),
        "split_plan_sha256": _sha(split_bytes),
        "input_protocol_config_sha256": "a" * 64 if bad_input_sha else input_config_sha,
        "artifact_inventory_sha256": "0" * 64 if bad_inventory_hash else _sha(inventory_bytes),
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
        "project_test_split_historically_observed": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "completion_self_hash_included": False,
        "production_conversion_executed": True,
    }
    (output / "completion.json").write_bytes(_canonical(completion))
    if unlisted:
        (output / "unlisted.txt").write_text("not bound", encoding="utf-8")

mode = os.environ.get("FAKE_CHILD_MODE", "success")
print("fake-child-mode=" + mode, flush=True)
if mode in {"success", "session"}:
    if mode == "session":
        print(f"independent-session={os.getsid(0) == os.getpid()}", flush=True)
    _write_success()
    raise SystemExit(0)
if mode == "v1_output":
    _write_success(output_protocol="P3-VISDRONE-DATA-PROTOCOL-V1")
    raise SystemExit(0)
if mode == "nested_test":
    _write_success(top_level_test=False)
    raise SystemExit(0)
if mode == "generated_false":
    _write_success(generated=False)
    raise SystemExit(0)
if mode == "fake_input_sha":
    _write_success(bad_input_sha=True)
    raise SystemExit(0)
if mode == "split_drift":
    _write_success(bad_split=True)
    raise SystemExit(0)
if mode == "split_planner":
    _write_success(bad_planner=True)
    raise SystemExit(0)
if mode == "split_count":
    _write_success(bad_count=True)
    raise SystemExit(0)
if mode == "split_type":
    _write_success(bad_type=True)
    raise SystemExit(0)
if mode == "split_checks":
    _write_success(bad_checks=True)
    raise SystemExit(0)
if mode == "mutate_config":
    _write_success()
    protocol_config = Path(args[args.index("--protocol-config") + 1])
    protocol_config.write_bytes(protocol_config.read_bytes() + b"\\n")
    raise SystemExit(0)
if mode == "unlisted":
    _write_success(unlisted=True)
    raise SystemExit(0)
if mode == "binding":
    _write_success(bad_inventory_hash=True)
    raise SystemExit(0)
if mode == "invalid_inventory":
    output = _output()
    output.mkdir(parents=True)
    (output / "artifact_inventory.json").write_text("{}", encoding="utf-8")
    (output / "completion.json").write_text('{"status":"COMPLETED"}', encoding="utf-8")
    raise SystemExit(0)
if mode == "missing_inventory":
    output = _output()
    output.mkdir(parents=True)
    input_config_sha = _sha(Path(args[args.index("--protocol-config") + 1]).read_bytes())
    (output / "completion.json").write_text(json.dumps({
        "schema_version": 1, "status": "COMPLETED",
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "config_sha256": "0" * 64, "split_plan_sha256": "0" * 64,
        "input_protocol_config_sha256": input_config_sha, "artifact_inventory_sha256": "0" * 64,
        "selection_allowed": False, "metrics_access_allowed": False,
        "single_final_access_only": True, "project_test_split_historically_observed": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False, "completion_self_hash_included": False,
        "production_conversion_executed": True,
    }), encoding="utf-8")
    raise SystemExit(0)
if mode == "invalid_completion":
    output = _output()
    output.mkdir(parents=True)
    (output / "completion.json").write_text("not-json", encoding="utf-8")
    raise SystemExit(0)
if mode == "noentry":
    print("no entry completion", flush=True)
    raise SystemExit(0)
if mode == "contract":
    print("synthetic contract failure", file=sys.stderr, flush=True)
    raise SystemExit(2)
print("synthetic generic failure", file=sys.stderr, flush=True)
raise SystemExit(1)
'''


def _fake_source(root: Path) -> Path:
    package = root / "sparse_rtdetr" / "data_protocol"
    package.mkdir(parents=True)
    (root / "sparse_rtdetr" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(FAKE_CLI, encoding="utf-8")
    return root


class ProcessLauncherTests(unittest.TestCase):
    def _run_fake(self, mode: str):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        source = _fake_source(root / "source")
        output = root / "output"
        evidence = root / "evidence"
        config = CONFIG
        if mode == "mutate_config":
            config = root / "protocol_v2.json"
            config.write_bytes(CONFIG.read_bytes())
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "FAKE_CHILD_MODE": mode,
        }
        with mock.patch.dict(os.environ, env, clear=False):
            code = run(root / "data", output, config, ("train", "val"), evidence, Path(sys.executable), source)
        return temp, root, output, evidence, code

    def test_launcher_contract_check_is_data_free(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}, clear=False):
            result = contract_check(SOURCE)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["real_data_accessed"])
        self.assertFalse(result["output_directory_created"])
        self.assertFalse(result["process_evidence_created"])
        self.assertFalse(result["dataset_test_accessed_by_this_process"])

    def test_fake_cli_has_no_static_duplicate_string_keys(self):
        duplicates = []
        tree = ast.parse(FAKE_CLI)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = set()
            for key in node.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if key.value in seen:
                    duplicates.append((key.lineno, key.value))
                seen.add(key.value)
        self.assertEqual(duplicates, [])

    def test_launcher_rejects_visible_cuda_and_test_split(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1"}, clear=False):
            with self.assertRaises(ProcessLauncherContractError):
                contract_check(SOURCE)
        with self.assertRaises(Exception):
            _split_arg("test,val")

    def test_v1_input_config_is_rejected_before_child_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _fake_source(root / "source")
            output = root / "output"
            evidence = root / "evidence"
            with mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"},
                clear=False,
            ):
                with self.assertRaises(ProcessLauncherContractError):
                    run(root / "data", output, V1_CONFIG, ("train", "val"), evidence, Path(sys.executable), source)
            self.assertFalse(output.exists())
            self.assertFalse(evidence.exists())

    def test_success_exit_code_and_completion_bindings(self):
        temp, root, output, evidence, code = self._run_fake("success")
        try:
            self.assertEqual(code, 0)
            completion = json.loads((evidence / "process_completion.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "COMPLETED")
            self.assertEqual(completion["child_returncode"], 0)
            self.assertEqual((evidence / "process_exit_code.txt").read_bytes(), b"0\n")
            console = evidence / "process_console.log"
            self.assertEqual(completion["console"]["sha256"], hashlib.sha256(console.read_bytes()).hexdigest())
            self.assertTrue(completion["entry_success_accepted"])
            self.assertTrue(completion["entry_artifact_inventory"]["present"])
            self.assertEqual(completion["entry_completion_status"], "COMPLETED")
            self.assertTrue(completion["input_protocol_config_unchanged"])
            self.assertEqual(
                completion["input_protocol_config_before"]["sha256"],
                hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                completion["input_protocol_config_before"],
                completion["input_protocol_config_after"],
            )
            self.assertTrue((evidence / "process_inventory.json").is_file())
            self.assertTrue((output / "completion.json").is_file())
            process_inventory = json.loads((evidence / "process_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(
                completion["process_inventory"]["sha256"],
                hashlib.sha256((evidence / "process_inventory.json").read_bytes()).hexdigest(),
            )
            for row in process_inventory["artifacts"]:
                path = evidence / row["relative_path"]
                self.assertEqual(path.stat().st_size, row["size_bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        finally:
            temp.cleanup()

    def test_contract_and_generic_child_exit_codes_are_persisted(self):
        for mode, expected in (("contract", 2), ("generic", 1)):
            temp, root, output, evidence, code = self._run_fake(mode)
            try:
                self.assertEqual(code, expected)
                completion = json.loads((evidence / "process_completion.json").read_text(encoding="utf-8"))
                self.assertEqual(completion["child_returncode"], expected)
                self.assertEqual((evidence / "process_exit_code.txt").read_text(encoding="ascii"), f"{expected}\n")
                self.assertFalse(completion["entry_success_accepted"])
                self.assertTrue(completion["child_failure_preserved_in_console"])
                self.assertTrue((evidence / "partial_inventory.json").is_file())
                self.assertFalse(output.exists())
            finally:
                temp.cleanup()

    def test_zero_exit_entry_failures_are_not_reported_as_completed(self):
        expected = {
            "invalid_completion": "FAILED_ENTRY_COMPLETION_INVALID",
            "missing_inventory": "FAILED_ENTRY_ARTIFACT_INVENTORY_MISSING",
            "invalid_inventory": "FAILED_ENTRY_COMPLETION_INVALID",
            "binding": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "unlisted": "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID",
            "v1_output": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "nested_test": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "generated_false": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "fake_input_sha": "FAILED_ENTRY_INPUT_PROTOCOL_CONFIG_BINDING",
            "split_drift": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "split_planner": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "split_count": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "split_type": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
            "split_checks": "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH",
        }
        for mode, status in expected.items():
            temp, root, output, evidence, code = self._run_fake(mode)
            try:
                self.assertEqual(code, 1, mode)
                completion = json.loads((evidence / "process_completion.json").read_text(encoding="utf-8"))
                self.assertEqual(completion["status"], status, mode)
                self.assertEqual(completion["child_returncode"], 0, mode)
                self.assertFalse(completion["entry_success_accepted"], mode)
                self.assertTrue((evidence / "partial_inventory.json").is_file(), mode)
            finally:
                temp.cleanup()

    def test_input_protocol_config_change_fails_closed(self):
        temp, root, output, evidence, code = self._run_fake("mutate_config")
        try:
            self.assertEqual(code, 1)
            completion = json.loads((evidence / "process_completion.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "FAILED_ENTRY_INPUT_PROTOCOL_CONFIG_BINDING")
            self.assertFalse(completion["input_protocol_config_unchanged"])
            self.assertNotEqual(
                completion["input_protocol_config_before"],
                completion["input_protocol_config_after"],
            )
        finally:
            temp.cleanup()

    def test_child_has_independent_process_group(self):
        temp, root, output, evidence, code = self._run_fake("session")
        try:
            self.assertEqual(code, 0)
            self.assertIn(b"independent-session=True", (evidence / "process_console.log").read_bytes())
        finally:
            temp.cleanup()

    def test_missing_entry_completion_fails_closed(self):
        temp, root, output, evidence, code = self._run_fake("noentry")
        try:
            self.assertEqual(code, 1)
            completion = json.loads((evidence / "process_completion.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "FAILED_ENTRY_COMPLETION_MISSING")
            self.assertEqual(completion["child_returncode"], 0)
            self.assertFalse(completion["entry_success_accepted"])
            self.assertTrue((evidence / "partial_inventory.json").is_file())
        finally:
            temp.cleanup()

    def test_launcher_does_not_own_or_precreate_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _fake_source(root / "source")
            output = root / "output"
            evidence = root / "evidence"
            output.mkdir()
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}, clear=False):
                with self.assertRaises(ProcessLauncherContractError):
                    run(root / "data", output, CONFIG, ("train", "val"), evidence, Path(sys.executable), source)
            self.assertFalse(evidence.exists())
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
