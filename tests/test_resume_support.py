"""Pure CPU lineage checks using actual JSON files and real SHA references.

These fixtures certify reference/identity plumbing only. They are not detector,
checkpoint-payload, GPU, or scientific acceptance evidence.
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("resume_support_under_test", ROOT / "resume_support.py")
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)
from sparse_rtdetr.localization.runio import digest, read, reference


class ResumeSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = self.base / "original" / "seed1"
        self.formal = self.old / "formal"
        self.a = self.old / "infer-A"
        self.formal.mkdir(parents=True)
        self.a.mkdir()
        self.parent_path = self.old / "execution-contract.json"
        self.cert_path = self.base / "PARENT_REUSE.json"
        self.parent = {
            "task_id": "r18-sahi-seed12-20261016-001-seed1",
            "seed": 1,
            "scope": "formal_R18_seed1_e120",
            "source_sha256": "1" * 64,
            "job_root": str(self.old),
            "config": {"seed": 1, "epochs": 120, "input_hw": [768, 1344]},
            "data_sha256": "3" * 64,
            "environment": {"torch": "fixture-only-not-imported"},
            "gpu_uuid": "GPU-fixture-not-accessed",
            "data": {"development": {"manifest": "fixed-development.json"}},
            "pretrained": {"path": "certified-imagenet.pth"},
            "development_ids": [format(i, "064x") for i in range(548)],
            "fixed64": [format(i, "064x") for i in range(64)],
            "evaluation_protocol": "fixed-evaluation.json",
            "evaluation_protocol_sha256": "4" * 64,
            "materials": [{"path": "fixture-material", "sha256": "5" * 64}],
            "plan": {"path": "PLAN-v002.json", "sha256": "6" * 64},
        }
        self._write_parent()
        binding_ref = self.write(self.formal / "binding.json", {
            "scope": self.parent["scope"], "source_sha256": self.parent["source_sha256"],
            "contract_sha256": self.parent["contract_sha256"],
        })
        self.lock = {
            "status": "R18_COMMON_E120_EMA_LOCKED", "seed": 1, "epoch": 120,
            "updates": 36480, "actual_updates": 36480,
            "source_sha256": self.parent["source_sha256"],
            "contract_sha256": self.parent["contract_sha256"],
            "used_acceptance_state": False, "binding": binding_ref,
            "checkpoint": {"path": "SERVER_ONLY_fixture-not-a-checkpoint.pth",
                           "sha256": "7" * 64, "completed_updates": 36480},
        }
        self.write(self.formal / "endpoint-lock.json", self.lock)
        self.write(self.formal / "RESULT.json", self.lock)
        self.write(self.formal / "runtime-archive-index.json", {"scope": "CPU_reference_fixture"})
        self.formal_closed = self.closed(self.formal, "formal")
        self.write(self.formal / "CLOSED.json", self.formal_closed)
        self.write(self.a / "RESULT.json", {
            "status": "PASS", "phase": "infer-A", "seed": 1,
            "source_sha256": self.parent["source_sha256"],
            "contract_sha256": self.parent["contract_sha256"],
            "checkpoint_sha256": self.lock["checkpoint"]["sha256"],
            "images": 548,
        })
        self.write(self.a / "runtime-archive-index.json", {"scope": "CPU_A_reference_fixture"})
        self.a_closed = self.closed(self.a, "infer-A")
        self.write(self.a / "CLOSED.json", self.a_closed)
        self.c = copy.deepcopy(self.parent)
        self.c.update(source_sha256="2" * 64, job_root=str(self.base / "resume" / "seed1"),
                      contract_sha256="8" * 64)
        self.c["parent_reuse"] = {
            "kind": "certified_closed_seed1_e120_and_A548",
            "contract": reference(self.parent_path),
            "endpoint": reference(self.formal / "endpoint-lock.json"),
            "formal_closed": reference(self.formal / "CLOSED.json"),
            "A_result": reference(self.a / "RESULT.json"),
            "A_closed": reference(self.a / "CLOSED.json"),
            "A_archive_index": reference(self.a / "runtime-archive-index.json"),
        }
        self.refresh_certificate()

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
        return reference(path)

    def _write_parent(self):
        self.parent.pop("contract_sha256", None)
        self.parent["contract_sha256"] = digest(self.parent)
        self.write(self.parent_path, self.parent)

    @staticmethod
    def closed(directory, mode):
        return {"status": "PASS", "seed": 1, "mode": mode,
                "result": reference(directory / "RESULT.json"),
                "runtime_index": reference(directory / "runtime-archive-index.json")}

    def refresh_certificate(self):
        reuse = self.c["parent_reuse"]
        certificate = {
            "status": "PASS", "old_contract": reuse["contract"],
            "old_source_sha256": self.parent["source_sha256"],
            "new_source_sha256": "2" * 64, "protocol_only_source_delta": True,
            "reuse_references": {k: v for k, v in reuse.items()
                                 if isinstance(v, dict) and k != "certification"},
            "scope": "CPU_REFERENCE_FIXTURE_NOT_GPU_EVIDENCE",
        }
        reuse["certification"] = self.write(self.cert_path, certificate)

    def replace_bound(self, key, path, value):
        self.c["parent_reuse"][key] = self.write(path, value)
        self.refresh_certificate()

    def test_valid_resolvers_preserve_parent_bytes_and_old_binding(self):
        before = {str(p): reference(p) for p in self.base.rglob("*.json")}
        self.assertEqual(support.parent_contract(self.c), self.parent)
        self.assertEqual(support.endpoint(self.c), self.lock)
        self.assertEqual(support.restore_contract(self.c), self.parent)
        cache = support.original_A_contract(self.c)
        self.assertEqual(cache["checkpoint"], self.lock["checkpoint"])
        self.assertEqual(cache["source_sha256"], self.parent["source_sha256"])
        self.assertNotEqual(cache["source_sha256"], self.c["source_sha256"])
        self.assertEqual(before, {str(p): reference(p) for p in self.base.rglob("*.json")})

    def test_all_five_authorized_remaining_modes_validate_real_files(self):
        for mode in ("accept-A", "accept-B", "accept-C", "full-C", "evaluation"):
            with self.subTest(mode=mode):
                self.assertIsNone(support.prerequisite(self.c, mode))

    def test_seed1_training_A548_and_gpu_acceptance_are_forbidden(self):
        for mode in ("formal", "gpu-acceptance", "infer-A", "A548", "cpu-state", "unknown"):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "cannot train/replay"):
                support.prerequisite(self.c, mode)

    def test_reuse_is_seed1_only(self):
        self.c["seed"] = 2
        with self.assertRaisesRegex(RuntimeError, "only original seed1"):
            support.parent_contract(self.c)

    def test_wrong_new_identity_and_recipe_are_rejected(self):
        for key, value in (("task_id", "foreign-task"), ("config", {"seed": 2}),
                           ("data_sha256", "9" * 64), ("environment", {"torch": "other"}),
                           ("gpu_uuid", "GPU-foreign")):
            candidate = copy.deepcopy(self.c)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "scientific/runtime parent differs"):
                support.parent_contract(candidate)

    def test_all_scientific_input_bindings_are_checked(self):
        for key in ("data", "pretrained", "development_ids", "fixed64", "evaluation_protocol",
                    "evaluation_protocol_sha256", "materials", "plan"):
            candidate = copy.deepcopy(self.c)
            candidate[key] = {"changed": key}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "scientific inputs differ: " + key):
                support.parent_contract(candidate)

    def test_wrong_new_source_is_rejected_by_certificate(self):
        self.c["source_sha256"] = "9" * 64
        with self.assertRaisesRegex(RuntimeError, "certification differs"):
            support.parent_contract(self.c)

    def test_parent_digest_mismatch_even_with_fresh_file_SHA_is_rejected(self):
        damaged = {**self.parent, "scope": "changed-without-redigest"}
        self.c["parent_reuse"]["contract"] = self.write(self.parent_path, damaged)
        self.refresh_certificate()
        with self.assertRaisesRegex(RuntimeError, "parent contract digest changed"):
            support.parent_contract(self.c)

    def test_corrupted_parent_reference_is_rejected(self):
        self.c["parent_reuse"]["contract"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "reference bytes differ"):
            support.parent_contract(self.c)

    def test_size_mismatch_and_missing_reference_are_rejected(self):
        ref = reference(self.parent_path)
        with self.assertRaisesRegex(RuntimeError, "reference bytes differ"):
            support.exact({**ref, "bytes": ref["bytes"] + 1})
        with self.assertRaises(FileNotFoundError):
            support.exact({**ref, "path": str(self.base / "missing.json")})

    def test_certificate_corruption_and_failed_delta_are_rejected(self):
        proof = read(self.cert_path)
        proof["protocol_only_source_delta"] = False
        self.c["parent_reuse"]["certification"] = self.write(self.cert_path, proof)
        with self.assertRaisesRegex(RuntimeError, "certification differs"):
            support.parent_contract(self.c)
        self.cert_path.write_text("{}", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "reference bytes differ"):
            support.parent_contract(self.c)

    def test_reuse_reference_set_cannot_be_changed_after_certification(self):
        alternate = self.write(self.base / "alternate.json", {"scope": "not-certified"})
        self.c["parent_reuse"]["A_closed"] = alternate
        with self.assertRaisesRegex(RuntimeError, "reference set differs"):
            support.parent_contract(self.c)

    def test_e120_cursor_source_seed_and_acceptance_parent_are_rejected(self):
        for key, value in (("epoch", 119), ("updates", 36479), ("actual_updates", 2),
                           ("seed", 2), ("source_sha256", "9" * 64),
                           ("contract_sha256", "9" * 64), ("used_acceptance_state", True)):
            lock = {**self.lock, key: value}
            self.replace_bound("endpoint", self.formal / "endpoint-lock.json", lock)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "endpoint parent/source/cursor differs"):
                support.endpoint(self.c)

    def test_original_binding_bytes_change_is_rejected(self):
        self.write(self.formal / "binding.json", {"changed": True})
        with self.assertRaisesRegex(RuntimeError, "checkpoint binding changed"):
            support.endpoint(self.c)

    def test_formal_failed_wrong_mode_seed_and_result_are_rejected(self):
        for key, value in (("status", "STOP"), ("mode", "gpu-acceptance"), ("seed", 2),
                           ("result", {"path": "wrong-result", "sha256": "0" * 64})):
            self.replace_bound("formal_closed", self.formal / "CLOSED.json", {**self.formal_closed, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "formal closure changed"):
                support.endpoint(self.c)

    def test_formal_archive_damage_is_rejected(self):
        self.write(self.formal / "runtime-archive-index.json", {"scope": "tampered"})
        with self.assertRaisesRegex(RuntimeError, "formal closure changed"):
            support.endpoint(self.c)

    def test_A_failed_wrong_mode_seed_and_result_are_rejected(self):
        for key, value in (("status", "STOP"), ("mode", "accept-A"), ("seed", 2),
                           ("result", {"path": "wrong-result", "sha256": "0" * 64})):
            self.replace_bound("A_closed", self.a / "CLOSED.json", {**self.a_closed, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "A closure changed"):
                support.original_A_contract(self.c)

    def test_A_result_and_archive_damage_are_rejected(self):
        self.write(self.a / "RESULT.json", {"scope": "tampered"})
        with self.assertRaisesRegex(RuntimeError, "reference bytes differ"):
            support.original_A_contract(self.c)

    def test_A_alias_paths_cannot_replace_expected_original_paths(self):
        for key, name in (("A_result", "RESULT.json"), ("A_closed", "CLOSED.json"),
                          ("A_archive_index", "runtime-archive-index.json")):
            original = copy.deepcopy(self.c["parent_reuse"])
            alias = self.base / ("alias-" + name)
            alias.write_bytes((self.a / name).read_bytes())
            self.c["parent_reuse"][key] = reference(alias)
            self.refresh_certificate()
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "A closure changed"):
                support.original_A_contract(self.c)
            self.c["parent_reuse"] = original
            self.refresh_certificate()

    def test_same_bytes_different_formal_and_endpoint_paths_are_rejected(self):
        for key, name, error in (("endpoint", "endpoint-lock.json", "endpoint path differs"),
                                 ("formal_closed", "CLOSED.json", "formal closure changed")):
            original = copy.deepcopy(self.c["parent_reuse"])
            alias = self.base / ("alias-formal-" + name)
            alias.write_bytes((self.formal / name).read_bytes())
            self.c["parent_reuse"][key] = reference(alias)
            self.refresh_certificate()
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, error):
                support.endpoint(self.c)
            self.c["parent_reuse"] = original
            self.refresh_certificate()

    def test_non_reuse_seed2_keeps_own_contract_and_endpoint(self):
        c = copy.deepcopy(self.parent)
        c["seed"] = 2
        self.write(self.formal / "endpoint-lock.json", {**self.lock, "seed": 2})
        self.assertIs(support.original_A_contract(c), c)
        self.assertIs(support.restore_contract(c), c)
        self.assertEqual(support.endpoint(c)["seed"], 2)

    def test_real_child_resolves_JSON_without_importing_torch_or_creating_artifacts(self):
        path = self.base / "resume-contract-fixture.json"
        self.write(path, self.c)
        code = (
            "import sys; from pathlib import Path; "
            "sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'src')]; "
            "import resume_support as r; from sparse_rtdetr.localization.runio import read; "
            "c=read(sys.argv[2]); r.parent_contract(c); r.endpoint(c); "
            "r.original_A_contract(c); r.restore_contract(c); r.prerequisite(c,'accept-A'); "
            "assert 'torch' not in sys.modules; print('CPU_RESOLVER_PASS_NO_TORCH')"
        )
        before = {str(p): reference(p) for p in self.base.rglob("*.json")}
        child = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(path)],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("CPU_RESOLVER_PASS_NO_TORCH", child.stdout)
        self.assertEqual(before, {str(p): reference(p) for p in self.base.rglob("*.json")})

    def test_continuation_chain_has_only_the_fixed_remaining_phase_tables(self):
        tree = ast.parse((ROOT / "resume_chain.py").read_text(encoding="utf8"))
        assignments = {node.targets[0].id: node.value for node in tree.body
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)}
        seed1 = ast.literal_eval(assignments["SEED1"])
        self.assertEqual(seed1, ("accept-A", "accept-B", "accept-C", "full-C", "evaluation"))
        seed2_node = assignments["SEED2"]
        self.assertIsInstance(seed2_node, ast.Tuple)
        self.assertEqual(tuple(ast.literal_eval(v) for v in seed2_node.elts[:-1]),
                         ("gpu-acceptance", "formal", "infer-A"))
        self.assertIsInstance(seed2_node.elts[-1], ast.Starred)
        self.assertEqual(seed2_node.elts[-1].value.id, "SEED1")

    def test_continuation_never_imports_or_calls_original_launcher_main(self):
        tree = ast.parse((ROOT / "resume_chain.py").read_text(encoding="utf8"))
        imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and node.module == "new_chain"]
        self.assertEqual(len(imported), 1)
        self.assertEqual([alias.name for alias in imported[0].names], ["phase"])
        self.assertFalse(any(isinstance(node, ast.Import)
                             and any(alias.name == "new_chain" for alias in node.names)
                             for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                             and node.value.id == "new_chain" and node.attr == "main"
                             for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
