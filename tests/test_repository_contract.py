import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "tools" / "repository_contract_check.py"
SPEC = importlib.util.spec_from_file_location("repository_contract_check", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


class RepositoryContractTests(unittest.TestCase):
    def test_contract_passes(self):
        self.assertTrue(CHECKER.check_repository(ROOT))

    def test_no_model_framework_imports(self):
        for directory in (ROOT / "src", ROOT / "tools", ROOT / "scripts"):
            for path in directory.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?m)^\\s*(from|import)\\s+(torch|ultralytics|rtdetr)\\b")

    def test_frozen_upstream_identity(self):
        manifest = (ROOT / "manifests" / "rtdetrv2_upstream.json").read_text(encoding="utf-8")
        self.assertIn(CHECKER.EXPECTED_UPSTREAM_COMMIT, manifest)
        self.assertIn(CHECKER.EXPECTED_UPSTREAM_SUBTREE, manifest)
        self.assertIn(CHECKER.EXPECTED_LICENSE_SHA256, manifest)

    def test_contract_sources_do_not_import_vendor(self):
        for directory in (ROOT / "src", ROOT / "tools", ROOT / "scripts"):
            for path in directory.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?m)^\s*(from|import)\s+vendor\b")


if __name__ == "__main__":
    unittest.main()
