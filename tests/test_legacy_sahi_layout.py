"""Exercise the file paths used by the SAHI supervisors without starting a job."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.legacy_sahi import confirmation_prepare, new_runtime, supplemental_job, supplemental_runtime


ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ROOT / "tools" / "legacy_sahi"


class LegacyLayoutTests(unittest.TestCase):
    def test_source_root_and_child_entry(self):
        for runtime in (confirmation_prepare, new_runtime, supplemental_runtime):
            self.assertEqual(runtime.ROOT, ROOT)
        contract = {"python": sys.executable, "contract_path": "example-contract.json"}
        self.assertEqual(
            supplemental_job.command(contract, "worker", seed=2, mode="full-C"),
            [sys.executable, "-B", str(DRIVERS / "supplemental_job.py"),
             "worker", "--contract", "example-contract.json", "--seed", "2", "--mode", "full-C"],
        )

    def test_standalone_help_from_another_directory(self):
        # The supervisor invokes scripts by path, including from outside the checkout.
        entries = ("new_chain", "new_entry", "confirmation_prepare", "confirmation_cpu_check",
                   "supplemental_job", "supplemental_entry_check", "supplemental_path_check")
        with tempfile.TemporaryDirectory() as work:
            for entry in entries:
                with self.subTest(entry=entry):
                    help_text = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", str(DRIVERS / (entry + ".py")), "--help"],
                        cwd=work, capture_output=True, text=True, check=True,
                    ).stdout
                    self.assertIn("usage:", help_text)
            self.assertEqual(list(Path(work).iterdir()), [])
