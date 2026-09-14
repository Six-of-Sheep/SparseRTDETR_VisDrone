import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def test_runtime_artifact_policy_uses_gitignore_and_checks_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "artifacts" / "ordinary.bin").write_bytes(b"ok")
            files, failures = CHECKER._file_policy(root)
            self.assertNotIn("artifacts/ordinary.bin", files)
            self.assertEqual(failures, [])

    def test_ignored_large_artifact_is_stat_checked_without_source_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            with (root / "artifacts" / "large.bin").open("wb") as handle:
                handle.truncate(11 * 1024 * 1024)
            files, failures = CHECKER._file_policy(root)
            self.assertNotIn("artifacts/large.bin", files)
            self.assertEqual(failures, [])

    def test_artifact_symlink_and_tracked_artifact_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "source.txt").write_text("source", encoding="utf-8")
            os.symlink(root / "source.txt", root / "artifacts" / "link.txt")
            with mock.patch.object(CHECKER, "_git_tracked_files", return_value={"artifacts/tracked.txt"}):
                (root / "artifacts" / "tracked.txt").write_text("tracked", encoding="utf-8")
                _files, failures = CHECKER._file_policy(root)
            self.assertIn("symlink: artifacts/link.txt", failures)
            self.assertIn("tracked runtime artifact: artifacts/tracked.txt", failures)

    def test_nonignored_artifact_and_source_limits_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n!artifacts/keep.txt\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "artifacts" / "keep.txt").write_text("source-visible", encoding="utf-8")
            os.symlink(root / "artifacts" / "keep.txt", root / "source-link.txt")
            with (root / "large-source.txt").open("wb") as handle:
                handle.truncate(11 * 1024 * 1024)
            files, failures = CHECKER._file_policy(root)
            self.assertIn("artifacts/keep.txt", files)
            self.assertIn("symlink: source-link.txt", failures)
            self.assertIn("large file: large-source.txt", failures)

    def test_source_only_never_enters_runtime_artifacts_or_follows_evidence_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            evidence = root / "artifacts" / "evidence"
            evidence.mkdir(parents=True)
            (evidence / "payload").write_bytes(b"runtime-only")
            os.symlink(evidence / "payload", evidence / "pytest-current")
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            scandir = os.scandir
            def source_scandir(path):
                candidate = Path(path)
                if candidate == root / "artifacts" or root / "artifacts" in candidate.parents:
                    raise AssertionError("source checker entered runtime evidence")
                return scandir(path)
            with mock.patch.object(CHECKER.os, "scandir", side_effect=source_scandir), \
                    mock.patch.object(CHECKER, "_git_tracked_files", return_value=set()):
                files, failures = CHECKER._file_policy(root, source_only=True)
            self.assertEqual(files, {".gitignore", "source.py"})
            self.assertEqual(failures, [])

    def test_source_only_still_rejects_source_links_and_tracked_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            os.symlink(root / "source.py", root / "source-link.py")
            os.symlink(root / "artifacts", root / "source-dir-link")
            with mock.patch.object(CHECKER, "_git_tracked_files", return_value={"artifacts/unread.pt"}):
                _, failures = CHECKER._file_policy(root, source_only=True)
            self.assertIn("symlink: source-link.py", failures)
            self.assertIn("symlink: source-dir-link", failures)
            self.assertIn("tracked runtime artifact: artifacts/unread.pt", failures)

    def test_source_only_rejects_dangling_artifact_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            os.symlink(root / "absent", root / "artifacts")
            _, failures = CHECKER._file_policy(root, source_only=True)
            self.assertIn("artifacts must be a real directory", failures)


if __name__ == "__main__":
    unittest.main()
