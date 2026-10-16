"""Isolated Git repositories test publication and freeze boundaries; no model."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repo_manage', Path(__file__).parents[1] / 'tools/repo_manage.py')
management = importlib.util.module_from_spec(spec); spec.loader.exec_module(management)

class RepositoryManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name); self.repo = self.root / 'repo'; self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test'); self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'core.autocrlf', 'false')
        (self.repo / 'config.json').write_bytes(b'{}\n')
        self.git('add', '--', 'config.json'); self.git('commit', '-qm', 'fixture')

    def tearDown(self): self.temp.cleanup()

    def git(self, *args): return management.git(self.repo, *args)

    def test_vendor_data_code_is_allowed_but_root_data_is_not(self):
        self.assertEqual(management.file_problems('vendor/pkg/src/data/loader.py', b'pass\n'), [])
        self.assertTrue(management.file_problems('data/image.jpg', b'pixels'))

    def test_actual_ignore_policy_keeps_vendor_data_source(self):
        policy = Path(__file__).parents[1] / '.gitignore'
        (self.repo / '.gitignore').write_bytes(policy.read_bytes())
        for name, expected in [('data/a.jpg', 0), ('vendor/pkg/src/data/loader.py', 1)]:
            result = subprocess.run(['git', '-C', str(self.repo), 'check-ignore', '-q', name])
            self.assertEqual(result.returncode, expected, name)

    def test_artifacts_and_credentials_rejected(self):
        for path in ['model.pth', 'weights/w.bin', '.env', '.env.local', 'x/id_ed25519', '.ssh/config']:
            self.assertTrue(management.file_problems(path, b'x'), path)
        secret = ('gh' + 'p_' + 'A' * 36).encode()
        self.assertTrue(management.file_problems('settings.py', secret))

    def test_size_and_symlink_rejected(self):
        self.assertTrue(management.file_problems('large.txt', b'x' * (management.MAX_BYTES + 1)))
        self.assertTrue(management.file_problems('link', b'/outside', '120000'))

    def test_traversal_and_windows_paths_rejected(self):
        for name in ['../x', '/x', 'C:/x', 'a\\b', 'a/../../b', 'a\nb']:
            with self.assertRaises(ValueError): management.relative_path(name)

    def test_staged_scan_reads_index_not_working_copy(self):
        path = self.repo / 'secret.py'; path.write_text('gh' + 'p_' + 'A' * 36)
        self.git('add', '--', 'secret.py'); path.write_text('safe now')
        with self.assertRaises(ValueError): management.check(self.repo, staged=True)

    def test_deleted_artifact_is_not_rejected(self):
        path = self.repo / 'old.pth'; path.write_bytes(b'old')
        self.git('add', '--', 'old.pth'); self.git('commit', '-qm', 'old artifact fixture')
        self.git('rm', '-q', '--', 'old.pth')
        self.assertEqual(management.check(self.repo, staged=True)['files'], 0)

    def test_freeze_binds_commit_and_refuses_overwrite(self):
        target = self.root / 'run/source.json'
        result = management.freeze(self.repo, 'unit-seed0', 'config.json', target)
        receipt = json.loads(target.read_text())
        self.assertEqual(result['commit'], self.git('rev-parse', 'HEAD').decode().strip())
        self.assertFalse(receipt['execution_authorized'])
        self.assertEqual(receipt['source_files'][0]['bytes'], 3)
        with self.assertRaises(FileExistsError): management.freeze(self.repo, 'unit-seed0', 'config.json', target)

    def test_dirty_and_untracked_source_block_freeze(self):
        (self.repo / 'new.py').write_text('pass')
        with self.assertRaises(ValueError): management.freeze(self.repo, 'unit-seed0', 'config.json', self.root / 'receipt.json')

    def test_config_must_be_tracked_and_receipt_outside_source(self):
        with self.assertRaises(ValueError): management.freeze(self.repo, 'unit-seed0', 'absent.json', self.root / 'r.json')
        with self.assertRaises(ValueError): management.freeze(self.repo, 'unit-seed0', 'config.json', self.repo / 'r.json')

    def test_import_verifies_immutable_commit_and_detects_tampering(self):
        commit = self.git('rev-parse', 'HEAD').decode().strip()
        directory = self.repo / 'manifests'; directory.mkdir()
        content = b'{}\n'
        manifest = json.dumps({'files': [{'path': 'config.json', 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}).encode()
        (directory / 'import.json').write_bytes(manifest)
        (directory / 'experiments.json').write_text(json.dumps({'source_import': {
            'commit': commit, 'manifest': 'manifests/import.json', 'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'files': 1}}))
        self.assertEqual(management.verify_import(self.repo, True)['files'], 1)
        (self.repo / 'config.json').write_text('changed')
        self.assertEqual(management.verify_import(self.repo)['files'], 1)
        with self.assertRaises(ValueError): management.verify_import(self.repo, True)
        (directory / 'import.json').write_bytes(manifest + b' ')
        with self.assertRaises(ValueError): management.verify_import(self.repo)

if __name__ == '__main__': unittest.main()
