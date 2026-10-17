"""CPU-only observer integration; exact read code runs in child processes.

Fixtures replace only SSH/OS adapters. No model, CUDA or launch entry is used.
"""
import base64
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'tools/sahi_observer.py'
spec = importlib.util.spec_from_file_location('sahi_observer', MODULE)
obs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obs)


def canonical(value):
    return obs.sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=False, allow_nan=False).encode())


class ObserverTests(unittest.TestCase):
    def test_windows_ssh_child_has_no_window_flag(self):
        self.assertEqual(obs.ssh_process_options('nt'), {'creationflags': 0x08000000})

    def test_non_windows_ssh_has_no_windows_options(self):
        self.assertEqual(obs.ssh_process_options('posix'), {})

    @unittest.skipUnless(os.name == 'nt', 'Windows console API')
    def test_windows_real_child_has_no_console_window(self):
        code = 'import ctypes; print(int(ctypes.windll.kernel32.GetConsoleWindow() or 0))'
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                                timeout=10, **obs.ssh_process_options())
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), b'0')

    def test_real_remote_adapter_applies_flag_without_losing_capture_or_timeout(self):
        with patch.object(obs, 'ssh_process_options', return_value={'creationflags': 0x08000000}), \
             patch.object(obs.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'{}', stderr=b'')) as launch:
            self.assertEqual(obs.remote(self.config, obs.REMOTE_STATUS, timeout=17), {})
        args, kwargs = launch.call_args
        self.assertEqual(args[0][0], 'ssh.exe')
        self.assertEqual(kwargs['creationflags'], 0x08000000)
        self.assertTrue(kwargs['capture_output'])
        self.assertEqual(kwargs['timeout'], 17)
        self.assertIsInstance(kwargs['input'], bytes)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.remote = self.project / 'remote'
        self.remote.mkdir()
        self.local = self.project / 'local'
        self.local.mkdir()
        (self.project / 'collaboration').mkdir()
        for name in ('CURRENT.md', 'HISTORY.md'):
            (self.project / 'collaboration' / name).write_text('# Shared\n\nOld record\n', encoding='utf8')
        self.source = self.remote / 'source'
        self.source.mkdir()
        (self.source / 'new_runtime.py').write_text('def install(c=None): pass\n', encoding='utf8')
        package = self.source / 'src/sparse_rtdetr/localization'
        package.mkdir(parents=True)
        (package / 'channel.py').write_text('', encoding='utf8')
        (package / 'guard.py').write_text("def identity(pid): return {'start':'42','state':'S'}\n", encoding='utf8')
        (package / 'runio.py').write_text(
            'import hashlib,json\nfrom pathlib import Path\n'
            'def read(path): return json.loads(Path(path).read_text(encoding="utf8"))\n'
            'def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),'
            'ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n', encoding='utf8')
        contracts = []
        for seed in (1, 2):
            c = {'seed': seed, 'task_id': f'fixture-seed{seed}', 'source_sha256': 'b'*64,
                 'contract_sha256': 'c'*64, 'job_root': (self.remote / f'seed{seed}').as_posix(),
                 'runtime_root': (self.remote / 'ssd' / f'seed{seed}').as_posix()}
            contracts.append(self.write(self.remote / f'seed{seed}/execution-contract.json', c))
        self.execution = {'task_id': 'fixture', 'seeds': [1, 2], 'no_retry': True, 'holdout': False,
                          'contracts': contracts, 'source_root': self.source.as_posix(),
                          'job_root': self.remote.as_posix(), 'python': sys.executable}
        self.execution['execution_sha256'] = canonical(self.execution)
        self.exec_ref = self.write(self.remote / 'EXECUTION.json', self.execution)
        self.write(self.local / 'EXECUTION.json', self.execution)
        launch = {'pid': 1234, 'process_identity': {'start': '42', 'state': 'S'},
                  'execution_sha256': self.execution['execution_sha256']}
        self.write(self.remote / 'LAUNCH_RECEIPT.json', launch)
        self.write(self.local / 'LAUNCH_RECEIPT.json', launch)
        self.terminal = self.write(self.remote / 'CHAIN_RESULT.json', {'status': 'FIXED_PAIRED_EXECUTION_COMPLETE'})
        aggregate = self.write(self.remote / 'summary/AGGREGATE.json', {'seeds': [0, 1, 2], 'decision': 'FAIL'})
        self.refs = [self.exec_ref, self.terminal, aggregate]
        self.manifest = {'task_id': 'fixture', 'status': 'READY_FOR_WINDOWS_SLIM_RETURN',
                         'not_full_audit_delivery': True, 'files': self.refs,
                         'files_count': 3, 'total_bytes': sum(r['bytes'] for r in self.refs)}
        self.manifest_ref = self.write(self.remote / 'SLIM_EXPORT.json', self.manifest)
        self.config = obs.config_for(self.local / 'EXECUTION.json', self.local)
        self.calls = []

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(obs.json_bytes(value))
        result = obs.reference(p)
        result['path'] = p.as_posix()
        return result

    def fetch(self, config, body, timeout=120):
        self.calls.append(body)
        code = 'CONFIG=' + repr(config) + '\n' + obs.REMOTE_HEADER + inspect.getsource(obs.read_tail) + body
        p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, timeout=15)
        if p.returncode:
            raise obs.EvidenceError(p.stderr.decode('utf8', 'replace'))
        return json.loads(p.stdout)

    def child_entry(self, mode='watch', script_prefix='', observer_id='cpu-entry'):
        # Actual production main in an independent process. The SSH boundary is
        # replaced by exact production read code in a further CPU subprocess.
        code = ('import importlib.util,inspect,json,subprocess,sys\n'
                f'spec=importlib.util.spec_from_file_location("observer",{str(MODULE)!r})\n'
                'o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)\n'
                'calls=[0]\n'
                'def cpu(config,body,timeout=120):\n'
                ' calls[0]+=1\n'
                + script_prefix +
                ' code="CONFIG="+repr(config)+"\\n"+o.REMOTE_HEADER+inspect.getsource(o.read_tail)+body\n'
                ' p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,timeout=15)\n'
                ' if p.returncode: raise o.EvidenceError(p.stderr.decode())\n'
                ' return json.loads(p.stdout)\n'
                'o.remote=cpu\no.time.sleep=lambda seconds: None\n'
                f'raise SystemExit(o.main({[mode,"--execution",str(self.local / "EXECUTION.json"),"--local-root",str(self.local),"--project",str(self.project),"--observer-id",observer_id]!r}))\n')
        return subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, timeout=40)

    def test_true_entry_terminal_export_and_scientific_FAIL_closes(self):
        p = self.child_entry()
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        folder = self.local / 'observers/cpu-entry'
        result = json.loads((folder / 'OBSERVER_COMPLETE.json').read_bytes())
        self.assertEqual(result['status'], 'SLIM_RETURN_AND_HANDOFF_COMPLETE')
        self.assertFalse(result['server_chain_restarted'])
        receipt = json.loads((folder / 'OBSERVER_TERMINAL_RETURN.json').read_bytes())
        self.assertTrue(all(row['SHA_PASS'] for row in receipt['files']))
        self.assertIn('FAIL', (self.project / 'collaboration/CURRENT.md').read_text('utf8'))
        self.assertEqual(len(list(folder.glob('OBSERVER_STARTED.json'))), 1)

    def test_network_reconnection_real_entry_never_restarts_server(self):
        p = self.child_entry(script_prefix=' if calls[0]==1: raise o.NetworkUnavailable("offline fixture")\n')
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        state = json.loads((self.local / 'observers/cpu-entry/OBSERVER_STATE.json').read_bytes())
        self.assertEqual(state['network_errors'], 1)
        self.assertEqual(state['polls'], 2)
        self.assertIn(b'NETWORK_UNAVAILABLE_REMOTE_STATUS_UNKNOWN', p.stdout)
        started = json.loads((self.local / 'observers/cpu-entry/OBSERVER_STARTED.json').read_bytes())
        self.assertEqual(started['remote_launcher_calls'], 0)

    def test_status_real_entry_always_new_read(self):
        # Even deliberately forged old local status is never consulted.
        self.write(self.local / 'OBSERVER_LATEST.json', {'status': 'FORGED_LIVE_CACHE'})
        p = self.child_entry('status')
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        view = json.loads(p.stdout)
        self.assertEqual(view['freshness'], 'fresh')
        self.assertEqual(view['status'], 'TERMINAL_READY')
        self.assertIn('checked_at', view)

    def test_status_network_real_entry_reports_unknown_nonzero(self):
        p = self.child_entry('status', ' raise o.NetworkUnavailable("network fixture")\n')
        self.assertEqual(p.returncode, 2)
        view = json.loads(p.stdout)
        self.assertEqual(view['freshness'], 'unknown')
        self.assertIsNone(view['snapshot'])

    def test_chain_disappeared_real_entry_stops_only_observer(self):
        (self.remote / 'CHAIN_RESULT.json').unlink()
        (self.remote / 'SLIM_EXPORT.json').unlink()
        (self.source / 'src/sparse_rtdetr/localization/guard.py').write_text(
            'def identity(pid): return None\n', encoding='utf8')
        p = self.child_entry()
        self.assertNotEqual(p.returncode, 0)
        folder = self.local / 'observers/cpu-entry'
        stop = json.loads((folder / 'OBSERVER_STOP.json').read_bytes())
        self.assertFalse(stop['remote_retry_or_restart'])
        self.assertTrue(stop['remote_training_failure_not_inferred'])
        self.assertIn('CHAIN_ABSENT_WITHOUT_TERMINAL', stop['error'])
        self.assertFalse((self.remote / 'STOP.json').exists())

    def test_remote_STOP_exports_evidence_without_inferring_completion(self):
        (self.remote / 'CHAIN_RESULT.json').unlink()
        stop = self.write(self.remote / 'STOP.json', {'status': 'STOP_PRESERVE_NO_RETRY',
            'seed': 1, 'mode': 'accept-A', 'error': 'fixture failure'})
        refs = [self.exec_ref, stop]
        self.write(self.remote / 'SLIM_EXPORT.json', {**self.manifest, 'files': refs,
            'files_count': len(refs), 'total_bytes': sum(r['bytes'] for r in refs)})
        p = self.child_entry()
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        value = json.loads((self.local / 'observers/cpu-entry/OBSERVER_COMPLETE.json').read_bytes())
        self.assertEqual(value['terminal'], 'STOP.json')
        text = (self.project / 'collaboration/CURRENT.md').read_text('utf8')
        self.assertIn('未形成三seed裁决', text)
        self.assertIn('STOP留证', text)

    def test_independent_concurrent_handoff_records_are_preserved(self):
        current = self.project / 'collaboration/CURRENT.md'
        original = current.read_bytes()
        code = ('import os,sys,time\n'
                'fd=os.open(sys.argv[1],os.O_APPEND|os.O_WRONLY)\n'
                'for i in range(60):\n'
                ' os.write(fd,("foreground-record-"+str(i)+"\\n").encode());time.sleep(.001)\n'
                'os.close(fd)\n')
        child = subprocess.Popen([sys.executable, '-B', '-c', code, str(current)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p = self.child_entry()
        _, err = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0, err.decode())
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        data = current.read_bytes()
        self.assertTrue(data.startswith(original.splitlines()[0]))
        self.assertIn(b'Old record', data)
        for i in range(60):
            self.assertIn(f'foreground-record-{i}'.encode(), data.splitlines())

    def test_cached_age_fresh_stale_unknown(self):
        self.assertEqual(obs.cached_state({'received_epoch': 10}, 11)['freshness'], 'fresh')
        self.assertEqual(obs.cached_state({'received_epoch': 10}, 191)['freshness'], 'stale')
        self.assertEqual(obs.cached_state(None)['freshness'], 'unknown')
        self.assertEqual(obs.cached_state({'received_epoch': 10}, 11)['remote_status'], 'UNKNOWN_UNTIL_FRESH_READ')

    def test_network_failure_does_not_publish_cached_RUNNING(self):
        def offline(*args):
            raise obs.NetworkUnavailable('unreachable')
        view = obs.fresh_status(self.config, {'received_epoch': 10}, offline)
        self.assertEqual(view['freshness'], 'unknown')
        self.assertIsNone(view['snapshot'])
        self.assertTrue(view['remote_training_failure_not_inferred'])

    def test_new_observer_identity_and_singleton_real_process(self):
        lock = obs.hold_singleton(self.local)
        try:
            p = self.child_entry()
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(b'singleton', p.stderr)
            self.assertFalse((self.local / 'observers/cpu-entry').exists())
        finally:
            lock.close()
        p = self.child_entry()
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        old = (self.local / 'observers/cpu-entry/OBSERVER_STARTED.json').read_bytes()
        p = self.child_entry()
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(old, (self.local / 'observers/cpu-entry/OBSERVER_STARTED.json').read_bytes())

    def test_remote_launch_pid_reuse_rejected(self):
        launch = {'pid': 1234, 'process_identity': {'start': 'different'},
                  'execution_sha256': self.execution['execution_sha256']}
        self.write(self.remote / 'LAUNCH_RECEIPT.json', launch)
        self.write(self.local / 'LAUNCH_RECEIPT.json', launch)
        self.config = obs.config_for(self.local / 'EXECUTION.json', self.local)
        with self.assertRaises(obs.EvidenceError):
            obs.fresh_status(self.config, fetch=self.fetch)

    def test_remote_corrupt_SHA_ref_stops_without_new_local_bytes(self):
        (self.remote / 'summary/AGGREGATE.json').write_bytes(b'corrupted')
        view = obs.fresh_status(self.config, fetch=self.fetch)
        with self.assertRaises(obs.EvidenceError):
            obs.export_terminal(self.local, self.config, view['snapshot'], self.fetch)
        self.assertFalse((self.local / 'returned-small/CHAIN_RESULT.json').exists())

    def test_same_SHA_reuse_and_local_mismatch_never_overwrites(self):
        p = self.local / 'returned-small/EXECUTION.json'
        p.parent.mkdir()
        p.write_bytes((self.remote / 'EXECUTION.json').read_bytes())
        view = obs.fresh_status(self.config, fetch=self.fetch)
        result = obs.export_terminal(self.local, self.config, view['snapshot'], self.fetch)
        row = next(row for row in result['files'] if row['reference'] == self.exec_ref)
        self.assertTrue(row['reused'])
        p.write_bytes(b'preserve prior local evidence')
        with self.assertRaises(obs.EvidenceError):
            obs.export_terminal(self.local, self.config, view['snapshot'], self.fetch)
        self.assertEqual(p.read_bytes(), b'preserve prior local evidence')

    def test_manifest_wrong_identity_missing_terminal_size_duplicate_reject(self):
        for changes in ({'task_id': 'wrong'}, {'files_count': 201}, {'total_bytes': obs.MAX_BYTES+1},
                        {'files': self.refs + self.refs[:1], 'files_count': 4,
                         'total_bytes': self.manifest['total_bytes'] + self.refs[0]['bytes']}):
            with self.subTest(changes=changes), self.assertRaises(obs.EvidenceError):
                obs.validate_manifest({**self.manifest, **changes}, self.manifest_ref, self.config)
        bad = {**self.manifest, 'files': [self.exec_ref], 'files_count': 1, 'total_bytes': self.exec_ref['bytes']}
        self.write(self.remote / 'SLIM_EXPORT.json', bad)
        view = obs.fresh_status(self.config, fetch=self.fetch)
        with self.assertRaises(obs.EvidenceError):
            obs.export_terminal(self.local, self.config, view['snapshot'], self.fetch)

    def test_forbidden_paths_and_allowed_new_small_receipts(self):
        for name in ('seed1/full-C/top600-predictions.json', 'seed1/formal/checkpoint-epoch-120.pth',
                     'seed2/formal/audits/update-1.json', '../escape.json', 'seed1/formal/worker.log'):
            with self.subTest(name=name), self.assertRaises(obs.EvidenceError):
                obs.allowed_relative({'path': self.remote.as_posix()+'/'+name,
                                      'bytes': 1, 'sha256': 'a'*64}, self.remote.as_posix())
        for name in ('PARENT_REUSE.json', 'PROTOCOL_DELTA.json', 'release-003/source-manifest.json',
                     'seed1-accept-A-exit.json', 'PREPARED.json', 'CPU_INTEGRATION.json',
                     'CPU_UNIT_GATE.json', 'SOURCE_GIT.json', 'seed1/gpu-execution-contract.json',
                     'seed2/gpu-execution-contract.json'):
            self.assertEqual(obs.allowed_relative({'path': self.remote.as_posix()+'/'+name,
                'bytes': 1, 'sha256': 'a'*64}, self.remote.as_posix()), name)

    def test_resume_execution_schema_and_all_new_export_refs_real_entry(self):
        execution = {**self.execution, 'task_id': 'protocol-resume-20261017-001'}
        execution['execution_sha256'] = canonical({k: v for k, v in execution.items() if k != 'execution_sha256'})
        execution_ref = self.write(self.remote / 'EXECUTION.json', execution)
        self.write(self.local / 'EXECUTION.json', execution)
        launch = {'pid': 1234, 'process_identity': {'start': '42', 'state': 'S'},
                  'execution_sha256': execution['execution_sha256']}
        self.write(self.remote / 'LAUNCH_RECEIPT.json', launch)
        self.write(self.local / 'LAUNCH_RECEIPT.json', launch)
        refs = [execution_ref, self.terminal, self.refs[-1]]
        names = ('PROTOCOL_DELTA.json', 'PARENT_REUSE.json', 'PREPARED.json', 'CPU_INTEGRATION.json',
                 'CPU_UNIT_GATE.json', 'SOURCE_GIT.json', 'seed1/gpu-execution-contract.json',
                 'seed2/gpu-execution-contract.json', 'release-001/source-manifest.json')
        for name in names:
            refs.append(self.write(self.remote / name, {'fixture': name}))
        manifest = {**self.manifest, 'task_id': execution['task_id'], 'files': refs,
                    'files_count': len(refs), 'total_bytes': sum(r['bytes'] for r in refs)}
        self.write(self.remote / 'SLIM_EXPORT.json', manifest)
        p = self.child_entry()
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        receipt = json.loads((self.local / 'observers/cpu-entry/OBSERVER_TERMINAL_RETURN.json').read_bytes())
        self.assertEqual(receipt['task_id'], execution['task_id'])
        self.assertTrue(all(row['SHA_PASS'] for row in receipt['files']))
        for name in names:
            self.assertTrue((self.local / 'returned-small' / name).is_file())

    def test_concurrent_handoff_append_keeps_other_executor_bytes(self):
        current = self.project / 'collaboration/CURRENT.md'
        current.write_bytes(b'# Shared\nnew foreground executor\n')
        before = current.read_bytes()
        self.assertTrue(obs.append_once_fresh(current, '<!-- unique -->', '<!-- unique -->\nterminal'))
        self.assertTrue(current.read_bytes().startswith(before))
        self.assertFalse(obs.append_once_fresh(current, '<!-- unique -->', '<!-- unique -->\nduplicate'))
        def edit(attempt):
            if attempt == 0:
                current.write_bytes(current.read_bytes()+b'parallel foreground edit\n')
        obs.prepend_fresh(current, '<!-- new -->', '<!-- new -->\nterminal', edit)
        self.assertIn(b'parallel foreground edit', current.read_bytes())

    def test_process_identity_records_creation_not_only_PID(self):
        identity = obs.process_identity()
        self.assertEqual(identity['pid'], os.getpid())
        self.assertTrue('creation_100ns' in identity or 'proc_start_ticks' in identity)

    def test_cursor_identity_regression_and_tail_integrity(self):
        cursor = {'device': 1, 'inode': 2, 'size': 100, 'seq': 3,
                  'record_sha256': 'a'*64, 'monotonic': 2.0}
        old = {'1:accept-A': cursor}
        for bad in ({'seq': 2}, {'size': 99}, {'inode': 3}, {'record_sha256': 'b'*64}):
            snapshot = {'seeds': [{'seed': 1, 'stream_cursors': {'accept-A': {**cursor, **bad}}}]}
            with self.subTest(bad=bad), self.assertRaises(obs.EvidenceError):
                obs.check_cursors(old, snapshot)
        record = {'identity': 'b'*64, 'stream': 'progress', 'seq': 3, 'writer_pid': 123,
                  'payload': {'monotonic': 2.0, 'event': 'sahi_complete'}}
        record['sha256'] = canonical(record)
        path = self.local / 'progress.jsonl'
        path.write_bytes((json.dumps(record)+'\n{"partial":').encode())
        event, _ = obs.read_tail(path, 'b'*64, canonical)
        self.assertEqual(event['event'], 'sahi_complete')
        with self.assertRaises(obs.EvidenceError):
            obs.read_tail(path, 'c'*64, canonical)


if __name__ == '__main__':
    unittest.main()
