"""Production identity interface checks; no torch import/CUDA/forward required."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sparse_rtdetr.protocol_r18 import cuda_identity as ci

A = 'GPU-11111111-1111-1111-1111-111111111111'
B = 'GPU-22222222-2222-2222-2222-222222222222'
RA = dict(uuid=A, pci='0000:01:00.0')
RB = dict(uuid=B, pci='0000:02:00.0')


class IdentityChecks(unittest.TestCase):
    def mapping(self, **changes):
        kw = dict(expected_uuid=A, runtime=[RA], inventory=[RA, RB], logical=0, visible=A)
        kw.update(changes)
        return ci.validate_mapping(**kw)

    def test_uuid_and_pci_normalization(self):
        self.assertEqual(ci.canonical_uuid(A[4:].upper()), A)
        self.assertEqual(ci.canonical_pci(RA['pci']), '00000000:01:00.0')

    def test_independently_measured_reordered_devices(self):
        v = self.mapping(expected_uuid=B, runtime=[RB, RA], visible='1,0')
        self.assertEqual(v['selected']['uuid'], B)
        self.assertEqual(v['logical_to_physical'][1]['uuid'], A)

    def test_physical_uuid_join_uses_runtime_pci_not_inventory_order(self):
        result = ci.join_physical_uuid([dict(pci=RB['pci']),dict(pci=RA['pci'])],[RA,RB])
        self.assertEqual([r['uuid'] for r in result],[B,A])
        for inventory in ([RA], [RB,RB]):
            with self.assertRaises(RuntimeError):ci.join_physical_uuid([dict(pci=RB['pci'])],inventory)

    def test_actual_runtime_api_wrapper_without_uuid_symbol(self):
        class Function:
            def __init__(self, impl):self.impl=impl
            def __call__(self,*args):return self.impl(*args)
        def integer(value):
            def impl(ptr):ctypes.cast(ptr,ctypes.POINTER(ctypes.c_int))[0]=value;return 0
            return Function(impl)
        def pci(buffer,length,index):buffer.value=RB['pci'].encode();return 0
        lib=types.SimpleNamespace(cudaGetDeviceCount=integer(1),cudaGetDevice=integer(0),
            cudaRuntimeGetVersion=integer(12040),cudaDeviceGetPCIBusId=Function(pci))
        torch=types.SimpleNamespace(version=types.SimpleNamespace(cuda='12.4'),cuda=types.SimpleNamespace(device_count=lambda:1))
        with tempfile.TemporaryDirectory() as d:
            library=Path(d)/'libcudart.so.12';library.write_bytes(b'CPU MOCK LIBRARY')
            with patch.object(Path,'read_text',return_value='mock mapping '+library.as_posix()),patch.object(ctypes,'CDLL',return_value=lib):
                rows, evidence=ci.runtime_mapping(torch)
            self.assertEqual(rows,[dict(pci=RB['pci'])]);self.assertEqual(evidence['version'],12040)

    def test_uuid_visibility_reordering(self):
        self.mapping(expected_uuid=B, runtime=[RB, RA], visible=B+','+A)
        with self.assertRaises(RuntimeError):
            self.mapping(expected_uuid=B, runtime=[RB, RA], visible=A+','+B)

    def test_wrong_logical_device_rejected(self):
        with self.assertRaises(RuntimeError):
            self.mapping(expected_uuid=A, runtime=[RB], visible='1')

    def test_uuid_pci_conflict_rejected(self):
        with self.assertRaises(RuntimeError):
            self.mapping(runtime=[dict(uuid=A, pci=RB['pci'])])

    def test_native_uuid_conflict_rejected(self):
        with self.assertRaises(RuntimeError):
            self.mapping(native_uuid=B)

    def test_missing_or_name_uuid_rejected(self):
        for value in ('', '0', 'NVIDIA GeForce RTX 4090 D', None):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.mapping(expected_uuid=value)

    def test_ambiguous_inventory_rejected(self):
        for rows in ([RA, RA], [RA, dict(uuid=B, pci=RA['pci'])], [RB]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.mapping(inventory=rows)

    def test_visibility_count_and_tokens_rejected(self):
        for visible in ('', '0,0', A+','+B, 'GPU-1111', '-1', ' 0'):
            with self.subTest(visible=visible), self.assertRaises(RuntimeError):
                self.mapping(visible=visible)

    def test_inventory_parser_rejects_missing_duplicate_and_garbage(self):
        for raw in ('', f'{A}, {RA["pci"]}\n{A}, {RA["pci"]}', 'N/A, N/A'):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                ci.physical_inventory(raw)

    def fake_torch(self, native):
        prop = types.SimpleNamespace(**native)
        cuda = types.SimpleNamespace(init=lambda:None, device_count=lambda:1, set_device=lambda n:None,
            get_device_properties=lambda n:prop, is_initialized=lambda:True, is_bf16_supported=lambda:True)
        return types.SimpleNamespace(cuda=cuda, __version__='2.4.1')

    def inspect(self, root, native, *, query_ok=True, mapping_ok=True):
        result = subprocess.CompletedProcess(['nvidia-smi'], 0 if query_ok else 1,
            f'{A}, {RA["pci"]}\n'.encode() if query_ok else b'', b'' if query_ok else b'query failure')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=A), \
             patch.object(ci, 'runtime_mapping', return_value=([RA], dict(path='CPU_MOCK')) if mapping_ok else None) as runtime, \
             patch.object(ci, 'recorded_subprocess', return_value=result):
            if not mapping_ok:
                runtime.side_effect = RuntimeError('runtime UUID query failed')
            return ci.inspect_cuda_identity(A, root, torch_module=self.fake_torch(native))

    def test_production_interface_without_native_uuid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'identity'; value=self.inspect(root,{})
            self.assertFalse(value['native_uuid_attribute'])
            self.assertEqual(value['mapping']['selected']['uuid'], A)
            self.assertEqual(json.loads((root/'IDENTITY_PASS.json').read_text())['model_forward_calls'], 0)

    def test_production_interface_with_native_uuid(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self.inspect(Path(d)/'identity',dict(uuid=A))['native_uuid_attribute'])

    def test_present_null_or_conflicting_native_uuid_rejected(self):
        for native in (dict(uuid=None), dict(uuid=B)):
            with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
                self.inspect(Path(d)/'identity', native)

    def test_fallback_query_failure_is_persisted_and_rejected(self):
        for kw in (dict(query_ok=False), dict(mapping_ok=False)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'identity'
                with self.assertRaises(RuntimeError):self.inspect(root,{},**kw)
                self.assertTrue((root/'IDENTITY_STOP.json').is_file())
                self.assertFalse((root/'IDENTITY_PASS.json').exists())

    def test_process_success_and_failure_streams_and_returncode(self):
        for rc in (0, 7):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'process'
                cmd=[sys.executable,'-B','-c',f'import sys; print("out"); print("err",file=sys.stderr); sys.exit({rc})']
                result=ci.recorded_subprocess(root,cmd,cwd=ROOT,env=dict(os.environ))
                v=json.loads((root/'PROCESS.json').read_text())
                self.assertEqual(result.returncode,rc);self.assertEqual(v['returncode'],rc)
                self.assertEqual(v['command'],cmd);self.assertIn('out',v['stdout']);self.assertIn('err',v['stderr'])
                self.assertEqual((root/'stdout.bin').read_bytes(),result.stdout)
                self.assertEqual((root/'stderr.bin').read_bytes(),result.stderr)
                with self.assertRaises(FileExistsError):ci.recorded_subprocess(root,cmd,cwd=ROOT,env=dict(os.environ))

    def test_spawn_failure_retains_record(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'process'
            with self.assertRaises(FileNotFoundError):
                ci.recorded_subprocess(root,[str(Path(d)/'absent-executable')],cwd=ROOT,env=dict(os.environ))
            v=json.loads((root/'PROCESS.json').read_text())
            self.assertIsNone(v['returncode']);self.assertIn('FileNotFoundError',v['exception'])
            self.assertEqual((root/'stderr.bin').read_bytes(),b'')


if __name__ == '__main__':
    unittest.main()
