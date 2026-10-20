"""Fail-closed CUDA logical/physical identity, including torch 2.4 without uuid.

No model or tensor operations. CUDA runtime PCI is uniquely joined to the real
UUID in nvidia-smi's physical inventory; names/enumeration indices are not IDs.
"""
from __future__ import annotations
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def recorded_subprocess(directory, command, *, cwd, env):
    """Persist argv, rc, both raw streams and exception before returning/raising."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = None; error = None
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
        return result
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        stdout = b'' if result is None else result.stdout
        stderr = b'' if result is None else result.stderr
        (directory/'stdout.bin').write_bytes(stdout)
        (directory/'stderr.bin').write_bytes(stderr)
        write(directory/'PROCESS.json', dict(command=list(map(str, command)), cwd=str(cwd),
            started=started, ended=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTHONPATH')},
            returncode=None if result is None else result.returncode, exception=error,
            stdout=stdout.decode('utf-8', errors='replace'), stderr=stderr.decode('utf-8', errors='replace'),
            stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest()))


def canonical_uuid(value):
    require(isinstance(value, str), 'GPU UUID must be explicit text')
    value = value.lower()
    if value.startswith('gpu-'):
        value = value[4:]
    require(re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', value), 'missing/invalid full GPU UUID')
    return 'GPU-' + value


def canonical_pci(value):
    require(isinstance(value, str), 'missing PCI address')
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    require(match is not None, 'invalid PCI address')
    domain, bus, device, function = (int(v, 16) for v in match.groups())
    return f'{domain:08x}:{bus:02x}:{device:02x}.{function:x}'


def physical_inventory(text):
    rows = []
    for line in text.splitlines():
        fields = [v.strip() for v in line.split(',')]
        require(len(fields) == 2, 'invalid nvidia-smi UUID/PCI record')
        rows.append(dict(uuid=canonical_uuid(fields[0]), pci=canonical_pci(fields[1])))
    require(rows, 'empty physical GPU inventory')
    require(len({v['uuid'] for v in rows}) == len(rows) == len({v['pci'] for v in rows}), 'ambiguous physical GPU identity')
    return rows


def validate_mapping(*, expected_uuid, runtime, inventory, logical, visible, native_uuid=None):
    """CPU-testable validator. Runtime list is measured in CUDA logical order.

    Numeric visibility is NEVER interpreted as nvidia-smi physical indices.
    Runtime PCI joined to physical UUID proves mapping even after reordering.
    """
    expected = canonical_uuid(expected_uuid)
    require(type(logical) is int and 0 <= logical < len(runtime), 'invalid logical CUDA device')
    require(len({canonical_uuid(r['uuid']) for r in runtime}) == len(runtime), 'duplicate logical CUDA identity')
    require(len({canonical_uuid(r['uuid']) for r in inventory}) == len(inventory), 'duplicate physical UUID')
    require(len({canonical_pci(r['pci']) for r in inventory}) == len(inventory), 'duplicate physical PCI')
    if visible is not None:
        tokens = visible.split(',')
        require(len(tokens) == len(runtime) and len(set(tokens)) == len(tokens), 'CUDA visibility/count ambiguity')
        for token, row in zip(tokens, runtime):
            require(token == token.strip() and token != '', 'invalid CUDA visibility token')
            if token.startswith(('GPU-', 'gpu-')):
                require(canonical_uuid(token) == canonical_uuid(row['uuid']), 'CUDA visibility UUID order conflict')
            else:
                require(re.fullmatch(r'0|[1-9][0-9]*', token) is not None, 'unsupported/ambiguous CUDA visibility')
    mapped = []
    for row in runtime:
        uid, pci = canonical_uuid(row['uuid']), canonical_pci(row['pci'])
        matches = [r for r in inventory if canonical_uuid(r['uuid']) == uid and canonical_pci(r['pci']) == pci]
        require(len(matches) == 1, 'CUDA runtime / physical UUID-PCI conflict')
        mapped.append(dict(uuid=uid, pci=pci))
    selected = mapped[logical]
    require(selected['uuid'] == expected, 'selected CUDA GPU UUID differs from contract')
    if native_uuid is not None:
        require(canonical_uuid(str(native_uuid)) == expected, 'torch native UUID conflicts with runtime/contract')
    return dict(logical_device=logical, selected=selected, logical_to_physical=mapped,
                CUDA_VISIBLE_DEVICES=visible, proof='CUDA runtime PCI uniquely joined to physical nvidia-smi UUID; native UUID cross-check when available')


def join_physical_uuid(runtime, inventory):
    """Obtain UUID ONLY from the unique physical entry for measured CUDA PCI."""
    result = []
    for row in runtime:
        pci = canonical_pci(row['pci'])
        matches = [r for r in inventory if canonical_pci(r['pci']) == pci]
        require(len(matches) == 1, 'logical CUDA PCI missing/ambiguous in physical inventory')
        result.append(dict(uuid=canonical_uuid(matches[0]['uuid']), pci=pci))
    return result


def runtime_mapping(torch_module):
    """Use the already-loaded cudart, so no second CUDA runtime is introduced."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        part = line.split()[-1]
        if '/libcudart.so.' in part:
            paths.add(str(Path(part).resolve(strict=True)))
    require(len(paths) == 1, 'loaded CUDA runtime library missing/ambiguous')
    library = paths.pop(); lib = ctypes.CDLL(library)
    def call(name, argtypes, *args):
        fn = getattr(lib, name); fn.argtypes = argtypes; fn.restype = ctypes.c_int
        code = fn(*args); require(code == 0, f'{name} failed with CUDA runtime status {code}')
    count = ctypes.c_int(); current = ctypes.c_int(); version = ctypes.c_int()
    call('cudaGetDeviceCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    call('cudaGetDevice', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(current))
    call('cudaRuntimeGetVersion', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(version))
    major, minor = map(int, torch_module.version.cuda.split('.'))
    require(version.value // 1000 == major and version.value % 1000 // 10 == minor, 'loaded CUDA runtime / torch build mismatch')
    require(count.value == torch_module.cuda.device_count() == 1 and current.value == 0, 'production logical CUDA device/count mismatch')
    rows = []
    for logical in range(count.value):
        pci = ctypes.create_string_buffer(64)
        call('cudaDeviceGetPCIBusId', [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], pci, len(pci), logical)
        rows.append(dict(pci=pci.value.decode('ascii')))
    return rows, dict(path=library, sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(), version=version.value)


def inspect_cuda_identity(expected_uuid, directory, *, torch_module=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == expected_uuid, 'production must expose the full contract UUID explicitly')
        if torch_module is None:
            import torch as torch_module
        torch_module.cuda.init()
        require(torch_module.cuda.device_count() == 1, 'production must expose exactly one CUDA device')
        torch_module.cuda.set_device(0)
        prop = torch_module.cuda.get_device_properties(0)
        native_present = hasattr(prop, 'uuid')
        if native_present:
            require(prop.uuid is not None, 'present torch UUID is null')
        rows, library = runtime_mapping(torch_module)
        write(directory/'RUNTIME_MAPPING.json', dict(devices=rows, library=library))
        result = recorded_subprocess(directory/'nvidia-smi',
            ['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'], cwd=Path.cwd(), env=dict(os.environ))
        require(result.returncode == 0, 'nvidia-smi UUID/PCI query failed')
        inventory = physical_inventory(result.stdout.decode('utf-8'))
        rows = join_physical_uuid(rows, inventory)
        mapping = validate_mapping(expected_uuid=expected_uuid, runtime=rows, inventory=inventory, logical=0,
            visible=os.environ.get('CUDA_VISIBLE_DEVICES'), native_uuid=prop.uuid if native_present else None)
        value = dict(status='PASS', mapping=mapping, native_uuid_attribute=native_present,
            cuda_runtime=library, torch=str(torch_module.__version__), CUDA_initialized=torch_module.cuda.is_initialized(),
            BF16_supported=torch_module.cuda.is_bf16_supported(), model_forward_calls=0)
        write(directory/'IDENTITY_PASS.json', value)
        return value
    except BaseException:
        write(directory/'IDENTITY_STOP.json', dict(status='STOP', traceback=traceback.format_exc(), model_forward_calls=0))
        raise
