"""Exercise the real prepare_cuda -> live_admission path, stop before CUDA."""
import argparse
import copy
import os
from pathlib import Path
import time
from unittest.mock import patch
from sixth_view_runtime import read, publish, reference, require, install
from sixth_view_admission import adapt_resources


class CUDABoundary(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only check')
    install()
    from sparse_rtdetr.localization import worker, guard, channel
    c = read(a.contract)
    start = time.monotonic(); records = []

    def exercise(name, contract, mutate=None, success=False):
        sample = guard.snapshot(out)
        if mutate:
            mutate(sample)
        before = copy.deepcopy(contract)
        reached = []
        def boundary():
            reached.append(True)
            raise CUDABoundary()
        error = None
        with patch.object(channel, 'resource', return_value=sample), patch.object(torch.cuda, 'init', side_effect=boundary), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': c['gpu_uuid'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            try:
                worker.prepare_cuda(adapt_resources(contract), out)
            except CUDABoundary:
                require(success, 'invalid contract/resource reached CUDA: ' + name)
            except (RuntimeError, KeyError) as exc:
                error = str(exc)
                require(not success, 'real valid admission failed: ' + name + ': ' + error)
            else:
                raise RuntimeError('prepare_cuda unexpectedly returned')
        require(bool(reached) == success and not torch.cuda.is_initialized(), 'CUDA boundary mismatch')
        require(contract == before, 'adapter mutated input')
        records.append({'case': name, 'expected_cuda_boundary': success, 'reached_boundary': bool(reached), 'error': error})

    for parent in c['parents']:
        require(parent['construction']['environment'] == c['environment'], 'parent environment conflict')
        cfg = {**c, **parent['construction'], 'seed': parent['seed'], 'runtime_root': c['runtime_root']}
        exercise('seed' + str(parent['seed']) + '-actual-contract', cfg, success=True)
    both = copy.deepcopy(c); both['storage'] = {'identity': copy.deepcopy(c['storage_identity'])}
    exercise('consistent-dual-fields', both, success=True)
    for key in ('runtime_root', 'boot_id', 'gpu_uuid', 'storage_identity', 'environment'):
        bad = copy.deepcopy(c); del bad[key]; exercise('missing-' + key, bad)
    for key in c['storage_identity']:
        bad = copy.deepcopy(c); del bad['storage_identity'][key]; exercise('missing-storage-' + key, bad)
    for key in ('torch', 'torchvision', 'cuda_build', 'cudnn'):
        bad = copy.deepcopy(c); del bad['environment'][key]; exercise('missing-environment-' + key, bad)
    bad = copy.deepcopy(both); bad['storage']['identity']['st_dev'] = -1
    exercise('conflicting-storage-fields', bad)
    bad = copy.deepcopy(c); bad['storage'] = {}; exercise('incomplete-legacy-storage', bad)
    mutations = {
        'storage-identity-mismatch': lambda s: s['storage'].__setitem__('st_dev', -1),
        'boot-mismatch': lambda s: s.__setitem__('boot_id', 'wrong'),
        'stale-sample': lambda s: s.__setitem__('monotonic', time.monotonic()-26),
        'insufficient-storage': lambda s: s['storage'].__setitem__('admitted_available_bytes', 0),
        'insufficient-RAM': lambda s: s['snapshot']['resources'].__setitem__('ram_available', 0),
        'missing-RAM': lambda s: s['snapshot']['resources'].pop('ram_available'),
        'insufficient-GPU-memory': lambda s: s['snapshot']['gpu']['devices'][0]['memory.free'].__setitem__('value', 0),
        'wrong-GPU': lambda s: s['snapshot']['gpu']['devices'][0]['uuid'].__setitem__('value', 'GPU-wrong'),
    }
    for name, mutate in mutations.items():
        exercise(name, copy.deepcopy(c), mutate)
    publish(out/'ADMISSION_CPU_GATE.json', {'status': 'PASS', 'contract': reference(a.contract), 'records': records,
        'actual_path': 'adapt_resources -> unchanged prepare_cuda -> unchanged live_admission -> installed real guard.screen -> intercepted CUDA init boundary',
        'resource_samples': 'fresh actual guard snapshots; only named negative fields mutated',
        'CUDA_initialized': torch.cuda.is_initialized(), 'forwards': 0, 'elapsed_seconds': time.monotonic()-start})
    print('PASS: ' + str(len(records)) + ' actual resource admission cases; CUDA not initialized')


if __name__ == '__main__':
    main()
