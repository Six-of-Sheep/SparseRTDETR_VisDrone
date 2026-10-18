"""Separate supplemental namespace; no relaxation of independent confirmation."""
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from sparse_rtdetr.localization.runio import read, require, reference, digest, publish, event

TASK = 'sahi-original-c-supplemental-20261018-001'
PURPOSE = 'supplemental_evaluation_history_unverified'
PROTOCOL = 'original-c-supplemental-progress-v1'
PHASES = ('accept-A', 'accept-B', 'accept-C', 'full-C', 'evaluation')
SEALED_SHA = '1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4'


def exact(ref):
    require(reference(ref['path']) == ref, 'immutable reference changed: ' + ref['path'])
    return read(ref['path'])


def steps(c, mode):
    if mode in ('data-admission', 'input-check'):
        return ['begin', 'data_checked', 'complete']
    if mode == 'evaluation':
        return ['begin'] + [f'score-{a}-{b}' for a in 'ABC' for b in (600, 300)] + ['complete']
    n = 2 if c['synthetic'] else 16 if mode.startswith('accept-') else 401
    return ['begin', 'cuda_initialized', 'restored', 'warmup'] + [f'batch-{i:03d}' for i in range(1, n + 1)] + ['state_exact', 'complete']


def install():
    from sparse_rtdetr.protocol_r18.control import install_resources
    install_resources()
    from sparse_rtdetr.localization import channel
    if getattr(channel, '_supplemental_installed', False):
        return
    native, NativeReader = channel.validate_event, channel.Reader

    def validate(p, b, stream):
        if b.get('progress_protocol') != PROTOCOL:
            return native(p, b, stream)
        require(b['task_id'] == TASK and b['purpose'] == PURPOSE and type(b['seed']) is int and b['seed'] in (0, 1, 2), 'supplemental identity')
        require(all(channel._hex64(b[k]) for k in ('source_sha256', 'contract_sha256', 'checkpoint_sha256')), 'immutable identity')
        require(type(p.get('monotonic')) in (int, float) and math.isfinite(p['monotonic']), 'finite event clock')
        if stream != 'progress':
            return None
        require(p['event'] == 'supplemental_step' and type(p.get('step')) is int and 0 <= p['step'] < len(b['steps']), 'event schema')
        require(p['label'] == b['steps'][p['step']] and p['seed'] == b['seed'] and p['checkpoint_sha256'] == b['checkpoint_sha256'], 'event order/identity')
        return digest([p['event'], p['step']])

    class Reader(NativeReader):
        def _accept(self, r):
            ours = self.bound_identity.get('progress_protocol') == PROTOCOL and self.stream == 'progress'
            if ours and r['seq'] not in self.seen:
                require(r['payload']['step'] == self.clock.get('_supp_next', 0), 'missing/duplicate/out-of-order supplemental event')
            changed = super()._accept(r)
            if ours and changed:
                self.clock['_supp_next'] = self.clock.get('_supp_next', 0) + 1
            return changed

    channel.validate_event, channel.Reader = validate, Reader
    channel._supplemental_installed = True


def load(path):
    install()
    c = read(path)
    require(c['task_id'] == TASK and c['purpose'] == PURPOSE, 'separate supplemental task/purpose required')
    require(c['contract_sha256'] == digest({k: v for k, v in c.items() if k != 'contract_sha256'}), 'contract digest')
    require(c['source_root'] == str(ROOT), 'source root')
    source = exact(c['source_reference'])
    require(c['source_sha256'] == c['source_reference']['sha256'] and source['commit'] == c['source_commit'], 'source/commit binding')
    for r in source['files']:
        p = ROOT / r['path']
        require(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()), 'source path')
        require(reference(p) == dict(path=str(p), bytes=r['bytes'], sha256=r['sha256']), 'source changed: ' + r['path'])
    plan = exact(c['plan'])
    require(plan['task_id'] == TASK and plan['purpose'] == PURPOSE and plan['independent_qualification'] == 'BLOCKED_PENDING_ACCESS_HISTORY', 'purpose/independent status')
    require(c['seeds'] == [0, 1, 2] and c['training_updates'] == 0 and c['forward_budget'] == 6423, 'fixed execution scope')
    parent = exact(c['parent_metadata'])
    require(parent['parents'] == c['parents'] and parent['environment'] == c['environment'], 'certified parent metadata/initialization differs')
    require([p['seed'] for p in c['parents']] == [0, 1, 2] and [p['checkpoint']['sha256'] for p in c['parents']] == plan['scientific_protocol']['checkpoint_sha256'], 'three own e120 seeds')
    require(c['synthetic'] in (True, False) and type(c['synthetic']) is bool, 'fixture flag')
    if c.get('CPU_input_only', False):
        require(c['synthetic'] is False and c['GPU_authorized'] is False and c['science_authorized'] is False and c['sealed_access_authorized'] is True, 'real input CPU-only scope')
    elif c['synthetic']:
        require(not any(c[k] for k in ('GPU_authorized', 'science_authorized', 'sealed_access_authorized')), 'fixture cannot authorize production')
    else:
        require(all(c[k] is True for k in ('GPU_authorized', 'science_authorized', 'sealed_access_authorized')), 'explicit execution authority absent')
        auth = exact(c['authorization'])
        require(auth['task_id'] == TASK and auth['purpose'] == PURPOSE and auth['manifest_sha256'] == SEALED_SHA and auth['scope'] == 'fixed_three_seed_ABC_two_budgets_only', 'new user authorization differs')
        require(auth['independent_qualification'] == 'BLOCKED_PENDING_ACCESS_HISTORY' and auth['near_duplicate_or_flight_site_access'] is False, 'evidence limits')
        gate = exact(c['CPU_gate'])
        require(gate['status'] == 'PASS' and gate['source_sha256'] == c['source_sha256'] and gate['entry'] == 'supplemental_job.py', 'final entry CPU gate/source absent')
    from sparse_rtdetr.data_protocol.supplemental_zero_gt import specification, VERSION
    require(exact(c['evaluation_input_extension']) == specification(), 'explicit evaluation input extension binding')
    if not c['synthetic']:
        auth = exact(c['authorization'])
        require(auth['task_id'] == TASK and auth['purpose'] == PURPOSE and auth['manifest_sha256'] == SEALED_SHA and
                auth['evaluation_input_protocol'] == VERSION and auth['old_confirmation_not_released'] is True and
                auth['authorized'] is True and auth['scope'] == 'fixed_three_seed_ABC_two_budgets_only' and
                auth['independent_qualification'] == 'BLOCKED_PENDING_ACCESS_HISTORY' and auth['near_duplicate_or_flight_site_access'] is False, 'accepted GT protocol authority')
    require(Path(c['job_root']).is_absolute() and Path(c['runtime_root']).is_absolute() and c['job_root'] != c['runtime_root'], 'separate output/runtime')
    return c


def identity(c, seed, mode):
    from sparse_rtdetr.localization import guard
    return dict(task_id=TASK, purpose=PURPOSE, seed=seed, mode=mode, progress_protocol=PROTOCOL,
                evaluation_input_protocol='supplemental-zero-gt-input-extension-v1',
                evaluation_input_extension_sha256=c['evaluation_input_extension']['sha256'],
                source_sha256=c['source_sha256'], contract_sha256=c['contract_sha256'],
                checkpoint_sha256=c['parents'][seed]['checkpoint']['sha256'],
                output_root=str(Path(c['job_root']) / f'seed{seed}' / mode),
                supervisor_pid=os.getpid(), supervisor_start=guard.identity(os.getpid())['start'], steps=steps(c, mode))


class Emitter:
    def __init__(self, c, seed, mode, out):
        self.c, self.seed, self.mode, self.out, self.n = c, seed, mode, out, 0

    def emit(self, label, **kw):
        names = steps(self.c, self.mode)
        require(self.n < len(names) and names[self.n] == label, 'writer order/duplicate terminal')
        event(self.out, 'supplemental_step', step=self.n, label=label, seed=self.seed,
              checkpoint_sha256=self.c['parents'][self.seed]['checkpoint']['sha256'], **kw)
        self.n += 1


def live(c, out, gpu):
    from sparse_rtdetr.localization import channel, guard
    from sparse_rtdetr.localization.storage import identity as sid
    sample = channel.resource()
    require(0 <= time.monotonic() - sample['monotonic'] < 25, 'stale supplemental resource')
    require(sample['boot_id'] == c['boot_id'] and sid(sample['storage']) == c['storage_identity'], 'boot/storage identity')
    guard.screen(sample, gpu=gpu, gpu_uuid=c['gpu_uuid'], disk_bytes=40 * 1024**3)
    return sample


def closed_stream(runtime, binding):
    from sparse_rtdetr.localization import channel
    reader = channel.Reader(Path(runtime) / 'progress.jsonl', binding, 'progress')
    last, _ = reader.poll()
    require(reader.clock.get('_supp_next') == len(binding['steps']) and last['payload']['label'] == 'complete', 'incomplete supplemental terminal')
    return last
