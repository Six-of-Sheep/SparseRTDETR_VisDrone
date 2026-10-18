"""Drive final launcher, real supervisor/workers/readers/evaluators/export/exit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import supplemental_runtime as rt
from supplemental_runtime import ROOT, read, exact, require, reference, digest, publish


def fixture(parent, base):
    from sparse_rtdetr.localization.data import parse_raw, read_bound
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    spec = parent['construction']['data']['development']
    manifest = read(spec['manifest'])
    require(reference(spec['manifest'])['sha256'] == '5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd', 'actual development fixture source')
    root = base / 'synthetic'; root.mkdir()
    (root / 'images').mkdir(); (root / 'annotations').mkdir()
    rows, lineage, source_refs = [], [], []
    for i, original in enumerate(manifest['records'][:6]):
        row = copy.deepcopy(original)
        name = f'synthetic-{i}.jpg'; ann = f'synthetic-{i}.txt'
        row.update(split='train', relative_path='train/images/' + name, annotation_relative_path='train/annotations/' + ann,
                   stable_image_id=stable_image_id('train', 'train/images/' + name), sequence_key='synthetic')
        for kind, field, sf, zf, name_ in (('images', 'relative_path', 'image_sha256', 'image_size_bytes', name),
                                         ('annotations', 'annotation_relative_path', 'annotation_sha256', 'annotation_size_bytes', ann)):
            p = Path(spec['roots'][kind]) / Path(original[field]).name
            b = read_bound(p, original[sf], original[zf]); (root / kind / name_).write_bytes(b)
            source_refs.append(reference(p))
        parsed = parse_raw((root / 'annotations' / ann).read_bytes(), row['stable_image_id'], expected_rows=row['raw_annotation_rows'])
        for x in parsed['rows']:
            lineage.append(dict(image_id=row['stable_image_id'], physical_line_number=x['physical_line'], annotation_id=x['id'],
                                raw_line_sha256=x['raw_line_sha256'], bbox_xyxy=x['box'], raw_category_id=x['category']))
        rows.append(row)
    publish(root / 'manifest.json', dict(records=rows, selection_allowed=False, metrics_access_allowed=False, single_final_access_only=True))
    publish(root / 'train.json', {'records': [dict(stable_image_id='unrelated', image_sha256='0' * 64, relative_path='train/images/unrelated.jpg')]})
    with (root / 'lineage.jsonl').open('x', encoding='utf-8') as f:
        f.write(''.join(json.dumps(x) + '\n' for x in lineage))
    return dict(fixture=True, manifest=reference(root / 'manifest.json'), lineage=reference(root / 'lineage.jsonl'),
                train_manifest=reference(root / 'train.json'), images=str(root / 'images'), annotations=str(root / 'annotations')), source_refs


def tree(root):
    return {str(p.relative_to(root)): reference(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True)
    parser.add_argument('--parent-contract', required=True); parser.add_argument('--parent-contract-sha256', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args(); rt.install()
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU check boundary')
    torch.set_num_threads(1)
    base = Path(a.output); base.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    parent_ref = reference(a.parent_contract)
    require(parent_ref['sha256'] == a.parent_contract_sha256, 'CPU metadata parent contract SHA')
    parent = exact(parent_ref)
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity as sid
    sample = guard.snapshot(base); guard.screen(sample, gpu=False, entry=True, disk_bytes=40 * 1024**3)
    data, source_refs = fixture(parent['parents'][0], base)
    template = dict(task_id=rt.TASK, purpose=rt.PURPOSE, source_root=str(ROOT), source_reference=reference(a.manifest),
        source_sha256=reference(a.manifest)['sha256'], source_commit=read(a.manifest)['commit'],
        plan=reference(ROOT / 'configs/sahi_supplemental_evaluation.json'), parent_metadata=parent_ref, parents=parent['parents'], seeds=[0, 1, 2],
        synthetic=True, GPU_authorized=False, science_authorized=False, sealed_access_authorized=False,
        training_updates=0, forward_budget=6423, python=sys.executable, boot_id=sample['boot_id'],
        gpu_uuid=parent['gpu_uuid'], storage_identity=sid(sample['storage']), environment=parent['environment'], evaluation_data=data)
    # Parent/source SHA is supplied from the actual metadata receipt, never a
    # guessed directory or an unbound fixture. The production build is separate.
    cases = []
    for failure in (False, True):
        name = 'worker-error' if failure else 'success'
        c = copy.deepcopy(template)
        c.update(job_root=str(base / name), runtime_root=str(Path('/home/lyy/.cache') / rt.TASK / base.name / name),
                 contract_path=str(base / (name + '-contract.json')), inject_worker_failure=failure)
        tok = base / (name + '-token.json')
        publish(tok, dict(task_id=rt.TASK, purpose=rt.PURPOSE, source_sha256=c['source_sha256'], job_root=c['job_root']))
        c['launch_token'] = reference(tok); c['contract_sha256'] = digest(c); publish(c['contract_path'], c)
        from supplemental_job import environment, verify_phase
        env = {**environment(c), 'SUPPLEMENTAL_PYTHON': sys.executable}
        argv = ['bash', str(ROOT / 'supplemental_launch.sh'), c['contract_path']]
        with (base / (name + '.log')).open('xb') as log:
            p = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        root = Path(c['job_root'])
        require(p.returncode == (2 if failure else 0), 'actual launcher unexpected return; inspect ' + str(base / (name + '.log')))
        require((root / ('STOP.json' if failure else 'CLOSED.json')).is_file(), 'actual terminal absent')
        require(read(root / 'EXIT.json')['returncode'] == p.returncode and read(root / 'SLIM_EXPORT.json')['status'] == 'READY_FOR_SLIM_RETURN', 'export/exit not closed')
        if not failure:
            verify_phase(c, 0, 'data-admission')
            for seed in (0, 1, 2):
                for mode in rt.PHASES:
                    r = verify_phase(c, seed, mode)
                    if mode != 'evaluation':
                        require(r['actual_batch_sizes'] == [4, 2] and r['forward_calls'] == 0, 'actual CPU partial batch/fixture scope')
            summary = read(root / 'SUMMARY.json')
            require(summary['synthetic'] and summary['dual_evaluator_groups'] == 18 and summary['forward_calls'] == 0, 'actual dual evaluator scope')
        before = tree(root)
        for action in ('start', 'run'):
            args = argv if action == 'start' else [sys.executable, '-B', str(ROOT / 'supplemental_job.py'), 'run', '--contract', c['contract_path']]
            with (base / (name + '-duplicate-' + action + '.log')).open('xb') as log:
                duplicate = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            require(duplicate.returncode != 0 and tree(root) == before, 'duplicate start/run changed evidence')
        cases.append(dict(case=name, returncode=p.returncode, command=argv, terminal=reference(root / ('STOP.json' if failure else 'CLOSED.json')), duplicate_start_run_rejected=True))
    # Authenticated stream mutations exercise the same Reader used by supervisor
    # and archive; they are refusal tests, not hand-created success evidence.
    from sparse_rtdetr.localization import channel
    src = base / 'success/seed0/evaluation/runtime-archive'
    b = read(src / 'identity.json'); lines = [json.loads(x) for x in (src / 'progress.jsonl').read_bytes().splitlines()]
    invalid = {'missing': lines[:1] + lines[2:], 'reordered': lines[:1] + [lines[2], lines[1]] + lines[3:],
               'duplicate_terminal': lines + [copy.deepcopy(lines[-1])]}
    for name, values in invalid.items():
        prev = None
        for i, r in enumerate(values):
            r.update(seq=i + 1, previous=prev); r.pop('sha256', None); r['sha256'] = digest(r); prev = r['sha256']
        path = base / (name + '.jsonl'); path.write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
        try:
            channel.Reader(path, b, 'progress').poll()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('invalid stream admitted: ' + name)
    from supplemental_data import dataset
    reader_refusals = []
    original = read(base / 'success-contract.json')
    for name in ('wrong_purpose', 'missing_member', 'duplicate_member', 'wrong_member_identity', 'training_overlap', 'manifest_SHA', 'lineage_SHA'):
        bad = copy.deepcopy(original)
        if name == 'wrong_purpose':
            bad['purpose'] = 'independent_confirmation'
        elif name in ('manifest_SHA', 'lineage_SHA'):
            bad['evaluation_data']['manifest' if name == 'manifest_SHA' else 'lineage']['sha256'] = '0' * 64
        elif name == 'training_overlap':
            train = read(bad['evaluation_data']['manifest']['path'])['records'][:1]
            path = base / 'synthetic' / 'overlap-train.json'; publish(path, {'records': train})
            bad['evaluation_data']['train_manifest'] = reference(path)
        else:
            value = read(bad['evaluation_data']['manifest']['path'])
            if name == 'missing_member':
                value['records'].pop()
            elif name == 'duplicate_member':
                value['records'][-1] = value['records'][0]
            else:
                value['records'][0]['stable_image_id'] = 'wrong'
            path = base / 'synthetic' / (name + '.json'); publish(path, value)
            bad['evaluation_data']['manifest'] = reference(path)
        try:
            dataset(bad)
        except (RuntimeError, ValueError):
            reader_refusals.append(name)
        else:
            raise RuntimeError('bad reader input admitted: ' + name)
    # Exercise each actual strict restore -> adapted prepare_cuda -> original
    # live_admission path, stopping at CUDA.init rather than constructing a model.
    from supplemental_gpu import restore
    saved_resource, saved_init = channel.resource, torch.cuda.init
    sample = guard.snapshot(base)
    admitted_seeds = []
    class BeforeCUDA(Exception):
        pass
    def boundary():
        raise BeforeCUDA('EXPECTED_BEFORE_CUDA_INIT')
    try:
        channel.resource = lambda: sample
        torch.cuda.init = boundary
        os.environ['CUDA_VISIBLE_DEVICES'] = original['gpu_uuid']
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        for seed in (0, 1, 2):
            sample['monotonic'] = time.monotonic()
            try:
                restore(original, seed, base)
            except BeforeCUDA:
                admitted_seeds.append(seed)
            else:
                raise RuntimeError('pre-CUDA sentinel missed')
        resource_refusals = []
        for name in ('missing_RAM', 'low_RAM', 'wrong_GPU', 'low_GPU_memory', 'CPU_overheat', 'stale', 'wrong_boot', 'wrong_storage'):
            bad = copy.deepcopy(sample); contract = copy.deepcopy(original)
            if name == 'missing_RAM':
                del bad['snapshot']['resources']['ram_available']
            elif name == 'low_RAM':
                bad['snapshot']['resources']['ram_available'] = 0
            elif name == 'wrong_GPU':
                contract['gpu_uuid'] = 'GPU-wrong'
            elif name == 'low_GPU_memory':
                bad['snapshot']['gpu']['devices'][0]['memory.free']['value'] = 0
            elif name == 'CPU_overheat':
                bad['snapshot']['cpu_temperature_max'] = 100
            elif name == 'stale':
                bad['monotonic'] = time.monotonic() - 26
            elif name == 'wrong_boot':
                bad['boot_id'] = 'wrong'
            elif name == 'wrong_storage':
                bad['storage']['st_dev'] = -1
            channel.resource = lambda: bad
            try:
                restore(contract, 0, base)
            except BeforeCUDA:
                raise RuntimeError('bad resources reached CUDA boundary: ' + name)
            except (RuntimeError, KeyError):
                resource_refusals.append(name)
            else:
                raise RuntimeError('bad resources admitted: ' + name)
    finally:
        channel.resource, torch.cuda.init = saved_resource, saved_init
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    publish(base / 'CPU_GATE.json', dict(status='PASS', entry='supplemental_job.py', source_sha256=template['source_sha256'],
        actual_chain='supplemental_launch.sh -> start -> run -> real guard.supervise -> worker -> real purpose reader/pixels/decode -> dual evaluators -> RESULT/CLOSED -> export -> exit',
        cases=cases, synthetic_dual_evaluator_groups=18, real_development_source_refs=source_refs, actual_partial_batches=[4, 2],
        invalid_stream_refusals=list(invalid), reader_refusals=reader_refusals,
        real_restore_pre_CUDA_seed_paths=admitted_seeds, resource_refusals=resource_refusals,
        sealed_content_access=False, CUDA_initialized=False, detector_updates=0,
        production_sealed_reader_not_yet_opened=True, elapsed_seconds=time.monotonic() - start))
    print(json.dumps({'status': 'PASS', 'CPU_gate': reference(base / 'CPU_GATE.json')}))


if __name__ == '__main__':
    main()
