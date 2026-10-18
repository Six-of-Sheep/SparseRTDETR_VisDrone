"""Unique supervised supplemental A/B/C chain; no training command."""
import argparse
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import supplemental_runtime as rt
from supplemental_runtime import ROOT, read, exact, require, reference, digest, publish


def environment(c, gpu=False):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': c['gpu_uuid'] if gpu else '',
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT / 'src'),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2' if gpu else '1',
            'MKL_NUM_THREADS': '2' if gpu else '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_THREADING_LAYER': 'GNU'}


def command(c, action, seed=None, mode=None):
    argv = [c['python'], '-B', str(ROOT / 'supplemental_job.py'), action, '--contract', c['contract_path']]
    if seed is not None:
        argv += ['--seed', str(seed), '--mode', mode]
    return argv


def verify_phase(c, seed, mode):
    rt.install()
    from sparse_rtdetr.localization import channel
    out = Path(c['job_root']) / f'seed{seed}' / mode
    runtime = Path(c['runtime_root']) / f'seed{seed}' / mode
    b = read(runtime / 'identity.json')
    expected = rt.identity(c, seed, mode)
    for key in ('task_id', 'purpose', 'seed', 'mode', 'source_sha256', 'contract_sha256', 'checkpoint_sha256', 'steps', 'output_root'):
        require(b[key] == expected[key], 'phase identity: ' + key)
    closed = read(out / 'CLOSED.json')
    require(closed['result'] == reference(out / 'RESULT.json') and closed['archive'] == reference(out / 'runtime-archive-index.json'), 'closed result/archive reference')
    channel.verify_archive(runtime, out, b); rt.closed_stream(runtime, b)
    ex = read(runtime / 'supervisor-exit.json')
    require(ex['status'] == 'PASS' and ex['returncode'] == 0 and not any(ex[k] for k in ('remaining_live', 'unknown_pids', 'cleanup_errors')), 'phase exit incomplete')
    require(read(out / 'execution-binding.json') == b, 'worker binding differs')
    value = read(out / 'RESULT.json'); require(value['status'] == 'PASS', 'phase failure')
    return value


def supervise_phase(c, seed, mode):
    from sparse_rtdetr.localization import guard, channel
    out = Path(c['job_root']) / f'seed{seed}' / mode
    runtime = Path(c['runtime_root']) / f'seed{seed}' / mode
    gpu = not c['synthetic'] and mode not in ('evaluation', 'data-admission')
    require(read(out.parent / (mode + '-CONSUMED.json'))['contract_sha256'] == c['contract_sha256'], 'phase token binding')
    code = guard.supervise(command(c, 'worker', seed, mode), out, environment(c, gpu),
                           gpu_uuid=c['gpu_uuid'], boot_id=c['boot_id'], gpu=gpu, formal=False,
                           checkpoint_bytes=0, runtime=runtime, run_identity=rt.identity(c, seed, mode))
    if code:
        return code
    b = read(runtime / 'identity.json')
    channel.verify_archive(runtime, out, b); rt.closed_stream(runtime, b)
    publish(out / 'CLOSED.json', {'status': 'PASS', 'result': reference(out / 'RESULT.json'), 'archive': reference(out / 'runtime-archive-index.json')})
    verify_phase(c, seed, mode)
    return 0


def phase(c, seed, mode):
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity as sid
    root = Path(c['job_root']) / f'seed{seed}'
    root.mkdir(exist_ok=True)
    require(not (root / mode).exists() and not (root / (mode + '-CONSUMED.json')).exists(), 'phase already consumed')
    sample = guard.snapshot(root)
    require(sample['boot_id'] == c['boot_id'] and sid(sample['storage']) == c['storage_identity'], 'fresh boot/storage differs')
    gpu = not c['synthetic'] and mode not in ('evaluation', 'data-admission')
    guard.screen(sample, gpu=gpu, gpu_uuid=c['gpu_uuid'], entry=True, disk_bytes=40 * 1024**3)
    publish(root / (mode + '-ADMISSION.json'), sample)
    publish(root / (mode + '-CONSUMED.json'), {'contract_sha256': c['contract_sha256'], 'seed': seed, 'mode': mode, 'no_retry': True})
    start = time.monotonic()
    with (root / (mode + '-supervisor.log')).open('xb') as log:
        p = subprocess.run(command(c, 'supervise', seed, mode), env=environment(c), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    publish(root / (mode + '-EXIT.json'), {'returncode': p.returncode, 'elapsed_seconds': time.monotonic() - start})
    require(p.returncode == 0, 'supervised phase failed: ' + mode)
    return verify_phase(c, seed, mode)


def synthesize(c, seed, mode, out, emit):
    """Explicit CPU fixture, shares reader/pixels/decode and dynamic batch path."""
    import torch
    from supplemental_data import dataset
    from supplemental_gpu import decode_merge
    from infer_seed import make_view_pixels
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized(), 'CPU fixture CUDA boundary')
    data, rows, parsed = dataset(c)
    emit.emit('cuda_initialized', fixture_only=True, CUDA_initialized=False)
    emit.emit('restored', fixture_only=True, model_constructed=False)
    emit.emit('warmup', fixture_only=True, forward_calls=0)
    preds = {a: {'600': [], '300': []} for a in ('ABC' if mode == 'full-C' else mode[-1])}
    sizes = []
    for k in range(2):
        br = rows[k * 4:k * 4 + 4]; sizes.append(len(br))
        views, _ = make_view_pixels(data, br, mode in ('accept-C', 'full-C'))
        records = []
        for vid, tensors in enumerate(views):
            require(all(x.shape == (3, 768, 1344) and torch.isfinite(x).all() for x in tensors), 'actual fixture resize/crops')
            logits = torch.full((len(br), 300, 10), -4., dtype=torch.float32)
            logits[:, 0, 0] = 2.
            boxes = torch.tensor([.5, .5, .1, .1], dtype=torch.float32).repeat(len(br), 300, 1)
            records.append(dict(view_id=vid, pred_logits=logits, pred_boxes=boxes))
        value = decode_merge(records, br, mode)
        for arm in preds:
            for budget in ('600', '300'):
                preds[arm][budget].extend(value[arm][budget])
        emit.emit(f'batch-{k + 1:03d}', fixture_only=True, actual_batch=len(br), forward_calls=0)
    require(sizes == [4, 2], 'real CPU partial batch must be unpadded')
    emit.emit('state_exact', fixture_only=True, model_constructed=False)
    refs = {}
    for arm in preds:
        refs[arm] = {}
        for budget in preds[arm]:
            path = out / f'{arm}-top{budget}-predictions.json'; publish(path, preds[arm][budget]); refs[arm][budget] = reference(path)
    publish(out / 'RESULT.json', dict(status='PASS', seed=seed, synthetic=True, predictions=refs,
                                    forward_calls=0, detector_updates=0, actual_batch_sizes=sizes, e2e_seconds=1.))
    emit.emit('complete', result=reference(out / 'RESULT.json'))


def evaluation(c, seed, out, emit):
    from supplemental_data import dataset
    from confirmation_prepare import evaluator_inputs
    from sparse_rtdetr.r0.evaluation import score_budget
    _, rows, parsed = dataset(c)
    infer = verify_phase(c, seed, 'full-C')
    metrics, refs = {}, []
    for arm in 'ABC':
        metrics[arm] = {}
        for budget in ('600', '300'):
            rt.live(c, out, False)
            preds = exact(infer['predictions'][arm][budget])
            proof = evaluator_inputs(rows, parsed, preds, int(budget))
            value = score_budget(rows, parsed, preds, ROOT)
            require(value['official']['canonical_input_sha256'] == proof['canonical_input_sha256'] and value['spatial_filter_counts'] == proof['counts'], 'scored input/ignore stage differs')
            require(value['official']['evaluated_detection_count'] == proof['counts']['detections_after'], 'evaluated detection count')
            value['supplemental_binding'] = {'task_id': rt.TASK, 'purpose': rt.PURPOSE, 'seed': seed, 'arm': arm, 'budget': int(budget),
                'checkpoint_sha256': c['parents'][seed]['checkpoint']['sha256'], 'source_sha256': c['source_sha256'], 'contract_sha256': c['contract_sha256'],
                'synthetic': c['synthetic'], 'independent_qualification': 'BLOCKED_PENDING_ACCESS_HISTORY'}
            path = out / f'{arm}-top{budget}-metrics.json'; publish(path, value)
            summary = {'COCO_bridge': value['COCO_bridge']['metrics'],
                       'official_pp': {k: value['official'][k] for k in ('AP', 'AP50', 'AP75')},
                       'counts': value['spatial_filter_counts'], 'canonical_input_sha256': proof['canonical_input_sha256']}
            metrics[arm][budget] = summary; refs.append(reference(path))
            emit.emit(f'score-{arm}-{budget}', metrics=refs[-1])
    decision = compare(metrics, exact(c['plan'])['scientific_protocol']['C_minus_A_minimum_pp'])
    publish(out / 'COMPARISON.json', decision)
    publish(out / 'RESULT.json', {'status': 'PASS', 'seed': seed, 'synthetic': c['synthetic'], 'scores': metrics,
                                 'dual_evaluator_groups': 6, 'metrics_refs': refs, 'comparison': reference(out / 'COMPARISON.json'),
                                 'forward_calls': 0, 'detector_updates': 0})
    emit.emit('complete', result=reference(out / 'RESULT.json'))


def compare(metrics, limits):
    comparisons = {}
    for new, old in (('C', 'A'), ('B', 'A'), ('C', 'B')):
        comparisons[new + '-' + old] = {}
        for budget in ('600', '300'):
            a, b = metrics[new][budget], metrics[old][budget]
            delta = {}
            for key in a['COCO_bridge']:
                x, y = a['COCO_bridge'][key], b['COCO_bridge'][key]
                require(math.isfinite(x) and math.isfinite(y), 'nonfinite science metric')
                delta[key] = None if x < 0 or y < 0 else (x - y) * 100.
            comparisons[new + '-' + old][budget] = {'COCO_bridge_delta_pp': delta,
                'official_delta_pp': {key: a['official_pp'][key] - b['official_pp'][key] for key in a['official_pp']}}
    gates = {}
    for name, limit in limits.items():
        budget, key = name.split('_', 1)
        value = comparisons['C-A'][budget]['COCO_bridge_delta_pp'][key]
        gates[name] = {'delta_pp': value, 'minimum_pp': limit, 'passed': value is not None and value >= limit,
                       'status': 'NOT_EVALUABLE' if value is None else 'PASS' if value >= limit else 'FAIL'}
    return {'comparisons': comparisons, 'quality_conditions': gates, 'quality_passed': all(v['passed'] for v in gates.values()),
            'conclusion_scope': rt.PURPOSE, 'independent_qualification': 'BLOCKED_PENDING_ACCESS_HISTORY'}


def worker(c, seed, mode):
    import torch
    from sparse_rtdetr.localization import channel, guard
    out = Path(c['job_root']) / f'seed{seed}' / mode
    b = channel.binding(); require(b is not None and read(Path(b['runtime_root']) / 'identity.json') == b, 'bound real worker')
    require(os.getppid() == b['supervisor_pid'] and guard.identity(os.getppid())['start'] == b['supervisor_start'], 'worker owner lifecycle')
    for key, value in rt.identity(c, seed, mode).items():
        if key not in ('supervisor_pid', 'supervisor_start'):
            require(b[key] == value, 'worker identity: ' + key)
    publish(out / 'execution-binding.json', b); emit = rt.Emitter(c, seed, mode, out)
    try:
        rt.live(c, out, not c['synthetic'] and mode not in ('data-admission', 'evaluation')); emit.emit('begin')
        if c['synthetic']:
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU fixture visibility')
            torch.set_num_threads(1)
            if c.get('inject_worker_failure'):
                raise RuntimeError('EXPECTED_CPU_WORKER_EXCEPTION')
        if mode == 'data-admission':
            from supplemental_data import dataset, verify_images
            data, rows, parsed = dataset(c)
            result = verify_images(data, c); result.update(seed=seed, synthetic=c['synthetic'], detector_updates=0, forward_calls=0)
            publish(out / 'DATA_ADMISSION.json', result); emit.emit('data_checked', proof=reference(out / 'DATA_ADMISSION.json'))
            publish(out / 'RESULT.json', {'status': 'PASS', 'proof': reference(out / 'DATA_ADMISSION.json'), 'forward_calls': 0})
            emit.emit('complete', result=reference(out / 'RESULT.json'))
        elif mode == 'evaluation':
            evaluation(c, seed, out, emit)
        elif c['synthetic']:
            synthesize(c, seed, mode, out, emit)
        else:
            from supplemental_gpu import run as gpu_run
            gpu_run(c, seed, mode, out, emit)
        publish(channel.receipt_root(out) / 'worker-exit.json', {'status': 'PASS', 'success': True, 'pid': os.getpid()})
    except BaseException as error:
        failure = {'status': 'STOP', 'error': repr(error), 'traceback': traceback.format_exc(), 'seed': seed, 'mode': mode, 'automatic_retry': False}
        publish(out / 'STOP.json', failure); publish(channel.receipt_root(out) / 'failure.json', failure)
        publish(channel.receipt_root(out) / 'worker-exit.json', {'status': 'STOP', 'success': False, 'pid': os.getpid()})
        raise


def export(c):
    root = Path(c['job_root']); files = []
    names = {'RESULT.json', 'CLOSED.json', 'STOP.json', 'COMPARISON.json', 'COST.json', 'RESTORE.json', 'DATA_ADMISSION.json', 'execution-binding.json', 'SUMMARY.json'}
    for path in sorted(root.rglob('*.json')):
        if path.name in names and 'runtime-archive' not in path.parts:
            files.append(reference(path))
    publish(root / 'SLIM_EXPORT.json', {'status': 'READY_FOR_SLIM_RETURN', 'files': files,
        'full_raw_predictions_logs_checkpoint': 'REMOTE_ONLY', 'index_is_not_backup': True})


def run(c):
    root = Path(c['job_root']); require(not (root / 'CONSUMED.json').exists(), 'attempt already consumed')
    token = exact(c['launch_token'])
    require(token['task_id'] == rt.TASK and token['purpose'] == rt.PURPOSE and token['source_sha256'] == c['source_sha256'] and token['job_root'] == c['job_root'], 'new launch token binding')
    publish(root / 'CONSUMED.json', {'contract_sha256': c['contract_sha256'], 'token': c['launch_token'], 'no_retry': True})
    started = time.monotonic(); completed = []
    try:
        phase(c, 0, 'data-admission')
        # Cost admission for all seeds precedes the fixed full-set scientific chain.
        for seed in c['seeds']:
            rs = {mode: phase(c, seed, mode) for mode in rt.PHASES[:3]}
            a = rs['accept-A']['e2e_seconds']; require(a > 0, 'positive A timing')
            cost = {'seed': seed, 'C_over_A': rs['accept-C']['e2e_seconds'] / a, 'B_over_A': rs['accept-B']['e2e_seconds'] / a,
                    'limit': 6., 'synthetic': c['synthetic'], 'all_real_views_included': not c['synthetic'],
                    'timings': {k: {x: v[x] for x in ('e2e_seconds', 'forward_calls')} for k, v in rs.items()}}
            cost['passed'] = cost['C_over_A'] <= 6.
            publish(root / f'seed{seed}' / 'COST.json', cost)
            require(cost['passed'], 'fixed cost admission failed; no retry')
        for seed in c['seeds']:
            phase(c, seed, 'full-C'); evaluation_result = phase(c, seed, 'evaluation')
            completed.append({'seed': seed, 'scores': evaluation_result['scores'], 'quality': read(root / f'seed{seed}/evaluation/COMPARISON.json'),
                              'cost': read(root / f'seed{seed}/COST.json')})
        stats = {}
        for contrast in ('C-A', 'B-A', 'C-B'):
            stats[contrast] = {}
            for budget in ('600', '300'):
                stats[contrast][budget] = {}
                for key in completed[0]['quality']['comparisons'][contrast][budget]['COCO_bridge_delta_pp']:
                    values = [x['quality']['comparisons'][contrast][budget]['COCO_bridge_delta_pp'][key] for x in completed]
                    stats[contrast][budget][key] = {'paired_delta_pp': values, 'mean_pp': statistics.mean(values) if None not in values else None,
                        'sample_sd_pp': statistics.stdev(values) if None not in values else None, 'worst_pp': min(values) if None not in values else None}
        forward_calls = sum(verify_phase(c, s, m)['forward_calls'] for s in c['seeds'] for m in rt.PHASES[:4])
        require(forward_calls == (0 if c['synthetic'] else 6423), 'total fixed forward budget')
        summary = {'status': 'COMPLETE', 'task_id': rt.TASK, 'purpose': rt.PURPOSE, 'seeds': completed, 'paired_statistics': stats,
                   'quality_passed': all(x['quality']['quality_passed'] for x in completed), 'cost_passed': all(x['cost']['passed'] for x in completed),
                   'forward_calls': forward_calls, 'detector_updates': 0, 'synthetic': c['synthetic'], 'dual_evaluator_groups': 18,
                   'independent_qualification': 'BLOCKED_PENDING_ACCESS_HISTORY', 'historical_use': 'unknown', 'near_duplicates': 'unknown', 'flight_site': 'unknown'}
        publish(root / 'SUMMARY.json', summary)
        publish(root / 'RESULT.json', {'status': 'COMPLETE', 'summary': reference(root / 'SUMMARY.json'), 'elapsed_seconds': time.monotonic() - started, 'synthetic': c['synthetic']})
        publish(root / 'CLOSED.json', {'status': 'PASS', 'result': reference(root / 'RESULT.json')}); export(c)
        return 0
    except BaseException as error:
        publish(root / 'STOP.json', {'status': 'STOP', 'error': repr(error), 'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started, 'automatic_retry': False})
        export(c); return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'run', 'supervise', 'worker'))
    parser.add_argument('--contract', required=True); parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--mode', choices=('data-admission', *rt.PHASES))
    a = parser.parse_args(); c = rt.load(a.contract)
    if a.action == 'worker':
        return worker(c, a.seed, a.mode)
    if a.action == 'supervise':
        return supervise_phase(c, a.seed, a.mode)
    if a.action == 'run':
        return run(c)
    root = Path(c['job_root']); require(not root.exists() and not Path(c['runtime_root']).exists(), 'attempt output/runtime already exists')
    root.mkdir(parents=True, exist_ok=False)
    publish(root / 'LAUNCH.json', {'contract': reference(a.contract), 'pid': os.getpid(), 'task_id': rt.TASK})
    code = subprocess.call(command(c, 'run'), env=environment(c))
    publish(root / 'EXIT.json', {'returncode': code, 'epoch': time.time()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
