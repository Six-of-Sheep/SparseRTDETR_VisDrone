"""Guarded single CPU cache diagnosis. No detector, training, new forward or AP/AR.

CLI check is read-only. execute consumes one exclusive output and one worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LIMITS = dict(wall_seconds=1800, cpu_threads=2, rss_bytes=4 * 1024**3,
              output_bytes=20 * 1024**2, return_bytes=5 * 1024**2)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def exact(ref):
    require(set(ref) == {'path', 'bytes', 'sha256'}, 'reference fields differ')
    p = Path(ref['path'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink() and p == p.resolve(), 'bound file missing or symlink')
    require(p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256'], 'bound SHA/size mismatch: ' + str(p))
    return p


def read(ref):
    return json.loads(exact(ref).read_text())


def publish(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def contract(path, expected):
    require(sha(path) == expected, 'contract SHA differs')
    c = json.loads(Path(path).read_text())
    require(c['schema'] == 'r18-cache-diagnostic-v1' and c['limits'] == LIMITS, 'contract schema/limits differ')
    require(c['profile'] in ('REAL_NUD_548', 'SYNTHETIC_FIXTURE'), 'unknown profile')
    require(c['arms_order'] == ['N', 'U', 'D'] and set(c['arms']) == {'N', 'U', 'D'}, 'arm identities differ')
    require(c['thresholds'] == [.5, .75] and c['budgets'] == [600, 300], 'statistics changed')
    require(c['images'] == (548 if c['profile'] == 'REAL_NUD_548' else 4), 'image coverage differs')
    require(c['no_GPU_forward_training_AP_AR'] is True and c['supplemental1602'] is False, 'scope changed')
    for f, value in c['runtime_files'].items():
        require('/' not in f and sha(Path(__file__).parent / f) == value, 'executed source differs')
    if c['profile'] == 'REAL_NUD_548':
        require(c['manifest']['sha256'] == '5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd', 'not original development548')
        expected_cp = {'N': 'be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83',
                       'U': 'd64b9bcbfed42ad2ce1e015827bafbd06d6ef3ec8ccc5738c723ae4d0b455619',
                       'D': '00eca9fefb5e9efbf9e05cad3292b75c82e125c66a4944b6817b5d1447ebd4a5'}
        require(all(c['arms'][a]['checkpoint']['sha256'] == v for a, v in expected_cp.items()), 'endpoint identity differs')
        expected_infer = {'N':'99609528d93f60dc79572b005c3cd3a5dd4d453adb3ef9c5e328751ac55102a9',
                          'U':'e7621f66edc3c97650726794084bd7e45808a895f0e1b5c279334fbb338fcecd',
                          'D':'5b43acf5a7bcbc2140da53f729120c50f14cee5d0271cea4b60b3f5ab3f3d919'}
        require(all(c['arms'][a]['infer_result']['sha256'] == v for a,v in expected_infer.items()), 'raw endpoint receipt changed')
        require(c['native_manifest']['sha256'] == 'c2200ead1a3a45620124b6c1fece499d64515c4cedca538e0231458da9728d07', 'native source differs')
        require(all(Path(c['arms'][a]['infer_result']['path']).parent.name == ('infer-B0' if a == 'N' else 'infer-' + a) for a in c['arms']), 'infer stage identity differs')
    return c


def preflight(c, guard=lambda: None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    require(all(os.environ.get(x) == '2' for x in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'thread environment differs')
    manifest = read(c['manifest'])
    rows = manifest['records']
    require(len(rows) == c['images'] and len({r['stable_image_id'] for r in rows}) == len(rows), 'manifest coverage/duplicate IDs')
    require(all(r['split'] == 'val' and r['width'] > 0 and r['height'] > 0 for r in rows), 'non-development member or invalid dimensions')
    ar = Path(c['annotation_root'])
    require(ar.is_absolute() and ar.name == 'annotations' and ar.parent.name == 'development', 'wrong annotation role root')
    forbidden = ('1602', 'holdout', 'testdev', 'test_dev', 'confirmatory')
    require(not any(x in str(ar).lower() for x in forbidden), 'forbidden role path')
    native = read(c['native_manifest'])
    require(native['commit'] == c['native_commit'], 'native commit differs')
    source_rows = native['source_files']
    require(len({r['path'] for r in source_rows}) == len(source_rows), 'duplicate native source path')
    for row in source_rows:
        require(not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts, 'source path escape')
        exact(dict(path=str(Path(c['native_root']) / row['path']), bytes=row['bytes'], sha256=row['sha256']))
        guard()
    if c['profile'] == 'REAL_NUD_548':
        old = read(c['N_source']['manifest'])
        require(set(old) == {'commit','scope','files'} and old['commit'] == 'a97e1eaf750c69c4ee72af232d07632b1aced931', 'N source identity/schema differs')
        require(c['N_source']['manifest']['sha256'] == 'c91451b3db676122b9f705ed3e4647f2318c4f4c1437e35107f856e383c7ecf1', 'N source manifest changed')
        require(len({r['path'] for r in old['files']}) == len(old['files']), 'duplicate N source path')
        for r in old['files']:
            require(not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts, 'N source path escape')
            exact(dict(path=str(Path(c['N_source']['root'])/r['path']),bytes=r['bytes'],sha256=r['sha256']))
            guard()
        for ref in c['origin_contracts']:
            original = read(ref)
            require(original['data']['development']['manifest']['sha256'] == c['manifest']['sha256'], 'origin development identity differs')
    proof = {}
    for arm in c['arms_order']:
        a = c['arms'][arm]
        ep, binding, result, closed = (read(a[k]) for k in ('endpoint', 'binding', 'infer_result', 'infer_closed'))
        require(ep['epoch'] == 120 and ep['updates'] == 36480 and ep['EMA'] and not ep['used_acceptance_state'], 'not fixed e120 EMA')
        expected_checkpoint = dict(path=ep['checkpoint']['path'],bytes=ep['checkpoint']['size_bytes'],sha256=ep['checkpoint']['sha256'])
        require(ep['binding'] == a['binding'] and expected_checkpoint == a['checkpoint'], 'endpoint binding differs')
        require(binding['arm'] == ('r18-sampling-span-v1/B0' if arm == 'N' else arm), 'checkpoint arm mismatch')
        require(binding['config']['input_hw'] == [768, 1344] and binding['config']['seed'] == 0, 'checkpoint geometry/seed mismatch')
        require(closed['status'] == result['status'] == 'PASS' and closed['result'] == a['infer_result'], 'inference closure differs')
        if arm != 'N':
            require(result['contract_sha256'] == binding['contract_sha256'] and binding['source_sha256'] == c['native_manifest']['sha256'], 'inference/checkpoint source or contract differs')
        require(result['images'] == c['images'] and result['EMA'] and result['forward_calls'] == (c['images'] + 3) // 4, 'inference coverage/EMA differs')
        require(len(result['raw_batches']) == (c['images'] + 3) // 4, 'truncated/missing raw cache')
        require(len({x['path'] for x in result['raw_batches']}) == len(result['raw_batches']), 'duplicate cache path')
        for i, ref in enumerate(result['raw_batches']):
            require(Path(ref['path']).name == f'raw-{i:03d}.pth' and Path(ref['path']).parent == Path(a['infer_result']['path']).parent, 'raw-stage correspondence differs')
            exact(ref)
            guard()
        for ref in result['predictions'].values():
            exact(ref)
            guard()
        require(set(a['scores']) == {'600','300'}, 'score reference budgets differ')
        for ref in a['scores'].values():
            read(ref)
            guard()
        exact(a['checkpoint'])  # chunked physical SHA; never torch.load(checkpoint)
        guard()
        if arm == 'N':
            read(c['original_N_STOP'])
            require(not Path(c['original_N_CLOSED']).exists(), 'historical missing closure changed')
        proof[arm] = dict(endpoint=a['endpoint'], checkpoint=a['checkpoint'], infer_result=a['infer_result'],
                          infer_closed=a['infer_closed'], raw_batches=len(result['raw_batches']),
                          raw_bytes=sum(x['bytes'] for x in result['raw_batches']), full_query_shape=[300, 10])
    for row in rows:
        name = Path(row['annotation_relative_path']).name
        require(row['annotation_relative_path'].startswith('val/annotations/') and name.endswith('.txt'), 'annotation role differs')
        exact(dict(path=str(ar / name), bytes=row['annotation_size_bytes'], sha256=row['annotation_sha256']))
        guard()
    return rows, proof


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def output_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def worker(c, out, started):
    def guard():
        require(time.monotonic() - started < c['limits']['wall_seconds'], 'wall budget exceeded')
        require(rss(os.getpid()) <= c['limits']['rss_bytes'], 'RSS budget exceeded')
        require(output_bytes(out.parent) <= c['limits']['output_bytes'], 'new storage budget exceeded')
    try:
        rows, proof = preflight(c, guard)
        publish(out / 'ADMISSION.json', dict(status='PASS', proof=proof, images=len(rows), full_cache=True,
                                            checkpoint_loaded=False, GPU_forward=0))
        from r18_cache_worker import diagnose
        result = diagnose(c, rows, out, guard)
        result.update(status='PASS', elapsed_seconds=time.monotonic() - started, max_rss_bytes=rss(os.getpid()))
        publish(out / 'RESULT.json', result)
        guard()
    except BaseException as e:
        publish(out / 'STOP.json', dict(status='STOP_NO_RETRY', error=type(e).__name__ + ': ' + str(e),
                                        elapsed_seconds=time.monotonic() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'execute', '_worker'])
    parser.add_argument('--contract', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    c = contract(args.contract, args.sha256)
    out = Path(c['output'])
    if args.mode == 'check':
        require(not out.exists(), 'output already consumed')
        preflight(c)
        print(json.dumps(dict(status='CHECK_PASS', profile=c['profile'], output_created=False)))
        return
    if args.mode == '_worker':
        require(os.environ.get('R18_DIAG_PARENT') == str(os.getppid()) and (out / 'CONSUMED.json').is_file(), 'worker has no single supervisor')
        worker(c, out, float(os.environ['R18_DIAG_STARTED']))
        return
    started = time.monotonic()
    out.mkdir(parents=False, exist_ok=False)
    publish(out / 'CONSUMED.json', dict(contract_sha256=args.sha256, no_retry=True, started=started))
    cpus = sorted(os.sched_getaffinity(0))[:2]
    require(len(cpus) == 2, 'two CPU affinity slots unavailable')
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ, R18_DIAG_PARENT=str(os.getpid()), R18_DIAG_STARTED=str(started))
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '_worker', '--contract', args.contract, '--sha256', args.sha256]
    violation = None
    peak = 0
    with (out / 'worker.stdout').open('xb') as stdout, (out / 'worker.stderr').open('xb') as stderr:
        child = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            peak = max(peak, rss(child.pid) + rss(os.getpid()))
            if time.monotonic() - started >= LIMITS['wall_seconds']:
                violation = 'wall budget exceeded'
            if peak > LIMITS['rss_bytes']:
                violation = 'RSS budget exceeded'
            if output_bytes(out.parent) > LIMITS['output_bytes']:
                violation = 'new storage budget exceeded'
            if violation:
                child.terminate()  # only this invocation's owned worker; never historical jobs
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            time.sleep(.1)
    ok = child.returncode == 0 and violation is None and (out / 'RESULT.json').exists() and not (out / 'STOP.json').exists()
    if not ok and not (out / 'STOP.json').exists():
        publish(out / 'STOP.json', dict(status='STOP_NO_RETRY', error=violation or 'worker exited before result'))
    publish(out / 'EXIT.json', dict(status='PASS' if ok else 'STOP_NO_RETRY', returncode=child.returncode,
                                  error=violation, elapsed_seconds=time.monotonic() - started,
                                  peak_total_rss_bytes=peak, cpu_affinity=cpus, argv=argv))
    if ok:
        files = {p.name: reference(p) for p in out.iterdir() if p.is_file()}
        publish(out / 'CLOSED.json', dict(status='PASS', result=files['RESULT.json'], exit=files['EXIT.json'], files=files))
    print(json.dumps(dict(status='PASS' if ok else 'STOP_NO_RETRY', output=str(out), returncode=child.returncode)))
    if not ok:
        sys.exit(2)


if __name__ == '__main__':
    main()
