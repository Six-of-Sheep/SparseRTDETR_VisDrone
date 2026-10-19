"""CPU-only frozen-cache diagnostic; never launches an historical job."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

TASK = 'large-recall-stage-20261019-001'
PARENT_SHA = 'b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def publish(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return {'path': str(p), 'bytes': Path(p).stat().st_size, 'sha256': sha(p)}


def dependency(parent):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[k] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    if sha(parent) != PARENT_SHA: raise ValueError('parent contract SHA differs')
    c = read(parent)
    root = Path(c['source_root'])
    for r in c['source_files']:
        p = Path(r['path'])
        if p.is_absolute() or '..' in p.parts or sha(root / p) != r['sha256']:
            raise ValueError('frozen dependency source differs')
    sys.path.insert(0, str(root))
    import softnms_job as j
    torch = j.cpu_only()
    c = j.load_contract(parent)
    specs, rows, data = j.context(c)  # Authenticates three small parent schemas; analyzes seed0 only.
    spec = next(s for s in specs if s['seed'] == 0)
    j.require(len(rows) == 548 and len(spec['full']['raw_view_refs']) == 137, 'scope differs')
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary
    def forbidden(*a, **kw): raise RuntimeError('diagnostic forbids AP/scoring/model APIs')
    ev.score_budget = ev.bridge_metrics = primary.evaluate_primary_v1 = forbidden
    parsed = j.parsed_gt(rows, data)
    return j, torch, spec, rows, parsed


def admit(parent, out):
    j, torch, spec, rows, parsed = dependency(parent)
    resource = j.resources(out, j.read(j.PLAN))
    refs = spec['full']['raw_view_refs']
    for r in refs:
        p = Path(r['path'])
        j.require(p.is_file() and p.stat().st_size == r.get('bytes', r.get('size_bytes')), 'cache missing/size differs')
    # No tensor is deserialized or diagnostic statistic computed in admission.
    ctx = publish(out/'CONTEXT.json', {'spec': spec, 'rows': rows})
    return {'status': 'ADMISSION_PASS', 'seed': 0, 'images': len(rows),
            'annotation_files_SHA_verified': len(parsed), 'raw_batches': len(refs),
            'raw_bytes': sum(Path(r['path']).stat().st_size for r in refs),
            'raw_SHA_recheck': 'deferred to one production read before every decode',
            'available': ['FP32 logits and boxes all five views', 'view/query/class identity',
                          'authenticated A/B/C pre-ignore predictions', 'canonical input/count proofs',
                          'original raw-ignore implementation and retained GT identities'],
            'source_files_verified': len(read(parent)['source_files']),
            'checkpoint': spec['identity']['checkpoint_sha256'], 'context': ctx,
            'resource': resource, 'model_calls': 0, 'GPU_calls': 0, 'AP_calls': 0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('admit', 'run', 'fixture'))
    p.add_argument('--parent', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contract', type=Path)
    p.add_argument('--inject-failure', action='store_true')
    a = p.parse_args(argv)
    if a.mode != 'fixture' and (a.parent is None or a.inject_failure): p.error('production arguments')
    out = a.output.resolve()
    if a.mode != 'fixture' and TASK not in out.parts: p.error('wrong task output')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    publish(out/'START.json', {'pid': os.getpid(), 'mode': a.mode, 'time': time.time()})
    try:
        if a.mode == 'admit': result = admit(a.parent, out)
        elif a.mode == 'fixture':
            if a.inject_failure: raise RuntimeError('injected worker failure')
            from recall_stage_core import pair_match
            result = {'status': 'COMPLETE', 'matched': pair_match([[1.]], [False], [False], .5).tolist()}
        else:
            from recall_stage_run import run
            result = run(a.parent, a.contract, out)
        result['elapsed_seconds'] = time.monotonic() - start
        ref = publish(out/'RESULT.json', result)
        publish(out/'CLOSED.json', {'result': ref, 'returncode': 0})
        print(json.dumps({'status': result['status'], 'result': ref}), flush=True)
        return 0
    except Exception:
        publish(out/'STOP.json', {'traceback': traceback.format_exc(),
                                'elapsed_seconds': time.monotonic()-start, 'automatic_retry': False})
        traceback.print_exc()
        return 2


if __name__ == '__main__': sys.exit(main())
