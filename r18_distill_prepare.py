"""Real preparation entry. Intentionally provides no GPU/train/score command."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def publish(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_identity():
    names = ['r18_distill_prepare.py', 'tools/distill_cpu_fixture.py',
             'tests/test_multiview_distill.py',
             'configs/research/r18_multiview_distill_v1.json',
             'src/sparse_rtdetr/protocol_r18/distill.py',
             'src/sparse_rtdetr/protocol_r18/distill_engine.py']
    rows = []
    for name in names:
        path = ROOT/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing/linked entry dependency: '+name)
        raw = path.read_bytes()
        rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','cpu-check'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/research/r18_multiview_distill_v1.json')
    parser.add_argument('--train-manifest', type=Path)
    parser.add_argument('--parent-binding', type=Path)
    args = parser.parse_args()
    # Reject repeats before import, side effects, or changing any prior output.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(args.output).free < 1024**3:
            raise ValueError('CPU preparation requires 1GiB free output space')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ.pop('PYTHONPATH',None)
        sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('preparation must not initialize CUDA')
        torch.set_num_threads(1)
        from sparse_rtdetr.protocol_r18 import distill as kd
        if Path(kd.__file__).resolve() != ROOT/'src/sparse_rtdetr/protocol_r18/distill.py':
            raise ValueError('loaded different release module')
        plan = kd.read_plan(args.plan)
        sources = source_identity()
        metadata = dict(training_members=None,parent_binding_verified=False,
                        teacher_cache='NOT_GENERATED', real_checkpoint='SERVER_NOT_READ')
        if args.mode == 'prepare':
            if args.train_manifest is None or args.parent_binding is None:
                raise ValueError('prepare requires explicit authenticated training metadata and parent binding')
            rows = kd.train_members(args.train_manifest)
            raw = args.parent_binding.read_bytes()
            if hashlib.sha256(raw).hexdigest() != '93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938':
                raise ValueError('wrong certified seed0 parent binding SHA')
            parent = json.loads(raw)
            if parent['scope'] != 'formal_R18_seed0_e120' or parent['config']['seed'] != 0:
                raise ValueError('parent binding is not seed0')
            metadata.update(training_members=len(rows), training_manifest_sha256=kd.TRAIN_SHA,
                            training_order_sha256=kd.digest([r['stable_image_id'] for r in rows]),
                            parent_binding_verified=True,
                            parent_binding_sha256=hashlib.sha256(raw).hexdigest())
        else:
            command = [sys.executable,'-B',str(ROOT/'tools/distill_cpu_fixture.py'),str(args.output/'fixture')]
            with (args.output/'CPU_ENTRY.log').open('x',encoding='utf-8') as stream:
                child = subprocess.run(command, cwd=ROOT, env=dict(os.environ), stdout=stream, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError('real CPU worker failed, exit='+str(child.returncode))
            proof = json.loads((args.output/'fixture/RESULT.json').read_text(encoding='utf-8'))
            if proof['status'] != 'CPU_PASS' or proof['arms'] != plan['arms'] or proof['CUDA_initialized'] is not False:
                raise ValueError('CPU child did not close required interfaces')
            metadata['fixture'] = proof
        result = dict(status='CPU_PREPARATION_PASS',mode=args.mode, metadata=metadata,
                      plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                      entry_sources=sources, actual_module=str(Path(kd.__file__).resolve()),
                      actual_python=sys.executable, actual_cwd=str(Path.cwd()),
                      elapsed_seconds=time.monotonic()-started, CUDA_initialized=torch.cuda.is_initialized(),
                      GPU_calls=0, scientific_updates=0, scientific_scoring_groups=0,
                      production_token_created=False, GPU_ready=False)
        publish(args.output/'RESULT.json',result)
        publish(args.output/'CLOSED.json',dict(status='CPU_PASS',result_sha256=hashlib.sha256((args.output/'RESULT.json').read_bytes()).hexdigest()))
        print(json.dumps(dict(status=result['status'],mode=args.mode,output=str(args.output)),ensure_ascii=True))
        return 0
    except BaseException as error:
        publish(args.output/'STOP.json',dict(status='STOP_CPU_PREPARATION',error=type(error).__name__,message=str(error),
                                           elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
