"""Recorded final-CLI CUDA identity preflight, no model/forward/contract/token."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sparse_rtdetr.protocol_r18 import cuda_identity as ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ci.require(ci.canonical_uuid(args.gpu_uuid) == args.gpu_uuid, 'canonical full GPU UUID required')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    cmd = [sys.executable, '-B', str(ROOT/'r18_distill.py'), 'cuda-preflight',
           '--gpu-uuid', args.gpu_uuid, '--output', str(out/'identity')]
    result = ci.recorded_subprocess(out/'process', cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        ci.write(out/'PREFLIGHT_STOP.json', dict(status='STOP_NO_RETRY', returncode=result.returncode,
            process_record=str(out/'process/PROCESS.json'), model_forward_calls=0))
        return 2
    value = json.loads(result.stdout)
    ci.require(value == json.loads((out/'identity/IDENTITY_PASS.json').read_text()), 'child stdout/audit differs')
    ci.require(value['status']=='PASS' and value['BF16_supported'] and value['model_forward_calls']==0, 'CUDA identity preflight failed')
    ci.write(out/'PREFLIGHT_PASS.json', dict(status='PASS', identity=value,
        process_record=str(out/'process/PROCESS.json'), model_forward_calls=0))
    print(json.dumps(value))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
