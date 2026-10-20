"""Acceptance-only CLI. Never dispatches formal training, scoring or timing."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT))
    from sparse_rtdetr.protocol_r18 import span_acceptance
    if Path(span_acceptance.__file__).resolve() != ROOT/'src/sparse_rtdetr/protocol_r18/span_acceptance.py':
        raise RuntimeError('foreign acceptance dispatch module')
    return span_acceptance.main()


if __name__ == '__main__':
    raise SystemExit(main())
