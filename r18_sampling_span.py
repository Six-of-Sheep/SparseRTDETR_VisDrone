"""Final one-attempt production/CPU-fixture CLI; GPU requires separate approval."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    # Identical import initialization for ALL branches and spawn imports.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
    from sparse_rtdetr.protocol_r18 import span_job
    if Path(span_job.__file__).resolve()!=ROOT/'src/sparse_rtdetr/protocol_r18/span_job.py':
        raise RuntimeError('foreign release dispatch module')
    return span_job.main()


if __name__=='__main__':
    raise SystemExit(main())
