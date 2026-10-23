"""Formal e120 chain with explicit external engineering acceptance evidence."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
    from sparse_rtdetr.protocol_r18 import span_formal
    if Path(span_formal.__file__).resolve()!=ROOT/'src/sparse_rtdetr/protocol_r18/span_formal.py':raise RuntimeError('foreign formal module')
    return span_formal.main()
if __name__=='__main__':raise SystemExit(main())
