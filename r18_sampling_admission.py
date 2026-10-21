"""Tracked acceptance admission/launch CLI; no external field assembler."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
    from sparse_rtdetr.protocol_r18 import span_admission
    if Path(span_admission.__file__).resolve()!=ROOT/'src/sparse_rtdetr/protocol_r18/span_admission.py':
        raise RuntimeError('foreign admission dispatch module')
    return span_admission.main()
if __name__=='__main__':raise SystemExit(main())
