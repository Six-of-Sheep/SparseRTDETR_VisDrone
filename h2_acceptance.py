"""Spawn-safe H2 acceptance-only CLI. No formal/evaluation/cost dispatch."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
_dispatch_started=False
if __name__=='__main__':
    _dispatch_started=True
    from sparse_rtdetr.research.h2_acceptance import main
    raise SystemExit(main())
