"""Spawn-safe H2 formal lifecycle; independent of the acceptance-only CLI."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
_dispatch_started=False
if __name__=='__main__':
    _dispatch_started=True
    from sparse_rtdetr.research.h2_formal import main
    raise SystemExit(main())
