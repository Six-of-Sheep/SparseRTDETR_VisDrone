"""Sole new entry; importing __mp_main__ never parses or dispatches."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
_dispatch_started=False
def main():
    global _dispatch_started
    _dispatch_started=True
    from sparse_rtdetr.protocol_r18.control import main as dispatch
    return dispatch()
if __name__=='__main__':
    raise SystemExit(main())
