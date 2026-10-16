"""Only R0 entry; historical F0/C2 and distribution-arm actions are forbidden."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from sparse_rtdetr.r0.control import main
if __name__=="__main__":raise SystemExit(main())
