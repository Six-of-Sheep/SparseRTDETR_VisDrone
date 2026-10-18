"""CPU preparation only. Import/spawn never parses arguments or dispatches."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['cpu-check'])
    p.add_argument('--plan',required=True)
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    # Check paths before creating an output. No torch/CUDA import at module top.
    for s in (a.plan,a.inputs):
        q=Path(s)
        if not q.is_absolute() or not q.is_file() or q.is_symlink():raise ValueError('bad CPU input path')
    out=Path(a.output)
    if not out.is_absolute() or out.exists():raise ValueError('new absolute CPU output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden for CPU preparation')
    # Match the certified lab launcher before importing torch/vendor or spawn.
    # Children inherit this; setting it after torch imports is too late.
    os.environ['MKL_THREADING_LAYER']='GNU'
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    started=time.monotonic()
    try:
        from sparse_rtdetr.protocol_r18.slice_finetune_cpu import run
        run(a.plan,a.inputs,out)
        result={'status':'CPU_PREPARATION_CLOSED','elapsed_seconds':time.monotonic()-started,
                'GPU_started':False,'training_authorized':False,'pid':os.getpid()}
        (out/'CLOSED.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    except BaseException as e:
        (out/'STOP.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),
                       'elapsed_seconds':time.monotonic()-started},indent=2),encoding='utf-8')
        raise


if __name__=='__main__':main()
