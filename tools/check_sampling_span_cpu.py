"""Run the isolated sampling-span prototype; never loads data or starts a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('run this CPU-only prototype with CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CUDA was already initialized')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sampling_span.py')
    with (args.output/'unittest.txt').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    # Bind every project/vendor Python file actually imported by this process.
    sources={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).resolve().is_relative_to(ROOT):
            path=Path(filename).resolve()
            if path.suffix=='.py':sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources['tools/check_sampling_span_cpu.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value=dict(status='CPU_PROTOTYPE_PASS' if result.wasSuccessful() and result.testsRun==13 and not result.skipped else 'FAIL',
               scope='synthetic CPU leaf forward/autograd and common model construction; no GPU/data/AP/production entry',
               tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
               torch=torch.__version__,python=sys.version,CUDA_initialized=torch.cuda.is_initialized(),sources=sources)
    if value['CUDA_initialized']:value['status']='FAIL'
    (args.output/'CPU_REPORT.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='sources'}))
    if value['status']!='CPU_PROTOTYPE_PASS':
        print((args.output/'unittest.txt').read_text(encoding='utf-8'))
    return 0 if value['status']=='CPU_PROTOTYPE_PASS' else 1


if __name__=='__main__':raise SystemExit(main())
