"""Full H2 synthetic CPU acceptance. No GPU/real-input/launch mode."""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output');g.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CUDA already initialized')
    out=Path(args.output) if args.output else None
    if out:out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    spec=importlib.util.spec_from_file_location('h2_full_tests',ROOT/'tests/test_local_detail_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO()
    forwards=[]
    original=torch.nn.Module._call_impl
    def counted(model,*a,**kw):
        if type(model).__name__=='RTDETR':forwards.append(dict(training=model.training,arm=getattr(model,'h2_arm','native_reference')))
        return original(model,*a,**kw)
    with patch.object(torch.nn.Module,'_call_impl',counted):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    print(stream.getvalue(),end='')
    report=dict(schema='H2-full-model-CPU-v1',status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,
        python=platform.python_version(),torch=str(torch.__version__),CUDA_initialized=torch.cuda.is_initialized(),
        GPU_forward_calls=0,real_images_or_annotations_read=False,old_checkpoint_loaded=False,
        fixture='CONSTRUCTED_128x160_PHYSICAL1_ACCUM2_NOT_PRODUCTION_DATA_OR_COST',
        full_state_replay_matrix=['N/FP32','U/FP32','D/FP32','N/CPU-BF16','U/CPU-BF16','D/CPU-BF16'],
        full_state_CPU_actual_updates=18 if result.wasSuccessful() else 'INCOMPLETE',
        full_state_CPU_training_batch_forwards=36 if result.wasSuccessful() else 'INCOMPLETE',
        actual_CPU_detector_batch_forwards=len(forwards),CPU_forward_inventory=forwards,
        scientific_benefit='UNKNOWN',production_GPU_BF16='NOT_TESTED')
    files=['src/sparse_rtdetr/research/local_detail_model.py','src/sparse_rtdetr/research/local_detail_engine.py',
           'src/sparse_rtdetr/research/local_detail_checkpoint.py','src/sparse_rtdetr/research/local_detail_reference.py','tests/test_local_detail_model.py',
           'tools/local_detail_model_cpu.py']
    source=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in files]
    if out:
        for name,value in [('CPU_REPORT.json',report),('CPU_SOURCE.json',source)]:
            (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8')
    print(json.dumps(report))
    if report['status']!='PASS' or report['CUDA_initialized']:raise SystemExit(1)


if __name__=='__main__':main()
