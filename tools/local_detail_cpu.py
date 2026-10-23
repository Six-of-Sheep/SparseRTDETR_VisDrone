"""Only constructed CPU cases, no model/data/optimizer or GPU mode."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from sparse_rtdetr.research.local_detail_center import LocalDetailCenter,local_samples,odd_even


def cases():
    torch.set_num_threads(1)
    dtype=torch.float64
    y,x=torch.meshgrid((torch.arange(33,dtype=dtype)+.5)/33,
                       (torch.arange(65,dtype=dtype)+.5)/65,indexing='ij')
    ref=torch.tensor([[[.5,.5,.4,.2]]],dtype=dtype)
    ctx=torch.tensor([[[.4,-.4,.1,-.1]]],dtype=dtype)
    leaf=LocalDetailCenter(1).double()
    with torch.no_grad():leaf.readout.weight.copy_(torch.eye(2,dtype=dtype))
    response=[]
    with torch.no_grad():
        for k in range(-8,9):
            offset=k*.01
            field=torch.exp(-((x-(.5+offset))**2+(y-.5)**2)/(2*.05**2))[None,None]
            v=leaf(field,ref,ctx)
            shifted=leaf(field+3,ref,ctx)
            response.append(dict(constructed_target_offset_x=offset,odd_x=float(v['odd'][0,0,0]),
                even_x=float(v['even'][0,0,0]),center_logit_delta_x=float(v['detail_delta'][0,0,0]),
                added_common_mode_change=float((v['detail_delta']-shifted['detail_delta']).abs().max())))
        flat=leaf(torch.ones(1,1,33,65,dtype=dtype),ref,ctx)
        flat_control=leaf(torch.ones(1,1,33,65,dtype=dtype),ref,ctx,additive_control=True)
        edge_ref=ref.clone();edge_ref[...,0]=.01
        edge=leaf(torch.ones(1,1,33,65,dtype=dtype),edge_ref,ctx)
        ramp=leaf(x[None,None],ref,ctx)
    return dict(label='CONSTRUCTED_SYNTHETIC_CPU_NOT_MODEL_OBSERVATIONS',hand_set_readout=True,
        trained=False,feature_hw=[33,65],reference=ref.tolist(),response=response,
        counterexamples=dict(flat_wrong_center=dict(predicted_center=flat['box'][0,0,:2].tolist(),
            constructed_GT_center=[.6,.5],cannot_correct=True),
            additive_control_on_flat=dict(center=flat_control['box'][0,0,:2].tolist()),
            zero_padding_constant_field=dict(reference=edge_ref.tolist(),odd=edge['odd'].tolist(),
                common_mode_invariance_does_not_hold=True),
            background_ramp_without_object=dict(odd=ramp['odd'].tolist(),
                nonzero_detail_is_not_object_evidence=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output');g.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU check must not initialize CUDA')
    spec=importlib.util.spec_from_file_location('local_detail_tests',ROOT/'tests/test_local_detail_center.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    print(stream.getvalue(),end='')
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('CPU checks failed/skipped')
    measurements=cases()
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized during synthetic checks')
    report=dict(status='PASS',tests=result.testsRun,skips=len(result.skipped),python=platform.python_version(),
        torch=str(torch.__version__),CUDA_initialized=False,GPU_forward_calls=0,detector_loaded=False,
        checkpoint_loaded=False,real_images_or_annotations_read=False,optimizer_updates=0,
        original_P_reopened=False,hypotheses_considered=2,hypotheses_retained=['H2_TASK_ROUTING_ONLY'],
        scientific_benefit='UNKNOWN',operator_originality='NO: existing PDC specialization',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if a.output:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        for name,value in [('CPU_REPORT.json',report),('CPU_CASES.json',measurements)]:
            (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8')
        names=['src/sparse_rtdetr/research/local_detail_center.py','tests/test_local_detail_center.py',
               'tools/local_detail_cpu.py','src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py']
        source=[dict(path=n,sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),bytes=(ROOT/n).stat().st_size) for n in names]
        (out/'CPU_SOURCE.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
