"""Independent full FT entry. Import/spawn has no parse, launch or CUDA effect."""
import argparse,os,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','launch','supervise','worker'])
    p.add_argument('--contract',required=True);p.add_argument('--phase');p.add_argument('--token')
    a=p.parse_args()
    os.environ['MKL_THREADING_LAYER']='GNU'
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as execution
    c=execution.load(a.contract)
    if a.action=='preflight':
        import torch,json
        if torch.cuda.is_initialized():raise RuntimeError('preflight initialized CUDA')
        print(json.dumps({'status':'PASS','scope':c['scope'],'model_constructed':False,'token_consumed':False,'CUDA_initialized':False}));return 0
    if a.action=='launch':return execution.launch(c,a.token)
    if a.action=='worker':execution.worker(c,a.phase);return 0
    execution.supervise(c,a.phase);return 0


if __name__=='__main__':raise SystemExit(main())
