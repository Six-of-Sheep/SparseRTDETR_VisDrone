"""CPU checkpoint-health regression with the frozen native anchor generator.

No detector, checkpoint, image, model forward or CUDA. Synthetic state only.
"""
import argparse,ast,copy,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from sparse_rtdetr.protocol_r18.span_B0_reuse import checkpoint_health
    torch.set_num_threads(2)
    if torch.version.cuda is not None or torch.cuda.is_initialized():raise RuntimeError('CPU-only runtime required')
    source=ROOT/'vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RTDETRTransformerv2')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_anchors'))
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    generate=ns['_generate_anchors'];checks=[]
    def check(name,ok):
        if not ok:raise RuntimeError(name)
        checks.append(name)
    def native(hw):
        return generate(SimpleNamespace(eval_spatial_size=hw,feat_strides=[8,16,32],eps=.01))
    anchors,mask=native([768,1344])
    base={'model':{'decoder.anchors':anchors,'decoder.valid_mask':mask,'weight':torch.ones(3)},
          'ema':{'module':{'decoder.anchors':anchors.clone(),'decoder.valid_mask':mask.clone(),'weight':torch.ones(3)},'updates':36480},
          'optimizer':{'state':{0:{'exp_avg':torch.zeros(3),'exp_avg_sq':torch.ones(3)}}},
          'engine':{'clock':torch.tensor(36480)},'rng':{'torch_cpu':torch.zeros(8,dtype=torch.uint8)}}
    health=checkpoint_health(base)
    check('native production canvas permits 3232 invalid +inf coordinates in each cache',
        anchors.shape==(1,21168,4) and all(x['invalid_coordinates']==3232 for x in health['native_anchor_caches'].values()))
    check('previous unconditional predicate rejects this legitimate native cache',not all(bool(x.isfinite().all()) for x in (anchors,base['model']['weight'])))
    valid=mask.expand_as(anchors);bad=~valid
    def reject(name,mutate):
        state=copy.deepcopy(base);mutate(state)
        try:checkpoint_health(state)
        except (RuntimeError,KeyError):checks.append(name+' rejected')
        else:raise RuntimeError(name+' accepted')
    for label,get in [('model',lambda s:s['model']),('EMA',lambda s:s['ema']['module'])]:
        for title,value in [('NaN',float('nan')),('negative infinity',float('-inf')),('finite invalid sentinel',0.)]:
            reject(label+' invalid anchor '+title,lambda s,get=get,value=value:get(s)['decoder.anchors'].masked_fill_(bad,value))
        for title,value in [('NaN',float('nan')),('positive infinity',float('inf')),('negative infinity',float('-inf'))]:
            reject(label+' valid anchor '+title,lambda s,get=get,value=value:get(s)['decoder.anchors'].masked_fill_(valid,value))
        reject(label+' nonboolean mask',lambda s,get=get:get(s).__setitem__('decoder.valid_mask',get(s)['decoder.valid_mask'].float()))
        reject(label+' wrong mask shape',lambda s,get=get:get(s).__setitem__('decoder.valid_mask',get(s)['decoder.valid_mask'].squeeze(-1)))
        reject(label+' inverted mask',lambda s,get=get:get(s).__setitem__('decoder.valid_mask',~get(s)['decoder.valid_mask']))
        reject(label+' missing mask',lambda s,get=get:get(s).pop('decoder.valid_mask'))
        reject(label+' wrong anchor dtype',lambda s,get=get:get(s).__setitem__('decoder.anchors',get(s)['decoder.anchors'].double()))
        reject(label+' nonfinite weight',lambda s,get=get:get(s)['weight'].fill_(float('inf')))
    reject('model/EMA finite cache mismatch',lambda s:s['ema']['module']['decoder.anchors'].masked_fill_(valid,0.))
    reject('optimizer nonfinite',lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')))
    reject('other similarly named anchors not exempt',lambda s:s['model'].__setitem__('foreign.anchors',torch.tensor(float('inf'))))
    reject('nested unrelated anchors not exempt',lambda s:s.__setitem__('extra',{'decoder.anchors':torch.tensor(float('inf'))}))
    small=copy.deepcopy(base);x,m=native([128,128])
    for c in (small['model'],small['ema']['module']):c.update({'decoder.anchors':x.clone(),'decoder.valid_mask':m.clone()})
    check('all-valid small CPU fixture cache accepted',checkpoint_health(small)['native_anchor_caches']['model']['invalid_coordinates']==0)
    check('no CUDA initialized',not torch.cuda.is_initialized())
    sources=[source,ROOT/'src/sparse_rtdetr/protocol_r18/span_B0_reuse.py',Path(__file__)]
    d=dict(status='ANCHOR_HEALTH_CPU_PASS',checks=checks,checks_run=len(checks),skips=0,CPU_only=True,
        CUDA_initialized=False,model_constructed=False,checkpoint_loaded=False,model_forward_calls=0,
        native_production_cache_health=health,sources=[dict(path=x.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sources])
    (a.output/'CPU_GATE.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k not in ('checks','sources')}))


if __name__=='__main__':main()
