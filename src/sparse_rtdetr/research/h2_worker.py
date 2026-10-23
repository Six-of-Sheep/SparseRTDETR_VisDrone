"""Real and synthetic input adapters to the SAME bounded H2 worker."""
import gc
import random
import time
from pathlib import Path

from ..localization.runio import require,read,publish,atomic,reference,digest,sha
from . import h2_contract as ct
from .h2_resources import GIB


def configuration(c):
    from ..protocol_r18.model import Config
    return Config() if c['profile']=='LAB_GPU_ONCE' else Config(input_hw=(128,160),physical_batch_size=1,amp_dtype='bfloat16')


def construct(c,arm):
    from .local_detail_model import build
    return build(configuration(c),arm,pretrained=c['pretrained']['path'] if c['profile']=='LAB_GPU_ONCE' else None,
                 synthetic=c['profile']=='CPU_FIXTURE_ONLY')


def training_data(c):
    if c['profile']=='CPU_FIXTURE_ONLY':return None
    from ..protocol_r18.data import BoundDataset
    d=c['data']['train_core']
    return BoundDataset(role='train_core',manifest_path=d['manifest']['path'],images_path=d['roots']['images'],
        annotations_path=d['roots']['annotations'],seed=0,root_binding=d['roots'])


def windows(c,dataset,*,completed=0,limit=2):
    import torch
    if c['profile']=='LAB_GPU_ONCE':
        from ..localization.data import training_loader
        from ..protocol_r18.data import input_record
        for images,targets,traces in training_loader(dataset,epoch=1,completed_windows=completed,window_limit=limit):
            identity=input_record(images,targets,traces);identity.pop('spawn_workers')
            yield images,targets,identity
    else:
        from ..protocol_r18.engine import output_digest
        for i in range(completed,completed+limit):
            generator=torch.Generator().manual_seed(1731+i)
            images=torch.rand(2,3,128,160,generator=generator)
            targets=[]
            for j,n in enumerate((2,0)):
                targets.append(dict(boxes=torch.tensor([[.4,.4,.15,.2],[.7,.7,.1,.12]])[:n],
                    labels=torch.tensor([1,4],dtype=torch.int64)[:n],gt_ids=[f'CONSTRUCTED/w{i}/im{j}/raw{k}' for k in range(n)],
                    ignore_union=torch.empty(0,4,dtype=torch.float64),ignore_class_boxes=torch.empty(0,4,dtype=torch.float64),ignore_class_labels=torch.empty(0,dtype=torch.int64)))
            yield images,targets,dict(fixture_only=True,window=i,sha256=output_digest(dict(images=images,targets=targets)),input_recipe_sha256=c['data']['train_core']['manifest']['sha256'])


def prepare(c,out):
    import torch
    from ..protocol_r18.engine import tensor_digest
    from . import h2_resources as res
    require(not torch.cuda.is_initialized(),'prepare initialized CUDA')
    torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    first=res.snapshot(c,entry=True);publish(out/'resource-entry.json',first)
    for role in ('train_core','development'):ct.exact(c['data'][role]['manifest'])
    if c['profile']=='LAB_GPU_ONCE':
        require(reference(c['pretrained']['path'])==c['pretrained'],'pretrained SHA/bytes differs')
        actual=dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        require(actual==c['environment'],'prepare environment differs')
    dataset=training_data(c);members=0
    if dataset is not None:
        for row in dataset.records:
            dataset.bound_bytes(row);members+=1
            if members%256==0:res.snapshot(c,entry=False,baseline=first)
        require(members==4869,'real train members not complete')
    else:members=0
    expected=[]
    for images,targets,identity in windows(c,dataset):expected.append(identity)
    require(len(expected)==2,'prepare inputs missing')
    initial={};ns=None;us=None
    for arm in ct.ARMS:
        m,criterion,info=construct(c,arm);s=m.state_dict()
        if arm=='N':ns={k:v.clone() for k,v in s.items()}
        elif arm=='U':
            for k,v in s.items():
                if 'detail_readout' in k:require(int(v.count_nonzero())==0,'detail head not zero')
                elif 'decoder.decoder.layers.2.cross_attn.sampling_offsets.' in k:
                    keep=torch.arange(192).reshape(8,12,2)[:,4:].flatten();require(torch.equal(v,ns[k][keep]),'coarse offset init differs')
                else:require(torch.equal(v,ns[k]),'unregistered U/N init difference: '+k)
            us={k:v.clone() for k,v in s.items()}
        else:
            for k,v in s.items():
                expected_value=us[k][2:] if k.startswith('decoder.dec_bbox_head.2.layers.2.') else us[k]
                require(torch.equal(v,expected_value),'unregistered D/U init difference: '+k)
        initial[arm]=dict(model_sha256=tensor_digest(s.items()),parameters=info['trainable_parameters'])
        del m,criterion,s;gc.collect()
    if c['profile']=='LAB_GPU_ONCE':require(initial['N']['model_sha256']=='40f9d83dff1acf1918855c17b1d7a000c03a6153394487a3e4b168a4c2bee995','N common ImageNet initial state differs')
    gate=dict(status='PREPARE_PASS',source_sha256=c['source']['sha256'],science=c['science'],data_sha256=digest(c['data']),
        initializations=initial,windows=expected,verified_train_files=members,development_images_read=0,
        model_forward_calls=0,CUDA_initialized=False,profile=c['profile'])
    publish(out/'PREPARE_PASS.json',gate);return gate


def run(c,arm,out,reserve_forward,reserve_update,guard_memory,complete_forward,complete_update):
    import torch,numpy as np
    from ..baseline.training_v2b import _optimizer
    from ..protocol_r18.model import CommonEMA
    from ..protocol_r18.engine import tensor_digest
    from .local_detail_engine import Engine
    from . import local_detail_checkpoint as cp
    out=Path(out);prepared=ct.exact(c['prepare_gate'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=c['profile']=='LAB_GPU_ONCE';device=torch.device('cuda:0' if gpu else 'cpu')
    if gpu:
        from .h2_cuda_identity import inspect_cuda_identity
        proof=inspect_cuda_identity(c['gpu_uuid'],out/'cuda-identity',torch_module=torch)
        require(proof['BF16_supported'],'BF16 not supported')
        require(str(torch.__version__)==c['environment']['torch'] and __import__('torchvision').__version__==c['environment']['torchvision'] and torch.version.cuda==c['environment']['cuda_build'] and torch.backends.cudnn.version()==c['environment']['cudnn'],'worker environment differs')
        torch.cuda.set_per_process_memory_fraction(18*GIB/torch.cuda.get_device_properties(0).total_memory,device=0)
        torch.cuda.reset_peak_memory_stats()
    elif torch.cuda.is_initialized():raise RuntimeError('CPU fixture initialized CUDA')
    config=configuration(c);dtype_records=[]
    def before(module,args):
        guard_memory();reserve_forward(arm)
    def after(module,args,result):
        complete_forward(arm)
        require(result['pred_logits'].dtype==torch.bfloat16,'actual logits are not BF16')
        dtype_records.append(str(result['pred_logits'].dtype));guard_memory()
    def make():
        m,criterion,info=construct(c,arm)
        initial=tensor_digest(m.state_dict().items())
        require(initial==prepared['initializations'][arm]['model_sha256'],'fresh worker initialization differs')
        m=m.to(device);criterion=criterion.to(device)
        e=Engine(m,criterion,_optimizer(m,config),CommonEMA(m),config)
        binding=dict(arm=arm,scope='gpu_H2_acceptance' if gpu else 'cpu_H2_synthetic',config=config.binding(),model=info,
            source_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],data_sha256=digest(c['data']),
            pretrained_sha256=c['pretrained']['sha256'],runtime=dict(gpu_uuid=c['gpu_uuid']))
        random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
        if gpu:torch.cuda.manual_seed_all(0)
        m.register_forward_pre_hook(before);m.register_forward_hook(after)
        require(e.completed_updates==e.ema.updates==0 and not e.optimizer.state,'not a fresh acceptance engine')
        return e,binding
    e,b=make();publish(out/'binding.json',b);rows=[];saved=None;data=training_data(c)
    fault=c.get('fixture',{}).get('fault','none')
    if fault=='worker_exception':raise RuntimeError('CONSTRUCTED_WORKER_EXCEPTION')
    def update(engine,images,targets,identity,index):
        require(identity==prepared['windows'][index],'input/GT/augmentation SHA differs')
        reserve_update(arm);r=engine.train_window(images,targets,capture=True);r['input']=identity
        complete_update(arm)
        guard_memory();return r
    for i,(images,targets,identity) in enumerate(windows(c,data)):
        r=update(e,images,targets,identity,i);rows.append(r)
        publish(out/f'update-{i+1}.json',r)
        atomic(out/'progress.json',dict(event='update',completed=i+1,monotonic=time.monotonic()))
        if i==0:saved=cp.save(e,b,out/'C1.pth')
    require(len(rows)==2,'unique acceptance updates differ')
    expected=cp.capture(e,b)
    del e;gc.collect()
    if gpu:torch.cuda.empty_cache()
    clone,new_binding=make();require(b==new_binding,'reconstruction binding differs')
    restored=cp.restore(clone,b,saved['path'],saved['sha256']);publish(out/'RESTORE.json',restored)
    n=0
    for images,targets,identity in windows(c,data,completed=1,limit=1):
        replay=update(clone,images,targets,identity,1);n+=1
    require(n==1 and {k:v for k,v in rows[1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'next input/output/DN/gradient exact replay differs')
    require(cp.exact_equal(expected,cp.capture(clone,b)),'full model/optimizer/EMA/RNG replay differs')
    require(len(dtype_records)==6,'actual BF16 forward coverage differs')
    if fault=='forward_overrun':reserve_forward(arm)
    if fault=='update_overrun':reserve_update(arm)
    peak=int(torch.cuda.max_memory_allocated()) if gpu else 0
    result=dict(status='PASS',arm=arm,actual_updates=3,unique_updates=2,sequence=[1,2,2],forward_calls=6,
        logits_dtypes=dtype_records,full_state_RNG_exact=True,same_next_result_exact=True,gradient_finite=True,
        checkpoint=saved,initial_model_sha256=prepared['initializations'][arm]['model_sha256'],
        state_usable_for_formal=False,peak_allocated_bytes=peak,
        update_seconds=[x['elapsed_seconds'] for x in rows]+[replay['elapsed_seconds']],
        fixture_only=not gpu,scientific_quality_evaluated=False)
    publish(out/'RESULT.json',result)
    del clone,expected;gc.collect()
    if gpu:torch.cuda.empty_cache()
    return result
