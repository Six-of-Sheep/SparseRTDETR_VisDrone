"""Actual preparation entry consumers: real spawn data, native CPU state replay.

No production model forward; native detector CPU replay uses a labeled 128px
synthetic fixture, separate from the real parent/no-forward and input checks.
"""
import copy
import hashlib
import json
import os
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from . import slice_finetune as ft
from .checkpoint import exact_equal, read as read_native
from .data import BoundDataset
from .engine import output_digest, tensor_digest
from .model import Config, build
from ..localization.data import training_loader,collate_window


def seed():
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)


def ref(path):
    p=Path(path)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(r):
    if ref(r['path'])!=r:raise ValueError('CPU input reference changed: '+str(r['path']))
    return json.loads(Path(r['path']).read_text())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


class Fixture(TorchDataset):
    """Uses exactly the production Transform and real spawn import, not a mock."""
    def __init__(self,arm):self.arm=arm
    def __len__(self):return 4
    def __getitem__(self,index):
        image=__import__('PIL.Image',fromlist=['Image']).fromarray(np.full((128,128,3),70+index,np.uint8))
        # Mix nonempty, ignore and empty targets without changing loss code.
        data=b'30,30,20,25,1,1,0,0\n90,20,20,20,0,0,0,0\n' if index%2==0 else b''
        raw=ft.parse_raw(data,'cpu-fixture-'+str(index))
        x,t,tr=ft.Transform(self.arm,(128,128))(image,raw,epoch=12,position=index,image_id='cpu-fixture-'+str(index))
        tr['spawn']={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()}
        return x,t,tr


def fixture_loader(arm,start=0):
    return DataLoader(Fixture(arm),batch_sampler=[[0,1],[2,3]][start:],num_workers=2,prefetch_factor=2,
                      multiprocessing_context='spawn',collate_fn=collate_window,
                      generator=torch.Generator().manual_seed(0),timeout=120)


def parent_initialization(inputs,plan,out):
    b=checked(inputs['parent_binding'])
    r=inputs['parent_checkpoint']
    if r['sha256']!=ft.PARENT_SHA or Path(r['path']).stat().st_size!=r['bytes']:
        raise ValueError('parent checkpoint reference/size differs')
    p=read_native(r['path'],r['sha256'],b)  # one necessary full SHA/read; never download
    model,criterion,info=build(Config(),synthetic=True)
    a=ft.initialize_from_parent(model,p,expected_binding=b)
    opt,ema=ft.make_optimizer_ema(model,Config())
    digest_a=tensor_digest(model.state_dict().items())
    digest_ema=tensor_digest(ema.module.state_dict().items())
    model2,_,_=build(Config(),synthetic=True)
    z=ft.initialize_from_parent(model2,p,expected_binding=b)
    opt2,ema2=ft.make_optimizer_ema(model2,Config())
    assert a==z==digest_a==digest_ema==tensor_digest(ema2.module.state_dict().items())
    assert not opt.state and not opt2.state and ema.updates==ema2.updates==0
    rejected=[]
    for key,value in [('engine',dict(p['engine'],completed_updates=36479)),('binding_sha256','0'*64)]:
        bad=dict(p);bad[key]=value
        try:ft.initialize_from_parent(model2,bad,expected_binding=b)
        except ValueError:rejected.append(key)
        else:raise AssertionError('invalid parent accepted')
    result={'status':'PASS','parent_checkpoint':r,'parent_binding':inputs['parent_binding'],
            'parent_source':inputs['parent_source_sha256'],'selected':'EMA.module only',
            'model_and_BN_bytes_both_arms_exact':True,'initial_model_sha256':a,
            'new_optimizer_empty':True,'new_EMA_updates':0,'old_RNG_not_restored':True,
            'model_forwards':0,'rejections':rejected}
    write(out/'PARENT_INITIALIZATION.json',result)
    del p,model,model2,opt,opt2,ema,ema2,criterion
    return ref(out/'PARENT_INITIALIZATION.json')


def real_data(inputs,out):
    d=inputs['train_core'];r=d['roots']
    args=dict(manifest_path=d['manifest'],images_path=r['images'],annotations_path=r['annotations'],root_binding=r)
    native=BoundDataset(role='train_core',seed=0,**args)
    rows=[]
    for arm in ft.ARMS:
        data=ft.Dataset(arm,**args)
        # Full frozen windows 1,2 and explicit bounded prefetch: no third window.
        count=0
        for batch,(x,t,tr) in enumerate(training_loader(data,epoch=109,window_limit=2)):
            assert len(x)==16 and not torch.cuda.is_initialized()
            assert all(not v['spawn_observer']['CUDA_initialized'] and v['spawn_observer']['pid']!=os.getpid() for v in tr)
            views=[v['fixed_view']['view_id'] for v in tr]
            assert [views.count(i) for i in range(5)]==([16,0,0,0,0] if arm=='T-control' else [8,2,2,2,2])
            if arm=='T-control':
                # Native sampler's exact index, transform and bytes; worker PIDs excluded.
                from ..baseline.training_v2b import logical_batch_indices
                indices=list(logical_batch_indices(4869,seed=0,epoch=109,logical_batch_size=16,completed_batches=batch))[0]
                for i,index in enumerate(indices):
                    ox,ot,otr=native[(109,batch*16+i,index)]
                    assert torch.equal(ox,x[i]) and exact_equal(ot,t[i])
                    for k in ('gt_ids','ignore_ids','augmented_tensor_sha256','geometry_steps'):
                        assert tr[i][k]==otr[k]
            rows.append({'arm':arm,'window':batch,'image_ids':[v['image_id'] for v in tr],
                         'GT_ids':[v['gt_ids'] for v in t],'view_ids':views,
                         'input_sha256':tensor_digest([('images',x)]),'target_sha256':output_digest(t),
                         'augmentation_sha256':[v['augmented_tensor_sha256'] for v in tr],
                         'worker_pids':sorted({v['spawn_observer']['pid'] for v in tr})})
            count+=1
        assert count==2
    assert rows[0]['image_ids']==rows[2]['image_ids'] and rows[1]['image_ids']==rows[3]['image_ids']
    write(out/'REAL_DATA_CPU.json',{'status':'PASS','source_epoch':109,'logical_windows_per_arm':2,
          'exact_T-control_parity_views':32,'workers':2,'prefetch':2,
          'prefetch_range':'only windows0..1; finite sampler excludes window2+',
          'rows':rows,'CUDA_initialized':False})
    return ref(out/'REAL_DATA_CPU.json')


def native_replay(plan,out):
    records=[]
    config=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    for arm in ft.ARMS:
        def construct():
            m,c,_=build(config,synthetic=True)
            opt,ema=ft.make_optimizer_ema(m,config)
            e=ft.FineTuneEngine(m,c,opt,ema,config,arm=arm,plan_sha256=ft.digest(plan),parent_sha256='a'*64)
            seed();return e
        e=construct();results=[];inputs=[]
        for i,(x,t,tr) in enumerate(fixture_loader(arm)):
            assert all(not r['spawn']['CUDA_initialized'] for r in tr)
            inp=output_digest({'images':x,'targets':t})
            r=e.train_window(x,t,capture=True);r['input_sha256']=inp;results.append(r);inputs.append(inp)
            if i==0:cp=ft.save(e,out/(arm+'-CPU1.pth'))
        assert len(results)==2 and e.completed_updates==e.ema.updates==2
        assert len(results[0]['microbatch_losses'][0])==21 and len(results[0]['microbatch_losses'][1])==12
        expected=copy.deepcopy(ft.capture(e));state2=ft.save(e,out/(arm+'-CPU2.pth'))
        clone=construct();restored=ft.restore(clone,cp)
        for x,t,tr in fixture_loader(arm,start=1):
            replay=clone.train_window(x,t,capture=True)
            replay['input_sha256']=output_digest({'images':x,'targets':t})
        assert {k:v for k,v in results[1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'}
        assert exact_equal(expected,ft.capture(clone))
        rejects=[]
        for label,reference,other in [('cross_arm',cp,ft.ARMS[1-ft.ARMS.index(arm)]),('bad_SHA',dict(cp,sha256='0'*64),arm)]:
            fresh=construct();fresh.arm=other
            try:ft.restore(fresh,reference)
            except (ValueError,RuntimeError):rejects.append(label)
            else:raise AssertionError('bad recovery accepted')
            assert fresh.failed
        fresh=construct();fresh.plan_sha256='b'*64
        try:ft.restore(fresh,cp)
        except ValueError:rejects.append('wrong_plan')
        else:raise AssertionError('wrong plan restored')
        records.append({'arm':arm,'actual_updates':3,'sequence':[1,2,2],
                        'input_DN_outputs_loss_preclip_gradient_exact':True,'full_state_RNG_exact':True,
                        'restore':restored,'checkpoints':[cp,state2],'rejections':rejects,
                        'loss_names':sorted(results[0]['microbatch_losses'][0]),
                        'loss_count_with_DN':len(results[0]['microbatch_losses'][0]),
                        'loss_count_empty_microbatch':len(results[0]['microbatch_losses'][1])})
        del e,clone,fresh,expected
    write(out/'CPU_NATIVE_RECOVERY.json',{'status':'PASS','scope':'native R18 at 128x128 FP32 synthetic; not GPU/production acceptance',
          'real_parent_used':False,'fixture_parent_sha256':'a'*64,'rows':records,'CUDA_initialized':False})
    return ref(out/'CPU_NATIVE_RECOVERY.json')


def run(plan_path,inputs_path,out):
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU-only environment missing or CUDA already initialized')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=ft.read_plan(plan_path);inputs=json.loads(Path(inputs_path).read_text())
    if inputs['plan_sha256']!=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest():
        raise ValueError('CPU plan reference differs')
    checked(inputs['parent_binding'])
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[3]/'tests'),pattern='test_slice_finetune.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('fixed-rules CPU checks failed')
    gates={'unit_tests':result.testsRun}
    gates['parent_initialization']=parent_initialization(inputs,plan,out)
    gates['real_data']=real_data(inputs,out)
    gates['native_CPU_restore']=native_replay(plan,out)
    if torch.cuda.is_initialized():raise RuntimeError('CPU check initialized CUDA')
    write(out/'CPU_GATE.json',{'status':'PASS','plan_sha256':inputs['plan_sha256'],
          'inputs_sha256':hashlib.sha256(Path(inputs_path).read_bytes()).hexdigest(),
          'scope':'fixed data/parent/native-CPU recovery only; GPU gate and formal launch NOT_IMPLEMENTED_NOT_AUTHORIZED',
          'detector_training_on_real_data':False,'GPU_calls':0,'CUDA_initialized':False,'gates':gates})
