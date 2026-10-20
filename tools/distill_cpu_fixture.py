"""Native R18 small CPU integration worker, called by the preparation entry.

All images, boxes and weights are synthetic. Runs real forward/backward/state
serialization for engineering only; no dataset AP or scientific training.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))


def main():
    out = Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('R18_DISTILL_CPU_INJECT_FAILURE')=='1':
        raise RuntimeError('intentional CPU fixture worker exception')
    import numpy as np
    import torch
    from sparse_rtdetr.protocol_r18 import distill as kd
    from sparse_rtdetr.protocol_r18.distill_engine import Engine, capture
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.protocol_r18.checkpoint import exact_equal
    from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
    from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs
    from sparse_rtdetr.baseline.primary_evaluator import _validate_input
    from test_multiview_distill import fixture, empty_teacher
    if torch.cuda.is_initialized() or torch.version.cuda is not None:
        raise RuntimeError('integration fixture requires CPU-only torch')
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_multiview_distill.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('required production-interface checks failed or skipped')
    cfg=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    parent,base,info=build(cfg,synthetic=True)
    initial=kd.tensor_digest(parent.state_dict().items())
    # Exercise actual frozen teacher forward and original five-view decode/merge,
    # then compact/raw SHA admission; synthetic pixels/weights, not training data.
    from sparse_rtdetr.protocol_r18 import checkpoint as native
    from sparse_rtdetr.baseline.training_v2b_checkpoint import _safe
    from sparse_rtdetr.localization.data import parse_raw
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    from test_multiview_distill import fixture
    im,raw=fixture()
    teacher=copy.deepcopy(parent).eval().requires_grad_(False)
    teacher_outputs=kd.teacher_batch(teacher,[im],input_hw=(128,128),synthetic=True)
    del teacher
    row=dict(stable_image_id=stable_image_id('train','train/images/CPU_FIXTURE.png'),
             split='train',relative_path='train/images/CPU_FIXTURE.png',height=101,width=157,
             image_sha256='a'*64,annotation_sha256='b'*64)
    cb=kd.cache_binding(source_manifest_sha256='c'*64,contract_sha256='d'*64)
    member=kd.write_cache_member(out/'synthetic-teacher',row,teacher_outputs[0],cb)
    index=out/'synthetic-teacher-index.json'
    index.write_bytes(kd.canonical(dict(closed=True,binding=cb,records=[member])))
    index_ref=dict(path=str(index),bytes=index.stat().st_size,sha256=native.file_sha(index))
    reader=kd.CacheReader(index_ref,[row],cb)
    assert reader(row)['member']==row
    try:reader(dict(row,stable_image_id='development-like-wrong-member'))
    except ValueError:pass
    else:raise RuntimeError('cross-member teacher read accepted')
    # Raw reference is actually checked at admission, not just written in metadata.
    damaged=copy.deepcopy(member);damaged['image_id']='wrong'
    bad_index=out/'bad-index.json';bad_index.write_bytes(kd.canonical(dict(closed=True,binding=cb,records=[damaged])))
    try:kd.CacheReader(dict(path=str(bad_index),bytes=bad_index.stat().st_size,sha256=native.file_sha(bad_index)),[row],cb)
    except ValueError:pass
    else:raise RuntimeError('cross-member index accepted')
    # Parent extraction uses real implementation on a deliberately SYNTHETIC
    # schema fixture; cannot satisfy load_parent's real checkpoint-byte gate.
    parent_binding=dict(scope='formal_R18_seed0_e120',config=dict(seed=0))
    pseudo_parent=dict(format=native.FORMAT,binding=parent_binding,binding_sha256=kd.digest(parent_binding),
                       engine=dict(completed_updates=36480),
                       ema=dict(module=_safe(parent.state_dict()),updates=36480,decay=.9999,warmups=2000))
    assert kd.initialize_parent(parent,pseudo_parent,parent_binding)==initial
    bad_parent=copy.deepcopy(pseudo_parent);bad_parent['ema']['updates']=36479
    try:kd.initialize_parent(parent,bad_parent,parent_binding)
    except ValueError:pass
    else:raise RuntimeError('nonterminal parent EMA accepted')
    x,t,tr=RawAugmentation((128,128))(im,raw,epoch=120,seed=0,position=0,image_id='CPU_FIXTURE')
    empty_raw=parse_raw(b'', 'CPU_EMPTY')
    xe,te,_=RawAugmentation((128,128))(im,empty_raw,epoch=120,seed=0,position=1,image_id='CPU_EMPTY')
    images=torch.stack([x,xe]);arms=[]
    for arm in kd.ARMS:
        model=copy.deepcopy(parent)
        assert initial==kd.tensor_digest(model.state_dict().items())
        criterion=kd.Criterion(copy.deepcopy(base),arm)
        opt,ema=kd.optimizer_ema(model,cfg)
        assert not opt.state and ema.updates==0
        binding=dict(scope='CPU_SYNTHETIC_ONLY',arm=arm,protocol=kd.VERSION,
                     parent_tensor_sha256=initial,plan_sha256=kd.digest(kd.read_plan(ROOT/'configs/research/r18_multiview_distill_v1.json')),
                     cache_identity='SYNTHETIC_ONLY_NOT_TRAIN_CACHE',source='this_preparation_entry_sources')
        engine=Engine(model,criterion,opt,ema,cfg,arm=arm,binding=binding)
        targets=copy.deepcopy([t,te])
        if arm!='T0':
            targets[0]['teacher']=dict(boxes=t['boxes'].clone(),labels=t['labels'].clone(),
                                      scores=torch.full((len(t['boxes']),),.75),identities=[])
            targets[1]['teacher']=empty_teacher()
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        a=engine.train_window(images,targets,capture=True)
        ref=kd.save(engine,out/(arm+'-synthetic-state.pt'))
        b=engine.train_window(images,targets,capture=True)
        expected=capture(engine)
        restore=kd.restore(engine,ref)
        replay=engine.train_window(images,targets,capture=True)
        actual=capture(engine)
        if not exact_equal(expected,actual):
            raise RuntimeError('actual native full state replay differs: '+arm)
        for key in ('output_sha256','dn_noise_sha256','microbatch_losses','gradient_sha256_before_clip','normalizers'):
            if b[key]!=replay[key]:
                raise RuntimeError('actual native replay differs '+key+': '+arm)
        # Different arm/cache/source identity cannot restore, before state load.
        changed=copy.deepcopy(engine.binding);engine.binding['cache_identity']='wrong'
        try:kd.restore(engine,ref)
        except ValueError:pass
        else:raise RuntimeError('cross-cache restore accepted')
        if not engine.failed:raise RuntimeError('restore rejection did not poison engine')
        engine.binding=changed
        Path(ref['path']).unlink() # Only own successful synthetic state; preserve compact SHA receipt.
        arms.append(dict(arm=arm,initial_sha256=initial,optimizer_initial_empty=True,
                         checkpoint={**ref,'retained':False,'reason':'synthetic CPU state only; successful exact replay'},
                         restored=restore,actual_CPU_synthetic_updates=3,
                         full_state_RNG_and_outputs_exact=True,empty_GT_and_teacher=True,
                         supervised_loss_counts=[len(v)-(arm!='T0') for v in a['microbatch_losses']],
                         source_cache_restore_rejected=True))
        del expected,actual,model,criterion,opt,ema,engine
    # Actual raw parser -> BOTH production evaluator input adapters, not AP.
    rows=[dict(stable_image_id='CPU_FIXTURE',width=157,height=101),dict(stable_image_id='CPU_EMPTY',width=157,height=101)]
    inputs=primary_inputs(rows,[raw,empty_raw],[]);_validate_input(inputs)
    coco,pred,counts=bridge_inputs(rows,[raw,empty_raw],[])
    assert len(coco['images'])==2 and counts['detections_before']==counts['detections_after']==0
    value=dict(status='CPU_PASS',arms=list(kd.ARMS),checks_run=result.testsRun,skips=0,
               actual_synthetic_teacher_forward_views=5,teacher_raw_compact_rebuild=True,
               teacher_member_SHA_admission_and_cross_member_rejection=True,
               parent_EMA_extraction_synthetic_schema_only=True,
               native_arms=arms,CPU_synthetic_updates_total=9,real_evaluator_input_adapters=True,
               scientific_AP_calls=0,scientific_updates=0,CUDA_initialized=torch.cuda.is_initialized(),GPU_calls=0,
               actual_entry='r18_distill_prepare.py cpu-check -> this worker -> native R18 -> state -> input adapters -> exit',
               data_and_checkpoint_scope='synthetic only; actual train membership checked separately by prepare',
               torch_version=str(torch.__version__),input_hw=[128,128],physical_batch=1)
    (out/'RESULT.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'CLOSED.json').write_text(json.dumps(dict(status='CPU_PASS',result_sha256=hashlib.sha256((out/'RESULT.json').read_bytes()).hexdigest()))+'\n',encoding='utf-8')
    print(json.dumps(dict(status=value['status'],checks_run=result.testsRun,CPU_synthetic_updates=9)))


if __name__=='__main__':main()
