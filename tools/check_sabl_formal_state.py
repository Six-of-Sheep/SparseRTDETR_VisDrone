"""Constructed CPU native-304 boundary/replay and local augmentation checks."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))


def main(base):
    import torch
    from PIL import Image
    from sparse_rtdetr.research import sabl_formal_state as state, sabl_state as short, sabl_formal_data as data
    from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
    from sparse_rtdetr.localization import data as raw
    import sabl_formal as cli
    base=Path(base);base.mkdir();began=time.monotonic();checks=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        (base/'CHECKS.json').write_text(json.dumps(checks))
        if not ok: raise AssertionError(name)
    fixture=data.Fixture(); images,targets,_=fixture.window(1)
    from sparse_rtdetr.research import sabl_formal_contract as ct
    c={'mode':ct.CPU,'manifest_sha256':'SYNTHETIC_NOT_SCIENTIFIC'}
    for arm in ct.ARMS:
        e,b,_,_=cli.fresh(c,arm,'cpu')
        # One real CPU update constructs all optimizer entries. The following
        # clock injection is a boundary fixture, not 303 performed updates.
        e.train_window(images,targets,capture=True)
        e.completed_updates=303;e.ema.updates=303
        for n,p in e.model.named_parameters():
            if p in e.optimizer.state and 'step' in e.optimizer.state[p]:
                e.optimizer.state[p]['step'].fill_(303);e.parameter_steps[n]=303
        for group in e.optimizer.param_groups: group['lr']=e.rate(303)
        e.train_window(images,targets,capture=True)
        check(arm+'_native_epoch_boundary',e.state_dict()['completed_epochs']==1 and e.state_dict()['next_window']==0)
        cp=state.save(e,b,base/(arm+'-304.pt'))
        direct=e.train_window(images,targets,capture=True); expected=state.capture(e,b)
        state.restore(e,b,cp['path'],cp['sha256']); replay=e.train_window(images,targets,capture=True)
        check(arm+'_305_exact_full_state_replay',_equal(expected,state.capture(e,b)) and all(direct[k]==replay[k] for k in ('microbatch_losses','gradient_sha256_before_clip','output_sha256','dn_noise_sha256')))
        check(arm+'_native_next_epoch_cursor',e.state_dict()['next_epoch']==2 and e.state_dict()['next_window']==1)
        short_path=base/(arm+'-forbidden-short.pt')
        forbidden=state.capture(e,b);forbidden['format']=short.FORMAT
        torch.save(forbidden,short_path)
        try:
            state.restore(e,b,short_path,hashlib.sha256(short_path.read_bytes()).hexdigest())
            refused=False
        except Exception: refused=True
        check(arm+'_formal_rejects_short_acceptance_format',refused)
        del forbidden
        # A failed restore seals the engine; create a fresh one for the reverse refusal.
        del e;gc.collect();e,_,_,_=cli.fresh(c,arm,'cpu')
        try:
            short.restore(e,dict(b,scope='CPU_SYNTHETIC_SABL'),cp['path'],cp['sha256'])
            refused=False
        except Exception: refused=True
        check(arm+'_short_scope_rejects_formal_state',refused)
        refs=[]
        for p in (Path(cp['path']),short_path):
            assert p.resolve().parent==base.resolve()
            refs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'synthetic_tensor_removed':True})
            p.unlink()
        (base/(arm+'-SYNTHETIC_TENSOR_INDEX.json')).write_text(json.dumps(refs))
        del e,expected;gc.collect()
    # Execute only the original augmentation source with the single registered
    # cutoff substitution as an independent oracle. Same sample RNG/local epoch.
    source=inspect.getsource(raw.RawAugmentation).replace('epoch < 118','epoch < 22')
    namespace=dict(vars(raw));exec(compile(source,'<constructed_cutoff22_oracle>','exec'),namespace)
    oracle=namespace['RawAugmentation']((128,128)); actual=data.LocalAugmentation((128,128))
    for epoch in (1,21,22,24):
        for position in range(3):
            row=fixture.records[0];gt=fixture.raw(row);kw=dict(epoch=epoch,seed=0,position=position,image_id=row['stable_image_id'],training=True)
            im=Image.new('RGB',(160,128),(80,90,110));a=actual(im.copy(),gt,**kw);b=oracle(im.copy(),gt,**kw);im.close()
            check('augmentation_e%d_p%d_bytes_RNG'%(epoch,position),torch.equal(a[0],b[0]) and _equal(a[1],b[1]) and a[2]['sample_seed']==b[2]['sample_seed'] and a[2]['geometry_steps']==b[2]['geometry_steps'] and a[2]['augmented_tensor_sha256']==b[2]['augmented_tensor_sha256'])
    check('CPU_no_CUDA',not torch.cuda.is_initialized())
    r={'status':'PASS','scope':'CONSTRUCTED_CPU_NOT_SCIENTIFIC','checks':checks,'seconds':time.monotonic()-began,
       'CPU_actual_updates':12,'CPU_detector_forward':24,'constructed_clock_injection':303,'clock_is_not_work_performed':True,'real_GPU_cost':0}
    (base/'ACCEPTANCE.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))


if __name__=='__main__':main(sys.argv[1])
