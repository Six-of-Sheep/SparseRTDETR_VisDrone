"""Targeted seed plumbing checks; no detector construction or CUDA call.

These tests supplement, and do not replace, the real spawn/two-window and
complete checkpoint recovery checks in the independent execution CPU gate.
"""
from dataclasses import replace
from types import SimpleNamespace
import json
import torch


class _IndexDataset:
    role = 'train_core'

    def __init__(self, seed):
        self.seed = seed
        self.records = [{'stable_image_id': str(i)} for i in range(4869)]

    def __len__(self): return len(self.records)

    def __getitem__(self, key):
        epoch, position, index = key
        return torch.tensor([index]), {'index': index}, {'epoch': epoch, 'position': position, 'index': index}


def run():
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.protocol_r18 import data as native_data, checkpoint
    from sparse_rtdetr.localization.data import RawRoleDataset, training_loader
    from sparse_rtdetr.localization.worker import plan_epoch
    from sparse_rtdetr.baseline.training_v2b import logical_batch_indices

    if torch.cuda.is_initialized(): raise RuntimeError('seed CPU tests require an uninitialized CUDA runtime')
    checks = []

    def check(label, condition):
        if not condition: raise AssertionError(label)
        checks.append(label)

    def rejected(label, fn, message=None):
        try: fn()
        except (ValueError, RuntimeError, KeyError) as error:
            if message is not None and message not in str(error): raise
            checks.append(label)
        else: raise AssertionError(label + ': unexpectedly accepted')

    baseline = Config().binding()
    nonseed_keys = set(baseline) - {'seed', 'initializer', 'training_rng'}
    for seed in (0,1,2):
        conf = Config(seed=seed);conf.validate()
        check(f'seed{seed}:production recipe unchanged', all(conf.binding()[k] == baseline[k] for k in nonseed_keys))
        check(f'seed{seed}:initialization label', f'CPU_seed{seed}_then' in conf.binding()['initializer'])
        check(f'seed{seed}:training RNG label', f'CUDA_seed{seed}_after' in conf.binding()['training_rng'])
        Config(seed=seed,input_hw=(128,128),physical_batch_size=1,amp_dtype='float32').validate(synthetic=True)
        checks.append(f'seed{seed}:CPU fixture permitted')
        for attr, value in [('learning_rate',2e-4),('num_denoising',101),('epochs',121),('sampling_backend','grid_sample')]:
            rejected(f'seed{seed}:recipe change rejected:{attr}', lambda attr=attr,value=value: replace(conf,**{attr:value}).validate())
        rejected(f'seed{seed}:production fixture rejected', lambda: replace(conf,input_hw=(128,128)).validate())
    for value in (-1,3,True,False,1.0,'1',None):
        rejected(f'invalid config seed:{value!r}', lambda value=value: Config(seed=value).validate())
        rejected(f'invalid dataset seed:{value!r}', lambda value=value: RawRoleDataset(role='train_core',manifest_path='unused',images_path='unused',annotations_path='unused',seed=value), 'unauthorized role or seed')

    orders = {}
    before_rng = torch.get_rng_state().clone()
    for seed in (0,1,2):
        data = _IndexDataset(seed)
        for epoch in (1,2,120):
            plan = plan_epoch(data,epoch)
            batches = list(logical_batch_indices(len(data),seed=seed,epoch=epoch,logical_batch_size=16))
            check(f'seed{seed}:epoch{epoch}:plan equals actual sampler', plan['order'][:4864] == [i for batch in batches for i in batch])
            check(f'seed{seed}:epoch{epoch}:five dropped retained', len(plan['dropped_indices']) == 5 and plan['dropped_indices'] == plan['order'][4864:])
            loader = training_loader(data,epoch=epoch,workers=0,window_limit=2)
            check(f'seed{seed}:epoch{epoch}:dedicated loader RNG', loader.generator.initial_seed() == seed and loader.batch_sampler.seed == seed)
            got = list(loader)
            check(f'seed{seed}:epoch{epoch}:two loader windows', len(got) == 2 and [v['index'] for v in got[1][1]] == list(batches[1]))
            resumed = list(training_loader(data,epoch=epoch,completed_windows=1,workers=0,window_limit=1))
            check(f'seed{seed}:epoch{epoch}:restored data cursor', len(resumed) == 1 and torch.equal(got[1][0],resumed[0][0]) and got[1][1:] == resumed[0][1:])
            if epoch == 1:orders[seed] = plan['order']
        fake_engine = SimpleNamespace(config=Config(seed=seed))
        for other in (0,1,2):
            if other != seed:
                rejected(f'seed{seed}:foreign checkpoint scope{other}', lambda other=other: checkpoint.capture(fake_engine,{'scope':f'formal_R18_seed{other}_e120'}), 'unknown common R18 state scope')
    check('three seed orders distinct', all(orders[a] != orders[b] for a,b in ((0,1),(0,2),(1,2))))
    check('sampler and loader do not consume model CPU RNG', torch.equal(before_rng,torch.get_rng_state()))

    # Verify the actual native dataset adapter forwards the explicit seed and
    # rejects contradictory metadata before any data reads. Real image/GT bytes
    # and spawn behavior are checked separately by the execution entry gate.
    original = native_data.BoundDataset
    try:
        native_data.BoundDataset = lambda **kwargs: kwargs
        for seed in (0,1,2):
            c = {'seed':seed,'config':Config(seed=seed).binding(),'data':{'train_core':{'manifest':'unused','roots':{'images':'unused','annotations':'unused'}}}}
            check(f'seed{seed}:contract reaches raw dataset',native_data.dataset(c)['seed'] == seed)
            wrong = {**c,'config':{**c['config'],'seed':(seed+1)%3}}
            rejected(f'seed{seed}:contradictory contract seed rejected',lambda wrong=wrong:native_data.dataset(wrong),'training dataset seed/config differs')
            if seed == 1:
                wrong_bool = {**c,'config':{**c['config'],'seed':True}}
                rejected('boolean config seed rejected',lambda:native_data.dataset(wrong_bool),'training dataset seed/config differs')
            missing = {k:v for k,v in c.items() if k != 'seed'}
            rejected(f'seed{seed}:missing explicit contract seed rejected',lambda missing=missing:native_data.dataset(missing))
    finally:
        native_data.BoundDataset = original
    check('CUDA remains uninitialized',not torch.cuda.is_initialized())
    return {'status':'PASS','checks':len(checks),'labels':checks,'CUDA_initialized':False,'detector_constructed':False,'actual_spawn_checked_here':False,'model_updates':0}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
