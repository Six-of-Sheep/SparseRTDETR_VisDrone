"""Same raw identity, augmentation, sampler and two-window prefetch bound."""
import os,sys
import torch
from ..localization.data import RawRoleDataset,training_loader

class BoundDataset(RawRoleDataset):
    def __getitem__(self,key):
        image,target,trace=super().__getitem__(key)
        trace['spawn_observer']={'pid':os.getpid(),'main_name':sys.modules['__main__'].__name__,
            'dispatch_started':getattr(sys.modules['__main__'],'_dispatch_started',False),'CUDA_initialized':torch.cuda.is_initialized()}
        return image,target,trace

def dataset(c):
    d=c['data']['train_core'];r=d['roots']
    seed=c['seed']
    if type(seed) is not int or seed not in (0,1,2) or type(c['config']['seed']) is not int or c['config']['seed']!=seed:
        raise ValueError('training dataset seed/config differs')
    return BoundDataset(role='train_core',manifest_path=d['manifest'],images_path=r['images'],annotations_path=r['annotations'],seed=seed,root_binding=r)

def input_record(images,targets,traces):
    from ..r0.engine import tensor_digest,output_digest
    from ..localization.runio import require,digest
    scientific=[{k:v for k,v in t.items() if k!='spawn_observer'} for t in traces]
    require(len(images)==len(targets)==len(traces)==16,'logical16 input differs')
    workers=[t['spawn_observer'] for t in traces]
    require(all(w['CUDA_initialized'] is False and w['dispatch_started'] is False and w['main_name']=='__mp_main__' for w in workers),'spawn import dispatched worker or initialized CUDA')
    import hashlib,json
    for image,target,trace in zip(images,targets,scientific):
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for key in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):
            h.update(key.encode());h.update(target[key].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode())
        require(h.hexdigest()==trace['augmented_tensor_sha256'],'actual augmented tensor/GT/ignore bytes differ')
    return {'ids':[t['image_id'] for t in traces],'GT_ids':[t['gt_ids'] for t in targets],
        'augmented_sha256':[t['augmented_tensor_sha256'] for t in scientific],'metadata_sha256':digest(scientific),
        'input_tensor_sha256':tensor_digest([('pixels',images)]),'target_sha256':output_digest(targets),'spawn_workers':workers}
