"""Native pinned architecture. Only explicit common-protocol adaptations."""
from dataclasses import asdict, dataclass
import copy, contextlib, importlib
from pathlib import Path
import torch
from ..baseline import training_v2b as vendor
from ..r0.model import R0EMA as CommonEMA

COMMIT='1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47'
ARM='RT-DETRv2-R18-common-protocol-v1'
PRETRAINED_SHA='911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720'
ROOT=Path(__file__).resolve().parents[3]

@dataclass(frozen=True)
class Config:
    seed:int=0
    input_hw:tuple=(768,1344)
    physical_batch_size:int=8
    accumulation_steps:int=2
    epochs:int=120
    updates_per_epoch:int=304
    amp_dtype:str='bfloat16'
    num_denoising:int=100
    sampling_backend:str='deterministic_gather'
    learning_rate:float=1e-4
    weight_decay:float=1e-4
    clip_max_norm:float=.1
    warmup_steps:int=2000
    ema_decay:float=.9999
    ema_warmups:int=2000
    def validate(self,*,synthetic=False):
        if type(self.seed) is not int or self.seed not in (0,1,2):
            raise ValueError('common protocol seed must be fixed 0, 1 or 2')
        fixed=Config(seed=self.seed)
        for key in asdict(fixed):
            if key not in ('input_hw','physical_batch_size','amp_dtype') and getattr(self,key)!=getattr(fixed,key):
                raise ValueError('common protocol recipe differs: '+key)
        if not synthetic and self!=fixed: raise ValueError('production recipe differs')
        if synthetic and (min(self.input_hw)<128 or any(x%32 for x in self.input_hw) or self.physical_batch_size<1 or self.amp_dtype not in ('float32','bfloat16')):
            raise ValueError('invalid CPU fixture')
    def binding(self):
        return {**asdict(self),'input_hw':list(self.input_hw),'arm':ARM,'official_commit':COMMIT,
                'representation':'native_iterative_sigmoid_cxcywh_v1','matcher':'own_head_Hungarian',
                'pre_head':False,'feature_history_add':False,'GO':False,'losses':['vfl','bbox','giou'],
                'weights':{'loss_vfl':1.,'loss_bbox':5.,'loss_giou':2.},'loss_count_with_DN':21,
                'loss_count_without_DN':12,'scheduler':'update_warmup2000_cosine_to_1e-6',
                'augmentation_stop_logged_epoch':118,'num_workers':2,'prefetch_factor':2,
                'position_embedding':'unchanged_native_width_first_meshgrid',
                'reference_detach':'unchanged_native_attention_detach_and_aux_gradient_path',
                'initializer':f'unchanged_native_heads_CPU_seed{self.seed}_then_strict_ImageNet_backbone',
                'training_rng':f'reset_python_numpy_torchCPU_CUDA_seed{self.seed}_after_independent_initialization',
                'raw_ignore':'normal_heads_unmatched_negative_class_only',
                'loss_dtype':'FP32_from_native_outputs_without_detach','endpoint':'e120_EMA'}

def resolved_config(config):
    config.validate(synthetic=config!=Config(seed=config.seed))
    with vendor._vendor_path(vendor._vendor_root(ROOT)):
        from src.core.yaml_utils import load_config
        original=copy.deepcopy(load_config(str(ROOT/'vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'),cfg={}))
    effective=copy.deepcopy(original)
    effective.update(num_classes=10,remap_mscoco_category=False,eval_spatial_size=list(config.input_hw),
                     device='cpu',use_amp=config.amp_dtype=='bfloat16',epoches=120)
    effective['PResNet']['pretrained']=False # offline strict load; no download
    effective['optimizer']={'type':'AdamW','lr':1e-4,'betas':[.9,.999],'weight_decay':1e-4,
        'params':[{'params':'^(?=.*(?:norm|bn)).*$','weight_decay':0.}]}
    effective['lr_scheduler']={'type':'CommonUpdateCosine','warmup_updates':2000,'total_updates':36480,'minimum_lr':1e-6}
    effective['lr_warmup_scheduler']={'type':'LinearWarmup','warmup_duration':2000}
    effective['ema'].update(decay=.9999,warmups=2000)
    effective.pop('scaler',None)
    for name,batch in (('train_dataloader',8),('val_dataloader',4)):
        loader=effective[name];loader.pop('batch_size',None)
        loader.update(total_batch_size=batch,num_workers=2,drop_last=name=='train_dataloader')
        loader['dataset']['img_folder']=None;loader['dataset']['ann_file']=None
        for op in loader['dataset']['transforms']['ops']:
            if op['type']=='Resize':op['size']=list(config.input_hw)
        if name=='train_dataloader':
            loader['dataset']['transforms']['policy']['epoch']=117
            loader['collate_fn'].update(scales=None,stop_epoch=117)
    return original,effective

def native_objects(config):
    original,effective=resolved_config(config)
    vendor._package_source_identities(ROOT);vendor._vendor_source_identities(ROOT)
    with vendor._vendor_path(vendor._vendor_root(ROOT)):
        for name in ('src.solver','src.data','src.nn','src.zoo.rtdetr'):importlib.import_module(name)
        from src.core import GLOBAL_CONFIG
        from src.core.workspace import create
        from src.core.yaml_utils import merge_config
        saved=vendor._snapshot_registry(GLOBAL_CONFIG)
        try:
            merged=merge_config(effective,vendor._snapshot_registry(GLOBAL_CONFIG),inplace=False,overwrite=False)
            with torch.random.fork_rng(devices=[]):
                vendor.seed_cpu_sources(config.seed)
                model=create(merged['model'],merged).cpu()
                criterion=create(merged['criterion'],merged).cpu()
                post=create(merged['postprocessor'],merged).cpu()
        finally:vendor._restore_registry(GLOBAL_CONFIG,saved)
    return model,criterion,post,original,effective

def build(config=Config(),*,pretrained_path=None,synthetic=False):
    config.validate(synthetic=synthetic)
    model,base,post,original,effective=native_objects(config)
    sampling=vendor._configure_model_sampling(model,config)
    if pretrained_path is None and not synthetic:raise ValueError('strict ImageNet source required')
    loaded=vendor._load_pretrained(model,Path(pretrained_path),PRETRAINED_SHA) if pretrained_path else None
    from .criterion import adapt
    criterion=adapt(base)
    info={'config':config.binding(),'official_commit':COMMIT,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'criterion':criterion.recipe(),'sampling':sampling,'pretrained':loaded,'effective_config':effective}
    return model,criterion,info

def fp32_outputs(value):
    if isinstance(value,torch.Tensor):return value.float() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:fp32_outputs(v) for k,v in value.items()}
    if isinstance(value,list):return [fp32_outputs(v) for v in value]
    if isinstance(value,tuple):return tuple(fp32_outputs(v) for v in value)
    return value

@contextlib.contextmanager
def dn_capture(model):
    """Transient observers; native forward, outputs, state and RNG remain intact."""
    d=model.decoder;observed={'classes':None,'boxes_unact':None,'attention_mask':None,'embedding_calls':0}
    old=d._get_decoder_input
    def classes(module,args):
        observed['classes']=args[0].detach().clone();observed['embedding_calls']+=1
    def input(*args,**kwargs):
        box=args[3] if len(args)>3 else kwargs.get('denoising_bbox_unact')
        observed['boxes_unact']=None if box is None else box.detach().clone()
        return old(*args,**kwargs)
    def decoder(module,args,kwargs):
        mask=kwargs.get('attn_mask');observed['attention_mask']=None if mask is None else mask.detach().clone()
    h1=d.denoising_class_embed.register_forward_pre_hook(classes)
    h2=d.decoder.register_forward_pre_hook(decoder,with_kwargs=True)
    d._get_decoder_input=input
    try:yield observed
    finally:
        del d._get_decoder_input
        h1.remove();h2.remove()
