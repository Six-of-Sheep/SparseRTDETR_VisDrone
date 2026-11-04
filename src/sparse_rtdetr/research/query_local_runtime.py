"""Name-mapped e060 bridge and arm-bound complete R25 checkpoints."""
import copy,hashlib,json,os,time
from pathlib import Path
import torch
from sparse_rtdetr.protocol_r18.model import Config,build,CommonEMA
from sparse_rtdetr.protocol_r18.engine import Engine,learning_rate,tensor_digest
from sparse_rtdetr.protocol_r18 import checkpoint as native
from sparse_rtdetr.baseline.training_v2b import _optimizer
from .query_local import install,VERSION

PARENT_SHA='a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a'

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def create(parent,arm,device='cpu',config=None,synthetic=False):
    cfg=config or Config()
    s=torch.load(parent,map_location='cpu',weights_only=True)
    if not synthetic:
        if native.file_sha(parent)!=PARENT_SHA or s['engine']['completed_updates']!=18240:raise ValueError('R25 parent identity')
        if s['binding']['config']!=cfg.binding():raise ValueError('R25 original recipe')
    model,criterion,info=build(cfg,synthetic=True)
    if not synthetic and (native.old._model_layout(model)!=s['model_layout'] or criterion.recipe()!=s['behavior']['criterion']):raise ValueError('R25 native model/criterion behavior differs')
    model.load_state_dict(s['model'],strict=True)
    opt=_optimizer(model,cfg)
    # Validate names AND map moments by name, never by new positional ids.
    saved={}
    for g in s['optimizer_layout']['groups']:
        for name,i in zip(g['names'],g['ids']):
            if name in saved:raise ValueError('duplicate parent optimizer name')
            saved[name]=s['optimizer']['state'].get(i,{})
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(named)!=set(saved):raise ValueError('shared optimizer named coverage')
    for n,p in named.items():opt.state[p]=copy.deepcopy(saved[n])
    for g,oldg in zip(opt.param_groups,s['optimizer']['param_groups']):
        if g['param_names']!=oldg['param_names']:raise ValueError('parent optimizer recipe/name order')
        for k,v in oldg.items():
            if k!='params':g[k]=copy.deepcopy(v)
    ema=CommonEMA(model);ema.load_state_dict(s['ema'])
    for n,m in model.named_modules():m.training=s['model_training'][n]
    if arm!='A0':
        install(model,arm);install(ema.module,arm)
        ema.module.local_head.load_state_dict(model.local_head.state_dict(),strict=True)
        ema.module.requires_grad_(False)
        new=[('local_head.'+n,p) for n,p in model.local_head.named_parameters()]
        opt.add_param_group({'params':[p for _,p in new],'param_names':[n for n,_ in new],
                 'name':'r25_new','lr':learning_rate(s['engine']['completed_updates']) if s['engine']['completed_updates'] else cfg.learning_rate,
                 'weight_decay':cfg.weight_decay})
    model.to(device);criterion.to(device);ema.module.to(device)
    # Follow torch optimizer load semantics: Adam step stays CPU, moments follow parameter.
    for param,state in opt.state.items():
        for k,v in state.items():
            if isinstance(v,torch.Tensor) and k!='step':state[k]=v.to(param.device)
    e=Engine(model,criterion,opt,ema,cfg);e.completed_updates=s['engine']['completed_updates']
    e.parameter_steps={n:s['engine']['parameter_optimizer_steps'].get(n,0) for n in e.parameter_steps}
    e.r25_arm=arm;e.r25_parent=PARENT_SHA if not synthetic else native.file_sha(parent)
    checks={'shared_model_exact':all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in s['model'].items()),
      'shared_EMA_exact':all(torch.equal(ema.module.state_dict()[n].cpu(),v) for n,v in s['ema']['module'].items()),
      'shared_optimizer_exact':all(native.exact_equal(cpu(opt.state[named[n]]),cpu(v)) for n,v in saved.items()),
      'new_state_empty':all(not opt.state.get(p) for n,p in model.named_parameters() if n.startswith('local_head.')),
      'shared_parameter_steps_exact':all(e.parameter_steps[n]==v for n,v in s['engine']['parameter_optimizer_steps'].items()),
      'new_parameter_steps_zero':all(v==0 for n,v in e.parameter_steps.items() if n.startswith('local_head.')),
      'ema_updates':ema.updates,'completed_updates':e.completed_updates,'lr_next':learning_rate(e.completed_updates+1),
      'new_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith('local_head.')),
      'shared_parameters_sha256':tensor_digest((n,p) for n,p in model.named_parameters() if not n.startswith('local_head.')),
      'new_parameters_sha256':tensor_digest((n,p) for n,p in model.named_parameters() if n.startswith('local_head.'))}
    if not all(checks[k] for k in ['shared_model_exact','shared_EMA_exact','shared_optimizer_exact','new_state_empty','shared_parameter_steps_exact','new_parameter_steps_zero']):raise ValueError(('bridge mismatch',checks))
    if device!='cpu':native.restore_rng(s['rng'],e.device)
    else:
        rng=dict(s['rng']);rng['cuda']=None;native.restore_rng(rng,e.device)
    return e,checks

def capture(e):
    if e.phase or e.failed or any(p.grad is not None for p in e.model.parameters()):raise ValueError('not a clean R25 update boundary')
    return cpu({'version':VERSION,'arm':e.r25_arm,'parent_sha256':e.r25_parent,'config':e.config.binding(),
      'model':e.model.state_dict(),'model_modes':{n:m.training for n,m in e.model.named_modules()},
      'optimizer':e.optimizer.state_dict(),'optimizer_names':[g['param_names'] for g in e.optimizer.param_groups],
      'ema':e.ema.state_dict(),'engine':e.state_dict(),'rng':native.capture_rng(e.device)})

def save(e,path):
    p=Path(path)
    if p.exists():raise FileExistsError(p)
    payload=capture(e)
    with p.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    r={'path':str(p),'bytes':p.stat().st_size,'sha256':native.file_sha(p),'arm':e.r25_arm,'update':e.completed_updates,'formal_usable':False}
    p.with_suffix('.json').write_text(json.dumps(r,indent=1));return r

def restore(e,path,sha):
    if native.file_sha(path)!=sha:raise ValueError('R25 saved SHA')
    s=torch.load(path,map_location='cpu',weights_only=True)
    if (s['version'],s['arm'],s['parent_sha256'],s['config'])!=(VERSION,e.r25_arm,e.r25_parent,e.config.binding()):raise ValueError('R25 cross-arm/parent/recipe state')
    if s['optimizer_names']!=[g['param_names'] for g in e.optimizer.param_groups]:raise ValueError('R25 optimizer names')
    e.model.load_state_dict(s['model'],strict=True);e.optimizer.load_state_dict(s['optimizer']);e.ema.load_state_dict(s['ema'])
    for n,m in e.model.named_modules():m.training=s['model_modes'][n]
    e.load_state_dict(s['engine']);native.restore_rng(s['rng'],e.device)
    if not native.exact_equal(capture(e),s):raise ValueError('R25 complete restored state not exact')
    return {'full_state_exact':True,'arm':e.r25_arm,'update':e.completed_updates,'optimizer_EMA_RNG_cursor_exact':True}
