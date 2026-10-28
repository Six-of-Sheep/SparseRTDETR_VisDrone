"""PR26 formal-only state; accepted layout/RNG validation with separate scope."""
import hashlib,json,os
from pathlib import Path
import torch
from ..protocol_r18 import checkpoint as common
from ..protocol_r18.runio import fsync_directory
from ..baseline import training_v2b_checkpoint as layout
from .sabl_control import need,VERSION

FORMAT='sparse_rtdetr.sabl_formal24_fullstate_v1'

def capture(engine,binding):
    need(binding['scope'] in ('CPU_SYNTHETIC_SABL_FORMAL','GPU_SABL_FORMAL_24'),'not a formal scope')
    need(binding['arm']==engine.model.arm and binding['version']==VERSION,'arm/version mismatch')
    need(not engine.phase and not engine.failed and all(p.grad is None for p in engine.model.parameters()),'not complete boundary')
    engine.health(engine.model,engine.optimizer,engine.ema)
    return layout._safe({'format':FORMAT,'binding':binding,'runtime':str(torch.__version__),'device':str(engine.device),
        'model':engine.model.state_dict(),'model_layout':layout._model_layout(engine.model),
        'model_training':{n:m.training for n,m in engine.model.named_modules()},
        'optimizer':engine.optimizer.state_dict(),'optimizer_layout':layout._optimizer_layout(engine.model,engine.optimizer),
        'ema':engine.ema.state_dict(),'engine':engine.state_dict(),'rng':common.capture_rng(engine.device),
        'criterion':engine.criterion.recipe()})

def save(engine,binding,path):
    path=Path(path);state=capture(engine,binding)
    need(not path.exists() and not path.with_suffix('.partial').exists(),'checkpoint overwrite')
    partial=path.with_suffix('.partial')
    with partial.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.link(partial,path);fsync_directory(path.parent);partial.unlink()
    receipt={'path':str(path),'sha256':common.file_sha(path),'bytes':path.stat().st_size,'binding':binding,'updates':engine.completed_updates}
    with Path(str(path)+'.complete.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    return receipt

def restore(engine,binding,path,expected_sha):
    try:
        need(common.file_sha(path)==expected_sha,'checkpoint SHA')
        state=torch.load(path,map_location='cpu',weights_only=True)
        need(state['format']==FORMAT and state['binding']==binding,'checkpoint scope/arm')
        need(state['runtime']==str(torch.__version__) and state['device']==str(engine.device),'checkpoint runtime')
        current=capture(engine,binding);need(set(state)==set(current),'checkpoint schema')
        for key in ('model_layout','optimizer_layout','criterion'):
            need(state[key]==current[key],'checkpoint layout/recipe '+key)
        layout._check_model_state(state['model'],current['model_layout'],'model')
        layout._check_model_state(state['ema']['module'],current['model_layout'],'ema')
        engine.validate_state_dict(state['engine']);u=state['engine']['completed_updates'];need(0<=u<=7296,'formal clock')
        layout._check_optimizer(state['optimizer'],current['optimizer_layout'],engine.model,u)
        for g in current['optimizer_layout']['groups']:
            for name,id in zip(g['names'],g['ids']):need(int(state['optimizer']['state'].get(id,{}).get('step',0))==state['engine']['parameter_optimizer_steps'][name],'optimizer clock')
        need(state['ema']['updates']==u and state['ema']['decay']==.9999 and state['ema']['warmups']==2000,'EMA clock')
        expected_lr=engine.rate(u) if u else engine.config.learning_rate
        need(all(g['lr']==expected_lr for g in state['optimizer']['param_groups']),'LR clock')
        for name,v in current['model'].items():
            if name.rsplit('.',1)[-1] in ('anchors','valid_mask','num_points_scale'):
                need(torch.equal(state['model'][name],v) and torch.equal(state['ema']['module'][name],v),'immutable cache')
        need(set(state['model_training'])=={n for n,m in engine.model.named_modules()} and all(type(v) is bool for v in state['model_training'].values()),'mode map')
        common.validate_rng(state['rng'],engine.device)
        engine.model.load_state_dict(state['model'],strict=True)
        for name,m in engine.model.named_modules():m.training=state['model_training'][name]
        engine.optimizer.load_state_dict(state['optimizer']);engine.ema.load_state_dict(state['ema']);engine.load_state_dict(state['engine']);common.restore_rng(state['rng'],engine.device)
        need(layout._equal(state,capture(engine,binding)),'full restored state mismatch')
        return {'full_state_exact':True,'updates':u}
    except BaseException:
        engine.failed=True;raise
