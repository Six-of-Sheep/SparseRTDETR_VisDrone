"""New namespace on certified SSD channel; 1 MiB and refusal rules unchanged."""
import time,os
from pathlib import Path
from ..localization import channel
from ..localization.runio import require,digest,read,sha,write,reference,event
from .model import ARM
PROTOCOL='native-r18-common-execution-progress-v2'
CURRENT_PROTOCOL='native-r18-common-e088-recovery-progress-v4'
TASK='rtdetrv2-r18-common-protocol-seed0-e120-20261013-001'

def validate_audit(payload,identity):
    ref=payload['audit'];path=Path(ref['path'])
    require(path.is_absolute() and path.is_relative_to(Path(identity['output_root'])/'audits') and not path.is_symlink(),'audit path/binding differs')
    require(reference(path)==ref,'scientific audit missing/corrupt')
    value=read(path)
    for key in ('task_id','source_sha256','contract_sha256'):require(value[key]==identity[key],'audit identity differs: '+key)
    require(value['update']==payload['update'] and value['role']==payload['audit_role'] and value['result']['input_sha256']==payload['input_sha256'],'audit update/input differs')
    require(value['result']['update']==value['update'],'audit result clock differs')
    return value

def _install_legacy():
    if getattr(channel,'_native_r18_installed',False):return
    original=channel.validate_event;NativeReader=channel.Reader
    def validate(payload,identity,stream):
        if identity.get('progress_protocol')!=PROTOCOL:return original(payload,identity,stream)
        require(identity.get('task_id')==TASK and identity.get('arm')==ARM,'new protocol task/model identity differs')
        require(all(channel._hex64(identity.get(k)) for k in ('source_sha256','contract_sha256')),'frozen source/contract missing')
        require(type(payload.get('monotonic')) in (int,float) and __import__('math').isfinite(payload['monotonic']),'finite clock missing')
        if stream!='progress':return None
        kind=payload.get('event');mode=identity['mode']
        if kind=='cpu_input_window_complete':
            require(mode=='cpu-spawn' and payload.get('cpu_window_number') in (1,2),'CPU window scope expanded')
            n=payload['cpu_window_number'];require(payload['input']['ids']==identity['fixed_ids'][n-1] and payload.get('CUDA_initialized') is False,'CPU input identity/CUDA differs')
            return digest([kind,n])
        if kind=='update_complete':
            require(mode in ('cpu-state','cpu-execution','gpu-acceptance','formal'),'updates outside fixed run')
            if mode in ('cpu-state','gpu-acceptance'):
                require(payload.get('update') in (1,2) and payload.get('audit_role') in ('direct','replay'),'update/role outside fixed3')
            else:
                require(type(payload.get('update')) is int and 1<=payload['update']<=(2 if mode=='cpu-execution' else 36480) and payload.get('audit_role')=='direct','formal update/role differs')
            validate_audit(payload,identity)
            return digest([kind,payload['update'],payload['audit_role']])
        if kind in ('evaluation_image_saved','evaluation_budget_complete'):
            auth=identity.get('evaluation_authorization');require(mode in ('inference','evaluation') and isinstance(auth,dict),'endpoint event phase/authority differs')
            require(payload.get('checkpoint_sha256')==auth['checkpoint_sha256'] and channel._hex64(auth['checkpoint_sha256']),'endpoint checkpoint differs')
            if kind=='evaluation_image_saved':
                n=payload.get('evaluation_image_number');require(mode=='inference' and type(n) is int and 1<=n<=548 and payload.get('image_id')==auth['image_ids'][n-1],'endpoint image order differs')
                return digest([kind,auth['checkpoint_sha256'],n])
            require(mode=='evaluation' and payload.get('budget') in (600,300) and payload.get('name')=='top'+str(payload['budget']) and payload.get('images')==548,'endpoint budget differs')
            return digest([kind,auth['checkpoint_sha256'],payload['budget']])
        require(kind in ('checkpoint_begin','checkpoint_complete','cuda_initialized','epoch_begin') and mode in ('cpu-state','cpu-execution','gpu-acceptance','formal','inference'),'unknown/unapproved new event')
        if kind=='cuda_initialized':require(mode in ('gpu-acceptance','formal','inference'),'CPU CUDA event refused')
        elif kind=='epoch_begin':require(mode=='formal' and type(payload.get('epoch')) is int and 0<=payload['epoch']<=120,'epoch phase differs')
        else:require(type(payload.get('update')) is int and 0<=payload['update']<=(36480 if mode=='formal' else 2),'checkpoint update scope differs')
        return digest([kind,payload.get('name'),payload.get('update')])
    class Reader(NativeReader):
        def _accept(self,record):
            result=super()._accept(record)
            if result and self.stream=='progress' and self.bound_identity.get('progress_protocol')==PROTOCOL:
                p=record['payload']
                if p['event']=='update_complete':
                    n=self.clock.get('_native_step',0)
                    if self.bound_identity['mode'] in ('cpu-state','gpu-acceptance'):
                        steps=[(1,'direct'),(2,'direct'),(2,'replay')]
                        require(n<3 and (p['update'],p['audit_role'])==steps[n],'fixed3 update/replay order differs')
                    else:require(p['update']==n+1 and p['audit_role']=='direct','formal update skipped/replayed')
                    self.clock['_native_step']=n+1
                if p['event']=='evaluation_image_saved':
                    n=self.clock.get('_native_eval_image',0)+1;require(p['evaluation_image_number']==n,'evaluation image skipped');self.clock['_native_eval_image']=n
                if p['event']=='evaluation_budget_complete':
                    n=self.clock.get('_native_budget',0);require(n<2 and p['budget']==[600,300][n],'evaluation budget order');self.clock['_native_budget']=n+1
            return result
    channel.validate_event=validate;channel.Reader=Reader;channel._native_r18_installed=True

def save_audit(out,identity,result,role):
    value={**{k:identity[k] for k in ('task_id','source_sha256','contract_sha256')},'update':result['update'],'role':role,'result':result}
    path=Path(out)/'audits'/f"update-{result['update']:05d}-{role}.json";path.parent.mkdir(exist_ok=True)
    write(path,value)
    payload={'update':result['update'],'audit_role':role,'input_sha256':result['input_sha256'],'audit':reference(path),
             'loss':result['loss'],'finite':result['finite'],'completed_epochs':result['update']//304}
    event(out,'update_complete',**payload)
    return payload

def recover_audit(payload,identity):return validate_audit(payload,identity)['result']


def install():
    _install_legacy()
    from .progress_v3 import install as v3_install
    v3_install()
    from .progress_v4 import install as current_install
    current_install()
