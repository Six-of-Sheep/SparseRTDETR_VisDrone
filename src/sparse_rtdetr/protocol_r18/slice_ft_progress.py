"""One namespace on the unchanged 1MiB transactional SSD transport."""
import math
from pathlib import Path
from ..localization import channel
from ..localization.runio import require,read,reference,digest,event,publish

PROTOCOL='r18-slice-finetune-full-execution-v1'


def audit(p,b):
    r=p['audit'];q=Path(r['path'])
    require(q.is_absolute() and q.is_relative_to(Path(b['output_root'])/'audits') and not q.is_symlink(),'FT audit escaped')
    require(reference(q)==r,'FT audit corrupt')
    x=read(q)
    require(all(x[k]==b[k] for k in ('task_id','source_sha256','contract_sha256','arm')),'FT audit identity differs')
    require(x['role']==p['role'] and x['result']['update']==p['update'] and x['result']['input_sha256']==p['input_sha256'],'FT audit clock/input differs')
    return x


def install():
    if getattr(channel,'_slice_ft_installed',False):return
    previous=channel.validate_event;PreviousReader=channel.Reader
    def validate(p,b,stream):
        if b.get('progress_protocol')!=PROTOCOL:return previous(p,b,stream)
        require(isinstance(p,dict) and type(p.get('monotonic')) in (int,float) and math.isfinite(p['monotonic']),'FT finite clock absent')
        require(b['task_id'].startswith('r18-slice-finetune-20261018-001') and b['arm'] in ('T-control','T-slice','paired'),'FT identity differs')
        require(all(channel._hex64(b[k]) for k in ('source_sha256','contract_sha256','plan_sha256')),'FT frozen identity missing')
        if stream!='progress':return None
        kind=p['event'];mode=b['phase']
        allowed={'cuda_initialized','epoch_begin','epoch_complete','checkpoint_begin','checkpoint_complete','update_complete','inference_batch','evaluation_budget_complete','phase_complete'}
        require(kind in allowed,'unknown FT event')
        if kind=='update_complete':
            require(mode.startswith(('accept-','formal-')),'FT updates in other phase')
            audit(p,b)
        if kind=='evaluation_budget_complete':
            require(mode=='evaluation' and p['budget'] in (600,300) and p['arm_name'] in ('T-control','T-slice') and p['view'] in ('A','C') and p['images']==b['evaluation_images'],'FT evaluation scope')
        if kind=='inference_batch':
            require(mode.startswith('infer-') and p['kind'] in ('profile','full'),'FT inference phase')
            ids=b['fixed64'] if p['kind']=='profile' else b['development_ids']
            require(p['ids']==ids[(p['batch']-1)*4:p['batch']*4],'FT image order differs')
        if kind=='phase_complete':require(reference(p['result']['path'])==p['result'],'FT terminal RESULT SHA differs')
        return digest([kind,p.get('update'),p.get('role'),p.get('epoch'),p.get('name'),p.get('kind'),p.get('batch'),p.get('ordinal')])
    class Reader(PreviousReader):
        def _accept(self,record):
            if self.bound_identity.get('progress_protocol')!=PROTOCOL or self.stream!='progress':return super()._accept(record)
            # Existing identical wire records stay transport-idempotent; a new
            # logical terminal record is forbidden even with the same payload.
            if record['seq'] in self.seen:return super()._accept(record)
            p=record['payload'];k=p['event'];b=self.bound_identity;mode=b['phase']
            require(not self.clock.get('_ft_terminal'),'FT event after terminal')
            n=self.clock.get('_ft_u',0);ep=self.clock.get('_ft_epoch',0);step=b['updates_per_epoch']
            if k=='epoch_begin':
                require(mode.startswith('formal-') and p['epoch']==ep+1 and p['update']==ep*step and n==ep*step and self.clock.get('_ft_epoch_closed',0)==ep,'FT cross-epoch begin gap/conflict')
                self.clock['_ft_epoch']=ep+1
            elif k=='update_complete':
                if mode.startswith('accept-'):
                    seq=[(1,'direct'),(2,'direct'),(2,'replay')];i=self.clock.get('_ft_accept',0)
                    require(i<3 and (p['update'],p['role'])==seq[i],'FT acceptance fixed3 order')
                    self.clock['_ft_accept']=i+1
                else:
                    require(p['role']=='direct' and p['update']==n+1 and ep==(n//step)+1,'FT formal update gap/epoch')
                self.clock['_ft_u']=p['update']
            elif k=='checkpoint_begin':
                require(not self.clock.get('_ft_cp_open') and p['update']==n,'FT checkpoint begin clock')
                self.clock['_ft_cp_open']=p['name']
            elif k=='checkpoint_complete':
                require(self.clock.get('_ft_cp_open')==p['name'] and p['update']==n and reference(p['checkpoint']['path'])=={key:p['checkpoint'][key] for key in ('path','bytes','sha256')},'FT checkpoint complete/binding')
                self.clock['_ft_cp_open']=None
            elif k=='epoch_complete':
                require(mode.startswith('formal-') and p['epoch']==ep and n==ep*step and not self.clock.get('_ft_cp_open'),'FT epoch completion gap')
                self.clock['_ft_epoch_closed']=ep
            elif k=='inference_batch':
                key='_ft_'+p['kind'];i=self.clock.get(key,0)+1
                require(p['batch']==i,'FT inference batch gap');self.clock[key]=i
            elif k=='evaluation_budget_complete':
                i=self.clock.get('_ft_eval',0)
                expected=[(a,v,z) for a in ('T-control','T-slice') for v in ('A','C') for z in (600,300)]
                require(i<8 and (p['arm_name'],p['view'],p['budget'])==expected[i] and p['ordinal']==i+1,'FT evaluation order/gap')
                self.clock['_ft_eval']=i+1
            elif k=='phase_complete':
                require(not self.clock.get('_ft_cp_open'),'FT checkpoint incomplete at terminal')
                if mode.startswith('accept-'):require(self.clock.get('_ft_accept')==3,'FT incomplete acceptance')
                elif mode.startswith('formal-'):require(n==b['total_updates'] and self.clock.get('_ft_epoch_closed')==b['epochs'],'FT incomplete formal')
                elif mode.startswith('infer-'):require(self.clock.get('_ft_profile')==len(b['fixed64'])//4 and self.clock.get('_ft_full')==b['evaluation_images']//4,'FT incomplete inference')
                else:require(mode=='evaluation' and self.clock.get('_ft_eval')==8,'FT incomplete evaluation')
                self.clock['_ft_terminal']=True
            return super()._accept(record)
    channel.Reader=Reader;channel.validate_event=validate;channel._slice_ft_installed=True


def save_audit(out,b,result,role):
    path=Path(out)/'audits'/f"update-{result['update']:05d}-{role}.json"
    publish(path,{**{k:b[k] for k in ('task_id','source_sha256','contract_sha256','arm')},'role':role,'result':result})
    event(out,'update_complete',update=result['update'],role=role,audit=reference(path),input_sha256=result['input_sha256'],loss=result['loss'])
