"""Cursor-bound recovery channel. Historical v2 streams retain their rules."""
from pathlib import Path
from ..localization import channel
from ..localization.runio import require,digest,reference,read
PROTOCOL='native-r18-common-epoch-recovery-progress-v3'

def install():
    if getattr(channel,'_native_r18_v3_installed',False):return
    from . import transport
    original=channel.validate_event;Base=channel.Reader
    def validate(p,b,stream):
        if b.get('progress_protocol')!=PROTOCOL:return original(p,b,stream)
        require(b.get('task_id')==transport.TASK and b.get('arm')==transport.ARM,'v3 task/arm differs')
        require(all(channel._hex64(b.get(k)) for k in ('source_sha256','contract_sha256')),'v3 source/contract missing')
        require(type(p.get('monotonic')) in (int,float) and __import__('math').isfinite(p['monotonic']),'v3 clock invalid')
        if stream!='progress':return None
        mode=b.get('protocol_mode',b['mode'])
        if 'protocol_mode' in b:require(b['mode']=='cpu-protocol' and b.get('CPU_protocol_only') is True,'CPU protocol identity invalid')
        kind=p.get('event');start=b.get('starting_cursor')
        require(type(start) is int and 0<=start<=36480,'starting cursor missing')
        if mode in ('formal','gpu-recovery'):require(start==304 or b.get('CPU_protocol_only') is True,'original recovery cursor differs')
        if kind=='resume_parent_exact':
            require(mode in ('formal','gpu-recovery') and p.get('completed_updates')==start and p.get('checkpoint_sha256')==b.get('resume_parent_sha256') and channel._hex64(p.get('checkpoint_sha256')),'resume identity/cursor differs')
            require(p.get('full_state_exact') is True and p.get('completed_epochs')==start//304,'resume state receipt differs')
            return digest([kind,p['checkpoint_sha256'],start])
        if kind=='epoch_begin':
            e=p.get('epoch');u=p.get('completed_updates')
            require(mode in ('formal','gpu-recovery') and type(e) is int and 1<=e<=120 and type(u) is int and u==(e-1)*304 and p.get('completed_epochs')==e-1,'epoch/cursor inconsistent')
            require(isinstance(p.get('dropped_ids'),list) and len(p['dropped_ids'])==5 and len(set(p['dropped_ids']))==5,'epoch dropped identity differs')
            return digest([kind,e,u])
        if kind=='update_complete':
            require(mode in ('formal','gpu-recovery'),'updates outside recovery phase')
            u=p.get('update');role=p.get('audit_role')
            require(type(u) is int and start<u<=36480,'recovery update range differs')
            require(role in ('direct','replay') if mode=='gpu-recovery' else role=='direct','update role differs')
            if mode=='gpu-recovery':require(u in (305,306),'fixed3 scope expanded')
            transport.validate_audit(p,b)
            return digest([kind,u,role])
        if kind in ('checkpoint_begin','checkpoint_complete'):
            u=p.get('update');name=p.get('name')
            require(mode in ('formal','gpu-recovery') and type(u) is int and start<=u<=36480 and isinstance(name,str) and name==Path(name).name and name.endswith('.pth'),'checkpoint scope/name differs')
            if mode=='gpu-recovery':require(u in (305,306),'acceptance checkpoint range differs')
            if kind=='checkpoint_complete':
                cp=p.get('checkpoint');require(isinstance(cp,dict) and cp.get('completed_updates')==u and cp.get('path')==str(Path(b['output_root'])/name) and channel._hex64(cp.get('sha256')),'checkpoint completion identity differs')
                require(reference(cp['path'])=={'path':cp['path'],'bytes':cp['size_bytes'],'sha256':cp['sha256']},'completed checkpoint bytes differ')
            return digest([kind,name,u])
        if kind=='cuda_initialized':
            require(mode in ('formal','gpu-recovery','inference') and not b.get('CPU_protocol_only'),'unapproved CUDA event')
            return digest([kind])
        if kind=='comparison_complete':
            auth=b.get('evaluation_authorization');require(mode=='comparison' and start==36480 and isinstance(auth,dict),'comparison endpoint authority missing')
            require(p.get('checkpoint_sha256')==auth['checkpoint_sha256'] and p.get('budgets')==[600,300],'comparison endpoint/budgets differ')
            ref=p.get('result');require(isinstance(ref,dict) and ref['path']==str(Path(b['output_root'])/'RESULT.json') and reference(ref['path'])==ref,'comparison result bytes differ')
            value=read(ref['path']);require(value['status']=='PASS' and value.get('source_sha256')==b['source_sha256'] and value.get('contract_sha256')==b['contract_sha256'],'comparison source identity differs')
            return digest([kind,p['checkpoint_sha256']])
        if kind in ('evaluation_image_saved','evaluation_budget_complete'):
            auth=b.get('evaluation_authorization');require(mode in ('inference','evaluation') and start==36480 and isinstance(auth,dict),'endpoint transition authority differs')
            cp=auth['checkpoint_sha256'];require(channel._hex64(cp) and p.get('checkpoint_sha256')==cp,'endpoint state differs')
            if kind=='evaluation_image_saved':
                n=p.get('evaluation_image_number');require(mode=='inference' and type(n) is int and 1<=n<=548 and p.get('image_id')==auth['image_ids'][n-1],'image order/identity differs')
                return digest([kind,cp,n])
            require(mode=='evaluation' and p.get('budget') in (600,300) and p.get('images')==548 and p.get('name')=='top'+str(p['budget']),'budget/name differs')
            return digest([kind,cp,p['budget']])
        raise RuntimeError('unknown v3 event: '+str(kind))
    class Reader(Base):
        def _accept(self,record):
            result=super()._accept(record)
            if not result or self.stream!='progress' or self.bound_identity.get('progress_protocol')!=PROTOCOL:return result
            b=self.bound_identity;p=record['payload'];kind=p['event'];mode=b.get('protocol_mode',b['mode']);start=b['starting_cursor']
            cursor=self.clock.get('_v3_cursor',start)
            if kind=='resume_parent_exact':
                require(not self.clock.get('_v3_resumed') and cursor==start,'duplicate/conflicting restore transition')
                self.clock['_v3_resumed']=True
            elif kind=='epoch_begin':
                require(self.clock.get('_v3_resumed') and p['completed_updates']==cursor,'epoch before restore or missing updates')
                previous=self.clock.get('_v3_epoch')
                require(previous is None or p['epoch']==previous+1,'epoch skipped/regressed')
                if previous is not None:require(self.clock.get('_v3_saved')==cursor,'epoch advanced before complete checkpoint')
                self.clock['_v3_epoch']=p['epoch']
            elif kind=='update_complete':
                require(self.clock.get('_v3_resumed') and self.clock.get('_v3_epoch')==(p['update']-1)//304+1,'update epoch/restore context missing')
                n=self.clock.get('_v3_steps',0)
                if mode=='gpu-recovery':
                    seq=[(305,'direct'),(306,'direct'),(306,'replay')]
                    require(n<3 and (p['update'],p['audit_role'])==seq[n],'fixed recovery sequence differs')
                else:require(p['update']==cursor+1,'formal update skipped/replayed')
                self.clock['_v3_steps']=n+1;self.clock['_v3_cursor']=p['update']
            elif kind=='checkpoint_begin':
                require(self.clock.get('_v3_resumed') and p['update']==cursor,'checkpoint before restore/wrong cursor')
                require(not self.clock.get('_v3_checkpoint_open'),'checkpoint already open')
                self.clock['_v3_checkpoint_open']=[p['name'],p['update']]
            elif kind=='checkpoint_complete':
                require(self.clock.get('_v3_checkpoint_open')==[p['name'],p['update']] and p['update']==cursor,'checkpoint completion without matching begin/cursor')
                self.clock['_v3_checkpoint_open']=None;self.clock['_v3_saved']=cursor
            elif kind=='evaluation_image_saved':
                n=self.clock.get('_v3_images',0)+1;require(p['evaluation_image_number']==n,'endpoint image skipped');self.clock['_v3_images']=n
            elif kind=='evaluation_budget_complete':
                n=self.clock.get('_v3_budgets',0);require(n<2 and p['budget']==[600,300][n],'endpoint budget skipped');self.clock['_v3_budgets']=n+1
            return result
    channel.validate_event=validate;channel.Reader=Reader;channel._native_r18_v3_installed=True
