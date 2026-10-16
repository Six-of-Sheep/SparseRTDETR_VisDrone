"""Identity-bound, single-writer append streams on local SSD.

Only complete authenticated records advance a cursor. NAS copies have no
control role. Any source disappearance/replacement/truncation is fatal.
"""
import json,os,time,hashlib,math,copy
from collections import ChainMap
from pathlib import Path
from .runio import require,read,write,digest,sha,canonical,fsync_directory

ENV='F0_RUNTIME_BINDING'
_writers={}
_resources={}

def binding():
    p=os.environ.get(ENV)
    if not p:return None
    value=read(p)
    require(sha(p)==os.environ.get(ENV+'_SHA256'),'runtime identity bytes changed')
    require(value['schema']=='f0-ssd-channel-v4','runtime channel schema changed')
    require(Path(p).parent==Path(value['runtime_root']),'runtime binding path changed')
    return value

def receipt_root(out):
    b=binding()
    return Path(out) if b is None else Path(b['runtime_root'])

def create(root,identity):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    value={'schema':'f0-ssd-channel-v4','runtime_root':str(root),**identity}
    write(root/'identity.json',value)
    for name in ('progress','resources'):
        with (root/(name+'.jsonl')).open('xb') as f:f.flush();os.fsync(f.fileno())
    fsync_directory(root)
    return value

def environment(value):
    p=Path(value['runtime_root'])/'identity.json'
    return {ENV:str(p),ENV+'_SHA256':sha(p)}


PROTOCOL_NAME='r0-training-evaluation-progress-v1'
OR_PROTOCOL_NAME='or-training-evaluation-trace-progress-v1'
RSMALL_PROTOCOL_NAME='rsmall-training-evaluation-geometry-statistics-progress-v1'
EVENTS={'cpu_audit_record_complete','cpu_input_window_complete','cuda_initialized','resume_parent_exact','epoch_begin','checkpoint_begin','checkpoint_complete','update_complete','trace_image_saved','trace_image_analyzed','diagnostic_micro_backward','forensic_layer_saved','forensic_forward_closed','terminal_dn_micro_backward','evaluation_image_saved','evaluation_budget_complete','comparison_draw_complete','comparison_budget_complete','geometry_image_complete'}

def _hex64(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _rsmall_event(payload,identity):
    kind=payload['event'];mode=identity.get('mode')
    require(identity.get('task_id') in ('rsmall-seed0-e120-20261012-001','rsmall-seed0-e120-20261012-002','rsmall-seed0-e120-20261012-003') and identity.get('arm')=='R-small' and identity.get('representation')=='r0_scalar_sigmoid_go_v1' and identity.get('progress_protocol')==RSMALL_PROTOCOL_NAME,'R-small progress namespace')
    require(all(_hex64(identity.get(k)) for k in ('source_sha256','contract_sha256','proposal_sha256')),'R-small frozen source/contract/proposal absent')
    if kind in ('comparison_draw_complete','comparison_budget_complete'):
        auth=identity.get('comparison_authorization');step=payload.get('comparison_step');budget=payload.get('budget')
        require(mode=='comparison' and isinstance(auth,dict) and auth.get('budgets')==[600,300] and auth.get('replicates_per_budget')==2000,'R-small comparison authority')
        require(all(_hex64(auth.get(k)) and payload.get(k)==auth[k] for k in ('checkpoint_sha256','R0_checkpoint_sha256','comparison_protocol_sha256')),'R-small comparison frozen endpoint/protocol')
        require(type(step) is int and 1<=step<=4000 and budget==(600 if step<=2000 else 300),'R-small comparison counter/budget')
        if kind=='comparison_budget_complete':require(step in (2000,4000),'R-small draw budget incomplete')
        return digest([kind,auth['checkpoint_sha256'],auth['R0_checkpoint_sha256'],step,budget])
    if kind=='geometry_image_complete':
        auth=identity.get('geometry_authorization');n=payload.get('geometry_image_number')
        require(mode=='geometry' and isinstance(auth,dict),'R-small geometry authority')
        ids=auth.get('image_ids');require(isinstance(ids,list) and len(ids)==len(set(ids))==548 and all(_hex64(i) for i in ids),'R-small fixed548 geometry IDs')
        require(type(n) is int and 1<=n<=548 and payload.get('image_id')==ids[n-1] and 'update' not in payload,'R-small geometry identity/order')
        require(all(_hex64(auth.get(k)) and payload.get(k)==auth[k] for k in ('checkpoint_sha256','geometry_protocol_sha256')),'R-small geometry endpoint/protocol')
        return digest([kind,auth['checkpoint_sha256'],auth['geometry_protocol_sha256'],n])
    if kind in ('evaluation_image_saved','evaluation_budget_complete'):
        auth=identity.get('evaluation_authorization')
        require(mode in ('inference','evaluation') and isinstance(auth,dict) and auth.get('images')==548 and auth.get('budgets')==[600,300] and _hex64(auth.get('checkpoint_sha256')) and payload.get('checkpoint_sha256')==auth['checkpoint_sha256'],'R-small endpoint authority')
        if kind=='evaluation_image_saved':
            require(mode=='inference' and type(payload.get('evaluation_image_number')) is int and 1<=payload['evaluation_image_number']<=548 and _hex64(payload.get('image_id')),'R-small endpoint image scope')
            return digest([kind,auth['checkpoint_sha256'],payload['evaluation_image_number']])
        require(mode=='evaluation' and payload.get('budget') in (600,300) and payload.get('name')=='top'+str(payload['budget']) and payload.get('images')==548,'R-small endpoint budget scope')
        return digest([kind,auth['checkpoint_sha256'],payload['budget']])
    allowed={'cuda_initialized':('acceptance','formal','inference'),'checkpoint_begin':('acceptance','formal'),
             'checkpoint_complete':('acceptance','formal'),'update_complete':('acceptance','formal'),'epoch_begin':('formal',)}
    require(kind in allowed and mode in allowed[kind],'R-small event outside authorized phase')
    if kind=='update_complete':require(type(payload.get('update')) is int and 1<=payload['update']<=(2 if mode=='acceptance' else 36480),'R-small exact update scope')
    if kind=='update_complete' and identity.get('task_id')=='rsmall-seed0-e120-20261012-003':
        from ..rsmall.audit import validate_event_reference
        validate_event_reference(payload,identity)
        return digest([kind,payload['update'],payload['audit_role']])
    return None

def validate_event(payload,identity,stream):
    require(isinstance(payload,dict) and type(payload.get('monotonic')) in (int,float) and math.isfinite(payload['monotonic']),'channel finite monotonic timestamp absent')
    if stream!='progress':return None
    kind=payload.get('event');require(kind in EVENTS,'unknown progress operation')
    if kind=='cpu_audit_record_complete':
        from ..rsmall.audit import validate_event_reference
        validate_event_reference(payload,identity)
        require(identity.get('task_id')=='rsmall-seed0-e120-20261012-003' and identity.get('mode')=='cpu-audit' and identity.get('progress_protocol')=='rsmall-audit-transport-cpu-v1','CPU real saved audit authority')
        return digest([kind,payload['audit']['sha256']])
    if kind=='cpu_input_window_complete':
        auth=identity.get('cpu_input_authorization');n=payload.get('cpu_window_number')
        require(identity.get('task_id') in ('rsmall-seed0-e120-20261012-002','rsmall-seed0-e120-20261012-003') and identity.get('arm')=='R-small' and identity.get('mode')=='cpu-spawn' and identity.get('progress_protocol')=='rsmall-entry-real-spawn-cpu-v1','CPU real-input event namespace')
        require(all(_hex64(identity.get(k)) for k in ('source_sha256','contract_sha256','proposal_sha256')),'CPU frozen source/contract/proposal')
        require(auth=={'epoch':1,'windows':[0,1],'positions':list(range(32)),'workers':2,'prefetch_factor':2},'CPU input authority differs')
        require(type(n) is int and n in (1,2) and payload.get('epoch')==1 and payload.get('window')==n-1 and payload.get('positions')==list(range((n-1)*16,n*16)) and 'update' not in payload,'CPU actual window scope')
        require(_hex64(payload.get('input_sha256')) and _hex64(payload.get('GT_base_weights_sha256')) and payload.get('CUDA_initialized') is False,'CPU actual input proof absent')
        return digest([kind,identity['contract_sha256'],n])
    if identity.get('task_id') in ('rsmall-seed0-e120-20261012-001','rsmall-seed0-e120-20261012-002','rsmall-seed0-e120-20261012-003') or identity.get('progress_protocol')==RSMALL_PROTOCOL_NAME:
        operation=_rsmall_event(payload,identity)
        if operation is not None:return operation
    require(kind!='geometry_image_complete','geometry event requires authenticated R-small identity')
    if identity.get('progress_protocol')==OR_PROTOCOL_NAME or kind in ('comparison_draw_complete','comparison_budget_complete'):
        require(identity.get('task_id')=='or-seed0-e120-20261011-001' and identity.get('arm')=='OR' and identity.get('progress_protocol')==OR_PROTOCOL_NAME,'OR protocol cannot use a foreign identity')
    if identity.get('task_id')=='or-seed0-e120-20261011-001':
        require(identity.get('arm')=='OR' and identity.get('representation')=='center_logextent_categorical_v1' and identity.get('progress_protocol')==OR_PROTOCOL_NAME,'OR progress identity/version')
        require(all(_hex64(identity.get(k)) for k in ('source_sha256','contract_sha256')),'OR source/contract identity')
        mode=identity.get('mode')
        if kind in ('comparison_draw_complete','comparison_budget_complete'):
            auth=identity.get('comparison_authorization');budget=payload.get('budget');step=payload.get('comparison_step')
            require(mode=='comparison' and isinstance(auth,dict) and auth.get('budgets')==[600,300] and auth.get('replicates_per_budget')==2000,'OR comparison authority')
            require(all(_hex64(auth.get(k)) for k in ('checkpoint_sha256','R0_checkpoint_sha256','comparison_protocol_sha256')) and all(payload.get(k)==auth[k] for k in ('checkpoint_sha256','R0_checkpoint_sha256','comparison_protocol_sha256')),'OR comparison frozen endpoints/protocol')
            require(type(step) is int and 1<=step<=4000 and budget==(600 if step<=2000 else 300),'OR comparison step/budget order')
            if kind=='comparison_budget_complete':require(step in (2000,4000),'OR comparison incomplete draw budget')
            return digest([kind,auth['checkpoint_sha256'],auth['R0_checkpoint_sha256'],step,budget])
        if kind in ('evaluation_image_saved','evaluation_budget_complete'):
            auth=identity.get('evaluation_authorization')
            require(mode in ('inference','evaluation') and isinstance(auth,dict) and auth.get('images')==548 and auth.get('budgets')==[600,300] and _hex64(auth.get('checkpoint_sha256')) and payload.get('checkpoint_sha256')==auth['checkpoint_sha256'],'OR endpoint progress authority')
            if kind=='evaluation_image_saved':
                require(mode=='inference' and type(payload.get('evaluation_image_number')) is int and 1<=payload['evaluation_image_number']<=548 and _hex64(payload.get('image_id')),'OR endpoint image scope')
                return digest([kind,auth['checkpoint_sha256'],payload['evaluation_image_number']])
            require(mode=='evaluation' and payload.get('budget') in (600,300) and payload.get('name')=='top'+str(payload['budget']) and payload.get('images')==548,'OR endpoint budget scope')
            return digest([kind,auth['checkpoint_sha256'],payload['budget']])
        if kind in ('trace_image_saved','trace_image_analyzed'):
            auth=identity.get('trace_authorization');n=payload.get('image_number')
            require(mode==('trace-cache' if kind=='trace_image_saved' else 'trace-analysis') and isinstance(auth,dict),'OR trace mode/identity')
            ids=auth.get('image_ids');require(isinstance(ids,list) and len(ids)==len(set(ids))==64 and all(_hex64(v) for v in ids),'OR trace fixed identity set')
            require(type(n) is int and 1<=n<=64 and payload.get('image_id')==ids[n-1] and 'update' not in payload,'OR trace image order/no update')
            require(all(_hex64(auth.get(k)) for k in ('selection_sha256','checkpoint_sha256','trace_protocol_sha256')) and all(payload.get(k)==auth[k] for k in ('selection_sha256','checkpoint_sha256','trace_protocol_sha256')),'OR trace frozen selection/checkpoint/protocol')
            return digest([kind,auth['checkpoint_sha256'],auth['selection_sha256'],n])
        allowed={'cuda_initialized':('acceptance','formal','inference','trace-cache'),
                 'checkpoint_begin':('acceptance','formal'),'checkpoint_complete':('acceptance','formal'),
                 'update_complete':('acceptance','formal'),'epoch_begin':('formal',)}
        require(kind in allowed and mode in allowed[kind],'OR event outside authorized phase')
    if kind in ('evaluation_image_saved','evaluation_budget_complete'):
        auth=identity.get('evaluation_authorization')
        require(identity.get('task_id')=='r0-seed0-e120-20261010-001' and identity.get('arm')=='R0' and identity.get('mode') in ('inference','evaluation') and identity.get('progress_protocol')==PROTOCOL_NAME and isinstance(auth,dict),'R0 evaluation progress identity')
        require(auth.get('images')==548 and auth.get('budgets')==[600,300] and _hex64(auth.get('checkpoint_sha256')) and payload.get('checkpoint_sha256')==auth['checkpoint_sha256'],'R0 endpoint progress authority')
        if kind=='evaluation_image_saved':
            require(identity['mode']=='inference' and type(payload.get('evaluation_image_number')) is int and 1<=payload['evaluation_image_number']<=548 and _hex64(payload.get('image_id')),'R0 evaluation image scope')
            return digest([kind,auth['checkpoint_sha256'],payload['evaluation_image_number']])
        require(identity['mode']=='evaluation' and payload.get('budget') in (600,300) and payload.get('name')=='top'+str(payload['budget']) and payload.get('images')==548,'R0 evaluation budget scope')
        return digest([kind,auth['checkpoint_sha256'],payload['budget']])
    if kind=='terminal_dn_micro_backward':
        auth=identity.get('diagnostic_authorization')
        require(identity.get('mode')=='diagnostic' and identity.get('progress_protocol')==PROTOCOL_NAME and isinstance(auth,dict),'terminal DN diagnostic identity not authenticated')
        require(all(_hex64(identity.get(k)) for k in ('source_sha256','contract_sha256')) and all(_hex64(auth.get(k)) for k in ('parent_checkpoint_sha256','parent_binding_sha256','parent_contract_sha256')),'terminal DN frozen identities absent')
        require(auth.get('parent_update')==13693 and auth.get('epoch')==46 and auth.get('window')==13 and auth.get('micro_phase_sequence')==[[0,1],[1,2]],'terminal DN frozen workload differs')
        require(payload.get('diagnostic_parent_sha256')==auth['parent_checkpoint_sha256'] and 'update' not in payload,'terminal DN parent differs or update submitted')
        require(all(type(payload.get(k)) is int and payload[k]==v for k,v in (('completed_updates',13693),('epoch',46),('window',13))),'terminal DN clock/epoch/window differs')
        require(type(payload.get('microbatch')) is int and payload['microbatch'] in (0,1) and type(payload.get('phase')) is int and payload['phase']==payload['microbatch']+1,'terminal DN micro/phase differs')
        return digest([kind,auth['parent_checkpoint_sha256'],13693,46,13,payload['microbatch'],payload['phase']])
    if kind in ('forensic_layer_saved','forensic_forward_closed'):
        a=identity.get('forensic_authorization')
        require(identity.get('mode')=='limited-geometry-diagnostic' and identity.get('progress_protocol')==PROTOCOL_NAME and isinstance(a,dict),'forensic identity not authenticated')
        require(_hex64(identity.get('source_sha256')) and _hex64(identity.get('contract_sha256')) and all(_hex64(a.get(k)) for k in ('parent_checkpoint_sha256','parent_binding_sha256','parent_contract_sha256')),'forensic frozen identities absent')
        require(a.get('parent_update')==13680 and a.get('saved_boundary_update')==13693 and a.get('candidate_update')==13694 and a.get('epoch')==46 and a.get('window')==13 and a.get('microbatch')==0 and a.get('layers')==[0,1,2],'forensic frozen workload differs')
        require(payload.get('parent_checkpoint_sha256')==a['parent_checkpoint_sha256'] and 'update' not in payload,'forensic parent differs or submitted update')
        require(all(type(payload.get(k)) is int and payload[k]==v for k,v in (('completed_updates',13693),('candidate_update',13694),('epoch',46),('window',13),('microbatch',0))),'forensic clock/window/micro scope differs')
        if kind=='forensic_layer_saved':require(type(payload.get('layer')) is int and payload['layer'] in (0,1,2) and _hex64(payload.get('capture_sha256')),'forensic layer/capture identity differs')
        else:require(payload.get('outcome') in ('EXPECTED_NEGATIVE_GEOMETRY_CAPTURED','TARGET_FORWARD_COMPLETED_NOT_REPRODUCED') and _hex64(payload.get('capture_manifest_sha256')),'forensic closure outcome/manifest differs')
        return digest([kind,a['parent_checkpoint_sha256'],13693,13694,46,13,0,payload.get('layer')])
    if kind=='diagnostic_micro_backward':
        auth=identity.get('diagnostic_authorization')
        require(identity.get('mode')=='diagnostic' and identity.get('progress_protocol')==PROTOCOL_NAME and isinstance(auth,dict),'diagnostic identity not authenticated')
        require(_hex64(identity.get('source_sha256')) and _hex64(identity.get('contract_sha256')) and _hex64(auth.get('parent_checkpoint_sha256')) and _hex64(auth.get('parent_binding_sha256')) and _hex64(auth.get('parent_contract_sha256')),'diagnostic frozen source/contract/parent identity missing')
        require(auth.get('parent_update')==7977 and auth.get('micro_phase_sequence')==[[0,1],[1,2]],'diagnostic frozen parent/window scope differs')
        require(payload.get('diagnostic_parent_sha256')==auth['parent_checkpoint_sha256'],'diagnostic event parent differs from frozen identity')
        require(type(payload.get('completed_updates')) is int and payload['completed_updates']==auth['parent_update'] and 'update' not in payload,'diagnostic event submitted update or wrong clock')
        require(type(payload.get('microbatch')) is int and payload['microbatch'] in (0,1) and type(payload.get('phase')) is int and payload['phase']==payload['microbatch']+1,'diagnostic micro/phase out of scope')
        return digest([kind,auth['parent_checkpoint_sha256'],payload['completed_updates'],payload['microbatch'],payload['phase']])
    return digest([kind,payload.get('update',payload.get('completed_updates')),payload.get('epoch'),payload.get('name'),payload.get('image_number')])

class Reader:
    def __init__(self,path,identity,stream):
        require(identity.get('schema')=='f0-ssd-channel-v4' and stream in ('progress','resources'),'channel protocol version/stream differs')
        self.path=Path(path);self.identity=digest(identity);self.bound_identity=copy.deepcopy(identity);self.stream=stream
        info=self.path.stat();require(self.path.is_file() and not self.path.is_symlink(),'not a regular channel')
        self.inode=(info.st_dev,info.st_ino);self.offset=0;self.size=0;self.last=None;self.seen={};self.clock={};self.owner=None;self.tail=None;self.operations={};self.semantic=0;self.visible=None
    def trial(self):
        other=copy.copy(self);other.clock=self.clock.copy();other.seen=ChainMap({},self.seen);other.operations=ChainMap({},self.operations)
        return other
    def poll(self):
        # Transactional batch: failed reads never commit partially accepted state.
        trial=self.trial();result=trial._poll()
        self.seen.update(trial.seen.maps[0]);self.operations.update(trial.operations.maps[0])
        for key in ('offset','size','last','clock','owner','tail','semantic','visible'):setattr(self,key,getattr(trial,key))
        return result
    def _poll(self):
        info=self.path.stat()
        require((info.st_dev,info.st_ino)==self.inode,'channel replaced')
        require(info.st_size>=self.size,'channel truncated')
        with self.path.open('rb') as f:
            actual=os.fstat(f.fileno());require((actual.st_dev,actual.st_ino)==self.inode,'channel open changed')
            f.seek(self.offset);block=f.read()
        self.size=max(self.size,info.st_size,self.offset+len(block))
        end=block.rfind(b'\n')+1
        require(len(block)-end<=1024*1024,'channel partial record exceeds1MiB')
        if not end:return self.visible,False
        advanced=False
        for raw in block[:end].splitlines():
            require(len(raw)<=1024*1024,'channel record exceeds1MiB')
            require(bool(raw),'empty channel record')
            advanced=self._accept(json.loads(raw)) or advanced
        self.offset+=end
        tail=block[:end].splitlines(keepends=True)[-1]
        self.tail={'offset':self.offset-len(tail),'bytes':len(tail),'sha256':hashlib.sha256(tail).hexdigest()}
        return self.visible,advanced

    def _accept(self,record):
        record=dict(record);proof=record.pop('sha256')
        require(proof==digest(record),'channel digest mismatch')
        require(record['identity']==self.identity and record['stream']==self.stream,'channel run/stream identity conflict')
        require(type(record['writer_pid']) is int and record['writer_pid']>0 and self.owner in (None,record['writer_pid']),'channel writer changed')
        self.owner=record['writer_pid']
        seq=record['seq'];require(type(seq) is int and seq>0,'invalid channel sequence')
        record['sha256']=proof
        if seq in self.seen:
            require(self.seen[seq]==proof,'conflicting duplicate channel record')
            return False
        require(seq==(1 if self.last is None else self.last['seq']+1),'channel sequence gap/regression')
        require(record['previous']==(None if self.last is None else self.last['sha256']),'channel chain conflict')
        p=record['payload'];token=validate_event(p,self.bound_identity,self.stream)
        require(self.last is None or p['monotonic']>=self.last['payload']['monotonic'],'channel time regressed')
        for field in ('update','completed_updates','completed_epochs','image_number','evaluation_image_number','geometry_image_number','comparison_step','cpu_window_number'):
            if field in p:
                require(type(p[field]) is int and p[field]>=self.clock.get(field,0),'channel progress regressed: '+field)
                self.clock[field]=p[field]
        u=p.get('update',p.get('completed_updates'))
        if u is not None:
            require(0<=u<=36480 and ('update' not in p or 'completed_updates' not in p or p['update']==p['completed_updates']),'channel update fields conflict/out of range')
            require(u>=self.clock.get('_updates',0),'channel update clock regressed across fields')
            self.clock['_updates']=u
        if 'completed_epochs' in p:
            require(0<=p['completed_epochs']<=120,'channel epoch out of range')
            known=self.clock.get('_updates')
            require(known is None or p['completed_epochs']==known//304,'channel epochs inconsistent with update clock')
        if 'image_number' in p:require(1<=p['image_number']<=64,'channel trace scope exceeded')
        useful=True
        if self.stream=='progress':
            body=digest({k:v for k,v in p.items() if k not in ('monotonic','wall_epoch','pid')})
            if token in self.operations:
                require(self.operations[token]==body,'conflicting repeated operation')
                useful=False
            else:
                if p['event']=='update_complete' and self.bound_identity.get('task_id')=='rsmall-seed0-e120-20261012-003':
                    if self.bound_identity['mode']=='acceptance':
                        steps=[(1,'direct'),(2,'direct'),(2,'replay')];n=self.clock.get('_audit_step',0)
                        require(n<3 and (p['update'],p['audit_role'])==steps[n],'fixed3 audit progress order')
                    else:
                        require(p['audit_role']=='direct' and p['update']==self.clock.get('_audit_step',0)+1,'formal audit update order')
                    self.clock['_audit_step']=self.clock.get('_audit_step',0)+1
                if p['event']=='cpu_input_window_complete':
                    require(p['cpu_window_number']==self.clock.get('_cpu_window',0)+1,'CPU actual window order regressed or skipped')
                    self.clock['_cpu_window']=p['cpu_window_number']
                if p['event'] in ('diagnostic_micro_backward','terminal_dn_micro_backward'):
                    require(p['microbatch']==self.clock.get('_diagnostic_micro',-1)+1,'diagnostic micro/phase order regressed or skipped')
                    self.clock['_diagnostic_micro']=p['microbatch']
                if p['event']=='forensic_layer_saved':
                    require(not self.clock.get('_forensic_closed') and p['layer']==self.clock.get('_forensic_layer',-1)+1,'forensic layer order/closed scope differs')
                    self.clock['_forensic_layer']=p['layer']
                if p['event']=='forensic_forward_closed':
                    require(self.clock.get('_forensic_layer')==2 and not self.clock.get('_forensic_closed'),'forensic closure before all layers or repeated conflicting closure')
                    self.clock['_forensic_closed']=True
                self.operations[token]=body;self.semantic+=1
        self.seen[seq]=proof;self.last=record
        if useful:self.visible=record
        return useful

class Writer:
    def __init__(self,identity,stream):
        self.identity=identity;self.stream=stream;self.path=Path(identity['runtime_root'])/(stream+'.jsonl')
        self.reader=Reader(self.path,identity,stream);self.owner=os.getpid()
        require(self.path.stat().st_size==0,'channel already has a writer; no restart')
        self.sidecar=Path(identity['runtime_root'])/(stream+'-writer.json')
        self.receipt_published=False
    def append(self,payload):
        require(os.getpid()==self.owner,'channel writer PID changed')
        # Use small transactional overlays, not copies of growing histories.
        trial=self.reader.trial();trial._poll()
        base_size=self.path.stat().st_size
        require(trial.offset==base_size,'writer refuses unclosed partial channel tail')
        last=trial.last
        record={'identity':digest(self.identity),'stream':self.stream,'writer_pid':self.owner,'seq':1 if last is None else last['seq']+1,
                'previous':None if last is None else last['sha256'],'payload':payload}
        record['sha256']=digest(record);blob=canonical(record)+b'\n'
        require(len(blob)<=1024*1024,'channel serialized record exceeds1MiB before write')
        trial._accept(record) # No stream, sidecar, cache, or real cursor mutation on refusal.
        if not self.receipt_published:
            write(self.sidecar,{'pid':self.owner,'identity':digest(self.identity),'stream':self.stream})
            self.receipt_published=True
        fd=os.open(self.path,os.O_WRONLY|os.O_APPEND)
        try:
            info=os.fstat(fd);require((info.st_dev,info.st_ino)==self.reader.inode and info.st_size==base_size,'channel writer open/size changed after precheck')
            while blob:
                n=os.write(fd,blob);require(n>0,'channel append stalled');blob=blob[n:]
            os.fsync(fd)
        finally:os.close(fd)
        self.reader.poll()
        return record

def event(value):
    b=binding();require(b is not None,'runtime channel missing')
    key=(b['runtime_root'],os.getpid())
    writer=_writers.get(key)
    if writer is None:
        writer=Writer(b,'progress');result=writer.append(value);_writers[key]=writer;return result
    return writer.append(value)

def resource():
    b=binding();require(b is not None,'runtime channel missing')
    key=(b['runtime_root'],os.getpid())
    if key not in _resources:_resources[key]=Reader(Path(b['runtime_root'])/'resources.jsonl',b,'resources')
    record,_=_resources[key].poll();require(record is not None,'resource channel has no complete record')
    require(record['writer_pid']==b['supervisor_pid'],'resource publisher is not supervisor')
    return record['payload']

def latest(root,stream):
    """Locked persistent outer-observer cursor; no full-log polling."""
    import fcntl
    root=Path(root);b=read(root/'identity.json')
    state=root/('observer-'+stream+'.json')
    from .runio import atomic
    with (root/('observer-'+stream+'.lock')).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reader=Reader(root/(stream+'.jsonl'),b,stream)
        if state.exists():
            saved=read(state)
            require(saved['identity']==reader.identity and tuple(saved['inode'])==reader.inode,'observer channel identity changed')
            require(reader.path.stat().st_size>=saved['size'],'observer channel truncated')
            tail=saved['tail']
            if tail is not None:
                with reader.path.open('rb') as f:f.seek(tail['offset']);raw=f.read(tail['bytes'])
                require(hashlib.sha256(raw).hexdigest()==tail['sha256'],'observer acknowledged record changed')
            for key in ('offset','size','last','clock','owner','tail','operations','semantic','visible'):setattr(reader,key,saved[key])
            reader.seen={int(k):v for k,v in saved['seen'].items()}
        record,_=reader.poll()
        atomic(state,{k:getattr(reader,k) for k in ('identity','inode','offset','size','last','seen','clock','owner','tail','operations','semantic','visible')})
        fcntl.flock(lock,fcntl.LOCK_UN)
    return None if record is None else record['payload']

def archive(root,out):
    """Final immutable NAS mirror; never consulted by progress readers."""
    import shutil
    root=Path(root);dest=Path(out)/'runtime-archive';dest.mkdir(exist_ok=False)
    b=read(root/'identity.json')
    for stream in ('progress','resources'):
        r=Reader(root/(stream+'.jsonl'),b,stream);r.poll()
        require(r.offset==r.path.stat().st_size,'unclosed partial channel record at exit')
    for p in sorted(root.iterdir()):
        if p.name.startswith('observer-'):continue
        require(p.is_file() and not p.is_symlink(),'unexpected runtime artifact')
        shutil.copyfile(p,dest/p.name)
        require(sha(p)==sha(dest/p.name),'runtime archive SHA differs')
    index={p.name:sha(p) for p in dest.iterdir()}
    write(Path(out)/'runtime-archive-index.json',{'identity_sha256':sha(root/'identity.json'),'files':index})
    return index

def verify_archive(root,out,expected):
    root=Path(root);b=read(root/'identity.json')
    for key,value in expected.items():require(b.get(key)==value,'completed channel belongs to another run')
    names={p.name for p in root.iterdir() if not p.name.startswith('observer-')}
    mandatory={'identity.json','progress.jsonl','resources.jsonl','progress-writer.json','resources-writer.json',
               'worker.log','activity.jsonl','supervisor-start.json','worker-exit.json','supervisor-exit.json'}
    require(mandatory<=names,'closed runtime artifact missing')
    index=read(Path(out)/'runtime-archive-index.json')
    require(set(index['files'])==names and index['identity_sha256']==sha(root/'identity.json'),'closed runtime archive membership differs')
    require({p.name for p in (Path(out)/'runtime-archive').iterdir()}==names,'runtime archive extra/missing member')
    for name,proof in index['files'].items():
        require(name==Path(name).name and sha(root/name)==sha(Path(out)/'runtime-archive'/name)==proof,'closed runtime archive differs')

    for location in (root,Path(out)/'runtime-archive'):
        for stream in ('progress','resources'):
            reader=Reader(location/(stream+'.jsonl'),b,stream);reader.poll()
            require(reader.offset==reader.path.stat().st_size,'archive has an unclosed partial record')
