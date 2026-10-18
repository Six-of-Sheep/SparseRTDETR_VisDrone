"""Independent six-view contract/channel. Legacy seed scopes remain untouched."""
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from sparse_rtdetr.localization.runio import read,require,reference,digest,sha,publish,event

TASK='sahi-sixth-view-20261018-001'
PROTOCOL='r18-sixth-view-production-v1'
PHASES=('accept-A','accept-C','accept-center','accept-flip','full-center','full-flip','evaluation')


def steps(c,mode):
    if mode=='evaluation':return ['begin']+[f'score-{v}-{b}' for v in ('C-center','C-flip') for b in (600,300)]+['complete']
    return ['begin','cuda_initialized','restored','warmup']+[f'batch-{i:03d}' for i in range(1,(1 if c['synthetic'] else 16 if mode.startswith('accept') else 137)+1)]+['state_exact','complete']


def install():
    from sparse_rtdetr.protocol_r18.control import install_resources
    install_resources()
    from sparse_rtdetr.localization import channel
    if getattr(channel,'_sixth_installed',False):return
    native=channel.validate_event;NativeReader=channel.Reader
    def validate(p,b,stream):
        if b.get('progress_protocol')!=PROTOCOL:return native(p,b,stream)
        require(b['task_id']==TASK and type(b['seed']) is int and b['seed'] in (0,1,2),'sixth task/real seed identity')
        require(all(channel._hex64(b[k]) for k in ('source_sha256','contract_sha256','checkpoint_sha256')),'sixth immutable identity')
        require(type(p.get('monotonic')) in (int,float) and math.isfinite(p['monotonic']),'sixth finite clock')
        if stream!='progress':return None
        require(p['event']=='six_step' and type(p.get('step')) is int and 0<=p['step']<len(b['steps']),'sixth event/step')
        require(p['label']==b['steps'][p['step']] and p['seed']==b['seed'] and p['checkpoint_sha256']==b['checkpoint_sha256'],'sixth phase/endpoint binding')
        return digest([p['event'],p['step']])
    class Reader(NativeReader):
        def _accept(self,r):
            ours=self.bound_identity.get('progress_protocol')==PROTOCOL and self.stream=='progress'
            if ours and r['seq'] not in self.seen:
                require(r['payload']['step']==self.clock.get('_six_next',0),'sixth missing/duplicate/out-of-order terminal')
            changed=super()._accept(r)
            if ours and changed:self.clock['_six_next']=self.clock.get('_six_next',0)+1
            return changed
    channel.validate_event=validate;channel.Reader=Reader;channel._sixth_installed=True


def load(path):
    install();c=read(path)
    require(c['task_id']==TASK and c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'sixth contract digest/task')
    require(c['source_root']==str(ROOT) and reference(c['source_manifest'])==c['source_reference'],'sixth source manifest')
    for r in read(c['source_manifest'])['files']:
        p=ROOT/r['path'];require(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()),'sixth source path')
        require(reference(p)=={'path':str(p),'bytes':r['bytes'],'sha256':r['sha256']},'sixth source changed: '+r['path'])
    require(c['source_sha256']==c['source_reference']['sha256'],'source digest binding')
    for r in (c['plan_json'],c['plan_md'],c['input_binding'],c['parent_contract']):
        require(reference(r['path'])==r,'sixth immutable input reference changed')
    require(c['training_updates']==0 and c['holdout'] is False and c['seeds']==[0,1,2],'sixth scope')
    inputs=read(c['input_binding']['path'])
    require(len(c['parents'])==3 and [p['seed'] for p in c['parents']]==[0,1,2],'all real seed parents required')
    for p,expected in zip(c['parents'],inputs['seeds']):
        require(p['seed']==expected['seed'] and p['identity']==expected['identity'] and p['original_contract']==expected['original_contract'],'sixth parent/seed/input binding')
        require(p['checkpoint']['sha256']==expected['checkpoint_sha256'] and p['fixed64']==expected['fixed64_ids'],'own checkpoint/fixed64 binding')
        require(reference(p['original_contract']['path'])==p['original_contract'],'parent original contract changed')
    require(c['synthetic'] is False or (c['GPU_authorized'] is False and c['science_authorized'] is False),'synthetic cannot authorize GPU/science')
    require(Path(c['job_root']).is_absolute() and Path(c['runtime_root']).is_absolute() and c['job_root']!=c['runtime_root'],'isolated output/runtime')
    return c


def identity(c,seed,mode):
    from sparse_rtdetr.localization import guard
    return dict(task_id=TASK,seed=seed,mode=mode,progress_protocol=PROTOCOL,
        source_sha256=c['source_sha256'],contract_sha256=c['contract_sha256'],
        checkpoint_sha256=c['parents'][seed]['checkpoint']['sha256'],
        output_root=str(Path(c['job_root'])/f'seed{seed}'/mode),
        supervisor_pid=os.getpid(),supervisor_start=guard.identity(os.getpid())['start'],steps=steps(c,mode))


class Emitter:
    def __init__(self,c,seed,mode,out):self.c,self.seed,self.mode,self.out,self.n=c,seed,mode,out,0
    def emit(self,label,**kw):
        names=steps(self.c,self.mode);require(self.n<len(names) and names[self.n]==label,'writer event order')
        event(self.out,'six_step',step=self.n,label=label,seed=self.seed,
              checkpoint_sha256=self.c['parents'][self.seed]['checkpoint']['sha256'],**kw)
        self.n+=1


def parent_context(c):
    """Call certified read-only authentication, never its launcher."""
    p=c['parent_contract'];require(reference(p['path'])==p,'parent contract SHA')
    parent=read(p['path']);source=Path(parent['source_root'])
    for r in parent['source_files']:
        require(reference(source/r['path'])=={'path':str(source/r['path']),'bytes':r['bytes'],'sha256':r['sha256']},'parent verifier source changed')
    sys.path.insert(0,str(source));import softnms_job as old
    bound=old.load_contract(p['path']);specs,rows,data_spec=old.context(bound)
    from softnms_reuse import validate_reuse
    cert=validate_reuse(bound)
    require(old.digest(specs)==cert['context_sha256'] and old.digest(rows)==cert['rows_sha256'],'parent authenticated context/rebuild evidence changed')
    sys.path.insert(0,str(ROOT))
    for s in specs:
        require(s['seed'] in (0,1,2) and s['identity']==c['parents'][s['seed']]['identity'],'parent identity conflicts')
    # context returns a role-bound data configuration, NOT parsed annotations.
    # Use the certified SHA-checking parser; never pass the configuration to GT.
    parsed=old.parsed_gt(rows,data_spec)
    from sixth_view_score_reuse import validate_annotations
    validate_annotations(rows,parsed,[r['stable_image_id'] for r in rows])
    return old,specs,rows,parsed


def live(c,out,gpu):
    from sparse_rtdetr.localization import channel,guard
    from sparse_rtdetr.localization.storage import identity as sid
    sample=channel.resource()
    require(0<=time.monotonic()-sample['monotonic']<25,'sixth stale resource')
    require(sample['boot_id']==c['boot_id'] and sid(sample['storage'])==c['storage_identity'],'sixth boot/storage identity')
    guard.screen(sample,gpu=gpu,gpu_uuid=c['gpu_uuid'],disk_bytes=40*1024**3)
    return sample
