"""Explicit one-attempt KD contract, resources and immutable stage protocol.

CPU_FIXTURE_ONLY contracts cannot initialize CUDA or consume GPU authorization.
No legacy task/channel identity is borrowed or widened.
"""
from __future__ import annotations
import copy, hashlib, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path, PurePosixPath
from ..localization.runio import (read, publish, atomic, reference, require, sha,
                                  digest, canonical)

ROOT = Path(__file__).resolve().parents[3]
TASK = 'r18-multiview-distill-20261020-001'
PROTOCOL = 'r18-distill-production-lifecycle-v1'
SCIENCE = 'configs/research/r18_multiview_distill_v1.json'
EXECUTION = 'configs/research/r18_multiview_distill_execution_v1.json'
STAGES = ('admission', 'teacher', 'accept-T0', 'accept-T1', 'accept-T2',
          'train-T0', 'train-T1', 'train-T2', 'infer-T0', 'infer-T1', 'infer-T2',
          'score', 'cost', 'compare', 'export')
GIB = 1024**3


def exact(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'}, 'explicit file reference required')
    p = Path(ref['path'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'missing/noncanonical bound file')
    require(reference(p) == ref, 'bound size/SHA differs: '+str(p))
    return read(p)


def recipe():
    return read(ROOT/SCIENCE)


def execution():
    return read(ROOT/EXECUTION)


def source_receipt(path):
    """Preparation only: complete tracked/new source, without exporting data."""
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().splitlines()
    rows = []
    for name in sorted(set(names)):
        p = ROOT/name
        require(p.is_file() and not p.is_symlink(), 'source file absent/linked: '+name)
        raw = p.read_bytes()
        rows.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    publish(path, dict(commit=commit, files=rows, scope='actual release bytes; no execution authority'))
    return reference(path)


def verify_source(ref):
    value = exact(ref); names = []
    for row in value['files']:
        name = row['path']; p = PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name, 'unsafe source member')
        target = ROOT/name
        require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(ROOT), 'source member escaped/missing')
        require(reference(target) == dict(path=str(target), bytes=row['bytes'], sha256=row['sha256']), 'actual source bytes changed: '+name)
        names.append(name)
    require(len(names) == len(set(names)) and {'r18_distill.py', SCIENCE, EXECUTION,
        'src/sparse_rtdetr/protocol_r18/distill_runtime.py',
        'src/sparse_rtdetr/protocol_r18/distill_job.py',
        'src/sparse_rtdetr/protocol_r18/distill_worker.py'} <= set(names), 'incomplete source manifest')
    return value


def profile(c):
    if c['profile'] == 'CPU_FIXTURE_ONLY':
        return dict(epochs=2, windows=1, physical=1, train_count=6,
                    eval_count=6, fixed_count=6, input_hw=(128,128), workers=0)
    require(c['profile'] == 'LAB_GPU_ONCE', 'unknown execution profile')
    return dict(epochs=12, windows=304, physical=8, train_count=4869,
                eval_count=548, fixed_count=64, input_hw=(768,1344), workers=2)


def load(path):
    require(platform.system() == 'Linux', 'final lifecycle requires POSIX evaluator/process interfaces; CPU via WSL/CI only')
    c = read(path)
    require(c['task_id'] == TASK and c['protocol'] == PROTOCOL, 'foreign task/protocol')
    require(c['contract_sha256'] == digest({k:v for k,v in c.items() if k != 'contract_sha256'}), 'contract digest differs')
    require(c['source_root'] == str(ROOT) and c['python'] == sys.executable, 'release/interpreter differs')
    require(c['science_sha256'] == digest(recipe()) and c['execution_sha256'] == digest(execution()), 'frozen recipe/execution differs')
    require(c['stages'] == list(STAGES), 'phase omission/reorder')
    require(type(c['attempt']) is str and c['attempt'].startswith('attempt-') and '/' not in c['attempt'] and '\\' not in c['attempt'], 'invalid new attempt identity')
    require(Path(c['job_root']).is_absolute() and Path(c['job_root']).name == c['attempt'], 'attempt/output identity differs')
    require(str(Path(c['job_root']).resolve())==c['job_root'],'noncanonical output path')
    require(type(c['nonce']) is str and len(c['nonce'])==32 and all(v in '0123456789abcdef' for v in c['nonce']),'new explicit nonce required')
    p = profile(c)
    source = verify_source(c['source'])
    require(source['commit'] == c['commit'], 'source commit conflict')
    if c['profile'] == 'CPU_FIXTURE_ONLY':
        require(c['GPU_authorized'] is False and c['science_authorized'] is False and c['authorization'] is None,
                'CPU fixture cannot carry production authority')
    else:
        require(platform.system() == 'Linux' and os.uname().machine == 'x86_64', 'production requires admitted lab Linux')
        require(c['GPU_authorized'] is True and c['science_authorized'] is True, 'GPU execution not authorized')
        a = exact(c['authorization'])
        require(a == dict(scope='GPU_TRAIN_EVAL_ONCE', task_id=TASK, attempt=c['attempt'],
                         commit=c['commit'], source_sha256=c['source']['sha256'],
                         science_sha256=c['science_sha256'], execution_sha256=c['execution_sha256'],
                         nonce=c['nonce']), 'separate exact-source GPU approval absent/conflicting')
        gate = exact(c['CPU_gate'])
        require(gate['status'] == 'PRODUCTION_CPU_ENTRY_PASS' and gate['source_sha256'] == c['source']['sha256'], 'final-source CPU entry gate absent')
        if 'byte_identical_parent_gate' in gate:
            parent=exact(gate['byte_identical_parent_gate']);ref=gate['byte_identical_parent_source']
            require(all(ref[k]==parent['source'][k] for k in ('bytes','sha256')),'relocated CPU source evidence differs')
            old=exact(ref)
            require(parent['status']=='PRODUCTION_CPU_ENTRY_PASS' and parent['skips']==0 and old['files']==source['files'], 'CPU evidence rebind changed source bytes')
        require(c['environment'] == execution()['environment'], 'certified lab runtime changed')
        require(c['resources'] == execution()['resources'], 'resource limits changed')
        require(Path(c['job_root']).is_relative_to('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/'+TASK), 'output not in this lab task')
    require(c['resources'] == execution()['resources'], 'resource contract differs')
    for role, count in (('train_core',p['train_count']),('development',p['eval_count'])):
        d = c['data'][role]; require(d['role'] == role and d['count'] == count, 'dataset role/count mapping differs')
    require(set(c['data']) == {'train_core','development'}, 'unauthorized data role')
    c['_path'] = str(Path(path).resolve())
    return c


def fixture_sample():
    """Schema fixture only; goes through the SAME resource screen as production."""
    dev = {k:dict(status='OBSERVED',value=v) for k,v in {
        'uuid':'CPU_FIXTURE_NO_GPU','name':'NO_GPU_SIMULATION','memory.total':24576,
        'memory.free':24576,'temperature.gpu':40,'utilization.gpu':0,
        'clocks.current.graphics':1500,'power.limit':250}.items()}
    return dict(sampled_epoch=time.time(),monotonic=time.monotonic(),boot_id='CPU_FIXTURE_BOOT',
        snapshot=dict(gpu=dict(devices=[dev]), resources=dict(cpu_count=16,ram_available=32*GIB),
                      cpu_temperature_max=40,shm_free=16*GIB),
        storage=dict(canonical_mount='CPU_FIXTURE',platform_entry='CPU_FIXTURE',mount=dict(source='FIXTURE',rw=True),
                     st_dev=0,planning_cap_bytes=200*GIB,admitted_available_bytes=100*GIB,ssd_free_bytes=10*GIB),
        memory_events={'vmstat':{'oom_kill':0}},compute_apps=[],compute_apps_status='OBSERVED',
        cpu_power_constraints=dict(constraint_0_name='long_term',constraint_1_name='short_term',
                                   constraint_0_power_limit_uw='125000000',constraint_1_power_limit_uw='150000000'),
        fixture_only=True)


def screen(sample, c, *, entry, gpu, baseline=None):
    from ..localization import lab_consumer, thermal_resume
    r = c['resources']; s = sample['snapshot']
    require(0 <= time.time()-sample['sampled_epoch'] < r['max_resource_age_seconds'], 'stale resource sample')
    require(sample['boot_id'] == c['boot_id'], 'boot identity differs')
    require(s['resources']['cpu_count'] >= r['minimum_CPUs'] and s['resources']['ram_available'] >= r['minimum_RAM_GiB']*GIB, 'CPU/RAM admission failed')
    require(s['shm_free'] >= (r['entry_shm_GiB'] if entry else r['live_shm_GiB'])*GIB, 'shared memory admission failed')
    require(sample['storage']['mount']['rw'], 'storage became readonly')
    identity = {k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')}
    require(identity == c['storage_identity'], 'storage identity differs')
    lab_consumer.screen(sample, gpu=gpu, entry=entry, gpu_uuid=c['gpu_uuid'],
                        disk_bytes=r['minimum_free_disk_GiB']*GIB,
                        baseline_events=None if baseline is None else baseline['memory_events'])
    # Preserve restored caps; do not write power/clock settings.
    if gpu:
        thermal_resume.cap_check(sample)
        require(sample['compute_apps_status']=='OBSERVED', 'compute inventory unavailable')
        require(all(a['uuid']==c['gpu_uuid'] for a in sample['compute_apps']), 'foreign GPU UUID')
    if baseline is not None:
        require(sample['memory_events']==baseline['memory_events'], 'OOM events changed')
    return sample


def snapshot(c, *, entry, gpu, baseline=None):
    if c['profile']=='CPU_FIXTURE_ONLY':
        sample=fixture_sample()
    else:
        from ..localization.thermal_resume import snapshot as observe
        target=Path(c['job_root'])
        while not target.exists():target=target.parent
        sample=observe(str(target))
        require(not sample.get('fixture_only'), 'production cannot use resource fixture')
    return screen(sample,c,entry=entry,gpu=gpu,baseline=baseline)


def identity(c,stage):
    require(stage in STAGES,'unknown stage')
    return dict(task_id=TASK,protocol=PROTOCOL,attempt=c['attempt'],stage=stage,
                source_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],
                science_sha256=c['science_sha256'],profile=c['profile'],parent_sha256=c['parent']['checkpoint']['sha256'])


def stage_path(c,stage):
    require(stage in STAGES,'unknown stage path')
    return Path(c['job_root'])/stage


def event_spec(c,stage):
    p=profile(c)
    if stage=='admission':return ['begin','training_members','development_members','complete']
    if stage=='teacher':return ['begin']+[f'batch-{i+1}' for i in range(math.ceil(p['train_count']/4))]+['cache_closed','complete']
    if stage.startswith('accept-'):return ['begin','parent_loaded','update-1','checkpoint-1','update-2','restore-1','replay-2','complete']
    if stage.startswith('train-'):
        names=['begin','parent_loaded']
        for ep in range(1,p['epochs']+1):
            names += [f'epoch-{ep}-begin']+[f'update-{(ep-1)*p["windows"]+w+1}' for w in range(p['windows'])]+[f'epoch-{ep}-saved']
        return names+['endpoint_locked','complete']
    if stage.startswith('infer-'):return ['begin']+[f'batch-{i+1}' for i in range(math.ceil(p['eval_count']/4))]+['complete']
    if stage=='score':return ['begin']+[f'score-{a}-{b}' for a in ('T0','T1','T2') for b in (600,300)]+['complete']
    if stage=='cost':return ['begin']+[f'warmup-{a}-{i}' for a in ('A','T0','T1','T2') for i in (1,2)]+[f'timed-{i+1}-{a}' for i,order in enumerate(execution()['timing_orders']) for a in order]+['complete']
    return ['begin','complete']


class Events:
    def __init__(self,c,stage,path):
        self.c,self.stage,self.path=c,stage,Path(path);self.names=event_spec(c,stage);self.cursor=0;self.previous='0'*64

    def emit(self,name,**fields):
        require(self.cursor<len(self.names) and name==self.names[self.cursor], 'missing/out-of-order/duplicate event')
        row=dict(identity=identity(self.c,self.stage),number=self.cursor+1,event=name,previous=self.previous,
                 monotonic=time.monotonic(),pid=os.getpid(),fields=fields)
        row['sha256']=digest(row)
        with (self.path/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        atomic(self.path/'progress.json',dict(event=name,monotonic=row['monotonic'],number=row['number'],sha256=row['sha256']))
        self.cursor+=1;self.previous=row['sha256']


def verify_events(c,stage,path,*,complete=True):
    names=event_spec(c,stage);previous='0'*64;records=[];last=-1.;pid=None
    raw=Path(path).read_bytes()
    require(not complete or not raw or raw.endswith(b'\n'), 'incomplete terminal event bytes')
    if not complete and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
    for line in raw.decode('utf-8').splitlines():
        r=json.loads(line);i=len(records)
        require(i<len(names) and r['event']==names[i] and r['number']==i+1 and r['identity']==identity(c,stage), 'event phase/identity/order differs')
        require(r['previous']==previous and r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}), 'event chain SHA differs')
        require(type(r['monotonic']) in (int,float) and math.isfinite(r['monotonic']) and r['monotonic']>=last, 'nonmonotonic event')
        require(type(r['pid']) is int and r['pid']>0 and pid in (None,r['pid']), 'event writer process changed')
        pid=r['pid'];last=r['monotonic'];previous=r['sha256'];records.append(r)
    require(not complete or len(records)==len(names), 'terminal event missing')
    return records


def verify_stage(c,stage):
    out=stage_path(c,stage);closed=read(out/'CLOSED.json')
    require(closed['identity']==identity(c,stage) and closed['status']=='PASS', 'stage not closed under this identity')
    value=exact(closed['result']);require(value['identity']==identity(c,stage) and value['status']=='PASS', 'result binding differs')
    require(closed['events']==reference(out/'events.jsonl'), 'closed event bytes differ')
    events=verify_events(c,stage,out/'events.jsonl')
    require(events[-1]['fields']['result']==closed['result'], 'terminal result reference differs')
    exit=exact(closed['supervisor_exit'])
    require(exit['status']=='PASS' and exit['returncode']==0 and not exit['remaining_live'] and not exit['cleanup_errors'], 'supervisor/process exit incomplete')
    worker=exact(closed['worker_exit']);require(worker['status']=='PASS' and worker['identity']==identity(c,stage), 'worker exit differs')
    return value


def seal_contract(c,path):
    value={k:v for k,v in c.items() if not k.startswith('_') and k!='contract_sha256'}
    value['contract_sha256']=digest(value);publish(path,value)
    return value
