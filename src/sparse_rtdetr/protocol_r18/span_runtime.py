"""Explicit one-attempt sampling-span contract, resources and immutable stage protocol.

CPU_FIXTURE_ONLY contracts cannot initialize CUDA or consume GPU authorization.
No legacy task/channel identity is borrowed or widened.
"""
from __future__ import annotations
import copy, hashlib, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path, PurePosixPath
from ..localization.runio import (read, publish, atomic, reference, require, sha,
                                  digest, canonical)

ROOT = Path(__file__).resolve().parents[3]
TASK = 'r18-sampling-span-20261020-001'
PROTOCOL = 'r18-sampling-span-production-v1'
ACCEPTANCE_PROTOCOL = 'r18-sampling-span-acceptance-v1'
ACCEPTANCE_CONFIG = 'configs/research/r18_sampling_span_acceptance_v1.json'
FORMAL_PROTOCOL = 'r18-sampling-span-formal-external-acceptance-v1'
FORMAL_CONFIG = 'configs/research/r18_sampling_span_formal_v1.json'
SCIENCE = 'configs/research/r18_sampling_span_seed0.json'
EXECUTION = 'configs/research/r18_sampling_span_execution_v1.json'
STAGES = ('admission', 'accept-B0', 'accept-B1', 'accept-P',
          'train-B0', 'train-B1', 'train-P', 'infer-B0', 'infer-B1', 'infer-P',
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
    require(len(names) == len(set(names)) and {'r18_sampling_span.py', 'r18_sampling_acceptance.py', 'r18_sampling_admission.py', 'r18_sampling_formal.py', FORMAL_CONFIG, SCIENCE, EXECUTION, ACCEPTANCE_CONFIG,
        'src/sparse_rtdetr/protocol_r18/span_formal.py',
        'src/sparse_rtdetr/protocol_r18/span_formal_admission.py',
        'src/sparse_rtdetr/protocol_r18/span_external_acceptance.py',
        'src/sparse_rtdetr/protocol_r18/span_admission.py',
        'src/sparse_rtdetr/protocol_r18/span_acceptance.py',
        'src/sparse_rtdetr/protocol_r18/span_acceptance_budget.py',
        'src/sparse_rtdetr/protocol_r18/span_runtime.py',
        'src/sparse_rtdetr/protocol_r18/span_job.py',
        'src/sparse_rtdetr/protocol_r18/span_worker.py',
        'src/sparse_rtdetr/protocol_r18/span_model.py',
        'src/sparse_rtdetr/protocol_r18/span_checkpoint.py',
        'src/sparse_rtdetr/protocol_r18/span_observation.py',
        'src/sparse_rtdetr/protocol_r18/span_budget.py',
        'src/sparse_rtdetr/protocol_r18/span_inputs.py',
        'src/sparse_rtdetr/protocol_r18/span_event_stream.py',
        'src/sparse_rtdetr/protocol_r18/span_B0_reuse.py',
        'src/sparse_rtdetr/research/sampling_span.py',
        'src/sparse_rtdetr/protocol_r18/cuda_identity.py'} <= set(names), 'incomplete source manifest')
    return value


def profile(c):
    if c['profile'] == 'CPU_FIXTURE_ONLY':
        return dict(epochs=2, windows=1, physical=1, train_count=6,
                    eval_count=6, fixed_count=6, input_hw=(128,128), workers=0)
    require(c['profile'] == 'LAB_GPU_ONCE', 'unknown execution profile')
    return dict(epochs=120, windows=304, physical=8, train_count=4869,
                eval_count=548, fixed_count=64, input_hw=(768,1344), workers=2)


def acceptance_config():
    return read(ROOT/ACCEPTANCE_CONFIG)


def acceptance_only(c):
    require(c['protocol'] in (PROTOCOL, ACCEPTANCE_PROTOCOL, FORMAL_PROTOCOL), 'foreign execution protocol')
    return c['protocol'] == ACCEPTANCE_PROTOCOL


def permitted_stages(c):
    if c['protocol']==FORMAL_PROTOCOL:return tuple(read(ROOT/FORMAL_CONFIG)['stages'])
    return tuple(acceptance_config()['stages']) if acceptance_only(c) else STAGES


def verify_cpu_gate(gate, source_ref, gate_status):
    """Validate original or byte-identical relocated CPU evidence, by protocol."""
    require(gate['status'] == gate_status and gate['source_sha256'] == source_ref['sha256'],
            'final-source CPU entry gate absent')
    if 'byte_identical_parent_gate' in gate:
        parent = exact(gate['byte_identical_parent_gate']); ref = gate['byte_identical_parent_source']
        require(all(ref[k] == parent['source'][k] for k in ('bytes','sha256')),
                'relocated CPU source evidence differs')
        old = exact(ref); source = exact(source_ref)
        require(parent['status'] == gate_status and parent['skips'] == 0 and old['files'] == source['files'],
                'CPU evidence rebind changed protocol/source bytes')
    return gate


def load(path, *, preparation=False, acceptance=False, formal=False):
    require(platform.system() == 'Linux', 'final lifecycle requires POSIX evaluator/process interfaces; CPU via WSL/CI only')
    c = read(path)
    require(not (acceptance and formal),'ambiguous protocol loader')
    expected_protocol = FORMAL_PROTOCOL if formal else (ACCEPTANCE_PROTOCOL if acceptance else PROTOCOL)
    require(c['task_id'] == TASK and c['protocol'] == expected_protocol, 'foreign task/protocol')
    require(c['contract_sha256'] == digest({k:v for k,v in c.items() if k != 'contract_sha256'}), 'contract digest differs')
    require(c['source_root'] == str(ROOT) and c['python'] == sys.executable, 'release/interpreter differs')
    require(c['science_sha256'] == digest(recipe()) and c['execution_sha256'] == digest(execution()), 'frozen recipe/execution differs')
    require(c['stages'] == list(permitted_stages(c)), 'phase omission/reorder')
    if acceptance:
        aconfig = acceptance_config()
        require(c['execution_scope'] == aconfig['scope'] and c['acceptance_limits'] == aconfig['limits']
                and c['acceptance_sha256'] == digest(aconfig), 'acceptance scope/limits differ')
        require(c['science_authorized'] is False, 'acceptance cannot authorize scientific execution')
    elif formal:
        policy=read(ROOT/FORMAL_CONFIG)
        require(c['execution_scope']==policy['scope'] and c['formal_sha256']==digest(policy), 'formal policy/scope differs')
        require('acceptance_limits' not in c and 'acceptance_sha256' not in c,'formal cannot carry acceptance authority')
    else:
        require('acceptance_limits' not in c and 'acceptance_sha256' not in c and
                c.get('execution_scope', 'GPU_TRAIN_EVAL_ONCE') == 'GPU_TRAIN_EVAL_ONCE', 'acceptance contract refused by full-chain entry')
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
        require(c['GPU_authorized'] is (not preparation) and c['science_authorized'] is (False if acceptance else not preparation), 'execution/preparation authority mismatch')
        if preparation:
            require(c['authorization'] is None, 'preparation cannot carry GPU authority')
        a = None if preparation else exact(c['authorization'])
        approval = dict(scope='GPU_FORMAL_E120_ONCE' if formal else ('GPU_ACCEPTANCE_ONLY' if acceptance else 'GPU_TRAIN_EVAL_ONCE'), task_id=TASK, attempt=c['attempt'],
                         commit=c['commit'], source_sha256=c['source']['sha256'],
                         science_sha256=c['science_sha256'], execution_sha256=c['execution_sha256'],
                         nonce=c['nonce'])
        if acceptance:
            approval.update(acceptance_sha256=c['acceptance_sha256'], limits=acceptance_config()['limits'])
        if formal:
            approval.update(formal_sha256=c['formal_sha256'],external_acceptance_sha256=c['external_acceptance']['sha256'],remaining_forward_calls=remaining_forward_calls(c))
            if 'external_B0' in c:approval['external_B0_sha256']=c['external_B0']['sha256']
        require(preparation or a == approval, 'separate exact-source GPU approval absent/conflicting')
        gate_status = 'FORMAL_CPU_ENTRY_PASS' if formal else ('ACCEPTANCE_CPU_ENTRY_PASS' if acceptance else 'PRODUCTION_CPU_ENTRY_PASS')
        gate = dict(status=gate_status, source_sha256=c['source']['sha256']) if preparation else exact(c['CPU_gate'])
        verify_cpu_gate(gate, c['source'], gate_status)
        require(c['environment'] == execution()['environment'], 'certified lab runtime changed')
        require(c['resources'] == execution()['resources'], 'resource limits changed')
        require(Path(c['job_root']).is_relative_to('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/'+TASK), 'output not in this lab task')
    require(c.get('seed') == 0 and type(c['seed']) is int, 'only fixed seed0 admitted')
    require(c['resources'] == execution()['resources'], 'resource contract differs')
    for role, count in (('train_core',p['train_count']),('development',p['eval_count'])):
        d = c['data'][role]; require(d['role'] == role and d['count'] == count, 'dataset role/count mapping differs')
        if c['profile']=='LAB_GPU_ONCE':
            from ..localization.material_paths import MANIFESTS,metadata_path
            require(d['manifest']['sha256']==MANIFESTS[role][0],'frozen manifest role SHA differs')
            metadata_path(role,d['manifest']['path'],MANIFESTS[role][0])
    require(set(c['data']) == {'train_core','development'}, 'unauthorized data role')
    require(c['pretrained']['sha256'] == recipe()['initialization']['ImageNet_sha256'] or c['profile']=='CPU_FIXTURE_ONLY', 'ImageNet binding differs')
    require(c['prior_cost']['path'] != c['job_root'], 'prior cost must be external evidence')
    if acceptance and c['profile']=='LAB_GPU_ONCE':
        prior=exact(c['prior_cost'])
        require(prior['task_id']==TASK and prior['profile']=='LAB_GPU_ONCE' and
            type(prior['GPU_seconds']) in (int,float) and prior['GPU_seconds']>=0 and
            type(prior['forward_calls']) is int and prior['forward_calls']>=0,
            'acceptance prior cumulative cost identity/values differ')
    require(not (c['profile']=='LAB_GPU_ONCE' and any(k.startswith(('fixture_','CPU_fixture_')) for k in c)), 'production fixture override refused')
    from .span_inputs import adapt
    before = digest(c)
    adapt(c)
    require(digest(c) == before, 'sealed contract lacks explicit input-list normalization')
    c['_path'] = str(Path(path).resolve())
    if formal:
        from .span_external_acceptance import verify
        verify(c)
        if 'external_B0' in c:
            from .span_B0_reuse import verify as verify_B0
            verify_B0(c)
    return c


def remaining_forward_calls(c):
    prior=exact(c['prior_cost'])
    expected=72978 if 'external_B0' in c else 18
    if c['profile']=='LAB_GPU_ONCE':require(prior['forward_calls']==expected,'formal prior forward count differs')
    return execution()['forward_budget']['total']-prior['forward_calls']


def reused_B0(c):return c.get('protocol')==FORMAL_PROTOCOL and 'external_B0' in c


def training_contract(c,arm):
    if arm=='B0' and reused_B0(c):
        from .span_B0_reuse import verify
        return verify(c)['parent']
    return c


def export_stage_evidence(c,stage):
    if stage=='train-B0' and reused_B0(c):
        from .span_B0_reuse import verify
        verify(c)
        return 'train-B0-REUSE.json',Path(c['external_B0']['path'])
    return stage+'-CLOSED.json',stage_path(c,stage)/'CLOSED.json'


def fixture_sample():
    """Schema fixture only; goes through the SAME resource screen as production."""
    began = dict(epoch=time.time(), monotonic=time.monotonic())
    dev = {k:dict(status='OBSERVED',value=v) for k,v in {
        'uuid':'CPU_FIXTURE_NO_GPU','name':'NO_GPU_SIMULATION','memory.total':24576,
        'memory.free':24576,'temperature.gpu':40,'utilization.gpu':0,
        'clocks.current.graphics':1500,'power.limit':250}.items()}
    sample = dict(sampled_epoch=time.time(),monotonic=time.monotonic(),boot_id='CPU_FIXTURE_BOOT',
        snapshot=dict(gpu=dict(devices=[dev]), resources=dict(cpu_count=16,ram_available=32*GIB),
                      cpu_temperature_max=40,shm_free=16*GIB),
        storage=dict(canonical_mount='CPU_FIXTURE',platform_entry='CPU_FIXTURE',mount=dict(source='FIXTURE',rw=True),
                     st_dev=0,planning_cap_bytes=200*GIB,admitted_available_bytes=100*GIB,ssd_free_bytes=10*GIB),
        memory_events={'vmstat':{'oom_kill':0}},compute_apps=[],compute_apps_status='OBSERVED',
        cpu_power_constraints=dict(constraint_0_name='long_term',constraint_1_name='short_term',
                                   constraint_0_power_limit_uw='125000000',constraint_1_power_limit_uw='150000000'),
        fixture_only=True)
    sample['sampling_timing'] = dict(started=began,
        completed=dict(epoch=time.time(),monotonic=time.monotonic()),
        source='fixture_sample: actual Python time.time/CLOCK_REALTIME and time.monotonic/CLOCK_MONOTONIC')
    return sample


def resource_context(c, stage=None):
    """Explicit stage argument, or the actual worker/supervisor CLI stage."""
    if stage is None and '--stage' in sys.argv:
        i=sys.argv.index('--stage')
        stage=sys.argv[i+1] if i+1<len(sys.argv) else None
    if stage is not None:require(stage in permitted_stages(c),'resource diagnostic stage outside contract')
    return dict(task_id=c.get('task_id'),protocol=c.get('protocol'),attempt=c.get('attempt'),
        contract_sha256=c.get('contract_sha256'),source_sha256=c.get('source',{}).get('sha256'),
        stage=stage,action=sys.argv[1] if len(sys.argv)>1 else None,pid=os.getpid())


def resource_age(sample,c,stage=None):
    """Observe the original wall-clock predicate; never replace a sample time.

    Monotonic values diagnose duration/clock disagreement, not an alternate
    permission to admit negative or expired wall-clock ages.
    """
    checked=dict(epoch=time.time(),monotonic=time.monotonic())
    timing=sample.get('sampling_timing',{})
    start=timing.get('started');finish=timing.get('completed')
    return dict(sampled_epoch=sample['sampled_epoch'],sampled_monotonic=sample.get('monotonic'),
        sampling_started=start,sampling_completed=finish,checked=checked,
        actual_age_seconds=checked['epoch']-sample['sampled_epoch'],
        monotonic_age_seconds=None if 'monotonic' not in sample else checked['monotonic']-sample['monotonic'],
        sampling_elapsed_seconds=None if start is None or finish is None else finish['monotonic']-start['monotonic'],
        threshold_seconds=c['resources']['max_resource_age_seconds'],
        clock_sources=dict(age='Python time.time / CLOCK_REALTIME',duration='Python time.monotonic / CLOCK_MONOTONIC'),
        sampling_time_source=timing.get('source','legacy sample: collection boundaries unavailable'),
        phase_identity=resource_context(c,stage),sampling_phase_identity=sample.get('sampling_phase_identity'))


def reject_resource_age(sample,c,record):
    """Persist the rejected sample before raising; never overwrite a failure."""
    root=Path(c['job_root']);stage=record['phase_identity']['stage']
    if root.is_dir():
        folder=root/stage if stage is not None and (root/stage).is_dir() else root
    else:
        # Do not consume the future job directory before run's exist_ok=False.
        folder=root.with_name(root.name+'-resource-rejections')
        folder.mkdir(parents=True,exist_ok=True)
    age=record['actual_age_seconds']
    publish(folder/('resource-age-rejected-'+str(os.getpid())+'-'+str(time.monotonic_ns())+'.json'),
        dict(status='REJECTED_RESOURCE_AGE_NO_RETRY',
             reason='negative age' if age<0 else ('nonfinite age' if not math.isfinite(age) else 'expired sample'),
             age=record,sample=sample))


def screen(sample, c, *, entry, gpu, baseline=None, stage=None):
    from ..localization import lab_consumer, thermal_resume
    require(isinstance(c.get('resources'),dict) and c['resources']==execution()['resources'],
            'resource screen requires constructed contract with frozen resources')
    r = c['resources']; s = sample['snapshot']
    age=resource_age(sample,c,stage)
    if not 0 <= age['actual_age_seconds'] < r['max_resource_age_seconds']:
        reject_resource_age(sample,c,age)
        raise RuntimeError('stale resource sample')
    sample['age_check']=age
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


def snapshot(c, *, entry, gpu, baseline=None, stage=None):
    began=dict(epoch=time.time(),monotonic=time.monotonic())
    if c['profile']=='CPU_FIXTURE_ONLY':
        sample=fixture_sample()
    else:
        from ..localization.thermal_resume import snapshot as observe
        target=Path(c['job_root'])
        while not target.exists():target=target.parent
        sample=observe(str(target))
        require(not sample.get('fixture_only'), 'production cannot use resource fixture')
    sample['sampling_timing']=dict(started=began,
        completed=dict(epoch=time.time(),monotonic=time.monotonic()),
        source='snapshot collection boundaries: actual CLOCK_REALTIME/CLOCK_MONOTONIC; sampled_epoch preserved from collector')
    sample['sampling_phase_identity']=resource_context(c,stage)
    return screen(sample,c,entry=entry,gpu=gpu,baseline=baseline,stage=stage)


def identity(c,stage):
    if c['protocol']==FORMAL_PROTOCOL and stage.startswith('accept-'):
        from .span_external_acceptance import parent
        return identity(parent(c),stage)
    require(stage in permitted_stages(c),'stage outside contract scope')
    return dict(task_id=TASK,protocol=c['protocol'],attempt=c['attempt'],stage=stage,
                source_sha256=c['source']['sha256'],contract_sha256=c['contract_sha256'],
                science_sha256=c['science_sha256'],profile=c['profile'],initialization_sha256=c['pretrained']['sha256'])


def stage_path(c,stage):
    if c['protocol']==FORMAL_PROTOCOL and stage.startswith('accept-'):
        from .span_external_acceptance import parent
        return stage_path(parent(c),stage)
    if stage=='train-B0' and reused_B0(c):return Path(training_contract(c,'B0')['job_root'])/stage
    require(stage in permitted_stages(c),'stage outside contract scope')
    return Path(c['job_root'])/stage


def event_spec(c,stage):
    p=profile(c)
    if stage=='admission':return ['begin','training_members','development_members','complete']
    if stage.startswith('accept-'):return ['begin','initial_loaded','update-1','checkpoint-1','update-2','restore-1','replay-2','complete']
    if stage.startswith('train-'):
        names=['begin','initial_loaded']
        for ep in range(1,p['epochs']+1):
            names += [f'epoch-{ep}-begin']+[f'update-{(ep-1)*p["windows"]+w+1}' for w in range(p['windows'])]+[f'epoch-{ep}-saved']
        return names+['endpoint_locked','complete']
    if stage.startswith('infer-'):return ['begin']+[f'batch-{i+1}' for i in range(math.ceil(p['eval_count']/4))]+['complete']
    if stage=='score':return ['begin']+[f'score-{a}-{b}' for a in ('B0','B1','P') for b in (600,300)]+['complete']
    if stage=='cost':return ['begin']+[f'warmup-{a}-{i}' for a in ('A','B0','B1','P') for i in (1,2)]+[f'timed-{i+1}-{a}' for i,order in enumerate(execution()['timing_orders']) for a in order]+['complete']
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
    from .span_event_stream import VerifiedEvents
    return VerifiedEvents(c,stage,path,complete)


class EventReader:
    """Incremental live verification; terminal closure still verifies all bytes.

    Only complete appended lines advance the cursor. Missing/replaced/truncated
    journals fail after first observation; incomplete tails are never progress.
    """
    def __init__(self,c,stage,path):
        self.c,self.stage,self.path=c,stage,Path(path)
        self.names=event_spec(c,stage);self.offset=0;self.count=0
        self.previous='0'*64;self.clock=-1.;self.pid=None;self.file_id=None

    def poll(self,writer):
        if not self.path.exists():
            require(self.file_id is None,'previously observed event journal disappeared')
            return None
        with self.path.open('rb') as stream:
            info=os.fstat(stream.fileno());file_id=(info.st_dev,info.st_ino)
            require(self.file_id in (None,file_id),'event journal replaced')
            require(info.st_size>=self.offset,'event journal truncated')
            self.file_id=file_id;stream.seek(self.offset);raw=stream.read()
        end=raw.rfind(b'\n')+1
        for line in raw[:end].decode('utf-8').splitlines():
            r=json.loads(line);i=self.count
            require(i<len(self.names) and r['event']==self.names[i] and r['number']==i+1
                    and r['identity']==identity(self.c,self.stage),'event phase/identity/order differs')
            require(r['previous']==self.previous and r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}),'event chain SHA differs')
            require(type(r['monotonic']) in (int,float) and math.isfinite(r['monotonic'])
                    and r['monotonic']>=self.clock,'nonmonotonic event')
            require(type(r['pid']) is int and r['pid']>0 and self.pid in (None,r['pid'])
                    and writer==r['pid'],'foreign event writer')
            self.count+=1;self.previous=r['sha256'];self.clock=r['monotonic'];self.pid=r['pid']
        self.offset+=end
        return None if not self.count else dict(number=self.count,sha256=self.previous)


def verify_stage(c,stage):
    if c['protocol']==FORMAL_PROTOCOL and stage.startswith('accept-'):
        from .span_external_acceptance import parent
        return verify_stage(parent(c),stage)
    if stage=='train-B0' and reused_B0(c):
        from .span_B0_reuse import verify
        return verify(c)['result']
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
