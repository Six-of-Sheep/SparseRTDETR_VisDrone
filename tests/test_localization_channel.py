"""New telemetry interfaces only; no GPU, model update or real image reads."""
import copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from sparse_rtdetr.localization import channel as C,guard,recovery as R,recovery_client as RC,worker
from sparse_rtdetr.localization.runio import canonical,digest,read,write,event

def setup(tmp_path):
    b=C.create(tmp_path/'ssd',{'contract_sha256':'contract','source_sha256':'source','attempt':'recovery-002',
                              'mode':'formal','supervisor_pid':os.getpid(),'supervisor_start':'start'})
    w=C.Writer(b,'progress');r=C.Reader(w.path,b,'progress');return b,w,r

def payload(u=913,t=1.):return {'event':'update_complete','update':u,'completed_epochs':u//304,'monotonic':t,'pid':os.getpid()}

def append(path,record):
    with path.open('ab') as f:f.write(canonical(record)+b'\n')

def test_interleaved_half_line_and_exact_duplicates_do_not_advance(tmp_path):
    b,w,r=setup(tmp_path);a=w.append(payload());assert r.poll()[1]
    append(w.path,a);assert not r.poll()[1] and r.last['seq']==1
    p=payload(914,2.);rec={'identity':digest(b),'stream':'progress','writer_pid':os.getpid(),'seq':2,'previous':a['sha256'],'payload':p}
    rec['sha256']=digest(rec);raw=canonical(rec)+b'\n';cut=len(raw)//2
    with w.path.open('ab') as f:f.write(raw[:cut])
    assert not r.poll()[1] and r.last['seq']==1
    with w.path.open('ab') as f:f.write(raw[cut:])
    assert r.poll()[1] and r.last['payload']['update']==914

@pytest.mark.parametrize('damage',['conflict','gap','regress','time','identity','writer','digest'])
def test_complete_corruption_stops(tmp_path,damage):
    b,w,r=setup(tmp_path);old=w.append(payload());r.poll()
    bad={**copy.deepcopy(old),'seq':2,'previous':old['sha256'],'payload':payload(914,2.)};bad.pop('sha256')
    if damage=='conflict':bad['seq']=1
    if damage=='gap':bad['seq']=3
    if damage=='regress':bad['payload']['update']=912
    if damage=='time':bad['payload']['monotonic']=.5
    if damage=='identity':bad['identity']='wrong'
    if damage=='writer':bad['writer_pid']+=1
    bad['sha256']=digest(bad)
    if damage=='digest':bad['sha256']='0'*64
    append(w.path,bad)
    with pytest.raises(RuntimeError):r.poll()

@pytest.mark.parametrize('damage',['missing','truncate','replace'])
def test_persistent_authoritative_channel_loss_is_fatal(tmp_path,damage):
    b,w,r=setup(tmp_path);w.append(payload());r.poll()
    if damage=='missing':w.path.unlink()
    if damage=='truncate':w.path.write_bytes(b'')
    if damage=='replace':
        p=w.path.with_suffix('.new');p.write_bytes(w.path.read_bytes());os.replace(p,w.path)
    with pytest.raises((RuntimeError,FileNotFoundError)):r.poll()

def test_duplicate_and_half_do_not_reset_semantic_timeout(tmp_path):
    b,w,r=setup(tmp_path);old=w.append(payload());r.poll();timer=guard.Progress(0)
    timer.observe(0,cpu=0,io=0,files=(),marker=r.last['seq'],gpu_busy=False)
    append(w.path,old)
    with w.path.open('ab') as f:f.write(b'{"partial":')
    assert not r.poll()[1]
    x=timer.observe(300,cpu=0,io=0,files=(),marker=r.last['seq'],gpu_busy=False)
    assert x['stop_reason']=='NO_PROGRESS_ALL_OBSERVABLES' and x['operation_seconds']==300
    y=timer.observe(1800,cpu=10000,io=20000,files=('real checkpoint',),marker=r.last['seq'],gpu_busy=True)
    assert y['stop_reason']=='SINGLE_OPERATION_NO_COMPLETION'

def test_new_sequence_same_operation_is_not_progress_but_conflicting_result_stops(tmp_path):
    b,w,r=setup(tmp_path);w.append(payload());r.poll();marker=r.semantic
    w.append(payload(913,2.));assert not r.poll()[1] and r.semantic==marker
    with pytest.raises(RuntimeError,match='conflicting repeated operation'):w.append({**payload(913,3.),'loss':9.})
    with pytest.raises(RuntimeError,match='conflicting repeated operation'):r.poll()

@pytest.mark.parametrize('bad',[{'update':913,'completed_updates':914},{'update':913,'completed_epochs':120},{'image_number':65},{'update':36481}])
def test_cross_field_and_scope_inconsistency_stops(tmp_path,bad):
    b,w,r=setup(tmp_path)
    with pytest.raises(RuntimeError):w.append({**payload(),**bad})

def test_attempt_label_only_normalization_cannot_hide_new_engine_work(tmp_path):
    from sparse_rtdetr.localization.acceptance_reuse import computation
    text="def make_engine(contract):\n binding={}\n if contract.get('recovery'):\n  binding['lineage']={'attempt':'recovery-001'}\n return binding\n"
    old=tmp_path/'old.py';new=tmp_path/'new.py';old.write_text(text)
    new.write_text(text.replace("'attempt':'recovery-001'","'attempt':contract.get('attempt', 'recovery-001')"))
    assert computation(old,'make_engine')==computation(new,'make_engine')
    new.write_text(text.replace(' return binding','  engine.train_window()\n return binding'))
    assert computation(old,'make_engine')!=computation(new,'make_engine')

def test_incomplete_archive_index_cannot_authorize_completion(tmp_path):
    b,w,r=setup(tmp_path);root=Path(b['runtime_root']);out=tmp_path/'nas';out.mkdir()
    w.append(payload());C.Writer(b,'resources').append({'monotonic':1.})
    for name in ('worker.log','activity.jsonl','supervisor-start.json','worker-exit.json','supervisor-exit.json'):write(root/name,{'test':True})
    C.archive(root,out);C.verify_archive(root,out,{'contract_sha256':'contract'})
    index=read(out/'runtime-archive-index.json');index['files'].pop('progress.jsonl')
    (out/'runtime-archive-index.json').write_text(json.dumps(index))
    with pytest.raises(RuntimeError,match='membership'):C.verify_archive(root,out,{'contract_sha256':'contract'})

def test_initial_failure_is_durable_before_nas_quarantine_attempt(tmp_path,monkeypatch):
    b=C.create(tmp_path/'ssd',{'supervisor_pid':os.getpid(),'run':'failure-fixture'});out=tmp_path/'nas';out.mkdir()
    for k,v in C.environment(b).items():monkeypatch.setenv(k,v)
    c={k:'fixture' for k in ('contract_sha256','source_sha256','data_sha256','selection_sha256','gpu_uuid','boot_id','environment')}
    c['runtime_root']=b['runtime_root']
    state=SimpleNamespace(state_dict=lambda:{})
    engine=SimpleNamespace(phase=0,completed_updates=913,model=state,optimizer=state,ema=state,device='cpu')
    engine.model=SimpleNamespace(state_dict=lambda:{},named_parameters=lambda:[])
    from sparse_rtdetr.localization import checkpoint
    monkeypatch.setattr(checkpoint,'capture_rng',lambda *a:{})
    observed=[]
    def save(*a):
        observed.append(read(Path(b['runtime_root'])/'failure.json')['type']);raise FileNotFoundError(2,'NAS quarantine unavailable','/synthetic/NAS')
    monkeypatch.setattr(worker.torch,'save',save)
    def fail(c,out,holder):holder['engine']=engine;raise FloatingPointError('synthetic numerical failure')
    with pytest.raises(FloatingPointError):worker.execute(c,'formal',out,actions={'formal':fail})
    assert observed==['FloatingPointError'] and read(Path(b['runtime_root'])/'worker-exit.json')['body_returned'] is False
    assert 'quarantine_save_error' in read(Path(b['runtime_root'])/'failure-evidence-final.json')

def test_nas_mirror_unreadability_has_no_authoritative_read_path(tmp_path,monkeypatch):
    b=C.create(tmp_path/'ssd',{'supervisor_pid':os.getpid(),'run':'test'});r=C.Reader(Path(b['runtime_root'])/'progress.jsonl',b,'progress')
    nas=tmp_path/'nas';nas.mkdir();monkeypatch.setenv(C.ENV,str(Path(b['runtime_root'])/'identity.json'))
    monkeypatch.setenv(C.ENV+'_SHA256',C.sha(Path(b['runtime_root'])/'identity.json'))
    # event owns a fresh stream, so do not claim this fixture writer separately.
    C._writers.clear();event(nas,'update_complete',update=913,completed_epochs=3)
    assert r.poll()[0]['payload']['update']==913
    assert not (nas/'progress.json').exists() and not (nas/'progress.jsonl').exists()
    with pytest.raises(FileNotFoundError):read(nas/'progress.json')
    event(nas,'update_complete',update=914,completed_epochs=3)
    assert r.poll()[0]['payload']['update']==914

def test_worker_resources_only_ssd_identity_and_staleness(tmp_path,monkeypatch):
    b,_,_=setup(tmp_path);w=C.Writer(b,'resources')
    monkeypatch.setenv(C.ENV,str(Path(b['runtime_root'])/'identity.json'));monkeypatch.setenv(C.ENV+'_SHA256',C.sha(Path(b['runtime_root'])/'identity.json'))
    from sparse_rtdetr.localization import storage
    monkeypatch.setattr(storage,'identity',lambda x:x);monkeypatch.setattr(guard,'screen',lambda *a,**k:None)
    c={'runtime_root':str(tmp_path),'boot_id':'boot','storage':{'identity':'mount'},'gpu_uuid':'gpu'}
    sample={'monotonic':worker.time.monotonic(),'boot_id':'boot','storage':'mount'};w.append(sample)
    worker.live_admission(tmp_path/'nonexistent-nas',c)
    w.append({**sample,'monotonic':sample['monotonic']+1})
    monkeypatch.setattr(worker.time,'monotonic',lambda:sample['monotonic']+30)
    with pytest.raises(RuntimeError,match='stale'):worker.live_admission(tmp_path,c)

def test_outer_incremental_reader_and_acknowledged_prefix_guard(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_UN=2,flock=lambda *a:None))
    b,w,_=setup(tmp_path);w.append(payload());root=Path(b['runtime_root'])
    assert C.latest(root,'progress')['update']==913
    state=read(root/'observer-progress.json');assert state['offset']>0
    w.append(payload(914,2.));assert C.latest(root,'progress')['update']==914
    state=read(root/'observer-progress.json')
    with w.path.open('r+b') as f:f.seek(state['tail']['offset']);f.write(b'!')
    with pytest.raises(RuntimeError,match='acknowledged'):C.latest(root,'progress')

def test_reused_gpu_acceptance_launch_is_disabled(tmp_path):
    dep=tmp_path/'dep.json';write(dep,{'task_id':R.TASK_ID,'status':'STAGED_SHA_PASS'})
    with pytest.raises(RuntimeError,match='prohibited'):RC.remote(dep,'acceptance',tmp_path/'receipt')
    assert 'acceptance' not in R.PHASES and R.WORKLOAD['acceptance_actual_updates']==0

def test_computation_unchanged_from_passed_release003():
    from sparse_rtdetr.localization.acceptance_reuse import compatibility
    old=Path('C:/Users/liu/Documents/Lzy_CV/workspace_management/delivery/experiments/localization-distribution/f0-seed0-e120-20261009-001/source-export-003/source')
    repo=Path(__file__).resolve().parents[1];proof=compatibility(old,repo)
    assert proof['new_gpu_acceptance_updates']==0 and len(proof['unchanged_dependencies'])>200

@pytest.mark.parametrize('changed',[False,True])
def test_actual_environment_extra_metadata_cannot_change_frozen_runtime(changed):
    from sparse_rtdetr.localization.acceptance_reuse import runtime_equal
    from sparse_rtdetr.localization.control import PYTHON_REMOTE
    old={'torch':'2.4.1','torchvision':'0.19.1','cuda_build':'12.4','cudnn':90100}
    actual={**old,'source_root':'new release','python':PYTHON_REMOTE,'python_version':'3.10.16','numpy':'1.26.4',
            'scipy':'1.13.1','pillow':'12.1.0','model_calls':0,'cuda_initialized':False}
    if changed:
        actual['scipy']='1.15.0'
        with pytest.raises(RuntimeError,match='runtime'):runtime_equal(old,actual)
    else:runtime_equal(old,actual)

def test_real_input_and_complete_checkpoint_absence_still_fatal(tmp_path):
    from sparse_rtdetr.localization import checkpoint
    with pytest.raises(FileNotFoundError):read(tmp_path/'actual-input.json')
    with pytest.raises(FileNotFoundError):checkpoint.read(tmp_path/'complete.pth','0'*64,{})
