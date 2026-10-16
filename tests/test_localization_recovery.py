"""New recovery entry/interfaces only: mocks, no network/GPU/real images."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from sparse_rtdetr.localization import recovery as R,recovery_client as C,worker,checkpoint,guard,control
from sparse_rtdetr.localization.runio import write,read,reference,execution_binding,digest

def minimal_contract():
    return {'schema':'f0-recovery-execution-v1','task_id':R.TASK_ID,'recovery':True,'state':'FROZEN_READY','job_root':R.ROOT,
            'runtime_root':R.RUNTIME_ROOT,'attempt':'recovery-002',
            'workload':R.WORKLOAD,'parent_checkpoint':{'path':R.PARENT_PATH,'sha256':R.PARENT_SHA},
            'parent_source_sha256':R.PARENT_SOURCE,'parent_contract_sha256':R.PARENT_CONTRACT,'python':control.PYTHON_REMOTE,
            'design_sha256':R.DESIGN_SHA,'selection_sha256':R.SELECTION_SHA,'checkpoint_bytes':512*1024**2}

@pytest.mark.parametrize('field,value',[('path','C913.pth'),('sha256','0'*64)])
def test_wrong_parent_cannot_be_admitted(field,value):
    c=minimal_contract();c['parent_checkpoint'][field]=value;c['contract_sha256']=digest(c)
    with pytest.raises(RuntimeError,match='originale003'):R.validate(c)

def test_fixed_recovery_workload_and_capacity_cannot_be_lowered():
    c=minimal_contract();c['contract_sha256']=digest(c);R.validate(c)
    for key,bad in [('checkpoint_bytes',327118113),('workload',{**R.WORKLOAD,'resume_updates':35519})]:
        changed={**c,key:bad};changed['contract_sha256']=digest({k:v for k,v in changed.items() if k!='contract_sha256'})
        with pytest.raises(RuntimeError):R.validate(changed)

@pytest.mark.parametrize('action',['prepare','acceptance','resume','trace-cache','trace-analysis','status','export'])
def test_remote_recovery_cli_routes_action_without_overwriting_command(monkeypatch,capsys,action):
    calls=[]
    monkeypatch.setattr(C,'remote',lambda *a,**k:calls.append((a,k)) or {'status':'MOCK_NO_NETWORK'})
    assert control.main(['remote-recovery','--deployment','dep','--action',action,'--receipt','receipt'])==0
    assert calls==[(('dep',action,'receipt'),{'request':None})]
    assert json.loads(capsys.readouterr().out)['status']=='MOCK_NO_NETWORK'

def test_restore_uses_original_binding_after_construction_and_keeps912(tmp_path,monkeypatch):
    origin={'scope':'formal_F0_seed0_e120','model':{'original':True},'runtime':{'old':True},'config':{'old':True},'source_sha256':R.PARENT_SOURCE}
    p=tmp_path/'binding.json';write(p,origin);c={'parent_binding_file':{'path':str(p)}};calls=[]
    e=SimpleNamespace(completed_updates=0,ema=SimpleNamespace(updates=0),phase=0,failed=False)
    active={**origin,'source_sha256':'newsource'}
    monkeypatch.setattr(worker,'make_engine',lambda *a:calls.append('construct') or (e,active))
    def restore(engine,binding,path,expected):
        calls.append('restore');assert binding==origin and path==R.PARENT_PATH and expected==R.PARENT_SHA
        engine.completed_updates=engine.ema.updates=912
        return {'full_state_exact':True,'completed_updates':912}
    monkeypatch.setattr(checkpoint,'restore',restore)
    engine,binding,r=R.restored_engine(c,{},'formal_F0_seed0_e120',{})
    assert calls==['construct','restore'] and engine.completed_updates==912 and binding==active

@pytest.mark.parametrize('damage',[None,'loss','gradient','input','state'])
def test_acceptance_exact_three_updates_and_every_result_field(tmp_path,monkeypatch,damage):
    calls=[];saved={};binding={'scope':'gpu_short_window','fixture':True};ordinal=[0]
    class Engine:
        def __init__(self,u):self.completed_updates=u;self.ema=SimpleNamespace(updates=u);self.phase=0;self.failed=False
        def train_window(self,*a,**k):
            self.completed_updates+=1;self.ema.updates=self.completed_updates;ordinal[0]+=1
            calls.append(('train',self.completed_updates))
            r={'update':self.completed_updates,'elapsed_seconds':ordinal[0],'loss':1.,'gradient_sha256_before_clip':'same','output_sha256':['same'],'normalizers':[1,2]}
            if ordinal[0]==3 and damage=='loss':r['loss']=2.
            if ordinal[0]==3 and damage=='gradient':r['gradient_sha256_before_clip']='changed'
            return r
    def state(e):return {'update':e.completed_updates,'rng':'changed' if damage=='state' and ordinal[0]==3 else 'same'}
    def parent_restore(c,runtime,scope,holder):
        e=Engine(912);holder['engine']=e;return e,binding,{'full_state_exact':True}
    monkeypatch.setattr(R,'restored_engine',parent_restore)
    monkeypatch.setattr(worker,'prepare_cuda',lambda *a:{})
    monkeypatch.setattr(worker,'dataset',lambda *a:'synthetic_dataset')
    monkeypatch.setattr(worker,'plan_epoch',lambda *a:{})
    monkeypatch.setattr(worker,'live_admission',lambda *a:None)
    monkeypatch.setattr(worker,'make_engine',lambda *a:(Engine(0),binding))
    def metadata(images,targets,traces,plan,epoch,window):
        r={'epoch':epoch,'window':window,'ids':[window*16+i for i in range(16)]}
        if ordinal[0]==2 and damage=='input':r['ids'][0]=-1
        return r
    monkeypatch.setattr(worker,'check_inputs',metadata)
    def save(e,b,out,name):
        saved[str(Path(out)/name)]=state(e)
        return {'path':str(Path(out)/name),'sha256':name,'size_bytes':1}
    monkeypatch.setattr(worker,'save_state',save)
    monkeypatch.setattr(checkpoint,'read',lambda p,*a:copy.deepcopy(saved[p]))
    monkeypatch.setattr(checkpoint,'capture',lambda e,*a:state(e))
    def restore(e,b,p,*a):e.completed_updates=e.ema.updates=saved[p]['update'];return {'full_state_exact':True}
    monkeypatch.setattr(checkpoint,'restore',restore)
    def loader(data,**kwargs):
        calls.append(('loader',kwargs))
        return iter([(None,None,None)]*kwargs['window_limit'])
    from sparse_rtdetr.localization import data
    monkeypatch.setattr(data,'training_loader',loader)
    for name in ('reset_peak_memory_stats','empty_cache'):monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:1)
    if damage:
        with pytest.raises(RuntimeError,match='mismatch'):R.acceptance({},tmp_path,{})
        assert not (tmp_path/'RESULT.json').exists()
    else:
        R.acceptance({},tmp_path,{})
        r=read(tmp_path/'RESULT.json');assert r['actual_updates']==3 and not r['formal_state_usable']
    assert [x[1] for x in calls if x[0]=='train']==[913,914,914]
    assert [x[1] for x in calls if x[0]=='loader']==[{'epoch':4,'window_limit':2},{'epoch':4,'completed_windows':1,'window_limit':1}]

def test_new_launch_token_precedes_process_and_original_launch_is_never_called(tmp_path,monkeypatch):
    c={'job_root':str(tmp_path),'boot_id':'boot','gpu_uuid':'gpu','checkpoint_bytes':512*1024**2,'contract_sha256':'new','python':'inert'}
    monkeypatch.setattr(R,'ROOT',str(tmp_path));monkeypatch.setattr(R,'load',lambda p:c);monkeypatch.setattr(R,'prerequisites',lambda *a:None)
    monkeypatch.setattr(guard,'snapshot',lambda *a:{'boot_id':'boot'});monkeypatch.setattr(guard,'screen',lambda *a,**k:None)
    monkeypatch.setattr(guard,'identity',lambda pid:{'start':'owned','state':'R'})
    monkeypatch.setattr(control,'dispatch',lambda *a:pytest.fail('old formal launcher used'))
    original=tmp_path/'original-formal-launch.json';original.write_text('preserved')
    calls=[]
    def popen(command,**k):
        assert (tmp_path/'formal-launch-consumed.json').is_file()
        assert 'recovery-supervise' in command and 'supervise' not in command
        calls.append(command);return SimpleNamespace(pid=123)
    monkeypatch.setattr(R.subprocess,'Popen',popen)
    R.dispatch('inert','formal')
    with pytest.raises(FileExistsError):R.dispatch('inert','formal')
    assert len(calls)==1 and original.read_text()=='preserved'

def test_final_backup_rejects_incomplete_and_duplicate_epochs_before_any_download(tmp_path,monkeypatch):
    from sparse_rtdetr.localization import client
    monkeypatch.setattr(client,'download',lambda *a,**k:pytest.fail('unexpected transfer'))
    for items in ([],[{'epoch':0},{'epoch':0}]):
        with pytest.raises(RuntimeError):C.sync_keys({'key_checkpoints':items},tmp_path,final=True)

def test_comparison_only_omits_actual_time_not_state_or_counts():
    a={'loss':1,'elapsed_seconds':1,'normalizers':[3],'gradient':'g','input':{'id':'i'},'optimizer_steps':1}
    assert R.comparable(a)==R.comparable({**a,'elapsed_seconds':99})
    for key,value in [('loss',2),('gradient','h'),('optimizer_steps',2),('normalizers',[4]),('input',{'id':'j'})]:
        assert R.comparable(a)!=R.comparable({**a,key:value})

def completed_fixture(tmp_path,mode):
    c={'job_root':str(tmp_path),'source_sha256':'n'*64,'contract_sha256':'c'*64,'data_sha256':'d'*64,
       'selection_sha256':R.SELECTION_SHA,'gpu_uuid':'GPU-fixture','boot_id':'boot','environment':{},'config':{'seed':0},
       'parent_binding_file':{'path':str(tmp_path/'parent.json')}}
    origin={'model':{'fixture':True},'runtime':{'gpu_uuid':'GPU-fixture'}};write(tmp_path/'parent.json',origin)
    out=tmp_path/mode;out.mkdir();b={**origin,'scope':'gpu_short_window' if mode=='acceptance' else 'formal_F0_seed0_e120',
         'run_id':R.TASK_ID,'config':c['config'],'data_sha256':c['data_sha256'],'pretrained_sha256':R.PRETRAINED_SHA,
         'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
         'lineage':{'attempt':'recovery-001','parent_checkpoint_sha256':R.PARENT_SHA,'parent_source_sha256':R.PARENT_SOURCE,
                    'parent_contract_sha256':R.PARENT_CONTRACT,'acceptance_state':mode=='acceptance'}}
    write(out/'binding.json',b);write(out/'execution-binding.json',execution_binding(c,mode));write(out/'worker-exit.json',{'returned':True})
    result={'status':'PASS'}
    if mode=='formal':result.update(checkpoint={'path':str(out/'checkpoint-epoch-120.pth'),'completed_updates':36480,'scope':'formal_F0_seed0_e120'},
                                    binding=reference(out/'binding.json'),parent_sha256=R.PARENT_SHA,contract_sha256=c['contract_sha256'])
    write(out/'RESULT.json',result)
    if mode=='formal':write(out/'endpoint-lock.json',result)
    exit={'status':'PASS','completion_receipts':{p.name:reference(p) for p in out.iterdir() if p.is_file() and p.name!='endpoint-lock.json'}}
    write(out/'supervisor-exit.json',exit)
    write(tmp_path/(mode+'-dispatch.json'),{'pid':123,'process_identity':{'start':'owned','state':'R'},'contract_sha256':c['contract_sha256']})
    return c,out,b,exit

@pytest.mark.parametrize('field,value',[('scope','formal_F0_seed0_e120'),('run_id','other'),('data_sha256','changed'),('runtime',{'gpu_uuid':'other'})])
def test_completed_acceptance_rejects_wrong_state_binding_even_when_hash_bound(tmp_path,monkeypatch,field,value):
    c,out,b,exit=completed_fixture(tmp_path,'acceptance');monkeypatch.setattr(guard,'identity',lambda p:None)
    R.completed(c,'acceptance')
    (out/'binding.json').write_text(json.dumps({**b,field:value}))
    exit['completion_receipts']['binding.json']=reference(out/'binding.json')
    (out/'supervisor-exit.json').write_text(json.dumps(exit))
    with pytest.raises(RuntimeError):R.completed(c,'acceptance')

def test_completed_phase_rejects_other_contract_and_unbound_endpoint(tmp_path,monkeypatch):
    c,out,b,exit=completed_fixture(tmp_path,'formal');monkeypatch.setattr(guard,'identity',lambda p:None)
    R.completed(c,'formal')
    with pytest.raises(RuntimeError):R.completed({**c,'contract_sha256':'different'},'formal')
    (out/'endpoint-lock.json').write_text(json.dumps({**read(out/'endpoint-lock.json'),'changed':True}))
    with pytest.raises(RuntimeError,match='endpoint lock differs'):R.completed(c,'formal')

def test_returned_keys_must_equal_authenticated_epoch_lineage():
    epochs=[]
    for epoch in range(121):
        base=R.REMOTE_ROOT+'/formal' if epoch<=3 else R.ROOT+'/formal'
        epochs.append({'epoch':epoch,'checkpoint':{'path':base+f'/checkpoint-epoch-{epoch:03d}.pth','completed_updates':epoch*304,
                       'scope':'formal_F0_seed0_e120','sha256':'same','size_bytes':100,'binding_sha256':'bound'}})
    s={'key_checkpoints':[{'epoch':3,'receipt':epochs[3]['checkpoint']}]}
    C.verify_epoch_lineage({'epoch_lineage':epochs},s)
    for key,bad in [('sha256','other'),('size_bytes',101),('binding_sha256','other-binding')]:
        changed={'key_checkpoints':[{'epoch':3,'receipt':{**epochs[3]['checkpoint'],key:bad}}]}
        with pytest.raises(RuntimeError,match='authenticated epoch lineage'):C.verify_epoch_lineage({'epoch_lineage':epochs},changed)

def test_fixed64_export_checks_actual_caches_and_reports_not_only_existing_scan(tmp_path):
    selected=[{'stable_image_id':f'id{i}'} for i in range(64)];selection=tmp_path/'selection.json';write(selection,{'selected':selected})
    entries=[];reports=[]
    for i in range(64):
        cache=tmp_path/'trace-cache'/f'trace-{i:03d}.npz';cache.parent.mkdir(exist_ok=True);cache.write_bytes(b'synthetic cache bytes')
        report=tmp_path/'trace-analysis'/f'image-{i:03d}.json';write(report,{'image_id':f'id{i}'})
        entries.append({**reference(cache),'image_id':f'id{i}'});reports.append(reference(report))
    write(tmp_path/'trace-cache/RESULT.json',{'files':entries});write(tmp_path/'trace-analysis/RESULT.json',{'image_reports':reports})
    c={'job_root':str(tmp_path),'selection_path':str(selection)};R.verify_fixed_trace_evidence(c)
    missing=tmp_path/'trace-cache/trace-017.npz';missing.unlink()
    with pytest.raises((OSError,RuntimeError)):R.verify_fixed_trace_evidence(c)
