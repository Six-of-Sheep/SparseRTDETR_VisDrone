"""Synthetic failure-path checks; no SSH, GPU or real image/annotation reads."""
from __future__ import annotations
import copy,io,json,signal,tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from sparse_rtdetr.localization import client,control,guard,worker,material_paths,local_ready
from sparse_rtdetr.localization.runio import (TASK_ID,KEY_EPOCHS,SHORT_STATES,PHASES,PRETRAINED_SHA,
    SELECTION_SHA,reference,read,write,sha,execution_binding,original_process_exited)


def contract(root):
    return {'job_root':str(root),'contract_sha256':'c'*64,'source_sha256':'a'*64,'data_sha256':'d'*64,
            'selection_sha256':SELECTION_SHA,'gpu_uuid':'GPU-fixture','boot_id':'boot-fixture',
            'environment':{'torch':'fixture'},'config':{'seed':0},'python':'never-executed'}


def completed_phase(root,c,phase):
    folder=root/phase;folder.mkdir()
    write(folder/'execution-binding.json',execution_binding(c,phase))
    result={'status':'PASS','unique_updates':6,'actual_updates':7,'same_next_state_exact':True,
            'same_next_window_exact':True,'checkpoint_bytes':1024}
    write(folder/'RESULT.json',result);write(folder/'worker-exit.json',{'body_returned':True})
    names=['execution-binding.json','RESULT.json','worker-exit.json']
    if phase in ('short','formal'):
        binding={'run_id':TASK_ID,'scope':{'short':'gpu_short_window','formal':'formal_F0_seed0_e120'}[phase],
                 'pretrained_sha256':PRETRAINED_SHA,'runtime':{**c['environment'],'gpu_uuid':c['gpu_uuid']},
                 **{k:c[k] for k in ('contract_sha256','source_sha256','data_sha256','config')}}
        write(folder/'binding.json',binding);names.append('binding.json')
    write(folder/'supervisor-exit.json',{'status':'PASS','completion_receipts':{n:reference(folder/n) for n in names}})
    write(root/(phase+'-dispatch.json'),{'pid':123,'process_identity':{'start':'12345','state':'R'}})


def test_prior_contract_and_live_supervisor_cannot_release_formal(tmp_path,monkeypatch):
    c=contract(tmp_path);completed_phase(tmp_path,c,'short')
    write(tmp_path/'windows-backup-ready.json',{'task_id':TASK_ID,'source_sha256':c['source_sha256'],
          'epochs':list(KEY_EPOCHS),'observed_epoch':control.time.time(),'free_bytes':10*1024**3})
    monkeypatch.setattr(guard,'identity',lambda pid:None)
    control.prerequisites(c,'formal')
    for key in ('source_sha256','contract_sha256','data_sha256'):
        with pytest.raises(RuntimeError,match='contract/source/data/runtime'):
            control.prerequisites({**c,key:'e'*64},'formal')
    monkeypatch.setattr(guard,'identity',lambda pid:{'start':'12345','state':'S'})
    with pytest.raises(RuntimeError,match='still running'):control.prerequisites(c,'formal')
    for current in (None,{'start':'12345','state':'Z'},{'start':'99999','state':'R'}):
        assert original_process_exited(read(tmp_path/'short-dispatch.json'),current)
    with pytest.raises(RuntimeError):original_process_exited({'process_identity':None},None)


def test_launch_token_precedes_process_and_cannot_be_reused(tmp_path,monkeypatch):
    c=contract(tmp_path);calls=[]
    monkeypatch.setattr(control,'load_contract',lambda p:c)
    monkeypatch.setattr(control,'prerequisites',lambda *a:None)
    monkeypatch.setattr(guard,'snapshot',lambda p:{'boot_id':c['boot_id']})
    monkeypatch.setattr(guard,'screen',lambda *a,**k:None)
    def launch(*args,**kwargs):
        calls.append(args)
        assert (tmp_path/'short-launch-consumed.json').is_file()
        raise OSError('synthetic process creation failure')
    monkeypatch.setattr(control.subprocess,'Popen',launch)
    with pytest.raises(OSError,match='synthetic'):control.dispatch('unused','short')
    token_sha=sha(tmp_path/'short-launch-consumed.json')
    with pytest.raises(FileExistsError):control.dispatch('unused','short')
    assert len(calls)==1 and sha(tmp_path/'short-launch-consumed.json')==token_sha


def test_remote_cli_is_inert_and_preserves_explicit_arguments(monkeypatch,capsys):
    seen=[]
    monkeypatch.setattr(client,'remote_call',lambda *a,**k:seen.append((a,k)) or {'status':'MOCK_NO_NETWORK'})
    assert control.main(['remote-call','--deployment','dep','--action','launch','--mode','short',
                         '--receipt','receipt','--contract','contract'])==0
    assert seen==[(('dep','launch','receipt'),{'mode':'short','request':None,'contract':'contract'})]
    assert json.loads(capsys.readouterr().out)['status']=='MOCK_NO_NETWORK'


def test_progress_uses_activity_and_semantic_completion_without_total_cap():
    p=guard.Progress(0)
    p.observe(0,cpu=0,io=0,files=(),marker=('load',0),gpu_busy=False)
    assert p.observe(299,cpu=0,io=0,files=(),marker=('load',0),gpu_busy=False)['stop_reason'] is None
    assert p.observe(300,cpu=0,io=0,files=(),marker=('load',0),gpu_busy=False)['stop_reason']=='NO_PROGRESS_ALL_OBSERVABLES'
    p=guard.Progress(0)
    for t in (0,250,500,1000,1500,1799):
        assert p.observe(t,cpu=t,io=t,files=(t,),marker=('load',0),gpu_busy=True)['stop_reason'] is None
    assert p.observe(1800,cpu=1800,io=1800,files=(1800,),marker=('load',0),gpu_busy=True)['stop_reason']=='SINGLE_OPERATION_NO_COMPLETION'
    p=guard.Progress(0)
    for t in range(0,10001,20):
        assert p.observe(t,cpu=t,io=t,files=(t,),marker=('update',t),gpu_busy=True)['stop_reason'] is None


@pytest.mark.parametrize('exits_after_term',[None,2.])
def test_guard_proc_failure_still_gives_owned_child_term_grace(tmp_path,monkeypatch,exits_after_term):
    result,signals=_guard_fixture(tmp_path,monkeypatch,kind='proc-fails',exits_after_term=exits_after_term)
    assert result['status']=='STOP' and result['process_table_cleanup_uncertain']
    assert signals[0]==('TERM',0.)
    kills=[t for kind,t in signals if kind=='KILL']
    if exits_after_term is None:assert len(kills)==1 and kills[0]>=8.
    else:assert not kills
    assert not result['remaining_live'] and not result['unknown_pids']


def test_guard_normal_helper_drain_uses_no_signal(tmp_path,monkeypatch):
    result,signals=_guard_fixture(tmp_path,monkeypatch,kind='natural-drain')
    assert result['status']=='PASS' and result['residual_at_leader_exit']==[124]
    assert result['residual_after_natural_drain']==[] and .2<=result['natural_drain_seconds']<1
    assert signals==[] and set(result['completion_receipts'])=={'RESULT.json','execution-binding.json','worker-exit.json'}


def test_guard_late_identity_failure_does_not_skip_term(tmp_path,monkeypatch):
    result,signals=_guard_fixture(tmp_path,monkeypatch,kind='identity-fails')
    assert result['status']=='STOP'
    assert signals[0]==('TERM',0.)
    assert next(t for s,t in signals if s=='KILL')>=8.


def _guard_fixture(root,monkeypatch,*,kind,exits_after_term=None):
    clock=SimpleNamespace(now=0.)
    clock.monotonic=lambda:clock.now;clock.time=lambda:1000.+clock.now
    clock.sleep=lambda dt:setattr(clock,'now',clock.now+dt)
    signals=[];state={'polls':0,'snapshots':0}
    class Child:
        pid=123;returncode=None;term=None
        def poll(self):
            if kind=='natural-drain':self.returncode=0
            if self.term is not None and exits_after_term is not None and clock.now-self.term>=exits_after_term:self.returncode=0
            return self.returncode
        def terminate(self):signals.append(('TERM',clock.now));self.term=clock.now
        def kill(self):signals.append(('KILL',clock.now));self.returncode=-9
        def wait(self,timeout):
            assert self.poll() is not None
            return self.returncode
    child=Child()
    def popen(*args,**kwargs):
        for name in ('RESULT.json','execution-binding.json','worker-exit.json'):write(root/'phase'/name,{'synthetic':True})
        if kind=='progress-gone':write(root/'phase/progress.json',{'event':'load','monotonic':0,'completed_epochs':0})
        return child
    def proc():
        if kind=='proc-fails':raise PermissionError('synthetic /proc unavailable')
        if kind=='progress-gone' and clock.now>=2:
            (root/'phase/progress.json').unlink(missing_ok=True)
            if clock.now>=4:child.returncode=0
        if child.poll() is not None:
            return {124:{'ppid':999,'start':'helper','state':'S','cpu':0}} if kind=='natural-drain' and clock.now<.2 else {}
        return {123:{'ppid':999,'start':'child','state':'R','cpu':0}}
    def snapshot(*args):
        state['snapshots']+=1
        if kind=='identity-fails' and state['snapshots']>1:raise PermissionError('synthetic monitor failure')
        if kind=='missing-file' and state['snapshots']>1:raise FileNotFoundError(2,'synthetic missing resource path','/synthetic/missing')
        return {'boot_id':'fixture','memory_events':{'fixture':{'oom':0}},'snapshot':{'gpu':{'devices':[]}},
                'storage':{'canonical_mount':'fixture','platform_entry':'fixture','mount':{'source':'fixture'},'st_dev':1,'planning_cap_bytes':1}}
    def ident(pid):
        if kind=='proc-fails':raise PermissionError('identity unavailable')
        return {'start':'child','state':'R'}
    def owned(pid,start,sig):
        if kind=='identity-fails':raise PermissionError('identity failed after the cleanup scan')
        signals.append(('UNEXPECTED_PIDFD',clock.now))
    def no_children(*args):raise ChildProcessError()
    monkeypatch.setattr(guard,'time',clock)
    monkeypatch.setattr(guard,'ctypes',SimpleNamespace(CDLL=lambda *a,**k:SimpleNamespace(prctl=lambda *a:0)))
    monkeypatch.setattr(guard,'os',SimpleNamespace(getpid=lambda:999,waitpid=no_children,WNOHANG=1))
    monkeypatch.setattr(guard,'signal',SimpleNamespace(SIGHUP=1,SIG_IGN=0,SIGINT=2,SIGTERM=15,SIGKILL=9,signal=lambda *a:None))
    monkeypatch.setattr(guard,'subprocess',SimpleNamespace(Popen=popen,DEVNULL=-3,STDOUT=-2,TimeoutExpired=RuntimeError))
    monkeypatch.setattr(guard,'processes',proc);monkeypatch.setattr(guard,'identity',ident)
    monkeypatch.setattr(guard,'snapshot',snapshot);monkeypatch.setattr(guard,'screen',lambda *a,**k:None)
    monkeypatch.setattr(guard,'signal_owned',owned)
    # Proc IO is synthetic too; no /proc access can occur on this Windows host.
    original_read=Path.read_text
    def read_text(path,*args,**kwargs):
        if str(path).replace('\\','/').endswith('/123/io'):return 'read_bytes: 0\nwrite_bytes: 0\n'
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    guard.supervise(['never-executed'],root/'phase',{},gpu_uuid='fixture',boot_id='fixture',gpu=False)
    return read(root/'phase/supervisor-exit.json'),signals


def test_guard_retains_missing_path_traceback_and_operation(tmp_path,monkeypatch):
    result,signals=_guard_fixture(tmp_path,monkeypatch,kind='missing-file',exits_after_term=2.)
    assert result['status']=='STOP' and result['returncode']==0
    error=result['guard_error']
    assert error['type']=='FileNotFoundError' and error['errno']==2 and error['filename']=='/synthetic/missing'
    assert error['operation']=='resource_snapshot' and 'snapshot' in error['traceback']
    assert error['received_signals']==[] and result['received_signals']==[]
    assert signals[0]==('TERM',0.) # Protection still stops its own worker; no retry or relaxed gate.


def test_atomic_publication_and_disappearing_partial(tmp_path,monkeypatch):
    from sparse_rtdetr.localization.runio import publish
    p=tmp_path/'receipt.json';publish(p,{'value':1});before=sha(p)
    with pytest.raises(FileExistsError):publish(p,{'value':2})
    assert sha(p)==before and read(p)=={'value':1}
    partial=tmp_path/'one.pth.partial';partial.write_bytes(b'synthetic')
    published=tmp_path/'one.pth';published.write_bytes(b'published')
    original_stat=Path.stat
    def stat(path,*args,**kwargs):
        if path==partial:raise FileNotFoundError('publication removed its partial name')
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',stat)
    rows=guard.file_activity(tmp_path)
    assert len(rows)==1 and rows[0][:2]==('one.pth',len(b'published'))


def test_finite_sampler_never_issues_out_of_window_inputs():
    from sparse_rtdetr.localization.data import WindowSampler
    direct=list(WindowSampler(4869,seed=0,epoch=1,completed_windows=0,window_limit=6))
    replay=list(WindowSampler(4869,seed=0,epoch=1,completed_windows=5,window_limit=1))
    assert len(direct)==6 and sum(map(len,direct))==96 and replay==[direct[-1]]
    assert [p for batch in direct for _,p,_ in batch]==list(range(96))
    for limit in (0,305,-1):
        with pytest.raises(ValueError):WindowSampler(4869,seed=0,epoch=1,completed_windows=0,window_limit=limit)


def test_metadata_forbidden_or_wrong_role_rejected_before_read(tmp_path,monkeypatch):
    reads=[]
    monkeypatch.setattr(Path,'read_bytes',lambda p:reads.append(str(p)) or b'not permitted')
    for kind,p,h in [('development',tmp_path/'holdout/development_manifest.json',material_paths.MANIFESTS['development'][0]),
                     ('train_core',tmp_path/'train_core_manifest.json','0'*64),
                     ('selection',tmp_path/'test-dev/trace-selection.json',SELECTION_SHA),
                     ('development',tmp_path/'unknown/development_manifest.json',material_paths.MANIFESTS['development'][0])]:
        with pytest.raises(RuntimeError):material_paths.metadata_path(kind,p,h).read_bytes()
    assert reads==[]


def _endpoint(root,updates=36480):
    folder=root/'formal';folder.mkdir();write(folder/'binding.json',{})
    return {'status':'F0_E120_EMA_LOCKED','epoch':120,'updates':36480,'binding':reference(folder/'binding.json'),
            'checkpoint':{'path':str(folder/'checkpoint-epoch-120.pth'),'scope':'formal_F0_seed0_e120',
                          'completed_updates':updates,'sha256':'f'*64}}


@pytest.mark.parametrize('wrong_receipt',[True,False])
def test_trace_rejects_early_checkpoint_before_any_forward(tmp_path,monkeypatch,wrong_receipt):
    c=contract(tmp_path);end=_endpoint(tmp_path,9120 if wrong_receipt else 36480)
    write(tmp_path/'formal/endpoint-lock.json',end);calls=[]
    engine=SimpleNamespace(completed_updates=9120,ema=SimpleNamespace(updates=9120))
    monkeypatch.setattr(worker,'prepare_cuda',lambda *a:calls.append('cuda_stub') or {})
    monkeypatch.setattr(worker,'make_engine',lambda *a:(engine,{}))
    monkeypatch.setattr(worker.checkpoint,'restore',lambda *a:calls.append('restore_stub'))
    monkeypatch.setattr(worker,'dataset',lambda *a:pytest.fail('data/model accessed for wrong endpoint'))
    with pytest.raises(RuntimeError,match='not.*e120'):worker.run_trace_cache(c,tmp_path,{})
    assert calls==([] if wrong_receipt else ['cuda_stub','restore_stub'])


def test_analysis_cannot_accept_self_consistent_other_endpoint_cache(tmp_path,monkeypatch):
    c=contract(tmp_path);end=_endpoint(tmp_path);write(tmp_path/'formal/endpoint-lock.json',end)
    write(tmp_path/'trace-cache/RESULT.json',{'status':'PASS','images':64,'checkpoint_sha256':'e'*64,'selection_sha256':SELECTION_SHA})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(worker,'dataset',lambda *a:SimpleNamespace(records=[]))
    with pytest.raises(RuntimeError,match='another endpoint'):worker.run_trace_analysis(c,tmp_path,{})


def _tar(path,contents):
    with tarfile.open(path,'w:gz') as tar:
        for name,raw in contents:
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))


@pytest.mark.parametrize('names',[['../outside'],['C:/outside'],['..\\outside'],['one','one']])
def test_unsafe_archives_rejected_before_creating_destination(tmp_path,names):
    archive=tmp_path/'bad.tar.gz';_tar(archive,[(n,b'x') for n in names])
    with pytest.raises(RuntimeError):client.safe_extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_raw_export_uses_exact_closed_members_without_reading_extra(tmp_path,monkeypatch):
    c=contract(tmp_path);c['data']={'synthetic':'roles'};records={};closure=[]
    for role,n in [('train_core',4869),('development',64)]:
        folder=tmp_path/'materials'/role/'annotations';folder.mkdir(parents=True)
        rows=[]
        for i in range(n):
            r={'stable_image_id':role+str(i),'annotation_relative_path':f'annotations/{i}.txt','annotation_size_bytes':1,'annotation_sha256':'e'*64}
            rows.append(r);closure.append({'role':role,'image_id':r['stable_image_id'],'annotation':{'path':str(folder/f'{i}.txt'),'bytes':1,'sha256':'e'*64}})
        records[role]=SimpleNamespace(records=rows,annotations_path=folder)
    extra=tmp_path/'materials/development/annotations/extra.txt';extra.write_text('must not read')
    p=tmp_path/'raw-preflight/RESULT.json';write(p,{'status':'PASS','raw_files':4933,'data_binding':c['data'],'files':closure})
    c['raw_closure']={'path':str(p)};c['data_sha256']=sha(p);seen=[]
    monkeypatch.setattr(worker,'dataset',lambda c,r:records[r])
    def observed_sha(path):seen.append(Path(path));assert Path(path)!=extra;return sha(path)
    monkeypatch.setattr(control,'sha',observed_sha)
    allowed=control.raw_export_files(c)
    assert len(allowed)==4933 and extra not in allowed and seen==[p]


def test_evidence_export_not_ready_leaves_no_directory_and_complete_is_reusable(tmp_path,monkeypatch):
    c=contract(tmp_path);c.update(source_manifest=str(tmp_path/'release/source-manifest.json'),local_checks={'path':str(tmp_path/'release/local-checks/LOCAL_READY.json')})
    write(tmp_path/'contract.json',c);write(c['source_manifest'],{'synthetic':True});write(c['local_checks']['path'],{'evidence_files':[]})
    for phase in PHASES:(tmp_path/phase).mkdir()
    ready={'value':False}
    def phase_check(*args):
        if not ready['value']:raise RuntimeError('not ready')
    monkeypatch.setattr(control,'REMOTE_ROOT',str(tmp_path));monkeypatch.setattr(control,'validate_contract',lambda c:None)
    monkeypatch.setattr(control,'phase_complete',phase_check);monkeypatch.setattr(control,'verify_local_checks',lambda c:None)
    monkeypatch.setattr(control,'raw_export_files',lambda c:[])
    with pytest.raises(RuntimeError,match='not ready'):control.evidence_export()
    assert not (tmp_path/'export').exists()
    ready['value']=True;one=control.evidence_export();two=control.evidence_export()
    assert one==two and one['archive']==reference(tmp_path/'export/evidence.tar.gz')


def _final_fixture(tmp_path,monkeypatch,existing):
    source=tmp_path/'remote';source.mkdir();out=tmp_path/'return';out.mkdir()
    payload=b'authenticated payload';file=source/'one.txt';file.write_bytes(payload)
    manifest={'task_id':TASK_ID,'source_sha256':'a'*64,'contract_sha256':'c'*64,'files':[{'path':'one.txt','bytes':len(payload),'sha256':sha(file)}]}
    raw=json.dumps(manifest).encode();archive=source/'evidence.tar.gz';_tar(archive,[('manifest.json',raw),('one.txt',payload)])
    deployment=tmp_path/'DEPLOYED.json';write(deployment,{'source_sha256':'a'*64})
    s={'phases':{p:{'supervisor-exit.json':{'status':'PASS'},'dispatch':{'process_identity':{'start':'x'}},'supervisor_current_identity':None} for p in PHASES},
       'key_checkpoints':[{'epoch':e} for e in KEY_EPOCHS],'short_checkpoints':[{'name':n} for n in SHORT_STATES]}
    exported={'task_id':TASK_ID,'source_sha256':'a'*64,'contract_sha256':'c'*64,'files':1,'archive':reference(archive)}
    monkeypatch.setattr(client,'remote_call',lambda d,a,r:s if a=='status' else exported)
    monkeypatch.setattr(client,'sync_keys',lambda *a:[]);monkeypatch.setattr(client,'sync_short',lambda *a:[])
    def download(remote,expected,dest,receipts,size):
        Path(dest).write_bytes(Path(remote).read_bytes());assert sha(dest)==expected;return reference(dest)
    monkeypatch.setattr(client,'download',download)
    if existing:
        evidence=out/'evidence';evidence.mkdir()
        if existing=='other':write(evidence/'manifest.json',{'task_id':TASK_ID,'files':[]})
        if existing=='partial':(evidence/'one.txt').write_bytes(payload)
    return deployment,out,s


def test_final_sync_rejects_missing_duplicate_keys_before_transfer(tmp_path,monkeypatch):
    deployment,out,s=_final_fixture(tmp_path,monkeypatch,None)
    s['key_checkpoints']=[]
    monkeypatch.setattr(client,'sync_keys',lambda *a:pytest.fail('download started before key inventory gate'))
    with pytest.raises(RuntimeError,match='each key checkpoint'):client.sync_final(deployment,out)
    s['key_checkpoints']=[{'epoch':0} for _ in KEY_EPOCHS]
    with pytest.raises(RuntimeError,match='each key checkpoint'):client.sync_final(deployment,out)
    assert not (out/'VERIFICATION.json').exists()


def test_final_sync_does_not_trust_existing_self_reported_manifest(tmp_path,monkeypatch):
    deployment,out,s=_final_fixture(tmp_path,monkeypatch,'other')
    with pytest.raises(RuntimeError,match='authenticated archive'):client.sync_final(deployment,out)
    assert not (out/'VERIFICATION.json').exists()


def test_final_sync_preserves_partial_and_extracts_complete_authenticated_copy(tmp_path,monkeypatch):
    deployment,out,s=_final_fixture(tmp_path,monkeypatch,'partial')
    result=client.sync_final(deployment,out)
    assert result['status']=='WINDOWS_ALL_RETURNED_SHA_PASS' and result['evidence_files']==1
    assert Path(result['evidence_root'])!=out/'evidence' and (out/'evidence/one.txt').is_file()
    assert (Path(result['evidence_root'])/'manifest.json').is_file()


def test_local_ready_requires_actual_passing_receipts_and_bound_source(tmp_path):
    op=tmp_path/'operator.json';write(op,{'status':'PASS','fixtures':{str(i):{} for i in range(16)}})
    xml=tmp_path/'gate.xml';xml.write_text('<testsuites><testsuite><testcase name="checked"/></testsuite></testsuites>')
    evidence=[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in (op,xml)]
    value={'schema':'f0-local-ready-v1','task_id':TASK_ID,'status':'LOCAL_READY','source_sha256':'a'*64,
           'evidence_files':evidence,'gates':[{'id':'operators','path':op.name}]+[
               {'id':key,'path':xml.name,'testcases':['checked']} for key in ('core_reused','affected','lifecycle')]}
    path=tmp_path/'LOCAL_READY.json';write(path,value)
    assert local_ready.verify(path,'a'*64)['status']=='LOCAL_READY'
    value['gates'][1]={'id':'core_reused','reports':[{'path':xml.name,'testcases':['checked']}]}
    path.write_text(json.dumps(value));assert local_ready.verify(path,'a'*64)['status']=='LOCAL_READY'
    with pytest.raises(RuntimeError,match='source mismatch'):local_ready.verify(path,'b'*64)
    xml.write_text('<testsuites><testsuite><testcase name="checked"><failure/></testcase></testsuite></testsuites>')
    with pytest.raises(RuntimeError,match='receipt changed'):local_ready.verify(path,'a'*64)
    value['evidence_files'][1].update(sha256=sha(xml),bytes=xml.stat().st_size)
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='did not pass'):local_ready.verify(path,'a'*64)


def test_stage_new_source_can_reuse_identical_materials_and_rejects_changed_materials(tmp_path,monkeypatch):
    import contextlib
    export=tmp_path/'export';(export/'source').mkdir(parents=True)
    code=export/'source/example.py';code.write_text('# inert synthetic export\n')
    manifest={'task_id':TASK_ID,'files':[{'path':'example.py','sha256':sha(code),'bytes':code.stat().st_size}]}
    write(export/'source-manifest.json',manifest)
    _tar(export/'source.tar.gz',[('source-manifest.json',(export/'source-manifest.json').read_bytes()),('source/example.py',code.read_bytes())])
    spec={'task_id':TASK_ID,'source_sha256':sha(export/'source-manifest.json'),'archive':reference(export/'source.tar.gz')};write(export/'EXPORT.json',spec)
    plan={'roles':{},'pretrained':{'path':str(tmp_path/'weight.pth')}};manifests={}
    for role,count,split in [('train_core',4869,'train'),('development',548,'val')]:
        path=tmp_path/(role+'_manifest.json');write(path,{'synthetic_role':role})
        plan['roles'][role]={'manifest_path':str(path),'manifest_sha256':sha(path),'members':count}
        manifests[role]=(sha(path),count,split)
    weight=tmp_path/'weight.pth';weight.write_bytes(b'inert weight fixture, not a model')
    selected=tmp_path/'trace-selection.json';write(selected,{'synthetic':True})
    plan_path=tmp_path/'materials-plan.json';write(plan_path,plan)
    ready_path=tmp_path/'LOCAL_READY.json';write(ready_path,{'evidence_files':[]})
    monkeypatch.setattr(local_ready,'verify',lambda path,expected:read(path))
    monkeypatch.setattr(material_paths,'metadata_path',lambda kind,path,expected:Path(path))
    monkeypatch.setattr(material_paths,'MANIFESTS',manifests)
    monkeypatch.setattr(client,'PRETRAINED_SHA',sha(weight));monkeypatch.setattr(client,'SELECTION_SHA',sha(selected))
    remote=tmp_path/'synthetic-cloud';remote.mkdir();monkeypatch.setattr(client,'REMOTE_ROOT',str(remote))
    monkeypatch.setattr(client,'upload',lambda path,receipt:reference(path))
    calls=[]
    def remote_python(source,receipt,timeout):
        calls.append('LOCAL_SYNTHETIC_EXEC_ONLY');stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):exec(compile(source,'<synthetic staging>','exec'),{})
        return stdout.getvalue()
    monkeypatch.setattr(client,'helpers',lambda:SimpleNamespace(remote_python=remote_python))
    first=client.stage(export,plan_path,ready_path,tmp_path/'stage-1','release-001')
    material_before=sha(remote/'materials/materials-manifest.json')
    second=client.stage(export,plan_path,ready_path,tmp_path/'stage-2','release-002')
    assert first['material_root']==second['material_root'] and first['source_root']!=second['source_root']
    assert sha(remote/'materials/materials-manifest.json')==material_before and len(calls)==2
    (remote/'materials/materials-manifest.json').write_bytes(b'changed; must preserve')
    with pytest.raises(AssertionError,match='material manifest differs'):
        client.stage(export,plan_path,ready_path,tmp_path/'stage-3','release-003')
    assert not (tmp_path/'stage-3/DEPLOYED.json').exists()
