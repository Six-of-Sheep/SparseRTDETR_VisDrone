"""Targeted CPU fault routing, no Linux syscalls/GPU/real image reads."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import pytest
from sparse_rtdetr.localization import guard,worker
from sparse_rtdetr.localization.data import read_bound
from sparse_rtdetr.localization.runio import read
from test_localization_lifecycle import _guard_fixture

def test_process_exit_between_inventory_and_stat_is_benign(monkeypatch):
    missing_root=[False]
    class Proc:
        def __init__(self,*parts):self.path=PurePosixPath(*parts)
        @property
        def name(self):return self.path.name
        def __truediv__(self,p):return Proc(str(self.path/p))
        def iterdir(self):
            if missing_root[0]:raise FileNotFoundError(2,'root missing','/proc')
            return iter([Proc('/proc/123'),Proc('/proc/124')])
        def read_text(self):
            if str(self.path)=='/proc/123/stat':raise FileNotFoundError(2,'process exited',str(self.path))
            fields=['0']*22;fields[0]='S';fields[1]='1';fields[11]='2';fields[12]='3';fields[19]='1000'
            return '124 (fixture) '+' '.join(fields)
    monkeypatch.setattr(guard,'Path',Proc)
    assert guard.processes()=={124:{'ppid':1,'start':'1000','state':'S','cpu':5}}
    assert guard.identity(123) is None
    missing_root[0]=True
    with pytest.raises(FileNotFoundError):guard.processes() # A missing /proc root is not a single PID exit.

@pytest.mark.parametrize('current',[None,{'start':'new-owner','state':'R'}])
def test_pidfd_never_signals_exited_or_reused_pid(monkeypatch,current):
    closed=[];sent=[]
    monkeypatch.setattr(guard,'ctypes',SimpleNamespace(CDLL=lambda *a,**k:SimpleNamespace()))
    monkeypatch.setattr(guard,'os',SimpleNamespace(pidfd_open=lambda pid:42,close=closed.append))
    monkeypatch.setattr(guard,'identity',lambda p:current)
    monkeypatch.setattr(guard,'signal',SimpleNamespace(pidfd_send_signal=lambda *a:sent.append(a)))
    guard.signal_owned(123,'old-owner',15)
    assert closed==[42] and sent==[]

def test_resource_file_missing_remains_fatal_and_diagnostic(tmp_path,monkeypatch):
    result,_=_guard_fixture(tmp_path,monkeypatch,kind='missing-file',exits_after_term=2.)
    assert result['status']=='STOP' and result['guard_error']['operation']=='resource_snapshot'
    assert result['guard_error']['filename']=='/synthetic/missing'

def test_monitor_file_disappearing_after_first_seen_is_not_success(tmp_path,monkeypatch):
    result,_=_guard_fixture(tmp_path,monkeypatch,kind='progress-gone',exits_after_term=2.)
    assert result['status']=='STOP' and result['guard_error']['operation']=='progress_read'
    assert result['guard_error']['filename'].endswith('progress.json')

@pytest.mark.parametrize('published',[True,False])
def test_partial_vanish_is_benign_only_with_published_counterpart(tmp_path,monkeypatch,published):
    partial=tmp_path/'state.pth.partial';partial.write_bytes(b'temporary')
    if published:(tmp_path/'state.pth').write_bytes(b'published')
    original=Path.stat
    def stat(p,*a,**k):
        if p==partial:raise FileNotFoundError(2,'partial vanished',str(partial))
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'stat',stat)
    if published:assert len(guard.file_activity(tmp_path))==1
    else:
        with pytest.raises(FileNotFoundError):guard.file_activity(tmp_path)

def test_complete_checkpoint_missing_is_not_silently_ignored(tmp_path,monkeypatch):
    p=tmp_path/'state.pth';p.write_bytes(b'complete')
    original=Path.stat
    def stat(path,*a,**k):
        if path==p:raise FileNotFoundError(2,'immutable state vanished',str(p))
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'stat',stat)
    with pytest.raises(FileNotFoundError):guard.file_activity(tmp_path)

def test_true_missing_annotation_stops_worker_with_preserved_failure(tmp_path,monkeypatch):
    missing=tmp_path/'train_core/annotations/required.txt'
    def run(*args):read_bound(missing,'0'*64,100)
    monkeypatch.setattr(worker,'run_formal',run)
    contract={k:'fixture' for k in ('contract_sha256','source_sha256','data_sha256','selection_sha256','gpu_uuid','boot_id','environment')}
    with pytest.raises(ValueError,match='bound regular file missing'):worker.execute(contract,'formal',tmp_path)
    failure=read(tmp_path/'failure.json')
    assert failure['status']=='STOP_PRESERVE_NO_RETRY' and str(missing) in failure['traceback']
    assert read(tmp_path/'worker-exit.json')['body_returned'] is False

def test_progress_file_does_not_fabricate_semantic_completion(tmp_path):
    p=tmp_path/'progress.json';p.write_text('{"event":"load","monotonic":1}')
    # Before the first writer publication, no marker is allowed. After one
    # has been observed, missing metadata must be classified and stopped.
    value=guard.read_progress(p,seen=False);assert value['monotonic']==1
    p.unlink()
    with pytest.raises(FileNotFoundError):guard.read_progress(p,seen=True)
    assert guard.read_progress(p,seen=False) is None
