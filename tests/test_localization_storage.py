"""Only storage adaptation/affected lifecycle checks; synthetic inputs."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sparse_rtdetr.localization import storage,control,runio,material_paths,guard

def test_all_task_paths_use_canonical_file_storage_and_keep_ssd_inputs():
    assert runio.REMOTE_ROOT=='/autodl-fs/data/'+runio.TASK_ID
    assert control.PYTHON_REMOTE.startswith('/root/autodl-tmp/')
    assert storage.PLATFORM_ENTRY=='/root/autodl-fs'
    p=Path(runio.REMOTE_ROOT)
    if p.is_absolute():assert storage.path_on_mount(p/'short/C5.pth')==p/'short/C5.pth'
    for path in ('/root/autodl-fs/'+runio.TASK_ID,'/root/autodl-tmp/'+runio.TASK_ID,'/autodl-fs/data/other-task'):
        with pytest.raises(RuntimeError):storage.path_on_mount(path)

def test_mount_parser_and_capacity_are_per_volume():
    mounts=storage.mounts('54 13 0:109 / /autodl-fs/data rw,relatime - fuse.autofs AutoFS:account rw\n')
    assert mounts==[{'mountpoint':'/autodl-fs/data','type':'fuse.autofs','source':'AutoFS:account','rw':True}]
    value={'admitted_available_bytes':100*1024**3,'ssd_free_bytes':12*1024**3}
    storage.screen(value,80*1024**3)
    with pytest.raises(RuntimeError,match='account capacity'):storage.screen(value,101*1024**3)
    with pytest.raises(RuntimeError,match='SSD headroom'):storage.screen({**value,'ssd_free_bytes':3*1024**3},80*1024**3)

def test_storage_gate_binds_source_mount_and_real_restoration(tmp_path,monkeypatch):
    value={'canonical_mount':storage.MOUNT,'platform_entry':storage.PLATFORM_ENTRY,
           'mount':{'mountpoint':storage.MOUNT,'type':'fuse.autofs','source':'AutoFS:account','rw':True},
           'st_dev':109,'planning_cap_bytes':200_000_000_000,'admitted_available_bytes':100*1024**3,'ssd_free_bytes':12*1024**3}
    io=tmp_path/'io.json';restore=tmp_path/'restore.json'
    runio.write(io,{'status':'PASS','mount':{'source':'AutoFS:account'}})
    runio.write(restore,{'status':'PASS','source_sha256':'a'*64,'full_state_exact':True,'same_next_window_exact':True})
    c={'job_root':runio.REMOTE_ROOT,'source_sha256':'a'*64,'storage':{'identity':storage.identity(value),
            'io_gate':runio.reference(io),'restore_gate':runio.reference(restore)}}
    monkeypatch.setattr(storage,'path_on_mount',lambda p:Path(p))
    monkeypatch.setattr(storage,'observe',lambda p:value)
    assert storage.verify(c)['st_dev']==109
    with pytest.raises(RuntimeError,match='another source'):storage.verify({**c,'source_sha256':'b'*64})
    for key,bad in [('st_dev',110),('mount',{**value['mount'],'source':'AutoFS:other'})]:
        with monkeypatch.context() as m:
            m.setattr(storage,'observe',lambda p:{**value,key:bad})
            with pytest.raises(RuntimeError,match='identity changed'):storage.verify(c)
    restore.write_text(json.dumps({'status':'FAIL'}))
    with pytest.raises(RuntimeError,match='receipt changed'):storage.verify(c)

def test_guard_uses_account_capacity_independently_of_ssd_report():
    # The NAS must hold the reserve/states; a small SSD is valid for unchanged
    # inputs as long as its own 4GiB margin remains.
    sample={'snapshot':{'resources':{
        'memory':{'status':'OBSERVED','visible_headroom_upper_bound':80*1024**3},
        'cpu':{'status':'OBSERVED'},'cpu_visible_upper_bound':16,
        'pids':{'status':'OBSERVED','visible_headroom_upper_bound':100}},
        'disks':{'SSD':{'status':'OBSERVED','value':{'free_bytes':12*1024**3}}},
        'shm':{'status':'OBSERVED','value':{'free_bytes':40*1024**3}}},
        'storage':{'admitted_available_bytes':100*1024**3,'ssd_free_bytes':12*1024**3},'memory_events':{'a':{'oom':0}}}
    assert guard.screen(sample,gpu=False,disk_bytes=80*1024**3) is None
    with pytest.raises(RuntimeError,match='account capacity'):
        guard.screen({**sample,'storage':{**sample['storage'],'admitted_available_bytes':70*1024**3}},gpu=False,disk_bytes=80*1024**3)

def test_contract_rejects_old_ssd_or_missing_storage_before_launch():
    c={'schema':'f0-execution-contract-v1','task_id':runio.TASK_ID,'workload':control.WORKLOAD,
       'job_root':'/root/autodl-tmp/'+runio.TASK_ID}
    with pytest.raises(RuntimeError,match='wrong F0 job root'):control.validate_contract(c)
    c['job_root']=runio.REMOTE_ROOT
    with pytest.raises(RuntimeError,match='file-storage contract missing'):control.validate_contract(c)
