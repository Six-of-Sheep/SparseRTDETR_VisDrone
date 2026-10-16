"""Canonical account file-storage root with SSD inputs/environment left in place."""
import os,shutil,time
from pathlib import Path
from .runio import REMOTE_ROOT,require,read,sha

MOUNT='/autodl-fs/data'
PLATFORM_ENTRY='/root/autodl-fs'
SSD='/root/autodl-tmp'
# Conservative planning bound from the user's account display (200 GB). This
# is capped again by current per-account AutoFS statvfs, never by host pool df.
ACCOUNT_PLANNING_BYTES=200_000_000_000
SSD_RESERVE_BYTES=4*1024**3

def path_on_mount(path):
    p=Path(path)
    require(p.is_absolute() and p==p.resolve(),'task must use canonical absolute storage paths')
    require(p==Path(REMOTE_ROOT) or p.is_relative_to(REMOTE_ROOT),'task output outside its authorized storage root')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'task canonical path traverses a symlink')
    return p

def mounts(text):
    rows=[]
    for line in text.splitlines():
        a,b=line.split(' - ',1);f,t=a.split(),b.split()
        rows.append({'mountpoint':f[4],'type':t[0],'source':t[1],'rw':'rw' in f[5].split(',')})
    return rows

def observe(path):
    path_on_mount(path);base=Path(MOUNT);entry=Path(PLATFORM_ENTRY)
    require(entry.is_symlink() and entry.resolve()==base,'platform storage entry no longer resolves to the admitted mount')
    rows=mounts(Path('/proc/self/mountinfo').read_text())
    matches=[r for r in rows if r['mountpoint']==MOUNT]
    require(len(matches)==1 and matches[0]['type']=='fuse.autofs' and matches[0]['rw'],'file storage mount missing/type changed/read-only')
    stat=os.statvfs(base);total=stat.f_blocks*stat.f_frsize;free=stat.f_bavail*stat.f_frsize
    available=min(free,max(0,ACCOUNT_PLANNING_BYTES-(total-free)))
    require(Path(path).stat().st_dev==base.stat().st_dev,'task root moved to a different filesystem')
    return {'canonical_mount':MOUNT,'platform_entry':PLATFORM_ENTRY,'mount':matches[0],
            'st_dev':base.stat().st_dev,'reported_total_bytes':total,'reported_available_bytes':free,
            'planning_cap_bytes':ACCOUNT_PLANNING_BYTES,'admitted_available_bytes':available,
            'capacity_basis':'min(account AutoFS statvfs available, conservative200GB account-display cap minus used); no independent quota API claim',
            'ssd_free_bytes':shutil.disk_usage(SSD).free,'observed_epoch':time.time()}

def identity(value):
    return {k:value[k] for k in ('canonical_mount','platform_entry','mount','st_dev','planning_cap_bytes')}

def screen(value,required_bytes):
    require(value['admitted_available_bytes']>=required_bytes,'file-storage account capacity below remaining states plus40GiB reserve')
    require(value['ssd_free_bytes']>=SSD_RESERVE_BYTES,'SSD headroom below4GiB for unchanged image/environment inputs')

def verify(contract,required_bytes=40*1024**3):
    saved=contract['storage'];current=observe(contract['job_root'])
    require(identity(current)==saved['identity'],'file-storage identity changed since freeze')
    for key in ('io_gate','restore_gate'):
        row=saved[key];p=path_on_mount(row['path'])
        require(sha(p)==row['sha256'],'storage gate receipt changed: '+key)
        gate=read(p);require(gate['status']=='PASS','storage gate not passed: '+key)
    require(read(saved['io_gate']['path'])['mount']['source']==current['mount']['source'],'IO gate is for another account mount')
    restore=read(saved['restore_gate']['path'])
    require(restore['source_sha256']==contract['source_sha256'] and restore['full_state_exact'] and restore['same_next_window_exact'],'restore gate is for another source or incomplete')
    screen(current,required_bytes);return current
