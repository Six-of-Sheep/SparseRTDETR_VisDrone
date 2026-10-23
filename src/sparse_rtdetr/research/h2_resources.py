"""The original lab resource predicates, with H2 acceptance deadlines added.

Read-only observations; never changes CPU power or GPU clocks. CPU fixtures use
the identical screen and are explicitly rejected by the lab profile.
"""
import os
from pathlib import Path
import time
from ..localization.runio import require, publish

GIB=1024**3
LIMITS=dict(minimum_CPUs=16,minimum_RAM_GiB=16,minimum_free_disk_GiB=80,
    entry_shm_GiB=8,live_shm_GiB=4,max_resource_age_seconds=25,monitor_interval_seconds=2,
    CPU_temperature_C_below=90,GPU_temperature_C_below=80,GPU_entry_free_MiB=20000,
    GPU_live_free_MiB=3072,CPU_PL1_uW_max=125000000,CPU_PL2_uW_max=150000000,
    GPU_graphics_MHz_max=1500,GPU_stage_seconds_cap=900,wall_seconds_cap=1800,
    allocated_GiB_cap=18,task_storage_GiB_cap=100)


def fixture(c):
    dev={k:dict(status='OBSERVED',value=v) for k,v in dict(uuid=c['gpu_uuid'],name='CPU_FIXTURE_NO_GPU',
        **{'memory.total':24576,'memory.free':24576,'temperature.gpu':40,'utilization.gpu':0,
           'clocks.current.graphics':1500}).items()}
    return dict(fixture_only=True,sampled_epoch=time.time(),monotonic=time.monotonic(),boot_id=c['boot_id'],
        snapshot=dict(gpu=dict(devices=[dev]),resources=dict(cpu_count=16,ram_available=32*GIB),cpu_temperature_max=40,shm_free=16*GIB),
        storage=dict(c['storage_identity'],admitted_available_bytes=200*GIB,ssd_free_bytes=20*GIB),
        compute_apps_status='OBSERVED',compute_apps=[],memory_events={'vmstat':{'oom_kill':0}},
        cpu_power_constraints=dict(constraint_0_name='long_term',constraint_1_name='short_term',
            constraint_0_power_limit_uw='125000000',constraint_1_power_limit_uw='150000000'))


def screen(sample,c,*,entry,baseline=None,owned=None):
    require(c['resources']==LIMITS,'H2 resource contract missing/changed')
    require(bool(sample.get('fixture_only'))==(c['profile']=='CPU_FIXTURE_ONLY'),'resource fixture in production')
    age=time.time()-sample['sampled_epoch']
    sample['age_check']=dict(actual_age_seconds=age,monotonic_age_seconds=time.monotonic()-sample['monotonic'],
        checked_epoch=time.time(),threshold_seconds=25,sampling_timing=sample.get('sampling_timing'),
        contract_sha256=c.get('contract_sha256'),pid=os.getpid())
    require(0<=age<25,'stale resource sample')
    require(sample['boot_id']==c['boot_id'],'boot identity differs')
    s=sample['snapshot'];r=c['resources']
    require(s['resources']['cpu_count']>=r['minimum_CPUs'] and s['resources']['ram_available']>=r['minimum_RAM_GiB']*GIB,'CPU/RAM admission failed')
    require(s['shm_free']>=(r['entry_shm_GiB'] if entry else r['live_shm_GiB'])*GIB,'shm admission failed')
    require({k:sample['storage'][k] for k in c['storage_identity']}==c['storage_identity'],'storage identity differs')
    require(sample['storage']['mount']['rw'] and sample['storage']['admitted_available_bytes']>=r['minimum_free_disk_GiB']*GIB,'storage free/rw admission failed')
    require(sample['storage']['ssd_free_bytes']>=4*GIB and s['cpu_temperature_max']<90,'SSD/CPU temperature failed')
    if baseline is not None:require(sample['memory_events']==baseline['memory_events'],'OOM events changed')
    require(sample['memory_events'] and sample['compute_apps_status']=='OBSERVED','OOM/compute inventory absent')
    devices=s['gpu']['devices'];require(len(devices)==1,'physical GPU inventory differs')
    g=devices[0]
    for key in ('uuid','memory.free','temperature.gpu','clocks.current.graphics'):
        require(g[key]['status']=='OBSERVED','GPU field unavailable: '+key)
    require(g['uuid']['value']==c['gpu_uuid'],'GPU UUID differs')
    require(g['memory.free']['value']>=(20000 if entry else 3072) and g['temperature.gpu']['value']<80,'GPU memory/temperature failed')
    cap=sample['cpu_power_constraints']
    require(cap['constraint_0_name']=='long_term' and cap['constraint_1_name']=='short_term','CPU cap names differ')
    require(0<int(cap['constraint_0_power_limit_uw'])<=125000000 and 0<int(cap['constraint_1_power_limit_uw'])<=150000000,'CPU caps differ')
    require(0<float(g['clocks.current.graphics']['value'])<=1500,'GPU clock cap differs')
    apps=sample['compute_apps']
    require(not apps if entry else len(apps)<=1,'concurrent GPU process')
    require(all(a['uuid']==c['gpu_uuid'] and (owned is None or a['pid'] in owned) for a in apps),'GPU process is not owned')
    return sample


def snapshot(c,*,entry,baseline=None,owned=None):
    began=dict(epoch=time.time(),monotonic=time.monotonic())
    if c['profile']=='CPU_FIXTURE_ONLY':sample=fixture(c)
    else:
        from ..localization.thermal_resume import snapshot as collect
        p=Path(c['job_root'])
        while not p.exists():p=p.parent
        sample=collect(str(p))
    sample['sampling_timing']=dict(started=began,completed=dict(epoch=time.time(),monotonic=time.monotonic()))
    try:return screen(sample,c,entry=entry,baseline=baseline,owned=owned)
    except BaseException as error:
        p=Path(c['job_root']).parent/('resource-rejected-'+str(time.time_ns())+'.json')
        publish(p,dict(status='REJECTED_RESOURCE_NO_RETRY',error=repr(error),sample=sample))
        raise
