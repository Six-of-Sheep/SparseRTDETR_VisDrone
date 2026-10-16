"""Independent Linux process/resource protection; no overall stage deadline.

PID/starttime + pidfd cleanup follows the verified release-003 supervisor.
Only worker-tree CPU/IO and model/checkpoint progress count as useful activity.
"""
import ctypes,json,os,signal,stat,subprocess,time,traceback
from pathlib import Path
from . import cloud_probe as probe
from .runio import require,read,write,atomic,publish,reference

GIB=1024**3

def processes():
    result={}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/"stat").read_text().rsplit(")",1)[1].split()
            result[int(p.name)]={"ppid":int(f[1]),"start":f[19],"state":f[0],"cpu":int(f[11])+int(f[12])}
        except (FileNotFoundError,ProcessLookupError):pass
    return result

def descendants(root,rows):
    result={root}
    while True:
        following={pid for pid,row in rows.items() if row["ppid"] in result}
        if following<=result:return result-{root}
        result|=following

def identity(pid):
    try:
        fields=Path("/proc",str(pid),"stat").read_text().rsplit(")",1)[1].split()
        return {"start":fields[19],"state":fields[0]}
    except (FileNotFoundError,ProcessLookupError):return None

def signal_owned(pid,start,sig):
    # Opening the pidfd first prevents a new process from inheriting the signal.
    libc=ctypes.CDLL(None,use_errno=True)
    def syscall(number,*args):
        require(os.uname().machine=="x86_64","unsupported pidfd syscall architecture")
        value=libc.syscall(ctypes.c_long(number),*args)
        if value<0:
            error=ctypes.get_errno();raise OSError(error,os.strerror(error))
        return value
    try:fd=os.pidfd_open(pid) if hasattr(os,"pidfd_open") else syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
    except ProcessLookupError:return
    try:
        row=identity(pid)
        if row and row["start"]==start and row["state"] not in ("Z","X"):
            try:
                if hasattr(signal,"pidfd_send_signal"):signal.pidfd_send_signal(fd,sig)
                else:syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(0),ctypes.c_uint(0))
            except ProcessLookupError:pass
    finally:os.close(fd)

def snapshot(path):
    from .storage import observe,SSD
    storage=observe(path)
    sample=probe.snapshot([str(path),SSD])
    result=probe.bounded_command(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                                  "--format=csv,noheader,nounits"],timeout=12)
    apps=[]
    if result["status"]=="OBSERVED":
        for line in result["stdout"].splitlines():
            bits=[b.strip() for b in line.split(",")]
            require(len(bits)==4 and bits[0].isdigit(),"GPU process telemetry parse failure")
            apps.append({"pid":int(bits[0]),"uuid":bits[1],"name":bits[2],"memory_mib":bits[3]})
    events={}
    for row in sample["resources"]["memory"].get("ancestors",[]):
        content=probe.read_text(Path(row["path"])/"memory.events")
        if content is not None:events[row["path"]]={k:int(v) for k,v in (line.split() for line in content.splitlines())}
    return {"sampled_epoch":time.time(),"monotonic":time.monotonic(),"snapshot":sample,
            "boot_id":probe.read_text("/proc/sys/kernel/random/boot_id"),"compute_apps_status":result["status"],
            "compute_apps":apps,"memory_events":events,"storage":storage}

def screen(sample,*,gpu_uuid=None,entry=False,gpu=True,disk_bytes=40*GIB,baseline_events=None):
    s=sample["snapshot"];r=s["resources"]
    require(r["memory"]["status"]=="OBSERVED" and isinstance(r["memory"]["visible_headroom_upper_bound"],(int,float)) and r["memory"]["visible_headroom_upper_bound"]>=4*GIB,"RAM headroom missing/below 4GiB")
    require(r["cpu"]["status"]=="OBSERVED" and r["cpu_visible_upper_bound"]>=16,"visible CPU bound below 16")
    require(r["pids"]["status"]=="OBSERVED" and isinstance(r["pids"]["visible_headroom_upper_bound"],(int,float)) and r["pids"]["visible_headroom_upper_bound"]>=32,"PID headroom below 32")
    from .storage import screen as storage_screen
    storage_screen(sample['storage'],disk_bytes)
    require(s["shm"]["status"]=="OBSERVED" and s["shm"]["value"]["free_bytes"]>=(8 if entry else 4)*GIB,"shared memory reserve")
    require(bool(sample["memory_events"]),"cgroup OOM visibility unavailable")
    if baseline_events is not None:
        require(set(sample["memory_events"])==set(baseline_events),"cgroup event scope changed")
        for path,values in sample["memory_events"].items():
            require(all(values.get(k,0)==baseline_events[path].get(k,0) for k in ("oom","oom_kill")),"cgroup OOM event changed")
    if not gpu:return None
    require(sample["compute_apps_status"]=="OBSERVED","GPU compute inventory missing")
    gs=s["gpu"].get("devices",[]);require(len(gs)==1,"GPU inventory is not exactly one")
    g=gs[0]
    for k in ("uuid","name","memory.total","memory.free","temperature.gpu","utilization.gpu"):
        require(g[k]["status"]=="OBSERVED","GPU mandatory telemetry missing: "+k)
    uid=g["uuid"]["value"]
    require(gpu_uuid is None or gpu_uuid==uid,"GPU UUID changed")
    require(g["memory.free"]["value"]>=(20000 if entry else 3072),"GPU memory reserve")
    require(g["temperature.gpu"]["value"]<80,"GPU temperature threshold")
    require((not sample["compute_apps"]) if entry else (len(sample["compute_apps"])<=1),"unexpected concurrent GPU process")
    require(all(a["uuid"]==uid for a in sample["compute_apps"]),"foreign GPU compute UUID")
    return uid

class Progress:
    def __init__(self,now):
        self.active=self.semantic=now;self.cpu=0;self.io=None;self.io_members={};self.files=None;self.marker=None
    def observe(self,now,*,cpu,io,files,marker,gpu_busy,idle=300,operation=1800,io_detail=None):
        semantic=marker!=self.marker
        io_active=False
        if io_detail is not None:
            for row in io_detail['members']:
                if row['status']!='OBSERVED':continue
                key=(row['pid'],row['start']);old=self.io_members.get(key)
                io_active=io_active or (old is not None and row['bytes']>old)
                self.io_members[key]=row['bytes']
        else:io_active=io is not None and self.io is not None and io>self.io
        productive=semantic or files!=self.files or cpu-self.cpu>=5 or io_active or gpu_busy
        if productive:self.active=now
        if semantic:self.semantic=now
        self.cpu,self.io,self.files,self.marker=cpu,io,files,marker
        reason="NO_PROGRESS_ALL_OBSERVABLES" if now-self.active>=idle else ("SINGLE_OPERATION_NO_COMPLETION" if now-self.semantic>=operation else None)
        return {"cpu_ticks":cpu,"io_bytes":io,"io_status":None if io_detail is None else io_detail["status"],"io_members":None if io_detail is None else io_detail["members"],"io_activity":io_active,"gpu_busy":gpu_busy,"semantic_change":semantic,
                "activity_idle_seconds":now-self.active,"operation_seconds":now-self.semantic,"stop_reason":reason}

def read_progress(path,*,seen):
    """Only absence before first publication is normal worker setup.

    Once observed, disappearing monitor metadata is a distinct fatal condition;
    never turn it into a new semantic marker or conceal unrelated missing data.
    """
    try:return read(path)
    except FileNotFoundError as error:
        if not seen:return None
        raise FileNotFoundError(2,'previously observed monitor progress file disappeared',str(path)) from error

def file_activity(out):
    rows=[];out=Path(out)
    for path in out.rglob("*"):
        if not (path.suffix in (".pth",".npz") or path.name.endswith((".pth.partial",".npz.partial"))):continue
        try:
            info=path.stat()
            if stat.S_ISREG(info.st_mode):rows.append((str(path.relative_to(out)),info.st_size,info.st_mtime_ns))
        except (FileNotFoundError,ProcessLookupError):
            # Only the tested exclusive-publication lifecycle can legitimately
            # remove a partial after linking its complete counterpart.
            if path.name.endswith(('.pth.partial','.npz.partial')):
                published=path.with_suffix('')
                if published.is_file() and not published.is_symlink():continue
            raise
    return tuple(sorted(rows))

def supervise(command,out,env,*,gpu_uuid,boot_id,gpu=True,checkpoint_bytes=0,formal=False,runtime=None,run_identity=None):
    out=Path(out);out.mkdir(exist_ok=False)
    from . import channel
    telemetry=out;progress_reader=None;resource_writer=None
    if runtime is not None:
        b=channel.create(runtime,run_identity)
        telemetry=Path(runtime);env={**env,**channel.environment(b)}
        progress_reader=channel.Reader(telemetry/'progress.jsonl',b,'progress')
        resource_writer=channel.Writer(b,'resources')
    require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"subreaper unavailable")
    signal.signal(signal.SIGHUP,signal.SIG_IGN);interrupted=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:interrupted.append(s))
    first=snapshot(out);required=40*GIB+(122*checkpoint_bytes if formal else 4*checkpoint_bytes)
    screen(first,gpu_uuid=gpu_uuid,entry=True,gpu=gpu,disk_bytes=required)
    require(first["boot_id"]==boot_id,"boot identity changed")
    if resource_writer:resource_writer.append(first)
    else:write(out/"resource-entry.json",first)
    tracked={};unknown=set();cleanup=[];table_unknown=False;started=time.time();progress=Progress(time.monotonic());stable_gpu_pid=None
    write(telemetry/"supervisor-start.json",{"pid":os.getpid(),"command":command,"started_epoch":started,"total_wallclock_cap":None,"idle_seconds":300,"operation_seconds":1800})
    child=None;reason="NOT_STARTED";last_snapshot=first;operation='worker_launch';guard_error=None;progress_seen=False
    try:
        with (telemetry/"worker.log").open("x",encoding="utf-8") as log,(telemetry/"activity.jsonl").open("x") as activity:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            born=identity(child.pid)
            if born is not None:tracked[child.pid]=born["start"]
            while True:
                operation='process_inventory'
                rows=processes();own=descendants(child.pid,rows)|({child.pid} if child.pid in rows else set())
                for pid in own:tracked[pid]=rows[pid]["start"]
                live=[p for p in own if rows[p]["state"] not in ("Z","X")]
                cpu=sum(rows[p]["cpu"] for p in live)
                operation='process_io'
                from .process_io import observe as observe_io
                io_detail=observe_io(live,rows,identity,runtime=bool(runtime));io=io_detail['bytes']
                marker=None;completed_epochs=0
                operation='progress_read'
                if progress_reader:
                    record,_=progress_reader.poll();p=None if record is None else record['payload']
                    require(record is None or record['writer_pid']==child.pid,'progress writer is not owned worker')
                else:p=read_progress(out/'progress.json',seen=progress_seen)
                if p is not None:
                    progress_seen=True
                    marker=progress_reader.semantic if progress_reader else (p["event"],p["monotonic"])
                    completed_epochs=min(120,p.get("completed_epochs",0))
                required=40*GIB+(max(0,122-completed_epochs)*checkpoint_bytes if formal else 0)
                operation='resource_snapshot'
                last_snapshot=snapshot(out)
                operation='resource_admission'
                from .storage import identity as storage_identity
                require(storage_identity(last_snapshot['storage'])==storage_identity(first['storage']),'file-storage mount changed during worker execution')
                screen(last_snapshot,gpu_uuid=gpu_uuid,gpu=gpu,disk_bytes=required,baseline_events=first["memory_events"])
                require(last_snapshot["boot_id"]==boot_id,"boot identity changed")
                if gpu and last_snapshot["compute_apps"]:
                    current=last_snapshot["compute_apps"][0]["pid"]
                    require(stable_gpu_pid in (None,current),"visible GPU compute PID changed")
                    stable_gpu_pid=current
                operation='monitor_publish'
                if resource_writer:resource_writer.append(last_snapshot)
                else:atomic(out/"monitor.json",last_snapshot)
                operation='checkpoint_file_activity'
                files=file_activity(out)
                busy=gpu and any(d["utilization.gpu"].get("value",0)>1 for d in last_snapshot["snapshot"]["gpu"].get("devices",[]))
                record=progress.observe(time.monotonic(),cpu=cpu,io=io,files=files,marker=marker,gpu_busy=busy,io_detail=io_detail)
                operation='activity_write_and_exit_checks'
                activity.write(json.dumps({"epoch":time.time(),"live":live,**({} if runtime else {"resource_sample":last_snapshot}),**record})+"\n");activity.flush()
                if (telemetry/"failure.json").exists():reason="WORKER_FAILURE_RECEIPT";break
                if child.poll() is not None:reason="WORKER_EXIT";break
                if interrupted:reason="INTERRUPTED";break
                if record["stop_reason"]:reason=record["stop_reason"];break
                operation='monitor_interval';time.sleep(2)
    except BaseException as error:
        reason="GUARD_ERROR:"+repr(error)
        guard_error={'type':type(error).__name__,'error':str(error),'repr':repr(error),'operation':operation,
                     'errno':getattr(error,'errno',None),'filename':getattr(error,'filename',None),
                     'filename2':getattr(error,'filename2',None),'traceback':traceback.format_exc(),
                     'received_signals':list(interrupted),'epoch':time.time()}
        print(json.dumps({'event':'guard_error',**guard_error}),flush=True)
    def scan():
        nonlocal table_unknown
        try:
            rows=processes()
            for pid in descendants(os.getpid(),rows):tracked[pid]=rows[pid]["start"]
            return [p for p,start in tracked.items() if p in rows and rows[p]["start"]==start and rows[p]["state"] not in ("Z","X")]
        except (OSError,ValueError,IndexError) as error:
            table_unknown=True
            if repr(error) not in cleanup:cleanup.append(repr(error))
            live=[]
            for pid,start in tracked.items():
                try:
                    found=identity(pid)
                    if found is not None and found["start"]==start and found["state"] not in ("Z","X"):live.append(pid)
                except (OSError,ValueError,IndexError):unknown.add(pid)
            return live
    live=scan();residual=list(live);drain_started=time.monotonic()
    if reason=='WORKER_EXIT' and child is not None and child.returncode==0:
        while live and time.monotonic()-drain_started<8:
            time.sleep(.1);live=scan()
    residual_after_drain=list(live);drain_seconds=time.monotonic()-drain_started
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if child is not None and child.poll() is None:
            # Popen owns this unreaped direct child. Send TERM before its grace
            # interval even if /proc becomes unreadable inside the pidfd loop.
            try:child.terminate() if sig==signal.SIGTERM else child.kill()
            except OSError as error:cleanup.append(repr(error))
        until=time.monotonic()+8
        while (live or (child is not None and child.poll() is None)) and time.monotonic()<until:
            for pid in live:
                try:signal_owned(pid,tracked[pid],sig)
                except (OSError,RuntimeError,AttributeError,ValueError,IndexError) as error:
                    cleanup.append(repr(error));unknown.add(pid)
            time.sleep(.1)
            if child is not None:child.poll()
            live=scan()
    if child is not None:
        try:
            child.wait(timeout=2);unknown.discard(child.pid)
        except subprocess.TimeoutExpired:unknown.add(child.pid)
    while True:
        try:
            if os.waitpid(-1,os.WNOHANG)[0]==0:break
        except ChildProcessError:break
    completion={}
    if reason=="WORKER_EXIT" and child is not None and child.returncode==0:
        try:
            for name in ('RESULT.json','execution-binding.json','worker-exit.json'):
                completion[name]=reference((telemetry if name=='worker-exit.json' else out)/name)
            if (out/'binding.json').is_file():completion['binding.json']=reference(out/'binding.json')
        except (OSError,ValueError) as error:
            cleanup.append('completion receipt: '+repr(error))
    result={"status":"PASS" if reason=="WORKER_EXIT" and child is not None and child.returncode==0 and not residual_after_drain and not live and not unknown and not cleanup and not table_unknown else "STOP",
            "reason":reason,"returncode":None if child is None else child.returncode,"residual_at_leader_exit":residual,
            "residual_after_natural_drain":residual_after_drain,"natural_drain_seconds":drain_seconds,
            "remaining_live":live,"unknown_pids":sorted(unknown),"cleanup_errors":cleanup,"tracked":tracked,
            "started_epoch":started,"finished_epoch":time.time(),"elapsed_seconds":time.time()-started,"total_wallclock_cap":None,
            "gpu_pid_namespace_relation":"not_assumed; stable single NVML PID after empty entry","visible_gpu_pid":stable_gpu_pid,
            "completion_receipts":completion,"guard_error":guard_error,"received_signals":list(interrupted)}
    result["process_table_cleanup_uncertain"]=table_unknown
    publish(telemetry/"supervisor-exit.json",result)
    if runtime:
        try:channel.archive(telemetry,out)
        except BaseException as error:
            publish(telemetry/'mirror-error.json',{'status':'ARCHIVE_FAILED_PRESERVE','error':repr(error),'traceback':traceback.format_exc()})
            return 2
    return 0 if result["status"]=="PASS" else 2
