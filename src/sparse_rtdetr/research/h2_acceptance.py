"""Acceptance-only CLI -> independent watchdog -> real worker -> SHA slim close.

Every profile uses this dispatch. Only input/resource/device observations differ
for explicitly labelled CPU fixtures. No formal/evaluation/speed entry exists.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback

from ..localization.runio import require,read,publish,atomic,reference,digest,sha
from . import h2_contract as ct,h2_resources as resource


def ledger_path(c):return Path(c['task_root'])/'CUMULATIVE_COST.json'


@contextlib.contextmanager
def ledger(c):
    p=ledger_path(c)
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=read(p)
        require(value['contract_sha256']==c['contract_sha256'],'ledger contract differs')
        yield value
        atomic(p,value)


def gpu_seconds(v):
    return v['GPU_seconds']+(time.monotonic()-v['active']['monotonic'] if v['active'] is not None else 0)


def budget_guard(c,v,*,forward=0,update=0,arm=None,allocated=0):
    require(time.monotonic()-v['started_monotonic']<1800,'total wall budget exhausted')
    require(gpu_seconds(v)<900,'cumulative GPU-stage budget exhausted')
    require(allocated<=18*resource.GIB,'allocated memory cap exceeded')
    require(v['forward_calls']+forward<=18 and v['forward_calls']+forward<=146596,'acceptance/future forward budget exhausted')
    require(v['actual_updates']+update<=9,'acceptance actual update budget exhausted')
    if arm is not None:
        require(v['arms'][arm]['forward_calls']+forward<=6 and v['arms'][arm]['actual_updates']+update<=3,'arm acceptance budget exhausted')
    require(sum(p.stat().st_size for p in Path(c['job_root']).rglob('*') if p.is_file())<100*resource.GIB,'task storage budget exhausted')


def reserve(c,arm,*,forward=0,update=0):
    with ledger(c) as v:
        require(v['active'] is not None and v['active']['arm']==arm,'forward/update outside active arm')
        budget_guard(c,v,forward=forward,update=update,arm=arm)
        v['forward_calls']+=forward;v['actual_updates']+=update
        v['arms'][arm]['forward_calls']+=forward;v['arms'][arm]['actual_updates']+=update
        # Charged BEFORE the operation; a failed call never receives a refund.


def completed(c,arm,*,forward=0,update=0):
    with ledger(c) as v:
        v['forwards_completed']+=forward;v['updates_completed']+=update
        v['arms'][arm]['forwards_completed']+=forward;v['arms'][arm]['updates_completed']+=update
        require(v['forwards_completed']<=v['forward_calls'] and v['updates_completed']<=v['actual_updates'],'completed operation lacks reservation')


def guard_memory(c):
    import torch
    used=int(torch.cuda.max_memory_allocated()) if c['profile']=='LAB_GPU_ONCE' else 0
    if c.get('fixture',{}).get('fault')=='allocated_overrun':used=18*resource.GIB+1
    with ledger(c) as v:budget_guard(c,v,allocated=used)


def supervised_process(c,arm):
    """Second watchdog bounds a lost/hung supervisor and reaps only its tree."""
    from ..localization import guard
    require(__import__('ctypes').CDLL(None).prctl(36,1,0,0,0)==0,'controller subreaper unavailable')
    with ledger(c) as v:
        timeout=min(900-gpu_seconds(v),1800-(time.monotonic()-v['started_monotonic']))-5
    require(timeout>0,'outer deadline reserve exhausted')
    p=subprocess.Popen(ct.command(c,'supervise',arm),cwd=ct.ROOT,env=ct.environment(c),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    start_id=guard.identity(p.pid);tracked={} if start_id is None else {p.pid:start_id['start']}
    error=None;cleanup=[];stdout=b'';stderr=b''
    try:stdout,stderr=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:error='outer supervisor deadline'
    finally:
        rows=guard.processes()
        for pid in guard.descendants(os.getpid(),rows):tracked[pid]=rows[pid]['start']
        for sig in (signal.SIGTERM,signal.SIGKILL):
            live=[pid for pid,s in tracked.items() if (v:=guard.identity(pid)) and v['start']==s and v['state'] not in ('Z','X')]
            if live and error is None:error='supervisor/worker residue'
            for pid in live:
                try:guard.signal_owned(pid,tracked[pid],sig)
                except OSError as exc:cleanup.append(repr(exc))
            if live:time.sleep(.2)
        if p.poll() is None:p.wait(timeout=2)
        if error is not None:
            stdout,stderr=p.communicate(timeout=2)
        while True:
            try:
                if os.waitpid(-1,os.WNOHANG)[0]==0:break
            except ChildProcessError:break
        remaining=[pid for pid,s in tracked.items() if (v:=guard.identity(pid)) and v['start']==s and v['state'] not in ('Z','X')]
    root=Path(c['job_root']);(root/('accept-'+arm+'-supervisor.stdout')).write_bytes(stdout)
    (root/('accept-'+arm+'-supervisor.stderr')).write_bytes(stderr)
    publish(root/('accept-'+arm+'-PROCESS-EXIT.json'),dict(returncode=p.returncode,error=error,cleanup_errors=cleanup,remaining_live=remaining,
        stdout_sha256=__import__('hashlib').sha256(stdout).hexdigest(),stderr_sha256=__import__('hashlib').sha256(stderr).hexdigest()))
    return p.returncode if error is None and not cleanup and not remaining else 2


def material_refs(c):
    for role in ('train_core','development'):ct.exact(c['data'][role]['manifest'])
    if c['profile']=='LAB_GPU_ONCE':require(reference(c['pretrained']['path'])==c['pretrained'],'pretrained changed')


def supervise(c,arm):
    from ..localization import guard
    root=Path(c['job_root']);out=root/('accept-'+arm);out.mkdir(exist_ok=False)
    require(read(root/('accept-'+arm+'-CONSUMED.json'))['contract_sha256']==c['contract_sha256'],'stage not consumed')
    require(__import__('ctypes').CDLL(None).prctl(36,1,0,0,0)==0,'subreaper unavailable')
    owner=dict(pid=os.getpid(),start=guard.identity(os.getpid())['start'],contract_sha256=c['contract_sha256'])
    publish(out/'supervisor-owner.json',owner)
    first=resource.snapshot(c,entry=True);publish(out/'resource-entry.json',first)
    child=None;tracked={};error=None;cleanup=[];remaining=[];started=time.monotonic()
    def deadline(*args):raise TimeoutError('hard acceptance stage/wall deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
    signal.signal(signal.SIGINT,lambda *a:(_ for _ in ()).throw(RuntimeError('supervisor interrupted')))
    try:
        with ledger(c) as v:
            remaining_seconds=min(900-gpu_seconds(v)-15,1800-(time.monotonic()-v['started_monotonic'])-15)
        require(remaining_seconds>0,'deadline cleanup reserve exhausted')
        signal.setitimer(signal.ITIMER_REAL,remaining_seconds)
        env=ct.environment(c,gpu=c['profile']=='LAB_GPU_ONCE');env['H2_SUPERVISOR_PID']=str(os.getpid())
        argv=ct.command(c,'worker',arm);publish(out/'worker-command.json',dict(argv=argv,cwd=str(ct.ROOT),CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES']))
        with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr,(out/'resources.jsonl').open('x') as journal:
            child=subprocess.Popen(argv,cwd=ct.ROOT,env=env,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
            born=guard.identity(child.pid)
            if born:tracked[child.pid]=born['start']
            while True:
                rows=guard.processes();own=guard.descendants(child.pid,rows)|({child.pid} if child.pid in rows else set())
                for pid in own:tracked[pid]=rows[pid]['start']
                with ledger(c) as v:budget_guard(c,v)
                sample=resource.snapshot(c,entry=False,baseline=first,owned=own)
                journal.write(json.dumps(sample,allow_nan=False)+'\n');journal.flush()
                code=child.poll()
                if code is not None:require(code==0,'worker exit '+str(code));break
                time.sleep(.1 if c['profile']=='CPU_FIXTURE_ONLY' else 2)
        exit_record=read(out/'worker-exit.json');result=read(out/'RESULT.json')
        require(exit_record['status']=='PASS' and exit_record['pid']==child.pid,'worker exit identity differs')
        require(result['status']=='PASS' and result['arm']==arm and result['actual_updates']==3 and result['forward_calls']==6 and result['full_state_RNG_exact'],'worker result incomplete')
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None:
            rows=guard.processes()
            for pid in guard.descendants(os.getpid(),rows):tracked[pid]=rows[pid]['start']
            for sig in (signal.SIGTERM,signal.SIGKILL):
                live=[p for p,s in tracked.items() if (v:=guard.identity(p)) and v['start']==s and v['state'] not in ('Z','X')]
                if live and error is None:error=dict(error='owned worker residue',live=live)
                for pid in live:
                    try:guard.signal_owned(pid,tracked[pid],sig)
                    except OSError as exc:cleanup.append(repr(exc))
                if live:time.sleep(.2)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:cleanup.append('direct child did not exit')
            # Reap only this supervisor's adopted descendants.
            while True:
                try:
                    if os.waitpid(-1,os.WNOHANG)[0]==0:break
                except ChildProcessError:break
            remaining=[p for p,s in tracked.items() if (v:=guard.identity(p)) and v['start']==s and v['state'] not in ('Z','X')]
        value=dict(status='PASS' if error is None and not cleanup and not remaining else 'STOP',arm=arm,
            contract_sha256=c['contract_sha256'],returncode=None if child is None else child.returncode,
            error=error,remaining_live=remaining,cleanup_errors=cleanup,tracked=tracked,elapsed_seconds=time.monotonic()-started)
        publish(out/'supervisor-exit.json',value)
    if value['status']!='PASS':publish(out/'SUPERVISOR_STOP.json',value);return 2
    publish(out/'CLOSED.json',dict(status='PASS',arm=arm,contract_sha256=c['contract_sha256'],result=reference(out/'RESULT.json'),
        supervisor_exit=reference(out/'supervisor-exit.json'),worker_exit=reference(out/'worker-exit.json')))
    return 0


def worker(c,arm):
    from ..localization import guard
    from . import h2_worker
    out=Path(c['job_root'])/('accept-'+arm)
    owner=read(out/'supervisor-owner.json');parent=guard.identity(os.getppid())
    require(owner['pid']==os.getppid() and str(os.getppid())==os.environ.get('H2_SUPERVISOR_PID') and parent['start']==owner['start'],'worker supervisor ownership differs')
    publish(out/'worker-start.json',dict(pid=os.getpid(),ppid=os.getppid(),contract_sha256=c['contract_sha256']))
    result=None;error=None
    try:
        fault=c.get('fixture',{}).get('fault','none')
        if fault=='watchdog_deadline':time.sleep(20)  # constructed worker stall; real watchdog must terminate it
        if fault in ('time_overrun','wall_overrun'):
            with ledger(c) as v:
                if fault=='time_overrun':v['GPU_seconds']=901
                else:v['started_monotonic']=time.monotonic()-1801
            with ledger(c) as v:budget_guard(c,v)
        result=h2_worker.run(c,arm,out,lambda a:reserve(c,a,forward=1),lambda a:reserve(c,a,update=1),lambda:guard_memory(c),
            lambda a:completed(c,a,forward=1),lambda a:completed(c,a,update=1))
        return 0
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc());publish(out/'STOP.json',dict(status='STOP_NO_RETRY',**error));return 2
    finally:
        publish(out/'worker-exit.json',dict(status='PASS' if error is None and result is not None else 'STOP',pid=os.getpid(),
            contract_sha256=c['contract_sha256'],error=error))


def verify(c):
    root=Path(c['job_root']);require(not (root/'STOP.json').exists(),'root has STOP')
    result=read(root/'RESULT.json');closed=read(root/'CLOSED.json')
    require(result['status']=='PASS' and result['actual_updates']==9 and result['forward_calls']==18 and result['scope']==ct.SCOPE,'root result differs')
    require(result['contract_sha256']==closed['contract_sha256']==c['contract_sha256'],'terminal contract differs')
    ct.exact(closed['result']);ct.exact(closed['export_index'])
    cost=read(ledger_path(c));require(cost['active'] is None and cost['actual_updates']==9 and cost['forward_calls']==18,'ledger not complete')
    require(cost['updates_completed']==9 and cost['forwards_completed']==18,'actual completed operations differ from reservations')
    require(cost['GPU_seconds']<=900 and result['wall_seconds']<=1800,'final resource budget exceeded')
    for arm in ct.ARMS:
        out=root/('accept-'+arm);cl=read(out/'CLOSED.json');r=ct.exact(cl['result']);exit_record=ct.exact(cl['supervisor_exit'])
        require(cl['contract_sha256']==c['contract_sha256'] and r['state_usable_for_formal'] is False,'stage identity/scope differs')
        require(exit_record['status']=='PASS' and not exit_record['remaining_live'] and not exit_record['cleanup_errors'],'supervisor residue')
        require(r['forward_calls']==6 and r['actual_updates']==3 and len(r['logits_dtypes'])==6 and set(r['logits_dtypes'])=={'torch.bfloat16'},'actual BF16/count coverage differs')
    return result


def export(c,*,success):
    root=Path(c['job_root']);target=root/'slim-export';target.mkdir(exist_ok=False)
    names=['RESULT.json','STOP.json','EXECUTION_EXIT.json']
    sources=[(n,root/n) for n in names if (root/n).exists()]
    sources += [('contract.json',Path(c['_path'])),('CUMULATIVE_COST.json',ledger_path(c)),('source-manifest.json',Path(c['source']['path']))]
    for arm in ct.ARMS:
        out=root/('accept-'+arm)
        for name in ('RESULT.json','CLOSED.json','supervisor-exit.json','worker-exit.json','STOP.json','RESTORE.json','C1.pth.complete.json'):
            if (out/name).is_file():sources.append((arm+'-'+name,out/name))
    refs=[]
    require(sum(p.stat().st_size for _,p in sources)<=20*1024**2,'slim export exceeds20MiB')
    for name,p in sources:
        require(not p.is_symlink() and p.stat().st_size<=2*1024**2,'unexpected large slim member')
        dest=target/name;dest.write_bytes(p.read_bytes());require(sha(dest)==sha(p),'slim export SHA differs')
        refs.append(dict(name=name,source=reference(p),returned=reference(dest)))
    publish(target/'SHA_INDEX.json',dict(status='PASS' if success else 'STOP',files=refs,
        checkpoints='SERVER_ONLY_NOT_DOWNLOADED',state_usable_for_formal=False,scientific_quality_evaluated=False))
    return reference(target/'SHA_INDEX.json')


def run(c):
    root=Path(c['job_root']);token=read(Path(c['_path']).parent/'LAUNCH_CONSUMED.json')
    require(token['contract_sha256']==c['contract_sha256'] and token['nonce']==c['nonce'],'unique launch token differs')
    root.mkdir(exist_ok=False)
    publish(root/'RUN-CONSUMED.json',dict(contract_sha256=c['contract_sha256'],nonce=c['nonce'],pid=os.getpid(),no_retry=True))
    publish(ledger_path(c),dict(task_id=ct.TASK,contract_sha256=c['contract_sha256'],total_forward_cap=146596,
        profile=c['profile'],fixture_only=c['profile']=='CPU_FIXTURE_ONLY',failed_reservations_refunded=False,
        reservation_semantics='forward_calls/actual_updates charge attempted entries before operation; completed fields prove successful completion',
        forwards_completed=0,updates_completed=0,
        prior=ct.exact(c['prior_cost']),forward_calls=0,actual_updates=0,GPU_seconds=0,started_monotonic=time.monotonic(),
        arms={a:dict(forward_calls=0,actual_updates=0,forwards_completed=0,updates_completed=0) for a in ct.ARMS},active=None,stages=[]))
    started=token['monotonic'];done=[];error=None
    with ledger(c) as v:v['started_monotonic']=started
    try:
        for arm in ct.ARMS:
            resource.snapshot(c,entry=True)
            publish(root/('accept-'+arm+'-CONSUMED.json'),dict(contract_sha256=c['contract_sha256'],arm=arm,no_retry=True))
            with ledger(c) as v:
                if c.get('fixture',{}).get('fault')=='watchdog_deadline':v['GPU_seconds']=875
                budget_guard(c,v);v['active']=dict(arm=arm,monotonic=time.monotonic())
            code=supervised_process(c,arm)
            with ledger(c) as v:
                elapsed=time.monotonic()-v['active']['monotonic'];v['GPU_seconds']+=elapsed;v['active']=None
                v['stages'].append(dict(arm=arm,seconds=elapsed,returncode=code))
            require(code==0,'supervised acceptance STOP: '+arm)
            cl=read(root/('accept-'+arm)/'CLOSED.json');ct.exact(cl['result']);ct.exact(cl['supervisor_exit']);done.append(arm)
        with ledger(c) as v:budget_guard(c,v)
        require(read(ledger_path(c))['forward_calls']==18 and read(ledger_path(c))['actual_updates']==9,'actual totals differ')
        publish(root/'RESULT.json',dict(status='PASS',scope=ct.SCOPE,contract_sha256=c['contract_sha256'],
            actual_updates=9,forward_calls=18,arms=done,scientific_quality_evaluated=False,state_usable_for_formal=False,
            fixture_only=c['profile']=='CPU_FIXTURE_ONLY',wall_seconds=time.monotonic()-started,remaining_future_forwards=146578))
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc())
        publish(root/'STOP.json',dict(status='STOP_NO_RETRY',completed_arms=done,contract_sha256=c['contract_sha256'],**error))
    finally:
        with ledger(c) as v:
            if v['active'] is not None:
                v['GPU_seconds']+=time.monotonic()-v['active']['monotonic'];v['active']=None
        publish(root/'EXECUTION_EXIT.json',dict(status='PASS' if error is None else 'STOP',returncode=0 if error is None else 2,
            contract_sha256=c['contract_sha256'],error=error,formal_training_started=False))
    index=export(c,success=error is None)
    if error is not None:return 2
    publish(root/'CLOSED.json',dict(status='PASS',contract_sha256=c['contract_sha256'],result=reference(root/'RESULT.json'),export_index=index))
    verify(c);return 0


def launch(c,session,foreground):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','launcher must hide CUDA')
    require(re.fullmatch(r'[a-z0-9-]{3,80}',session) is not None,'invalid session')
    root=Path(c['job_root']);token=Path(c['_path']).parent/'LAUNCH_CONSUMED.json'
    require(not root.exists() and not token.exists() and not ledger_path(c).exists(),'duplicate launch/run/ledger refused')
    material_refs(c);resource.snapshot(c,entry=True)
    if foreground:require(c['profile']=='CPU_FIXTURE_ONLY','production foreground refused')
    else:
        require(c['profile']=='LAB_GPU_ONCE','CPU tmux launch refused')
        probe=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
        require(probe.returncode==1,'tmux name exists or inventory unavailable')
    publish(token,dict(contract_sha256=c['contract_sha256'],nonce=c['nonce'],session=session,no_retry=True,epoch=time.time(),monotonic=time.monotonic()))
    if foreground:return subprocess.run(ct.command(c,'run'),cwd=ct.ROOT,env=ct.environment(c)).returncode
    env=ct.environment(c)
    command=['env',*[f'{k}={env[k]}' for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_THREADING_LAYER')],*ct.command(c,'run')]
    proc=subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ct.ROOT),shlex.join(command)],capture_output=True)
    publish(token.with_name('LAUNCH_RECEIPT.json'),dict(returncode=proc.returncode,stdout=proc.stdout.decode(),stderr=proc.stderr.decode(),argv=command))
    require(proc.returncode==0,'unique tmux launch failed; no retry')
    print(json.dumps(dict(status='STARTED',session=session,job_root=str(root))))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','bind','contract-check','cuda-check','launch','run','supervise','worker','verify'))
    for n in ('spec','source','output','contract','cpu-gate','prepare-gate','authorization'):p.add_argument('--'+n,type=Path)
    p.add_argument('--arm',choices=ct.ARMS);p.add_argument('--session');p.add_argument('--foreground',action='store_true')
    a=p.parse_args()
    try:
        if a.action in ('prepare','bind'):
            require(a.spec and a.source and a.output,'raw spec/source/output required')
            c=ct.base(a.spec,reference(a.source))
            if a.action=='prepare':
                require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','prepare must hide CUDA')
                from .h2_worker import prepare
                prepare(c,a.output);return 0
            require(a.cpu_gate and a.prepare_gate and a.authorization,'bind evidence missing')
            c.update(CPU_gate=reference(a.cpu_gate),prepare_gate=reference(a.prepare_gate),authorization=reference(a.authorization))
            ct.validate(c);material_refs(c);resource.snapshot(c,entry=True)
            ct.seal(c,a.output);print(json.dumps(dict(status='BOUND_ACCEPTANCE_ONLY')));return 0
        require(a.contract,'explicit contract required');c=ct.load(a.contract)
        if a.action=='contract-check':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','contract-check must hide CUDA')
            import torch
            require(not torch.cuda.is_initialized() and not torch.cuda.is_available(),'contract-check CUDA exposed')
            material_refs(c);resource.snapshot(c,entry=True)
            print(json.dumps(dict(status='PASS',scope=ct.SCOPE,resources=c['resources'],CUDA_initialized=False,forward_calls=0)));return 0
        if a.action=='cuda-check':
            require(c['profile']=='LAB_GPU_ONCE' and a.output,'real CUDA check output required')
            from .h2_cuda_identity import inspect_cuda_identity
            inspect_cuda_identity(c['gpu_uuid'],a.output);return 0
        if a.action=='launch':require(a.session,'session required');return launch(c,a.session,a.foreground)
        if a.action=='run':return run(c)
        if a.action=='verify':verify(c);print(json.dumps(dict(status='VERIFIED_PASS')));return 0
        require(a.arm,'explicit arm required')
        return supervise(c,a.arm) if a.action=='supervise' else worker(c,a.arm)
    except BaseException as exc:
        print(json.dumps(dict(status='STOP_NO_RETRY',error=repr(exc),traceback=traceback.format_exc())),file=sys.stderr)
        return 2
