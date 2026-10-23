"""H2 formal lifecycle. Watchdog ownership derived from accepted 7527c464.

New stages/counters are independent; the acceptance-only CLI is unchanged.
"""
import argparse,contextlib,fcntl,hashlib,json,os,re,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
from ..localization.runio import require,read,publish,atomic,reference,digest,sha
from . import h2_formal_contract as ct,h2_formal_resources as resource

def ledger_path(c):return Path(c['task_root'])/'CUMULATIVE_COST.json'

@contextlib.contextmanager
def ledger(c):
    p=ledger_path(c)
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);v=read(p)
        require(v['contract_sha256']==c['contract_sha256'],'formal ledger identity differs')
        yield v
        atomic(p,v)

def gpu_stage(s):return s.startswith(('train-','infer-')) or s=='cost'
def gpu_seconds(v):return v['GPU_seconds']+(time.monotonic()-v['active']['monotonic'] if v['active'] and v['active']['GPU_stage'] else 0)

def budget_guard(c,v,*,forward=0,update=0,arm=None,allocated=0):
    require(time.monotonic()-v['started_monotonic']<237600,'formal wall budget exhausted')
    require(gpu_seconds(v)<216000,'cumulative 60 GPU-stage-hour cap exhausted')
    require(allocated<=18*resource.GIB,'allocated cap exceeded')
    require(v['forward_calls']+forward<=18+ct.total_new(c)<=146596,'cumulative forward cap exhausted')
    require(v['actual_updates']+update<=2*ct.profile(c)['epochs']*ct.profile(c)['windows'],'formal update cap exhausted')
    if arm:
        require(v['arms'][arm]['forward_calls']+forward<=ct.stage_budget(c,arm),'stage forward cap exhausted')
        cap=ct.profile(c)['epochs']*ct.profile(c)['windows'] if arm.startswith('train-') else 0
        require(v['arms'][arm]['actual_updates']+update<=cap,'stage update cap exhausted')
    require(sum(p.stat().st_size for p in Path(c['task_root']).rglob('*') if p.is_file())<100*resource.GIB,'task storage cap exhausted')

def reserve(c,arm,*,forward=0,update=0):
    with ledger(c) as v:
        require(v['active'] and v['active']['arm']==arm,'operation outside active stage')
        budget_guard(c,v,forward=forward,update=update,arm=arm)
        v['forward_calls']+=forward;v['actual_updates']+=update
        v['arms'][arm]['forward_calls']+=forward;v['arms'][arm]['actual_updates']+=update

def guard_memory(c):
    import torch
    used=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
    if c.get('fixture',{}).get('fault')=='allocated_overrun':used=18*resource.GIB+1
    with ledger(c) as v:budget_guard(c,v,allocated=used)

def verify_events(c,stage,path):
    # Bounded full-log scan: SHA, chain, sequence and identity; never read_text().
    previous='0'*64;n=0
    with Path(path).open('rb') as stream:
        for line in stream:
            require(len(line)<=4*1024**2,'oversize event')
            row=json.loads(line);body=row['body'];n+=1
            require(row['sequence']==n and row['previous']==previous and row['sha256']==digest(dict(sequence=n,previous=previous,body=body)),'event chain differs')
            require(body['contract_sha256']==c['contract_sha256'] and body['stage']==stage,'event identity differs')
            previous=row['sha256']
    receipt=read(Path(path).with_name('EVENTS_COMPLETE.json'))
    require(n==receipt['events'] and previous==receipt['last_sha256'] and reference(path)==receipt['file'],'event stream incomplete')
    return receipt

def supervised_process(c,arm):
    """Second watchdog bounds a lost/hung supervisor and reaps only its tree."""
    from ..localization import guard
    require(__import__('ctypes').CDLL(None).prctl(36,1,0,0,0)==0,'controller subreaper unavailable')
    with ledger(c) as v:
        timeout=min(216000-gpu_seconds(v),237600-(time.monotonic()-v['started_monotonic']))-5
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
    root=Path(c['job_root']);(root/(arm+'-supervisor.stdout')).write_bytes(stdout)
    (root/(arm+'-supervisor.stderr')).write_bytes(stderr)
    publish(root/(arm+'-PROCESS-EXIT.json'),dict(returncode=p.returncode,error=error,cleanup_errors=cleanup,remaining_live=remaining,
        stdout_sha256=__import__('hashlib').sha256(stdout).hexdigest(),stderr_sha256=__import__('hashlib').sha256(stderr).hexdigest()))
    return p.returncode if error is None and not cleanup and not remaining else 2


def supervise(c,arm):
    from ..localization import guard
    root=Path(c['job_root']);out=root/(arm);out.mkdir(exist_ok=False)
    require(read(root/(arm+'-CONSUMED.json'))['contract_sha256']==c['contract_sha256'],'stage not consumed')
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
            remaining_seconds=min(216000-gpu_seconds(v)-15,237600-(time.monotonic()-v['started_monotonic'])-15)
        require(remaining_seconds>0,'deadline cleanup reserve exhausted')
        signal.setitimer(signal.ITIMER_REAL,remaining_seconds)
        env=ct.environment(c,gpu=c['profile']=='LAB_GPU_ONCE' and gpu_stage(arm));env['H2_SUPERVISOR_PID']=str(os.getpid())
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
        require(result['status']=='PASS' and result['stage']==arm and result['forward_calls']==ct.stage_budget(c,arm),'worker result incomplete')
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
    verify_events(c,arm,out/'events.jsonl')
    publish(out/'CLOSED.json',dict(status='PASS',arm=arm,contract_sha256=c['contract_sha256'],result=reference(out/'RESULT.json'),
        supervisor_exit=reference(out/'supervisor-exit.json'),worker_exit=reference(out/'worker-exit.json')))
    return 0


def worker(c,arm):
    from ..localization import guard
    from . import h2_formal_worker
    out=Path(c['job_root'])/arm;owner=read(out/'supervisor-owner.json');parent=guard.identity(os.getppid())
    require(owner['pid']==os.getppid() and str(os.getppid())==os.environ.get('H2_SUPERVISOR_PID') and parent['start']==owner['start'],'formal worker owner differs')
    publish(out/'worker-start.json',dict(pid=os.getpid(),ppid=os.getppid(),contract_sha256=c['contract_sha256']))
    result=None;error=None;n=0;previous='0'*64;pending=0
    journal=(out/'events.jsonl').open('xb')
    def emit(event,**value):
        nonlocal n,previous
        n+=1;body=dict(event=event,stage=arm,contract_sha256=c['contract_sha256'],**value)
        row=dict(sequence=n,previous=previous,body=body);row['sha256']=digest(row);previous=row['sha256']
        journal.write((json.dumps(row,allow_nan=False)+'\n').encode());journal.flush()
    def forward():
        nonlocal pending
        if c.get('fixture',{}).get('fault')=='forward_overrun':reserve(c,arm,forward=ct.stage_budget(c,arm)+1)
        reserve(c,arm,forward=1);pending+=1
    def update(*,done=False):
        if done:
            with ledger(c) as v:
                v['arms'][arm]['updates_completed']+=1;v['updates_completed']+=1
        else:
            if c.get('fixture',{}).get('fault')=='update_overrun':reserve(c,arm,update=ct.profile(c)['epochs']*ct.profile(c)['windows']+1)
            reserve(c,arm,update=1)
    def memory():
        nonlocal pending
        if pending:
            with ledger(c) as v:
                v['forwards_completed']+=pending;v['arms'][arm]['forwards_completed']+=pending
            pending=0
        guard_memory(c)
    try:
        if c.get('fixture',{}).get('fault')=='time_overrun':
            with ledger(c) as v:v['GPU_seconds']=216001
            with ledger(c) as v:budget_guard(c,v)
        result=h2_formal_worker.execute(c,arm,out,emit,forward,update,memory)
        with ledger(c) as v:
            a=v['arms'][arm];require(a['forward_calls']==a['forwards_completed']==result['forward_calls'],'stage actual forward count differs')
            require(a['actual_updates']==result.get('actual_updates',0),'stage actual update count differs')
            require(a['updates_completed']==a['actual_updates'],'successful update completion counter differs')
        publish(out/'RESULT.json',dict(result,contract_sha256=c['contract_sha256']))
        return 0
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc());publish(out/'STOP.json',dict(status='STOP_NO_RETRY',**error));return 2
    finally:
        journal.flush();os.fsync(journal.fileno());journal.close()
        publish(out/'EVENTS_COMPLETE.json',dict(file=reference(out/'events.jsonl'),events=n,last_sha256=previous))
        publish(out/'worker-exit.json',dict(status='PASS' if result is not None and error is None else 'STOP',pid=os.getpid(),contract_sha256=c['contract_sha256'],error=error))

def material_refs(c):
    for role in ('train_core','development'):ct.exact(c['data'][role]['manifest'])
    if c['profile']=='LAB_GPU_ONCE':require(reference(c['pretrained']['path'])==c['pretrained'],'ImageNet changed')
    ct.exact(c['selection']);ct.acceptance(c);ct.historical(c)

def export(c,*,success):
    root=Path(c['job_root']);target=root/'slim-export';target.mkdir(exist_ok=False)
    sources=[(n,root/n) for n in ('RESULT.json','STOP.json','EXECUTION_EXIT.json') if (root/n).is_file()]
    sources += [('contract.json',Path(c['_path'])),('CUMULATIVE_COST.json',ledger_path(c)),('source-manifest.json',Path(c['source']['path']))]
    for stage in ct.STAGES:
        out=root/stage
        for n in ('RESULT.json','CLOSED.json','supervisor-exit.json','worker-exit.json','endpoint.json','binding.json','EVENTS_COMPLETE.json','CENTER_SUMMARY.json','STOP.json'):
            if (out/n).is_file():sources.append((stage+'-'+n,out/n))
        if stage=='score':
            for a in ct.ARMS:
                for b in ('600','300'):
                    if (out/f'{a}-{b}.json').is_file():sources.append((f'{a}-{b}.json',out/f'{a}-{b}.json'))
    refs=[];require(sum(p.stat().st_size for _,p in sources)<=20*1024**2,'slim export exceeds20MiB')
    for n,p in sources:
        require(not p.is_symlink() and p.stat().st_size<2*1024**2,'large/unexpected slim artifact')
        dest=target/n;dest.write_bytes(p.read_bytes());require(sha(dest)==sha(p),'export SHA differs');refs.append(dict(name=n,source=reference(p),returned=reference(dest)))
    publish(target/'SHA_INDEX.json',dict(status='PASS' if success else 'STOP',files=refs,full_state_predictions_events_centers='SERVER_ONLY',original_N_STOP_preserved=True))
    return reference(target/'SHA_INDEX.json')

def verify(c,*,closing=False):
    root=Path(c['job_root']);require(not (root/'STOP.json').exists(),'root STOP exists')
    cl=(dict(contract_sha256=c['contract_sha256'],result=reference(root/'RESULT.json'),export_index=reference(root/'slim-export/SHA_INDEX.json')) if closing else read(root/'CLOSED.json'))
    r=ct.exact(cl['result']);ct.exact(cl['export_index'])
    require(cl['contract_sha256']==r['contract_sha256']==c['contract_sha256'] and r['status']=='PASS','root binding/closure differs')
    v=read(ledger_path(c));require(v['active'] is None and v['forward_calls']==v['forwards_completed']==18+ct.total_new(c) and v['actual_updates']==v['updates_completed']==2*ct.profile(c)['epochs']*ct.profile(c)['windows'],'terminal cost/counts differ')
    budget_guard(c,v)
    for s in ct.STAGES:
        p=root/s;cl=read(p/'CLOSED.json');res=ct.exact(cl['result']);e=ct.exact(cl['supervisor_exit']);ct.exact(cl['worker_exit'])
        require(cl['contract_sha256']==c['contract_sha256'] and e['status']=='PASS' and not e['remaining_live'] and not e['cleanup_errors'],'stage not closed')
        require(res['forward_calls']==ct.stage_budget(c,s),'stage forward mismatch');verify_events(c,s,p/'events.jsonl')
    return r

def run(c):
    root=Path(c['job_root']);token=read(Path(c['_path']).parent/'LAUNCH_CONSUMED.json')
    require(token['contract_sha256']==c['contract_sha256'] and token['nonce']==c['nonce'],'launch token differs')
    root.mkdir(exist_ok=False);publish(root/'RUN-CONSUMED.json',dict(contract_sha256=c['contract_sha256'],nonce=c['nonce'],pid=os.getpid(),no_retry=True))
    prior=ct.exact(c['prior_cost'])
    publish(ledger_path(c),dict(task_id=ct.TASK,contract_sha256=c['contract_sha256'],total_forward_cap=146596,prior=prior,profile=c['profile'],fixture_only=c['profile']=='CPU_FIXTURE_ONLY',failed_reservations_refunded=False,
        forward_calls=18,forwards_completed=18,actual_updates=0,updates_completed=0,GPU_seconds=ct.PRIOR_SECONDS,started_monotonic=token['monotonic'],
        arms={a:dict(forward_calls=0,actual_updates=0,forwards_completed=0,updates_completed=0) for a in ct.STAGES},active=None,stages=[]))
    done=[];error=None
    try:
        for stage in ct.STAGES:
            resource.snapshot(c,entry=True)
            publish(root/(stage+'-CONSUMED.json'),dict(contract_sha256=c['contract_sha256'],stage=stage,no_retry=True))
            with ledger(c) as v:budget_guard(c,v);v['active']=dict(arm=stage,monotonic=time.monotonic(),GPU_stage=gpu_stage(stage))
            code=supervised_process(c,stage)
            with ledger(c) as v:
                elapsed=time.monotonic()-v['active']['monotonic'];charged=elapsed if v['active']['GPU_stage'] else 0
                v['GPU_seconds']+=charged;v['active']=None;v['stages'].append(dict(stage=stage,GPU_seconds=charged,wall_seconds=elapsed,returncode=code))
            require(code==0,'supervised formal STOP: '+stage)
            ct.exact(read(root/stage/'CLOSED.json')['result']);done.append(stage)
        v=read(ledger_path(c));budget_guard(c,v)
        require(v['forward_calls']==v['forwards_completed']==18+ct.total_new(c),'final forward counts differ')
        comp=read(root/'compare/RESULT.json')
        publish(root/'RESULT.json',dict(status='PASS',scope=ct.SCOPE,contract_sha256=c['contract_sha256'],stages=done,forward_calls=v['forward_calls'],GPU_seconds=v['GPU_seconds'],
            quality_passed=comp['quality_passed'],cost_passed=comp['cost_passed'],comparison=reference(root/'compare/RESULT.json'),historical_N_status=c['historical_N']['status'],fixture_only=c['profile']=='CPU_FIXTURE_ONLY'))
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc());publish(root/'STOP.json',dict(status='STOP_NO_RETRY',completed_stages=done,contract_sha256=c['contract_sha256'],**error))
    finally:
        with ledger(c) as v:
            if v['active']:
                if v['active']['GPU_stage']:v['GPU_seconds']+=time.monotonic()-v['active']['monotonic']
                v['active']=None
        publish(root/'EXECUTION_EXIT.json',dict(status='PASS' if error is None else 'STOP',returncode=0 if error is None else 2,contract_sha256=c['contract_sha256'],error=error))
    try:
        index=export(c,success=error is None)
        if error:return 2
        verify(c,closing=True)
    except BaseException as exc:
        final_error=dict(error=repr(exc),traceback=traceback.format_exc(),phase='finalization',no_retry=True)
        publish(root/'FINALIZATION_STOP.json',final_error)
        if not (root/'STOP.json').exists():publish(root/'STOP.json',dict(status='STOP_NO_RETRY',**final_error))
        publish(root/'EXECUTION_EXIT_BEFORE_FINALIZATION.json',read(root/'EXECUTION_EXIT.json'))
        atomic(root/'EXECUTION_EXIT.json',dict(status='STOP',returncode=2,contract_sha256=c['contract_sha256'],error=final_error))
        return 2
    publish(root/'CLOSED.json',dict(status='PASS',contract_sha256=c['contract_sha256'],result=reference(root/'RESULT.json'),export_index=index));return 0

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
    p.add_argument('--arm',choices=ct.STAGES);p.add_argument('--session');p.add_argument('--foreground',action='store_true')
    a=p.parse_args()
    try:
        if a.action in ('prepare','bind'):
            require(a.spec and a.source and a.output,'raw spec/source/output required')
            c=ct.base(a.spec,reference(a.source))
            if a.action=='prepare':
                require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','prepare must hide CUDA')
                from .h2_formal_worker import prepare
                prepare(c,a.output);return 0
            require(a.cpu_gate and a.prepare_gate and a.authorization,'bind evidence missing')
            c.update(CPU_gate=reference(a.cpu_gate),prepare_gate=reference(a.prepare_gate),authorization=reference(a.authorization))
            ct.validate(c);material_refs(c);resource.snapshot(c,entry=True)
            ct.seal(c,a.output);print(json.dumps(dict(status='BOUND_FORMAL_UD_WITH_HISTORICAL_N')));return 0
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
