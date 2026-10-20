"""Final CLI -> run -> supervisor -> worker -> strict closure -> slim export.

No automatic resume/retry. The production and explicit CPU profile use the
same dispatch, stage protocol, resource contract and serialization interfaces.
"""
from __future__ import annotations
import argparse, json, os, platform, signal, subprocess, sys, time, traceback
from pathlib import Path
from . import distill_runtime as rt


def environment(c,gpu=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'] if gpu else '',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='2' if gpu else '1',MKL_NUM_THREADS='2' if gpu else '1',
             OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
    env.pop('PYTHONPATH',None)
    return env


def command(c,action,stage=None):
    argv=[c['python'],'-B',str(rt.ROOT/'r18_distill.py'),action,'--contract',c['_path']]
    if stage is not None:argv+=['--stage',stage]
    return argv


def gpu_stage(c,stage):
    return c['profile']=='LAB_GPU_ONCE' and (stage=='teacher' or stage.startswith(('accept-','train-','infer-')) or stage=='cost')


def predecessors(c,stage):
    for previous in rt.STAGES[:rt.STAGES.index(stage)]:rt.verify_stage(c,previous)


def supervise(c,stage):
    from ..localization import guard
    from ..localization.process_io import observe as observe_io
    out=rt.stage_path(c,stage)
    consumed=rt.read(Path(c['job_root'])/(stage+'-CONSUMED.json'))
    rt.require(consumed['identity']==rt.identity(c,stage),'phase token differs')
    predecessors(c,stage)
    out.mkdir(exist_ok=False)
    gpu=gpu_stage(c,stage);first=rt.snapshot(c,entry=True,gpu=gpu)
    rt.publish(out/'resource-entry.json',first)
    owner=dict(pid=os.getpid(),identity=rt.identity(c,stage))
    if platform.system()=='Linux':
        import ctypes
        rt.require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'subreaper unavailable')
        owner['start']=guard.identity(os.getpid())['start']
    else:
        rt.require(c['profile']=='CPU_FIXTURE_ONLY','non-Linux production supervisor refused')
        owner['start']='CPU_FIXTURE_WINDOWS'
    rt.publish(out/'supervisor-owner.json',owner)
    child=None;tracked={};cleanup=[];remaining=[];error=None;seen=False;writer=None
    interrupted=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:interrupted.append(s))
    progress=guard.Progress(time.monotonic());start=time.monotonic();last_snapshot=first
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command(c,'worker',stage),cwd=rt.ROOT,env=environment(c,gpu),
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            with (out/'resources.jsonl').open('x',encoding='utf-8') as resources:
                while True:
                    last_snapshot=rt.snapshot(c,entry=False,gpu=gpu,baseline=first)
                    rt.atomic(out/'resource-live.json',last_snapshot)
                    resources.write(json.dumps(last_snapshot,allow_nan=False)+'\n');resources.flush()
                    if writer is None and (out/'worker-start.json').exists():
                        start_record=rt.read(out/'worker-start.json')
                        rt.require(start_record['identity']==rt.identity(c,stage) and start_record['pid']==child.pid,'worker process handshake differs')
                        writer=start_record['pid']
                        rt.publish(out/'supervisor-ack.json',dict(identity=rt.identity(c,stage),writer_pid=writer,supervisor_pid=os.getpid()))
                    marker=guard.read_progress(out/'progress.json',seen=seen)
                    if marker is not None:
                        events=rt.verify_events(c,stage,out/'events.jsonl',complete=False)
                        rt.require(writer is not None and all(e['pid']==writer for e in events),'foreign event writer')
                        rt.require(0<marker['number']<=len(events) and events[marker['number']-1]['sha256']==marker['sha256'],'progress/event binding differs')
                        seen=True
                    if platform.system()=='Linux':
                        rows=guard.processes();own=guard.descendants(child.pid,rows)|({child.pid} if child.pid in rows else set())
                        for pid in own:tracked[pid]=rows[pid]['start']
                        if gpu:rt.require(all(a['pid'] in own for a in last_snapshot['compute_apps']),'foreign GPU process outside owned worker tree')
                        live=[pid for pid in own if rows[pid]['state'] not in ('Z','X')]
                        io=observe_io(live,rows,guard.identity,runtime=True)
                        cpu=sum(rows[pid]['cpu'] for pid in live)
                        busy=gpu and any(d['utilization.gpu']['value']>1 for d in last_snapshot['snapshot']['gpu']['devices'])
                        activity=progress.observe(time.monotonic(),cpu=cpu,io=io['bytes'],files=guard.file_activity(out),
                            marker=None if marker is None else (marker['number'],marker['sha256']),gpu_busy=busy,io_detail=io)
                        rt.require(not activity['stop_reason'],'progress protection: '+str(activity['stop_reason']))
                    rt.require(not interrupted,'supervisor interrupted')
                    rt.require(not (out/'STOP.json').exists(),'worker STOP receipt')
                    code=child.poll()
                    if code is not None:
                        rt.require(code==0,'worker exit '+str(code));break
                    time.sleep(.05 if c['profile']=='CPU_FIXTURE_ONLY' else c['resources']['monitor_interval_seconds'])
        final_events=rt.verify_events(c,stage,out/'events.jsonl')
        # On Windows the redirector/worker may already have exited; bind its
        # observed writer while live and the exact terminal worker receipt.
        rt.require(writer is not None and rt.read(out/'worker-exit.json')['pid']==writer and all(e['pid']==writer for e in final_events),'terminal writer receipt differs')
        result=rt.read(out/'RESULT.json')
        rt.require(result['identity']==rt.identity(c,stage) and result['status']=='PASS','worker result identity/status')
        rt.require(rt.read(out/'worker-exit.json')['status']=='PASS','worker exit receipt absent')
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        # Only the owned direct child and PID/start-bound descendants are signaled.
        if child is not None:
            if platform.system()=='Linux':
                rows=guard.processes()
                for pid in guard.descendants(child.pid,rows)|guard.descendants(os.getpid(),rows):
                    tracked[pid]=rows[pid]['start']
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    live=[pid for pid,start_id in tracked.items() if (v:=guard.identity(pid)) and v['start']==start_id and v['state'] not in ('Z','X')]
                    if live and error is None:
                        # DataLoader workers may drain after their leader returns.
                        deadline=time.monotonic()+2
                        while live and time.monotonic()<deadline:
                            time.sleep(.05)
                            live=[pid for pid in live if (v:=guard.identity(pid)) and v['start']==tracked[pid] and v['state'] not in ('Z','X')]
                    if live and error is None:error=dict(type='ResidualProcess',message='owned worker tree did not exit naturally')
                    for pid in live:
                        try:guard.signal_owned(pid,tracked[pid],sig)
                        except OSError as exc:cleanup.append(repr(exc))
                    if live:time.sleep(.1)
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill()
            child.wait(timeout=5)
            if platform.system()=='Linux':
                while True:
                    try:
                        if os.waitpid(-1,os.WNOHANG)[0]==0:break
                    except ChildProcessError:break
                remaining=[pid for pid,start_id in tracked.items() if (v:=guard.identity(pid)) and v['start']==start_id and v['state'] not in ('Z','X')]
    value=dict(status='PASS' if error is None and not cleanup and not remaining else 'STOP',
        identity=rt.identity(c,stage),returncode=None if child is None else child.returncode,
        remaining_live=remaining,cleanup_errors=cleanup,error=error,elapsed_seconds=time.monotonic()-start,
        fixture_only=c['profile']=='CPU_FIXTURE_ONLY',last_resource_sample=last_snapshot,tracked=tracked)
    rt.publish(out/'supervisor-exit.json',value)
    if value['status']!='PASS':
        if not (out/'SUPERVISOR_STOP.json').exists():rt.publish(out/'SUPERVISOR_STOP.json',value)
        return 2
    rt.publish(out/'CLOSED.json',dict(status='PASS',identity=rt.identity(c,stage),result=rt.reference(out/'RESULT.json'),
        events=rt.reference(out/'events.jsonl'),supervisor_exit=rt.reference(out/'supervisor-exit.json'),worker_exit=rt.reference(out/'worker-exit.json')))
    rt.verify_stage(c,stage)
    return 0


def run(c):
    root=Path(c['job_root']);root.mkdir(parents=True,exist_ok=False)
    rt.publish(root/'RUN-CONSUMED.json',dict(identity=rt.identity(c,'admission'),nonce=c['nonce'],no_retry=True))
    start=time.monotonic();done=[]
    try:
        for stage in rt.STAGES:
            rt.snapshot(c,entry=True,gpu=gpu_stage(c,stage))
            rt.publish(root/(stage+'-CONSUMED.json'),dict(identity=rt.identity(c,stage),no_retry=True))
            with (root/(stage+'-supervisor.log')).open('x',encoding='utf-8') as log:
                p=subprocess.run(command(c,'supervise',stage),cwd=rt.ROOT,env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            rt.require(p.returncode==0,'supervised stage stopped: '+stage)
            rt.verify_stage(c,stage);done.append(stage)
        comparison=rt.verify_stage(c,'compare')
        rt.publish(root/'RESULT.json',dict(status='PASS',task_id=rt.TASK,contract_sha256=c['contract_sha256'],
            source_sha256=c['source']['sha256'],profile=c['profile'],stages=done,
            quality_passed=comparison['quality_passed'],cost_passed=comparison['cost_passed'],
            elapsed_seconds=time.monotonic()-start,no_retry=True))
        rt.publish(root/'CLOSED.json',dict(status='PASS',result=rt.reference(root/'RESULT.json'),
                   stage_closures=[rt.reference(rt.stage_path(c,s)/'CLOSED.json') for s in rt.STAGES]))
        print(json.dumps(dict(status='PASS',profile=c['profile'],completed_stages=len(done))))
        return 0
    except BaseException as exc:
        rt.publish(root/'STOP.json',dict(status='STOP_NO_RETRY',completed_stages=done,
            error=repr(exc),traceback=traceback.format_exc(),contract_sha256=c['contract_sha256'],
            elapsed_seconds=time.monotonic()-start))
        raise


def bind(spec_path,source_ref,cpu_gate_ref,approval_ref,output):
    """Future authorized binding, no forward. Caller supplies current lab facts."""
    c=rt.read(spec_path)
    c.update(task_id=rt.TASK,protocol=rt.PROTOCOL,source_root=str(rt.ROOT),python=sys.executable,
             commit=rt.verify_source(source_ref)['commit'],source=source_ref,
             science_sha256=rt.digest(rt.recipe()),execution_sha256=rt.digest(rt.execution()),
             stages=list(rt.STAGES),resources=rt.execution()['resources'],CPU_gate=cpu_gate_ref,
             authorization=approval_ref,GPU_authorized=True,science_authorized=True)
    rt.require(c['profile']=='LAB_GPU_ONCE','bind only production; CPU fixtures have separate builder')
    # Validate a complete candidate before publishing any execution contract/token.
    tmp=Path(output).with_suffix('.candidate.json')
    rt.seal_contract(c,tmp)
    valid=rt.load(tmp)
    rt.snapshot(valid,entry=True,gpu=True)
    rt.publish(output,rt.read(tmp))
    return rt.reference(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','supervise','worker','verify','make-source','bind','cuda-preflight'))
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--stage',choices=rt.STAGES)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--cpu-gate',type=Path)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    if args.action=='make-source':
        rt.require(args.output is not None,'source output required');rt.source_receipt(args.output);return 0
    if args.action=='cuda-preflight':
        rt.require(args.output is not None and args.gpu_uuid is not None,'explicit GPU UUID/audit output required')
        from .cuda_identity import inspect_cuda_identity
        value=inspect_cuda_identity(args.gpu_uuid,args.output)
        rt.require(value['BF16_supported'],'frozen BF16 runtime unavailable')
        print(json.dumps(value));return 0
    if args.action=='bind':
        rt.require(all(v is not None for v in (args.spec,args.source,args.cpu_gate,args.authorization,args.output)),'all bind references required')
        bind(args.spec,rt.reference(args.source),rt.reference(args.cpu_gate),rt.reference(args.authorization),args.output);return 0
    rt.require(args.contract is not None,'explicit contract required')
    c=rt.load(args.contract)
    if args.action=='run':return run(c)
    if args.action=='verify':
        for s in rt.STAGES:rt.verify_stage(c,s)
        return 0
    rt.require(args.stage is not None,'explicit stage required')
    if args.action=='supervise':return supervise(c,args.stage)
    from .distill_worker import worker
    return worker(c,args.stage)
