"""Real denied /proc/io and exit on owned stdlib children; no model/CUDA."""
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
from ..localization import guard,channel,process_io
from ..localization.runio import require,publish,event,read,reference

CHILD="""import ctypes,time,sys,os
assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
print('READY',flush=True)
sys.stdin.readline()
assert ctypes.CDLL(None).prctl(4,1,0,0,0)==0
print('READABLE',flush=True)
sys.stdin.readline()
"""

def worker(c,out,holder):
    import torch
    require(not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']=='','CPU monitor initialized CUDA')
    child=subprocess.Popen([sys.executable,'-u','-c',CHILD],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    checks=[]
    try:
        require(child.stdout.readline().strip()=='READY','CPU child did not enter denied state')
        ident=guard.identity(child.pid);require(ident is not None,'CPU child identity unavailable')
        rows={child.pid:ident};denied=process_io.observe([child.pid],rows,guard.identity,runtime=True)
        require(denied['bytes'] is None and denied['members'][0]['status']=='UNAVAILABLE' and denied['members'][0]['errno']==13,'real permission denial not represented')
        checks.append('real_PR_SET_DUMPABLE0_permission_denied')
        p=guard.Progress(0);p.observe(0,cpu=0,io=None,io_detail=denied,files=(),marker=1,gpu_busy=False)
        for n in range(1,5):r=p.observe(n,cpu=0,io=None,io_detail=denied,files=(),marker=1,gpu_busy=False,idle=5)
        require(p.active==p.semantic==0 and not r['semantic_change'],'unavailable I/O reset no-progress timer')
        child.stdin.write('\n');child.stdin.flush();require(child.stdout.readline().strip()=='READABLE','CPU child did not restore readable state')
        readable=process_io.observe([child.pid],rows,guard.identity,runtime=True);require(readable['members'][0]['status']=='OBSERVED','real accessible I/O not read')
        r=p.observe(5,cpu=0,io=readable['bytes'],io_detail=readable,files=(),marker=1,gpu_busy=False,idle=5)
        require(r['stop_reason']=='NO_PROGRESS_ALL_OBSERVABLES' and not r['io_activity'],'availability toggle bypassed real inactivity')
        checks.append('real_live_denied_then_readable_no_progress_stops')
        p2=guard.Progress(0);p2.observe(0,cpu=0,io=None,io_detail=denied,files=(),marker=1,gpu_busy=False)
        r=p2.observe(1800,cpu=999,io=None,io_detail=denied,files=(),marker=1,gpu_busy=True)
        require(r['stop_reason']=='SINGLE_OPERATION_NO_COMPLETION','CPU/GPU activity bypassed semantic deadline');checks.append('busy_CPU_GPU_still_semantic_timeout1800')
        # Real disappearance between stat and I/O. The callback ends only this owned child.
        fired=False
        def race(pid):
            nonlocal fired
            if not fired:
                fired=True;before=guard.identity(pid);child.stdin.write('\n');child.stdin.flush();child.wait(timeout=5);return before
            return guard.identity(pid)
        ended=process_io.observe([child.pid],rows,race,runtime=True)
        require(ended['bytes'] is None and ended['members'][0]['status']=='EXITED','real exit race turned into available zero');checks.append('real_exit_between_identity_and_io')
        require(process_io.observe([child.pid],rows,guard.identity,runtime=True)['members'][0]['status']=='EXITED','reaped child not exited');checks.append('real_reaped_process')
        try:process_io.observe([os.getpid()],{os.getpid():{'start':'wrong'}},guard.identity,runtime=True)
        except RuntimeError:checks.append('wrong_PID_start_rejected')
        else:raise RuntimeError('wrong identity accepted')
        # Targeted malformed /proc fixture is explicitly synthetic; unknown I/O errors are fatal.
        fixture=out/'proc-fixture'/str(os.getpid());fixture.mkdir(parents=True);(fixture/'io').write_text('read_bytes: invalid\nwrite_bytes: 0\n')
        try:process_io.observe([os.getpid()],{os.getpid():guard.identity(os.getpid())},guard.identity,runtime=True,proc_root=fixture.parent)
        except ValueError:checks.append('malformed_counter_not_ignored')
        else:raise RuntimeError('malformed counter accepted')
        p3=guard.Progress(0);row={'status':'OBSERVED','bytes':100,'members':[{'pid':1,'start':'fixed','status':'OBSERVED','bytes':100}]}
        p3.observe(0,cpu=0,io=100,io_detail=row,files=(),marker=1,gpu_busy=False)
        row['bytes']=101;row['members'][0]['bytes']=101;r=p3.observe(4,cpu=0,io=101,io_detail=row,files=(),marker=1,gpu_busy=False)
        require(r['io_activity'] and p3.active==4,'authenticated increasing I/O not activity');checks.append('same_PID_monotonic_IO_counts_activity')
        missing={'status':'PARTIALLY_UNAVAILABLE','bytes':None,'members':[{'pid':1,'start':'fixed','status':'UNAVAILABLE','bytes':None}]}
        r=p3.observe(9,cpu=0,io=None,io_detail=missing,files=(),marker=1,gpu_busy=False,idle=5)
        require(r['stop_reason']=='NO_PROGRESS_ALL_OBSERVABLES','readability loss reset timer');checks.append('unavailable_transition_no_timer_reset')
        value={'status':'PASS','checks':checks,'count':len(checks),'real_denied':denied,'real_readable':readable,'real_exit_race':ended,
            'CPU_models_constructed':0,'CUDA_initialized':False,'owned_child_reaped':child.poll() is not None,'child_identity':ident,'source_sha256':c['source_sha256']}
        publish(out/'RESULT.json',value);event(out,'cpu_monitor_verified',checks=len(checks),CUDA_initialized=False,result=reference(out/'RESULT.json'))
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
        child.stdin.close();child.stdout.close()
