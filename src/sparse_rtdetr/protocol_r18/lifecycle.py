"""Server-owned one-shot conditional chain, independent of Windows lifetime."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from ..localization import guard,channel
from ..localization.runio import require,read,publish,reference,original_process_exited
from .model import ROOT
from . import transport
MODES=('gpu-recovery','formal','inference','evaluation','comparison')

def env(c):
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
def command(c,mode):return [c['python'],'-B',str(ROOT/'tools/protocol_r18/deploy.py'),'supervise','--contract',c['contract_path'],'--mode',mode]

def verify_token(c,mode):
    transport.install()
    tok=read(Path(c['job_root'])/(mode+'-launch-consumed.json'))
    require(tok['command']==command(c,mode) and tok['source_sha256']==c['source_sha256'] and tok['contract_sha256']==c['contract_sha256'] and tok['no_retry'] is True,'new launch token differs')

def complete(c,mode):
    transport.install()
    if mode=='gpu-acceptance' and c.get('acceptance_reuse'):
        from .reuse import verify
        return verify(c)
    root=Path(c['job_root']);runtime=Path(c['runtime_root'])/mode;out=root/mode
    d=read(root/(mode+'-dispatch.json'));require(original_process_exited(d,guard.identity(d['pid'])),'phase supervisor has not exited')
    ex=read(runtime/'supervisor-exit.json')
    require(ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'phase not complete clean PASS: '+mode)
    from .control import identity
    b={**identity(c,mode),'supervisor_pid':d['pid']};channel.verify_archive(runtime,out,b)
    require(read(out/'execution-binding.json')==read(runtime/'identity.json'),'worker execution binding differs')
    for n,ref in ex['completion_receipts'].items():require(reference(ref['path'])==ref,'complete receipt bytes changed: '+n)
    require(read(out/'LIFECYCLE_CHECK.json')['status']=='PASS','outer channel/exit verification absent')
    r=read(out/'RESULT.json')
    if mode=='gpu-recovery':
        require(r['status']=='PASS' and r['actual_updates']==3 and r['sequence']==[26753,26754,26754] and r['original_parent']==c['resume_parent']['checkpoint'] and r['same_next_result_exact'] and r['complete_state_RNG_exact'] and not r['state_usable_for_formal'],'fixed3 GPU acceptance not exact')
        from ..localization.thermal_resume import under_load_proof
        under_load_proof(c,runtime)
    elif mode=='formal':require(r['status']=='R18_COMMON_E120_EMA_LOCKED' and r['updates']==36480 and r['actual_updates']==9728 and r['original_completed_updates']==26752 and r['original_parent']==c['resume_parent']['checkpoint'] and not r['used_acceptance_state'] and not r['used_R0_or_v2a_state'],'native independent fixed endpoint absent')
    elif mode=='inference':require(r['status']=='PASS' and r['images']==548 and r['model_batch_forwards']==137 and r['complete_parent_EMA_optimizer_RNG_unchanged'],'native inference scope/state failed')
    else:require(r['status']=='PASS','endpoint CPU stage failed')
    return r

def prerequisites(c,mode):
    transport.install()
    require(mode in MODES,'unknown unique phase')
    for earlier in MODES[:MODES.index(mode)]:complete(c,earlier)

def launch(c,mode):
    from .control import install_resources
    install_resources();prerequisites(c,mode);root=Path(c['job_root'])
    require(not (mode=='gpu-acceptance' and c.get('acceptance_reuse')),'reused acceptance must never be relaunched')
    require(not (root/mode).exists() and not (root/(mode+'-launch-consumed.json')).exists(),'phase already used; no relaunch')
    s=guard.snapshot(root);gpu=mode in ('gpu-recovery','formal','inference')
    guard.screen(s,gpu_uuid=c['gpu_uuid'],gpu=gpu,entry=True,disk_bytes=40*1024**3+(122*400*1024**2 if mode=='formal' else 0))
    require(s['boot_id']==c['boot_id'],'boot changed before unique phase')
    cmd=command(c,mode);publish(root/(mode+'-launch-consumed.json'),{'command':cmd,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'no_retry':True,'epoch':time.time()})
    with (root/(mode+'-dispatcher.log')).open('x') as f:p=subprocess.Popen(cmd,env=env(c),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    d={'pid':p.pid,'process_identity':guard.identity(p.pid),'mode':mode,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':time.time()}
    publish(root/(mode+'-dispatch.json'),d);return p,d

def start_chain(c,path):
    transport.install()
    require(c.get('GPU_execution_authorized') is True and c.get('formal_training_authorized') is True,'complete chain not authorized')
    root=Path(c['job_root']);cmd=[c['python'],'-B',str(ROOT/'tools/protocol_r18/deploy.py'),'chain','--contract',str(path)]
    require(not c.get('acceptance_reuse') and c['recovery_workload']['sequence']==[26753,26754,26754],'dedicated e001 recovery chain only')
    modes=list(MODES)
    publish(root/'chain-consumed.json',{'command':cmd,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'modes':modes,'no_retry':True,'epoch':time.time()})
    with (root/'chain.log').open('x') as f:p=subprocess.Popen(cmd,env=env(c),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    d={'pid':p.pid,'process_identity':guard.identity(p.pid),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':time.time()}
    publish(root/'chain-dispatch.json',d);return d

def chain(c,path):
    from .control import install_resources
    install_resources();root=Path(c['job_root']);tok=read(root/'chain-consumed.json')
    require(not c.get('acceptance_reuse') and c['recovery_workload']['sequence']==[26753,26754,26754],'dedicated e001 recovery chain only')
    modes=list(MODES)
    require(tok['contract_sha256']==c['contract_sha256'] and tok['source_sha256']==c['source_sha256'] and tok['modes']==modes,'unique chain identity differs')
    start=time.time();mode=None
    try:
        for mode in modes:
            p,d=launch(c,mode)
            while p.poll() is None:
                ex=Path(c['runtime_root'])/mode/'supervisor-exit.json'
                if ex.is_file():require(read(ex)['status']=='PASS','supervisor STOP; no retry: '+mode)
                time.sleep(5)
            require(p.returncode==0,'phase process failed: '+mode);complete(c,mode)
        publish(root/'CHAIN_RESULT.json',{'status':'R18_COMMON_E120_EVALUATION_COMPARISON_COMPLETE','source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
            'started_epoch':start,'finished_epoch':time.time(),'elapsed_seconds':time.time()-start,'GPU_retry':False,'hardware_or_power_operations':False,'holdout_accessed':False})
        return 0
    except BaseException as e:
        publish(root/'CHAIN_STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','mode':mode,'error':repr(e),'traceback':traceback.format_exc(),'started_epoch':start,'finished_epoch':time.time(),
            'elapsed_seconds':time.time()-start,'GPU_retry':False,'phase_worker_may_still_be_running':'inspect exact PID/start and phase supervisor; do not relaunch'})
        raise
