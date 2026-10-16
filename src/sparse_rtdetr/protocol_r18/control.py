"""Independent safe CLI. Preparation contracts cannot launch CUDA or training."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from ..localization import guard,channel,thermal_resume,lab_consumer
from ..localization.runio import require,read,write,publish,sha,digest,reference,event
from .model import ROOT,ARM
from . import transport

def install_resources():
    transport.install()
    guard.snapshot=thermal_resume.snapshot
    def screen(sample,**kwargs):
        if kwargs.get('gpu',True):thermal_resume.cap_check(sample)
        require(sample['snapshot']['resources']['cpu_count']>=16,'common protocol CPU16 reserve')
        require(sample['snapshot']['shm_free']>=(8 if kwargs.get('entry') else 4)*1024**3,'common protocol shm reserve')
        return lab_consumer.screen(sample,**kwargs)
    guard.screen=screen

def load(path):
    transport.install()
    c=read(path)
    require(c['task_id']==transport.TASK and c['arm']==ARM,'wrong new task/model identity')
    require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'contract bytes changed')
    require(c['source_root']==str(ROOT) and sha(c['source_manifest'])==c['source_sha256'],'source root/manifest differs')
    for row in read(c['source_manifest'])['files']:
        require(reference(ROOT/row['path'])=={'path':str(ROOT/row['path']),'sha256':row['sha256'],'bytes':row['bytes']},'source differs: '+row['path'])
    for ref in c.get('materials',[]):require(reference(ref['path'])==ref,'frozen material bytes changed: '+ref['path'])
    if c.get('GPU_execution_authorized'):
        require(c.get('formal_training_authorized') is True and c['config']==__import__('sparse_rtdetr.protocol_r18.model',fromlist=['Config']).Config().binding(),'execution recipe/authority differs')
        proof=c['CPU_gate'];require(reference(proof['path'])==proof and read(proof['path'])['status']=='PASS' and read(proof['path'])['source_sha256']==c['source_sha256'],'new CPU source-bound gate absent')
        require(read(c['data_verification'])['status']=='PASS' and sha(c['data_verification'])==c['data_sha256'],'fresh data closure changed')
    return c

def identity(c,mode):
    transport.install()
    result={**{k:c[k] for k in ('task_id','arm','source_sha256','contract_sha256')},'mode':mode,
       'progress_protocol':transport.CURRENT_PROTOCOL,'output_root':str(Path(c['job_root'])/mode),
       'fixed_ids':c['fixed_ids'],'supervisor_pid':os.getpid()}
    result['starting_cursor']=26752 if mode in ('formal','gpu-recovery') else 36480
    if mode in ('formal','gpu-recovery'):result['resume_parent_sha256']=c['resume_parent']['checkpoint']['sha256']
    if mode=='cpu-protocol':
        spec=c['CPU_protocol'];result.update(CPU_protocol_only=True,protocol_mode=spec['mode'],starting_cursor=spec['starting_cursor'],resume_parent_sha256=spec['parent_sha256'])
        if spec['mode'] in ('inference','evaluation','comparison'):result['evaluation_authorization']={'checkpoint_sha256':spec['parent_sha256'],'image_ids':c['development_ids'],'budgets':[600,300]}
    if mode in ('inference','evaluation','comparison'):
        endpoint=read(Path(c['job_root'])/'formal/endpoint-lock.json')
        require(endpoint['status']=='R18_COMMON_E120_EMA_LOCKED' and endpoint['source_sha256']==c['source_sha256'] and endpoint['contract_sha256']==c['contract_sha256'],'endpoint runtime parent identity differs')
        result['evaluation_authorization']={'checkpoint_sha256':endpoint['checkpoint']['sha256'],'image_ids':c['development_ids'],'budgets':[600,300]}
    return result

def worker(c,mode):
    install_resources();out=Path(c['job_root'])/mode;b=channel.binding();require(b is not None and b['mode']==mode,'worker runtime identity missing')
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU entry CUDA visibility/initialization differs') if mode.startswith('cpu') else None
    require(read(Path(b['runtime_root'])/'identity.json')==b and b['task_id']==c['task_id'] and b['source_sha256']==c['source_sha256'] and b['contract_sha256']==c['contract_sha256'],'worker parent identity differs')
    write(out/'execution-binding.json',b)
    holder={}
    try:
        if mode=='cpu-spawn':
            from .data import dataset,input_record,training_loader
            data=dataset(c);rows=[];started=time.monotonic()
            sample=channel.resource();guard.screen(sample,gpu=False)
            require(sample['boot_id']==c['boot_id'] and 0<=time.time()-sample['sampled_epoch']<=30,'CPU resource identity/staleness differs')
            for n,(images,targets,traces) in enumerate(training_loader(data,epoch=1,window_limit=2),1):
                rec=input_record(images,targets,traces)
                require(rec['ids']==c['fixed_ids'][n-1],'frozen two-window identity differs')
                if c.get('R0_fixed_inputs'):
                    old=read(c['R0_fixed_inputs'])['inputs'][str(n-1)]['metadata']
                    require(rec['ids']==old['ids'] and rec['augmented_sha256']==old['augmented_sha256'] and rec['metadata_sha256']==old['metadata_sha256'],'R0 fixed augmented bytes differ')
                rows.append(rec);event(out,'cpu_input_window_complete',cpu_window_number=n,input=rec,CUDA_initialized=False)
                sample=channel.resource();guard.screen(sample,gpu=False)
                require(sample['boot_id']==c['boot_id'] and 0<=time.time()-sample['sampled_epoch']<=30,'CPU running resource identity/staleness differs')
            require(len(rows)==2,'CPU input workload incomplete')
            publish(out/'RESULT.json',{'status':'PASS','actual_real_windows':2,'unique_presentations':32,'maximum_prefetch_positions':[0,31],
                'workers':2,'prefetch_factor':2,'windows':rows,'CUDA_initialized':torch.cuda.is_initialized(),'elapsed_seconds':time.monotonic()-started})
        elif mode=='cpu-state':
            from .worker import cpu_state
            cpu_state(c,out,holder)
        elif mode=='cpu-monitor':
            from .monitor_cpu import worker
            worker(c,out,holder)
        elif mode=='cpu-protocol':
            from .protocol_cpu_v4 import produce
            produce(c,out,holder)
        elif mode=='gpu-recovery':
            require(c.get('GPU_execution_authorized') is True,'GPU recovery not authorized')
            from .recovery_e088 import acceptance
            acceptance(c,out,holder)
        elif mode=='gpu-acceptance':
            require(c.get('GPU_execution_authorized') is True,'this preparation contract does not authorize GPU')
            from .worker import acceptance
            acceptance(c,out,holder)
        elif mode=='cpu-execution':
            from .worker import cpu_execution
            cpu_execution(c,out,holder)
        elif mode in ('formal','inference','evaluation','comparison'):
            from . import worker as native,endpoint
            if mode in ('formal','inference'):
                from . import recovery_e088 as recovery
                getattr(recovery,mode)(c,out,holder)
            elif mode=='evaluation':endpoint.evaluate(c,out,holder)
            else:
                from .recovery_e088 import comparison
                comparison(c,out)
        else:raise ValueError('unsupported independent worker mode')
        write(channel.receipt_root(out)/'worker-exit.json',{'status':'PASS','pid':os.getpid(),'elapsed_epoch':time.time()})
    except BaseException as error:
        evidence={'status':'STOP','error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False}
        try:
            from .failure import preserve
            evidence['forensic']=preserve(holder,out)
        except BaseException as save_error:evidence['forensic_save_error']=repr(save_error)
        publish(out/'failure.json',evidence)
        publish(channel.receipt_root(out)/'worker-exit.json',{'status':'STOP','pid':os.getpid(),'error':repr(error)})
        raise

def main(argv=None):
    # Every fresh process must install the authenticated Reader before prerequisite reads.
    install_resources()
    p=argparse.ArgumentParser(description='Independent RT-DETRv2-R18 common protocol preparation')
    p.add_argument('action',choices=['plan','worker','supervise','start-recovery-chain','chain','preflight','parent-audit']);p.add_argument('--contract');p.add_argument('--mode',choices=['cpu-spawn','cpu-state','cpu-execution','cpu-monitor','cpu-protocol','gpu-acceptance','gpu-recovery','formal','inference','evaluation','comparison'])
    a=p.parse_args(argv)
    if a.action=='plan':
        print(json.dumps({'task_id':transport.TASK,'arm':ARM,'GPU_permission_requires_frozen_execution_contract':True,'formal_permission_requires_GPU_PASS_and_frozen_execution_contract':True,'recovery_acceptance_updates':[26753,26754,26754],'formal_start':26752,'remaining_updates':9728}));return 0
    require(a.contract is not None,'contract required')
    c=load(a.contract)
    if a.action=='parent-audit':
        require(not c.get('GPU_execution_authorized') and not c.get('formal_training_authorized'),'CPU audit authority differs')
        from .recovery_e088 import cpu_audit
        print(json.dumps(cpu_audit(c,Path(c['job_root'])/'E088_CPU_AUDIT.json')));return 0
    if a.action=='preflight':
        require(c.get('CPU_preflight_only') is True and not c.get('GPU_execution_authorized') and not c.get('formal_training_authorized') and a.mode=='formal','CPU preflight scope differs')
        import torch
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU preflight CUDA visibility/initialization differs')
        from .lifecycle import prerequisites
        prerequisites(c,a.mode)
        print(json.dumps({'status':'PASS','entry':'actual_independent_formal_prerequisites','source_sha256':c['source_sha256'],'CUDA_initialized':False,'formal_model_constructed':False,'launch_token_consumed':False}));return 0
    if a.action=='start-recovery-chain':
        from .lifecycle import start_chain
        print(json.dumps(start_chain(c,a.contract)));return 0
    if a.action=='chain':
        from .lifecycle import chain
        return chain(c,a.contract)
    require(a.mode is not None,'mode required')
    if a.mode in ('gpu-recovery','formal','inference'):require(c.get('GPU_execution_authorized') is True,'GPU execution not authorized')
    if a.mode in ('formal','inference','evaluation','comparison'):require(c.get('formal_training_authorized') is True,'formal/endpoint authority absent')
    if not a.mode.startswith('cpu'):
        from .lifecycle import prerequisites,verify_token
        prerequisites(c,a.mode);verify_token(c,a.mode)
    if a.action=='worker':worker(c,a.mode);return 0
    install_resources();out=Path(c['job_root'])/a.mode;runtime=Path(c['runtime_root'])/a.mode
    command=[sys.executable,str(ROOT/'tools/protocol_r18/deploy.py'),'worker','--contract',a.contract,'--mode',a.mode]
    b=identity(c,a.mode);env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'CUDA_VISIBLE_DEVICES':c['gpu_uuid'] if a.mode in ('gpu-recovery','formal','inference') else ''}
    code=guard.supervise(command,out,env,gpu_uuid=c.get('gpu_uuid'),boot_id=c['boot_id'],gpu=a.mode in ('gpu-recovery','formal','inference'),checkpoint_bytes=400*1024**2 if a.mode in ('gpu-recovery','formal') else 0,formal=a.mode=='formal',runtime=runtime,run_identity=b)
    channel.verify_archive(runtime,out,b)
    result=read(runtime/'supervisor-exit.json');require(result['status']=='PASS' and not result['remaining_live'] and not result['unknown_pids'],'CPU lifecycle/exit failed')
    latest=channel.latest(runtime,'progress')
    publish(out/'LIFECYCLE_CHECK.json',{'status':'PASS','progress_latest':latest,'archive_verified':True,'complete_exit':True,'supervisor':result})
    return code
