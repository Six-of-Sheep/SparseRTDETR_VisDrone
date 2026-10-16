"""Original e045 -> exactly13 updates -> exactlyone micro0 forward, never formal."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import channel,checkpoint,guard,worker,thermal_resume,numerics
from .runio import read,write,publish,reference,sha,digest,event,require,execution_binding,original_process_exited
SOURCE=Path(__file__).resolve().parents[3];ROOT=SOURCE.parent.parent;BASE=ROOT.parent
PARENT='3362fcc7be58377e0b70f7cca0bdee997a9f8f17b981cc855147a827046e7c7a'
OLD_SOURCE='6956c5ecb309bebed9256fe32496cfa46b91352e26a18983cc1537690ec26351'
OLD_CONTRACT='236a97c1896ea2201ff1aa21441ecbb20a58a44f0f42d62ff8de11a40636af58'
MODE='limited-geometry-diagnostic'
WORKLOAD={'parent_update':13680,'replay_updates':list(range(13681,13694)),'saved_boundary_update':13693,'target_epoch':46,'target_window':13,
          'target_microbatch':0,'target_forward_images':8,'target_forward_calls':1,'target_loss_backward_calls':0,'forbidden_update':13694,'formal_training':False}

def auth(c):return {'parent_checkpoint_sha256':PARENT,'parent_binding_sha256':sha(c['parent_binding']),'parent_contract_sha256':OLD_CONTRACT,
                   'parent_update':13680,'saved_boundary_update':13693,'candidate_update':13694,'epoch':46,'window':13,'microbatch':0,'layers':[0,1,2]}
def identity(c):return {**execution_binding(c,MODE),'progress_protocol':channel.PROTOCOL_NAME,'forensic_authorization':c['forensic_authorization']}

def verify_source(c):
    require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'forensic frozen source differs')
    for r in read(c['source_manifest'])['files']:
        p=SOURCE/r['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'forensic source changed: '+r['path'])
    require(c['numerical_implementation']==numerics.identity(),'forensic arithmetic identity differs')
    old=Path(c['original_source_root'])
    for n in ('model','distribution','consumer_geometry','engine','checkpoint','data','criterion','ignore','cache_adapter','numerics'):
        require(sha(SOURCE/'src/sparse_rtdetr/localization'/f'{n}.py')==sha(old/'src/sparse_rtdetr/localization'/f'{n}.py'),'protected original core changed: '+n)

def verify_materials(c):
    require(len(c['materials'])==len({r['path'] for r in c['materials']}),'duplicated forensic material')
    mandatory={c[k] for k in ('parent_binding','parent_sidecar','parent_contract','fixed_inputs','replay_reference','old_failure','cpu_protocol_positive','cpu_protocol_negative','cpu_protocol_program','caps_receipt')}
    mandatory|={c['data'][r]['manifest'] for r in ('train_core','development')}|{c['pretrained']['path'],c['selection_path']}
    require(mandatory<={r['path'] for r in c['materials']},'required forensic material absent')
    for r in c['materials']:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(BASE) and not p.is_symlink() and reference(p)==r,'forensic material changed: '+str(p))
    b=read(c['parent_binding']);s=read(c['parent_sidecar']);old=read(c['parent_contract'])
    import hashlib
    require(sha(c['parent_checkpoint'])==PARENT and s['sha256']==PARENT and s['size_bytes']==327152097 and s['completed_updates']==13680 and s['scope']=='formal_F0_seed0_e120' and s['path']==c['parent_checkpoint'],'original e045 differs')
    require(s['binding_sha256']==hashlib.sha256(checkpoint.canonical(b)).hexdigest() and b['source_sha256']==OLD_SOURCE and b['contract_sha256']==OLD_CONTRACT and old['contract_sha256']==OLD_CONTRACT and digest({k:v for k,v in old.items() if k!='contract_sha256'})==OLD_CONTRACT,'original e045 source/binding differs')
    require(c['forensic_authorization']==auth(c) and c['progress_protocol']==channel.PROTOCOL_NAME,'frozen forensic event authority differs')
    require(c['parent_lineage']=={'attempt':'limited-forensics-001','parent_checkpoint_sha256':PARENT,'parent_binding_sha256':sha(c['parent_binding']), 'parent_source_sha256':OLD_SOURCE,'parent_contract_sha256':OLD_CONTRACT,'parent_update':13680,'formal_forbidden':True},'forensic lineage differs')
    pos=read(c['cpu_protocol_positive']);neg=read(c['cpu_protocol_negative'])
    require(pos['status']=='REAL_FORENSIC_CPU_CHAIN_PASS' and pos['semantic']==4 and pos['torch_imported'] is False and pos['gpu_calls']==pos['model_calls']==0 and pos['supervisor_exit']['status']=='PASS','real no-model protocol chain incomplete')
    require(neg['status']=='REAL_FORENSIC_CPU_NEGATIVE_PASS' and len(neg['cases'])==19 and neg['torch_imported'] is False and neg['gpu_calls']==neg['model_calls']==0,'real protocol negative cases incomplete')
    for name,r in pos['implementation'].items():require(sha(SOURCE/'src/sparse_rtdetr/localization'/name)==r['sha256'],'executing protocol differs from actual CPU-tested implementation')
    fixed=read(c['fixed_inputs']);require(fixed['epoch']==46 and fixed['windows']==list(range(14)) and fixed['gpu_calls']==0,'forensic input scope differs')
    require(read(c['replay_reference'])['updates']==list(range(13681,13694)),'forensic replay reference scope differs')
    cap=read(c['caps_receipt']);require(cap['status']=='PRIOR_CPU_125W_150W_GPU_1500MHZ_REQUEST_APPLIED' and cap['gpu_uuid']==c['expected_gpu_uuid'] and cap['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'restored caps authority/boot differs')

def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'forensic contract digest differs')
    require(c['job_root']==str(ROOT) and ROOT.name=='limited-forensics-001' and c['workload']==WORKLOAD and c['runtime_root']=='/home/lyy/.cache/f0-e026-labdiag-20261009-001/runtime/limited-forensics-001','forensic workload/root expanded')
    verify_source(c);verify_materials(c);require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'forensic boot differs');return c

def prepare(path):
    c=read(path);require(not (ROOT/'contract.json').exists() and c['workload']==WORKLOAD and c['job_root']==str(ROOT),'forensic freeze already used or scope differs')
    verify_source(c);verify_materials(c)
    import torch,torchvision,xml.etree.ElementTree as ET
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'forensic freeze must be CPU only')
    require(sys.version.split()[0]=='3.10.16' and str(torch.__version__)=='2.4.1' and torch.version.cuda=='12.4' and torch.backends.cudnn.version()==90100 and str(Path(sys.executable))==c['python'],'forensic runtime changed')
    for p in c['new_cpu_checks']:
        require(reference(p['xml']['path'])==p['xml'] and reference(p['log']['path'])==p['log'],'new forensic CPU evidence changed')
        cases=list(ET.parse(p['xml']['path']).getroot().iter('testcase'));require(len(cases)==p['tests'] and all(not any(t.find(k) is not None for k in ('error','failure','skipped')) for t in cases),'new forensic CPU test incomplete')
    s=thermal_resume.snapshot(ROOT);uid=thermal_resume.screen(s,entry=True,disk_bytes=40*1024**3);require(uid==c['expected_gpu_uuid'],'forensic GPU differs')
    from .storage import identity as storage_identity
    c.update(schema='f0-one-e045-geometry-forensic-v1',gpu_uuid=uid,boot_id=s['boot_id'],environment={'torch':str(torch.__version__),'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
             storage={'identity':storage_identity(s['storage'])},resource_observation=s,freeze_epoch=time.time())
    c['contract_sha256']=digest(c);publish(ROOT/'contract.json',c);return c

def command(c,path):return [c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'forensic-supervise','--contract',str(path)]
def token(c,path):
    r=read(ROOT/'launch-consumed.json');require(r['contract_sha256']==c['contract_sha256'] and r['command']==command(c,path) and r['no_retry'] and r['workload']==WORKLOAD,'forensic token differs')

def launch(path):
    c=load(path);require(not (ROOT/'diagnostic').exists(),'single forensic attempt already used; no retry');thermal_resume.screen(thermal_resume.snapshot(ROOT),entry=True,disk_bytes=40*1024**3)
    cmd=command(c,path);publish(ROOT/'launch-consumed.json',{'contract_sha256':c['contract_sha256'],'command':cmd,'workload':WORKLOAD,'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/'dispatcher.log').open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'started_epoch':time.time()};publish(ROOT/'dispatch.json',r);return r

def supervise(path):
    c=load(path);token(c,path);runtime=Path(c['runtime_root'])/'diagnostic';guard.snapshot=thermal_resume.snapshot;guard.screen=thermal_resume.screen;thermal_resume.ACTIVE_RUNTIME=runtime
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'forensic-worker','--contract',str(path)]
    return guard.supervise(cmd,ROOT/'diagnostic',env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=True,checkpoint_bytes=512*1024**2,formal=False,
        runtime=runtime,run_identity={**identity(c),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})

def input_check(c,meta,targets,window):
    expected=read(c['fixed_inputs'])['inputs'][str(window)];require(meta==expected['metadata'] and digest(meta)==expected['metadata_digest'] and [t['gt_ids'] for t in targets]==expected['GT_ids'],'original e046 sampler/augment/GT identity differs')

def body(c,folder,holder):
    import torch
    from .data import training_loader
    from .cache_adapter import strict_restore
    from .engine import learning_rate,tensor_digest
    from ..baseline.training_v2b_engine import _batchnorm_backward_layout,_move
    from .forensic_capture import Capture
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,46)
    engine,binding=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=engine
    restored=strict_restore(engine,read(c['parent_binding']),c['parent_checkpoint'],PARENT,Path(c['runtime_root'])/'diagnostic/parent-cache-import.json')
    require(restored['full_state_exact'] and engine.completed_updates==engine.ema.updates==13680 and engine.phase==0,'original e045 full restore failed');numerics.validate_binding(binding,c)
    write(folder/'binding.json',binding);receipt={'status':'ORIGINAL_E045_STRICT_FULL_RESTORE_PASS','result':restored,'execution':identity(c),'scope':'diagnostic_only_no_formal'}
    publish(Path(c['runtime_root'])/'diagnostic/original-restore.json',receipt);publish(folder/'original-restore.json',receipt);event(folder,'resume_parent_exact',**restored,scope='gpu_short_window')
    reference_rows=read(c['replay_reference'])['records'];count=0;results=[]
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=46,window_limit=13)):
        meta=worker.check_inputs(images,targets,traces,plan,46,win);holder['input']=meta;input_check(c,meta,targets,win);worker.live_admission(folder,c)
        r=engine.train_window(images,targets);thermal_resume.lab_consumer.clock_check(engine,r,13681+win)
        compare={k:v for k,v in r.items() if k not in ('elapsed_seconds','input_sha256','event','pid','wall_epoch','monotonic','completed_epochs')}
        old=reference_rows[str(13681+win)];require(compare==old,'replayed original update result differs: '+str(13681+win))
        event(folder,'update_complete',**r,completed_epochs=45,input_sha256=digest(meta));results.append({'update':r['update'],'input_sha256':digest(meta),'original_recorded_result_exact':True,'elapsed_seconds':r['elapsed_seconds']});count+=1
    require(count==13 and engine.completed_updates==engine.ema.updates==13693 and engine.phase==0,'forensic13 window scope differs')
    isolated=worker.save_state(engine,binding,folder,'C13693_DIAGNOSTIC_ONLY.pth');publish(folder/'diagnostic-state-policy.json',{'checkpoint':isolated,'formal_usable':False,'only_isolated_diagnostic':True})
    before_params=tensor_digest(engine.model.named_parameters());before_ema=tensor_digest(engine.ema.module.state_dict().items());before_opt=copy.deepcopy(engine.optimizer.state_dict());before_rng=checkpoint.capture_rng(engine.device)
    engine.model.train();engine.criterion.train();engine.optimizer.zero_grad(set_to_none=True)
    for g in engine.optimizer.param_groups:g['lr']=learning_rate(13694)
    forbidden_calls=[]
    def forbidden(*args,**kwargs):forbidden_calls.append(True);raise RuntimeError('target stage forbids loss/backward/update/micro1')
    engine.optimizer.step=forbidden;engine.ema.update=forbidden;engine.criterion.forward=forbidden
    original_backward=torch.autograd.backward;original_grad=torch.autograd.grad;torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
    cap=None;target_count=0;failure=None;outcome=None
    try:
        for images,targets,traces in training_loader(data,epoch=46,completed_windows=13,window_limit=1):
            meta=worker.check_inputs(images,targets,traces,plan,46,13);holder['input']=meta;input_check(c,meta,targets,13);worker.live_admission(folder,c);target_count+=1
            subset=_move(targets[:8],engine.device);batch=images[:8].to(engine.device);cap=Capture(folder/'captures',identity(c),subset,meta,microbatch=0)
            try:
                with _batchnorm_backward_layout(engine.model,engine.device),cap.activate(engine.model.decoder):
                    with torch.autocast(engine.device.type,dtype=torch.bfloat16):out=engine.model(batch,targets=subset)
                outcome='TARGET_FORWARD_COMPLETED_NOT_REPRODUCED';del out
            except Exception as error:
                from .operators import NumericalIntegrityError
                if not isinstance(error,NumericalIntegrityError) or error.evidence.get('consumer')!='decode/box64' or error.evidence.get('decoder_layer')!=2 or len(cap.records)!=3:raise
                failure={'error':repr(error),'traceback':traceback.format_exc(),'evidence':error.evidence};outcome='EXPECTED_NEGATIVE_GEOMETRY_CAPTURED'
            del batch,subset
    finally:
        torch.autograd.backward=original_backward;torch.autograd.grad=original_grad
    require(target_count==1 and cap is not None and len(cap.records)==3 and not forbidden_calls,'forensic target scope/capture incomplete')
    require(engine.completed_updates==engine.ema.updates==13693 and engine.phase==0 and all(p.grad is None for p in engine.model.parameters()),'forensic target submitted gradient/update')
    require(before_params==tensor_digest(engine.model.named_parameters()) and before_ema==tensor_digest(engine.ema.module.state_dict().items()),'forensic target changed parameters or EMA')
    after_opt=engine.optimizer.state_dict()
    require(checkpoint.exact_equal(before_opt['state'],after_opt['state']),'forensic target optimizer moments changed')
    capture_manifest={'schema':'actual-forensic-capture-manifest-v1','execution':identity(c),'layers':cap.records,'outcome':outcome,'target_forward_calls':1,'target_microbatch':0}
    publish(folder/'capture-manifest.json',capture_manifest)
    event(folder,'forensic_forward_closed',parent_checkpoint_sha256=PARENT,completed_updates=13693,candidate_update=13694,epoch=46,window=13,microbatch=0,outcome=outcome,capture_manifest_sha256=sha(folder/'capture-manifest.json'))
    result={'status':outcome,'actual_updates':13,'completed_updates':13693,'replay':results,'original_restore':restored,'isolated_diagnostic_state':isolated,
            'target_forward_calls':1,'target_forward_images':8,'target_microbatch':0,'target_loss_calls':0,'target_backward_calls':0,'update13694_submitted':False,'micro1_executed':False,
            'parameters_exact_unchanged_during_target':True,'EMA_exact_unchanged_during_target':True,'optimizer_moments_unchanged':True,'candidate_lr_write_matches_original_training':True,
            'target_RNG_unchanged':checkpoint.exact_equal(before_rng,checkpoint.capture_rng(engine.device)),'failure':failure,'capture_manifest':reference(folder/'capture-manifest.json'),'state_for_formal':False}
    engine.failed=True;publish(folder/'RESULT.json',result)

def worker_entry(path):
    c=load(path);token(c,path);folder=ROOT/'diagnostic';runtime=Path(c['runtime_root'])/'diagnostic';holder={};start=time.time();d=read(ROOT/'dispatch.json')
    current=guard.identity(os.getppid());require(current and current['start']==d['process_identity']['start'] and os.getppid()==d['pid'],'forensic worker parent differs')
    b=read(runtime/'identity.json');require(all(b.get(k)==v for k,v in identity(c).items()),'forensic runtime identity differs')
    guard.snapshot=thermal_resume.snapshot;guard.screen=thermal_resume.screen;thermal_resume.ACTIVE_RUNTIME=runtime;publish(folder/'execution-binding.json',execution_binding(c,MODE))
    try:body(c,folder,holder)
    except BaseException as error:
        publish(runtime/'failure.json',{'status':'UNEXPECTED_STOP_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),'evidence':getattr(error,'evidence',{}),'input':holder.get('input')});raise
    finally:write(runtime/'worker-exit.json',{'pid':os.getpid(),'started_epoch':start,'finished_epoch':time.time(),'body_returned':not (runtime/'failure.json').exists()})
    return 0

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('forensic-prepare','forensic-launch','forensic-supervise','forensic-worker'));p.add_argument('--request');p.add_argument('--contract');a=p.parse_args(argv)
    if a.action=='forensic-prepare':r=prepare(a.request)
    elif a.action=='forensic-launch':r=launch(a.contract)
    elif a.action=='forensic-supervise':return supervise(a.contract)
    else:return worker_entry(a.contract)
    print(json.dumps(r,allow_nan=False));return 0
