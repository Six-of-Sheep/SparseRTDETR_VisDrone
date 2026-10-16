"""Unique R0 fixed-workload lifecycle; no resume, foreign checkpoint, other arms or retry."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import worker,checkpoint,geometry
from .model import R0Config
from .runio import require,read,write,publish,sha,digest,reference,execution_binding,original_process_exited,TASK_ID
from ..localization import channel,guard,lab_run,thermal_resume
SOURCE=Path(__file__).resolve().parents[3];ROOT=SOURCE.parent.parent
LAB_BASE=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution')
SSD_BASE=Path('/home/lyy/.cache')/TASK_ID
MODES=('acceptance','formal','inference','evaluation')
WORKLOAD={'acceptance_updates':[1,2,2],'acceptance_actual_updates':3,'formal_updates':36480,'formal_epochs':120,'evaluation_images':548,'inference_batches':137,'budgets':[600,300]}
ACTIVE=None
def snapshot(path):return thermal_resume.snapshot(path)
def screen(sample,**kwargs):
 try:
  thermal_resume.cap_check(sample)
  return thermal_resume.screen(sample,**kwargs)
 except BaseException as error:
  if ACTIVE is not None and Path(ACTIVE).is_dir():publish(Path(ACTIVE)/('rejected-resource-'+str(time.time_ns())+'.json'),{'error':repr(error),'sample':sample})
  raise
def identity(c,mode):
 b={**execution_binding(c,mode),'progress_protocol':channel.PROTOCOL_NAME}
 if mode in ('inference','evaluation'):
  endpoint=read(ROOT/'formal/endpoint-lock.json')
  b['evaluation_authorization']={'images':548,'budgets':[600,300],'checkpoint_sha256':endpoint['checkpoint']['sha256']}
 return b
def verify_source(c):
 require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'R0 executing source differs')
 for r in read(c['source_manifest'])['files']:
  p=SOURCE/r['path'];require(p.is_file() and not p.is_symlink() and reference(p)=={'path':str(p),'bytes':r['bytes'],'sha256':r['sha256']},'R0 frozen file changed: '+r['path'])
 require(c['config']==R0Config().binding() and c['geometry']==geometry.identity() and c['workload']==WORKLOAD and c['progress_protocol']==channel.PROTOCOL_NAME,'R0 recipe/geometry/workload/channel differs')
def verify_materials(c):
 require(len(c['materials'])==len({r['path'] for r in c['materials']}),'duplicate R0 material')
 must={c[k] for k in ('data_verification','fixed_inputs','evaluation_protocol','caps_receipt','environment_receipt')}|{c['pretrained']['path']}|{s['manifest'] for s in c['data'].values()}
 require(must<={r['path'] for r in c['materials']},'R0 required frozen material missing')
 for r in c['materials']:
  p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(LAB_BASE) and not p.is_symlink() and reference(p)==r,'R0 material bytes/path changed')
 proof=read(c['data_verification']);require(proof['status']=='R0_ALL_TRAIN4869_DEV548_IMAGES_AND_RAW_SHA_PASS' and proof['pairs']==5417 and proof['data']==c['data'],'R0 complete data closure absent')
 require(c['data_sha256']==sha(c['data_verification']) and c['evaluation_protocol_sha256']==sha(c['evaluation_protocol']),'R0 dataset/evaluation digest changed')
 require(read(c['fixed_inputs'])['windows']==[0,1] and read(c['fixed_inputs'])['epoch']==1 and read(c['fixed_inputs'])['GPU_calls']==0,'R0 pre-GPU input scope changed')
 from .evaluation import POLICY
 require(read(c['evaluation_protocol'])==POLICY,'R0 frozen endpoint policy differs')
 require(not any(k in c for k in ('parent_checkpoint','diagnostic_checkpoint','resume_checkpoint')),'R0 cannot have a trained parent')
 happy=read(c['protocol_CPU']['positive']['path']);negative=read(c['protocol_CPU']['negative']['path'])
 require(happy['status']=='R0_REAL_CPU_CHANNEL_SUPERVISOR_ARCHIVE_PASS' and happy['CUDA_calls']==0 and happy['model_calls']==0 and not happy['torch_imported'],'R0 real stdlib protocol proof absent')
 require(negative['status']=='R0_REAL_CPU_PROTOCOL_NEGATIVES_PASS' and len(negative['cases'])>=12 and not negative['torch_imported'],'R0 real protocol negative coverage absent')
 for n,r in happy['implementation'].items():
  require(n in ('channel.py','guard.py','runio.py') and all(reference(SOURCE/'src/sparse_rtdetr/localization'/n)[k]==r[k] for k in ('bytes','sha256')),'R0 protocol tested implementation changed')
 require(set(happy['implementation'])=={'channel.py','guard.py','runio.py'},'R0 complete protocol implementation evidence absent')
def load(path):
 c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'R0 contract digest mismatch')
 require(c['task_id']==TASK_ID and c['job_root']==str(ROOT) and ROOT.name==TASK_ID and c['runtime_root']==str(SSD_BASE/'runtime') and c['python']==str(SSD_BASE/'environment/bin/python'),'R0 task/disk/runtime identity mismatch')
 require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'R0 boot identity changed')
 verify_source(c);verify_materials(c);return c
def prepare(path):
 c=read(path);require('contract_sha256' not in c and not (ROOT/'contract.json').exists(),'R0 freeze request already bound')
 verify_source(c);verify_materials(c)
 import torch,torchvision,numpy,importlib.metadata,xml.etree.ElementTree as ET
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'R0 prepare must be CPU only')
 observed={'python':sys.version.split()[0],'torch':str(torch.__version__),'torchvision':torchvision.__version__,'numpy':numpy.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'pycocotools':importlib.metadata.version('pycocotools'),'prefix':sys.prefix}
 require(observed['python']=='3.10.16' and observed['torch']=='2.4.1' and observed['cuda_build']=='12.4' and observed['cudnn']==90100 and Path(sys.prefix)==SSD_BASE/'environment','R0 relocated runtime differs')
 passed_names=set()
 for p in c['CPU_checks']:
  require(reference(p['xml']['path'])==p['xml'] and reference(p['log']['path'])==p['log'],'R0 CPU evidence changed')
  cases=list(ET.parse(p['xml']['path']).getroot().iter('testcase'));passed=[v for v in cases if not any(v.find(k) is not None for k in ('failure','error','skipped'))]
  require(len(passed)==p['passed'] and len(cases)==p['total'],'R0 actual CPU pass count differs')
  passed_names.update(t.get('name') for t in passed)
 require(set(c['required_CPU_test_names'])<=passed_names,'R0 unresolved required CPU test remains')
 s=snapshot(ROOT);uid=screen(s,entry=True,disk_bytes=40*1024**3+122*512*1024**2);require(uid==c['expected_gpu_uuid'],'R0 GPU UUID differs')
 from ..localization.storage import identity as storage_identity
 c.update(schema='r0-fresh-seed0-e120-fixed3-v1',gpu_uuid=uid,boot_id=s['boot_id'],environment={k:observed[k] for k in ('torch','torchvision','cuda_build','cudnn')},
          environment_observation=observed,resource_observation=s,storage={'identity':storage_identity(s['storage'])},freeze_epoch=time.time())
 c['contract_sha256']=digest(c);publish(ROOT/'contract.json',c);load(ROOT/'contract.json');return c
def complete(c,mode):
 out=ROOT/mode;runtime=Path(c['runtime_root'])/mode;d=read(ROOT/(mode+'-dispatch.json'));ex=read(runtime/'supervisor-exit.json')
 require(ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')) and original_process_exited(d,guard.identity(d['pid'])),'R0 prior phase not clean PASS')
 require(d['contract_sha256']==c['contract_sha256'] and d['mode']==mode,'R0 dispatch wrong contract')
 channel.verify_archive(runtime,out,{**identity(c,mode),'supervisor_pid':d['pid'],'supervisor_start':d['process_identity']['start']})
 require(read(out/'execution-binding.json')==execution_binding(c,mode),'R0 phase execution binding differs')
 for name in ('RESULT.json','execution-binding.json','binding.json','worker-exit.json'):
  p=runtime/name if name=='worker-exit.json' else out/name;require(reference(p)==ex['completion_receipts'][name],'R0 PASS receipt changed')
 r=read(out/'RESULT.json')
 if mode=='acceptance':
  require(r['status']=='PASS' and r['actual_updates']==3 and r['updates']==[1,2,2] and r['input_DN_noise_result_gradient_exact'] and r['full_state_RNG_exact'] and not r['state_for_formal'],'R0 fixed3 not exact')
  thermal_resume.under_load_proof(c,runtime)
 elif mode=='formal':
  require(r['status']=='R0_E120_EMA_LOCKED' and r['actual_updates']==r['updates']==36480 and r['initial_model_sha256']==read(ROOT/'acceptance/RESULT.json')['initial_model_sha256'] and not r['used_acceptance_state'] and not r['used_e050_state'],'R0 complete independent endpoint absent')
 elif mode=='inference':require(r['status']=='PASS' and r['images']==548 and r['model_batch_forwards']==137 and r['EMA_and_RNG_unchanged'] and r['inference_FP32'],'R0 endpoint inference incomplete')
 else:require(r['status']=='PASS' and r['images']==548 and r['budgets']==[600,300] and r['model_forward_calls']==0 and not r['holdout_accessed'],'R0 both endpoint budgets incomplete')
 return r
def prerequisite(c,mode):
 require(mode in MODES,'R0 unknown phase')
 for p in MODES[:MODES.index(mode)]:complete(c,p)
def command(c,path,mode):return [c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'r0-supervise','--contract',str(path),'--mode',mode]
def launch(path,mode):
 c=load(path);prerequisite(c,mode);require(not (ROOT/mode).exists() and not (ROOT/(mode+'-launch-consumed.json')).exists(),'R0 phase already used; no GPU retry')
 screen(snapshot(ROOT),entry=mode!='evaluation',gpu=mode!='evaluation',disk_bytes=40*1024**3+(122*512*1024**2 if mode=='formal' else 0))
 cmd=command(c,path,mode);publish(ROOT/(mode+'-launch-consumed.json'),{'command':cmd,'mode':mode,'contract_sha256':c['contract_sha256'],'no_retry':True,'epoch':time.time()})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (ROOT/(mode+'-dispatcher.log')).open('x') as f:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d={'pid':p.pid,'process_identity':guard.identity(p.pid),'mode':mode,'contract_sha256':c['contract_sha256'],'started_epoch':time.time()};publish(ROOT/(mode+'-dispatch.json'),d);return d
def token(c,path,mode):
 r=read(ROOT/(mode+'-launch-consumed.json'));require(r['command']==command(c,path,mode) and r['contract_sha256']==c['contract_sha256'] and r['no_retry'],'R0 launch token mismatch')
def supervise(path,mode):
 global ACTIVE
 c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen;ACTIVE=Path(c['runtime_root'])/mode
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='evaluation' else c['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'r0-worker','--contract',str(path),'--mode',mode]
 return guard.supervise(cmd,ROOT/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='evaluation',checkpoint_bytes=512*1024**2,formal=mode=='formal',runtime=ACTIVE,
                       run_identity={**identity(c,mode),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})
def worker_entry(path,mode):
 global ACTIVE
 c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen
 out=ROOT/mode;runtime=Path(c['runtime_root'])/mode;ACTIVE=runtime;holder={};started=time.time();d=read(ROOT/(mode+'-dispatch.json'))
 current=guard.identity(os.getppid());require(current and current['start']==d['process_identity']['start'] and os.getppid()==d['pid'] and read(runtime/'supervisor-start.json')['pid']==os.getppid(),'R0 worker supervisor identity')
 b=read(runtime/'identity.json');require(all(b.get(k)==v for k,v in identity(c,mode).items()),'R0 runtime binding mismatch');publish(out/'execution-binding.json',execution_binding(c,mode))
 try:
  if mode in ('inference','evaluation'):write(out/'binding.json',{'execution':execution_binding(c,mode),'evaluation_protocol_sha256':c['evaluation_protocol_sha256']})
  getattr(worker,{'acceptance':'acceptance','formal':'formal','inference':'inference','evaluation':'evaluate'}[mode])(c,out,holder)
 except BaseException as error:
  e=holder.get('engine');evidence=getattr(error,'evidence',{})
  if e is not None:evidence={**evidence,'engine_phase':e.phase,'completed_updates':e.completed_updates,'parameter_optimizer_steps':e.parameter_steps}
  evidence={**evidence,'active_R0_geometry':geometry.identity()}
  publish(runtime/'failure.json',{'status':'STOP_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),'evidence':evidence,'input':holder.get('input')});raise
 finally:write(runtime/'worker-exit.json',{'pid':os.getpid(),'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'body_returned':not (runtime/'failure.json').exists()})
 return 0
def start_chain(path):
 c=load(path);complete(c,'acceptance');cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'r0-chain-worker','--contract',str(path)]
 publish(ROOT/'chain-consumed.json',{'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],'command':cmd,'modes':list(MODES[1:]),'no_retry':True,'epoch':time.time()})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (ROOT/'chain.log').open('x') as f:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'started_epoch':time.time()};publish(ROOT/'chain-dispatch.json',d);return d
def chain_worker(path):
 c=load(path);tok=read(ROOT/'chain-consumed.json');require(tok['contract_sha256']==c['contract_sha256'] and tok['source_sha256']==c['source_sha256'] and tok['modes']==list(MODES[1:]) and tok['no_retry'],'R0 chain identity mismatch')
 started=time.time();mode=None
 try:
  for mode in MODES[1:]:
   d=launch(path,mode)
   while True:
    ex=Path(c['runtime_root'])/mode/'supervisor-exit.json'
    if ex.is_file():require(read(ex)['status']=='PASS','R0 supervisor stopped; no retry: '+mode)
    if original_process_exited(d,guard.identity(d['pid'])):complete(c,mode);break
    time.sleep(5)
  publish(ROOT/'CHAIN_RESULT.json',{'status':'R0_E120_AND_BOTH_ENDPOINT_BUDGETS_COMPLETE','source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'no_gpu_retry':True,'power_operation':False});return 0
 except BaseException as error:
  publish(ROOT/'CHAIN_STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','mode':mode,'error':repr(error),'traceback':traceback.format_exc(),'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'GPU_retry':False,'worker_may_still_be_running':'inspect actual phase worker/supervisor'});raise
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('action',choices=('r0-prepare','r0-launch','r0-supervise','r0-worker','r0-start-chain','r0-chain-worker'));p.add_argument('--request');p.add_argument('--contract');p.add_argument('--mode',choices=MODES);a=p.parse_args(argv)
 if a.action=='r0-prepare':r=prepare(a.request)
 elif a.action=='r0-launch':r=launch(a.contract,a.mode)
 elif a.action=='r0-supervise':return supervise(a.contract,a.mode)
 elif a.action=='r0-worker':return worker_entry(a.contract,a.mode)
 elif a.action=='r0-chain-worker':return chain_worker(a.contract)
 else:r=start_chain(a.contract)
 print(json.dumps(r,allow_nan=False),flush=True);return 0

