"""One-shot laboratory-only paired chain; scientific FAIL completes fixed workload."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import new_runtime as rt
MODES=('gpu-acceptance','formal','infer-A','accept-A','accept-B','accept-C','full-C','evaluation')

def verify(c,mode):
 from sparse_rtdetr.localization import channel
 from sparse_rtdetr.localization.runio import read,require,reference
 root=Path(c['job_root']);out=root/mode;runtime=Path(c['runtime_root'])/mode
 closed=read(out/'CLOSED.json');b=read(runtime/'identity.json');expected=rt.identity(c,mode)
 for k,v in expected.items():
  if k not in ('supervisor_pid','supervisor_start'):require(b[k]==v,'phase identity differs: '+k)
 require(closed['status']=='PASS' and closed['seed']==c['seed'] and closed['result']==reference(out/'RESULT.json') and closed['runtime_index']==reference(out/'runtime-archive-index.json'),'closed result/index changed')
 ex=read(runtime/'supervisor-exit.json');require(ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'phase did not exit cleanly')
 require(read(out/'execution-binding.json')==b,'worker bound to different supervisor')
 r=read(out/'RESULT.json')
 if mode=='gpu-acceptance':
  require(r['status']=='PASS' and r['sequence']==[1,2,2] and r['actual_updates']==3 and r['same_next_result_exact'] and r['complete_state_RNG_exact'] and not r['state_usable_for_formal'],'GPU replay evidence differs')
  from sparse_rtdetr.localization.thermal_resume import under_load_proof
  under_load_proof(c,runtime)
 elif mode=='formal':require(r['status']=='R18_COMMON_E120_EMA_LOCKED' and r['seed']==c['seed'] and r['updates']==r['actual_updates']==36480 and not r['used_acceptance_state'],'formal endpoint not exact')
 else:
  require(r['status']=='PASS','inference/evaluation engineering stage failed')
  endpoint=read(root/'formal/endpoint-lock.json')
  for k in ('task_id','seed','scope','source_sha256','contract_sha256'):require(r[k]==c[k],'inference/eval result identity differs: '+k)
  require(r['checkpoint_sha256']==endpoint['checkpoint']['sha256'] and r['detector_updates']==0,'inference/eval endpoint differs')
  if mode=='evaluation':require(r['new_score_budget_calls']==6 and r['forward_calls']==0,'endpoint score scope differs')
  else:
   count,warmup,factor={'infer-A':(548,0,1),'accept-A':(64,2,1),'accept-B':(64,2,1),'accept-C':(64,2,5),'full-C':(548,2,5)}[mode]
   require(r['images']==count and r['model_batch_forwards']==factor*(count//4+warmup) and r['state_RNG_exact'] is True,'fixed inference workload/state differs')
 return r

def admission(c,mode):
 from sparse_rtdetr.localization import guard
 from sparse_rtdetr.localization.storage import identity
 from sparse_rtdetr.localization.runio import require
 rt.install(c);s=guard.snapshot(c['job_root']);gpu=mode!='evaluation'
 require(s['boot_id']==c['boot_id'] and identity(s['storage'])==c['storage']['identity'],'entry boot/storage changed')
 require(0<=time.monotonic()-s['monotonic']<25,'entry resource stale')
 guard.screen(s,gpu_uuid=c['gpu_uuid'],entry=True,gpu=gpu,disk_bytes=(200 if mode=='formal' else 40)*1024**3)
 return s

def phase(c,mode):
 from sparse_rtdetr.localization.runio import publish,require,reference,read
 from new_entry import prerequisite
 prerequisite(c,mode);root=Path(c['job_root']);require(not (root/mode).exists() and not (root/(mode+'-consumed.json')).exists(),'phase already used; no replay')
 sample=admission(c,mode);publish(root/(mode+'-admission.json'),{'status':'PASS','mode':mode,'seed':c['seed'],'snapshot':sample})
 command=[c['python'],'-B',str(Path(__file__).with_name('new_entry.py')),'supervise','--contract',c['contract_path'],'--mode',mode]
 publish(root/(mode+'-consumed.json'),{'mode':mode,'seed':c['seed'],'task_id':c['task_id'],'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'command':command,'no_retry':True,'started_epoch':time.time()})
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':c['source_root']+'/src','MKL_THREADING_LAYER':'GNU','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 start=time.time()
 with (root/(mode+'-supervisor.log')).open('xb') as log:
  p=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 publish(root/(mode+'-exit.json'),{'mode':mode,'seed':c['seed'],'returncode':p.returncode,'started_epoch':start,'finished_epoch':time.time(),'elapsed_seconds':time.time()-start,'log':reference(root/(mode+'-supervisor.log'))})
 require(p.returncode==0,'phase failed no retry: '+mode)
 return verify(c,mode)

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);a=p.parse_args();execution=json.loads(Path(a.execution).read_text())
 first=rt.load(execution['contracts'][0]['path']);rt.install(first)
 from sparse_rtdetr.localization.runio import publish,read,require,reference,digest
 root=Path(execution['job_root']);start=time.time();seed=None;mode=None;done=[]
 require(execution['execution_sha256']==digest({k:v for k,v in execution.items() if k!='execution_sha256'}),'execution digest differs')
 require(execution['seeds']==[1,2] and execution['no_retry'] is True and execution['holdout'] is False,'paired scope differs')
 require(not (root/'CHAIN_RESULT.json').exists() and not (root/'STOP.json').exists() and not (root/'CHAIN_STARTED.json').exists(),'chain already used; no retry')
 token=read(root/'LAUNCH_CONSUMED.json');require(token['execution_sha256']==execution['execution_sha256'],'launch identity differs')
 publish(root/'CHAIN_STARTED.json',{'started_epoch':start,'pid':os.getpid(),'seeds':[1,2],'source_sha256':first['source_sha256'],'execution_sha256':execution['execution_sha256']})
 try:
  for ref in execution['contracts']:
   require(reference(ref['path'])==ref,'seed contract changed');c=rt.load(ref['path']);rt.install(c);seed=c['seed'];results={}
   for mode in MODES:
    if mode=='formal':
     cmd=[c['python'],'-B',str(Path(__file__).with_name('new_entry.py')),'preflight','--contract',c['contract_path'],'--mode','formal']
     p=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True)
     publish(Path(c['job_root'])/'FORMAL_PREFLIGHT.json',{'status':'PASS' if p.returncode==0 else 'FAIL','returncode':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode(),'model_constructed':False,'CUDA_initialized':False})
     require(p.returncode==0,'real formal preflight failed')
    results[mode]=phase(c,mode)
    publish(root/f'seed{seed}-{mode}-complete.json',{'seed':seed,'mode':mode,'result':reference(Path(c['job_root'])/mode/'RESULT.json'),'finished_epoch':time.time()})
   # Scientific and matched cost failures are results, not permission to alter completion.
   done.append({'seed':seed,'comparison':reference(Path(c['job_root'])/'evaluation/COMPARISON.json'),'result':reference(Path(c['job_root'])/'evaluation/RESULT.json')})
   publish(root/f'seed{seed}-COMPLETE.json',{'status':'FIXED_SEED_COMPLETE','seed':seed,'results':done[-1],'finished_epoch':time.time()})
  import aggregate_seeds
  summary=aggregate_seeds.run(execution,root/'summary')
  publish(root/'CHAIN_RESULT.json',{'status':'FIXED_PAIRED_EXECUTION_COMPLETE','seeds':[1,2],'completed':done,'summary':summary,'started_epoch':start,'finished_epoch':time.time(),'elapsed_seconds':time.time()-start,'automatic_retry':False,'holdout_accessed':False})
  import slim_delivery
  slim_delivery.server_export(execution)
  return 0
 except BaseException as e:
  publish(root/'STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','seed':seed,'mode':mode,'error':repr(e),'traceback':traceback.format_exc(),'started_epoch':start,'finished_epoch':time.time(),'completed':done,'automatic_retry':False})
  try:
   import slim_delivery
   slim_delivery.server_export(execution)
  except BaseException as x:publish(root/'EXPORT_ERROR.json',{'error':repr(x),'traceback':traceback.format_exc()})
  return 2

if __name__=='__main__':raise SystemExit(main())
