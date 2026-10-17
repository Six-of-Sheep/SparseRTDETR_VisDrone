"""Import-safe independent seed1/2 entry. No legacy training dispatcher."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import new_runtime as rt

def prerequisite(c,mode,*,preview=False):
 from sparse_rtdetr.localization.runio import read,require,reference
 root=Path(c['job_root'])
 if mode.startswith('cpu'):return
 if preview and c.get('CPU_preflight_only'):
  require(not c.get('GPU_execution_authorized') and not c.get('formal_training_authorized'),'CPU prerequisite fixture carries execution authority')
 else:require(c.get('GPU_execution_authorized') and c.get('formal_training_authorized'),'chain authority absent')
 gate=read(c['CPU_gate']['path']);require(reference(c['CPU_gate']['path'])==c['CPU_gate'] and gate['status']=='PASS' and gate['source_sha256']==c['source_sha256'] and gate['seed']==c['seed'],'CPU gate differs')
 if c.get('parent_reuse'):
  from resume_support import prerequisite as resumed_prerequisite
  resumed_prerequisite(c,mode);return
 if mode!='gpu-acceptance':
  d=root/'gpu-acceptance';a=read(d/'RESULT.json');closed=read(d/'CLOSED.json');bound=read(d/'execution-binding.json')
  require(closed['status']=='PASS' and closed['seed']==c['seed'] and closed['mode']=='gpu-acceptance' and closed['result']==reference(d/'RESULT.json') and closed['runtime_index']==reference(d/'runtime-archive-index.json'),'acceptance archive/reference identity differs')
  require(all(bound[k]==c[k] for k in ('task_id','seed','source_sha256','contract_sha256')) and bound['mode']=='gpu-acceptance','acceptance source/seed parent differs')
  require(a['status']=='PASS' and a['actual_updates']==3 and a['sequence']==[1,2,2] and a['same_next_result_exact'] and a['complete_state_RNG_exact'] and not a['state_usable_for_formal'],'save/restore gate failed')
 if mode not in ('gpu-acceptance','formal'):
  e=read(root/'formal/endpoint-lock.json')
  require(e['seed']==c['seed'] and e['updates']==36480 and e['epoch']==120 and e['actual_updates']==36480 and e['source_sha256']==c['source_sha256'] and e['contract_sha256']==c['contract_sha256'] and not e['used_acceptance_state'],'own e120 endpoint missing')
  require(read(root/'formal/CLOSED.json')['status']=='PASS','formal lifecycle missing')

def formal(c,out,holder):
 import torch
 from sparse_rtdetr.localization import worker as base
 from sparse_rtdetr.localization.runio import read,require,write,event,publish,reference
 from sparse_rtdetr.protocol_r18.worker import make_engine,save_state,execute_window,full_clock
 from sparse_rtdetr.protocol_r18.data import dataset,training_loader,input_record
 from sparse_rtdetr.protocol_r18 import checkpoint
 short=read(Path(c['job_root'])/'gpu-acceptance/RESULT.json')
 runtime=base.prepare_cuda(c,out);e,b,initial=make_engine(c,runtime,c['scope']);holder.update(engine=e,binding=b)
 require(initial==short['initial_model_sha256'] and not e.optimizer.state and e.completed_updates==e.ema.updates==0,'formal state was not independent own-seed untrained initialization')
 write(out/'binding.json',b);zero=save_state(e,b,out,'checkpoint-epoch-000.pth')
 data=dataset(c);started=time.monotonic();times=[];last=None
 for epoch in range(1,121):
  rt.load(c['contract_path']);require(e.completed_updates==(epoch-1)*304,'epoch start clock differs')
  plan=base.plan_epoch(data,epoch);write(out/f'input-plan-epoch-{epoch:03d}.json',plan)
  event(out,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,completed_updates=e.completed_updates,dropped_ids=plan['ids'][4864:])
  t=time.monotonic();count=0
  for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
   inp=input_record(images,targets,traces);observers=inp.pop('spawn_workers')
   require(inp['ids']==plan['ids'][win*16:(win+1)*16] and all(x['epoch']==epoch and x['position']==win*16+i for i,x in enumerate(traces)),'formal data order/cursor differs')
   r=execute_window(e,images,targets,inp,out,holder,c);full_clock(e,r,(epoch-1)*304+win+1);count+=1
  require(count==304 and e.completed_updates==epoch*304,'epoch incomplete')
  last=save_state(e,b,out,f'checkpoint-epoch-{epoch:03d}.pth');elapsed=time.monotonic()-t;times.append(elapsed)
  publish(out/f'epoch-{epoch:03d}-complete.json',{'seed':c['seed'],'epoch':epoch,'updates':e.completed_updates,'checkpoint':last,'epoch_wall_seconds':elapsed,'strong_augmentation':epoch<118})
 require(e.completed_updates==e.ema.updates==36480 and checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],b),checkpoint.capture(e,b)),'complete endpoint state differs')
 endpoint={'status':'R18_COMMON_E120_EMA_LOCKED','arm':c['arm'],'seed':c['seed'],'epoch':120,'updates':36480,'actual_updates':36480,'checkpoint':last,'initial_checkpoint':zero,'initial_model_sha256':initial,'binding':reference(out/'binding.json'),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'used_acceptance_state':False,'used_R0_or_v2a_state':False,'training_wall_seconds':time.monotonic()-started,'epoch_wall_seconds':times,'evaluation_before_endpoint':False,'holdout_accessed':False}
 publish(out/'endpoint-lock.json',endpoint);publish(out/'RESULT.json',endpoint)

def cpu_spawn(c,out):
 import torch
 from sparse_rtdetr.localization import channel,guard,worker as base
 from sparse_rtdetr.localization.runio import write,publish,event,require
 from sparse_rtdetr.protocol_r18.data import dataset,training_loader,input_record
 require(not torch.cuda.is_initialized(),'CPU CUDA initialized')
 data=dataset(c);plan=base.plan_epoch(data,1);rows=[]
 for n,(images,targets,traces) in enumerate(training_loader(data,epoch=1,window_limit=2),1):
  rec=input_record(images,targets,traces)
  require(rec['ids']==c['fixed_ids'][n-1]==plan['ids'][(n-1)*16:n*16],'seed first2 input order differs')
  rows.append(rec);event(out,'cpu_input_window_complete',cpu_window_number=n,input=rec,CUDA_initialized=False)
  sample=channel.resource();guard.screen(sample,gpu=False)
  require(sample['boot_id']==c['boot_id'] and 0<=time.monotonic()-sample['monotonic']<25,'CPU resource identity/staleness differs')
 require(len(rows)==2 and all(len(r['ids'])==16 for r in rows),'spawn workload incomplete')
 publish(out/'RESULT.json',{'status':'PASS','seed':c['seed'],'actual_real_windows':2,'unique_presentations':32,'maximum_prefetch_positions':[0,31],'workers':2,'prefetch_factor':2,'windows':rows,'CUDA_initialized':False,'spawn_import_dispatch_and_CUDA_checks':True})

def infer(c,out,mode,holder):
 from sparse_rtdetr.localization import worker as base
 from sparse_rtdetr.localization.runio import read,require,reference,publish
 from sparse_rtdetr.protocol_r18.worker import make_engine
 from sparse_rtdetr.protocol_r18 import checkpoint
 import infer_seed
 from resume_support import endpoint as get_endpoint,restore_contract
 endpoint=get_endpoint(c);parent=restore_contract(c);binding=read(Path(parent['job_root'])/'formal/binding.json')
 require(reference(Path(parent['job_root'])/'formal/binding.json')==endpoint['binding'],'own binding changed')
 runtime=base.prepare_cuda(c,out);engine,b,_=make_engine(parent,runtime,c['scope']);holder.update(engine=engine,binding=b)
 require(b==binding,'own endpoint construction changed')
 cp=endpoint['checkpoint'];restored=checkpoint.restore(engine,b,cp['path'],cp['sha256'])
 require(restored['full_state_exact'] and engine.completed_updates==engine.ema.updates==36480,'own e120 restore differs');publish(out/'RESTORE.json',restored)
 infer_seed.gpu({**c,'checkpoint':cp},out,mode,engine,b)

def worker(c,mode):
 import torch
 from sparse_rtdetr.localization import channel
 from sparse_rtdetr.localization.runio import require,read,write,publish,event,reference
 b=channel.binding();out=Path(c['job_root'])/mode
 require(b is not None and b['mode']==mode and b['seed']==c['seed'] and b['task_id']==c['task_id'],'worker runtime identity missing')
 require(read(Path(b['runtime_root'])/'identity.json')==b,'worker runtime identity differs')
 from sparse_rtdetr.localization import guard
 require(os.getppid()==b['supervisor_pid'] and guard.identity(os.getppid())['start']==b['supervisor_start'],'worker process parent differs')
 for k in ('source_sha256','contract_sha256'):require(b[k]==c[k],'worker parent binding differs')
 if mode.startswith('cpu'):require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU initialization guard differs')
 if not mode.startswith('cpu'):
  token=read(Path(c['job_root'])/(mode+'-consumed.json'));require(token['contract_sha256']==c['contract_sha256'] and token['mode']==mode,'unique phase credential differs')
 write(out/'execution-binding.json',b);holder={}
 try:
  if mode=='cpu-spawn':cpu_spawn(c,out)
  elif mode=='cpu-state':
   from sparse_rtdetr.protocol_r18.worker import cpu_state
   cpu_state(c,out,holder)
  elif mode=='cpu-protocol':
   import test_seed_protocol,seed_cpu_tests
   protocol=test_seed_protocol.run(out/'protocol',c)
   seed=seed_cpu_tests.run()
   import cpu_front,infer_seed
   front=cpu_front.run(c,out/'front')
   if c['seed']==1:publish(out/'CPU_INFERENCE_INTERFACE.json',infer_seed.cpu_checks(c))
   else:
    old=Path(c['job_root']).parent/'seed1/cpu-protocol/CPU_INFERENCE_INTERFACE.json'
    prior=read(old);require(prior['status']=='PASS','shared inference CPU evidence absent')
    publish(out/'CPU_INFERENCE_INTERFACE.json',{'status':'PASS','reused_unchanged_inference_interface':reference(old),'reason':'input geometry and candidate science independent of training seed; both perseed spawn checked separately'})
   publish(out/'RESULT.json',{'status':'PASS','seed':c['seed'],'protocol':protocol,'seed_interfaces':seed,'actual_formal_front':front,'CUDA_initialized':False})
  elif mode=='gpu-acceptance':
   from sparse_rtdetr.protocol_r18.worker import acceptance
   acceptance(c,out,holder)
  elif mode=='formal':formal(c,out,holder)
  elif mode=='evaluation':
   from sparse_rtdetr.localization.runio import read
   import infer_seed
   from resume_support import endpoint as get_endpoint
   endpoint=get_endpoint(c)
   infer_seed.evaluate({**c,'checkpoint':endpoint['checkpoint']},out)
  else:infer(c,out,mode,holder)
  result=read(out/'RESULT.json')
  event(out,'seed_phase_complete',mode=mode,phase=mode,seed=c['seed'],result=reference(out/'RESULT.json'),actual_updates=(36480 if mode=='formal' else 3 if mode in ('cpu-state','gpu-acceptance') else 0),detector_updates=(36480 if mode=='formal' else 3 if mode in ('cpu-state','gpu-acceptance') else 0),state_RNG_exact=True,CUDA_initialized=torch.cuda.is_initialized())
  publish(channel.receipt_root(out)/'worker-exit.json',{'success':True,'status':'PASS','pid':os.getpid()})
 except BaseException as error:
  evidence={'status':'STOP','mode':mode,'error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False}
  try:
   from sparse_rtdetr.protocol_r18.failure import preserve
   evidence['forensic']=preserve(holder,out)
  except BaseException as e:evidence['forensic_save_error']=repr(e)
  publish(out/'failure.json',evidence)
  publish(channel.receipt_root(out)/'failure.json',{'status':'STOP','mode':mode,'error':repr(error)})
  publish(channel.receipt_root(out)/'worker-exit.json',{'success':False,'status':'STOP','pid':os.getpid(),'error':repr(error)})
  raise

def supervise(c,mode):
 from sparse_rtdetr.localization import channel,guard
 from sparse_rtdetr.localization.runio import require,publish,read,reference
 out=Path(c['job_root'])/mode;runtime=Path(c['runtime_root'])/mode
 gpu=mode in ('gpu-acceptance','formal','infer-A','accept-A','accept-B','accept-C','full-C')
 env={**os.environ,'PYTHONPATH':c['source_root']+'/src','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','CUBLAS_WORKSPACE_CONFIG':':4096:8','CUDA_VISIBLE_DEVICES':c['gpu_uuid'] if gpu else ''}
 cmd=[c['python'],'-B',str(Path(__file__).resolve()),'worker','--contract',c['contract_path'],'--mode',mode]
 code=guard.supervise(cmd,out,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=gpu,checkpoint_bytes=400*1024**2 if mode in ('gpu-acceptance','formal','cpu-state') else 0,formal=mode=='formal',runtime=runtime,run_identity=rt.identity(c,mode))
 binding=read(runtime/'identity.json');channel.verify_archive(runtime,out,binding)
 ex=read(runtime/'supervisor-exit.json');require(code==0 and ex['status']=='PASS' and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'phase lifecycle not clean')
 reader=channel.Reader(out/'runtime-archive/progress.jsonl',binding,'progress');record,_=reader.poll()
 require(record is not None and record['payload']['event']=='seed_phase_complete','phase terminal event absent')
 publish(out/'CLOSED.json',{'status':'PASS','mode':mode,'seed':c['seed'],'result':reference(out/'RESULT.json'),'runtime_index':reference(out/'runtime-archive-index.json'),'supervisor_exit':reference(runtime/'supervisor-exit.json')})
 return 0

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','worker','supervise']);p.add_argument('--contract',required=True);p.add_argument('--mode',required=True);a=p.parse_args(argv)
 c=rt.load(a.contract);rt.install(c);prerequisite(c,a.mode,preview=a.action=='preflight')
 if a.action=='preflight':
  import torch
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  print(json.dumps({'status':'PASS','seed':c['seed'],'mode':a.mode,'actual_entry':True,'CUDA_initialized':False,'model_constructed':False,'token_consumed':False}));return 0
 if a.action=='worker':worker(c,a.mode);return 0
 return supervise(c,a.mode)

if __name__=='__main__':raise SystemExit(main())
