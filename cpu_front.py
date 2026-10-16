"""Actual import-safe formal preflight with explicitly non-executable CPU fixtures."""
import json,os,subprocess,sys
from pathlib import Path

def run(c,out):
 from sparse_rtdetr.localization.runio import publish,read,require,reference,digest
 import torch
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 root=out/f'seed{c["seed"]}';root.mkdir();(root/'gpu-acceptance').mkdir()
 gate=root/'CPU_GATE.json';publish(gate,{'status':'PASS','seed':c['seed'],'source_sha256':c['source_sha256'],'scope':'CPU_SYNTHETIC_PREREQUISITE_FIXTURE_NOT_GPU_EVIDENCE'})
 acc={'status':'PASS','actual_updates':3,'sequence':[1,2,2],'same_next_result_exact':True,'complete_state_RNG_exact':True,'state_usable_for_formal':False,'scope':'CPU_SYNTHETIC_PREREQUISITE_FIXTURE_NOT_GPU_EVIDENCE'}
 publish(root/'gpu-acceptance/RESULT.json',acc)
 publish(root/'gpu-acceptance/runtime-archive-index.json',{'scope':'CPU_SYNTHETIC_FIXTURE_ONLY'})
 fixture={k:v for k,v in c.items() if k!='contract_sha256'}
 fixture.update(job_root=str(root),runtime_root=str(out/'runtime'/root.name),CPU_gate=reference(gate),CPU_preflight_only=True,GPU_execution_authorized=False,formal_training_authorized=False,contract_path=str(root/'fixture-contract.json'))
 fixture['contract_sha256']=digest(fixture);publish(Path(fixture['contract_path']),fixture)
 publish(root/'gpu-acceptance/execution-binding.json',{**{k:fixture[k] for k in ('task_id','seed','source_sha256','contract_sha256')},'mode':'gpu-acceptance','scope':'CPU_SYNTHETIC_FIXTURE_ONLY'})
 publish(root/'gpu-acceptance/CLOSED.json',{'status':'PASS','seed':c['seed'],'mode':'gpu-acceptance','result':reference(root/'gpu-acceptance/RESULT.json'),'runtime_index':reference(root/'gpu-acceptance/runtime-archive-index.json'),'scope':'CPU_SYNTHETIC_FIXTURE_ONLY'})
 command=[sys.executable,'-B',str(Path(__file__).with_name('new_entry.py')),'preflight','--contract',fixture['contract_path'],'--mode','formal']
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
 for key in list(env):
  if key.startswith('SPARSE_RTDETR_') or key.startswith('SPARSE_LOCALIZATION_'):env.pop(key,None)
 good=subprocess.run(command,env=env,capture_output=True)
 require(good.returncode==0,'actual formal CPU preflight positive branch failed: '+good.stderr.decode()[-2000:])
 result=json.loads(good.stdout.decode());require(result['status']=='PASS' and not result['CUDA_initialized'] and not result['model_constructed'] and not result['token_consumed'],'CPU preflight crossed scope')
 bad={**fixture,'CPU_gate':{**fixture['CPU_gate'],'sha256':'0'*64}};bad.pop('contract_sha256');bad['contract_path']=str(root/'bad-contract.json');bad['contract_sha256']=digest(bad);publish(Path(bad['contract_path']),bad)
 refused=subprocess.run([*command[:command.index('--contract')+1],bad['contract_path'],'--mode','formal'],env=env,capture_output=True)
 require(refused.returncode!=0 and b'CPU gate differs' in refused.stderr,'formal CPU prerequisite corrupted reference not rejected')
 # Separate non-executable fixtures test actual formal acceptance-parent checks.
 extra=[]
 for case in ('foreign-seed','corrupt-result'):
  d=out/case/root.name;(d/'gpu-acceptance').mkdir(parents=True)
  nextc={k:v for k,v in fixture.items() if k!='contract_sha256'}
  nextc.update(job_root=str(d),runtime_root=str(out/case/'runtime'/d.name),contract_path=str(d/'fixture-contract.json'))
  nextc['contract_sha256']=digest(nextc);publish(Path(nextc['contract_path']),nextc)
  publish(d/'gpu-acceptance/RESULT.json',acc);publish(d/'gpu-acceptance/runtime-archive-index.json',{'scope':'CPU_SYNTHETIC_FIXTURE_ONLY'})
  b={**{k:nextc[k] for k in ('task_id','seed','source_sha256','contract_sha256')},'mode':'gpu-acceptance'}
  if case=='foreign-seed':b['seed']=3-c['seed']
  publish(d/'gpu-acceptance/execution-binding.json',b)
  rr=reference(d/'gpu-acceptance/RESULT.json')
  if case=='corrupt-result':rr['sha256']='0'*64
  publish(d/'gpu-acceptance/CLOSED.json',{'status':'PASS','seed':c['seed'],'mode':'gpu-acceptance','result':rr,'runtime_index':reference(d/'gpu-acceptance/runtime-archive-index.json')})
  badp=subprocess.run([*command[:command.index('--contract')+1],nextc['contract_path'],'--mode','formal'],env=env,capture_output=True)
  expected=b'acceptance source/seed parent differs' if case=='foreign-seed' else b'acceptance archive/reference identity differs'
  require(badp.returncode!=0 and expected in badp.stderr,'formal acceptance negative not rejected: '+case)
  extra.append({'case':case,'returncode':badp.returncode})
 # Real worker entry with no supervisor channel must reject before any data/model work.
 from sparse_rtdetr.localization import channel
 for key in channel.environment(channel.binding()):env.pop(key,None)
 orphan=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('new_entry.py')),'worker','--contract',c['contract_path'],'--mode','cpu-spawn'],env=env,capture_output=True)
 require(orphan.returncode!=0 and b'worker runtime identity missing' in orphan.stderr,'orphan entry not rejected before work')
 require(not list(root.glob('*consumed*')) and not (root/'formal').exists() and not torch.cuda.is_initialized(),'CPU fixture consumed execution state')
 result={'status':'PASS','positive_formal_preflight':True,'corrupt_gate_rejected':True,'acceptance_negative_cases':extra,'orphan_worker_rejected':True,'CUDA_initialized':False,'model_constructed':False,'GPU_authority':False,'token_consumed':False,'fixture_scope':'CPU_SYNTHETIC_ONLY','returncodes':[good.returncode,refused.returncode,orphan.returncode]}
 publish(out/'RESULT.json',result);return result
