"""Actual CLI subprocess checks; constructed CPU inputs only, no optimizer."""
import copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import sabl_forward_check as f

def main():
 import torch
 torch.set_num_threads(2)
 passed=[]
 def check(ok,name):
  if not ok:raise AssertionError(name)
  passed.append(name)
 def reject(call,name):
  try:call()
  except (ValueError,TypeError,RuntimeError,KeyError):passed.append(name);return
  raise AssertionError('must reject '+name)
 with tempfile.TemporaryDirectory(prefix='SYNTHETIC_FORWARD_ONLY_') as tmp:
  tmp=Path(tmp);p=tmp/'protected.bin';p.write_bytes(b'constructed CPU fixture')
  r={'bytes':p.stat().st_size,'sha256':f.sha(p)}
  wrapped={'protected_files':{str(p):r},'all_unchanged':True,'original_science_files':451}
  check(f.protected_records(wrapped)=={str(p):r},'nested manifest explicit extraction')
  reject(lambda:f.protected_records({str(p):r}),'flat old bug rejected')
  reject(lambda:f.protected_records({'protected_files':[] }),'nonmapping rejected')
  reject(lambda:f.protected_records({'protected_files':{} }),'empty rejected')
  check(f.preserve(f.protected_records(wrapped))['all_unchanged'],'physical protected SHA')
  bad=copy.deepcopy(wrapped);bad['protected_files'][str(p)]['sha256']='0'*64
  reject(lambda:f.preserve(f.protected_records(bad)),'protected corruption rejected')
  spec=tmp/'spec.json';spec.write_text(json.dumps({'scope':'SYNTHETIC_NOT_SCIENTIFIC'}))
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
  def cli(args):
   r=subprocess.run([sys.executable,'-B',str(ROOT/'tools/sabl_forward_check.py'),*args],cwd=ROOT,env=env,capture_output=True,text=True)
   check(len(r.stdout.encode())+len(r.stderr.encode())<24000,'bounded CLI output '+str(len(passed)))
   return r
  out=tmp/'actual-cli';r=cli(['fixture','--spec',str(spec),'--out',str(out)])
  if r.returncode:raise AssertionError('actual CLI normal: '+r.stderr[-8000:])
  result=f.read(out/'RESULT.json');check(result['status']=='SYNTHETIC_PASS_NOT_SCIENTIFIC','CLI two-arm closed')
  check((out/'CLOSED.json').exists() and f.read(out/'CLOSED.json')['result_sha256']==f.sha(out/'RESULT.json'),'closed SHA')
  check(result['counts']['detector_forward_reserved']==2 and result['counts']['backward']==result['counts']['optimizer_updates']==result['counts']['EMA_updates']==0,'two forwards zero prohibited operations')
  check(result['CUDA_initialized'] is False,'hidden CUDA')
  for arm in result['reports']:
   check(arm['bridge']['all_raw_bytes_exact'] and not arm['bridge']['full_training_state_restored'],'strict raw bridge '+arm['forward']['arm'])
   check(arm['forward']['parameters_unchanged'] and arm['forward']['BN_changes_in_memory_only'],'params/BN '+arm['forward']['arm'])
   check(len(arm['forward']['local'])==2 and all(x['new_finite'] for x in arm['forward']['local']),'local axes '+arm['forward']['arm'])
  check(cli(['fixture','--spec',str(spec),'--out',str(out)]).returncode!=0,'duplicate CLI rejected')
  for fault in ('version','missing_tensor','dtype','mode','nonfinite','forward_cap'):
   spec.write_text(json.dumps({'scope':'SYNTHETIC_NOT_SCIENTIFIC','fault':fault}));r=cli(['fixture','--spec',str(spec),'--out',str(tmp/fault)])
   check(r.returncode!=0 and not (tmp/fault/'CLOSED.json').exists(),'actual CLI rejects '+fault)
  from unittest.mock import patch
  sc,_=f.import_model();model=f.FixtureModel.create('A');saw=[]
  meter=f.Meter(tmp,tmp,'cpu');saved={'roi_before_A_mean':torch.ones(1,256,7,7)}
  obs=f.Observe(model,sc,meter,'A',saved);obs.index=0
  try:reject(lambda:model.boundary(torch.zeros(1,256,7,7),'A'),'first RoI exact mismatch stops')
  finally:obs.close()
  check(obs.matches[0]['exact'] is False,'mismatch evidence retained')
  check(sc.normalized_sigmoid_axis is obs.original,'observer restoration')
  with f.no_updates():
   reject(lambda:torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]),'optimizer construction forbidden')
   reject(lambda:torch.ones(1,requires_grad=True).backward(),'backward forbidden')
  meter=f.Meter(tmp,tmp,'cpu', {**f.CAPS,'wall_seconds':-1});reject(meter.guard,'wall cap')
  meter=f.Meter(tmp,tmp,'cpu', {**f.CAPS,'storage_bytes':1});reject(meter.guard,'storage cap reserve')
  check(f.CAPS=={'forward':2,'backward':0,'optimizer_updates':0,'EMA_updates':0,'GPU_seconds':300,'wall_seconds':600,'allocated_bytes':18<<30,'storage_bytes':128<<20,'return_bytes':1<<20},'fixed budgets')
 result={'status':'PASS_CPU_CLI_NOT_SCIENTIFIC','checks':passed,'count':len(passed),'CUDA_initialized':torch.cuda.is_initialized(),'full_detector_forward':0,'optimizer_updates':0,'production_inputs_used':False}
 Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'checks':len(passed)}))
if __name__=='__main__':main()
