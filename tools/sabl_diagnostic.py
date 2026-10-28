"""One PR26 trajectory reproduction. Imports the frozen science, never repairs it."""
import argparse, datetime, gc, hashlib, inspect, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path

SCIENCE='79ba59371c21ecfd4fa76885c413941403797214'
MANIFEST='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab'
CHECKPOINT='3c4972fcb245f1a9890b9f65080da4441cdc37f2ecb060c25478eb574d14d899'
TASK=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/r18-sabl-formal24-20261027-001')
RELEASE=TASK/'releases/79ba59371c21-formal-001'
CAPS={'forward':661,'backward':660,'updates':330,'GPU_seconds':1200,'wall_seconds':1800,'allocated_bytes':18<<30,'storage_bytes':512<<20}
RESULT_KEYS=('update','epoch','window','microsteps','lr','coefficients','microbatch_losses','loss','normalizers','gradient_norm_before_clip','gradient_sha256_before_clip','output_sha256','dn_noise_sha256','finite','optimizer_steps','ema_updates')

def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(path,value):
 p=Path(path);q=p.with_suffix('.tmp')
 with q.open('w') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def reduced(result):return {k:result[k] for k in RESULT_KEYS}
def expected_row(row):return {'update':row['update'],'input':digest(row['input']),'result':digest(reduced(row['result'])),'loss':row['result']['loss'],'gradient_norm_before_clip':row['result']['gradient_norm_before_clip']}

def load_science(root):
 for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1'}.items():os.environ[k]=v
 sys.path.insert(0,str(root/'tools'));sys.path.insert(0,str(root/'src'))
 import sabl_formal
 from sparse_rtdetr.research import sabl_formal_state, sabl_formal_data, sabl_control
 need(Path(sabl_formal.__file__).resolve()==(root/'tools/sabl_formal.py').resolve(),'science import origin')
 return sabl_formal,sabl_formal_state,sabl_formal_data,sabl_control

def verify_science(root):
 need(sha(root/'RELEASE_MANIFEST.json')==MANIFEST,'frozen manifest SHA')
 m=read(root/'RELEASE_MANIFEST.json');need(m['commit']==SCIENCE and len(m['files'])==451,'science identity')
 for r in m['files']:
  p=root/r['path'];need(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'science bytes '+r['path'])
 return {'commit':SCIENCE,'manifest':MANIFEST,'files':451}

def prepare(a):
 """CPU-hidden admission; publishes evidence inputs only, not a launch marker."""
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','prepare must hide CUDA')
 root=Path(a.root).resolve();science=verify_science(root)
 original=TASK/'attempt-formal-001';cp=original/'train-A/checkpoint-e002.pt'
 need(sha(cp)==CHECKPOINT,'e2 checkpoint SHA');ref=read(str(cp)+'.complete.json')
 need(ref['updates']==608 and ref['sha256']==CHECKPOINT and ref['bytes']==cp.stat().st_size,'e2 sidecar')
 c=read(TASK/'contract.json');need(digest(c)==ref['binding']['contract'],'original contract')
 need(read(original/'STOP.json')['error']=="ValueError('zero attention denominator')" and read(original/'EXIT.json')['returncode']==1,'original STOP')
 rows=[]
 with (original/'train-A/remaining-events.jsonl').open() as f:
  for line in f:
   row=json.loads(line);need(row['update']==609+len(rows),'reference update order');rows.append(expected_row(row))
 need(len(rows)==330 and rows[-1]['update']==938,'reference length')
 protected=[]
 for p in sorted(original.rglob('*')):
  if p.is_file():protected.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
 endpoint={}
 for arm in ('N','S'):
  ep=read(original/('train-'+arm)/'endpoint.json');r=ep['checkpoint'];need(sha(r['path'])==r['sha256'],'preserved '+arm);endpoint[arm]=r
 for spec in c['metadata']['data'].values():need(sha(spec['manifest']['path'])==spec['manifest']['sha256'],'data manifest')
 stage=Path(a.out);stage.mkdir()
 write(stage/'EXPECTED.json',rows);write(stage/'PROTECTED.json',protected)
 definition={'schema':'PR26_DIAGNOSTIC_REPLAY_V1','science':science,'root':str(root),'original_contract':str(TASK/'contract.json'),'original_contract_sha':sha(TASK/'contract.json'),'original_checkpoint':ref,'original_STOP':{'path':str(original/'STOP.json'),'sha256':sha(original/'STOP.json')},'reference_log':{'path':str(original/'train-A/remaining-events.jsonl'),'sha256':sha(original/'train-A/remaining-events.jsonl')},'expected':{'path':str(stage/'EXPECTED.json'),'sha256':sha(stage/'EXPECTED.json')},'protected':{'path':str(stage/'PROTECTED.json'),'sha256':sha(stage/'PROTECTED.json')},'NS':endpoint,'caps':CAPS,'comparison':'exact canonical JSON digest; full input identity and all RESULT_KEYS except elapsed_seconds; no tolerance','result_keys':list(RESULT_KEYS),'prior':read(original/'ledger.json'),'diagnostic_root':str(stage.parent),'production':True}
 write(stage/'DEFINITION.json',definition)
 print(json.dumps({'status':'PREPARED','definition_sha256':sha(stage/'DEFINITION.json'),'expected_rows':330,'forward':0}))

class StopMismatch(Exception):pass
class TargetSucceeded(Exception):pass

class Meter:
 def __init__(self,out,caps,cuda,storage_root,started=None):
  self.out=out;self.caps=caps;self.cuda=cuda;self.storage_root=storage_root;self.start=time.monotonic() if started is None else started;self.count={'forward':0,'backward':0,'updates':0};self.peak=0;self.publish()
 def publish(self):atomic(self.out/'ledger.json',{'counts':self.count,'seconds':time.monotonic()-self.start,'peak_allocated_bytes':self.peak,'diagnostic_cost_additional':True})
 def guard(self):
  elapsed=time.monotonic()-self.start;need(elapsed<=self.caps['wall_seconds'],'wall cap');need(elapsed<=self.caps['GPU_seconds'],'GPU/CPU diagnostic phase cap')
  if self.cuda:
   import torch
   self.peak=max(self.peak,torch.cuda.max_memory_allocated());need(self.peak<=self.caps['allocated_bytes'],'allocated cap')
  # Unique inodes prevent hardlink publication from double-counting physical storage.
  seen=set();size=0
  for p in self.storage_root.rglob('*'):
   if p.is_file():
    s=p.stat();key=(s.st_dev,s.st_ino)
    if key not in seen:seen.add(key);size+=s.st_size
  need(size+(64<<20 if self.cuda else 0)<=self.caps['storage_bytes'],'storage cap including conservative development/freeze reserve')
 def tick(self,key):
  if key!='updates':self.guard()
  need(self.count[key]<self.caps[key],key+' cap');self.count[key]+=1;self.publish()
  if key=='updates':self.guard()

class Observation:
 def __init__(self,model,meter,control):
  import torch
  self.torch=torch;self.model=model;self.meter=meter;self.control=control;self.active=False;self.rows=[];self.refs={};self.handles=[];self.first=None
  self.head_code=control.BoundaryHead.forward.__code__;self.backward_code=torch.Tensor.backward.__code__
  lines,start=inspect.getsourcelines(control.BoundaryHead.forward)
  self.positions={start+i:tag for i,line in enumerate(lines) for text,tag in [('x=self.pre(roi)','roi_after_A_mean'),('dx=ax.sum','sigmoid'),('need(bool((dx>0)','denominators')] if text in line}
  for name,module in model.boundary.named_modules():
   if name in ('pre.0','pre.1','pre.2','pre.3','att_x','att_y'):self.handles.append(module.register_forward_hook(self.hook(name)))
  self.handles.append(model.boundary.register_forward_pre_hook(lambda module,args:self.record('roi_before_A_mean',args[0]) if self.active else None))
  self.handles.append(model.encoder.register_forward_hook(lambda module,args,value:self.record('F8',value[0],False) if self.active else None))
 def hook(self,name):
  def observe(module,args,value):
   if self.active:self.record(name,value)
  return observe
 def record(self,name,t,keep=True,denominator=False):
  torch=self.torch;x=t.detach();finite=torch.isfinite(x);zero=x==0
  row={'name':name,'dtype':str(x.dtype),'shape':list(x.shape),'finite':int(finite.sum()),'nonfinite':int((~finite).sum()),'zero':int(zero.sum()),'negative':int((x<0).sum())}
  vals=x[finite];row['finite_min']=float(vals.min()) if vals.numel() else None;row['finite_max']=float(vals.max()) if vals.numel() else None
  bad=(~finite)|((x<=0) if denominator else torch.zeros((),device=x.device,dtype=torch.bool))
  if bool(bad.any()):
   idx=bad.nonzero()[0].tolist();row['first_bad_index']=idx
   if self.first is None:self.first={'name':name,'index':idx,'dtype':str(x.dtype),'kind':'nonfinite' if not bool(finite[tuple(idx)]) else 'nonpositive_denominator'}
  if row['zero']:row['first_zero_index']=zero.nonzero()[0].tolist()
  self.rows.append(row)
  if keep:self.refs[name]=x
  self.meter.guard()
 def trace(self,frame,event,arg):
  if event=='call':
   if frame.f_code is self.backward_code:self.meter.tick('backward');return None
   if self.active and frame.f_code is self.head_code:return self.trace
   return None
  if event=='line' and frame.f_code is self.head_code:
   tag=self.positions.get(frame.f_lineno);v=frame.f_locals
   if tag=='roi_after_A_mean':self.record(tag,v['roi'])
   elif tag=='sigmoid':self.record('ax',v['ax']);self.record('ay',v['ay'])
   elif tag=='denominators':self.record('dx',v['dx'],denominator=True);self.record('dy',v['dy'],denominator=True)
  return self.trace
 def close(self):
  for h in self.handles:h.remove()
 def export(self,out):
  index=0
  if self.first and self.first['name'] in self.refs:index=self.first['index'][0]
  blocks={}
  for name,x in self.refs.items():
   if x.ndim and x.shape[0]>index:blocks[name]=x[index:index+1].detach().cpu().clone()
  p=out/'small-tensors.pt';self.torch.save(blocks,p);need(p.stat().st_size<=2<<20,'tensor attachment cap')
  write(out/'OBSERVATIONS.json',{'first_anomaly':self.first,'rows':self.rows,'selected_flat_roi_index':index,'tensor_file':{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size},'only_target_forward_observed':True})

def run_core(out,root,definition,fixture=None):
 started=time.monotonic()
 formal,state,data,control=load_science(root)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 cuda=fixture is None;device='cuda' if cuda else 'cpu'
 c=read(definition['original_contract']) if cuda else fixture['contract']
 if cuda:
  need(sha(definition['original_contract'])==definition['original_contract_sha'],'contract bytes');verify_science(root)
  need(definition['caps']==CAPS and definition['result_keys']==list(RESULT_KEYS),'frozen comparison/caps')
  os.environ['CUDA_VISIBLE_DEVICES']=c['gpu_uuid']
  from sabl_formal_preflight import resources
  write(out/'RESOURCE.json',resources(c,out))
  import torchvision
  env=c['metadata']['environment'];need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'runtime identity')
  need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'GPU/BF16')
  need(sha(definition['expected']['path'])==definition['expected']['sha256'] and sha(definition['reference_log']['path'])==definition['reference_log']['sha256'],'reference bytes')
  need(sha(definition['original_checkpoint']['path'])==CHECKPOINT,'start checkpoint')
  need(sha(definition['original_STOP']['path'])==definition['original_STOP']['sha256'],'preserve original STOP')
  for spec in c['metadata']['data'].values():need(sha(spec['manifest']['path'])==spec['manifest']['sha256'],'data manifest')
  torch.cuda.reset_peak_memory_stats();expected=read(definition['expected']['path']);start_u=608;target=938
 else:expected=fixture['expected'];start_u=0;target=len(expected)
 caps=CAPS if cuda else {'forward':2*target+1,'backward':2*target,'updates':target,'GPU_seconds':300,'wall_seconds':300,'allocated_bytes':18<<30,'storage_bytes':512<<20}
 if fixture and fixture.get('fault')=='forward_cap':caps={**caps,'forward':0}
 meter=Meter(out,caps,cuda,Path(definition['diagnostic_root']) if cuda else out,started)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('diagnostic deadline')));signal.alarm(caps['GPU_seconds'])
 engine,binding,_,_=formal.fresh(c,'A',device)
 cp=definition['original_checkpoint'] if cuda else fixture['checkpoint']
 audit=state.restore(engine,binding,cp['path'],cp['sha256']);need(audit['full_state_exact'] and engine.completed_updates==start_u,'full starting restore')
 write(out/'RESTORE.json',audit)
 if fixture:
  # Only newly constructed CPU test inputs are retired, never a production state.
  cp_path=Path(cp['path']);need(cp_path.name=='fixture-initial.pt' and cp_path.parent==out.parent,'CPU temporary input identity')
  cp_path.unlink();Path(str(cp_path)+'.complete.json').unlink()
 ds=data.Dataset(c['metadata']['data']['train_core']) if cuda else data.Fixture()
 observation=Observation(engine.model,meter,control)
 def forward_count(module,args):meter.tick('forward')
 def precision(module,args,value):
  need(str(value['pred_logits'].dtype)==('torch.bfloat16' if cuda else 'torch.float32'),'native dtype');need(module.last_boundary_dtype=='torch.float32','boundary dtype')
 def end_target(module,args,value):
  if observation.active:raise TargetSucceeded('target forward completed; prohibit criterion/backward')
 handles=[engine.model.register_forward_pre_hook(forward_count),engine.model.register_forward_hook(precision),engine.model.register_forward_hook(end_target)]
 old_trace=sys.gettrace();need(old_trace is None,'unexpected existing tracer');sys.settrace(observation.trace)
 status='STOP';error=None;matched=0;checkpoint=None
 try:
  windows=((epoch,values) for epoch in (3,4) for values in data.windows(ds,epoch)) if cuda else ((i+1,ds.window(i+1)) for i in range(target+1))
  with (out/'REPLAY.jsonl').open('x') as log:
   for epoch,(images,targets,identity) in windows:
    if engine.completed_updates==target:
     need(matched==len(expected),'missing replay comparisons');checkpoint=state.save(engine,binding,out/'checkpoint-u938.pt' if cuda else out/'checkpoint-fixture-boundary.pt');meter.guard()
     if fixture and fixture.get('fault')=='target_zero':
      with torch.no_grad():engine.model.boundary.att_x.weight.zero_();engine.model.boundary.att_x.bias.fill_(-1000)
     write(out/'TARGET_INPUT.json',{'epoch':epoch,'update':engine.completed_updates+1,'window_zero_based':26 if cuda else target,'microbatch':0,'identity_sha256':digest(identity),'input_tensor_sha256':identity.get('input_tensor_sha256',identity.get('input_SHA')),'target_sha256':identity.get('target_sha256'),'ids':identity.get('ids')})
     observation.active=True
     try:engine.train_window(images,targets,capture=False)
     except TargetSucceeded:status='NOT_REPRODUCED'
     except ValueError as e:
      if str(e)=='zero attention denominator':status='REPRODUCED_ORIGINAL_CHECK_FAILURE';error=repr(e)
      else:raise
     need(meter.count=={'forward':2*target-2*start_u+1,'backward':2*(target-start_u),'updates':target-start_u},'terminal counts')
     observation.export(out);break
    ref=expected[matched]
    if digest(identity)!=ref['input']:raise StopMismatch('input/GT identity at update '+str(engine.completed_updates+1))
    meter.guard();need(meter.count['updates']<caps['updates'],'update hard cap before call')
    result=engine.train_window(images,targets,capture=False);meter.tick('updates')
    if digest(reduced(result))!=ref['result']:raise StopMismatch('loss/gradient/result at update '+str(engine.completed_updates))
    matched+=1;log.write(json.dumps({'update':engine.completed_updates,'input_match':True,'result_match':True,'loss':result['loss'],'gradient_norm_before_clip':result['gradient_norm_before_clip']})+'\n');log.flush();meter.guard()
   else:raise RuntimeError('missing target window')
 except StopMismatch as e:status='STOP_TRAJECTORY_MISMATCH';error=str(e)
 except Exception as e:
  status='STOP_DIAGNOSTIC_EXCEPTION';error=repr(e);(out/'stderr.txt').write_text(traceback.format_exc())
 finally:
  sys.settrace(old_trace);signal.alarm(0);observation.close()
  for h in handles:h.remove()
  meter.publish()
 if observation.active and not (out/'OBSERVATIONS.json').exists():
  write(out/'OBSERVATIONS.json',{'first_anomaly':observation.first,'rows':observation.rows,'tensor_extraction_skipped_on_other_exception':True})
 result={'status':status,'error':error,'matched_replay_updates':matched,'counts':meter.count,'seconds':time.monotonic()-meter.start,'peak_allocated_bytes':meter.peak,'saved_boundary':checkpoint,'first_anomaly':observation.first,'not_training_continuation':True,'no_retry':True,'scientific_quality':'UNKNOWN','fixture':not cuda}
 if cuda:
  protected=read(definition['protected']['path']);changes=[]
  for r in protected:
   p=Path(r['path']);s=p.stat()
   if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:changes.append(r['path'])
  need(not changes,'original scene changed');verify_science(root)
  result['original_scene_unchanged']=True;result['prior_cost']=definition['prior'];result['cumulative_forward']=definition['prior']['total_forward']+meter.count['forward'];result['cumulative_GPU_stage_seconds']=definition['prior']['GPU_stage_seconds']+result['seconds']
 write(out/'RESULT.json',result);write(out/'CLOSED.json',{'status':status,'result_sha256':sha(out/'RESULT.json'),'no_followup':True})
 print(json.dumps({'status':status,'matched':matched,'counts':meter.count,'seconds':result['seconds']}))
 return 0 if status in ('NOT_REPRODUCED','REPRODUCED_ORIGINAL_CHECK_FAILURE') else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('prepare');q.add_argument('--root',default=str(RELEASE));q.add_argument('--out',required=True)
 q=s.add_parser('run');q.add_argument('--definition',required=True);q.add_argument('--authorization',required=True);q.add_argument('--out',required=True)
 q=s.add_parser('fixture');q.add_argument('--source-root',required=True);q.add_argument('--fixture',required=True);q.add_argument('--out',required=True)
 a=p.parse_args()
 if a.command=='prepare':prepare(a);return
 out=Path(a.out)
 if a.command=='fixture':
  need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixture CUDA must be hidden');fx=read(a.fixture);out.mkdir();return run_core(out,Path(a.source_root).resolve(),{'diagnostic_root':str(out)},fx)
 definition=read(a.definition);need(definition['production'] is True and definition['schema']=='PR26_DIAGNOSTIC_REPLAY_V1','production definition')
 auth=read(a.authorization);need(auth=={'scope':'PR26_DIAGNOSTIC_REPLAY_V1','definition_sha256':sha(a.definition),'diagnostic_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'attempt':str(out.resolve()),'authorize_once':True},'unique authorization binding')
 need(out.parent.resolve()==Path(definition['diagnostic_root']).resolve() and out.name=='attempt-diagnostic-001','diagnostic output identity')
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'dirty diagnostic source')
 out.mkdir();write(out/'CONSUMED.json',{'definition_sha256':sha(a.definition),'no_retry':True})
 try:return run_core(out,Path(definition['root']),definition)
 except BaseException as e:
  if not (out/'RESULT.json').exists():write(out/'STOP.json',{'error':repr(e),'no_retry':True});(out/'stderr.txt').write_text(traceback.format_exc())
  raise

if __name__=='__main__':raise SystemExit(main())
