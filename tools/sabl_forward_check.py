"""Read-only raw-model bridge and exactly two forward-only engineering probes.

This is not a training-state restore or an evaluation/benchmark launcher.
"""
import argparse,contextlib,gc,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT/'tools'):sys.path.insert(0,str(p))
BASE='992b6b13f0b866700a9bd631ec1a557da96e025e'
OLD_VERSION='r18-sabl-control-v1'
NEW_VERSION='r18-sabl-control-logsigmoid-v2'
OLD_MANIFEST='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab'
OLD_CONTRACT='f734c400e942c265eafbb5180617a09c3b827a9b238f2a65bf08f24b5557356e'
CAPS={'forward':2,'backward':0,'optimizer_updates':0,'EMA_updates':0,'GPU_seconds':300,'wall_seconds':600,'allocated_bytes':18<<30,'storage_bytes':128<<20,'return_bytes':1<<20}
CP_SHA={'A':'b9e7a005d356ab67ffd4004d568e8eca222c1198c6e2e00d9af167be34100878','S':'deb67cf4bfc7e1a25998fb8490309a0ee812471859bacfbe8a42d8be4673f21f'}
CP_UPDATES={'A':938,'S':7296}
FORMAT='sparse_rtdetr.sabl_formal24_fullstate_v1'
STATE_KEYS={'format','binding','runtime','device','model','model_layout','model_training','optimizer','optimizer_layout','ema','engine','rng','criterion'}
TASK=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/r18-sabl-forward-validation-20261028-001')

def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def exact(ref):
 p=Path(ref['path']);need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'bound file '+str(p));return p
def protected_records(data):
 need(isinstance(data,dict) and isinstance(data.get('protected_files'),dict),'protected manifest schema')
 rows=data['protected_files'];need(bool(rows),'empty protected manifest')
 for p,r in rows.items():
  need(Path(p).is_absolute() and isinstance(r,dict) and type(r.get('bytes')) is int and r['bytes']>0 and isinstance(r.get('sha256'),str) and len(r['sha256'])==64,'protected record schema')
 return rows
def preserve(rows):
 for p,r in rows.items():exact({'path':p,**r})
 return {'files':len(rows),'all_unchanged':True}
def bytes_of(t):return t.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()
def tensor_sha(t):return hashlib.sha256(bytes_of(t)).hexdigest()
def equal_tensor(a,b):return a.shape==b.shape and a.dtype==b.dtype and bytes_of(a)==bytes_of(b)
def finite_tree(x,rows=None,path='out'):
 import torch
 rows=[] if rows is None else rows
 if isinstance(x,torch.Tensor):
  need(bool(torch.isfinite(x).all()),'nonfinite output '+path)
  rows.append({'path':path,'shape':list(x.shape),'dtype':str(x.dtype),'finite':True,'sha256':tensor_sha(x)})
 elif isinstance(x,dict):
  for k,v in sorted(x.items()):finite_tree(v,rows,path+'/'+k)
 elif isinstance(x,(tuple,list)):
  for i,v in enumerate(x):finite_tree(v,rows,path+'/'+str(i))
 elif x is None or type(x) in (bool,int,float,str):canonical(x)
 else:raise TypeError('unknown output '+path)
 return rows

def import_model():
 import torch
 from sparse_rtdetr.research import sabl_control as sc
 need(sc.VERSION==NEW_VERSION,'new computation version')
 need(Path(sc.__file__).resolve()==ROOT/'src/sparse_rtdetr/research/sabl_control.py','module origin')
 return sc,torch

def expected_binding(arm,synthetic=False):
 return {'scope':'CPU_FORWARD_FIXTURE' if synthetic else 'GPU_SABL_FORMAL_24','arm':arm,'version':OLD_VERSION,'contract':OLD_CONTRACT,'release_manifest':OLD_MANIFEST,'N_SHA':'be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'}

def bridge(model,state,arm,synthetic=False):
 """Explicit OLD->NEW raw tensor bridge, never restore optimizer/EMA/engine."""
 import torch
 from sparse_rtdetr.baseline import training_v2b_checkpoint as layout
 need(set(state)==STATE_KEYS and state['format']==FORMAT,'complete source schema')
 need(state['binding']==expected_binding(arm,synthetic),'registered source binding')
 need(state['runtime']==str(torch.__version__) and state['device']==('cpu' if synthetic else 'cuda:0'),'source runtime/device')
 e=state['engine'];u=1 if synthetic else CP_UPDATES[arm]
 need(type(e['completed_updates']) is int and e['completed_updates']==u and e['phase']==0 and e['failed'] is False,'source boundary')
 if not synthetic:
  from sparse_rtdetr.research.sabl_control import TrainConfig
  cfg=dict(TrainConfig().binding());cfg['arm']=OLD_VERSION
  need(e['config']==cfg,'source frozen calculation config')
  need(state['criterion']['version']==OLD_VERSION and state['criterion']['arm']==arm,'source supervision version')
 current=layout._model_layout(model)
 need(state['model_layout']==current,'full parameter/buffer/cache layout')
 layout._check_model_state(state['model'],current,'explicit raw-model bridge')
 modes=state['model_training'];need(set(modes)=={n for n,m in model.named_modules()} and all(type(v) is bool and v for v in modes.values()),'source training modes')
 need(set(state['rng'])=={'python','numpy','torch_cpu','cuda'},'source RNG schema')
 need((state['rng']['cuda'] is None)==synthetic,'source RNG device')
 model.load_state_dict(state['model'],strict=True)
 need(all(equal_tensor(v,state['model'][k]) for k,v in model.state_dict().items()),'raw state copy exact')
 model.train()
 return {'source_version':OLD_VERSION,'target_version':NEW_VERSION,'raw_model_only':True,'full_training_state_restored':False,'optimizer_loaded':False,'EMA_loaded':False,'engine_loaded':False,'updates_identity':u,'state_tensors':len(state['model']),'parameter_tensors':len(list(model.parameters())),'buffer_tensors':len(list(model.buffers())),'layout_exact':True,'all_raw_bytes_exact':True}

class Meter:
 def __init__(self,out,storage,device,cap=None):
  self.out=out;self.storage=storage;self.device=device;self.cap=CAPS if cap is None else cap;self.start=time.monotonic();self.gpu_start=None;self.count=0;self.peak=0
 def guard(self):
  import torch
  need(time.monotonic()-self.start<=self.cap['wall_seconds'],'wall limit')
  if self.gpu_start is not None:need(time.monotonic()-self.gpu_start<=self.cap['GPU_seconds'],'GPU phase limit')
  if self.device=='cuda':
   self.peak=max(self.peak,torch.cuda.max_memory_allocated());need(self.peak<=self.cap['allocated_bytes'],'allocated memory limit')
  size=sum(p.stat().st_size for p in self.storage.rglob('*') if p.is_file())
  need(size+(32<<20)<=self.cap['storage_bytes'],'storage limit with development reserve')
 def forward(self,module,args):
  self.guard();need(self.count<self.cap['forward'],'forward limit');self.count+=1;self.publish()
 def publish(self):
  p=self.out/'ledger.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(self.snapshot()));os.replace(t,p)
 def snapshot(self):return {'detector_forward_reserved':self.count,'backward':0,'optimizer_updates':0,'EMA_updates':0,'wall_seconds':time.monotonic()-self.start,'GPU_seconds':0 if self.gpu_start is None else time.monotonic()-self.gpu_start,'peak_allocated_bytes':self.peak}

class Observe:
 def __init__(self,model,sc,meter,arm,saved=None):
  self.model=model;self.sc=sc;self.meter=meter;self.arm=arm;self.saved=saved;self.index=352 if saved is not None else 0;self.matches=[];self.local=[];self.x=None;self.h=[];self.original=sc.normalized_sigmoid_axis
  def check(name,value):
   import torch
   need(bool(torch.isfinite(value).all()),'upstream nonfinite '+name)
   if saved is not None:
    actual=value[self.index:self.index+1];expected=saved[name]
    match=equal_tensor(actual,expected);row={'name':name,'actual_dtype':str(actual.dtype),'shape':list(actual.shape),'actual_sha256':tensor_sha(actual),'expected_sha256':tensor_sha(expected),'exact':match};self.matches.append(row)
    need(match,'first A evidence mismatch '+name)
   self.meter.guard()
  self.h.append(model.boundary.register_forward_pre_hook(lambda m,a:check('roi_before_A_mean',a[0])))
  self.h.append(model.boundary.pre.register_forward_pre_hook(lambda m,a:check('roi_after_A_mean',a[0])))
  def post(m,a,v):check('pre.3',v);self.x=v.detach()
  self.h.append(model.boundary.pre[3].register_forward_hook(post))
  for name in ('att_x','att_y'):
   self.h.append(getattr(model.boundary,name).register_forward_hook(lambda m,a,v,n=name:check(n,v)))
  def observed(logits,axis):
   import torch
   value=self.original(logits,axis)
   with torch.no_grad():
    error=float((value.sum(axis)-1).abs().max());need(error<=1e-6,'axis normalization error')
    old=logits.detach().sigmoid();den=old.sum(axis,keepdim=True);bad=den<=0;oldw=old/den
    newpool=(self.x*value.detach()).sum(axis);oldpool=(self.x*oldw).sum(axis)
    valid=torch.isfinite(oldpool)&torch.isfinite(newpool)
    diff=(newpool[valid]-oldpool[valid]).abs();row={'axis':axis,'logits_dtype':str(logits.dtype),'weight_dtype':str(value.dtype),'pool_dtype':str(newpool.dtype),'new_finite':bool(torch.isfinite(value).all() and torch.isfinite(newpool).all()),'axis_error':error,'old_zero_denominators':int(bad.sum()),'old_nonfinite_weights':int((~torch.isfinite(oldw)).sum()),'old_nonfinite_pool':int((~torch.isfinite(oldpool)).sum()),'finite_pool_elements':int(valid.sum()),'finite_pool_maxabs':float(diff.max()) if diff.numel() else None}
    if saved is not None:
     col=value[self.index,0,:,1] if axis==2 else value[self.index,0,1,:]
     row['specified_new_weights']=col.cpu().tolist();row['specified_old_denominator']=float(den[self.index,0,0,1] if axis==2 else den[self.index,0,1,0])
     if axis==2:need(row['specified_old_denominator']==0,'specified original denominator no longer zero')
    self.local.append(row);need(row['new_finite'],'new local nonfinite');self.meter.guard()
   return value
  sc.normalized_sigmoid_axis=observed
 def close(self):
  self.sc.normalized_sigmoid_axis=self.original
  for h in self.h:h.remove()
  self.x=None

def forbidden(*a,**kw):raise RuntimeError('backward/optimizer/EMA operations forbidden')
@contextlib.contextmanager
def no_updates():
 import torch
 from unittest.mock import patch
 from sparse_rtdetr.protocol_r18.model import CommonEMA
 with patch.object(torch.Tensor,'backward',forbidden),patch.object(torch.autograd,'backward',forbidden),patch.object(torch.autograd,'grad',forbidden),patch.object(torch.optim.Optimizer,'__init__',forbidden),patch.object(CommonEMA,'update',forbidden):yield

def one(model,sc,state,images,targets,meter,arm,saved=None,synthetic=False):
 import torch
 from sparse_rtdetr.protocol_r18.checkpoint import restore_rng
 from sparse_rtdetr.baseline.training_v2b_engine import _batchnorm_backward_layout,_move
 from sparse_rtdetr.protocol_r18.model import dn_capture
 from sparse_rtdetr.protocol_r18.geometry import forward_guards
 from sparse_rtdetr.protocol_r18.engine import tensor_digest
 device=torch.device(meter.device)
 param_before=tensor_digest(model.named_parameters());buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
 restore_rng(state['rng'],device)
 before_rng=tensor_sha(torch.get_rng_state());model.train()
 observation=Observe(model,sc,meter,arm,saved);h=model.register_forward_pre_hook(meter.forward)
 try:
  with torch.enable_grad(),no_updates(),_batchnorm_backward_layout(model,device) if not synthetic else contextlib.nullcontext():
   batch=images[:8].to(device);subset=_move(targets[:8],device)
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if not synthetic else contextlib.nullcontext()
   with ctx,dn_capture(model) if not synthetic else contextlib.nullcontext(),forward_guards(model) if not synthetic else contextlib.nullcontext():output=model(batch,targets=subset)
   need(model.last_dtype==('torch.float32' if synthetic else 'torch.bfloat16') and model.last_boundary_dtype=='torch.float32','actual native/local precision')
   rows=finite_tree(output);need('sabl' in output,'training forward SABL output')
   if not synthetic:need('dn_aux_outputs' in output and len(output['aux_outputs'])==2,'unchanged DN/aux output path')
   need(all(p.grad is None for p in model.parameters()),'gradient unexpectedly populated')
   del output,batch,subset
  need(tensor_digest(model.named_parameters())==param_before,'parameters mutated')
  changed=[k for k,v in model.named_buffers() if not equal_tensor(v,buffers[k])]
  allowed={n+'.'+k if n else k for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for k in ('running_mean','running_var','num_batches_tracked')}
  need(set(changed)<=allowed,'unexpected non-BN buffer mutation')
  details=[]
  for k,v in model.named_buffers():
   if k in changed:
    now=v.detach().cpu();before=buffers[k];delta=(now.double()-before.double()).abs()
    details.append({'name':k,'dtype':str(v.dtype),'before_sha256':tensor_sha(before),'after_sha256':tensor_sha(now),'maxabs':float(delta.max()),'changed_elements':int((now!=before).sum())})
  meter.guard()
  return {'arm':arm,'raw_bridge':True,'training_mode':True,'grad_enabled_forward':True,'evidence_matches':observation.matches,'local':observation.local,'output_tensors':rows,'parameters_unchanged':True,'BN_mutated_buffers':changed,'BN_change_details':details,'BN_changes_in_memory_only':True,'state_not_saved':True,'CPU_rng_before_SHA':before_rng,'native_dtype':model.last_dtype,'boundary_dtype':model.last_boundary_dtype}
 except BaseException:
  write(meter.out/(arm+'_FAILED_OBSERVATION.json'),{'matches':observation.matches,'local':observation.local});raise
 finally:h.remove();observation.close()

def verify_release(path):
 m=read(path);need(m['base_commit']==BASE and m['scope']=='SABL_TWO_FORWARD_ONLY_V1','source manifest identity')
 for r in m['files']:exact({'path':str(ROOT/r['path']),**{k:r[k] for k in ('bytes','sha256')}})
 return m

def prepare(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU prepare hidden CUDA')
 sc,torch=import_model();torch.set_num_threads(2)
 out=Path(a.out);need(out.parent==TASK and not out.exists(),'fresh prepare target');out.mkdir()
 refs=read(TASK/'INPUT_REFS.json')
 for ref in refs.values():exact(ref)
 need(refs['A']['sha256']==CP_SHA['A'] and refs['S']['sha256']==CP_SHA['S'],'fixed raw states')
 protected=protected_records(read(TASK.parent/'r18-sabl-logsigmoid-fix-20261028-001/PROTECTED_AFTER.json'));preserve(protected)
 need(read(TASK/'STOP.json')['status']=='STOP_DEVELOPMENT_PREPARATION_EXCEPTION','retain original preparation STOP')
 c=read(refs['contract']['path']);need(digest(c)==OLD_CONTRACT and c['manifest_sha256']==OLD_MANIFEST,'original contract')
 target=read(refs['target']['path']);need(target['epoch']==4 and target['window_zero_based']==26 and target['update']==939,'target identity')
 from sparse_rtdetr.protocol_r18.model import Config
 audits={}
 with no_updates():
  for arm in ('A','S'):
   s=torch.load(refs[arm]['path'],map_location='cpu',weights_only=True)
   model,_=sc.build(arm,Config());audits[arm]=bridge(model,s,arm);del model,s;gc.collect()
 need(not torch.cuda.is_initialized(),'CPU prepare initialized CUDA')
 definition={'schema':'SABL_TWO_FORWARD_ONLY_V1','refs':refs,'protected':protected,'target':target,'raw_bridge_CPU':audits,'caps':CAPS,'root':str(TASK),'release':str(ROOT),'python':c['metadata']['python'],'GPU':c['gpu_uuid'],'environment':c['metadata']['environment'],'prior':{'forward':31740,'GPU_seconds':36518.94948015002},'no_retry':True}
 write(out/'DEFINITION.json',definition);print(json.dumps({'status':'PASS_CPU_PREPARE','definition_sha256':sha(out/'DEFINITION.json'),'raw_bridge_CPU':audits,'CUDA_initialized':False}))

def run(a):
 started=time.monotonic();out=Path(a.out);definition=read(a.definition)
 need(definition['schema']=='SABL_TWO_FORWARD_ONLY_V1' and definition['caps']==CAPS and Path(definition['root'])==TASK,'frozen definition')
 m=verify_release(a.manifest);auth=read(a.authorization)
 need(auth=={'scope':definition['schema'],'commit':m['commit'],'manifest_sha256':sha(a.manifest),'definition_sha256':sha(a.definition),'attempt':str(out),'authorize_once':True},'execution authorization binding')
 need(out==TASK/'attempt-forward-001' and not out.exists(),'unique attempt')
 with (TASK/'FORWARD_ONCE.json').open('x') as f:json.dump(auth,f)
 out.mkdir();meter=None;result=None
 try:
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s wall hard deadline')));signal.alarm(600)
  for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1'}.items():os.environ[k]=v
  need(os.environ.get('CUDA_VISIBLE_DEVICES')==definition['GPU'],'visible GPU binding')
  need(Path(sys.executable).resolve()==Path(definition['python']).resolve(),'interpreter')
  refs=definition['refs']
  for ref in refs.values():exact(ref)
  preserve(definition['protected'])
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();need(len(lines)==1,'one GPU')
  uuid,name,pci,free=[v.strip() for v in lines[0].split(',')]
  need(uuid==definition['GPU'] and name=='NVIDIA GeForce RTX 4090 D' and pci=='00000000:01:00.0' and int(free)>=18432,'fresh GPU resource identity')
  need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing GPU job')
  sc,torch=import_model();import torchvision
  env=definition['environment'];need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'environment identity')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
  gpu_started=time.monotonic()
  need(torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'BF16 GPU')
  torch.cuda.set_per_process_memory_fraction(CAPS['allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
  torch.cuda.reset_peak_memory_stats();meter=Meter(out,TASK,'cuda');meter.start=started;meter.gpu_start=gpu_started
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s GPU hard deadline')));signal.setitimer(signal.ITIMER_REAL,max(.001,300-(time.monotonic()-gpu_started)))
  c=read(refs['contract']['path'])
  from sparse_rtdetr.research import sabl_formal_data as data
  from sparse_rtdetr.localization.data import training_loader
  from sparse_rtdetr.protocol_r18.data import input_record
  ds=data.Dataset(c['metadata']['data']['train_core'])
  loader=training_loader(ds,epoch=4,completed_windows=26,window_limit=1)
  iterator=iter(loader)
  try:images,targets,traces=next(iterator);identity=input_record(images,targets,traces);identity.pop('spawn_workers')
  finally:
   del iterator,loader;gc.collect()
  target=definition['target'];need(digest(identity)==target['identity_sha256'] and identity['input_tensor_sha256']==target['input_tensor_sha256'] and identity['target_sha256']==target['target_sha256'] and identity['ids']==target['ids'],'exact logical16 input/GT before first8')
  write(out/'INPUT.json',{'identity_sha256':digest(identity),'ids':identity['ids'],'logical_images':16,'used_images':8,'target_sha256':identity['target_sha256']})
  saved=torch.load(refs['small']['path'],map_location='cpu',weights_only=True);reports=[]
  from sparse_rtdetr.protocol_r18.model import Config
  with no_updates():
   for arm in ('A','S'):
    meter.guard();s=torch.load(refs[arm]['path'],map_location='cpu',weights_only=True)
    model,_=sc.build(arm,Config());audit=bridge(model,s,arm);model.to('cuda')
    rng=s['rng'];del s;gc.collect()
    report=one(model,sc,{'rng':rng},images,targets,meter,arm,saved if arm=='A' else None)
    report['bridge']=audit;reports.append(report);write(out/(arm+'.json'),report)
    del model,rng,report;gc.collect();torch.cuda.empty_cache();meter.guard()
  need(meter.count==2,'exactly two detector forwards')
  torch.cuda.synchronize();meter.guard();meter.publish();preserved=preserve(definition['protected'])
  result={'status':'PASS_TWO_FORWARD_ENGINEERING_ONLY','reports':[{'arm':r['arm'],'output_tensors':len(r['output_tensors']),'evidence_exact':all(x['exact'] for x in r['evidence_matches']),'native_dtype':r['native_dtype'],'boundary_dtype':r['boundary_dtype'],'BN_changed':len(r['BN_mutated_buffers']),'local':r['local']} for r in reports],'cost':meter.snapshot(),'protected':preserved,'source_commit':m['commit'],'manifest_sha256':sha(a.manifest),'quality':'UNKNOWN','long_training_stability':'UNKNOWN','no_followup':True}
  write(out/'RESULT.json',result);write(out/'CLOSED.json',{'status':result['status'],'result_sha256':sha(out/'RESULT.json')})
  print(json.dumps({'status':result['status'],'cost':result['cost']}));return 0
 except BaseException as e:
  cost=meter.snapshot() if meter else {'detector_forward_reserved':0,'GPU_seconds':0,'wall_seconds':time.monotonic()-started}
  write(out/'STOP.json',{'status':'STOP_FORWARD_EXCEPTION','error':repr(e),'cost':cost,'no_retry':True});(out/'stderr.txt').write_text(traceback.format_exc());print(json.dumps({'status':'STOP_FORWARD_EXCEPTION','error':repr(e),'cost':cost}));return 1
 finally:signal.setitimer(signal.ITIMER_REAL,0)


class FixtureModel:
 @staticmethod
 def create(arm):
  import torch
  from torch import nn
  sc,_=import_model()
  class Model(nn.Module):
   def __init__(self):
    super().__init__();self.arm=arm;self.norm=nn.BatchNorm2d(256);self.boundary=sc.BoundaryHead();self.last_dtype=None;self.last_boundary_dtype=None
   def forward(self,images,targets=None):
    x=self.norm(images);z,d=self.boundary(x,self.arm)
    self.last_dtype='torch.float32';self.last_boundary_dtype=str(z.dtype)
    return {'pred_logits':z,'pred_boxes':torch.ones(len(z),1,4),'sabl':{'logits':z,'offsets':d,'reference':torch.ones(len(z),1,4)}}
  return Model()

def fixture(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixture hidden CUDA')
 sc,torch=import_model();torch.set_num_threads(2);spec=read(a.spec)
 need(spec['scope']=='SYNTHETIC_NOT_SCIENTIFIC','fixture scope')
 out=Path(a.out);need(not out.exists(),'fixture duplicate attempt');out.mkdir();meter=Meter(out,out,'cpu');reports=[]
 from sparse_rtdetr.baseline.training_v2b_checkpoint import _model_layout
 from sparse_rtdetr.protocol_r18.checkpoint import capture_rng
 with no_updates():
  for arm in ('A','S'):
   torch.manual_seed(37);model=FixtureModel.create(arm)
   raw={k:v.clone() for k,v in model.state_dict().items()}
   source={'format':FORMAT,'binding':expected_binding(arm,True),'runtime':str(torch.__version__),'device':'cpu','model':raw,'model_layout':_model_layout(model),'model_training':{n:True for n,m in model.named_modules()},'optimizer':{'not_loaded':True},'optimizer_layout':{'not_loaded':True},'ema':{'not_loaded':True},'engine':{'completed_updates':1,'phase':0,'failed':False},'rng':capture_rng(torch.device('cpu')),'criterion':{'not_loaded':True}}
   fault=spec.get('fault')
   if fault=='version':source['binding']['version']='unknown'
   if fault=='missing_tensor':source['model'].pop(next(iter(raw)))
   if fault=='dtype':k=next(iter(raw));source['model'][k]=source['model'][k].double()
   if fault=='mode':source['model_training']['']=False
   if fault=='nonfinite':k=next(iter(raw));source['model'][k].fill_(float('nan'))
   audit=bridge(model,source,arm,True)
   if fault=='forward_cap':meter.cap={**CAPS,'forward':0}
   images=torch.linspace(-1,1,8*256*7*7).reshape(8,256,7,7);targets=[{'boxes':torch.ones(1,4)} for _ in range(8)]
   reports.append({'bridge':audit,'forward':one(model,sc,source,images,targets,meter,arm,synthetic=True)})
   del model,source;gc.collect()
 need(not torch.cuda.is_initialized(),'CPU initialized CUDA');need(meter.count==2,'fixture counts')
 result={'status':'SYNTHETIC_PASS_NOT_SCIENTIFIC','reports':reports,'counts':meter.snapshot(),'CUDA_initialized':False};write(out/'RESULT.json',result);write(out/'CLOSED.json',{'result_sha256':sha(out/'RESULT.json')});print(json.dumps({'status':result['status'],'counts':result['counts']}))

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('prepare');q.add_argument('--out',required=True)
 q=s.add_parser('run');q.add_argument('--definition',required=True);q.add_argument('--manifest',required=True);q.add_argument('--authorization',required=True);q.add_argument('--out',required=True)
 q=s.add_parser('fixture');q.add_argument('--spec',required=True);q.add_argument('--out',required=True)
 a=p.parse_args();return {'prepare':prepare,'run':run,'fixture':fixture}[a.command](a)
if __name__=='__main__':raise SystemExit(main())
