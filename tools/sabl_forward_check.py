"""Read-only raw-model bridge and exactly two forward-only engineering probes.

This is not a training-state restore or an evaluation/benchmark launcher.
"""
import argparse,base64,contextlib,gc,hashlib,json,os,re,signal,subprocess,sys,time,traceback
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
TASK=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/r18-sabl-forward-validation-20261028-002')

def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def read(p):
 def unique(pairs):
  out={}
  for key,value in pairs:
   need(key not in out,'duplicate JSON field '+key);out[key]=value
  return out
 return json.loads(Path(p).read_text(),object_pairs_hook=unique)
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def normalized_ref(ref,allow_empty=False):
 need(isinstance(ref,dict) and set(ref)<={'path','sha256','bytes','size_bytes'} and {'path','sha256'}<=set(ref),'reference fields')
 sizes=[ref[k] for k in ('bytes','size_bytes') if k in ref]
 need(bool(sizes) and all(type(n) is int and n>=(0 if allow_empty else 1) for n in sizes) and len(set(sizes))==1,'reference size or conflict')
 need(isinstance(ref['path'],str) and isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'reference path/SHA types')
 return {'path':ref['path'],'bytes':sizes[0],'sha256':ref['sha256']}

def exact(ref,allow_empty=False):
 ref=normalized_ref(ref,allow_empty);p=Path(ref['path'])
 need(p.is_absolute() and '..' not in p.parts and p.is_file() and not p.is_symlink() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'bound file '+str(p));return p
def protected_records(data):
 need(isinstance(data,dict) and isinstance(data.get('protected_files'),dict),'protected manifest schema')
 rows=data['protected_files'];need(bool(rows),'empty protected manifest')
 for p,r in rows.items():
  need(isinstance(r,dict) and set(r)<={'bytes','size_bytes','sha256'},'protected record fields')
  normalized_ref({'path':p,**r})
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

def tree_sizes(root):
 root=Path(root);rows={}
 if root.exists():
  need(not root.is_symlink(),'cache root symlink '+str(root))
  for p in root.rglob('*'):
   try:
    need(not p.is_symlink(),'storage symlink '+str(p))
    if p.is_file():rows[str(p)]=p.stat().st_size
   except FileNotFoundError:continue  # compiler atomic rename; final audit is repeated after exit
 return rows

CACHE_VARIABLES={'TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton','TORCH_EXTENSIONS_DIR':'extensions','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}
def cache_environment(task):return {key:str(Path(task)/'compile-cache'/suffix) for key,suffix in CACHE_VARIABLES.items()}

class CacheAccounting:
 def __init__(self,task):
  import getpass
  self.task=Path(task);self.environment=cache_environment(task)
  need(not (self.task/'compile-cache').exists(),'compiler cache must start empty')
  for key,value in self.environment.items():
   need(os.environ.get(key) in (None,value),'unbound compiler cache '+key)
   os.environ[key]=value
  self.external=[Path('/tmp')/('torchinductor_'+getpass.getuser()),Path.home()/'.triton/cache',Path.home()/'.nv/ComputeCache',Path.home()/'.cache/torch/inductor',Path.home()/'.cache/triton']
  self.before={str(p):tree_sizes(p) for p in self.external}
 def snapshot(self):
  rows=[];growth=0
  for p in self.external:
   before=self.before[str(p)];after=tree_sizes(p);delta=sum(max(0,n-before.get(k,0)) for k,n in after.items());growth+=delta
   rows.append({'path':str(p),'before_bytes':sum(before.values()),'after_bytes':sum(after.values()),'positive_growth_bytes':delta})
  return {'environment':self.environment,'bound_cache_bytes':sum(tree_sizes(self.task/'compile-cache').values()),'external_positive_growth_bytes':growth,'external_roots':rows,'scope':'task-bound Inductor/Triton/extensions/CUDA/XDG caches plus explicit default cache growth; not whole-filesystem allocation'}

class Meter:
 def __init__(self,out,storage,device,cap=None,cache=None):
  self.out=out;self.storage=storage;self.device=device;self.cap=CAPS if cap is None else cap;self.start=time.monotonic();self.gpu_start=None;self.count=0;self.peak=0;self.peak_available=(device!='cuda');self.resource_errors=[];self.cache=cache;self.storage_snapshot=None
 def sample_resources(self):
  if self.device=='cuda':
   try:
    import torch
    self.peak=max(self.peak,torch.cuda.max_memory_allocated());self.peak_available=True
   except BaseException as e:self.peak_available=False;self.resource_errors.append('allocator peak: '+repr(e))
  try:
   measured=sum(tree_sizes(self.storage).values());c=self.cache.snapshot() if self.cache is not None else None
   self.storage_snapshot={'task_bytes':measured,'development_reserve_bytes':32<<20,'compiler_cache':c,'charged_storage_bytes':measured+(32<<20)+(0 if c is None else c['external_positive_growth_bytes'])}
  except BaseException as e:self.storage_snapshot=None;self.resource_errors.append('storage/cache: '+repr(e))
 def guard(self):
  self.sample_resources()
  need(time.monotonic()-self.start<=self.cap['wall_seconds'],'wall limit')
  if self.gpu_start is not None:need(time.monotonic()-self.gpu_start<=self.cap['GPU_seconds'],'GPU phase limit')
  need(self.peak_available,'allocated memory UNKNOWN')
  need(self.peak<=self.cap['allocated_bytes'],'allocated memory limit')
  need(self.storage_snapshot is not None,'storage/cache UNKNOWN')
  need(self.storage_snapshot['charged_storage_bytes']<=self.cap['storage_bytes'],'storage limit including compiler cache and development reserve')
 def forward(self,module,args):
  self.guard();need(self.count<self.cap['forward'],'forward limit');self.count+=1;self.publish()
 def publish(self):
  p=self.out/'ledger.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(self.snapshot()));os.replace(t,p)
 def snapshot(self):return {'detector_forward_reserved':self.count,'training_target_backward':0,'backward':0,'optimizer_updates':0,'EMA_updates':0,'wall_seconds':time.monotonic()-self.start,'GPU_seconds':0 if self.gpu_start is None else time.monotonic()-self.gpu_start,'peak_allocated_bytes':self.peak if self.peak_available else 'UNKNOWN','last_known_peak_allocated_bytes':self.peak,'final_resource_sampling_errors':self.resource_errors,'storage':self.storage_snapshot if self.storage_snapshot is not None else 'UNKNOWN'}

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

def model_audit(model,param_before,buffers,gradient_before):
 import torch
 from sparse_rtdetr.protocol_r18.engine import tensor_digest
 grads={k:('NONE' if v.grad is None else {'shape':list(v.grad.shape),'dtype':str(v.grad.dtype),'sha256':tensor_sha(v.grad),'finite':bool(torch.isfinite(v.grad).all())}) for k,v in model.named_parameters()}
 changed=[k for k,v in model.named_buffers() if not equal_tensor(v,buffers[k])]
 allowed={n+'.'+k if n else k for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for k in ('running_mean','running_var','num_batches_tracked')}
 details=[]
 for k,v in model.named_buffers():
  if k in changed:
   now=v.detach().cpu();before=buffers[k];delta=(now.double()-before.double()).abs()
   maxabs=float(delta.max());details.append({'name':k,'dtype':str(v.dtype),'before_sha256':tensor_sha(before),'after_sha256':tensor_sha(now),'maxabs':maxabs if bool(torch.isfinite(delta).all()) else 'NONFINITE','changed_elements':int((now!=before).sum())})
 param_after=tensor_digest(model.named_parameters())
 return {'parameter_SHA_before':param_before,'parameter_SHA_after':param_after,'parameters_unchanged':param_after==param_before,'gradient_state_before':gradient_before,'gradient_state_after':grads,'all_parameter_gradients_NONE':all(v=='NONE' for v in grads.values()),'BN_mutated_buffers':changed,'BN_change_details':details,'only_BN_buffers_changed':set(changed)<=allowed,'BN_changes_in_memory_only':True,'state_not_saved':True}

def one(model,sc,state,images,targets,meter,arm,saved=None,synthetic=False):
 import torch
 from sparse_rtdetr.protocol_r18.checkpoint import restore_rng
 from sparse_rtdetr.baseline.training_v2b_engine import _batchnorm_backward_layout,_move
 from sparse_rtdetr.protocol_r18.model import dn_capture
 from sparse_rtdetr.protocol_r18.geometry import forward_guards
 from sparse_rtdetr.protocol_r18.engine import tensor_digest
 device=torch.device(meter.device)
 param_before=tensor_digest(model.named_parameters());buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()};gradient_before={k:('NONE' if v.grad is None else 'PRESENT') for k,v in model.named_parameters()}
 need(all(v=='NONE' for v in gradient_before.values()),'initial parameter gradients must be NONE')
 restore_rng(state['rng'],device)
 before_rng=tensor_sha(torch.get_rng_state());model.train()
 observation=Observe(model,sc,meter,arm,saved);h=model.register_forward_pre_hook(meter.forward)
 try:
  with torch.enable_grad(),contextlib.nullcontext(),_batchnorm_backward_layout(model,device) if not synthetic else contextlib.nullcontext():
   batch=images[:8].to(device);subset=_move(targets[:8],device)
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if not synthetic else contextlib.nullcontext()
   with ctx,dn_capture(model) if not synthetic else contextlib.nullcontext(),forward_guards(model) if not synthetic else contextlib.nullcontext():output=model(batch,targets=subset)
   need(model.last_dtype==('torch.float32' if synthetic else 'torch.bfloat16') and model.last_boundary_dtype=='torch.float32','actual native/local precision')
   rows=finite_tree(output);need('sabl' in output,'training forward SABL output')
   if not synthetic:need('dn_aux_outputs' in output and len(output['aux_outputs'])==2,'unchanged DN/aux output path')
   need(all(p.grad is None for p in model.parameters()),'gradient unexpectedly populated')
   del output,batch,subset
  audit=model_audit(model,param_before,buffers,gradient_before)
  need(audit['parameters_unchanged'],'parameters mutated');need(audit['all_parameter_gradients_NONE'],'parameter gradients populated');need(audit['only_BN_buffers_changed'],'unexpected non-BN buffer mutation')
  meter.guard()
  return {'arm':arm,'raw_bridge':True,'training_mode':True,'grad_enabled_forward':True,'evidence_matches':observation.matches,'local':observation.local,'output_tensors':rows,**audit,'CPU_rng_before_SHA':before_rng,'native_dtype':model.last_dtype,'boundary_dtype':model.last_boundary_dtype}

 except BaseException:
  meter.sample_resources()
  try:audit=model_audit(model,param_before,buffers,gradient_before)
  except BaseException as audit_error:audit={'status':'UNKNOWN','error':repr(audit_error)}
  try:write(meter.out/(arm+'_FAILED_OBSERVATION.json'),{'matches':observation.matches,'local':observation.local,'model_audit':audit,'resources_at_exception':meter.snapshot()})
  except BaseException as receipt_error:meter.resource_errors.append('failed observation receipt: '+repr(receipt_error))
  raise
 finally:h.remove();observation.close()

def verify_release(path):
 m=read(path);need(m['base_commit']==BASE and m['scope']=='SABL_TWO_FORWARD_ONLY_V1' and re.fullmatch('[0-9a-f]{40}',m['commit']) is not None,'source manifest identity')
 need(isinstance(m['files'],list) and len(m['files'])>0,'source files')
 seen=set()
 for r in m['files']:
  name=r['path'];p=Path(name)
  need(isinstance(name,str) and not p.is_absolute() and '..' not in p.parts and name not in seen,'unsafe or duplicate manifest path');seen.add(name)
  exact({'path':str(ROOT/p),**{k:r[k] for k in ('bytes','sha256')}},allow_empty=True)
 return m

def stop_receipt(out,meter,error,tb,phase='forward'):
 secondary=[]
 try:meter.sample_resources()
 except BaseException as e:secondary.append('final sampling: '+repr(e));meter.peak_available=False;meter.storage_snapshot=None
 cost=meter.snapshot()
 try:meter.publish()
 except BaseException as e:secondary.append('ledger write: '+repr(e))
 try:(out/'stderr.txt').write_text(tb)
 except BaseException as e:secondary.append('stderr write: '+repr(e))
 result={'status':'STOP_PREPARE_EXCEPTION' if phase=='prepare' else 'STOP_FORWARD_EXCEPTION','phase':phase,'error':repr(error),'cost':cost,'secondary_errors':secondary,'no_retry':True}
 try:write(out/'STOP.json',result)
 except BaseException as e:result['secondary_errors'].append('STOP write: '+repr(e))
 print(json.dumps({'status':result['status'],'error':repr(error)[:1600],'cost':cost,'secondary_errors':result['secondary_errors']}))
 return 1

def expected_authorization(definition,manifest,definition_path,manifest_path,out):
 return {'scope':definition['schema'],'commit':manifest['commit'],'manifest_sha256':sha(manifest_path),'definition_sha256':sha(definition_path),'attempt':str(out),'authorize_once':True}

def admit(a,check_only=False):
 definition=read(a.definition);out=Path(a.out)
 need(definition['schema']=='SABL_TWO_FORWARD_ONLY_V1' and definition['caps']==CAPS and Path(definition['root'])==TASK and Path(definition['release']).resolve()==ROOT,'frozen definition/release')
 need(Path(sys.executable).resolve()==Path(definition['python']).resolve(),'interpreter')
 need(Path.cwd().resolve()==ROOT,'release working directory')
 need(out==TASK/'attempt-forward-001' and not out.exists() and not (TASK/'FORWARD_ONCE.json').exists(),'unique attempt/once target')
 manifest=verify_release(a.manifest)
 need(manifest['inputs_reference_sha256']==sha(TASK/'INPUT_REFS.json') and manifest['protection_index_sha256']==sha(TASK/'PROTECTED_CONVERGENCE.json') and manifest['caps']==CAPS,'manifest material/protection/budget binding')
 need(check_only or manifest.get('candidate_not_for_real_execution') is False,'CPU candidate cannot execute')
 need(definition['refs']==read(TASK/'INPUT_REFS.json') and definition['protected']==protected_records(read(TASK/'PROTECTED_CONVERGENCE.json')),'current material/protection bindings')
 need(definition['target']==read(definition['refs']['target']['path']),'target metadata binding')
 expected=expected_authorization(definition,manifest,a.definition,a.manifest,out)
 if not check_only:need(canonical(read(a.authorization))==canonical(expected),'execution authorization binding')
 else:need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','check-only hidden CUDA')
 for ref in definition['refs'].values():exact(ref)
 preserve(definition['protected'])
 # Bind compiler caches before importing detector dependencies. No directories are created here.
 for key,value in cache_environment(TASK).items():
  need(os.environ.get(key) in (None,value),'unbound compiler cache '+key);os.environ[key]=value
 contract=read(definition['refs']['contract']['path'])
 need(digest(contract)==OLD_CONTRACT and contract['manifest_sha256']==OLD_MANIFEST,'original contract identity')
 need(definition['python']==contract['metadata']['python'] and definition['GPU']==contract['gpu_uuid'] and definition['environment']==contract['metadata']['environment'],'original runtime metadata binding')
 # No training state restoration and no detector invocation in this admission stage.
 sc,torch=import_model();import torchvision
 env=definition['environment']
 need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'environment identity')
 prior=read(TASK.with_name('r18-sabl-forward-validation-20261028-001')/'COST_ADDITION.json')['cumulative']
 need(definition['prior']=={'forward':prior['forward'],'GPU_seconds':prior['GPU_stage_seconds']},'cost inheritance')
 need(not torch.cuda.is_initialized(),'admission must not initialize CUDA')
 return definition,manifest,expected


def prepare(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU prepare hidden CUDA')
 sc,torch=import_model();torch.set_num_threads(2)
 out=Path(a.out);need(out.parent==TASK and not out.exists(),'fresh prepare target');out.mkdir()
 refs=read(TASK/'INPUT_REFS.json')
 for ref in refs.values():exact(ref)
 need(refs['A']['sha256']==CP_SHA['A'] and refs['S']['sha256']==CP_SHA['S'],'fixed raw states')
 protected=protected_records(read(TASK/'PROTECTED_CONVERGENCE.json'));preserve(protected)
 previous=TASK.with_name('r18-sabl-forward-validation-20261028-001')
 need(read(previous/'RELEASE_STATUS.json')['status']=='RUNTIME_STOP_DO_NOT_REEXECUTE','retain prior runtime STOP')
 prior=read(previous/'COST_ADDITION.json')['cumulative'];need(prior=={'forward':31741,'GPU_stage_seconds':36528.27560652013},'prior cost includes failed forward')
 c=read(refs['contract']['path']);need(digest(c)==OLD_CONTRACT and c['manifest_sha256']==OLD_MANIFEST,'original contract')
 target=read(refs['target']['path']);need(target['epoch']==4 and target['window_zero_based']==26 and target['update']==939,'target identity')
 from sparse_rtdetr.protocol_r18.model import Config
 audits={}
 with contextlib.nullcontext():
  for arm in ('A','S'):
   s=torch.load(refs[arm]['path'],map_location='cpu',weights_only=True)
   model,_=sc.build(arm,Config());audits[arm]=bridge(model,s,arm);del model,s;gc.collect()
 need(not torch.cuda.is_initialized(),'CPU prepare initialized CUDA')
 definition={'schema':'SABL_TWO_FORWARD_ONLY_V1','refs':refs,'protected':protected,'target':target,'raw_bridge_CPU':audits,'caps':CAPS,'root':str(TASK),'release':str(ROOT),'python':c['metadata']['python'],'GPU':c['gpu_uuid'],'environment':c['metadata']['environment'],'prior':{'forward':prior['forward'],'GPU_seconds':prior['GPU_stage_seconds']},'no_retry':True}
 write(out/'DEFINITION.json',definition);print(json.dumps({'status':'PASS_CPU_PREPARE','definition_sha256':sha(out/'DEFINITION.json'),'raw_bridge_CPU':audits,'CUDA_initialized':False}))

def bind(a):
 # Same admission as run --check-only. Only future explicit GPU authorization may write.
 definition,manifest,expected=admit(a,True)
 need(a.authorize_two_forwards is True and a.approve_commit==manifest['commit'] and a.approve_manifest_sha256==sha(a.manifest),'explicit two-forward version approval')
 need(Path(a.authorization)==TASK/'FORWARD_AUTHORIZATION.json' and not Path(a.authorization).exists(),'fresh authorization target')
 if not a.check_only:
  need(manifest.get('candidate_not_for_real_execution') is False,'CPU candidate cannot bind')
  write(Path(a.authorization),expected)
 print(json.dumps({'status':'PASS_BIND_CHECK_ONLY_NO_WRITE' if a.check_only else 'BOUND_TWO_FORWARD_AUTHORIZATION','commit':manifest['commit'],'manifest_sha256':sha(a.manifest),'definition_sha256':sha(a.definition),'authorization_created':not a.check_only,'attempt_created':False,'GPU_started':False}))
 return 0


def run(a):
 started=time.monotonic()
 if not a.check_only:
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s wall hard deadline')));signal.setitimer(signal.ITIMER_REAL,600)
 try:return run_body(a,started)
 finally:
  if not a.check_only:signal.setitimer(signal.ITIMER_REAL,0)

def run_body(a,started):
 definition,m,auth=admit(a,a.check_only)
 if a.check_only:
  print(json.dumps({'status':'PASS_CPU_REAL_LAUNCH_ADMISSION_NOT_CUDA','definition_sha256':sha(a.definition),'manifest_sha256':sha(a.manifest),'module_origin':str(ROOT/'src/sparse_rtdetr/research/sabl_control.py'),'cwd':str(Path.cwd()),'python':sys.executable,'forward':0,'CUDA_initialized':False,'authorization_created':False,'attempt_created':False}));return 0
 out=Path(a.out)
 with (TASK/'FORWARD_ONCE.json').open('x') as f:json.dump(auth,f)
 out.mkdir();meter=Meter(out,TASK,'cpu');meter.start=started;result=None
 try:
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
  cache=CacheAccounting(TASK);write(out/'cache-before.index.json',cache.before)
  write(out/'CACHE_BASELINE.json',{'external':[{'path':p,'files':len(rows),'bytes':sum(rows.values())} for p,rows in cache.before.items()],'index_sha256':sha(out/'cache-before.index.json'),'environment':cache.environment})
  meter.cache=cache
  sc,torch=import_model();import torchvision
  env=definition['environment'];need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'environment identity')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
  gpu_started=time.monotonic();meter.gpu_start=gpu_started;meter.device='cuda';meter.peak_available=False
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('GPU/wall hard deadline')));signal.setitimer(signal.ITIMER_REAL,max(.001,min(300,600-(gpu_started-started))))
  need(torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'BF16 GPU')
  torch.cuda.set_per_process_memory_fraction(CAPS['allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
  torch.cuda.reset_peak_memory_stats();meter.device='cuda';meter.peak_available=False;meter.gpu_start=gpu_started
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
  with contextlib.nullcontext():
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
  return stop_receipt(out,meter,e,traceback.format_exc())
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

def fixture_body(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixture hidden CUDA')
 sc,torch=import_model();torch.set_num_threads(2);spec=read(a.spec)
 need(spec['scope']=='SYNTHETIC_NOT_SCIENTIFIC','fixture scope')
 out=Path(a.out);meter=a.meter;reports=[]
 from sparse_rtdetr.baseline.training_v2b_checkpoint import _model_layout
 from sparse_rtdetr.protocol_r18.checkpoint import capture_rng
 with contextlib.nullcontext():
  for arm in ('A','S'):
   torch.manual_seed(37);model=FixtureModel.create(arm)
   raw={k:v.clone() for k,v in model.state_dict().items()}
   source={'format':FORMAT,'binding':expected_binding(arm,True),'runtime':str(torch.__version__),'device':'cpu','model':raw,'model_layout':_model_layout(model),'model_training':{n:True for n,m in model.named_modules()},'optimizer':{'not_loaded':True},'optimizer_layout':{'not_loaded':True},'ema':{'not_loaded':True},'engine':{'completed_updates':1,'phase':0,'failed':False},'rng':capture_rng(torch.device('cpu')),'criterion':{'not_loaded':True}}
   fault=spec.get('fault')
   if fault=='wall_cap':meter.cap={**CAPS,'wall_seconds':-1}
   if fault=='storage_cap':meter.cap={**CAPS,'storage_bytes':1}
   if fault=='version':source['binding']['version']='unknown'
   if fault=='missing_tensor':source['model'].pop(next(iter(raw)))
   if fault=='dtype':k=next(iter(raw));source['model'][k]=source['model'][k].double()
   if fault=='mode':source['model_training']['']=False
   if fault=='nonfinite':k=next(iter(raw));source['model'][k].fill_(float('nan'))
   audit=bridge(model,source,arm,True)
   if fault=='forward_cap':meter.cap={**CAPS,'forward':0}
   if fault=='populated_gradient':next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
   images=torch.linspace(-1,1,8*256*7*7).reshape(8,256,7,7);targets=[{'boxes':torch.ones(1,4)} for _ in range(8)]
   reports.append({'bridge':audit,'forward':one(model,sc,source,images,targets,meter,arm,synthetic=True)})
   del model,source;gc.collect()
 need(not torch.cuda.is_initialized(),'CPU initialized CUDA');need(meter.count==2,'fixture counts')
 result={'status':'SYNTHETIC_PASS_NOT_SCIENTIFIC','reports':reports,'counts':meter.snapshot(),'CUDA_initialized':False};write(out/'RESULT.json',result);write(out/'CLOSED.json',{'result_sha256':sha(out/'RESULT.json')});print(json.dumps({'status':result['status'],'counts':result['counts']}))

def fixture(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixture hidden CUDA')
 out=Path(a.out);need(not out.exists(),'fixture duplicate attempt');out.mkdir();a.meter=Meter(out,out,'cpu')
 try:return fixture_body(a)
 except BaseException as e:return stop_receipt(out,a.meter,e,traceback.format_exc())

EXPORT_NAMES=('INPUT.json','A.json','S.json','RESULT.json','CLOSED.json','STOP.json','stderr.txt','A_FAILED_OBSERVATION.json','S_FAILED_OBSERVATION.json','ledger.json','CACHE_BASELINE.json')
def export(a):
 root=Path(a.attempt);need(root.is_dir() and not root.is_symlink(),'export root')
 need((root/'STOP.json').exists() != (root/'CLOSED.json').exists(),'exactly one terminal marker')
 if (root/'CLOSED.json').exists():need(read(root/'CLOSED.json')['result_sha256']==sha(root/'RESULT.json'),'closed result SHA')
 need(type(a.already_returned_bytes) is int and 0<=a.already_returned_bytes<CAPS['return_bytes'],'prior return bytes')
 rows=[]
 for name in EXPORT_NAMES:
  p=root/name
  if p.exists():
   need(p.is_file() and not p.is_symlink(),'export regular file')
   size=p.stat().st_size;need(size<=CAPS['return_bytes'],'single return file too large')
   raw=p.read_bytes();rows.append({'path':name,'bytes':size,'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode('ascii')})
 payload=canonical({'status':'TERMINAL_WHITELIST_EXPORT','files':rows})+b'\n'
 need(a.already_returned_bytes+len(payload)+sum(r['bytes'] for r in rows)<=CAPS['return_bytes'],'total return cap including wire and attachment bytes')
 sys.stdout.buffer.write(payload);return 0

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('prepare');q.add_argument('--out',required=True)
 q=s.add_parser('run');q.add_argument('--definition',required=True);q.add_argument('--manifest',required=True);q.add_argument('--authorization',required=True);q.add_argument('--out',required=True);q.add_argument('--check-only',action='store_true')
 q=s.add_parser('bind');q.add_argument('--definition',required=True);q.add_argument('--manifest',required=True);q.add_argument('--authorization',required=True);q.add_argument('--out',required=True);q.add_argument('--approve-commit',required=True);q.add_argument('--approve-manifest-sha256',required=True);q.add_argument('--authorize-two-forwards',action='store_true');q.add_argument('--check-only',action='store_true')
 q=s.add_parser('fixture');q.add_argument('--spec',required=True);q.add_argument('--out',required=True)
 q=s.add_parser('export');q.add_argument('--attempt',required=True);q.add_argument('--already-returned-bytes',required=True,type=int)
 a=p.parse_args();target=Path(a.out) if hasattr(a,'out') else None;existed=target.exists() if target is not None else False
 try:return {'prepare':prepare,'bind':bind,'run':run,'fixture':fixture,'export':export}[a.command](a)
 except BaseException as e:
  tb=traceback.format_exc()
  if a.command=='prepare' and target is not None and not existed and target.is_dir():return stop_receipt(target,Meter(target,TASK,'cpu'),e,tb,phase='prepare')
  # Admission/duplicate/export rejection must not write into existing or unbound outputs.
  print(json.dumps({'status':'BLOCKED_'+a.command.upper(),'error':repr(e)[:1600],'output_existed_before':existed,'output_exists':target.exists() if target is not None else False,'once_exists':(TASK/'FORWARD_ONCE.json').exists(),'no_retry':True}))
  sys.stderr.write(tb);return 1
if __name__=='__main__':raise SystemExit(main())
