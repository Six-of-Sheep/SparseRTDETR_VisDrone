"""Actual corrected full models/engine on synthetic CPU only; no real data or GPU."""
import os,sys,time,json,resource,signal,gc,copy,subprocess,importlib.util
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_DATA,(4<<30,4<<30));signal.alarm(300)
# The three actual-model checks use separate processes so CPU allocator caches
# from a prior model cannot consume the next check's unchanged 4 GiB AS cap.
if len(sys.argv)==2:
 root=Path(sys.argv[1]);root.mkdir();begin=time.monotonic();parts={};peak_rss_tree=0
 def rss_tree(pid):
  total=0
  try:
   status=Path('/proc/%d/status'%pid).read_text();total=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))*1024
   childfile=Path('/proc/%d/task/%d/children'%(pid,pid));children=childfile.read_text().split()
  except (OSError,StopIteration,ValueError):return total
  return total+sum(rss_tree(int(child)) for child in children)
 for arm in ('N','S','A'):
  proc=subprocess.Popen([sys.executable,'-B',__file__,str(root/arm),arm],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  while proc.poll() is None:
   rss=rss_tree(os.getpid());peak_rss_tree=max(peak_rss_tree,rss)
   if rss>(4<<30) or time.monotonic()-begin>295:
    os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate();(root/(arm+'.stdout')).write_bytes(stdout);(root/(arm+'.stderr')).write_bytes(stderr);(root/'RESOURCE_STOP.json').write_text(json.dumps({'RSS_tree_bytes':rss,'wall':time.monotonic()-begin,'CPU_child_group':proc.pid}));raise RuntimeError('CPU resident-memory/wall budget')
   time.sleep(.05)
  stdout,stderr=proc.communicate()
  (root/(arm+'.stdout')).write_bytes(stdout);(root/(arm+'.stderr')).write_bytes(stderr)
  if proc.returncode:
   (root/'FAILED_PHASE.json').write_text(json.dumps({'arm':arm,'returncode':proc.returncode,'stderr_tail':stderr.decode()[-5000:]}));sys.stderr.write(stderr.decode());raise SystemExit(proc.returncode)
  parts[arm]=json.loads((root/arm/'RESULT.json').read_text())
 assert parts['S']['init']['S']==parts['A']['init']['A'],'S/A independent initial state mismatch'
 result={'status':'PASS_CPU_FULL_CORRECTED_MODEL_PATH','version':'r18-sabl-control-lrtb-logsigmoid-v3','checks':sum(x['checks'] for x in parts.values())+1,'CPU_detector_forward':sum(x['CPU_detector_forward'] for x in parts.values()),'CPU_actual_optimizer_updates':sum(x['CPU_actual_optimizer_updates'] for x in parts.values()),'CPU_extra_autodiff_calls':sum(x.get('CPU_extra_autodiff_calls',0) for x in parts.values()),'same_S_A_native_and_head_init':True,'CPU_synthetic_not_experiment_cost':True,'CPU_wall_seconds':time.monotonic()-begin,'max_child_RSS_KiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'max_parent_and_children_RSS_bytes':peak_rss_tree,'RSS_tree_limit_bytes':4<<30,'DATA_limit_bytes':4<<30,'virtual_address_space_is_not_RAM':True,'GPU_forward':0,'parts':{arm:str(root/arm/'RESULT.json') for arm in parts},'scientific_benefit':'UNKNOWN'}
 (root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0)

import torch
from sparse_rtdetr.research import sabl_control as sc,sabl_formal_state as formal,sabl_state as short
from sparse_rtdetr.protocol_r18.engine import tensor_digest,output_digest
from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
from sparse_rtdetr.protocol_r18.checkpoint import file_sha
from sparse_rtdetr.protocol_r18.model import Config
def edges_from_GT(g):
 cx,cy,w,h=g.unbind(-1)
 return torch.stack((cx-w/2,cx+w/2,cy-h/2,cy+h/2),-1)

def main(path):
 out=Path(path);out.mkdir();phase=sys.argv[2];start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);checks=[];forwards=0;updates=0;arms={}
 def check(name,ok):
  checks.append({'name':name,'pass':bool(ok)});(out/'CHECKS.json').write_text(json.dumps(checks))
  if not ok:raise AssertionError(name)
 if phase=='N':
  # Old/new package source checks each run in their own complete checkout.
  import io,tarfile
  declared=os.environ.get('SABL_ORIGINAL_REFERENCE_ROOT')
  if declared:
   legacy=Path(declared).resolve();check('explicit frozen original manifest identity',file_sha(legacy/'RELEASE_MANIFEST.json')=='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab')
  else:
   legacy=out/'original-v1-release';legacy.mkdir()
   archive=subprocess.check_output(['git','archive','--format=tar','79ba59371c21ecfd4fa76885c413941403797214'],cwd=ROOT)
   with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tf:
    for member in tf.getmembers():
     name=Path(member.name);assert not name.is_absolute() and '..' not in name.parts
     if member.isdir():continue
     assert member.isfile();dest=legacy/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tf.extractfile(member).read())
  check('legacy source SHA',file_sha(legacy/'src/sparse_rtdetr/research/sabl_control.py')=='c5945edd78ba7159ce83f433a1cc9a266543675ba4f3edac478d059d7875f05b')
  code="import sys,json\nfrom pathlib import Path\nsys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd()/'tools')]\nimport torch\nfrom sparse_rtdetr.research import sabl_control as sc\nfrom sparse_rtdetr.protocol_r18.model import Config\nfrom sparse_rtdetr.protocol_r18.engine import output_digest,tensor_digest\ntorch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)\nm=sc.build('N',Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32'))[0].eval();images=torch.rand((2,3,128,128),generator=torch.Generator().manual_seed(311))\nwith torch.no_grad():out=m(images)\nprint(json.dumps({'version':sc.VERSION,'outputs_SHA':output_digest(out),'state_SHA':tensor_digest(sorted(m.state_dict().items())),'CUDA_initialized':torch.cuda.is_initialized()}))\n\n"
  child=subprocess.run([sys.executable,'-B','-c',code],cwd=legacy,capture_output=True,timeout=300);(out/'legacy-N.stdout').write_bytes(child.stdout);(out/'legacy-N.stderr').write_bytes(child.stderr);check('legacy native CPU process passed its own source checks',child.returncode==0)
  original=json.loads(child.stdout);cfg=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32');n2=sc.build('N',cfg)[0].eval();before=tensor_digest(sorted(n2.state_dict().items()));images=torch.rand((2,3,128,128),generator=torch.Generator().manual_seed(311))
  with torch.no_grad():b=n2(images)
  forwards+=2;check('N bypass original/new complete outputs bitwise digest',output_digest(b)==original['outputs_SHA']);check('N complete parameter/BN/cache state identical',before==original['state_SHA']);check('N eval buffers/params unchanged',before==tensor_digest(sorted(n2.state_dict().items())));check('N has no boundary head',not hasattr(n2,'boundary'));del b,n2;gc.collect()
  result={'status':'PASS_CPU_NATIVE_N_EQUIVALENCE','checks':len(checks),'CPU_detector_forward':2,'CPU_actual_optimizer_updates':0,'CPU_extra_autodiff_calls':0,'GPU_forward':0};(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
 images=torch.rand((2,3,128,128),generator=torch.Generator().manual_seed(311))
 init={};observed=[]
 targets=[]
 for i in range(2):
  targets.append({'boxes':torch.tensor([[.35,.45,.18,.12],[.66,.65,.12,.2]]),'labels':torch.tensor([1,2]),'gt_ids':['SYNTHETIC-%d-A'%i,'SYNTHETIC-%d-B'%i],'raw_area':torch.tensor([354.,393.]),'ignore_spatial_boxes':torch.empty(0,4),'ignore_union':torch.empty(0,4),'ignore_class_boxes':torch.empty(0,4),'ignore_class_labels':torch.empty(0,dtype=torch.long),'ignore_ids':[]})
 for arm in (phase,):
  e=sc.make_engine(arm,True);init[arm]={'native':tensor_digest(sorted(e.model.native.state_dict().items())),'head':tensor_digest(sorted(e.model.boundary.state_dict().items()))};check(arm+' new optimizer/EMA/RNG clock',not e.optimizer.state and e.completed_updates==0 and e.ema.updates==0)
  seen=[];grad_records=[];head_grad=[]
  def observe_feature(m,a,value):
   seen.append(value[0]);value[0].register_hook(lambda g:grad_records.append({'F8_total_objective_nonzero':bool(torch.isfinite(g).all() and (g!=0).any())}))
  capture=e.model.encoder.register_forward_hook(observe_feature)
  head_hook=e.model.boundary.offset.weight.register_hook(lambda g:head_grad.append(bool(torch.isfinite(g).all() and (g!=0).any())))
  real_encode=sc.encode;real_criterion=e.criterion.forward;encode_records=[]
  def witness_encode(reference,gt):
   answer=real_encode(reference,gt);x,s=sc.buckets(reference);expected=(x-edges_from_GT(gt.detach())[...,None])/s[...,None]
   check(arm+' actual training correct GT side targets call%d'%len(encode_records),torch.equal(answer[0],expected));encode_records.append({'matches':len(gt),'target_LRTB_first':edges_from_GT(gt.detach()).tolist()[0] if len(gt) else None,'correct':True});return answer
  def witness_criterion(outputs,target):
   losses=real_criterion(outputs,target)
   check(arm+' actual coarse reference detached',not outputs['sabl']['reference'].requires_grad)
   return losses
  sc.encode=witness_encode;e.criterion.forward=witness_criterion;torch.manual_seed(713)
  try:r=e.train_window(images,targets,capture=True)
  finally:sc.encode=real_encode;e.criterion.forward=real_criterion;capture.remove();head_hook.remove()
  forwards+=2;updates+=1;check(arm+' two microbatches used actual corrected encode',len(encode_records)==2 and len(grad_records)==2 and all(v['F8_total_objective_nonzero'] for v in grad_records) and len(head_grad)==2 and all(head_grad));check(arm+' native losses and ordinary-only extra',all(len(v)==23 and 'loss_bucket' in v and 'loss_offset' in v for v in r['microbatch_losses']) and all(len([d for d in ds if d['head']=='SABL_ordinary_only'])==1 for ds in r['normalizers']))
  binding={'scope':'CPU_SYNTHETIC_SABL_FORMAL','arm':arm,'version':sc.VERSION,'contract':'SYNTHETIC_CORRECTION_V3','release_manifest':'SYNTHETIC_NOT_SCIENTIFIC','N_SHA':sc.N_SHA};cp=formal.save(e,binding,out/(arm+'-u1.pt'));expected=copy.deepcopy(formal.capture(e,binding))
  with torch.no_grad():
   next(e.model.parameters()).view(-1)[0]+=.01;next(e.ema.module.parameters()).view(-1)[0]+=.02
   first=next(iter(e.optimizer.state.values()));first['exp_avg'].view(-1)[0]+=.03
  torch.rand(3);restored=formal.restore(e,binding,cp['path'],cp['sha256']);check(arm+' full model/optimizer/EMA/RNG restored exactly',restored['full_state_exact'] and _equal(expected,formal.capture(e,binding)));del expected;gc.collect()
  e.model.eval();digest_before=tensor_digest(sorted(e.model.state_dict().items()))
  with torch.no_grad():pred=e.model(images[:1])
  forwards+=1;check(arm+' corrected eval branch shape and finite',pred['pred_boxes'].shape==(1,300,4) and pred['refined_valid'].shape==(1,300) and bool(torch.isfinite(pred['pred_boxes']).all()));check(arm+' eval state unchanged',digest_before==tensor_digest(sorted(e.model.state_dict().items())));del pred
  arms[arm]={'actual_encode_calls':encode_records,'gradient_evidence':{'F8_combined_objective':grad_records,'offset_head_boundary_only':head_grad,'no_extra_backward_added':True},'CPU_window_losses':r['microbatch_losses'],'new_checkpoint':cp,'model_parameters':sum(p.numel() for p in e.model.parameters()),'full_state_restore':True};del e;gc.collect()
 # Actual restore interfaces: even a caller supplying the legacy binding is
 # rejected before model health/load. Header fixtures are explicitly synthetic.
 refusals=[]
 for module,scope in ((formal,'CPU_SYNTHETIC_SABL_FORMAL'),(short,'CPU_SYNTHETIC_SABL')):
  for arm in (phase,):
   for version in ('r18-sabl-control-v1','r18-sabl-control-logsigmoid-v2'):
    binding={'scope':scope,'arm':arm,'version':version,'contract':'SYNTHETIC_OLD_VERSION'};state={'format':module.FORMAT,'binding':binding,'runtime':str(torch.__version__),'device':'cpu'};p=out/(arm+'-'+('formal' if module is formal else 'short')+'-'+('v1' if version.endswith('v1') else 'v2')+'-HEADER_ONLY.pt');torch.save(state,p)
    engine=SimpleNamespace(model=SimpleNamespace(arm=arm),device=torch.device('cpu'),failed=False,phase=0)
    try:module.restore(engine,binding,p,file_sha(p))
    except ValueError as error:
     check(arm+' legacy '+version+' '+scope+' refuses before load',str(error)=='arm/version mismatch' and engine.failed);refusals.append({'arm':arm,'scope':scope,'old_version':version,'reason':str(error),'synthetic_header_only':True,'engine_sealed':True})
    else:raise AssertionError('legacy state accepted')
 check('CPU no CUDA',not torch.cuda.is_initialized());result={'status':'PASS_CPU_FULL_CORRECTED_MODEL_PATH','version':sc.VERSION,'checks':len(checks),'CPU_detector_forward':forwards,'CPU_actual_optimizer_updates':updates,'CPU_extra_autodiff_calls':0,'CPU_synthetic_not_experiment_cost':True,'init':init,'arms':arms,'legacy_refusals':refusals,'CPU_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_forward':0,'real_images_or_real_detector_weights_loaded':False,'scientific_benefit':'UNKNOWN'};(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('init','arms','legacy_refusals')}))
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as error:
  target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True);(target/'FAILURE_RESOURCE.json').write_text(json.dumps({'error':repr(error),'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'proc_status':Path('/proc/self/status').read_text(),'CUDA_initialized':torch.cuda.is_initialized()}));raise
