"""Actual CLI CPU acceptance; synthetic data only, bounded output and temporary states."""
import copy,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import sabl_diagnostic as d
formal,state,data,control=d.load_science(ROOT)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
checks=[];consumed={'baseline_updates':0,'baseline_forward':0,'CLI_forward':0,'CLI_backward':0,'CLI_updates':0,'head_only_forward':0,'head_only_backward':0}
def ok(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def run(args,expected=0):
 r=subprocess.run([sys.executable,'-B',str(ROOT/'tools/sabl_diagnostic.py'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
 ok('CLI bounded output',len(r.stdout.encode())+len(r.stderr.encode())<32768)
 if r.returncode!=expected:raise AssertionError('CLI return: '+r.stderr[-2500:])
 return r
started=time.monotonic()
with tempfile.TemporaryDirectory(prefix='pr26-diagnostic-cpu-') as temp:
 base=Path(temp);contract={'mode':formal.ct.CPU,'manifest_sha256':'CPU_SYNTHETIC_NOT_A_RELEASE'}
 e,binding,_,_=formal.fresh(contract,'A','cpu');ref=state.save(e,binding,base/'fixture-initial.pt')
 payload=Path(ref['path']).read_bytes();sidecar=Path(ref['path']+'.complete.json').read_bytes();ds=data.Fixture();expected=[]
 for i in range(2):
  images,targets,identity=ds.window(i+1);r=e.train_window(images,targets,capture=False);expected.append(d.expected_row({'update':e.completed_updates,'input':identity,'result':r}));consumed['baseline_updates']+=1;consumed['baseline_forward']+=2
 del e;__import__('gc').collect()
 cases=[('success',None,'NOT_REPRODUCED',0),('original-zero','target_zero','REPRODUCED_ORIGINAL_CHECK_FAILURE',0),('input-mismatch','input_mismatch','STOP_TRAJECTORY_MISMATCH',1),('result-mismatch','result_mismatch','STOP_TRAJECTORY_MISMATCH',1),('forward-cap','forward_cap','STOP_DIAGNOSTIC_EXCEPTION',1)]
 for name,fault,status,rc in cases:
  cp=base/'fixture-initial.pt'
  if not cp.exists():cp.write_bytes(payload);Path(str(cp)+'.complete.json').write_bytes(sidecar)
  rows=copy.deepcopy(expected)
  if fault=='input_mismatch':rows[0]['input']='0'*64
  if fault=='result_mismatch':rows[0]['result']='0'*64
  fixture={'contract':contract,'checkpoint':ref,'expected':rows,'fault':fault}
  fp=base/(name+'.json');fp.write_text(json.dumps(fixture));out=base/name
  run(['fixture','--source-root',str(ROOT),'--fixture',str(fp),'--out',str(out)],rc)
  result=d.read(out/'RESULT.json');ok(name+' classification',result['status']==status)
  for k in ('forward','backward','updates'):consumed['CLI_'+k]+=result['counts'][k]
  if name in ('success','original-zero'):
   ok(name+' exact replay',result['matched_replay_updates']==2 and result['counts']=={'forward':5,'backward':4,'updates':2})
   ok(name+' boundary exists',Path(result['saved_boundary']['path']).is_file())
   loaded=torch.load(result['saved_boundary']['path'],map_location='cpu',weights_only=True);ok(name+' full boundary clock',loaded['engine']['completed_updates']==loaded['ema']['updates']==2);del loaded
   obs=d.read(out/'OBSERVATIONS.json');ok(name+' observation dtype',all(x['dtype']=='torch.float32' for x in obs['rows']))
   ok(name+' denominator observation',any(x['name']=='dx' for x in obs['rows']))
  if name=='original-zero':ok('zero localization',result['first_anomaly']['name']=='dx' and result['first_anomaly']['kind']=='nonpositive_denominator')
  if name=='input-mismatch':ok('input mismatch before forward',result['counts']=={'forward':0,'backward':0,'updates':0})
  if name=='result-mismatch':ok('result mismatch charged',result['counts']=={'forward':2,'backward':2,'updates':1})
  before=d.sha(out/'RESULT.json');run(['fixture','--source-root',str(ROOT),'--fixture',str(fp),'--out',str(out)],1);ok('duplicate output rejected '+name,d.sha(out/'RESULT.json')==before)
  shutil.rmtree(out)
 # Head computation with and without hooks/line tracing: exact values, gradients and RNG.
 torch.manual_seed(3);head=control.BoundaryHead();roi=torch.randn(2,256,7,7,requires_grad=True)
 z,v=head(roi,'A');(z.sum()+v.sum()).backward();gold=(z.detach().clone(),v.detach().clone(),roi.grad.clone(),[p.grad.clone() for p in head.parameters()]);consumed['head_only_forward']+=1;consumed['head_only_backward']+=1
 head.zero_grad(set_to_none=True);roi.grad=None
 class Wrapped:
  boundary=head
  encoder=torch.nn.Identity()
 class Meter:
  n=0
  def guard(self):pass
  def tick(self,key):self.n+=1
 meter=Meter();obs=d.Observation(Wrapped(),meter,control);obs.active=True;before=torch.get_rng_state().clone();sys.settrace(obs.trace)
 try:z,v=head(roi,'A');(z.sum()+v.sum()).backward()
 finally:sys.settrace(None);obs.close()
 consumed['head_only_forward']+=1;consumed['head_only_backward']+=1
 ok('observed exact output',torch.equal(z,gold[0]) and torch.equal(v,gold[1]));ok('observed exact gradients',torch.equal(roi.grad,gold[2]) and all(torch.equal(p.grad,x) for p,x in zip(head.parameters(),gold[3])));ok('observer RNG unchanged',torch.equal(before,torch.get_rng_state()));ok('backward counted once',meter.n==1)
 ok('loss and gradient part of frozen comparison',all(k in d.RESULT_KEYS for k in ('microbatch_losses','loss','gradient_norm_before_clip','normalizers')))
 ok('no epsilon in diagnostic',not any(x in (ROOT/'tools/sabl_diagnostic.py').read_text() for x in ('clamp_min','nan_to_num','softmax(')))
 # Independent hard-limit refusal without executing numerical work.
 limit_dir=base/'limit-tests';limit_dir.mkdir()
 small={'forward':1,'backward':0,'updates':0,'GPU_seconds':10,'wall_seconds':10,'allocated_bytes':18<<30,'storage_bytes':512<<20}
 meter=d.Meter(limit_dir,small,False,limit_dir)
 for key in ('backward','updates'):
  try:meter.tick(key)
  except ValueError:checks.append(key+' hard cap refuses')
  else:raise AssertionError(key+' cap missing')
 meter.start-=20
 try:meter.guard()
 except ValueError:checks.append('elapsed wall cap refuses')
 else:raise AssertionError('time cap missing')
 meter.start=time.monotonic();meter.caps={**small,'storage_bytes':1}
 try:meter.guard()
 except ValueError:checks.append('storage cap refuses')
 else:raise AssertionError('storage cap missing')
 meter.caps=small
 class WrappedNan:
  boundary=control.BoundaryHead()
  encoder=torch.nn.Identity()
 obs=d.Observation(WrappedNan(),meter,control);obs.record('constructed-nonfinite',torch.tensor([[float('nan'),0.]]));obs.close()
 ok('nonfinite distinguished from zero',obs.first['kind']=='nonfinite' and obs.rows[0]['zero']==1)
 before=base/'bad-source-out'
 run(['prepare','--root',str(base),'--out',str(before)],1);ok('bad frozen identity before directory',not before.exists())
 ok('CUDA not initialized',not torch.cuda.is_initialized())
result={'status':'PASS_CPU_SYNTHETIC_ONLY','checks':checks,'count':len(checks),'consumed':consumed,'elapsed_seconds':time.monotonic()-started,'CUDA_initialized':torch.cuda.is_initialized(),'real_experiment_cost':0}
path=Path(sys.argv[1]);d.write(path,result);print(json.dumps({'status':result['status'],'checks':len(checks),'seconds':result['elapsed_seconds'],'consumed':consumed}))
