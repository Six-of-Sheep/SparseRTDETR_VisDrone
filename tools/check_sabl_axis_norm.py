"""CPU operators/local heads only. No detector, optimizer, image, AP or CUDA execution."""
import ast,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
stable=load(ROOT/'src/sparse_rtdetr/research/sabl_axis_norm.py','axis_norm').normalized_sigmoid_axis
CRITERIA=ROOT/'docs/research/r18_sabl_control/LOGSIGMOID_CRITERIA.json'
criteria=json.loads(CRITERIA.read_text());need_sha='5895115d1e591c25571d3146c0dd1939ae0ca9cc04b45ac7971d383cb578a6f1'
if hashlib.sha256(CRITERIA.read_bytes()).hexdigest()!=need_sha:raise ValueError('predeclared criteria changed')
OUT=Path(sys.argv[1]);need_parent=OUT.parent;need_parent.mkdir(parents=True,exist_ok=True)
if OUT.exists():raise FileExistsError('existing CPU evidence')
result={'status':'RUNNING','criteria_sha256':need_sha,'checks':[],'comparisons':[],'counterexamples':[],'optimizer_updates':0,'detector_forward':0,'GPU_forward':0,'local_head_forward':0,'operator_autodiff_calls':0,'CUDA_initialized':False}
started=time.monotonic()
def persist():OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def check(name,condition):
 result['checks'].append({'name':name,'pass':bool(condition)});persist()
 if not condition:raise AssertionError(name)
def compare(name,actual,reference,tolerance):
 a=actual.detach().double();r=reference.detach().double();diff=(a-r).abs();mask=r.abs()>=1e-5
 row={'name':name,'max_abs':float(diff.max()),'RMS':float(diff.square().mean().sqrt()),'max_relative_ref_abs_ge_1e_5':float((diff[mask]/r.abs()[mask]).max()) if bool(mask.any()) else None,'atol':tolerance['atol'],'rtol':tolerance['rtol'],'exact_equal':torch.equal(a,r),'pass':bool(torch.isfinite(a).all() and torch.isfinite(r).all() and torch.allclose(a,r,**tolerance))}
 result['comparisons'].append(row);persist()
 if not row['pass']:raise AssertionError(name)
def weights_math64(z,axis):
 moved=z.double().movedim(axis,-1);flat=moved.reshape(-1,moved.shape[-1]);rows=[]
 for values in flat.tolist():
  logs=[-math.log1p(math.exp(-x)) if x>=0 else x-math.log1p(math.exp(x)) for x in values];m=max(logs);p=[math.exp(x-m) for x in logs];den=math.fsum(p);rows.append([x/den for x in p])
 return torch.tensor(rows,dtype=torch.float64).reshape(moved.shape).movedim(-1,axis)
def evaluate(fn,z,x,axis,g):
 z=z.detach().clone().requires_grad_(True);x=x.detach().clone().requires_grad_(True);w=fn(z,axis);pooled=(x*w).sum(axis,keepdim=True);grads=torch.autograd.grad((pooled*g).sum(),(x,z));result['operator_autodiff_calls']+=1
 return w.detach(),pooled.detach(),grads[0].detach(),grads[1].detach()
def old(z,axis):
 w=z.sigmoid();return w/w.sum(axis,keepdim=True)
def oracle(z,x,axis,g):
 z=z.double();x=x.double();g=g.double();w=weights_math64(z,axis);pooled=(x*w).sum(axis,keepdim=True);dx=w*g;dz=w*torch.sigmoid(-z)*((x-pooled)*g).sum(1,keepdim=True)
 return w,pooled,dx,dz
def run_case(label,z,x,axis,normal=False):
 shape=list(x.shape);shape[axis]=1;gen=torch.Generator().manual_seed(801+axis);g=(torch.rand(shape,generator=gen,dtype=x.dtype)*2-1)/x.shape[1]
 new=evaluate(stable,z,x,axis,g);ref=oracle(z,x,axis,g)
 check(label+' finite weights/output/grad',all(bool(v.isfinite().all()) for v in new))
 check(label+' weights nonnegative',bool((new[0]>=0).all()))
 check(label+' original axis normalized',torch.allclose(new[0].sum(axis),torch.ones_like(new[0].sum(axis)),atol=criteria['normalization_axis_sum_FP32_atol'],rtol=0))
 for i,name in enumerate(('weights','output','gradient_features','gradient_logits')):
  tol=criteria['FP32_vs_FP64_saved_and_extreme_output'] if i<2 else criteria['FP32_vs_FP64_saved_and_extreme_gradients'];compare(label+'/FP32_vs_independent_FP64/'+name,new[i],ref[i],tol)
 double=evaluate(stable,z.double(),x.double(),axis,g.double())
 for i,name in enumerate(('weights','output','gradient_features','gradient_logits')):compare(label+'/FP64_autograd_vs_analytic/'+name,double[i],ref[i],criteria['FP64_reference_forward_and_analytic_gradient'])
 if bool((z.double().sigmoid().sum(axis)>0).all()):compare(label+'/direct_sigmoid_FP64_reference',old(z.double(),axis),ref[0],criteria['FP64_reference_forward_and_analytic_gradient'])
 else:result['counterexamples'].append({'case':label,'direct_FP64_sigmoid_denominator_positive':False,'interpretation':'FP64 direct sigmoid/sum is also not a universal extreme-input oracle; independent stable math FP64 is used'})
 if normal:
  legacy=evaluate(old,z,x,axis,g)
  for i,name in enumerate(('weights','output','gradient_features','gradient_logits')):compare(label+'/old_new_FP32/'+name,new[i],legacy[i],criteria['FP32_normal_old_new_weights_and_output'] if i<2 else criteria['FP32_normal_old_new_gradients'])
 return new,ref

def main():
 # All optimizer entry points used by the existing project are forbidden here.
 with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimizer step forbidden')),patch.object(torch.optim.SGD,'step',side_effect=AssertionError('optimizer step forbidden')):
  fixture=json.loads((ROOT/'tests/fixtures/sabl_axis_failure.json').read_text());fields={}
  check('authenticated saved local tensor provenance',fixture['source_tensor_sha256']==criteria['failure_fixture_sha256'])
  for name,row in fixture['fields'].items():
   t=torch.tensor(row['values'],dtype=torch.float32);check('portable fixture bytes '+name,list(t.shape)==row['shape'] and hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==row['float32_bytes_sha256']);fields[name]=t
  for axis,key in ((2,'lx'),(3,'ly')):
   run_case('saved_failure_axis%d'%axis,fields[key],fields['x'],axis)
  wx_old=old(fields['lx'],2);dx=fields['lx'].sigmoid().sum(2,keepdim=True)
  check('old selected zero denominator retained',float(dx[0,0,0,1])==0.)
  result['counterexamples'].append({'case':'saved_failure_old_formula','zero_denominator_count':int((dx==0).sum()),'unguarded_old_weights_finite':bool(wx_old.isfinite().all()),'production_original_stopped_before_division':True})
  for seed in range(4):
   gen=torch.Generator().manual_seed(seed);z=torch.rand((2,1,7,9),generator=gen)*24-12;x=torch.rand((2,4,7,9),generator=gen)*8-4
   for axis in (2,3):run_case('normal_seed%d_axis%d'%(seed,axis),z,x,axis,True)
  vectors=criteria['extreme_vectors']+[[torch.finfo(torch.float32).min]*7,[torch.finfo(torch.float32).max]*7,[torch.finfo(torch.float32).min,-1e20,-1000,0,1000,1e20,torch.finfo(torch.float32).max]]
  for i,vector in enumerate(vectors):
   for axis in (2,3):
    shape=(1,1,7,5) if axis==2 else (1,1,5,7);z=torch.tensor(vector,dtype=torch.float32).reshape(1,1,7,1) if axis==2 else torch.tensor(vector,dtype=torch.float32).reshape(1,1,1,7);z=z.expand(shape).clone();gen=torch.Generator().manual_seed(40+i);x=torch.rand((shape[0],3,shape[2],shape[3]),generator=gen)*8-4
    run_case('extreme%d_axis%d'%(i,axis),z,x,axis)
  z=torch.linspace(-6,6,7).reshape(1,1,7,1);x=torch.zeros(1,2,7,3);_,_,_,gz=evaluate(stable,z.expand(1,1,7,3),x,2,torch.ones(1,2,1,3)/2)
  check('constant zero features give exactly zero logit gradient',torch.count_nonzero(gz)==0)
  result['counterexamples'].append({'case':'constant_features','logit_gradient_max_abs':float(gz.abs().max()),'interpretation':'finite normalized weights do not imply a nonzero loss gradient or learned benefit'})
  z=torch.tensor([0.,1.,2.]).reshape(1,1,3,1);difference=float((stable(z,2)-torch.softmax(z,2)).abs().max());check('plain softmax logits is not equivalent',difference>.1);result['counterexamples'].append({'case':'softmax_of_logits','max_difference':difference,'interpretation':'the fix preserves normalized sigmoid, not plain softmax(logits)'})
  overflowing=F.conv2d(torch.full((1,1,1,1),torch.finfo(torch.float32).max),torch.full((1,1,1,1),2.))
  check('finite convolution inputs can still overflow',not bool(overflowing.isfinite().all()));result['counterexamples'].append({'case':'upstream_convolution','finite_inputs':True,'finite_output':False,'interpretation':'normalization repair does not ensure full-model numerical stability'})
  for value in (float('nan'),float('inf'),-float('inf')):
   try:stable(torch.full((1,1,7,7),value),2)
   except ValueError:check('nonfinite rejected '+str(value),True)
   else:check('nonfinite rejected '+str(value),False)
  for dtype in (torch.float16,torch.bfloat16,torch.int64):
   try:stable(torch.ones((1,1,7,7),dtype=dtype),2)
   except ValueError:check('unexpected dtype rejected '+str(dtype),True)
   else:check('unexpected dtype rejected '+str(dtype),False)
  # Extract the actual new Head/guard definitions without importing a detector.
  source=(ROOT/'src/sparse_rtdetr/research/sabl_control.py').read_text();tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))};impact=json.loads((ROOT/'tests/fixtures/sabl_axis_source_impact.json').read_text());finger=lambda n:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
  for name,h in impact['unchanged_AST'].items():check('unchanged source '+name,finger(nodes[name])==h)
  init=next(n for n in nodes['BoundaryHead'].body if isinstance(n,ast.FunctionDef) and n.name=='__init__');check('unchanged Head initialization',finger(init)==impact['BoundaryHead_init_AST'])
  check('new science version rejects silent v1 reuse',"VERSION = 'r18-sabl-control-logsigmoid-v2'" in source)
  namespace={'torch':torch,'nn':nn,'F':F,'normalized_sigmoid_axis':stable};selected=ast.Module(body=[nodes[k] for k in ('need','finite','BoundaryHead')],type_ignores=[]);exec(compile(selected,'<actual_new_BoundaryHead>','exec'),namespace)
  legacy=load(ROOT/'tests/fixtures/sabl_boundary_legacy.py','legacy_boundary')
  for arm in ('S','A'):
   with torch.random.fork_rng(devices=[]):torch.manual_seed(17);old_head=legacy.BoundaryHead()
   with torch.random.fork_rng(devices=[]):torch.manual_seed(17);new_head=namespace['BoundaryHead']()
   check(arm+' same parameters and initialization',list(old_head.state_dict())==list(new_head.state_dict()) and all(torch.equal(a,b) for a,b in zip(old_head.state_dict().values(),new_head.state_dict().values())))
   check(arm+' unchanged parameter count',sum(p.numel() for p in new_head.parameters())==1837572)
   gen=torch.Generator().manual_seed(5);roi=torch.rand((2,256,7,7),generator=gen)*2-1
   def head_values(h):
    r=roi.clone().requires_grad_(True);z,d=h(r,arm);g=torch.autograd.grad((z.sum()+d.sum())/z.numel(),(r,*h.parameters()));result['local_head_forward']+=1;return [z,d,*g]
   before=torch.get_rng_state().clone();a=head_values(old_head);b=head_values(new_head);check(arm+' forward/autodiff RNG unchanged',torch.equal(before,torch.get_rng_state()))
   for i,(actual,ref) in enumerate(zip(b,a)):compare(arm+'/Head_forward_gradient_%d'%i,actual,ref,criteria['FP32_normal_old_new_weights_and_output'] if i<2 else criteria['FP32_normal_old_new_gradients'])
  check('CUDA remains uninitialized',not torch.cuda.is_initialized());check('optimizer updates remain zero',result['optimizer_updates']==0)
 result['status']='PASS_CPU_OPERATOR_AND_LOCAL_HEAD_ONLY';result['seconds']=time.monotonic()-started;result['CUDA_initialized']=torch.cuda.is_initialized();persist()
 print(json.dumps({'status':result['status'],'checks':len(result['checks']),'comparisons':len(result['comparisons']),'seconds':result['seconds'],'optimizer_updates':0,'detector_forward':0,'local_head_forward':result['local_head_forward']}))
try:main()
except BaseException as e:result['status']='FAIL_CPU';result['error']=repr(e);result['seconds']=time.monotonic()-started;persist();raise
