"""Independent GT oracle against actual production v3 coder/loss/decoder, CPU only."""
import os,sys,json,time,resource,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.alarm(300)
import torch
from torch.nn import functional as F
from sparse_rtdetr.research import sabl_control as sc
ATOL=1e-6

def edges_from_GT(g):
 # Independent coordinate oracle: do NOT call corners, encode, decode or permute XYXY.
 cx,cy,w,h=g.unbind(-1)
 return torch.stack((cx-w/2,cx+w/2,cy-h/2,cy+h/2),-1)

def main(out):
 out=Path(out);out.mkdir();start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);checks=[];rows=[]
 def check(name,value):
  checks.append({'name':name,'pass':bool(value)})
  (out/'CHECKS.json').write_text(json.dumps(checks))
  if not value:raise AssertionError(name)
 check('v3 scientific identity',sc.VERSION=='r18-sabl-control-lrtb-logsigmoid-v3')
 cases=json.loads((ROOT/'tests/fixtures/sabl_gt_roundtrip_cases.json').read_text())['cases'];check('old eleven retained',sum(x['from_original_11'] for x in cases)==11)
 for row in cases:
  r=torch.tensor([row['reference']],dtype=torch.float32);g=torch.tensor([row['GT']],dtype=torch.float32);truth=edges_from_GT(g)
  x,scale=sc.buckets(r);target=(x-truth[...,None])/scale[...,None];idx=torch.argsort(target.abs(),stable=True);y=torch.zeros_like(target).scatter_(-1,idx[...,:1],1.)
  expected_cw=(~((target.abs()<1)&(y==0))).float();expected_ow=torch.zeros_like(target).scatter_(-1,idx[...,:1],1.);second=idx[...,1:2];expected_ow.scatter_(-1,second,(target.abs().gather(-1,second)<1).float())
  t,actual_y,cw,ow=sc.encode(r,g)
  check(row['name']+' correct independent GT edge targets',torch.equal(target,t))
  check(row['name']+' bucket and mask semantics',torch.equal(y,actual_y) and torch.equal(cw,expected_cw) and torch.equal(ow,expected_ow))
  z=torch.where(y>0,torch.tensor(20.),torch.tensor(-20.));decoded,valid=sc.decode(r,z,target)
  err=float((decoded-g).abs().max());check(row['name']+' GT-oracle ideal prediction roundtrip',bool(valid.all()) and err<=ATOL)
  actual_decoded,actual_valid=sc.decode(r,torch.where(actual_y>0,20.,-20.),t)
  check(row['name']+' actual encode ideal prediction roundtrip',bool(actual_valid.all()) and torch.allclose(actual_decoded,g,atol=ATOL,rtol=0))
  losses,diag=sc.boundary_loss({'reference':r[:,None],'logits':z[:,None],'offsets':target[:,None]},[{'boxes':g}],[(torch.tensor([0]),torch.tensor([0]))])
  check(row['name']+' actual loss minimized at correct GT',float(losses['loss_offset'])==0 and float(losses['loss_bucket'])<1e-6)
  rows.append({**row,'decoded':decoded.tolist()[0],'valid':bool(valid.item()),'max_abs_error':err,'loss_bucket':float(losses['loss_bucket']),'loss_offset':float(losses['loss_offset'])})
 # Nonideal predictions: independent GT edge perturbation and correct-side gradients.
 nonideal=[]
 for side in range(4):
  r=torch.tensor([[[.5,.5,.4,.3]]],requires_grad=True);g=torch.tensor([[.47,.58,.18,.26]],requires_grad=True);x,scale=sc.buckets(r);truth=edges_from_GT(g.detach())[:,None];oracle=(x-truth[...,None])/scale[...,None];k=oracle.abs().argmin(-1);y=torch.zeros_like(oracle).scatter_(-1,k[...,None],1.)
  d=oracle.detach().clone();d[0,0,side,k[0,0,side]]+=.25;d.requires_grad_();z=torch.where(y>0,2.,-2.).detach().requires_grad_()
  losses,diag=sc.boundary_loss({'reference':r,'logits':z,'offsets':d},[{'boxes':g}],[(torch.tensor([0]),torch.tensor([0]))]);gz,gd,gr,gg=torch.autograd.grad(sum(losses.values()),(z,d,r,g),allow_unused=True)
  index=int(k[0,0,side]);nz=(gd.abs()>1e-7).nonzero().tolist();check('side%d offset gradient correct index/sign'%side,nz==[[0,0,side,index]] and float(gd[0,0,side,index])>1e-7)
  check('side%d positive bucket and mask gradients'%side,bool((gz[y>0]<-1e-7).all()) and bool((gz[(y==0)&(oracle.abs()<1)]==0).all()))
  check('side%d GT/reference detached'%side,gr is None and gg is None)
  dec,v=sc.decode(r.detach().reshape(1,4),z.detach().reshape(1,4,7),d.detach().reshape(1,4,7));pred_edges=edges_from_GT(dec);desired=truth.reshape(1,4).clone();desired[0,side]-=scale[0,0,side]*.25
  check('side%d decoded shift on the correct edge'%side,bool(v.all()) and torch.allclose(pred_edges,desired,atol=ATOL,rtol=0))
  # Misclassify a non-excluded negative bucket on exactly this semantic edge.
  cw=(~((oracle.abs()<1)&(y==0))).float();wrong=int(((cw[0,0,side]>0)&(y[0,0,side]==0)).nonzero()[0]);badz=z.detach().clone();badz[0,0,side,index]=-2.;badz[0,0,side,wrong]=2.;badz.requires_grad_();cl,_=sc.boundary_loss({'reference':r,'logits':badz,'offsets':oracle.detach()},[{'boxes':g}],[(torch.tensor([0]),torch.tensor([0]))]);grad=torch.autograd.grad(cl['loss_bucket'],badz)[0]
  check('side%d wrong-class gradient penalizes the correct edge'%side,float(grad[0,0,side,index])<-1e-7 and float(grad[0,0,side,wrong])>1e-7)
  nonideal.append({'side':('L','R','T','B')[side],'nearest_bucket':index,'wrong_bucket':wrong,'offset_gradient':float(gd[0,0,side,index]),'predicted_edges':pred_edges.tolist()[0],'expected_shifted_edges':desired.tolist()[0]})
 # Keep invalid readout counterexamples; no positive-size clipping, swap or fallback.
 r=torch.tensor([[.5,.5,.2,.2]]);x,scale=sc.buckets(r);z=torch.zeros_like(x);counter=[]
 for name,bad_edges in [('LR_reverse',[.7,.3,.4,.6]),('TB_reverse',[.4,.6,.7,.3]),('both_reverse',[.7,.3,.7,.3]),('zero_width',[.5,.5,.4,.6]),('zero_height',[.4,.6,.5,.5])]:
  e=torch.tensor([bad_edges]);d=(x-e[...,None])/scale[...,None];box,valid=sc.decode(r,z,d);check(name+' still rejected without repair',not bool(valid.any()) and bool((box[...,2:]==0).all()));counter.append({'name':name,'constructed_edges_LRTB':bad_edges,'returned_box':box.tolist()[0],'valid':False})
 check('CPU only',not torch.cuda.is_initialized());result={'status':'PASS_CPU_INDEPENDENT_GT_CORRECTION','version':sc.VERSION,'checks':len(checks),'GT_cases':len(rows),'old_11_all_pass':all(r['valid'] and r['max_abs_error']<=ATOL for r in rows if r['from_original_11']),'all_GT_cases_pass':True,'atol':ATOL,'rtol':0,'cases':rows,'nonideal_gradients':nonideal,'counterexamples':counter,'CPU_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'detector_forward':0,'optimizer_updates':0,'GPU_forward':0,'scientific_benefit':'UNKNOWN'};(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('cases','nonideal_gradients','counterexamples')}))
if __name__=='__main__':main(sys.argv[1])
