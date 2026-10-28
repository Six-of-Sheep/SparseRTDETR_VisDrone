"""CPU audit of ORIGINAL v1 side semantics. Detects the frozen defect; never fixes it."""
import argparse,copy,hashlib,importlib,json,os,resource,signal,sys,time
from pathlib import Path
EXPECTED_SC_SHA='c5945edd78ba7159ce83f433a1cc9a266543675ba4f3edac478d059d7875f05b'
ATOL=1e-6;ROI_ATOL=2e-5

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def dump(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--out',required=True);p.add_argument('--criteria');a=p.parse_args();out=Path(a.out);out.mkdir();release=Path(a.release).resolve();start=time.monotonic()
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
 resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.alarm(300)
 assert sha(release/'src/sparse_rtdetr/research/sabl_control.py')==EXPECTED_SC_SHA
 if a.criteria:
  criteria=json.loads(Path(a.criteria).read_text());assert criteria['roundtrip_atol_normalized']==ATOL and criteria['roundtrip_rtol']==0 and criteria['roundtrip_requires_valid_true'] and criteria['RoI_linear_field_atol']==ROI_ATOL
 sys.path[:0]=[str(release/'src'),str(release/'tools')]
 import torch
 from torch.nn import functional as F
 from sparse_rtdetr.research import sabl_control as sc
 assert sc.VERSION=='r18-sabl-control-v1' and Path(sc.__file__).resolve()==release/'src/sparse_rtdetr/research/sabl_control.py'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 checks=[]
 def check(name,condition):
  if not condition:raise AssertionError(name)
  checks.append(name)
 def box_from_edges(l,t,r,b):return [(l+r)/2,(t+b)/2,r-l,b-t]
 cases=[('square_centered',[.5,.5,.2,.2],[.5,.5,.2,.2]),('non_square',[.4,.6,.3,.2],[.4,.6,.2,.1]),('asymmetric_centers_valid_wrong_output',[.4,.6,.2,.1],[.4,.6,.2,.1]),('one_pixel',[.6,.3,.02,.03],[.6,.3,1/1344,1/768]),('thin',[.7,.2,.5,.1],[.7,.2,.4,.002]),('top_left_edge',[.03,.04,.1,.08],[.01,.02,.02,.04]),('bottom_right_edge',[.96,.95,.1,.12],[.98,.97,.04,.06]),('accidental_top_equals_right',[.2,.4,.2,.2],[.2,.4,.2,.2])]
 exact_reference=[.5,.5,.875/1.7,.4375/1.7]
 cases += [('bucket_centers',exact_reference,box_from_edges(.09375,.296875,.90625,.703125)),('bucket_midpoint_ties',exact_reference,box_from_edges(.25,.375,.75,.625)),('expanded_domain_boundary',exact_reference,box_from_edges(.0625,.28125,.9375,.71875))]
 records=[]
 for name,r,g in cases:
  reference=torch.tensor([r],dtype=torch.float32);gt=torch.tensor([g],dtype=torch.float32);true_xyxy=sc.corners(gt)
  check(name+' GT legal',bool((gt[:,2:]>0).all()) and bool((true_xyxy>=-1e-7).all()) and bool((true_xyxy<=1+1e-7).all()))
  target,label,cw,ow=sc.encode(reference,gt);logits=torch.where(label>0,torch.tensor(20.),torch.tensor(-20.));offset=target.clone();decoded,valid=sc.decode(reference,logits,offset)
  losses,diag=sc.boundary_loss({'reference':reference[:,None,:],'logits':logits[:,None,:,:],'offsets':offset[:,None,:,:]},[{'boxes':gt}],[(torch.tensor([0]),torch.tensor([0]))])
  error=float((decoded-gt).abs().max());record={'name':name,'reference_cxcywh':r,'GT_cxcywh':g,'corners_GT_LTRB_xyxy':true_xyxy.tolist()[0],'actual_encoded_nearest_bins':label.argmax(-1).tolist()[0],'actual_decoded_cxcywh':decoded.tolist()[0],'valid':bool(valid.item()),'max_abs_error_normalized':error,'nominal_roundtrip_pass':bool(valid.item()) and error<=ATOL,'ideal_actual_loss_bucket':float(losses['loss_bucket']),'ideal_actual_loss_offset':float(losses['loss_offset'])};records.append(record)
  check(name+' actual target minimizes offset loss',float(losses['loss_offset'])==0.)
  check(name+' actual ideal BCE small',float(losses['loss_bucket'])<1e-6)
 check('minimal square counterexample confirmed',not records[0]['nominal_roundtrip_pass'] and not records[0]['valid'])
 check('valid but incorrect box counterexample exists',any(r['valid'] and not r['nominal_roundtrip_pass'] for r in records))
 check('accidental equality case retained',next(r for r in records if r['name']=='accidental_top_equals_right')['nominal_roundtrip_pass'])
 # Real RoIAlign operator on a constructed linear feature; not a detector forward.
 field=(torch.arange(16)[None,:]+10*torch.arange(8)[:,None]).float()[None,None]
 roi=sc.roi_features(field,torch.tensor([[[.5,.5,.25,.25]]]),(64,128))
 x=5.5+(torch.arange(7)+.5)*4/7;y=2.5+(torch.arange(7)+.5)*2/7;expected=x[None,:]+10*y[:,None]
 roi_error=float((roi[0,0]-expected).abs().max());check('non-square pixel/stride8/aligned RoI coordinates',roi_error<=ROI_ATOL)
 # Build ONLY the real native criterion and matcher from the resolved production config.
 cm=sc.common;_,effective=cm.resolved_config(cm.Config())
 with cm.vendor._vendor_path(cm.vendor._vendor_root(release)):
  for name in ('src.solver','src.data','src.nn','src.zoo.rtdetr'):importlib.import_module(name)
  from src.core import GLOBAL_CONFIG
  from src.core.workspace import create
  from src.core.yaml_utils import merge_config
  saved=cm.vendor._snapshot_registry(GLOBAL_CONFIG)
  try:
   merged=merge_config(effective,cm.vendor._snapshot_registry(GLOBAL_CONFIG),inplace=False,overwrite=False);base=create(merged['criterion'],merged).cpu()
  finally:cm.vendor._restore_registry(GLOBAL_CONFIG,saved)
 from sparse_rtdetr.protocol_r18.criterion import adapt
 native=adapt(base);criterion=sc.Criterion(native,'S')
 boxes=torch.tensor([.85,.85,.05,.05]).repeat(1,300,1);boxes[0,3]=torch.tensor([.2,.4,.2,.2]);boxes[0,17]=torch.tensor([.6,.3,.2,.1]);boxes.requires_grad_()
 logits=torch.full((1,300,10),-7.);logits[0,3,1]=7.;logits[0,17,2]=7.;logits.requires_grad_()
 gt=torch.tensor([[.6,.3,.2,.1],[.2,.4,.2,.2]],requires_grad=True)
 targets=[{'boxes':gt,'labels':torch.tensor([2,1]),'gt_ids':['SYNTHETIC-gB','SYNTHETIC-gA'],'raw_area':torch.tensor([200.,300.]),'ignore_spatial_boxes':torch.empty(0,4),'ignore_union':torch.empty(0,4),'ignore_class_boxes':torch.empty(0,4),'ignore_class_labels':torch.empty(0,dtype=torch.long),'ignore_ids':[]}]
 def coarse():return {'pred_logits':logits.clone(),'pred_boxes':boxes.clone()}
 ordinary=coarse();ordinary['aux_outputs']=[coarse(),coarse()];ordinary['enc_aux_outputs']=[coarse()];ordinary['enc_meta']={'class_agnostic':False};ordinary['dn_aux_outputs']=[{'pred_logits':torch.full((1,2,10),-1.),'pred_boxes':gt.detach()[None].clone()} for _ in range(3)];ordinary['dn_meta']={'dn_positive_idx':[torch.tensor([0,1])],'dn_num_group':1}
 z=torch.zeros(1,300,4,7,requires_grad=True);offsets=torch.zeros(1,300,4,7,requires_grad=True);reference=boxes.detach().clone().requires_grad_();pred={'logits':z,'offsets':offsets,'reference':reference}
 native_losses=native(ordinary,targets);losses=criterion({**ordinary,'sabl':pred},targets);pairs=criterion.last_matches;diag=copy.deepcopy(criterion.last_diagnostics)
 check('actual native matcher binds query3 to GT idA/dst1 and query17 to idB/dst0',pairs[0][0].tolist()==[3,17] and pairs[0][1].tolist()==[1,0])
 check('native final aux encoder DN losses unchanged',all(torch.equal(v,losses[k]) for k,v in native_losses.items()) and len(native_losses)==21)
 check('one ordinary boundary supervision head only',len([v for v in diag if v['head']=='SABL_ordinary_only'])==1 and len(losses)==23)
 check('actual loss denominators 4M/M',diag[-1]=={'head':'SABL_ordinary_only','matches':2,'bucket_denominator':8,'offset_denominator':2})
 selected_gt=gt[pairs[0][1]];target,label,cw,ow=sc.encode(reference[0,pairs[0][0]],selected_gt)
 extra,info=sc.boundary_loss(pred,targets,pairs)
 check('actual weighted BCE input and target masks',torch.equal(extra['loss_bucket'],(F.binary_cross_entropy_with_logits(z[0,pairs[0][0]],label,reduction='none')*cw).sum()/8))
 check('actual weighted offset input and targets',torch.equal(extra['loss_offset'],(F.smooth_l1_loss(offsets[0,pairs[0][0]],target,beta=.1,reduction='none')*ow).sum()/2))
 def variable_ids(value):
  stack=[value.grad_fn];seen=set();found=set()
  while stack:
   fn=stack.pop()
   if fn is None or fn in seen:continue
   seen.add(fn)
   if hasattr(fn,'variable'):found.add(id(fn.variable))
   stack.extend(n for n,_ in fn.next_functions)
  return found
 check('loss graph reaches supplied logits and offsets without backward',id(z) in variable_ids(extra['loss_bucket']) and id(offsets) in variable_ids(extra['loss_offset']))
 check('encoded GT/reference graph detached',all(id(reference) not in variable_ids(v) and id(gt) not in variable_ids(v) for v in extra.values()))
 z2=z.detach().clone();d2=offsets.detach().clone();z2[0,299]+=10;d2[0,299]+=10
 untouched,_=sc.boundary_loss({**pred,'logits':z2,'offsets':d2},targets,pairs);check('unmatched query not supervised',all(torch.equal(extra[k],untouched[k]) for k in extra))
 # Mask interventions use the actual encoded masks and preserve all other values.
 for kind,mask,tensor,key in [('classification',cw,z.detach().clone(),'loss_bucket'),('offset',ow,offsets.detach().clone(),'loss_offset')]:
  index=(mask==0).nonzero()[0].tolist();m,s,k=index;q=int(pairs[0][0][m]);tensor[0,q,s,k]+=5
  variant={**pred,('logits' if kind=='classification' else 'offsets'):tensor};value,_=sc.boundary_loss(variant,targets,pairs);check(kind+' zero-mask perturbation has no effect',torch.equal(extra[key],value[key]))
 empty,empty_diag=sc.boundary_loss(pred,[{**targets[0],'boxes':gt[:0]}],[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]);check('empty match differentiable zero without backward',all(float(v)==0 and v.grad_fn is not None for v in empty.values()) and empty_diag=={'matches':0,'bucket_denominator':1,'offset_denominator':1})
 check('no leaf gradients created',all(v.grad is None for v in (z,offsets,reference,gt,boxes,logits)))
 check('CUDA never initialized',not torch.cuda.is_initialized())
 report={'status':'AUDIT_COMPLETED_CONFIRMED_IMPLEMENTATION_INCONSISTENCY','scientific_result_unchanged':True,'source_module':str(Path(sc.__file__).resolve()),'source_SHA':EXPECTED_SC_SHA,'predeclared_atol':ATOL,'predeclared_rtol':0,'nominal_roundtrip_pass':all(v['nominal_roundtrip_pass'] for v in records),'nominal_roundtrip_pass_count':sum(v['nominal_roundtrip_pass'] for v in records),'case_count':len(records),'cases':records,'checks':checks,'check_count':len(checks),'RoI_max_abs_error':roi_error,'actual_criterion_module':type(native).__module__,'actual_matcher_module':type(native.matcher).__module__,'actual_recipe':criterion.recipe(),'CPU_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'detector_constructed':False,'detector_forward':0,'backward':0,'optimizer_updates':0,'AP_AR_recomputed':0,'checkpoint_or_real_image_loaded':False,'CUDA_initialized':False,'no_patch_or_repair':True}
 dump(out/'CPU_AUDIT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('cases','checks','actual_recipe')}));return 0
if __name__=='__main__':raise SystemExit(main())
