"""One-forward paired geometry diagnostic; no training, AP/AR, speed or method change."""
import os,sys,json,hashlib,math,copy,ast,inspect,textwrap,contextlib,time,signal,gc,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}.items():os.environ[k]=v
import numpy as np
import torch
from sparse_rtdetr.research import sabl_control as sc
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.engine import tensor_digest
from sparse_rtdetr.protocol_r18.geometry import forward_guards
from sparse_rtdetr.research.sabl_formal_contract import sha,digest,read,need,reference
from sparse_rtdetr.r0.evaluation_data import Development548
from sparse_rtdetr.r0.evaluation import primary,truth
from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2
FIXTURE=ROOT/'tests/fixtures/sabl_pair_base.json'
SCHEMA='SABL_COARSE_REFINED_SAME_QUERY_DIAGNOSTIC_V1'
POLICY={'question':'same-model same-query coarse-to-refined localization by original GT scale',
 'arms':['S','A'],'endpoint':'fixed_corrected_e24_EMA_only','real_images':32,'batch':4,'queries':300,'classes':10,
 'precision':'FP32_no_autocast','eval':True,'DN':False,'padding':False,'input_hw':[768,1344],
 'selection':'four balanced density rank quartiles; 8 distinct images/band; 2 slots each S M L presence then fill; SHA256 salt pair32-v1; final original manifest order',
 'scale':'exclusive raw-area S<1024 M[1024,9216) L>=9216; diagnostic bins differ from COCO overlapping endpoints',
 'GT':'category1..10,score>0,positive original raw area, official category0 raster ignore fraction<0.5',
 'association':'all coarse query-GT IoUs FP64; greedy (-IoU,stable_GT_ID,query_ID), one-to-one, no IoU/class/score threshold; include zero-IoU; fixed before refined',
 'ignore':'GT ignore/crowd excluded with counts; no prediction ignore filtering for pairing; both stage raster ignore flags recorded separately',
 'delta_tolerance':1e-6,'invalid_IoU':0.,'invalid_category_priority':True,
 'denominators':'all eligible GT and paired GT both; unmatched capacity explicitly UNKNOWN geometry; all Q300 validity/unmatched-query counts',
 'no_AP_AR':True,'no_optimizer_EMA_or_backward':True,'no_extra_detector_forward':True,'no_method_proposal':True}
CAPS={'forward':16,'GPU_seconds':600,'wall_seconds':1200,'allocated_bytes':21<<30,'storage_bytes':128<<20,'return_bytes':2<<20,'family_bytes':30<<30}
CPU='CPU_SYNTHETIC_PAIR';GPU='GPU_PAIR_DIAGNOSTIC_PENDING_SEPARATE_AUTHORIZATION'
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())
def exact(r):need(Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'reference SHA/size');return read(r['path'])
def scale(area):return 'S' if area<1024 else ('M' if area<9216 else 'L')
def GT(row,raw):
 image=PrimaryImageV2(row['stable_image_id'],int(row['width']),int(row['height']));regions=[g for g in truth(row,raw) if g.category_id==0]
 integral=primary._build_integral(image,regions,gt_extension=None);eligible=[];counts={'raw':len(raw['rows']),'category0_or_other':0,'crowd_score0':0,'nonpositive':0,'raw_ignore':0}
 for obj in raw['rows']:
  if obj['category'] not in range(1,11):counts['category0_or_other']+=1;continue
  if obj['score']<=0:counts['crowd_score0']+=1;continue
  if obj['raw_area']<=0:counts['nonpositive']+=1;continue
  box=tuple(float(x) for x in obj['box'])
  if primary._ignored_fraction(integral,image,box,gt_extension=None)>=.5:counts['raw_ignore']+=1;continue
  eligible.append({'id':obj['id'],'category_id':obj['category'],'bbox_xyxy':list(box),'area':obj['raw_area'],'scale':scale(obj['raw_area'])})
 eligible.sort(key=lambda x:x['id']);need(len({g['id'] for g in eligible})==len(eligible),'GT identity')
 return eligible,counts,(image,integral)
def choose(rows,raw):
 candidates=[]
 for position,row in enumerate(rows):
  gt,count,_=GT(row,raw(row));candidates.append({'row':row,'original_position':position,'eligible':len(gt),'scales':{s:sum(g['scale']==s for g in gt) for s in 'SML'},'excluded':count})
 ranked=sorted(candidates,key=lambda x:(x['eligible'],x['row']['stable_image_id']));selected=[];bands=[]
 for band in range(4):
  group=ranked[len(ranked)*band//4:len(ranked)*(band+1)//4];picked=[]
  def order(tag,items):return sorted(items,key=lambda x:(hashlib.sha256(('pair32-v1/'+str(band)+'/'+tag+'/'+x['row']['stable_image_id']).encode()).hexdigest(),x['row']['stable_image_id']))
  for s in 'SML':
   options=order(s,[x for x in group if x['scales'][s]>0 and x not in picked]);picked+=options[:2]
  picked+=order('fill',[x for x in group if x not in picked])[:8-len(picked)]
  need(len(picked)==8,'density band cannot supply8');selected += [dict(x,density_band=band) for x in picked]
  bands.append({'band':band,'members':len(group),'eligible_count_min':min(x['eligible'] for x in group),'eligible_count_max':max(x['eligible'] for x in group),'selected':8})
 selected.sort(key=lambda x:x['original_position']);need(len({x['row']['stable_image_id'] for x in selected})==32,'selection unique32')
 return {'selected':selected,'bands':bands,'selection_SHA':digest(selected),'uses_predictions':False,'independent_validation':False}
def iou_matrix(boxes,gt):
 a=np.asarray(boxes,dtype=np.float64).reshape(-1,4);b=np.asarray([g['bbox_xyxy'] for g in gt],dtype=np.float64).reshape(-1,4)
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);wh=np.maximum(hi-lo,0);inter=wh[...,0]*wh[...,1]
 aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1);union=aa[:,None]+bb[None,:]-inter
 return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def associate(boxes,gt):
 if not gt:return [],[]
 need([g['id'] for g in gt]==sorted(g['id'] for g in gt),'GT stable lexical order')
 mat=iou_matrix(boxes,gt);q,g=np.indices(mat.shape);order=np.lexsort((q.ravel(),g.ravel(),-mat.ravel()));uq=set();ug=set();pairs=[]
 # GT is sorted by stable ID, so g is the exact lexical tie order. No epsilon costs.
 for k in order:
  qi,gi=int(q.ravel()[k]),int(g.ravel()[k])
  if qi not in uq and gi not in ug:uq.add(qi);ug.add(gi);pairs.append((qi,gi))
  if len(pairs)==min(len(boxes),len(gt)):break
 return pairs,[i for i in range(len(gt)) if i not in ug]
def classify(before,after,valid):
 if not valid:return 'to_invalid'
 d=after-before
 return 'improved' if d>POLICY['delta_tolerance'] else ('degraded' if d<-POLICY['delta_tolerance'] else 'unchanged')
def xy(boxes):return torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),-1)
def instrument(fn,fields,observer):
 tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));node=tree.body[0];need(isinstance(node.body[-1],ast.Return),'observer single final return')
 old=copy.deepcopy(tree);ret=node.body.pop();node.body.extend([ast.Assign(targets=[ast.Name(id='_pair_return',ctx=ast.Store())],value=ret.value),ast.Expr(value=ast.Call(func=ast.Name(id='_pair_observer',ctx=ast.Load()),args=[ast.Dict(keys=[ast.Constant(k) for k in fields],values=[ast.Name(id=k,ctx=ast.Load()) for k in fields])],keywords=[])),ast.Return(value=ast.Name(id='_pair_return',ctx=ast.Load()))])
 # Removing the sole observation restores the original full computation AST exactly.
 proof=copy.deepcopy(tree);tail=proof.body[0].body[-3:];proof.body[0].body[-3:]=[ast.Return(value=tail[0].value)]
 need(ast.dump(proof,include_attributes=False)==ast.dump(old,include_attributes=False),'observer computation AST drift')
 ast.fix_missing_locations(tree);ns=dict(fn.__globals__);ns['_pair_observer']=observer;exec(compile(tree,fn.__code__.co_filename,'exec'),ns)
 return ns[fn.__name__]
class Capture:
 def __init__(self):self.calls={'decode':0,'roi':0};self.data={}
 def receive(self,stage,fields):
  self.calls[stage]+=1;need(self.calls[stage]==1,'multiple observed calls')
  self.data[stage]={k:v.detach().cpu().clone() for k,v in fields.items()}
 def __enter__(self):
  self.original=(sc.decode,sc.roi_features)
  dec=instrument(sc.decode,['reference','e','wh','valid','box'],lambda x:self.receive('decode',x));roi=instrument(sc.roi_features,['r','rois'],lambda x:self.receive('roi',x))
  sc.decode,sc.roi_features=dec,roi;return self
 def __exit__(self,*_):sc.decode,sc.roi_features=self.original
 def record(self,out):
  need(self.calls=={'decode':1,'roi':1},'missing observer');d=self.data['decode'];r=self.data['roi'];b,q=out['pred_boxes'].shape[:2]
  need(q==300 and torch.equal(d['reference'],out['coarse_pred_boxes'].cpu()) and torch.equal(d['reference'],r['r']),'same query reference')
  need(torch.equal(d['box'],out['pred_boxes'].cpu()) and torch.equal(d['valid'],out['refined_valid'].cpu()),'actual final output identity')
  logits=out['pred_logits'].float().cpu();need(bool(torch.isfinite(logits).all()),'nonfinite scores');need(logits.shape==(b,300,10) and all(p not in out or out[p] is None for p in ['dn_meta','dn_aux_outputs']),'ordinary no DN/padding')
  return {'coarse_cxcywh_normalized':d['reference'],'reference_cxcywh_normalized':r['r'],'roi_input_pixel_xyxy_with_batch':r['rois'].reshape(b,300,5),'raw_edges_LRTB_normalized':d['e'],'raw_wh_normalized':d['wh'],'final_cxcywh_normalized':d['box'],'valid':d['valid'],'logits':logits,'scores_sigmoid_FP32_CPU':logits.sigmoid(),'query_id':torch.arange(300)}
def source(manifest):
 from r18_deploy import verify_release
 m=read(manifest);verify_release(ROOT,sha(manifest));f=read(FIXTURE)
 need(Path(manifest).resolve()==ROOT/'RELEASE_MANIFEST.json','release manifest')
 for n,r in f['scientific_sources'].items():need(sha(ROOT/n)==r['sha256'],'scientific bytes '+n)
 for n in ['decode','roi_features']:need(hashlib.sha256(inspect.getsource(getattr(sc,n)).encode()).hexdigest()==f['observed_function_SHA'][n],'observed function identity')
 return m
def validate(c):
 need(c['schema']==SCHEMA and c['mode'] in [CPU,GPU] and c['policy']==POLICY and c['caps']==CAPS,'fixed definition policy/caps')
 need(c['GPU_execution_authorized'] is False and c['future_separate_authorization_required'] is True,'preparation not GPU authorization')
 need(c['planned_forward']==(4 if c['mode']==CPU else 16),'forward definition')
 need(len(c['selection']['selected'])==(8 if c['mode']==CPU else 32) and c['selection']['selection_SHA']==digest(c['selection']['selected']),'selection identity')
 need(len({x['row']['stable_image_id'] for x in c['selection']['selected']})==len(c['selection']['selected']),'duplicate IDs')
 if c['mode']==GPU:
  f=read(FIXTURE);base=exact(f['formal_contract']);need(c['runtime']=={'python':base['metadata']['python'],'environment':base['metadata']['environment']} and c['data']==base['metadata']['data']['development'] and c['GPU_uuid']==base['metadata']['gpu_uuid'],'runtime/data binding');need(c['cost_lineage']['reference']==f['cost'] and c['cost_lineage']['prior_all_research']==exact(f['cost'])['cumulative_all_research'] and c['cost_lineage']['formal_forward_remaining_preserved']==0,'cost lineage no transfer');need(c['endpoints']==f['endpoints'] and c['formal_commit']==f['formal_commit'] and c['formal_manifest_SHA']==f['formal_manifest']['sha256'],'fixed real endpoint binding')
 return c
def fixture_data():
 from sparse_rtdetr.localization.data import parse_raw
 rows=[{'stable_image_id':'SYNTHETIC_PAIR_ONLY/%03d'%i,'width':160,'height':128} for i in range(8)]
 def raw(row):
  if row['stable_image_id'].endswith('/000'):return {'rows':[],'counts':{}}
  return parse_raw(b'12,16,8,8,1,1,0,0\n30,30,40,40,1,2,0,0\n50,10,100,100,1,3,0,0\n0,0,10,10,0,0,0,0\n110,5,8,8,0,4,0,0\n',row['stable_image_id'],expected_rows=5)
 def item(row,position):return torch.rand((3,128,128),generator=torch.Generator().manual_seed(100+position)),raw(row)
 return rows,raw,item
def prepare(a):
 m=source(a.manifest);cpu=a.synthetic;f=read(FIXTURE);sel=None
 if cpu:
  rows,raw,_=fixture_data();selected=[{'row':r,'original_position':i,'density_band':i//2} for i,r in enumerate(rows)];sel={'selected':selected,'selection_SHA':digest(selected),'synthetic':True};endpoints=read(a.synthetic_checkpoints)
 else:
  base=exact(f['formal_contract'])
  need(sha(f['formal_manifest']['path'])==f['formal_manifest']['sha256'],'formal source binding');done=exact(f['terminal_audit']);need(done['status']=='VERIFIED_CLOSED_ENGINEERING_PASS_SCIENTIFIC_FAIL','formal closed identity')
  reader=Development548({'manifest':base['metadata']['data']['development']['manifest']['path'],'roots':base['metadata']['data']['development']['roots']});sel=choose(reader.records,reader.raw)
  endpoints=f['endpoints']
  for arm in ['S','A']:
   ep=exact(endpoints[arm]);need(ep['arm']==arm and ep['updates']==7296 and ep['epochs']==24 and ep['EMA'] and not ep['acceptance_state_used'],'fixed e24 EMA');need(sha(ep['checkpoint']['path'])==ep['checkpoint']['sha256'],'physical checkpoint')
 c={'schema':SCHEMA,'mode':CPU if cpu else GPU,'policy':POLICY,'caps':CAPS,'execution_commit':m['commit'],'execution_manifest_SHA':sha(a.manifest),'formal_commit':f['formal_commit'],'formal_manifest_SHA':f['formal_manifest']['sha256'],'endpoints':endpoints,'selection':sel,'planned_forward':4 if cpu else 16,'data':None if cpu else base['metadata']['data']['development'],'GPU_uuid':None if cpu else base['metadata']['gpu_uuid'],'runtime':None if cpu else {'python':base['metadata']['python'],'environment':base['metadata']['environment']},'GPU_execution_authorized':False,'future_separate_authorization_required':True,'formal_forward_balance':0,'attempt_root':str(Path(a.attempt_root).resolve()),'requested_separate_diagnostic_forward':16,'budget_not_borrowed_from_formal_or_unused_debug':True,'cost_lineage':None if cpu else {'reference':f['cost'],'prior_all_research':exact(f['cost'])['cumulative_all_research'],'formal_forward_remaining_preserved':exact(f['cost'])['remaining']['forward']}}
 validate(c);write(a.out,c);write(Path(a.out).parent/'PREPARATION_CLOSED.json',{'definition_SHA':sha(a.out),'GPU_authorized':False,'real_images_read':False,'model_instantiated':False,'annotations_only_for_selection':not cpu,'selection_SHA':sel['selection_SHA'],'selected_scale_GT_counts':None if cpu else {s:sum(x['scales'][s] for x in sel['selected']) for s in 'SML'}})
def load_EMA(c,arm,device):
 ep=exact(c['endpoints'][arm]);cp=ep['checkpoint'];need(sha(cp['path'])==cp['sha256'] and Path(cp['path']).stat().st_size==cp['bytes'],'checkpoint bytes')
 p=torch.load(cp['path'],map_location='cpu',weights_only=True);need(p['format']==('CPU_SYNTHETIC_EMA_PAIR_FIXTURE' if c['mode']==CPU else 'sparse_rtdetr.sabl_formal24_fullstate_v1'),'EMA format')
 need(p['binding']==ep['binding'] and p['binding']['version']==sc.VERSION and p['binding']['arm']==arm and p['ema']['updates']==7296,'EMA identity')
 need(ep['epochs']==24 and ep['updates']==7296 and ep['EMA'] and not ep['acceptance_state_used'],'endpoint clock')
 if c['mode']==GPU:need(p['engine']['completed_updates']==7296 and p['ema']['decay']==.9999 and p['ema']['warmups']==2000,'certified EMA clock/settings')
 cfg=Config(input_hw=(128,128),physical_batch_size=4,amp_dtype='float32') if c['mode']==CPU else Config()
 model,_=sc.build(arm,cfg);model.eval().requires_grad_(False);expected=tensor_digest(sorted(p['ema']['module'].items()));model.load_state_dict(p['ema']['module'],strict=True)
 need(expected==tensor_digest(sorted(model.state_dict().items())),'EMA tensor mapping');del p;gc.collect();model.to(device)
 return model,{'EMA_module_SHA':expected,'only_EMA_model_loaded':True,'optimizer_EMA_updater_engine_RNG_restored':False,'raw_model_used':False}
def summarize(records):
 rows={s:{'all_eligible_GT':0,'paired_GT':0,'unpaired_capacity_GT':0,'improved':0,'unchanged':0,'degraded':0,'to_invalid':0,'IoU_before':[],'IoU_after':[],'delta':[]} for s in 'SML'}
 query={'all':0,'unpaired':0,'invalid_all':0,'invalid_unpaired':0};exclusions={}
 for r in records:
  query['all']+=300;query['unpaired']+=r['unpaired_query_count'];query['invalid_all']+=r['invalid_all_queries'];query['invalid_unpaired']+=r['invalid_unpaired_queries']
  for k,v in r['excluded_GT'].items():exclusions[k]=exclusions.get(k,0)+v
  for g in r['GT']:rows[g['scale']]['all_eligible_GT']+=1
  for g in r['unpaired_GT']:rows[g['scale']]['unpaired_capacity_GT']+=1
  for p in r['pairs']:
   row=rows[p['scale']];row['paired_GT']+=1;row[p['change']]+=1;row['IoU_before'].append(p['IoU_before']);row['IoU_after'].append(p['IoU_after']);row['delta'].append(p['delta'])
 for r in rows.values():
  n=r['paired_GT'];alln=r['all_eligible_GT'];need(sum(r[k] for k in ['improved','unchanged','degraded','to_invalid'])==n and n+r['unpaired_capacity_GT']==alln,'complete denominators')
  r['proportions_of_paired']={k:None if not n else r[k]/n for k in ['improved','unchanged','degraded','to_invalid']}
  r['proportions_of_all_GT']={k:None if not alln else r[k]/alln for k in ['improved','unchanged','degraded','to_invalid','unpaired_capacity_GT']}
  for k in ['IoU_before','IoU_after','delta']:
   v=r[k];r[k]={'n':len(v),'mean':None if not v else float(np.mean(v)),'quantiles_0_25_50_75_100':None if not v else np.quantile(v,[0,.25,.5,.75,1]).tolist()}
 return {'scales':rows,'query_denominators':query,'excluded_GT':exclusions,'no_AP_AR_or_training_cause_claim':True}
def pair_image(record,row,raw):
 gt,excluded,ignore=GT(row,raw);factor=torch.tensor([row['width'],row['height']]*2,dtype=torch.float32)
 coarse=xy(record['coarse_cxcywh_normalized'])*factor;final=xy(record['final_cxcywh_normalized'])*factor
 pairs,unpaired=associate(coarse.numpy(),gt);refined_iou=iou_matrix(final.numpy(),gt);before_iou=iou_matrix(coarse.numpy(),gt);image,integral=ignore;paired=[];used=set()
 for q,g in pairs:
  valid=bool(record['valid'][q]);before=float(before_iou[q,g]);after=float(refined_iou[q,g]) if valid else 0.;used.add(q)
  paired.append({'query_id':q,'GT_id':gt[g]['id'],'scale':gt[g]['scale'],'GT_category':gt[g]['category_id'],'IoU_before':before,'IoU_after':after,'delta':after-before,'refined_valid':valid,'change':classify(before,after,valid),'coarse_ignore':primary._ignored_fraction(integral,image,tuple(coarse[q].tolist()),gt_extension=None)>=.5,'refined_ignore':None if not valid else primary._ignored_fraction(integral,image,tuple(final[q].tolist()),gt_extension=None)>=.5})
 return {'image_id':row['stable_image_id'],'GT':gt,'excluded_GT':excluded,'pairs':paired,'unpaired_GT':[gt[i] for i in unpaired],'unpaired_query_count':300-len(used),'invalid_all_queries':int((~record['valid']).sum()),'invalid_unpaired_queries':sum(not bool(record['valid'][q]) for q in range(300) if q not in used)}
def family_bytes():
 from sabl_ipc import owned_family_bytes
 ex=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution');private,_=owned_family_bytes(ex)
 return sum(size(p) for p in ex.glob('r18-sabl-*') if p.is_dir())+private+sum(size(p) for p in map(Path,['/tmp/torchinductor_lyy','/home/lyy/.triton/cache','/home/lyy/.nv/ComputeCache','/home/lyy/.cache/torch/inductor','/home/lyy/.cache/triton']) if p.exists())

def run(a):
 c=validate(read(a.definition));m=source(a.manifest);need(sha(a.definition)==a.approve_definition_SHA and sha(a.manifest)==a.approve_manifest_SHA==c['execution_manifest_SHA'] and m['commit']==c['execution_commit'],'execution binding')
 cpu=c['mode']==CPU;need((cpu and a.authorize_cpu and not a.authorize_gpu) or (not cpu and a.authorize_gpu and not a.authorize_cpu and a.grant_forward==16),'separate execution authority')
 out=Path(a.out);need(str(out.resolve())==c['attempt_root'],'fixed attempt binding');need(not out.exists() and not out.parent.joinpath(out.name+'-ONCE.json').exists(),'duplicate attempt')
 out.mkdir();write(out.parent/(out.name+'-ONCE.json'),{'definition_SHA':sha(a.definition),'no_retry':True,'CPU_synthetic':cpu});start=time.monotonic();ledger={'forward':0,'backward':0,'optimizer_updates':0,'EMA_updates':0};peak=0
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 def guard():
  nonlocal peak
  need(time.monotonic()-start<(300 if cpu else CAPS['GPU_seconds']),'phase time');need(size(out)<CAPS['storage_bytes'],'attempt storage')
  if not cpu:
   peak=max(peak,torch.cuda.max_memory_allocated());need(peak<=CAPS['allocated_bytes'],'allocated cap')
   from sabl_ipc import owned_family_bytes
   family=family_bytes();need(family<CAPS['family_bytes'],'family storage cap')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('paired diagnostic deadline')));signal.alarm(300 if cpu else CAPS['GPU_seconds'])
 try:
  if not cpu:
   from sabl_corrected_formal import allocator_policy,GPU_memory_gate
   import torchvision
   env=c['runtime']['environment'];need(Path(sys.executable).resolve()==Path(c['runtime']['python']).resolve() and str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'certified runtime')
   need(not m['candidate_not_for_real_execution'],'synthetic release no GPU');allocator_policy(torch)
   need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU jobs')
   parts=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');need(len(parts)==4 and parts[0].strip()==c['GPU_uuid'],'GPU identity')
   GPU_memory_gate(*[int(x.strip())*(1<<20) for x in parts[1:]])
   need(family_bytes()+CAPS['storage_bytes']<CAPS['family_bytes'],'family available storage');need(torch.cuda.is_available() and torch.cuda.device_count()==1,'CUDA')
   torch.cuda.set_per_process_memory_fraction(CAPS['allocated_bytes']/torch.cuda.get_device_properties(0).total_memory,0);free,total=map(int,torch.cuda.mem_get_info());GPU_memory_gate(total,total-free,free,True)
   cache=out/'compiler-cache'
   for key,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions')]:
    q=cache/sub;q.mkdir(parents=True);os.environ[key]=str(q)
  if cpu:rows,raw,item=fixture_data()
  else:
   reader=Development548({'manifest':c['data']['manifest']['path'],'roots':c['data']['roots']});raw=reader.raw;item=reader.item
  selected=c['selection']['selected'];all_results={};audits={}
  for arm in ['S','A']:
   model,loaded=load_EMA(c,arm,'cpu' if cpu else 'cuda');before=tensor_digest(sorted(model.state_dict().items()));records=[];saved=[]
   need(all(p.grad is None for p in model.parameters()) and not any(x.training for x in model.modules()),'eval/gradient state')
   for start_i in range(0,len(selected),4):
    guard();batch=selected[start_i:start_i+4];images=torch.stack([item(x['row'],x['original_position'])[0] for x in batch]).to('cpu' if cpu else 'cuda')
    need(ledger['forward']<c['planned_forward'],'forward cap');ledger['forward']+=1;write(out/('forward-%03d.json'%ledger['forward']),dict(ledger))
    with torch.inference_mode(),forward_guards(model),Capture() as cap:value=model(images)
    need(value['pred_logits'].dtype==torch.float32,'original evaluation FP32');data=cap.record(value)
    for i,x in enumerate(batch):
     rec={k:(v[i] if k!='query_id' else v) for k,v in data.items()};row=x['row'];factor=torch.tensor([row['width'],row['height']]*2,dtype=torch.float32)
     rec['coarse_xyxy_original_pixels']=xy(rec['coarse_cxcywh_normalized'])*factor;rec['final_xyxy_original_pixels']=xy(rec['final_cxcywh_normalized'])*factor
     rec['reference_roi_xyxy_original_pixels']=rec['roi_input_pixel_xyxy_with_batch'][:,1:]/torch.tensor([images.shape[3],images.shape[2]]*2)*factor
     rec['raw_edges_LRTB_original_pixels']=rec['raw_edges_LRTB_normalized']*torch.tensor([row['width'],row['width'],row['height'],row['height']])
     need(torch.allclose(rec['reference_roi_xyxy_original_pixels'],rec['coarse_xyxy_original_pixels'],atol=.002,rtol=1e-6),'RoI coordinate conversion')
     saved.append({'image_id':x['row']['stable_image_id'],'original_position':x['original_position'],**rec});records.append(pair_image(rec,x['row'],raw(x['row'])))
    del value,images,data;guard()
   need(before==tensor_digest(sorted(model.state_dict().items())) and all(p.grad is None for p in model.parameters()) and not any(x.training for x in model.modules()),'no parameter/buffer/gradient update')
   torch.save(saved,out/(arm+'-all-query-pairs.pt'));write(out/(arm+'-GT-pairs.json'),records);all_results[arm]=summarize(records);audits[arm]={**loaded,'state_unchanged':True,'gradients_all_None':True,'BN_unchanged':True,'covered_images':len(saved),'ordinary_queries':len(saved)*300,'forward':len(selected)//4}
   del model,saved;gc.collect();guard()
  need(ledger['forward']==c['planned_forward'] and ledger['backward']==ledger['optimizer_updates']==ledger['EMA_updates']==0,'terminal exact counts')
  result={'status':'CLOSED_PAIRED_LOCAL_DIAGNOSTIC_ONLY','CPU_synthetic':cpu,'ledger':ledger,'arms':all_results,'model_audits':audits,'seconds':time.monotonic()-start,'peak_allocated_bytes':None if cpu else peak,'scientific_scope':'32 repeated-development images only; geometric diagnostic, not AP, generalization or training cause','no_followup':True}
  write(out/'RESULT.json',result);write(out/'SHA_INDEX.json',{p.name:reference(p) for p in out.iterdir() if p.is_file()});guard();write(out/'CLOSED.json',{'result':reference(out/'RESULT.json'),'definition_SHA':sha(a.definition),'must_exit':True});write(out/'EXIT.json',{'returncode':0,'seconds':time.monotonic()-start,'ledger':ledger,'CPU_synthetic':cpu});return 0
 except BaseException as e:
  import traceback
  if not cpu and torch.cuda.is_initialized():
   try:peak=max(peak,torch.cuda.max_memory_allocated())
   except BaseException:peak='UNKNOWN_OBSERVATION_FAILED'
  write(out/'STOP.json',{'error':repr(e),'traceback':traceback.format_exc(),'ledger':ledger,'seconds':time.monotonic()-start,'peak_allocated_bytes':None if cpu else peak,'no_retry':True});write(out/'EXIT.json',{'returncode':1,'seconds':time.monotonic()-start,'ledger':ledger,'CPU_synthetic':cpu});return 1
 finally:signal.alarm(0)
def execute(a):
    # Bounded parent for the exact same CLI, including import/preparation overhead.
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'run',*sys.argv[2:]]
    out=Path(a.out);need(str(out.resolve())==read(a.definition)['attempt_root'],'fixed parent attempt binding');need(not out.exists() and not out.parent.joinpath(out.name+'-PROCESS_EXIT.json').exists(),'duplicate parent execution')
    start=time.monotonic();proc=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    cpu=read(a.definition)['mode']==CPU
    try:stdout,stderr=proc.communicate(timeout=295 if cpu else CAPS['GPU_seconds']-5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:stdout,stderr=proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
    write(out.parent/(out.name+'-PROCESS_EXIT.json'),{'returncode':proc.returncode,'complete_process_wall_seconds':time.monotonic()-start,'no_retry':True,'CPU_synthetic':read(a.definition)['mode']==CPU})
    (out.parent/(out.name+'-worker.stdout')).write_bytes(stdout);(out.parent/(out.name+'-worker.stderr')).write_bytes(stderr)
    if not cpu:
        c=read(a.definition);prefix=out.name;ex=read(out/'EXIT.json') if (out/'EXIT.json').exists() else None
        count=ex['ledger']['forward'] if ex is not None else len(list(out.glob('forward-*.json')))
        seconds=time.monotonic()-start;prior=c['cost_lineage']['prior_all_research']
        write(out.parent/'COST_ADDITION.json',{'scope':'INDEPENDENT_PAIRED_DIAGNOSTIC_NOT_FORMAL_TRAINING','prior_all_research':prior,'new_forward':count,'new_backward_optimizer_EMA_updates':0,'new_GPU_seconds_conservative_parent_wall':seconds,'cumulative_all_research':{'forward':prior['forward']+count,'GPU_stage_seconds':prior['GPU_stage_seconds']+seconds},'formal_forward_balance_preserved':0,'borrowed_old_budget':False,'no_refund':True})
    return 0 if proc.returncode==0 else 1

def main():
 import argparse
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--synthetic',action='store_true');q.add_argument('--synthetic-checkpoints');q.add_argument('--attempt-root',required=True)
 for command in ['run','execute']:
  q=s.add_parser(command)
  for x in ['definition','manifest','approve-definition-SHA','approve-manifest-SHA','out']:q.add_argument('--'+x,required=True)
  q.add_argument('--authorize-cpu',action='store_true');q.add_argument('--authorize-gpu',action='store_true');q.add_argument('--grant-forward',type=int,default=0)
 a=p.parse_args();return globals()[a.command](a)
if __name__=='__main__':raise SystemExit(main())
