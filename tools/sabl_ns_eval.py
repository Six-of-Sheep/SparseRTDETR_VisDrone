"""Original-v1 N/S e24 EMA evaluation only; no training or v2 fallback."""
import argparse,ast,base64,gc,hashlib,inspect,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from sparse_rtdetr.research import sabl_formal_contract as ct
from sabl_formal_preflight import checkpoint_bytes
SCOPE='PR26_NS_E24_EVALUATION_V1'
VERSION='r18-sabl-control-v1'
OLD_COMMIT='79ba59371c21ecfd4fa76885c413941403797214'
OLD_MANIFEST='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab'
OLD_CONTRACT='f734c400e942c265eafbb5180617a09c3b827a9b238f2a65bf08f24b5557356e'
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace')
FORMAL=WS/'experiments/localization-distribution/r18-sabl-formal24-20261027-001'
TASK=WS/'experiments/localization-distribution/r18-sabl-ns-e24-evaluation-20261028-001'
LATEST_COST=WS/'experiments/localization-distribution/r18-sabl-forward-validation-20261028-002/COST_ADDITION.json'
CP_SHA={'N':'fbf798c33649253a410c2feeac01bb62757089221709726a02493117469de921','S':'deb67cf4bfc7e1a25998fb8490309a0ee812471859bacfbe8a42d8be4673f21f'}
CAPS={'forward':274,'training_updates':0,'GPU_seconds':1800,'wall_seconds':3600,'allocated_bytes':18<<30,'new_storage_bytes':512<<20,'cumulative_storage_bytes':30<<30,'return_bytes':5<<20}
POLICY={'arms':['N','S'],'images':548,'batch':4,'dtype':'FP32','canvas':[768,1344],'query':300,'classes':10,'budgets':[600,300],'new_score_groups':4,'evaluator_calls':8,'original_N_recomputed':False,'EMA_only':True,'training_restore':False,'cost_benchmark':False,'A_missing':True,'holdout_access':False,'warmup_forward':0,'automatic_retry':False}
need=ct.need;read=ct.read;sha=ct.sha;digest=ct.digest;ref=ct.reference

def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def bound_file(row):
 p=Path(row['path']);n=checkpoint_bytes(row)
 need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size==n and sha(p)==row['sha256'],'bound physical file '+str(p));return p

def require_v1():
 from sparse_rtdetr.research import sabl_control as sc
 need(sc.VERSION==VERSION and Path(sc.__file__).resolve()==ROOT/'src/sparse_rtdetr/research/sabl_control.py','original v1 only; repaired v2 forbidden')
 return sc

def gates():
 import sabl_results
 tree=ast.parse(inspect.getsource(sabl_results.render));nodes=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='thresholds' for t in n.targets)]
 need(len(nodes)==1,'original threshold definition');return ast.literal_eval(nodes[0])

def ema_model(ep,device='cpu',fixture_model=None):
 import torch
 from sparse_rtdetr.baseline import training_v2b_checkpoint as layout
 from sparse_rtdetr.protocol_r18.model import Config
 from sparse_rtdetr.protocol_r18.engine import tensor_digest
 sc=require_v1();arm=ep['arm'];path=bound_file(ep['checkpoint']);state=torch.load(path,map_location='cpu',weights_only=True)
 need(state['format']=='sparse_rtdetr.sabl_formal24_fullstate_v1' and state['binding']==ep['binding'],'checkpoint schema/binding')
 need(state['binding']['version']==VERSION and state['binding']['arm']==arm and state['binding']['contract']==OLD_CONTRACT and state['binding']['release_manifest']==OLD_MANIFEST and state['binding']['N_SHA']==ct.N_SHA,'original checkpoint source/version/parent')
 need(state['runtime']==str(torch.__version__) and state['device']=='cuda:0','checkpoint runtime identity')
 cfg=dict(sc.TrainConfig().binding());need(state['engine']['config']==cfg and state['engine']['completed_updates']==7296 and state['engine']['phase']==0 and state['engine']['failed'] is False,'e24 training config/clock')
 need(state['ema']['updates']==7296 and state['ema']['decay']==.9999 and state['ema']['warmups']==2000,'EMA clock/config')
 need(state['criterion']['arm']==arm and state['criterion']['version']==VERSION,'supervision identity')
 model=fixture_model if fixture_model is not None else sc.build(arm,Config())[0]
 current=layout._model_layout(model);need(state['model_layout']==current,'model complete layout')
 layout._check_model_state(state['model'],current,'original raw state');layout._check_model_state(state['ema']['module'],current,'original e24 EMA')
 for name,value in model.state_dict().items():
  if name.rsplit('.',1)[-1] in ('anchors','valid_mask','num_points_scale'):need(torch.equal(value,state['ema']['module'][name]),'immutable EMA cache')
 expected=tensor_digest(sorted(state['ema']['module'].items()));raw_digest=tensor_digest(sorted(state['model'].items()))
 model.load_state_dict(state['ema']['module'],strict=True);actual=tensor_digest(sorted(model.state_dict().items()));need(actual==expected,'EMA tensor copy exact')
 model.eval().requires_grad_(False);need(all(not m.training for m in model.modules()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),'eval no gradients')
 report={'arm':arm,'checkpoint':ep['checkpoint'],'EMA_module_SHA':expected,'raw_module_SHA':raw_digest,'EMA_differs_from_raw':expected!=raw_digest,'EMA_loaded_exact':True,'EMA_updates':7296,'model_state_tensors':len(state['ema']['module']),'source_version':VERSION,'source_manifest':OLD_MANIFEST,'optimizer_restored':False,'engine_restored':False,'RNG_restored':False,'device':device}
 del state;gc.collect();model.to(device);return model,report

def endpoint(c,arm):
 import sabl_formal as formal
 folder=FORMAL/'attempt-formal-001'/('train-'+arm);result=formal.closed(folder,c);ep=ct.exact(result['endpoint'])
 binding={'scope':ct.GPU,'arm':arm,'version':VERSION,'contract':digest(c),'release_manifest':OLD_MANIFEST,'N_SHA':ct.N_SHA}
 need(ep['binding']==binding and ep['arm']==arm and ep['contract']==OLD_CONTRACT and ep['manifest']==OLD_MANIFEST,'endpoint source/cohort')
 need(ep['epochs']==24 and ep['updates']==7296 and ep['EMA'] is True and ep['acceptance_state_used'] is False and ep['synthetic'] is False and ep['parent_N_SHA']==ct.N_SHA,'fixed certified e24 EMA')
 need(ep['checkpoint']['sha256']==CP_SHA[arm] and ep['checkpoint']['updates']==7296 and ep['checkpoint']['binding']==binding,'checkpoint endpoint binding')
 need(read(str(ep['checkpoint']['path'])+'.complete.json')==ep['checkpoint'],'complete receipt binding');bound_file(ep['checkpoint'])
 return ep,result,{'closed':ref(folder/'CLOSED.json'),'result':ref(folder/'RESULT.json'),'endpoint':result['endpoint'],'complete':ref(str(ep['checkpoint']['path'])+'.complete.json')}

def verify_baseline(c):
 from sparse_rtdetr.baseline.primary_evaluator import primary_evaluator_contract_binding
 from sparse_rtdetr.r0.evaluation import POLICY as evaluation_policy
 history=c['metadata']['historical_N'];need(history['status']=='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE','historical limitation')
 values={k:ct.exact(history[k]) for k in ('endpoint','authentication','source','original_STOP','score_result','infer_result')}
 need(not Path(history['original_CLOSED_path']).exists(),'historical closure changed')
 need(values['endpoint']['checkpoint']['sha256']==c['metadata']['checkpoint']['sha256']==ct.N_SHA,'original N parent');bound_file(c['metadata']['checkpoint'])
 need(values['infer_result']['images']==548 and values['infer_result']['forward_calls']==137 and values['infer_result']['EMA'] is True and values['infer_result']['fixture_only'] is False,'historical inference profile')
 need(values['score_result']['results']['B0']==history['scores'] and values['score_result']['scope']=='development548' and values['score_result']['fixture_only'] is False,'historical score receipt')
 original_run=WS/'experiments/localization-distribution/r18-sampling-span-20261020-001/formal-execution-003/contract.json'
 need(sha(original_run)=='6bce42c6875747779d1e614309143103c3ce337051eddfa7638e60e7202180b9','historical evaluator run contract')
 old_contract=read(original_run);need(old_contract['contract_sha256']==values['infer_result']['identity']['contract_sha256'] and old_contract['source']==history['source'],'historical inference source binding')
 need(old_contract['data']['development']==c['metadata']['data']['development'],'historical and current exact 548 membership/coordinate roots')
 old={r['path']:r for r in values['source']['files']};shared=['src/sparse_rtdetr/r0/evaluation.py','src/sparse_rtdetr/r0/evaluation_data.py','src/sparse_rtdetr/baseline/primary_evaluator.py','src/sparse_rtdetr/data_protocol/evaluation.py','src/sparse_rtdetr/localization/consumer_geometry.py']
 need(all(sha(ROOT/p)==old[p]['sha256'] for p in shared),'same evaluator/data/raster source')
 binding=primary_evaluator_contract_binding(str(ROOT));profiles={}
 for budget,reference in history['scores'].items():
  score=ct.exact(reference);official=score['official'];coco=score['COCO_bridge']
  need(official['evaluated_image_count']==548 and official['protocol']=='visdrone_official_style_v1' and official['contract_raw_sha256']==binding['config_raw_sha256'] and official['authority_manifest_raw_sha256']==binding['authority_manifest_raw_sha256'],'original N evaluator authority/profile')
  need(coco['protocol']==evaluation_policy['id'] and coco['maxDets']==[1,10,100] and coco['areaRng']==[[0,1e10],[0,1024],[1024,9216],[9216,1e10]] and score['official_APS'] is None,'COCO/area/official APS semantics')
  profiles[budget]={'score':reference,'official_protocol':official['protocol'],'COCO_protocol':coco['protocol'],'COCO_maxDets':coco['maxDets'],'COCO_iouThrs':coco['iouThrs'],'COCO_areaRng':coco['areaRng'],'authority_SHA':official['authority_manifest_raw_sha256'],'evaluator_contract_SHA':official['contract_raw_sha256']}
 return {'reusable':True,'original_eval_contract':ref(original_run),'same_development_spec':True,'status':history['status'],'scores':history['scores'],'profiles':profiles,'shared_source_files':{p:sha(ROOT/p) for p in shared},'no_recomputation':True,'MATLAB_executed':False}

def development_verify(c,physical):
 from sparse_rtdetr.research.sabl_formal_data import development
 from sparse_rtdetr.localization.data import read_bound
 from PIL import Image
 import io
 spec=c['metadata']['data']['development'];ct.exact(spec['manifest']);rows,reader=development(c)
 need(len(rows)==548 and len({r['stable_image_id'] for r in rows})==548 and all(r['split']=='val' for r in rows),'fixed development membership')
 if physical:
  for row in rows:
   reader.raw(row);blob=read_bound(Path(spec['roots']['images'])/Path(row['relative_path']).name,row['image_sha256'],row['image_size_bytes'])
   with Image.open(io.BytesIO(blob)) as image:need(image.size==(row['width'],row['height']),'image dimensions');image.verify()
 return rows,reader

def resources(c):
 import shutil
 import torch,torchvision
 need(Path(sys.executable).resolve()==Path(c['metadata']['python']).resolve(),'certified interpreter')
 env=c['metadata']['environment'];need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'],'environment version')
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();need(len(lines)==1,'single physical GPU')
 uuid,name,pci,free=[v.strip() for v in lines[0].split(',')]
 need(uuid==c['gpu_uuid']=='GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8' and name=='NVIDIA GeForce RTX 4090 D' and pci=='00000000:01:00.0' and int(free)>=18432,'fresh GPU identity/free memory')
 need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing GPU job')
 disk=shutil.disk_usage(TASK);need(disk.free>=30<<30,'original free disk floor 30GiB')
 cumulative=sum(p.stat().st_size for root in TASK.parent.glob('r18-sabl-*') if root.is_dir() for p in root.rglob('*') if p.is_file());need(cumulative+CAPS['new_storage_bytes']<=CAPS['cumulative_storage_bytes'],'cumulative storage including old evidence')
 return {'GPU_uuid':uuid,'name':name,'pci':pci,'free_MiB':int(free),'disk_free_bytes':disk.free,'prior_task_family_storage_bytes':cumulative,'CUDA_initialized':torch.cuda.is_initialized(),'read_only_no_launch':True}

def prepare(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hidden CUDA preparation');sc=require_v1();import torch;torch.set_num_threads(2)
 out=Path(a.out);need(out.parent==TASK and not out.exists(),'fresh preparation');out.mkdir()
 c=read(FORMAL/'contract.json');need(digest(c)==OLD_CONTRACT and c['manifest_sha256']==OLD_MANIFEST,'original contract')
 old_release=FORMAL/'releases/79ba59371c21-formal-001';need(sha(old_release/'RELEASE_MANIFEST.json')==OLD_MANIFEST,'original source manifest')
 old=read(old_release/'RELEASE_MANIFEST.json');shared=[]
 for row in old['files']:
  if row['path'].startswith(('src/','configs/','vendor/')):need(sha(ROOT/row['path'])==row['sha256'],'original scientific bytes');shared.append(row)
 eps={};proofs={};audits={};input_hashes=[]
 for arm in POLICY['arms']:
  ep,r,proof=endpoint(c,arm);eps[arm]=ep;proofs[arm]=proof;input_hashes.append(r['training_input_SHA']);model,audit=ema_model(ep);audits[arm]=audit;del model;gc.collect()
 need(len(set(input_hashes))==1,'N/S ordered train input/GT/augmentation mismatch')
 baseline=verify_baseline(c);rows,_=development_verify(c,True);budget=read(LATEST_COST);resource_proof=resources(c)
 need(budget['cumulative']['forward']==31743 and budget['remaining']['forward']==12846 and budget['remaining']['GPU_stage_seconds']>=CAPS['GPU_seconds'],'current inherited balance')
 refs={'original_parent_N':c['metadata']['checkpoint'],'original_contract':ref(FORMAL/'contract.json'),'original_manifest':ref(old_release/'RELEASE_MANIFEST.json'),'budget':ref(LATEST_COST),'development_manifest':c['metadata']['data']['development']['manifest']}
 protected={str(FORMAL/'attempt-formal-001/STOP.json'):ref(FORMAL/'attempt-formal-001/STOP.json')}
 for arm in ('N','S'):
  for r in proofs[arm].values():protected[r['path']]=r
  protected[eps[arm]['checkpoint']['path']]=eps[arm]['checkpoint']
 for key in ('original_STOP','endpoint','source','authentication','score_result','infer_result'):protected[c['metadata']['historical_N'][key]['path']]=c['metadata']['historical_N'][key]
 for r in [*baseline['scores'].values(),baseline['original_eval_contract']]:protected[r['path']]=r
 d={'schema':SCOPE,'policy':POLICY,'caps':CAPS,'scientific_version':VERSION,'old_scientific_commit':OLD_COMMIT,'old_source_manifest':OLD_MANIFEST,'release':str(ROOT),'python':c['metadata']['python'],'environment':c['metadata']['environment'],'gpu_uuid':c['gpu_uuid'],'refs':refs,'endpoints':eps,'proofs':proofs,'EMA_CPU_audits':audits,'baseline':baseline,'development_ids':[r['stable_image_id'] for r in rows],'development_ids_SHA':digest([r['stable_image_id'] for r in rows]),'ordered_training_input_SHA':input_hashes[0],'gates':gates(),'scientific_files':shared,'protected':protected,'resource_proof':resource_proof,'prior':budget['cumulative'],'remaining':budget['remaining'],'only_two_questions':True,'overall_three_arm_verdict':'INCOMPLETE_MISSING_A_AND_COST'}
 need(not torch.cuda.is_initialized(),'CPU preparation initialized CUDA');write(out/'DEFINITION.json',d);print(json.dumps({'status':'PASS_CPU_REAL_EMA_AND_BASELINE_ADMISSION','definition_SHA':sha(out/'DEFINITION.json'),'EMA':audits,'baseline_reusable':True,'development_members_verified':548,'same_training_input_SHA':True,'CUDA_initialized':False,'forward':0,'new_scores':0}))

def compare_partial(scores):
 from sabl_results import metrics
 need(set(scores)=={'N24','S24','N_original'} and all(set(x)=={'600','300'} for x in scores.values()),'only two controls and two budgets')
 rows={arm:{b:metrics(score) for b,score in by_budget.items()} for arm,by_budget in scores.items()};differences={}
 for label,right in [('S-N24','N24'),('S-originalN','N_original')]:differences[label]={b:{k:None if rows['S24'][b][k] is None or rows[right][b][k] is None else rows['S24'][b][k]-rows[right][b][k] for k in rows['S24'][b]} for b in ('600','300')}
 checks={b:{k:None if differences['S-N24'][b][k] is None else differences['S-N24'][b][k]>=threshold for k,threshold in limits.items()} for b,limits in gates().items()}
 passed=all(value is True for budget in checks.values() for value in budget.values());unknown=any(value is None for budget in checks.values() for value in budget.values())
 negatives={b:[k for k,v in differences['S-originalN'][b].items() if v is not None and v<0] for b in ('600','300')}
 context='相对控制改善，原N净变化须逐项报告' if passed else '相对同期N的原条件未通过或存在UNKNOWN'
 if passed and any(differences['S-originalN'][b]['COCO_'+k] is not None and differences['S-originalN'][b]['COCO_'+k]<0 for b in ('600','300') for k in ('AP','APS','AP75_small')):context='相对控制改善，尚未形成现有基线净收益'
 result={'absolute_percent':rows,'differences_pp':differences,'S_N24_original_conditions':checks,'S_N24_conditions_pass':None if unknown else passed,'S_originalN_negative_metrics':negatives,'interpretation':context,'S_A24':'UNKNOWN_MISSING_A','full_original_quality_controls_pass':None,'cost_gate':{'passed':None,'measured':False,'original_limits':{'E2E_S_over_N':1.30,'allocated_S_over_N':1.20}},'overall_three_arm_verdict':'INCOMPLETE_MISSING_A_AND_COST','spatial_arrangement_mechanism':'NOT_IDENTIFIED_WITHOUT_A24','new_gate_added':False,'original_N_status':'AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE'}
 keys=list(rows['N24']['600']);table=['|budget|arm/comparison|'+'|'.join(keys)+'|','|---|---|'+'---:|'*len(keys)]
 for collection in (rows,differences):
  for arm,values in collection.items():
   for b in ('600','300'):table.append('|'+b+'|'+arm+'|'+'|'.join('UNKNOWN' if values[b][k] is None else '%.8f'%values[b][k] for k in keys)+'|')
 return result,'\n'.join(table)+'\n'

class Meter:
 def __init__(self,out,prior,fixture=False):
  self.out=out;self.prior=prior;self.fixture=fixture;self.start=time.monotonic();self.count=0;self.peak=0;self.cuda=None;self.limit=4 if fixture else 274
 def snapshot(self):return {'prior':self.prior,'training_optimizer_created':False,'new_forward':self.count,'total_forward':self.prior['forward']+self.count,'remaining_forward':44589-self.prior['forward']-self.count,'total_GPU_stage_seconds':self.prior['GPU_stage_seconds']+(0 if self.fixture else time.monotonic()-self.start),'remaining_GPU_stage_seconds':86400-self.prior['GPU_stage_seconds']-(0 if self.fixture else time.monotonic()-self.start),'training_updates':0,'new_stage_seconds':0 if self.fixture else time.monotonic()-self.start,'CPU_fixture_seconds':time.monotonic()-self.start if self.fixture else None,'peak_allocated_bytes':self.peak,'peak_status':getattr(self,'peak_status','NOT_INITIALIZED'),'cache_paths':[] if self.fixture else [str(TASK/'compiler-cache')]}
 def publish(self):
  from sabl_formal import atomic
  atomic(self.out/'ledger.json',self.snapshot())
 def sample(self):
  if self.cuda is not None:
   try:self.peak=max(self.peak,self.cuda.max_memory_allocated());self.peak_status='MEASURED'
   except BaseException:self.peak_status='UNKNOWN_FINAL_SAMPLE_FAILED'
 def guard(self):
  self.sample()
  need(time.monotonic()-self.start<(300 if self.fixture else CAPS['GPU_seconds']),'stage time cap')
  size=sum(p.stat().st_size for p in (self.out if self.fixture else TASK).rglob('*') if p.is_file());need(size<=CAPS['new_storage_bytes'],'new storage cap')
  if not self.fixture:
   cumulative=sum(p.stat().st_size for root in TASK.parent.glob('r18-sabl-*') if root.is_dir() for p in root.rglob('*') if p.is_file());need(cumulative<=CAPS['cumulative_storage_bytes'],'cumulative original 30GiB cap')
  if self.cuda is not None:self.peak=max(self.peak,self.cuda.max_memory_allocated());need(self.peak<=CAPS['allocated_bytes'],'allocated cap')
 def forward(self):self.guard();need(self.count<self.limit,'forward budget');self.count+=1;self.publish()


def execute(d,out,meter,fixture=False,fault='none'):
 import torch
 import sabl_formal as formal
 from sparse_rtdetr.research import sabl_formal_data as data
 from sparse_rtdetr.r0.evaluation import score_budget
 c=None if fixture else ct.exact(d['refs']['original_contract']);scores={};prediction_refs={};proofs={}
 reader=data.Fixture() if fixture else None
 if fixture:rows=reader.records
 else:rows,reader=development_verify(c,False)
 for arm in ('N','S'):
  if fixture:model=fixture_predictor(arm);audit={'EMA_loaded_exact':True,'scope':'SYNTHETIC_NOT_SCIENTIFIC'}
  else:model,audit=ema_model(d['endpoints'][arm],'cuda')
  from sparse_rtdetr.protocol_r18.engine import tensor_digest
  before=tensor_digest(sorted(model.state_dict().items()));folder=out/('infer-'+arm);folder.mkdir();pred,raw,invalid=formal.predictions(c,model,rows,reader,'cpu' if fixture else 'cuda',meter)
  need([k for batch in raw for k in batch['ids']]==[r['stable_image_id'] for r in rows],'prediction coverage/order')
  if not fixture:need([r['stable_image_id'] for r in rows]==d['development_ids'],'fixed development IDs')
  prediction_refs[arm]={}
  for budget,values in pred.items():path=folder/('predictions-%d.json'%budget);write(path,values);prediction_refs[arm][str(budget)]=ref(path)
  need(tensor_digest(sorted(model.state_dict().items()))==before and all(p.grad is None for p in model.parameters()),'evaluation state or parameter gradients changed')
  proofs[arm]={'eval_state_unchanged':True,'gradients_none':True,'EMA':audit,'invalid_ordinary_queries':invalid,'forward_calls':(len(rows)+3)//4};write(folder/'PROVENANCE.json',proofs[arm]);del model,raw,pred;gc.collect()
  if not fixture:torch.cuda.empty_cache()
 if fault=='score_error':raise RuntimeError('SYNTHETIC_SCORE_ROUTE_FAILURE')
 truth=None if fixture else [reader.raw(row) for row in rows];score_out=out/'score';score_out.mkdir()
 for arm in ('N','S'):
  scores[arm+'24']={}
  for budget in ('600','300'):
   pred=ct.exact(prediction_refs[arm][budget]);need(all(p['image_id'] in {r['stable_image_id'] for r in rows} for p in pred),'unknown image ID')
   score=synthetic_score() if fixture else score_budget(rows,truth,pred,str(ROOT))
   score['limited_evaluation_binding']={'scope':'SYNTHETIC_NOT_SCIENTIFIC' if fixture else SCOPE,'arm':arm,'budget':int(budget),'EMA_only':True,'endpoint':None if fixture else d['proofs'][arm]['endpoint'],'scientific_version':VERSION,'old_source_manifest':OLD_MANIFEST,'development_ids_SHA':digest([r['stable_image_id'] for r in rows]),'predictions':prediction_refs[arm][budget]}
   write(score_out/(arm+'-'+budget+'.json'),score);scores[arm+'24'][budget]=score;meter.guard()
 scores['N_original']={b:synthetic_score() if fixture else ct.exact(d['baseline']['scores'][b]) for b in ('600','300')}
 comparison,table=compare_partial(scores);write(out/'COMPARISON.json',comparison);(out/'TABLE.md').write_text(('# SYNTHETIC_NOT_SCIENTIFIC\n' if fixture else '# Original-v1 N/S e24 limited comparison\n')+'\n'+table)
 if not fixture:
  for row in d['protected'].values():bound_file(row)
  write(out/'PROTECTED_AFTER.json',{'status':'UNCHANGED_SHA','files':list(d['protected'].values())})
 need(meter.count==(4 if fixture else 274),'exact planned forwards');meter.guard();meter.publish()
 result={'status':'SYNTHETIC_CLOSED_NOT_SCIENTIFIC' if fixture else 'PASS_LIMITED_NS_EVALUATION_ENGINEERING','scope':'SYNTHETIC_NOT_SCIENTIFIC' if fixture else SCOPE,'source_version':VERSION,'cost':meter.snapshot(),'models':proofs,'new_score_groups':0 if fixture else 4,'evaluator_calls':0 if fixture else 8,'mock_score_groups':4 if fixture else 0,'original_N_recomputed':False,'A_evaluated':False,'cost_benchmark_executed':False,'overall_three_arm_verdict':'INCOMPLETE_MISSING_A_AND_COST','comparison':ref(out/'COMPARISON.json'),'no_followup':True}
 write(out/'RESULT.json',result);write(out/'CLOSED.json',{'status':result['status'],'result_sha256':sha(out/'RESULT.json')});return result

def stop(out,meter,error):
 meter.sample()
 try:meter.guard()
 except BaseException:pass
 try:meter.publish()
 except BaseException:pass
 write(out/'STOP.json',{'status':'STOP_LIMITED_NS_EVALUATION','error':repr(error),'cost':meter.snapshot(),'no_retry':True});(out/'stderr.txt').write_text(traceback.format_exc())
 print(json.dumps({'status':'STOP_LIMITED_NS_EVALUATION','error':repr(error),'cost':meter.snapshot()}));return 1

def packet(a,check=False):
 d=read(a.definition);m=read(a.manifest)
 need(d['schema']==SCOPE and d['policy']==POLICY and d['caps']==CAPS and d['scientific_version']==VERSION and Path(d['release'])==ROOT,'limited definition/source')
 need(Path.cwd()==ROOT and Path(sys.executable).resolve()==Path(d['python']).resolve(),'actual cwd/interpreter')
 need(len(m['files'])>0 and len({r['path'] for r in m['files']})==len(m['files']),'source duplicate paths')
 for r in m['files']:
  p=Path(r['path']);need(not p.is_absolute() and '..' not in p.parts and (ROOT/p).stat().st_size==r['bytes'] and sha(ROOT/p)==r['sha256'],'frozen source bytes')
 need(m['scope']==SCOPE and m['scientific_version']==VERSION and m['scientific_commit']==OLD_COMMIT and m['scientific_manifest_sha256']==OLD_MANIFEST,'frozen scope');require_v1()
 for row in d['refs'].values():bound_file(row)
 for row in d['protected'].values():bound_file(row)
 for row in d['scientific_files']:need(sha(ROOT/row['path'])==row['sha256'],'scientific bytes changed')
 need(d['gates']==json.loads(json.dumps(gates())) and d['old_source_manifest']==OLD_MANIFEST and d['old_scientific_commit']==OLD_COMMIT,'gates/scientific identity')
 need(read(LATEST_COST)['cumulative']==d['prior'] and read(LATEST_COST)['remaining']==d['remaining'],'no budget reset')
 need(Path(a.out)==TASK/'attempt-evaluate-001' and not Path(a.out).exists() and not (TASK/'EVAL_ONCE.json').exists(),'unique attempt')
 expected={'scope':SCOPE,'commit':m['commit'],'manifest_sha256':sha(a.manifest),'definition_sha256':sha(a.definition),'attempt':a.out,'authorize_once':True}
 if not check:need(digest(read(a.authorization))==digest(expected),'explicit frozen authorization')
 return d,m,expected

def bind(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hidden CUDA bind');d,m,expected=packet(a,True)
 need(a.approve_commit==m['commit'] and a.approve_manifest_sha256==sha(a.manifest) and a.authorize_evaluation,'explicit limited evaluation scope approval')
 need(Path(a.authorization)==TASK/'authorization.json' and not Path(a.authorization).exists(),'fresh authorization')
 if not a.check_only:write(a.authorization,expected)
 print(json.dumps({'status':'BIND_CHECK_ONLY' if a.check_only else 'BOUND_NS_EVALUATION','authorization_created':not a.check_only,'GPU':False}));return 0

def run(a):
 d,m,auth=packet(a,a.check_only)
 if a.check_only:need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hidden CUDA check');print(json.dumps({'status':'PASS_CPU_CALLER_ADMISSION','GPU':False,'scoring':False}));return 0
 write(TASK/'EVAL_ONCE.json',auth);out=Path(a.out);out.mkdir();meter=Meter(out,d['prior'])
 try:
  import torch,torchvision
  need(os.environ.get('CUDA_VISIBLE_DEVICES')==d['gpu_uuid'],'GPU binding')
  need(str(torch.__version__)==d['environment']['torch'] and str(torchvision.__version__)==d['environment']['torchvision'] and torch.version.cuda==d['environment']['cuda_build'],'runtime')
  need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'existing GPU job')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  need(os.environ.get('TORCHINDUCTOR_CACHE_DIR')==str(TASK/'compiler-cache/inductor') and os.environ.get('TRITON_CACHE_DIR')==str(TASK/'compiler-cache/triton'),'compiler cache accounting')
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('limited evaluation deadline')));signal.alarm(CAPS['GPU_seconds']);signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('SIGTERM')))
  torch.cuda.set_per_process_memory_fraction(CAPS['allocated_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats();meter.cuda=torch.cuda
  result=execute(d,out,meter);print(json.dumps({'status':result['status'],'cost':result['cost']}));return 0
 except BaseException as e:return stop(out,meter,e)
 finally:signal.alarm(0)

def synthetic_score():
 import sabl_results
 return {'official':{k:-1. for k in sabl_results.PRIMARY},'COCO_bridge':{'metrics':{k:-1. for k in sabl_results.COCO}},'synthetic_score_stub_no_AP_computation':True}

def fixture_predictor(arm):
 import torch
 class Predictor(torch.nn.Module):
  def __init__(self):
   super().__init__();self.arm=arm
   from types import SimpleNamespace
   self.decoder=SimpleNamespace(decoder=SimpleNamespace(layers=[]))
  def forward(self,images):
   batch=len(images);return {'pred_logits':torch.linspace(-2,2,3000).reshape(1,300,10).repeat(batch,1,1),'pred_boxes':torch.tensor([.5,.5,.1,.1]).reshape(1,1,4).repeat(batch,300,1)}
 return Predictor().eval()

def fixture(a):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixture no CUDA');out=Path(a.out);need(not out.exists(),'fixture duplicate');out.mkdir();meter=Meter(out,{'forward':31743,'GPU_stage_seconds':36543.70207419119},True)
 if a.fault=='forward_cap':meter.limit=1
 try:
  result=execute(None,out,meter,True,a.fault);import torch;need(not torch.cuda.is_initialized(),'CPU initialized CUDA');print(json.dumps({'status':result['status'],'cost':result['cost'],'scientific_scores':0,'detector_forward':0}));return 0
 except BaseException as e:return stop(out,meter,e)


def export(a):
 import zipfile
 out=Path(a.out);need(out.is_dir() and (out/'CLOSED.json').exists() != (out/'STOP.json').exists(),'exactly one terminal state')
 names=['RESULT.json','CLOSED.json','STOP.json','ledger.json','stderr.txt','COMPARISON.json','TABLE.md','infer-N/PROVENANCE.json','infer-S/PROVENANCE.json','score/N-600.json','score/N-300.json','score/S-600.json','score/S-300.json','EXIT.json','PROTECTED_AFTER.json']
 files=[out/n for n in names if (out/n).is_file()];target=Path(a.archive);need(not target.exists() and target.parent==out.parent,'fresh local export')
 need(sum(p.stat().st_size for p in files)<CAPS['return_bytes']-262144,'return reserve incl stdout/stderr')
 with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(out).as_posix())
  z.writestr('RETURN_INDEX.json',json.dumps([ref(p) for p in files],indent=2))
 need(target.stat().st_size<CAPS['return_bytes']-262144,'compressed return size')
 print(json.dumps({'export':ref(target),'files':len(files),'stdout_stderr_reserve_bytes':262144,'large_predictions_returned':False}));return 0

def launch(a):
 begin=time.monotonic()
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-first caller')
 manifest=read(a.manifest);need(manifest['commit']==a.approve_commit and sha(a.manifest)==a.approve_manifest_sha256 and a.authorize_evaluation,'explicit fixed authorization')
 need(Path(a.manifest)==ROOT/'RELEASE_MANIFEST.json' and Path.cwd()==ROOT,'fixed installed release')
 # Never create an attempt or authorization in CPU-only inspection.
 if a.check_only:
  args=argparse.Namespace(definition=a.definition,manifest=a.manifest,authorization=str(TASK/'authorization.json'),out=str(TASK/'attempt-evaluate-001'),check_only=True)
  packet(args,True);print(json.dumps({'status':'PASS_CPU_FINAL_CALLER_CHECK','GPU':False,'authorization_created':False,'attempt_created':False}));return 0
 need(not (TASK/'attempt-evaluate-001').exists() and not (TASK/'EVAL_ONCE.json').exists() and not (TASK/'authorization.json').exists(),'no repeated launch')
 fresh=TASK/'fresh-evaluation-admission-001';prepare(argparse.Namespace(out=str(fresh)))
 args=argparse.Namespace(definition=str(fresh/'DEFINITION.json'),manifest=a.manifest,authorization=str(TASK/'authorization.json'),out=str(TASK/'attempt-evaluate-001'),approve_commit=a.approve_commit,approve_manifest_sha256=a.approve_manifest_sha256,authorize_evaluation=True,check_only=False)
 bind(args)
 env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=read(args.definition)['gpu_uuid'];env['TORCHINDUCTOR_CACHE_DIR']=str(TASK/'compiler-cache/inductor');env['TRITON_CACHE_DIR']=str(TASK/'compiler-cache/triton');env['XDG_CACHE_HOME']=str(TASK/'compiler-cache/xdg');env['CUDA_CACHE_PATH']=str(TASK/'compiler-cache/cuda')
 worker_begin=time.monotonic()
 cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'run','--definition',args.definition,'--manifest',args.manifest,'--authorization',args.authorization,'--out',args.out]
 try:
  with (TASK/'GPU-stdout.txt').open('xb') as stdout,(TASK/'GPU-stderr.txt').open('xb') as stderr:
   process=subprocess.run(cmd,env=env,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=CAPS['wall_seconds']-(time.monotonic()-begin))
 except BaseException as error:
  write(TASK/'LAUNCH_STOP.json',{'error':repr(error),'no_retry':True});raise
 elapsed=time.monotonic()-worker_begin;ledger=read(Path(args.out)/'ledger.json');previous=read(args.definition)['prior'];new_count=ledger['new_forward'];debit=max(elapsed,ledger['new_stage_seconds'])
 write(Path(args.out)/'EXIT.json',{'returncode':process.returncode,'cost':{'previous':previous,'new':{'forward':new_count,'conservative_GPU_stage_seconds':debit},'cumulative':{'forward':previous['forward']+new_count,'GPU_stage_seconds':previous['GPU_stage_seconds']+debit},'remaining':{'forward':44589-previous['forward']-new_count,'GPU_stage_seconds':86400-previous['GPU_stage_seconds']-debit}},'authoritative_exit_cost':True,'no_retry':True})
 need(sum((TASK/n).stat().st_size for n in ('GPU-stdout.txt','GPU-stderr.txt'))<=262144,'stdout/stderr return reserve exceeded')
 export(argparse.Namespace(out=args.out,archive=str(TASK/'TERMINAL_RETURN.zip')))
 print(json.dumps({'returncode':process.returncode,'no_retry':True,'terminal_archive':ref(TASK/'TERMINAL_RETURN.zip')}));return process.returncode

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare');q.add_argument('--out',required=True)
 for name in ('bind','run'):
  q=sub.add_parser(name)
  for key in ('definition','manifest','authorization','out'):q.add_argument('--'+key,required=True)
  q.add_argument('--check-only',action='store_true')
  if name=='bind':q.add_argument('--approve-commit',required=True);q.add_argument('--approve-manifest-sha256',required=True);q.add_argument('--authorize-evaluation',action='store_true')
 q=sub.add_parser('export');q.add_argument('--out',required=True);q.add_argument('--archive',required=True)
 q=sub.add_parser('launch');q.add_argument('--definition',required=True);q.add_argument('--manifest',required=True);q.add_argument('--approve-commit',required=True);q.add_argument('--approve-manifest-sha256',required=True);q.add_argument('--authorize-evaluation',action='store_true');q.add_argument('--check-only',action='store_true')
 q=sub.add_parser('fixture');q.add_argument('--out',required=True);q.add_argument('--fault',choices=('none','score_error','forward_cap'),default='none')
 a=p.parse_args();return {'prepare':prepare,'bind':bind,'run':run,'fixture':fixture,'export':export,'launch':launch}[a.command](a)
if __name__=='__main__':raise SystemExit(main())
