"""CPU-only real parent initialization and N24 endpoint reuse qualification."""
import os,sys,json,subprocess,gc,copy,time,ast,resource,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_DATA,(4<<30,4<<30));signal.alarm(300)
import torch
from sparse_rtdetr.research import sabl_control as sc
from sparse_rtdetr.protocol_r18.engine import tensor_digest
from sparse_rtdetr.protocol_r18.checkpoint import capture_rng,file_sha
from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
import sabl_formal
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace');OLD=WS/'experiments/localization-distribution/r18-sabl-ns-e24-evaluation-20261028-001/release-68d990fc7b90';FORMAL=WS/'experiments/localization-distribution/r18-sabl-formal24-20261027-001';LATEST=WS/'experiments/localization-distribution/r18-sabl-ns-e24-evaluation-20261028-001/COST_ADDITION.json'

def main(dest):
 out=Path(dest);out.mkdir();start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=[]
 def check(name,ok):
  checks.append({'name':name,'pass':bool(ok)})
  if not ok:raise AssertionError(name)
 code="""import sys,json,gc
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd()/'tools')]
import sabl_ns_eval as e
from sparse_rtdetr.research import sabl_control as sc
c=e.read(e.FORMAL/'contract.json');ep,r,proof=e.endpoint(c,'N');model,audit=e.ema_model(ep,'cpu');scores={}
for b in ('600','300'):
 ref=e.ref(e.TASK/'attempt-evaluate-001/score'/('N-'+b+'.json'))
 v=e.ct.exact(ref);binding=v['limited_evaluation_binding'];assert binding['arm']=='N' and binding['EMA_only'] and binding['endpoint']==proof['endpoint'];scores[b]=ref
initial=e.read(e.FORMAL/'attempt-formal-001/train-N/INITIAL.json')
for role in ('train_core','development'):
 spec=c['metadata']['data'][role];material=e.ct.exact(spec['manifest']);assert len(material['records'])==spec['count']
print(json.dumps({'endpoint':ep,'EMA_CPU_audit':audit,'proof':proof,'scores':scores,'initial':initial,'training_input_SHA':r['training_input_SHA'],'old_train_binding':sc.TrainConfig().binding(),'parent_N':c['metadata']['checkpoint'],'data':c['metadata']['data'],'source_manifest':c['manifest_sha256'],'current_cost':e.ref(e.WS/'experiments/localization-distribution/r18-sabl-ns-e24-evaluation-20261028-001/COST_ADDITION.json'),'CUDA_initialized':__import__('torch').cuda.is_initialized()}))
"""
 child=subprocess.run([sys.executable,'-B','-c',code],cwd=OLD,capture_output=True,timeout=300);(out/'old-N24-admission.stdout').write_bytes(child.stdout);(out/'old-N24-admission.stderr').write_bytes(child.stderr);check('actual original-v1 endpoint/EMA/score/data admission',child.returncode==0)
 old=json.loads(child.stdout);new=sc.TrainConfig().binding();oldcfg=dict(old['old_train_binding']);newcfg=dict(new);oldcfg.pop('arm');newcfg.pop('arm');check('same 24-epoch/seed/input/optimizer/schedule/precision config except version identity',json.loads(json.dumps(newcfg))==oldcfg)
 before=(FORMAL/'releases/79ba59371c21-formal-001/src/sparse_rtdetr/research/sabl_control.py').read_text();after=(ROOT/'src/sparse_rtdetr/research/sabl_control.py').read_text();a=ast.parse(before);b=ast.parse(after)
 def fingerprint(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),include_attributes=False)
 check('N Model and Criterion calculation AST unchanged',fingerprint(a,'Model')==fingerprint(b,'Model') and fingerprint(a,'Criterion')==fingerprint(b,'Criterion'))
 check('Engine rate/health/update AST unchanged',fingerprint(a,'Engine')==fingerprint(b,'Engine'))
 parent=old['parent_N'];check('certified common parent physical SHA',file_sha(parent['path'])==parent['sha256']==sc.N_SHA)
 checkpoint=torch.load(parent['path'],map_location='cpu',weights_only=True);expected=tensor_digest(checkpoint['ema']['module'].items());del checkpoint;gc.collect();init={};rng0=None
 for arm in ('S','A'):
  sabl_formal.seed('cpu');e=sc.make_engine(arm,False,parent['path'],'cpu');sabl_formal.seed('cpu');rng=capture_rng(torch.device('cpu'))
  check(arm+' actual production factory fresh state',not e.optimizer.state and e.completed_updates==e.ema.updates==0 and e.config.epochs==24 and e.config.amp_dtype=='bfloat16' and tuple(e.config.input_hw)==(768,1344) and e.config.physical_batch_size==8)
  native=tensor_digest(e.model.native.state_dict().items());check(arm+' parent N EMA native/BN/cache exact',native==expected==old['initial']['native_SHA']);check(arm+' fresh EMA is this model',_equal(e.model.state_dict(),e.ema.module.state_dict()))
  init[arm]={'native_SHA':native,'head_SHA':tensor_digest(e.model.boundary.state_dict().items()),'optimizer_entries':len(e.optimizer.state),'EMA_updates':e.ema.updates,'RNG_seed':0,'Config':e.config.binding(),'parameter_count':sum(p.numel() for p in e.model.parameters())}
  if rng0 is None:rng0=copy.deepcopy(rng)
  else:check('S/A independent RNG reset identical',_equal(rng0,rng))
  del e,rng;gc.collect()
 check('S/A additional head initialization identical',init['S']['head_SHA']==init['A']['head_SHA'])
 check('N24 is original e24/7296 EMA, no acceptance state',old['endpoint']['epochs']==24 and old['endpoint']['updates']==7296 and old['endpoint']['EMA'] and not old['endpoint']['acceptance_state_used'])
 check('CUDA remains hidden/uninitialized',not torch.cuda.is_initialized() and old['CUDA_initialized'] is False)
 result={'status':'PASS_CPU_N24_REUSE_AND_REAL_PARENT_FRESH_SA','N24_reuse':'YES_ENDPOINT_EMA_AND_CERTIFIED_SAME_PROTOCOL_SCORES_ONLY','do_not_resume_old_N_state_as_v3':True,'reason':['unchanged native N path verified AST and synthetic full-output bytes in model test','same certified parent and full 24-epoch training recipe','N24 source/EMA/548 score identities revalidated by original loader','incorrect encode and stable norm are S/A-only paths'],'old_N24':old,'new_production_SA_initialization':init,'checks':checks,'CPU_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'detector_forward':0,'optimizer_updates':0,'AP_AR_recomputed':0,'GPU_forward':0,'budget_unchanged':json.loads(LATEST.read_text())['remaining']};(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('old_N24','new_production_SA_initialization','checks')}))
if __name__=='__main__':main(sys.argv[1])
