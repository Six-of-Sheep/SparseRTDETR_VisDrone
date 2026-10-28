"""CPU-only tests of the actual limited N/S evaluation entry, never real scoring."""
import argparse,copy,hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import sabl_ns_eval as e
from sparse_rtdetr.baseline import training_v2b_checkpoint as layout
from sparse_rtdetr.research import sabl_control as sc
from sabl_results import PRIMARY,COCO

def main():
 p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();checks=[]
 def ok(name,condition=True):
  if not condition:raise AssertionError(name)
  checks.append(name)
 def rejects(name,fn):
  try:fn()
  except (ValueError,RuntimeError,AssertionError,KeyError,TypeError):ok(name);return
  raise AssertionError('accepted invalid '+name)
 def cli(args,rc=0):
  run=subprocess.run([sys.executable,'-B',str(ROOT/'tools/sabl_ns_eval.py'),*args],cwd=ROOT,env=dict(os.environ),capture_output=True)
  tag='cli-%02d'%len(checks);(out/(tag+'.stdout')).write_bytes(run.stdout);(out/(tag+'.stderr')).write_bytes(run.stderr)
  ok('actual CLI '+str(args[:1])+' rc '+str(rc),run.returncode==rc)
  return run
 torch.set_num_threads(2)
 # Constructed state, real complete-layout and original EMA-only load code.
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.]));self.register_buffer('counter',torch.tensor(0))
 m=Tiny();binding={'scope':e.ct.GPU,'arm':'N','version':e.VERSION,'contract':e.OLD_CONTRACT,'release_manifest':e.OLD_MANIFEST,'N_SHA':e.ct.N_SHA}
 state={'format':'sparse_rtdetr.sabl_formal24_fullstate_v1','binding':binding,'runtime':str(torch.__version__),'device':'cuda:0','model_layout':layout._model_layout(m),'model':dict(copy.deepcopy(m.state_dict())),'ema':{'module':{'weight':torch.tensor([2.]),'counter':torch.tensor(0)},'updates':7296,'decay':.9999,'warmups':2000},'engine':{'config':sc.TrainConfig().binding(),'completed_updates':7296,'phase':0,'failed':False},'criterion':{'arm':'N','version':e.VERSION}}
 def load(value):
  path=out/'SYNTHETIC-state.pt';torch.save(value,path);ep={'arm':'N','binding':binding,'checkpoint':e.ref(path)};return e.ema_model(ep,fixture_model=Tiny())
 loaded,audit=load(state);ok('EMA chosen, raw ignored',loaded.weight.item()==2 and audit['EMA_loaded_exact'] and audit['EMA_differs_from_raw']);ok('eval frozen no gradient',not loaded.training and not loaded.weight.requires_grad and loaded.weight.grad is None)
 for name,mutate in [('wrong version',lambda v:v['binding'].update(version='v2')),('wrong parent',lambda v:v['binding'].update(N_SHA='0'*64)),('wrong epoch clock',lambda v:v['engine'].update(completed_updates=7295)),('wrong EMA clock',lambda v:v['ema'].update(updates=7295)),('nonfinite EMA',lambda v:v['ema']['module'].update(weight=torch.tensor([float('nan')]))),('wrong EMA layout',lambda v:v['ema']['module'].update(weight=torch.ones(2))),('missing EMA tensor',lambda v:v['ema']['module'].pop('counter'))]:
  broken=copy.deepcopy(state);mutate(broken);rejects(name,lambda:load(broken))
 raw=out/'bound.bin';raw.write_bytes(b'abc');reference=e.ref(raw);ok('bytes reference',e.bound_file(reference)==raw);legacy=dict(reference);legacy['size_bytes']=legacy.pop('bytes');ok('legacy size_bytes reference',e.bound_file(legacy)==raw)
 for name,v in [('missing size',{k:v for k,v in reference.items() if k!='bytes'}),('conflicting size',dict(reference,size_bytes=4)),('bool size',dict(reference,bytes=True)),('wrong SHA',dict(reference,sha256='0'*64))]:rejects(name,lambda v=v:e.bound_file(v))
 # Actual subprocess, cwd, interpreter and prediction/budget/terminal/export routing.
 fixture=out/'fixture';cli(['fixture','--out',str(fixture)]);result=e.read(fixture/'RESULT.json');ok('4 synthetic batch calls; zero real forward/score',result['cost']['new_forward']==4 and result['new_score_groups']==0 and result['evaluator_calls']==0);ok('two budgets share forward',result['mock_score_groups']==4);ok('closed exact result SHA',e.read(fixture/'CLOSED.json')['result_sha256']==e.sha(fixture/'RESULT.json'));ok('no fake gain',all(v is None for arm in e.read(fixture/'COMPARISON.json')['differences_pp'].values() for b in arm.values() for v in b.values()))
 cli(['fixture','--out',str(fixture)],1)
 for fault,expected in [('forward_cap',1),('score_error',4)]:
  target=out/fault;cli(['fixture','--out',str(target),'--fault',fault],1);ok('STOP cost retained '+fault,e.read(target/'STOP.json')['cost']['new_forward']==expected and not (target/'CLOSED.json').exists())
 archive=out/'return.zip';cli(['export','--out',str(fixture),'--archive',str(archive)]);ok('small export',archive.stat().st_size<1<<20)
 cli(['export','--out',str(fixture),'--archive',str(archive)],1)
 # Pure constructed metrics: preserve every negative/sentinel and missing A/cost.
 def score(x):return {'official':{k:x*100 for k in PRIMARY},'COCO_bridge':{'metrics':{k:x for k in COCO}},'SYNTHETIC_NOT_SCIENTIFIC':True}
 values={'N24':{b:score(.20) for b in ('600','300')},'S24':{b:score(.21) for b in ('600','300')},'N_original':{b:score(.22) for b in ('600','300')}}
 comparison,table=e.compare_partial(values);ok('old baseline negative retained',comparison['differences_pp']['S-originalN']['600']['COCO_AP']<0);ok('net benefit limitation',comparison['interpretation']=='相对控制改善，尚未形成现有基线净收益');ok('A unknown, no complete quality verdict',comparison['S_A24']=='UNKNOWN_MISSING_A' and comparison['full_original_quality_controls_pass'] is None);ok('cost unknown unchanged gate',comparison['cost_gate']['passed'] is None and comparison['cost_gate']['original_limits']=={'E2E_S_over_N':1.30,'allocated_S_over_N':1.20});ok('both comparisons present',set(comparison['differences_pp'])=={'S-N24','S-originalN'});ok('all metrics in table',all(k in table for k in e.metrics(values['N24']['600'])) if hasattr(e,'metrics') else 'primary_AP' in table)
 # Registered original production admission parser, isolated paths and real source hashes.
 from types import SimpleNamespace
 old_task,old_cost=e.TASK,e.LATEST_COST;e.TASK=out/'isolated-packet';e.TASK.mkdir();e.LATEST_COST=e.TASK/'prior.json'
 prior={'forward':31743,'GPU_stage_seconds':36543.70207419119};remaining={'forward':12846,'GPU_stage_seconds':49856.29792580881};e.write(e.LATEST_COST,{'cumulative':prior,'remaining':remaining})
 source='tools/sabl_ns_eval.py';source_ref={'path':source,'bytes':(ROOT/source).stat().st_size,'sha256':e.sha(ROOT/source)}
 definition={'schema':e.SCOPE,'policy':e.POLICY,'caps':e.CAPS,'scientific_version':e.VERSION,'old_source_manifest':e.OLD_MANIFEST,'old_scientific_commit':e.OLD_COMMIT,'release':str(ROOT),'python':sys.executable,'refs':{'prior':e.ref(e.LATEST_COST)},'protected':{'test':reference},'scientific_files':[source_ref],'prior':prior,'remaining':remaining,'gates':json.loads(json.dumps(e.gates()))}
 definition_path=e.TASK/'definition.json';manifest_path=e.TASK/'manifest.json';e.write(definition_path,definition);manifest={'scope':e.SCOPE,'scientific_version':e.VERSION,'scientific_commit':e.OLD_COMMIT,'scientific_manifest_sha256':e.OLD_MANIFEST,'commit':'SYNTHETIC_NOT_FROZEN','files':[source_ref]};e.write(manifest_path,manifest)
 args=SimpleNamespace(definition=str(definition_path),manifest=str(manifest_path),authorization=str(e.TASK/'authorization.json'),out=str(e.TASK/'attempt-evaluate-001'))
 e.packet(args,True);ok('actual packet check, no attempt or authorization',not Path(args.out).exists() and not Path(args.authorization).exists())
 for name,alter in [('manifest duplicate paths',lambda v:v['files'].append(source_ref)),('manifest wrong scientific version',lambda v:v.update(scientific_version='v2')),('manifest damaged source',lambda v:v['files'][0].update(sha256='0'*64))]:
  damaged=copy.deepcopy(manifest);alter(damaged);manifest_path.write_text(json.dumps(damaged));rejects(name,lambda:e.packet(args,True))
 manifest_path.write_text(json.dumps(manifest));raw.write_bytes(b'changed');rejects('protected historical file changed',lambda:e.packet(args,True));raw.write_bytes(b'abc')
 Path(args.out).mkdir();rejects('existing attempt refused',lambda:e.packet(args,True));Path(args.out).rmdir()
 e.write(e.TASK/'EVAL_ONCE.json',{});rejects('consumed launch marker refused',lambda:e.packet(args,True))
 e.TASK, e.LATEST_COST=old_task,old_cost
 budget_dir=out/'budget';budget_dir.mkdir();meter=e.Meter(budget_dir,prior,True);meter.start-=301;rejects('time cap enforced',meter.guard)
 meter=e.Meter(budget_dir,prior,True)
 class Peak:
  @staticmethod
  def max_memory_allocated():return (18<<30)+1
 meter.cuda=Peak();rejects('allocated cap enforced',meter.guard);ok('final peak sampled even at rejection',meter.peak==(18<<30)+1)
 (fixture/'stderr.txt').write_bytes(b'x'*(5<<20));cli(['export','--out',str(fixture),'--archive',str(out/'too-large.zip')],1);ok('oversize export not created',not (out/'too-large.zip').exists())
 # Structural guarantee, not names-based blocking: no optimizer, engine restore or backward path.
 import ast,inspect
 for function in (e.ema_model,e.execute):
  text=inspect.getsource(function);ok(function.__name__+' no train/update path',all(token not in text for token in ('make_engine(','.backward(','.step(','.restore(')))
 ok('original v1 module',sc.VERSION==e.VERSION);ok('no CUDA initialized',not torch.cuda.is_initialized())
 summary={'status':'PASS_CPU_SYNTHETIC_ENTRY','checks':len(checks),'names':checks,'seconds':time.monotonic()-start,'real_detector_forward':0,'real_scores':0,'CUDA_initialized':False,'CPU_fixture_not_experiment_cost':True,'GPU_unknown':['original-v1 S EMA full CUDA evaluation','CUDA resource peaks','real endpoint scores','real scoring latency']};e.write(out/'SUMMARY.json',summary);print(json.dumps(summary));return 0
if __name__=='__main__':raise SystemExit(main())
