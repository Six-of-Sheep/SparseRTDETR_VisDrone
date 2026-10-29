"""Real paired CLI on clearly synthetic CPU inputs; never real extraction or GPU."""
import os,sys,json,subprocess,tempfile,shutil,time,resource,gc,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import torch
import sabl_pair_diagnostic as p
from sparse_rtdetr.research import sabl_control as sc
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.engine import tensor_digest,output_digest
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def main(path):
 out=Path(path);out.mkdir();start=time.monotonic();checks=[];baseline_F=0
 def check(name,value):
  checks.append({'name':name,'pass':bool(value)});(out/'CHECKS.json').write_text(json.dumps(checks,indent=2));assert value,name
 # Constructed deterministic association/invalid/zero-support/denominator cases.
 gt=[{'id':'a','bbox_xyxy':[0,0,10,10],'scale':'S'},{'id':'b','bbox_xyxy':[0,0,10,10],'scale':'S'}]
 pairs,unmatched=p.associate([[0,0,10,10],[0,0,10,10]],gt);check('exact ties stable GT then query',pairs==[(0,0),(1,1)] and not unmatched)
 pairs,_=p.associate([[20,20,30,30]],gt);check('zero IoU not excluded',pairs==[(0,0)])
 pairs,unmatched=p.associate([[0,0,10,10]],gt);check('capacity explicit unpaired GT',pairs==[(0,0)] and unmatched==[1]);check('empty GT no fabricated pairing',p.associate([[0,0,1,1]],[])==([],[]))
 # Coarse association stays fixed even if a different refined query is better.
 fixed,_=p.associate([[0,0,10,10],[1,1,9,9]],[gt[0]])
 check('refined does not reselect query',fixed==[(0,0)] and p.classify(1.,0.,False)=='to_invalid')
 rgt=[{'id':'s1','scale':'S'},{'id':'s2','scale':'S'},{'id':'m','scale':'M'},{'id':'l','scale':'L'},{'id':'u','scale':'L'}]
 cases=[{'scale':'S','change':'improved','IoU_before':0.,'IoU_after':.2,'delta':.2},{'scale':'S','change':'to_invalid','IoU_before':0.,'IoU_after':0.,'delta':0.},{'scale':'M','change':'unchanged','IoU_before':.4,'IoU_after':.4,'delta':0.},{'scale':'L','change':'degraded','IoU_before':.9,'IoU_after':.1,'delta':-.8}]
 summary=p.summarize([{'GT':rgt,'pairs':cases,'unpaired_GT':[rgt[-1]],'excluded_GT':{},'unpaired_query_count':296,'invalid_all_queries':3,'invalid_unpaired_queries':2}])
 check('all versus paired denominator and unpaired retained',summary['scales']['L']['paired_GT']==1 and summary['scales']['L']['all_eligible_GT']==2 and summary['scales']['L']['proportions_of_all_GT']['degraded']==.5 and summary['scales']['L']['proportions_of_paired']['degraded']==1)
 check('zero-support invalid included in continuous delta',summary['scales']['S']['to_invalid']==1 and summary['scales']['S']['delta']['n']==2)
 synth_rows=[{'stable_image_id':'SELECTION_ONLY/%03d'%i,'width':160,'height':128} for i in range(64)]
 def synth_raw(row):
  n=int(row['stable_image_id'].rsplit('/',1)[1])%12
  return {'rows':[{'id':row['stable_image_id']+'/'+str(j),'category':1,'score':1,'box':[1,1,9,9] if j%3==0 else ([20,20,60,60] if j%3==1 else [40,10,150,120]),'raw_area':64 if j%3==0 else (1600 if j%3==1 else 12100)} for j in range(n)],'counts':{}}
 chosen=p.choose(synth_rows,synth_raw);check('GT-only deterministic32 in four density bands',chosen==p.choose(synth_rows,synth_raw) and len(chosen['selected'])==32 and all(sum(x['density_band']==b for x in chosen['selected'])==8 for b in range(4)) and not chosen['uses_predictions'])
 check('exclusive original area boundaries',[p.scale(x) for x in [1023,1024,9215,9216]]==['S','M','M','L'])
 check('invalid counted before delta even zero before',p.classify(0,0,False)=='to_invalid');check('tolerance fixed',p.classify(.5,.5+1e-7,True)=='unchanged' and p.classify(.5,.51,True)=='improved' and p.classify(.5,.4,True)=='degraded')
 # Direct existing production decoder retains reversed/zero raw edges, no repair.
 r=torch.tensor([[.5,.5,.2,.4]]);x,scale=sc.buckets(r);z=torch.full((1,4,7),-20.);z[:,:,0]=20.;want=torch.tensor([[.8,.2,.1,.9]])
 d=torch.zeros_like(z);d[:,:,0]=(x[:,:,0]-want)/scale
 with p.Capture() as cap:
  boxes,valid=sc.decode(r,z,d)
 check('raw invalid LRTB retained before zeroing',torch.allclose(cap.data['decode']['e'],want) and not valid.item() and torch.equal(boxes[:,2:],torch.zeros(1,2)))
 # Ignore annotation treatment without reading any actual annotation/image.
 rows,raw,item=p.fixture_data();g,counts,_=p.GT(rows[1],raw(rows[1]));check('ordinary labels and score0/category0 excluded',len(g)==3 and counts['category0_or_other']==1 and counts['crowd_score0']==1)
 with tempfile.TemporaryDirectory(prefix='paired-CLI-',dir=out) as temp:
  root=Path(temp);release=root/'release';release.mkdir();names=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines());names|={'tools/sabl_pair_diagnostic.py','tools/check_sabl_pair_diagnostic.py','tests/fixtures/sabl_pair_base.json'};files=[]
  for n in sorted(names):
   a=ROOT/n;b=release/n;b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b);files.append({'path':n,'bytes':b.stat().st_size,'sha256':p.sha(b)})
  manifest={'commit':'c'*40,'candidate_not_for_real_execution':True,'files':files};mf=release/'RELEASE_MANIFEST.json';p.write(mf,manifest);p.write(release/'DEPLOYED.json',{'status':'PASS','manifest_sha256':p.sha(mf),**manifest})
  endpoints={};images=torch.stack([item(rows[i],i)[0] for i in range(4)])
  for arm in ['S','A']:
   torch.manual_seed(17);model,_=sc.build(arm,Config(input_hw=(128,128),physical_batch_size=4,amp_dtype='float32'));model.eval().requires_grad_(False);state_before=tensor_digest(sorted(model.state_dict().items()))
   with torch.inference_mode():baseline=model(images)
   baseline_F+=1
   with torch.inference_mode(),p.Capture() as cap:observed=model(images)
   baseline_F+=1;rec=cap.record(observed)
   check(arm+' original/observed full eval output bitwise identical',output_digest(baseline)==output_digest(observed));check(arm+' allQ300 and no DN/padding',rec['logits'].shape==(4,300,10) and rec['valid'].shape==(4,300))
   check(arm+' one ROI and one decoder call',cap.calls=={'roi':1,'decode':1});check(arm+' same-query reference exact',torch.equal(rec['reference_cxcywh_normalized'],observed['coarse_pred_boxes']))
   check(arm+' state and gradient unchanged',state_before==tensor_digest(sorted(model.state_dict().items())) and all(x.grad is None for x in model.parameters()))
   binding={'scope':'CPU_SYNTHETIC_PAIR_EMA','arm':arm,'version':sc.VERSION};cp=root/(arm+'-EMA.pt');torch.save({'format':'CPU_SYNTHETIC_EMA_PAIR_FIXTURE','binding':binding,'ema':{'module':model.state_dict(),'updates':7296}},cp)
   ep=root/(arm+'-endpoint.json');p.write(ep,{'arm':arm,'binding':binding,'epochs':24,'updates':7296,'EMA':True,'acceptance_state_used':False,'checkpoint':p.reference(cp)});endpoints[arm]=p.reference(ep)
   del model,baseline,observed,rec,cap;gc.collect()
  ef=root/'synthetic-endpoints.json';p.write(ef,endpoints);definition=root/'definition.json';entry=release/'tools/sabl_pair_diagnostic.py';env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
  def cli(name,args,success=True):
   r=subprocess.run([sys.executable,'-B',str(entry),*args],cwd=release,env=env,capture_output=True,timeout=280);(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr);check('actual CLI '+name,(r.returncode==0)==success);return r
  cli('prepare',['prepare','--manifest',str(mf),'--out',str(definition),'--synthetic','--synthetic-checkpoints',str(ef),'--attempt-root',str(root/'attempt')])
  def launch(name,defn,folder,ok=True,extra=[]):
   c=p.read(defn)
   if c['attempt_root']!=str(folder.resolve()):
    c['attempt_root']=str(folder.resolve());defn=root/(name+'-bound-definition.json');p.write(defn,c)
   return cli(name,['execute','--definition',str(defn),'--manifest',str(mf),'--approve-definition-SHA',p.sha(defn),'--approve-manifest-SHA',p.sha(mf),'--out',str(folder),'--authorize-cpu',*extra],ok)
  attempt=root/'attempt';launch('positive',definition,attempt);v=p.read(attempt/'RESULT.json');check('exact4 CPU forward no update',v['ledger']=={'forward':4,'backward':0,'optimizer_updates':0,'EMA_updates':0})
  for arm in ['S','A']:
   check(arm+' complete populations include unpaired and invalid',v['arms'][arm]['query_denominators']['all']==2400 and all(x['paired_GT']+x['unpaired_capacity_GT']==x['all_eligible_GT'] for x in v['arms'][arm]['scales'].values()))
   check(arm+' read-only EMA and BN audit',v['model_audits'][arm]['only_EMA_model_loaded'] and v['model_audits'][arm]['BN_unchanged'] and not v['model_audits'][arm]['optimizer_EMA_updater_engine_RNG_restored'])
   saved=torch.load(attempt/(arm+'-all-query-pairs.pt'),weights_only=True);check(arm+' raw fields per-image one query namespace',len(saved)==8 and saved[0]['raw_edges_LRTB_normalized'].shape==(300,4) and torch.equal(saved[0]['query_id'],torch.arange(300)))
  before=p.sha(attempt/'CLOSED.json');launch('duplicate',definition,attempt,False);check('duplicate closure preserved',p.sha(attempt/'CLOSED.json')==before)
  for key,mutate in [('wrong-policy',lambda c:c['policy'].__setitem__('delta_tolerance',.01)),('duplicate-ID',lambda c:c['selection']['selected'].__setitem__(1,c['selection']['selected'][0]))]:
   c=p.read(definition);mutate(c);q=root/(key+'.json');p.write(q,c);dest=root/key;launch(key,q,dest,False);check(key+' no attempt',not dest.exists())
  q=root/'bad-identity.json';c=p.read(definition);r=p.read(endpoints['S']['path']);r['binding']['arm']='A';bad=root/'wrong-arm-endpoint.json';p.write(bad,r);c['endpoints']['S']=p.reference(bad);p.write(q,c);dest=root/'bad-arm';launch('wrong-EMA-arm',q,dest,False);check('wrong arm STOP before forward',p.read(dest/'STOP.json')['ledger']['forward']==0)
  launch('mixed-GPU-authority',definition,root/'mixed',False,['--authorize-gpu']);check('no production authority made',not (root/'mixed').exists())
  c=p.read(definition);c['planned_forward']=1;q=root/'wrong-forward-cap.json';p.write(q,c);launch('wrong-forward-cap',q,root/'bad-cap',False);check('budget tamper no forward',not (root/'bad-cap').exists())
  cli('wrong-manifest-SHA',['execute','--definition',str(definition),'--manifest',str(mf),'--approve-definition-SHA',p.sha(definition),'--approve-manifest-SHA','0'*64,'--out',str(root/'bad-SHA'),'--authorize-cpu'],False);check('wrong SHA no forward',not (root/'bad-SHA').exists())
  cli('wrong-attempt-root',['execute','--definition',str(definition),'--manifest',str(mf),'--approve-definition-SHA',p.sha(definition),'--approve-manifest-SHA',p.sha(mf),'--out',str(root/'unbound'),'--authorize-cpu'],False);check('unbound attempt rejected',not (root/'unbound').exists())
  (out/'POSITIVE_RESULT.json').write_bytes((attempt/'RESULT.json').read_bytes())
  index=[{'path':str(x.relative_to(root)),'bytes':x.stat().st_size,'sha256':p.sha(x)} for x in root.rglob('*.pt')];p.write(out/'SYNTHETIC_STATE_INDEX_BEFORE_TEMP_REMOVAL.json',index)
 check('CPU CUDA remains uninitialized',not torch.cuda.is_initialized());check('CPU resource envelope',time.monotonic()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4<<30);v={'status':'PASS_ACTUAL_PAIRED_CPU_CLI','checks':len(checks),'synthetic_CPU_baseline_forward':baseline_F,'synthetic_CPU_CLI_forward':4,'CPU_backward_updates':0,'real_images_or_real_weights_loaded':False,'GPU_forward':0,'seconds':time.monotonic()-start,'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_compilation_status':'UNKNOWN_NOT_EXECUTED','protocol_not_changed_after_tests':True};p.write(out/'RESULT.json',v);print(json.dumps(v))
if __name__=='__main__':main(sys.argv[1])
