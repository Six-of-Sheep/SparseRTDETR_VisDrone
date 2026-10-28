"""Real CLI CPU acceptance; synthetic local features, no detector/GPU/optimizer."""
import argparse,base64,copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import sabl_forward_check as f


def main():
 parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--definition');parser.add_argument('--manifest');args=parser.parse_args()
 if bool(args.definition)!=bool(args.manifest):parser.error('both real material definition and manifest required')
 import torch
 from unittest.mock import patch
 torch.set_num_threads(2);passed=[]
 def check(ok,name):
  if not ok:raise AssertionError(name)
  passed.append(name)
 def reject(call,name):
  try:call()
  except (ValueError,TypeError,RuntimeError,KeyError):passed.append(name);return
  raise AssertionError('must reject '+name)
 with tempfile.TemporaryDirectory(prefix='SYNTHETIC_FORWARD_ONLY_') as tmp:
  tmp=Path(tmp);p=tmp/'protected.bin';p.write_bytes(b'constructed CPU fixture');r={'bytes':p.stat().st_size,'sha256':f.sha(p)}
  check(f.sha(p)==hashlib.sha256(p.read_bytes()).hexdigest(),'hashlib.update remains allowed')
  duplicate_json=tmp/'duplicate.json';duplicate_json.write_text('{"scope":"old","scope":"new"}');reject(lambda:f.read(duplicate_json),'duplicate JSON fields rejected')
  ref={'path':str(p),**r};check(f.exact(ref)==p,'bytes physical reference')
  legacy={'path':str(p),'size_bytes':r['bytes'],'sha256':r['sha256']};check(f.exact(legacy)==p,'historical size_bytes physical reference')
  check(f.exact({**ref,'size_bytes':r['bytes']})==p,'consistent dual size fields')
  for value in [None,0,-1,True,1.1,'25']:
   bad={**ref,'bytes':value};reject(lambda b=bad:f.exact(b),'invalid size '+repr(value))
  reject(lambda:f.exact({k:v for k,v in ref.items() if k!='bytes'}),'missing size')
  reject(lambda:f.exact({**ref,'size_bytes':r['bytes']+1}),'conflicting dual size')
  reject(lambda:f.exact({**ref,'unknown':1}),'unknown reference field')
  reject(lambda:f.exact({**ref,'sha256':'g'*64}),'invalid SHA alphabet')
  wrapped={'protected_files':{str(p):r},'all_unchanged':True};check(f.protected_records(wrapped)=={str(p):r},'nested extraction')
  for bad in [{str(p):r},{'protected_files':[]},{'protected_files':{}}]:reject(lambda b=bad:f.protected_records(b),'invalid protection schema '+str(len(passed)))
  check(f.preserve(f.protected_records(wrapped))['all_unchanged'],'physical protected SHA')
  reject(lambda:f.preserve({str(p):{**r,'sha256':'0'*64}}),'protected corruption')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
  def cli(args,cwd=ROOT):
   v=subprocess.run([sys.executable,'-B',str(ROOT/'tools/sabl_forward_check.py'),*args],cwd=cwd,env=env,capture_output=True,text=True)
   check(len(v.stdout.encode())+len(v.stderr.encode())<24000,'bounded CLI output '+str(len(passed)));return v
  spec=tmp/'spec.json';spec.write_text(json.dumps({'scope':'SYNTHETIC_NOT_SCIENTIFIC'}));out=tmp/'actual-cli'
  v=cli(['fixture','--spec',str(spec),'--out',str(out)],cwd=tmp)
  if v.returncode:raise AssertionError('actual CLI normal: '+v.stderr[-8000:]+v.stdout[-4000:])
  result=f.read(out/'RESULT.json');check(result['status']=='SYNTHETIC_PASS_NOT_SCIENTIFIC','two-arm synthetic closed')
  check(f.read(out/'CLOSED.json')['result_sha256']==f.sha(out/'RESULT.json'),'closed SHA')
  check(result['counts']['detector_forward_reserved']==2 and result['counts']['training_target_backward']==result['counts']['optimizer_updates']==result['counts']['EMA_updates']==0,'synthetic forward counts zero training updates')
  check(result['CUDA_initialized'] is False,'CUDA never initialized')
  for arm in result['reports']:
   a=arm['forward'];check(arm['bridge']['all_raw_bytes_exact'] and not arm['bridge']['full_training_state_restored'],'strict raw bridge '+a['arm'])
   check(a['parameters_unchanged'] and a['all_parameter_gradients_NONE'] and all(x=='NONE' for x in a['gradient_state_before'].values()),'parameter/gradient audit '+a['arm'])
   check(a['BN_changes_in_memory_only'] and a['only_BN_buffers_changed'] and bool(a['BN_change_details']),'BN buffer audit '+a['arm'])
   check(len(a['local'])==2 and all(x['new_finite'] for x in a['local']),'local axes '+a['arm'])
  check(cli(['fixture','--spec',str(spec),'--out',str(out)]).returncode!=0,'duplicate CLI rejection')
  for fault in ('version','missing_tensor','dtype','mode','nonfinite','forward_cap','wall_cap','storage_cap','populated_gradient'):
   spec.write_text(json.dumps({'scope':'SYNTHETIC_NOT_SCIENTIFIC','fault':fault}));dst=tmp/fault;v=cli(['fixture','--spec',str(spec),'--out',str(dst)])
   check(v.returncode!=0 and not (dst/'CLOSED.json').exists() and (dst/'STOP.json').exists(),'same terminal path rejects '+fault)
   stop=f.read(dst/'STOP.json');check(stop['status']=='STOP_FORWARD_EXCEPTION' and 'cost' in stop,'STOP cost retained '+fault)
  # The exact production export implementation is exercised on closed and failed fixtures.
  for dst in (out,tmp/'nonfinite'):
   v=cli(['export','--attempt',str(dst),'--already-returned-bytes','1000']);check(v.returncode==0,'actual export CLI '+dst.name)
   packet=json.loads(v.stdout)
   check(all(hashlib.sha256(base64.b64decode(row['base64'])).hexdigest()==row['sha256'] for row in packet['files']),'export SHA '+dst.name)
   check(all(row['path'] in f.EXPORT_NAMES for row in packet['files']),'whitelist '+dst.name)
  check(cli(['export','--attempt',str(out),'--already-returned-bytes',str(f.CAPS['return_bytes']-1)]).returncode!=0,'total wire plus attachments cap')
  check(cli(['export','--attempt',str(tmp),'--already-returned-bytes','0']).returncode!=0,'export requires terminal marker')
  # Real native CPU RoIAlign and CPU AOT tracing, never claimed to be CUDA compilation.
  from torchvision.ops import roi_align
  APIs=(torch.autograd.grad,torch.autograd.backward,torch.Tensor.backward)
  x=torch.linspace(-1,1,16).reshape(1,1,4,4).requires_grad_();boxes=torch.tensor([[0.,0.,0.,3.,3.]])
  pooled=roi_align(x,boxes,(2,2),sampling_ratio=2,aligned=True)
  grads=torch.autograd.grad(pooled.sum(),x)[0]
  check(torch.isfinite(grads).all() and x.grad is None,'native CPU RoIAlign local autodiff')
  fn=torch.compile(lambda a:a.sin()*a,backend='aot_eager',fullgraph=True)
  v=fn(x);check(torch.allclose(v,x.sin()*x,rtol=0,atol=0) and x.grad is None,'CPU AOT forward compatibility')
  check(APIs==(torch.autograd.grad,torch.autograd.backward,torch.Tensor.backward),'autograd APIs never globally patched')
  sc,_=f.import_model();model=f.FixtureModel.create('A');meter=f.Meter(tmp,tmp,'cpu')
  saved={'roi_before_A_mean':torch.ones(1,256,7,7)};obs=f.Observe(model,sc,meter,'A',saved);obs.index=0
  try:reject(lambda:model.boundary(torch.zeros(1,256,7,7),'A'),'first local mismatch stops')
  finally:obs.close()
  check(obs.matches[0]['exact'] is False and sc.normalized_sigmoid_axis is obs.original,'observer restored and mismatch retained')
  # GPU allocator is only simulated here, including inability to sample.
  meter=f.Meter(tmp,tmp,'cuda')
  with patch.object(torch.cuda,'max_memory_allocated',return_value=12345):meter.sample_resources()
  check(meter.snapshot()['peak_allocated_bytes']==12345,'final peak sampling method')
  with patch.object(torch.cuda,'max_memory_allocated',side_effect=RuntimeError('synthetic allocator unavailable')):meter.sample_resources()
  check(meter.snapshot()['peak_allocated_bytes']=='UNKNOWN' and meter.snapshot()['last_known_peak_allocated_bytes']==12345,'sampling failure stays UNKNOWN')
  failure=tmp/'secondary-failure';failure.mkdir();meter=f.Meter(failure,failure,'cpu')
  with patch.object(meter,'publish',side_effect=OSError('synthetic ledger write failure')):f.stop_receipt(failure,meter,RuntimeError('PRIMARY_SYNTHETIC_ERROR'),'constructed traceback')
  check('PRIMARY_SYNTHETIC_ERROR' in f.read(failure/'STOP.json')['error'] and f.read(failure/'STOP.json')['secondary_errors'],'primary exception survives secondary write failure')
  croot=tmp/'cache';croot.mkdir();before={k:os.environ.get(k) for k in f.CACHE_VARIABLES}
  with patch.dict(os.environ,{k:v for k,v in f.cache_environment(croot).items()}):
   cache=f.CacheAccounting(croot);cache.external=[];cache.before={};p=croot/'compile-cache/inductor/generated.bin';p.parent.mkdir(parents=True);p.write_bytes(b'x'*101)
   meter=f.Meter(tmp,croot,'cpu',cache=cache);meter.sample_resources();snap=meter.snapshot()['storage']
   check(snap['compiler_cache']['bound_cache_bytes']==101 and snap['charged_storage_bytes']==101+(32<<20),'compiler cache charged once in task storage')
  meter=f.Meter(tmp,tmp,'cpu',{**f.CAPS,'wall_seconds':-1});reject(meter.guard,'wall hard cap')
  meter=f.Meter(tmp,tmp,'cpu',{**f.CAPS,'storage_bytes':1});reject(meter.guard,'storage hard cap')
  # Safe source manifest validation uses the same function as production admission.
  release=tmp/'release';release.mkdir();(release/'source.py').write_text('synthetic = True\n')
  (release/'empty.py').write_bytes(b'')
  rows=[{'path':'source.py','bytes':(release/'source.py').stat().st_size,'sha256':f.sha(release/'source.py')}]
  rows.append({'path':'empty.py','bytes':0,'sha256':f.sha(release/'empty.py')})
  manifest=tmp/'manifest.json';m={'base_commit':f.BASE,'scope':'SABL_TWO_FORWARD_ONLY_V1','commit':'a'*40,'files':rows};manifest.write_text(json.dumps(m))
  with patch.object(f,'ROOT',release):
   check(f.verify_release(manifest)['commit']=='a'*40,'manifest physical source verification')
   for badrows in [rows+rows,[{**rows[0],'path':'../source.py'}]]:
    manifest.write_text(json.dumps({**m,'files':badrows}));reject(lambda:f.verify_release(manifest),'duplicate/unsafe source manifest '+str(len(passed)))
  # Explicit state checks detect changes; no global training/crypto API interception.
  from sparse_rtdetr.protocol_r18.engine import tensor_digest
  model=f.FixtureModel.create('S');param_before=tensor_digest(model.named_parameters());buffers={k:v.detach().clone() for k,v in model.named_buffers()};grads={k:'NONE' for k,v in model.named_parameters()}
  with torch.no_grad():next(model.parameters()).add_(1)
  a=f.model_audit(model,param_before,buffers,grads);check(not a['parameters_unchanged'],'parameter mutation detected by bytes')
  next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
  check(not f.model_audit(model,param_before,buffers,grads)['all_parameter_gradients_NONE'],'populated parameter gradient detected')
  if args.definition:
   definition=Path(args.definition);manifest=Path(args.manifest);d=f.read(definition);m=f.read(manifest);attempt=Path(d['root'])/'attempt-forward-001';authorization=Path(d['root'])/'FORWARD_AUTHORIZATION.json'
   common=['--definition',str(definition),'--manifest',str(manifest),'--authorization',str(authorization),'--out',str(attempt)]
   v=cli(['run',*common,'--check-only']);check(v.returncode==0 and json.loads(v.stdout)['CUDA_initialized'] is False,'actual real-material launch CLI dry admission')
   check(cli(['run',*common,'--check-only'],cwd=tmp).returncode!=0,'real launcher rejects wrong cwd')
   approvals=['--approve-commit',m['commit'],'--approve-manifest-sha256',f.sha(manifest)]
   v=cli(['bind',*common,*approvals,'--authorize-two-forwards','--check-only']);check(v.returncode==0 and json.loads(v.stdout)['authorization_created'] is False,'actual registered binder check-only')
   check(cli(['bind',*common,*approvals,'--check-only']).returncode!=0,'binder rejects missing explicit scope approval')
   check(cli(['bind',*common,'--approve-commit','0'*40,*approvals[2:],'--authorize-two-forwards','--check-only']).returncode!=0,'binder rejects version conflict')
   cpu_candidate=tmp/'never-executable-binding-candidate.json';cpu_candidate.write_text(json.dumps({**m,'candidate_not_for_real_execution':True}))
   candidate_common=[*common[:2],'--manifest',str(cpu_candidate),*common[4:]];candidate_approvals=['--approve-commit',m['commit'],'--approve-manifest-sha256',f.sha(cpu_candidate)]
   check(cli(['bind',*candidate_common,*candidate_approvals,'--authorize-two-forwards']).returncode!=0,'CPU candidate cannot create production authorization')
   edits=[('caps',lambda x:x['caps'].__setitem__('forward',3)),('checkpoint',lambda x:x['refs']['A'].__setitem__('sha256','0'*64)),('release',lambda x:x.__setitem__('release',str(tmp))),('python',lambda x:x.__setitem__('python','/usr/bin/false')),('GPU',lambda x:x.__setitem__('GPU','unknown')),('target',lambda x:x['target'].__setitem__('update',940)),('prior',lambda x:x['prior'].__setitem__('forward',31740))]
   for name,edit in edits:
    bad=copy.deepcopy(d);edit(bad);path=tmp/(name+'-definition.json');path.write_text(json.dumps(bad))
    cmd=['run','--definition',str(path),*common[2:],'--check-only'];check(cli(cmd).returncode!=0,'real CLI rejects '+name)
   bad=copy.deepcopy(m);bad['files'][0]['sha256']='0'*64;mp=tmp/'corrupt-source.json';mp.write_text(json.dumps(bad))
   check(cli(['run',*common[:2],'--manifest',str(mp),*common[4:],'--check-only']).returncode!=0,'real CLI source corruption rejection')
   # Isolated invalid authorizations only; production auth/attempt/once remain absent.
   unsigned=tmp/'cpu-negative-binding-manifest.json';unsigned.write_text(json.dumps({**m,'candidate_not_for_real_execution':False}))
   expected=f.expected_authorization(d,m,definition,unsigned,attempt)
   for name,edit in [('numeric-bool',lambda x:x.__setitem__('authorize_once',1)),('head-conflict',lambda x:x.__setitem__('commit','0'*40))]:
    auth=copy.deepcopy(expected);edit(auth);ap=tmp/(name+'-authorization.json');ap.write_text(json.dumps(auth))
    cmd=['run','--definition',str(definition),'--manifest',str(unsigned),'--authorization',str(ap),'--out',str(attempt)]
    check(cli(cmd).returncode!=0,'actual caller rejects authorization '+name)
   definition_before=f.sha(definition);v=cli(['prepare','--out',str(definition.parent)]);check(v.returncode!=0 and f.sha(definition)==definition_before,'prepare duplicate preserves existing definition')
   check(not attempt.exists() and not authorization.exists() and not (Path(d['root'])/'FORWARD_ONCE.json').exists(),'CPU checks create no production attempt/auth/once')
  check(f.CAPS=={'forward':2,'backward':0,'optimizer_updates':0,'EMA_updates':0,'GPU_seconds':300,'wall_seconds':600,'allocated_bytes':18<<30,'storage_bytes':128<<20,'return_bytes':1<<20},'unchanged scientific execution caps')
 result={'status':'PASS_CPU_CLI_NOT_SCIENTIFIC','checks':passed,'count':len(passed),'CUDA_initialized':torch.cuda.is_initialized(),'full_detector_forward':0,'optimizer_updates':0,'production_inputs_used':bool(args.definition),'real_images_used':False,'CPU_compile_scope':'aot_eager plus native CPU RoIAlign, not CUDA Inductor','still_GPU_only':['actual CUDA deterministic RoIAlign compilation','BF16 native detector forward','A saved tensor exactness','S full output','GPU memory peak and cache growth']}
 result['real_materials_readonly_admission']=bool(args.definition);Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'checks':len(passed)}))
if __name__=='__main__':main()
