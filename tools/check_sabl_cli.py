"""Actual CPU package/deploy/plan/run/table checks; caller PYTHONPATH stripped."""
import sys,os,json,hashlib,subprocess,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(base):
    base=Path(base);base.mkdir();checks=[];commands=[];start=time.monotonic()
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
    def write(p,x):Path(p).write_text(json.dumps(x),encoding='utf-8')
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def call(tool,*argv):
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        p=subprocess.run([sys.executable,'-B',str(tool),*map(str,argv)],cwd=base,env=env,capture_output=True,text=True,timeout=300)
        commands.append({'script':str(tool),'argv':list(map(str,argv)),'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr});write(base/'COMMANDS.json',commands);return p.returncode
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:print(json.dumps(commands[-1]));raise AssertionError(name)
    pkg=base/'package';release=base/'release'
    check('package_actual_entry',call(ROOT/'tools/package_sabl.py',pkg,'--candidate')==0)
    manifest=pkg/'MANIFEST.json';h=sha(manifest);deploy=pkg/'deploy.py'
    check('deploy_actual_entry',call(deploy,'--bundle',pkg/'source.tar.gz','--manifest',manifest,'--sha256',h,'--out',release)==0)
    check('deploy_repeat_refused',call(deploy,'--bundle',pkg/'source.tar.gz','--manifest',manifest,'--sha256',h,'--out',release)!=0)
    bad=base/'bad.tar.gz';bad.write_bytes((pkg/'source.tar.gz').read_bytes()+b'corrupt')
    check('corrupt_package_refused',call(deploy,'--bundle',bad,'--manifest',manifest,'--sha256',h,'--out',base/'bad-release')!=0)
    m=read(manifest);m['files']=m['files'][1:];missing=base/'missing.json';write(missing,m)
    check('missing_package_member_refused',call(deploy,'--bundle',pkg/'source.tar.gz','--manifest',missing,'--sha256',sha(missing),'--out',base/'missing-release')!=0)
    entry=release/'tools/sabl_acceptance.py';contract=base/'cpu-contract.json';run=base/'attempt'
    check('CPU_plan_actual_entry',call(entry,'plan','--synthetic','--manifest-sha',h,'--attempt-root',run,'--out',contract)==0)
    check('CPU_full_actual_run_no_PYTHONPATH',call(entry,'run','--contract',contract,'--out',run)==0)
    closed=read(run/'CLOSED.json');result=read(run/'slim/RESULT.json')
    check('18_forwards_9_updates_closed_exit',closed['ledger']=={'detector_forward':18,'actual_updates':9} and read(run/'EXIT.json')['returncode']==0)
    check('all_three_full_state_replays',all(a['replay_full_state_exact'] and a['restore']['full_state_exact'] for a in result['arms'].values()))
    check('no_formal_or_science',not result['formal_training_started'] and result['scientific_quality']=='NOT_SCIENTIFIC' and closed['no_followup'])
    index=read(run/'slim/SHA_INDEX.json');check('export_SHA',all(sha(run/'slim'/k)==v for k,v in index.items()))
    before=sha(run/'CLOSED.json');check('run_repeat_no_overwrite',call(entry,'run','--contract',contract,'--out',run)!=0 and sha(run/'CLOSED.json')==before)
    c=read(contract)
    for name,key,value in [('budget','forward_cap',19),('formal','formal_training',True),('wrong_arm','arms',['N','S','M']),('scope','mode','formal'),('time','wall_seconds',3600),('identity','schema','wrong')]:
        x=dict(c);dest=base/name;x['attempt_root']=str(dest);x[key]=value;path=base/(name+'.json');write(path,x)
        check(name+'_actual_CLI_refusal',call(entry,'run','--contract',path,'--out',dest)!=0 and (dest/'STOP.json').exists())
    x=dict(c);dest=base/'GPU-denied';x.update(attempt_root=str(dest),mode='GPU_SABL_ACCEPTANCE_ONLY',gpu_seconds=900,wall_seconds=1800);path=base/'GPU-denied.json';write(path,x)
    check('GPU_denied_before_model',call(entry,'run','--contract',path,'--out',dest)!=0 and read(dest/'STOP.json')['ledger']['detector_forward']==0)
    ep=base/'endpoint.json';write(ep,{'checkpoint':{'path':'/SYNTHETIC/not-a-real-N.pth','sha256':c['N_checkpoint_SHA']}})
    meta=base/'legacy-record.json';write(meta,{'historical_N':{'endpoint':{'path':str(ep),'sha256':sha(ep)}},'data':{'train_core':{'manifest':{'path':'/SYNTHETIC/members.json'},'roots':{'role':'train_core','images':'/SYNTHETIC/train_core/images','annotations':'/SYNTHETIC/train_core/annotations'}}},'python':sys.executable})
    check('real_plan_legacy_metadata',call(entry,'plan','--metadata',meta,'--gpu-uuid','GPU-SYNTHETIC','--manifest-sha',h,'--attempt-root',base/'not-started','--out',base/'real-plan.json')==0)
    e=read(ep);e['checkpoint']['sha256']='0'*64;write(ep,e);md=read(meta);md['historical_N']['endpoint']['sha256']=sha(ep);write(meta,md)
    check('real_plan_wrong_N_refused',call(entry,'plan','--metadata',meta,'--gpu-uuid','GPU-SYNTHETIC','--manifest-sha',h,'--attempt-root',base/'not-started2','--out',base/'wrong-N.json')!=0)
    spec=importlib.util.spec_from_file_location('tables',release/'tools/sabl_results.py');tables=importlib.util.module_from_spec(spec);spec.loader.exec_module(tables)
    inp={'scope':'SYNTHETIC_NOT_SCIENTIFIC','N_checkpoint_sha256':c['N_checkpoint_SHA'],'scores':{}}
    for arm,value in [('N_original',.32),('N24',.30),('A24',.30),('S24',.31)]:
        inp['scores'][arm]={}
        for budget in ('600','300'):
            score=base/(arm+budget+'.json');write(score,{'COCO_bridge':{'metrics':{k:value for k in tables.COCO}},'official':{k:value*100 for k in tables.PRIMARY}})
            inp['scores'][arm][budget]={'path':str(score),'sha256':sha(score)}
    inputs=base/'table-inputs.json';write(inputs,inp)
    check('table_actual_CLI_original_N',call(release/'tools/sabl_results.py','--inputs',inputs,'--out',base/'table','--synthetic')==0)
    table=read(base/'table/RESULT.json');check('below_original_N_interpretation',table['quality_controls_pass'] and table['interpretation']=='相对控制改善，尚未形成现有基线净收益' and not table['new_gate_added'] and all(token in (base/'table/TABLE.md').read_text(encoding='utf-8') for token in ('S-N24','S-A24','S-originalN','SYNTHETIC_NOT_SCIENTIFIC')))
    check('synthetic_cannot_be_formal',call(release/'tools/sabl_results.py','--inputs',inputs,'--out',base/'fake-formal-table')!=0)
    del inp['scores']['N_original'];write(inputs,inp)
    check('missing_original_N_refused',call(release/'tools/sabl_results.py','--inputs',inputs,'--out',base/'missing-N-table','--synthetic')!=0)
    file=release/'src/sparse_rtdetr/research/sabl_control.py';file.write_bytes(file.read_bytes()+b'\n# damaged\n');dest=base/'source-tamper';x=dict(c);x['attempt_root']=str(dest);path=base/'source-tamper.json';write(path,x)
    check('source_tamper_refused',call(entry,'run','--contract',path,'--out',dest)!=0 and read(dest/'STOP.json')['ledger']['detector_forward']==0)
    summary={'status':'PASS','checks':checks,'seconds':time.monotonic()-start,'scope':'SYNTHETIC_NOT_SCIENTIFIC','CPU_detector_forward':18,'CPU_actual_updates':9,'GPU_cost':0,'manifest_sha256':h}
    write(base/'ACCEPTANCE.json',summary);print(json.dumps(summary))
if __name__=='__main__':main(sys.argv[1])
