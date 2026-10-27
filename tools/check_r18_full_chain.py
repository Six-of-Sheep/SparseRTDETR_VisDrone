"""Actual deployment/extraction/bind/readout CLI integration, synthetic only."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(base):
    base=Path(base);base.mkdir();checks=[];commands=[];start=time.monotonic()
    def load(p):return json.loads(Path(p).read_text())
    def write(p,x):Path(p).write_text(json.dumps(x))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def call(script,*args):
        t=time.monotonic();p=subprocess.run([sys.executable,'-B',str(script),*map(str,args)],capture_output=True,text=True,timeout=300)
        commands.append({'script':str(script),'args':list(map(str,args)),'rc':p.returncode,'seconds':time.monotonic()-t,'stdout':p.stdout,'stderr':p.stderr})
        write(base/'COMMANDS.json',commands);return p.returncode
    def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
    pkg=base/'package';check('package',call(ROOT/'tools/package_r18_readout.py',pkg,'--candidate')==0)
    manifest=pkg/'MANIFEST.json';release=base/'release'
    check('deploy-actual-CLI',call(pkg/'deploy.py','--bundle',pkg/'source.tar.gz','--manifest',manifest,'--sha256',sha(manifest),'--out',release)==0)
    check('deploy-repeat-refused',call(pkg/'deploy.py','--bundle',pkg/'source.tar.gz','--manifest',manifest,'--sha256',sha(manifest),'--out',release)!=0)
    bad=base/'corrupt.tar.gz';bad.write_bytes((pkg/'source.tar.gz').read_bytes()+b'tamper')
    check('deploy-corruption-refused',call(pkg/'deploy.py','--bundle',bad,'--manifest',manifest,'--sha256',sha(manifest),'--out',base/'badrelease')!=0)
    wrong=load(manifest);wrong['files']=wrong['files'][1:];wm=base/'missing-file-manifest.json';write(wm,wrong)
    check('deploy-missing-file-refused',call(pkg/'deploy.py','--bundle',pkg/'source.tar.gz','--manifest',wm,'--sha256',sha(wm),'--out',base/'missingrelease')!=0)
    check('native-behavior-on-off',call(release/'tools/check_r18_capture_behavior.py',base/'BEHAVIOR.json')==0)
    plan=base/'plan.json';entry=release/'tools/r18_full_chain.py'
    check('plan-CLI',call(entry,'plan','--synthetic','--manifest-sha',sha(manifest),'--out',plan)==0)
    run=base/'chain';check('full-chain-CLI',call(entry,'run','--contract',plan,'--out',run)==0)
    if not (run/'CLOSED.json').exists():
        write(base/'ACCEPTANCE.json',{'status':'FAIL','checks':checks,'commands':commands});return 1
    closed=load(run/'CLOSED.json');check('budgets-inherited-once',closed['ledger']=={'detector_forward':6,'head_update':9,'head_eval':12})
    check('chain-repeat-refused',call(entry,'run','--contract',plan,'--out',run)!=0)
    ext=release/'tools/r18_extract.py'
    for name in ('layers','IDs','coordinates','budget'):
        c=load(plan)
        if name=='layers':c['layer_order']=[2,1,0]
        if name=='IDs':c['ids'][0]='wrong'
        if name=='coordinates':c['coordinate']='pixel_xyxy'
        if name=='budget':c['forward_cap']=1
        path=base/(name+'.json');write(path,c);dest=base/name
        check('extract-reject-'+name,call(ext,'--contract',path,'--out',dest)!=0 and (dest/'STOP.json').exists())
    selection=run/'synthetic-selection.json';consumer=release/'tools/r18_readout_cli.py';receipt=run/'extract/CLOSED.json';features=run/'extract/features.pt'
    corrupt=base/'badfeatures.pt';corrupt.write_bytes(features.read_bytes()+b'x')
    check('bind-corrupt-feature-refused',call(consumer,'bind','--synthetic','--selection',selection,'--features',corrupt,'--receipt',receipt,'--out',base/'badbind.json')!=0)
    check('bind-missing-closure-refused',call(consumer,'bind','--synthetic','--selection',selection,'--features',features,'--receipt',base/'no-closure.json','--out',base/'noclosebind.json')!=0)
    index=load(run/'slim/SHA_INDEX.json');check('combined-export-SHA',all(sha(run/'slim'/n)==h for n,h in index.items()))
    need_source=release/'tools/r18_readout_math.py';need_source.write_bytes(need_source.read_bytes()+b'\n# tamper\n')
    check('deployed-source-tamper-refused',call(entry,'run','--contract',plan,'--out',base/'source-tamper')!=0)
    result={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks,'commands':commands,
            'scope':'CPU_CONSTRUCTED_NATIVE_NO_REAL_WEIGHTS_OR_IMAGES','seconds':time.monotonic()-start,
            'CPU_detector_forward':2+closed['ledger']['detector_forward']+load(base/'budget/STOP.json')['detector_forward'],
            'CPU_head_updates':closed['ledger']['head_update'],'CPU_head_eval':closed['ledger']['head_eval'],
            'real_GPU_budget_consumed':0,'candidate_manifest_sha256':sha(manifest)}
    write(base/'ACCEPTANCE.json',result);print(json.dumps({k:result[k] for k in ('status','seconds','CPU_detector_forward','CPU_head_updates','CPU_head_eval')}))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
