"""Actual-CLI acceptance, synthetic inputs only; outputs stay outside Git."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/'tools/r18_readout_cli.py'


def main(base):
    base=Path(base);base.mkdir()
    started=time.monotonic();checks=[];commands=[];cost={'detector_forward':0,'head_update':0,'head_eval':0}
    def call(*args):
        t=time.monotonic()
        r=subprocess.run([sys.executable,'-B',str(CLI),*map(str,args)],capture_output=True,text=True,timeout=300)
        commands.append({'args':list(map(str,args)),'rc':r.returncode,'seconds':time.monotonic()-t,'stdout':r.stdout,'stderr':r.stderr})
        return r.returncode
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
    def load(p):return json.loads(p.read_text())
    def write(p,x):p.write_text(json.dumps(x))
    for variant in ('normal','empty','empty_batch','invalid','nonfinite','identity','budget','probe_changed'):
        f=base/variant;run=base/(variant+'-run')
        check('fixture-'+variant,call('fixture','--variant',variant,'--out',f)==0)
        rc=call('run','--contract',f/'contract.json','--features',f/'features.pt','--out',run)
        good=variant in ('normal','probe_changed')
        check('CLI-'+variant,rc==0 if good else rc!=0)
        receipt=load(run/('CLOSED.json' if good else 'STOP.json'))
        for k,v in receipt['ledger'].items():cost[k]+=v
        if variant=='normal':
            result=load(run/'slim/RESULT.json')
            check('full-roundtrip-all-arms',result['roundtrip_arms']==['C0','C1','M'])
            check('physical-CPU-budget-separated',result['ledger']=={'detector_forward':0,'head_update':9,'head_eval':12})
            check('frozen-features-and-matching',result['feature_and_matching_unchanged'])
            index=load(run/'slim/SHA_INDEX.json')
            check('closed-export-SHA',all(hashlib.sha256((run/'slim'/p).read_bytes()).hexdigest()==s for p,s in index.items()))
            old=hashlib.sha256((run/'CLOSED.json').read_bytes()).hexdigest()
            check('duplicate-start-rejected',call('run','--contract',f/'contract.json','--features',f/'features.pt','--out',run)!=0)
            check('duplicate-preserves-closed',hashlib.sha256((run/'CLOSED.json').read_bytes()).hexdigest()==old)
        elif not good:check('STOP-'+variant,receipt['no_retry'] and not (run/'CLOSED.json').exists())
    # Compare serialized states in a separate bounded CPU subprocess: changing
    # only probe features/targets must not change model or optimizer at all.
    code="""
import os,resource,torch
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));torch.set_num_threads(2)
from pathlib import Path
base=Path(__import__('sys').argv[1])
def eq(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
for arm in ('C0','C1','M'):
 a=torch.load(base/'normal-run'/(arm+'-final.pt'),weights_only=False)
 b=torch.load(base/'probe_changed-run'/(arm+'-final.pt'),weights_only=False)
 assert eq(a['model'],b['model']) and eq(a['optimizer'],b['optimizer'])
import gzip,json
m=json.loads(gzip.decompress((base/'normal-run/slim/METRICS.json.gz').read_bytes()))
for arm in ('C1','M'):
 for a,b in zip(m['C0-initial']['rows'],m[arm+'-initial']['rows']):
  assert abs(a['iou']-b['iou'])<2e-6 and abs(a['edge_l1']-b['edge_l1'])<2e-6
"""
    r=subprocess.run([sys.executable,'-B','-c',code,str(base)],capture_output=True,text=True,timeout=60)
    check('probe-never-updates-parameters-and-initial-predictions-equal',r.returncode==0)
    commands.append({'state_comparison_rc':r.returncode,'stderr':r.stderr})
    normal=base/'normal'
    for name in ('source','ids','eval-budget','detector-budget','export-budget','real-authorization'):
        c=load(normal/'contract.json');target=base/(name+'.json');run=base/(name+'-run')
        if name=='source':c['sources'][next(iter(c['sources']))]='0'*64
        elif name=='ids':c['ids'][0]='wrong'
        elif name=='eval-budget':c['limits']['head_eval']=0
        elif name=='detector-budget':c['limits']['detector_forward']=-1
        elif name=='export-budget':c['export_bytes']=1
        else:c['mode']='REAL_REQUIRES_SEPARATE_AUTHORIZATION'
        write(target,c)
        check('CLI-'+name,call('run','--contract',target,'--features',normal/'features.pt','--out',run)!=0)
        stop=load(run/'STOP.json');check('STOP-'+name,stop['no_retry'] and not (run/'CLOSED.json').exists())
        for k,v in stop['ledger'].items():cost[k]+=v
    # Content changed after binding must reject before deserializing.
    corrupted=base/'corrupted.pt';corrupted.write_bytes((normal/'features.pt').read_bytes()+b'tamper')
    run=base/'tamper-run'
    check('feature-byte-tamper',call('run','--contract',normal/'contract.json','--features',corrupted,'--out',run)!=0)
    stop=load(run/'STOP.json');check('tamper-no-work',all(v==0 for v in stop['ledger'].values()))
    result={'scope':'SYNTHETIC_CPU_ONLY_NOT_REAL_EXPERIMENT','status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'checks':checks,'commands':commands,'physical_CPU_consumption':cost,'seconds':time.monotonic()-started,
            'children_peak_RSS_MiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/r18_readout_cli.py','tools/r18_readout_math.py', 'tools/check_r18_readout_cli.py']}}
    write(base/'ACCEPTANCE.json',result);print(json.dumps({k:result[k] for k in ('status','physical_CPU_consumption','seconds','children_peak_RSS_MiB')}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
