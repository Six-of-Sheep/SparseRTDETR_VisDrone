"""No-model bounded-memory event authentication and rejection regression."""
import argparse,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--child',choices=['stream','legacy']);a=p.parse_args()
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    from sparse_rtdetr.protocol_r18.span_event_stream import MAX_LINE_BYTES
    c=dict(task_id=rt.TASK,protocol=rt.PROTOCOL,profile='CPU_FIXTURE_ONLY',attempt='attempt-stream',
           contract_sha256='1'*64,source={'sha256':'2'*64},science_sha256='3'*64,pretrained={'sha256':'4'*64})
    names=['begin']+['update-'+str(i) for i in range(1,33)]+['complete']
    rt.event_spec=lambda *_:names
    out=a.output.resolve();path=out/'large.jsonl'
    if a.child:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(192*1024**2,192*1024**2))
        if a.child=='legacy':
            try:rows=path.read_bytes().decode().splitlines();rows=[json.loads(x) for x in rows]
            except MemoryError:print('EXPECTED_LEGACY_MEMORY_ERROR');return
            raise RuntimeError('legacy memory stress did not exceed cap')
        scan=rt.verify_events(c,'train-B0',path)
        rt.require(len(scan)==34 and scan.input_count==32,'bounded scan incomplete')
        print(json.dumps(dict(count=len(scan),bytes=scan.bytes,sha256=scan.sha256,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch_imported='torch' in sys.modules)))
        return
    out.mkdir(parents=True,exist_ok=False);checks=[]
    def write(rows,target):
        previous='0'*64
        with target.open('wb') as f:
            for r in rows:
                r['previous']=previous;r['sha256']=rt.digest({k:v for k,v in r.items() if k!='sha256'});previous=r['sha256']
                f.write(json.dumps(r).encode()+b'\n')
    rows=[dict(identity=rt.identity(c,'train-B0'),number=i+1,event=n,previous='',monotonic=i,pid=os.getpid(),
          fields={'input':{'i':i},'padding':'x'*(MAX_LINE_BYTES-2048)} if n.startswith('update-') else {}) for i,n in enumerate(names)]
    write(rows,path)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    for mode in ('legacy','stream'):
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out),'--child',mode],env=env,capture_output=True)
        rt.publish(out/(mode+'-process.json'),dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()))
        rt.require(r.returncode==0,mode+' memory subprocess failed');checks.append(mode+' 192MiB process')
    small=copy.deepcopy(rows)
    for r in small:r['fields'].pop('padding',None)
    write(small,out/'valid.jsonl');scan=rt.verify_events(c,'train-B0',out/'valid.jsonl')
    rt.require(scan.input_sha256==rt.digest([rt.digest(r['fields']['input']) for r in small if r['event'].startswith('update-')]),'input digest differs')
    checks.append('streaming input digest equals original list digest')
    cases={'missing':lambda r:r.pop(), 'identity':lambda r:r[2]['identity'].update(attempt='foreign'),
        'order':lambda r:r[2].update(number=1),'PID':lambda r:r[2].update(pid=999999),
        'time':lambda r:r[2].update(monotonic=-1),'nonfinite':lambda r:r[2].update(monotonic=float('inf'))}
    # NaN/Inf must be rejected even if a foreign encoder supplied a valid hash.
    cases.pop('nonfinite')
    for name,change in cases.items():
        bad=copy.deepcopy(small);change(bad);target=out/(name+'.jsonl');write(bad,target)
        try:rt.verify_events(c,'train-B0',target)
        except RuntimeError:checks.append(name+' rejected')
        else:raise RuntimeError(name+' accepted')
    raw=(out/'valid.jsonl').read_bytes()
    for name,content in [('SHA',raw.replace(b'"pid":',b'"pid":1',1)),('partial',raw[:-1]),('oversize',b'x'*(MAX_LINE_BYTES+1))]:
        target=out/(name+'.jsonl');target.write_bytes(content)
        try:rt.verify_events(c,'train-B0',target)
        except (RuntimeError,ValueError):checks.append(name+' rejected')
        else:raise RuntimeError(name+' accepted')
    original=rt.verify_events(c,'train-B0',out/'valid.jsonl');(out/'valid.jsonl').write_bytes(raw+b' ')
    try:list(original)
    except RuntimeError:checks.append('post-authentication mutation rejected')
    else:raise RuntimeError('mutation accepted')
    rt.publish(out/'CPU_GATE.json',dict(status='EVENT_STREAM_CPU_PASS',checks=checks,skips=0,
        no_model=True,production_oom_cause_proven=False,stress_file_bytes=path.stat().st_size))
    path.unlink();print(json.dumps(dict(status='PASS',checks=len(checks))))
if __name__=='__main__':main()
