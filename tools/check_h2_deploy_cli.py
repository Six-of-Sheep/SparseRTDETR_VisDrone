"""Exercise the actual self-contained installer CLI with synthetic source tarballs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
INSTALLER=ROOT/'tools/deploy_h2_source.py'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);commit='1'*40
    payload={'tools/deploy_h2_source.py':INSTALLER.read_bytes(),'src/constructed.txt':b'CONSTRUCTED SOURCE\n'+bytes(range(256))*8193}
    manifest=out/'manifest.json';manifest.write_text(json.dumps(dict(commit=commit,source_files=[dict(path=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest(),mode='100644') for k,v in payload.items()])),encoding='utf-8')
    def archive(name,values,extra=None):
        path=out/name
        with tarfile.open(path,'w') as t:
            for k,v in values.items():
                member=tarfile.TarInfo(k);member.size=len(v);t.addfile(member,io.BytesIO(v))
            if extra:t.addfile(extra)
        return path
    normal=archive('normal.tar',payload);cases=[];calls=0
    def run(name,tar=normal,man=manifest,ok=False,target=None,archive_sha=None):
        nonlocal calls
        calls+=1;target=target or out/(name+'-release');receipt=out/(name+'-receipt.json')
        argv=[sys.executable,'-B',str(INSTALLER),'--archive',str(tar),'--archive-sha256',archive_sha or sha(tar),
            '--manifest',str(man),'--manifest-sha256',sha(man),'--commit',commit,'--self-sha256',sha(INSTALLER),
            '--target',str(target),'--receipt',str(receipt)]
        r=subprocess.run(argv,capture_output=True);(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
        assert (r.returncode==0)==ok,(name,r.stdout,r.stderr)
        v=json.loads(receipt.read_text());assert v['status']==('PASS' if ok else 'STOP_NO_RETRY') and not v['execution_started'] and v['model_forward_calls']==0
        cases.append(name);return target,v
    target,v=run('normal',ok=True);assert v['files']==2 and v['all_member_SHA_verified'] and v['exact_source_inventory']
    for k,data in payload.items():assert (target/k).read_bytes()==data
    old={k:sha(target/k) for k in payload}
    run('duplicate-target',target=target);assert {k:sha(target/k) for k in payload}==old
    wrong=out/'bad-archive.tar';wrong.write_bytes(normal.read_bytes()+b'corruption');t,_=run('archive-SHA-corruption',tar=wrong,archive_sha=sha(normal));assert not t.exists()
    missing=out/'missing-archive.tar';calls+=1
    proc=subprocess.run([sys.executable,'-B',str(INSTALLER),'--archive',str(missing),'--archive-sha256','0'*64,'--manifest',str(manifest),'--manifest-sha256',sha(manifest),'--commit',commit,'--self-sha256',sha(INSTALLER),'--target',str(out/'missing-input-release'),'--receipt',str(out/'missing-input-receipt.json')],capture_output=True)
    assert proc.returncode==2 and not (out/'missing-input-release').exists();cases.append('missing-archive-input')
    omitted=archive('missing-member.tar',{'tools/deploy_h2_source.py':payload['tools/deploy_h2_source.py']});t,_=run('missing-manifest-member',tar=omitted);assert not t.exists()
    corrupt=dict(payload);corrupt['src/constructed.txt']=b'X'+corrupt['src/constructed.txt'][1:];t,v=run('per-file-SHA-corruption',tar=archive('wrong-member.tar',corrupt));assert t.exists() and 'extracted member SHA' in v['error']
    extra=dict(payload);extra['unindexed.txt']=b'X';t,_=run('unindexed-member',tar=archive('extra.tar',extra));assert not t.exists()
    evil=dict(payload);evil['../escaped.txt']=b'X';t,_=run('traversal-refusal',tar=archive('escape.tar',evil));assert not t.exists() and not (out/'escaped.txt').exists()
    symlink=tarfile.TarInfo('link');symlink.type=tarfile.SYMTYPE;symlink.linkname='../outside';t,_=run('symlink-refusal',tar=archive('link.tar',payload,symlink));assert not t.exists()
    bad=out/'bad-manifest.json';bad.write_text(json.dumps(dict(commit='2'*40,source_files=[])));t,_=run('wrong-manifest-commit',man=bad);assert not t.exists()
    proof=dict(status='PASS',tests=len(cases),skips=0,actual_CLI_invocations=calls,cases=cases,python=sys.version.split()[0],
        real_materials_or_GPU_used=False,model_forward_calls=0,source_files=[dict(path='tools/'+p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in (INSTALLER,Path(__file__))])
    (out/'DEPLOY_CLI_GATE.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':main()
