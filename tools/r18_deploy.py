"""Strict stdlib-only deployment; never imports torch or launches anything."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def deploy(bundle,manifest,expected,out):
    if sha(manifest)!=expected:raise ValueError('manifest SHA')
    m=json.loads(Path(manifest).read_text());rows=m['files'];index={r['path']:r for r in rows}
    if len(index)!=len(rows) or not rows:raise ValueError('duplicate/empty manifest')
    for name in index:
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('unsafe manifest path')
    if sha(bundle)!=m['bundle_sha256']:raise ValueError('source package SHA')
    with tarfile.open(bundle,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(index) or {x.name for x in members}!=set(index) or any(not x.isfile() for x in members):raise ValueError('missing/duplicate/nonregular archive entry')
        if any(x.size!=index[x.name]['bytes'] for x in members):raise ValueError('archive size')
        out=Path(out);out.mkdir()
        try:
            for x in members:
                dest=out/x.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(x) as source,dest.open('xb') as target:
                    for b in iter(lambda:source.read(1<<20),b''):target.write(b)
                if sha(dest)!=index[x.name]['sha256']:raise ValueError('file SHA '+x.name)
            (out/'DEPLOYED.json').write_text(json.dumps({'status':'PASS','commit':m['commit'],'candidate_not_for_real_execution':m['candidate_not_for_real_execution'],'manifest_sha256':expected,'files':rows}))
            (out/'RELEASE_MANIFEST.json').write_bytes(Path(manifest).read_bytes())
        except Exception as e:
            (out/'DEPLOY_STOP.json').write_text(json.dumps({'error':repr(e),'no_retry':True}));raise

def verify_release(root,expected):
    root=Path(root);r=json.loads((root/'DEPLOYED.json').read_text())
    if r['status']!='PASS' or r['manifest_sha256']!=expected:raise ValueError('release identity')
    if sha(root/'RELEASE_MANIFEST.json')!=expected:raise ValueError('release manifest bytes')
    m=json.loads((root/'RELEASE_MANIFEST.json').read_text())
    if any(r[k]!=m[k] for k in ('commit','candidate_not_for_real_execution','files')):raise ValueError('deployment receipt conflicts with manifest')
    for f in r['files']:
        p=root/f['path']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise ValueError('deployed file changed '+f['path'])
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--manifest',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();deploy(a.bundle,a.manifest,a.sha256,a.out)
