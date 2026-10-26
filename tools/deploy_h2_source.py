"""Strict one-target source installer; stdlib only, Python 3.10 and newer.

The bootstrap copy must match its manifest entry and the caller-pinned SHA.
No environment installation, model import, launch, overwrite or retry exists.
Failure evidence and any partially extracted target are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile


SELF_PATH='tools/deploy_h2_source.py'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):
    """Same SHA-256 digest; bounded reads, without the Python 3.11-only API."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def reference(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),'missing/symlink input: '+str(p))
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p))


def safe_name(name):
    p=PurePosixPath(name)
    require(isinstance(name,str) and bool(name) and not p.is_absolute() and '\\' not in name
        and '..' not in p.parts and '.' not in p.parts and p.as_posix()==name,'unsafe source member: '+str(name))
    return name


def no_links(path):
    p=Path(path)
    require(p.is_absolute(),'absolute deployment paths required')
    require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink in deployment path')
    require(p==p.resolve(),'noncanonical deployment path')


def install(a):
    target=a.target;receipt=a.receipt
    for p in (target,receipt):no_links(p)
    require(not target.exists(),'duplicate target refused')
    require(not receipt.exists(),'duplicate receipt refused')
    require(target.parent.is_dir() and receipt.parent.is_dir(),'deployment parent missing')
    require(not receipt.is_relative_to(target),'receipt must be outside source release')
    for value in (a.archive_sha256,a.manifest_sha256,a.self_sha256):
        require(re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid pinned SHA')
    require(re.fullmatch('[0-9a-f]{40}',a.commit) is not None,'invalid commit')
    bootstrap=reference(Path(__file__).resolve());require(bootstrap['sha256']==a.self_sha256,'bootstrap SHA differs')
    archive=reference(a.archive);manifest=reference(a.manifest)
    require(archive['sha256']==a.archive_sha256,'archive SHA differs')
    require(manifest['sha256']==a.manifest_sha256,'manifest SHA differs')
    spec=json.loads(a.manifest.read_text(encoding='utf-8'))
    require(spec['commit']==a.commit,'manifest commit differs')
    rows=spec.get('source_files',spec.get('files'));require(isinstance(rows,list) and bool(rows),'empty source inventory')
    by={}
    for row in rows:
        name=safe_name(row['path'])
        require(name not in by,'duplicate manifest member')
        require(type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'invalid member size/SHA')
        require(row.get('mode','100644') in ('100644','100755'),'unsupported source mode')
        by[name]=row
    require(SELF_PATH in by and by[SELF_PATH]['sha256']==bootstrap['sha256'] and by[SELF_PATH]['bytes']==bootstrap['bytes'],'bootstrap is not the frozen installer')
    with tarfile.open(a.archive,'r:*') as source:
        members=source.getmembers();names=set();files={}
        for member in members:
            name=safe_name(member.name.rstrip('/') if member.isdir() else member.name)
            require(name not in names,'duplicate archive member');names.add(name)
            require(member.isdir() or member.isfile(),'nonregular archive member refused')
            if member.isfile():
                require(name in by and member.size==by[name]['bytes'],'unindexed member or size differs: '+name);files[name]=member
        require(set(files)==set(by),'missing source member')
        target.mkdir(exist_ok=False)  # atomic exclusive reservation, never overwrite
        for name,row in by.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(files[name]) as incoming,path.open('xb') as output:
                shutil.copyfileobj(incoming,output,1024*1024);output.flush();os.fsync(output.fileno())
            os.chmod(path,0o755 if row.get('mode')=='100755' else 0o644)
            require(path.stat().st_size==row['bytes'] and sha256(path)==row['sha256'],'extracted member SHA/size differs: '+name)
    actual={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
    require(actual==set(by),'installed source inventory differs')
    if os.name!='nt':
        fd=os.open(target,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return dict(status='PASS',commit=a.commit,release=str(target),files=len(rows),bytes=sum(r['bytes'] for r in rows),
        manifest=manifest,archive=archive,installer=bootstrap,python=sys.version.split()[0],
        exact_source_inventory=True,all_member_SHA_verified=True,model_forward_calls=0,execution_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','manifest','target','receipt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('archive-sha256','manifest-sha256','self-sha256','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:
        result=install(a);code=0
    except BaseException as error:
        result=dict(status='STOP_NO_RETRY',error=repr(error),phase='source_install',target=str(a.target),
            target_exists=a.target.exists(),python=sys.version.split()[0],model_forward_calls=0,execution_started=False);code=2
    # An earlier receipt remains immutable, including on duplicate invocation.
    if a.receipt.parent.is_dir() and not a.receipt.exists() and not a.receipt.is_symlink():
        no_links(a.receipt)
        with a.receipt.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result,ensure_ascii=False));return code


if __name__=='__main__':raise SystemExit(main())
