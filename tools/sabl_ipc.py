"""SABL-only short IPC directory bootstrap. No model or CUDA imports."""
import os,sys,json,hashlib,subprocess
from pathlib import Path

def root_for(task):
    task=Path(task).resolve()
    return Path('/tmp')/('sabl-'+hashlib.sha256(str(task).encode()).hexdigest()[:16])

def declaration(task):
    return {'schema':'SABL_PRIVATE_IPC_V1','task_root':str(Path(task).resolve()),'uid':os.getuid()}

def env_for(task,base=None,create=False):
    root=root_for(task);owner=declaration(task)
    if create:
        try:
            root.mkdir(mode=0o700)
            with (root/'OWNER.json').open('x',encoding='utf8') as f:json.dump(owner,f,sort_keys=True)
        except FileExistsError:pass
    if not root.is_dir() or root.is_symlink() or root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077:
        raise ValueError('private IPC owner/mode/path')
    marker=root/'OWNER.json'
    if marker.is_symlink() or json.loads(marker.read_text())!=owner:raise ValueError('private IPC binding conflict')
    env=dict(os.environ if base is None else base)
    for k in ['TMPDIR','TEMP','TMP']:env[k]=str(root)
    return env

def verify(task):
    import tempfile,multiprocessing
    root=root_for(task);expected=env_for(task)
    if any(os.environ.get(k)!=expected[k] for k in ['TMPDIR','TEMP','TMP']):raise ValueError('IPC environment must precede Python startup')
    if Path(tempfile.gettempdir()).resolve()!=root:raise ValueError('cached tempfile path differs')
    cached=multiprocessing.current_process()._config.get('tempdir')
    if cached is not None and not Path(cached).resolve().is_relative_to(root):raise ValueError('cached multiprocessing path differs')
    return {'TMPDIR':str(root),'tempfile':tempfile.gettempdir(),'multiprocessing_cached':cached,'before_fresh_python':True}

def owned_family_bytes(experiment_root):
    total=0;rows=[]
    for p in sorted(Path('/tmp').glob('sabl-*')):
        marker=p/'OWNER.json'
        if p.is_symlink() or not marker.is_file():continue
        try:v=json.loads(marker.read_text());task=Path(v['task_root'])
        except (OSError,ValueError,KeyError,TypeError):continue
        if v.get('schema')=='SABL_PRIVATE_IPC_V1' and task.is_relative_to(Path(experiment_root)) and task.relative_to(Path(experiment_root)).parts[0].startswith('r18-sabl-'):
            if p!=root_for(task) or p.stat().st_uid!=os.getuid():raise ValueError('family IPC identity')
            n=sum(x.stat().st_size for x in p.rglob('*') if x.is_file());total+=n;rows.append({'path':str(p),'task':str(task),'bytes':n})
    return total,rows

def main():
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);q=sub.add_parser('exec');q.add_argument('--task',required=True);q.add_argument('argv',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.argv
    if argv and argv[0]=='--':argv=argv[1:]
    if not argv or Path(argv[0]).resolve()!=Path(sys.executable).resolve():raise ValueError('same explicit interpreter required')
    os.execve(sys.executable,argv,env_for(a.task,create=True))
if __name__=='__main__':main()
