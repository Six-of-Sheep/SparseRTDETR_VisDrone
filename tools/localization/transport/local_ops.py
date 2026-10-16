"""Windows OpenSSH transport for the new F0 root; no training-control primitive."""
import json,os,shutil,subprocess,time
from pathlib import Path

REMOTE='/autodl-fs/data/localization-f0-seed0-e120-20261009-001'
BOOTSTRAP='/root/miniconda3/bin/python'
SSH_OPTIONS=['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',
             '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']

def native_ssh():
    if os.name!='nt':raise RuntimeError('transport requires Windows ssh.exe autodl-c2')
    path=Path(os.environ.get('WINDIR','C:/Windows'))/'System32/OpenSSH/ssh.exe'
    if not path.is_file():raise RuntimeError('Windows OpenSSH executable not found')
    # Resolve bare ssh.exe used inside the unchanged, verified transfer primitives.
    os.environ['PATH']=str(path.parent)+os.pathsep+os.environ.get('PATH','')
    if Path(shutil.which('ssh.exe')).resolve()!=path.resolve():raise RuntimeError('native ssh.exe resolution mismatch')
    return str(path)

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())

def remote_python(code,receipt,timeout=60):
    receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    script=receipt.with_suffix('.remote.py')
    with script.open('x',encoding='utf-8',newline='\n') as stream:stream.write(code)
    started=time.time();command=[native_ssh(),*SSH_OPTIONS,'autodl-c2',BOOTSTRAP+' -B -']
    with script.open('rb') as source,receipt.with_suffix('.stdout').open('xb') as stdout,receipt.with_suffix('.stderr').open('xb') as stderr:
        child=subprocess.Popen(command,stdin=source,stdout=stdout,stderr=stderr)
        timed_out=False
        try:returncode=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();returncode=child.wait(timeout=10)
    write(receipt,{'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,
                   'returncode':returncode,'timed_out':timed_out,'timeout_seconds':timeout,'pid':child.pid,
                   'ssh_alias':'autodl-c2','executable':command[0],'child_exited':True,'automatic_retry':False})
    if timed_out or returncode!=0:raise RuntimeError('remote request failed; inspect saved receipt '+str(receipt))
    return receipt.with_suffix('.stdout').read_text(encoding='utf-8')
