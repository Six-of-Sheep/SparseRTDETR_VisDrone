"""Immutable receipts and scoped files for the one F0 lifecycle."""
import hashlib,json,os,time,uuid
from pathlib import Path

TASK_ID="localization-f0-seed0-e120-20261009-001"
REMOTE_ROOT="/autodl-fs/data/"+TASK_ID
DESIGN_SHA="27147b8bc81c595422300d4aad46b08b25051ecf2e17a8d28f1aad94a57876cb"
SELECTION_SHA="b9d1ce181a91f168865b3a717ad635b57c53c898a3a3d3519952f9197cc794ec"
PRETRAINED_SHA="911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720"
KEY_EPOCHS=(0,1,30,60,90,120)
SHORT_STATES={'initial.pth':0,'C5.pth':5,'C6-direct.pth':6,'C6-replay.pth':6}
PHASES=('short','formal','trace-cache','trace-analysis')
def request_id():return str(time.time_ns())+'-'+uuid.uuid4().hex[:12]

def execution_binding(contract,mode):
    return {'task_id':TASK_ID,'mode':mode,**{k:contract[k] for k in (
        'contract_sha256','source_sha256','data_sha256','selection_sha256','gpu_uuid','boot_id','environment')}}

def original_process_exited(dispatch,current):
    recorded=dispatch.get('process_identity')
    require(isinstance(recorded,dict) and isinstance(recorded.get('start'),str),'original supervisor identity unavailable')
    return current is None or current['start']!=recorded['start'] or current['state'] in ('Z','X')

def require(ok,reason):
    if not ok: raise RuntimeError(reason)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def fsync_directory(path):
    if os.name=="posix":
        fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    fsync_directory(path.parent)
def publish(path,value):
    path=Path(path);temporary=path.with_name(path.name+".publishing")
    write(temporary,value)
    os.link(temporary,path)
    fsync_directory(path.parent)
    temporary.unlink()
    fsync_directory(path.parent)
def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+".next")
    with temporary.open("w",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fsync_directory(path.parent)
def event(out,kind,**fields):
    value={"wall_epoch":time.time(),"monotonic":time.monotonic(),"pid":os.getpid(),"event":kind,**fields}
    from . import channel
    if os.environ.get(channel.ENV):channel.event(value)
    else:
        with (Path(out)/"progress.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
        atomic(Path(out)/"progress.json",value)
    if kind!="update_complete" or value.get("update",0)%20==0: print(json.dumps(value,ensure_ascii=False),flush=True)
    return value
def reference(path):
    p=Path(path);return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def scoped(root,name):
    root=Path(root).resolve();p=Path(name)
    require(not p.is_absolute() and ".." not in p.parts,"unsafe relative artifact path")
    resolved=(root/p).resolve()
    require(resolved.is_relative_to(root) and not (root/p).is_symlink(),"artifact escapes task root")
    return resolved
