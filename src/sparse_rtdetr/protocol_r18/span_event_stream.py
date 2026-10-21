"""Bounded-memory full event authentication, with no lost validation fields."""
import hashlib,json,math,os
from pathlib import Path

MAX_LINE_BYTES=4*1024*1024


def fingerprint(path):
    p=Path(path)
    if not p.is_file() or p.is_symlink():raise RuntimeError('event file missing/linked')
    s=p.stat()
    return dict(path=str(p.resolve()),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


class VerifiedEvents:
    """Full scan, retaining only one record; iteration reauthenticates the file.

    Existing callers use len, terminal [-1], or iteration. Random indexing and
    slicing are intentionally unsupported rather than silently materializing.
    """
    def __init__(self,c,stage,path,complete=True):
        self.c,self.stage,self.path,self.complete=c,stage,Path(path),complete
        self.file_identity=fingerprint(path);self.count=0;self.last=None;self.pid=None
        for _ in self._scan():pass

    def _scan(self):
        from . import span_runtime as rt
        rt.require(fingerprint(self.path)==self.file_identity,'event file changed before scan')
        names=rt.event_spec(self.c,self.stage);previous='0'*64;clock=-1.;pid=None;count=0
        h=hashlib.sha256();size=0;inputs=hashlib.sha256(b'[');input_count=0
        with self.path.open('rb') as stream:
            while True:
                line=stream.readline(MAX_LINE_BYTES+1)
                if not line:break
                rt.require(len(line)<=MAX_LINE_BYTES,'event line exceeds bounded verifier limit')
                h.update(line);size+=len(line)
                if not line.endswith(b'\n'):
                    rt.require(not self.complete,'incomplete terminal event bytes');break
                r=json.loads(line.decode('utf-8'))
                rt.require(count<len(names) and r['event']==names[count] and r['number']==count+1
                    and r['identity']==rt.identity(self.c,self.stage),'event phase/identity/order differs')
                rt.require(r['previous']==previous and r['sha256']==rt.digest({k:v for k,v in r.items() if k!='sha256'}),'event chain SHA differs')
                rt.require(type(r['monotonic']) in (int,float) and math.isfinite(r['monotonic']) and r['monotonic']>=clock,'nonmonotonic event')
                rt.require(type(r['pid']) is int and r['pid']>0 and pid in (None,r['pid']),'event writer process changed')
                previous=r['sha256'];clock=r['monotonic'];pid=r['pid'];count+=1
                if r['event'].startswith('update-') and 'input' in r['fields']:
                    if input_count:inputs.update(b',')
                    inputs.update(json.dumps(rt.digest(r['fields']['input'])).encode())
                    input_count+=1
                self.last=r
                yield r
        rt.require(not self.complete or count==len(names),'terminal event missing')
        rt.require(fingerprint(self.path)==self.file_identity,'event file changed during scan')
        self.count,self.pid,self.sha256,self.bytes=count,pid,h.hexdigest(),size
        inputs.update(b']');self.input_count,self.input_sha256=input_count,inputs.hexdigest()

    def __len__(self):return self.count
    def __iter__(self):return self._scan()
    def __getitem__(self,index):
        if self.count and index in (-1,self.count-1):return self.last
        raise IndexError('only authenticated terminal indexing is supported')
