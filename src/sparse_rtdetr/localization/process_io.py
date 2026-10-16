"""Auxiliary /proc I/O: unavailable is explicit, identity failures stay fatal."""
from pathlib import Path
from .runio import require

def observe(live,rows,identity,*,runtime,proc_root=Path('/proc')):
    result=[]
    def check(pid,start):
        current=identity(pid)
        require(current is None or current['start']==start,'process identity changed during I/O sampling')
        return current is not None and current['state'] not in ('Z','X')
    for pid in sorted(live):
        start=rows[pid]['start'];item={'pid':pid,'start':start,'status':'EXITED','bytes':None}
        if not check(pid,start):result.append(item);continue
        try:
            fields=dict(line.split(': ') for line in (proc_root/str(pid)/'io').read_text().splitlines())
            rd=int(fields['read_bytes']);wr=int(fields['write_bytes']);require(rd>=0 and wr>=0,'negative process I/O counter')
            if check(pid,start):item.update(status='OBSERVED',bytes=rd+(0 if runtime else wr))
        except (PermissionError,FileNotFoundError,ProcessLookupError) as error:
            # Identity/stat errors are not caught here. Same live PID can have
            # inaccessible auxiliary I/O, but gains no artificial activity.
            if check(pid,start):item.update(status='UNAVAILABLE',reason=type(error).__name__,errno=error.errno,filename=error.filename)
            else:item.update(reason='EXIT_RACE')
        result.append(item)
    available=all(x['status']=='OBSERVED' for x in result)
    return {'status':'OBSERVED' if available else 'PARTIALLY_UNAVAILABLE','bytes':sum(x['bytes'] for x in result) if available else None,'members':result}
