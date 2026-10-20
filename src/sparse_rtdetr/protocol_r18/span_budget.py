"""Single-owner cumulative reservation ledger; uncertain/failed work is charged.

GPU time is allocated stage wall time (includes construction/checkpoint/I/O),
not utilization. Prior attempts must have an explicit certified cost receipt.
No retry/resume is implemented. Graceful reserve stops save a new state; hard
supervisor limit still applies if the worker cannot reach a safe boundary.
"""
import time
from pathlib import Path
from . import span_runtime as rt


def start(c, stage):
    root=Path(c['job_root']); prior=rt.exact(c['prior_cost'])
    p=root/'COST_LEDGER.json'
    if p.exists():
        ledger=rt.read(p)
    else:
        rt.require(prior['task_id']==rt.TASK and prior['profile']=='LAB_GPU_ONCE'
                   and type(prior['GPU_seconds']) in (int,float) and prior['GPU_seconds']>=0
                   and type(prior['forward_calls']) is int and prior['forward_calls']>=0,
                   'prior cumulative cost identity/values differ')
        ledger=dict(identity=rt.identity(c,stage),prior=prior,completed_GPU_seconds=prior['GPU_seconds'],
                    forward_calls=prior['forward_calls'],stages=[],active=None)
    rt.require(ledger['active'] is None, 'unfinished GPU allocation cannot be resumed')
    ledger['active']=dict(stage=stage,started=time.monotonic())
    rt.atomic(p,ledger)


def seconds(ledger, now=None):
    active=ledger['active']
    return ledger['completed_GPU_seconds']+(0 if active is None else
        (time.monotonic() if now is None else now)-active['started'])


def storage_bytes(c):
    return sum(p.stat().st_size for p in Path(c['job_root']).rglob('*') if p.is_file())


def guard(c, *, forward_calls=0, save=None, hard=False):
    if c['profile']!='LAB_GPU_ONCE':
        return
    path=Path(c['job_root'])/'COST_LEDGER.json'; ledger=rt.read(path)
    r=c['resources']; used=seconds(ledger); size=storage_bytes(c)
    reason=None
    if used >= r['GPU_hours_cap']*3600-(0 if hard else r['checkpoint_reserve_seconds']):
        reason='cumulative 60 GPU-hour cap/checkpoint reserve reached'
    if ledger['forward_calls']+forward_calls > rt.execution()['forward_budget']['total']:
        reason='cumulative forward budget exhausted'
    if size >= (r['task_storage_GiB_cap']-(0 if hard else 2))*rt.GIB:
        reason='task storage cap/checkpoint reserve reached'
    if reason:
        receipt=None
        if save is not None:receipt=save()
        rt.atomic(Path(c['job_root'])/'BUDGET_STOP.json',dict(reason=reason,checkpoint=receipt,
            GPU_seconds=used,forward_calls=ledger['forward_calls'],storage_bytes=size,no_retry=True))
        raise RuntimeError(reason)
    if forward_calls:
        rt.require(ledger['active'] is not None, 'forward outside admitted GPU stage')
        ledger['forward_calls']+=forward_calls
        rt.atomic(path,ledger)  # reserve BEFORE execution; exceptions never refund


def finish(c):
    path=Path(c['job_root'])/'COST_LEDGER.json'; ledger=rt.read(path)
    elapsed=time.monotonic()-ledger['active']['started']
    ledger['stages'].append(dict(stage=ledger['active']['stage'],GPU_seconds=elapsed))
    ledger['completed_GPU_seconds']+=elapsed;ledger['active']=None
    rt.atomic(path,ledger)
