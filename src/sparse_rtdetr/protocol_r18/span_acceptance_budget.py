"""Acceptance-only update/forward reservations, charged before each operation.

Failed reservations are never refunded. Completed counts are separate; a
forward pre-hook checks each microbatch, not a view-count estimate.
"""
import contextlib
from pathlib import Path
from . import span_runtime as rt

ARMS = ('B0', 'B1', 'P')


def path(c):
    rt.require(rt.acceptance_only(c), 'acceptance counter cannot serve full chain')
    return Path(c['job_root'])/'ACCEPTANCE_COUNTS.json'


def start(c):
    row = dict(updates_reserved=0, updates_completed=0, forward_calls_reserved=0,
               forward_calls_completed=0, forward_output_dtypes=[])
    rt.publish(path(c), dict(identity=rt.identity(c, 'admission'),
        limits=c['acceptance_limits'], arms={a:dict(row, forward_output_dtypes=[]) for a in ARMS}))


def read(c):
    v = rt.read(path(c))
    rt.require(v['identity'] == rt.identity(c, 'admission') and
               v['limits'] == rt.acceptance_config()['limits'] and set(v['arms']) == set(ARMS),
               'acceptance counter identity/limits differ')
    for row in v['arms'].values():
        rt.require(all(type(row[k]) is int and row[k] >= 0 for k in
            ('updates_reserved', 'updates_completed', 'forward_calls_reserved', 'forward_calls_completed')),
            'acceptance counter must be nonnegative native integer')
        rt.require(row['updates_completed'] <= row['updates_reserved'] <= 3 and
            row['forward_calls_completed'] <= row['forward_calls_reserved'] <= 6,
            'acceptance counter already exceeds per-arm limit')
    return v


def reserve_update(c, arm):
    v = read(c); rt.require(arm in ARMS, 'unknown acceptance arm'); r = v['arms'][arm]; limits = v['limits']
    rt.require(r['updates_reserved'] == r['updates_completed'] and
               r['forward_calls_reserved'] == r['forward_calls_completed'] == 2*r['updates_completed'],
               'incomplete update cannot be retried')
    rt.require(r['updates_reserved'] < limits['actual_updates_per_arm'] and
        sum(x['updates_reserved'] for x in v['arms'].values()) < limits['actual_updates_total'],
        'acceptance update budget exhausted before execution')
    r['updates_reserved'] += 1; rt.atomic(path(c), v)


def reserve_forward(c, arm):
    v = read(c); r = v['arms'][arm]; limits = v['limits']
    rt.require(r['updates_reserved'] == r['updates_completed']+1 and
        r['forward_calls_reserved'] == r['forward_calls_completed'], 'forward outside active update/retry')
    rt.require(r['forward_calls_reserved'] < limits['batch_forward_calls_per_arm'] and
        r['forward_calls_reserved'] < 2*r['updates_reserved'] and
        sum(x['forward_calls_reserved'] for x in v['arms'].values()) < limits['batch_forward_calls_total'],
        'acceptance forward budget exhausted before execution')
    r['forward_calls_reserved'] += 1; rt.atomic(path(c), v)


def complete_forward(c, arm, dtype):
    v = read(c); r = v['arms'][arm]
    rt.require(r['forward_calls_reserved'] == r['forward_calls_completed']+1, 'forward completion without reservation')
    rt.require(dtype == ('torch.bfloat16' if c['profile'] == 'LAB_GPU_ONCE' else 'torch.float32'),
               'actual acceptance logit compute dtype differs')
    r['forward_calls_completed'] += 1; r['forward_output_dtypes'].append(dtype); rt.atomic(path(c), v)


def complete_update(c, arm):
    v = read(c); r = v['arms'][arm]
    rt.require(r['updates_reserved'] == r['updates_completed']+1 and
        r['forward_calls_reserved'] == r['forward_calls_completed'] == 2*r['updates_reserved'],
        'update completion/actual forward count differs')
    r['updates_completed'] += 1; rt.atomic(path(c), v)


@contextlib.contextmanager
def forward_counter(c, model, arm):
    def before(module, args):
        reserve_forward(c, arm)
    def after(module, args, output):
        complete_forward(c, arm, str(output['pred_logits'].dtype))
    pre = model.register_forward_pre_hook(before)
    post = model.register_forward_hook(after)
    try:
        yield
    finally:
        pre.remove(); post.remove()


def verify_complete(c):
    v = read(c)
    rt.require(all(r['updates_reserved'] == r['updates_completed'] == 3 and
        r['forward_calls_reserved'] == r['forward_calls_completed'] == 6 for r in v['arms'].values()),
        'acceptance did not complete exactly 9 updates / 18 actual forwards')
    return v
