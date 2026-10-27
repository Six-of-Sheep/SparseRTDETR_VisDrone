"""PR26 fixed 24-epoch policy; no model imports and no execution authority."""
import hashlib
import json
import re
from pathlib import Path

SCOPE = 'PR26_FORMAL_NSA_24_V1'
CPU = 'CPU_SYNTHETIC_SABL_FORMAL'
GPU = 'GPU_SABL_FORMAL_24'
ARMS = ('N', 'S', 'A')
PRIOR = {'detector_forward': 18, 'GPU_stage_seconds': 53.45389260002412}
CAPS = {'detector_forward': 44589, 'GPU_stage_seconds': 86400,
        'wall_seconds': 172800, 'allocated_bytes': 18 << 30,
        'storage_bytes': 30 << 30, 'return_bytes': 5 << 20}
POLICY = {'epochs': 24, 'windows_per_epoch': 304, 'engineering_epochs': 2,
          'strong_epochs': [1, 21], 'weak_epochs': [22, 24], 'seed': 0,
          'endpoint': 'local_e24_EMA_only', 'budgets': [600, 300],
          'train_forwards': 43776, 'endpoint_forwards': 411, 'cost_forwards': 384,
          'new_forwards': 44571, 'actual_updates': 21888,
          'acceptance_state_used': False, 'intermediate_AP': False,
          'holdout_access': False, 'automatic_retry': False}
COST_ORDERS = [('N', 'S', 'A'), ('S', 'A', 'N'), ('A', 'N', 'S'),
               ('N', 'A', 'S'), ('A', 'S', 'N'), ('S', 'N', 'A')]
ACCEPT_COMMIT = '0aecdf798565bab5e4628f3b2b2da94e55808afd'
ACCEPT_MANIFEST = '3915bbf03f5c335b2a499979e55876cf5856b39bbee92fec45b7d68324f4b7c1'
ACCEPT_COST_SHA = 'b92492895e06675cd11dcf9b9a8691436fdcf255c1ebb0d30589bc4ab3ee6666'
ACCEPT_CONTRACT_SHA = 'b770efea229fe0d4a9a1df39fa0acadf2d609faebca9dd7586e1a9464440b5ce'
N_SHA = 'be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exact(ref):
    need(isinstance(ref, dict) and set(ref) >= {'path', 'bytes', 'sha256'}, 'reference fields')
    need(type(ref['bytes']) is int and ref['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'reference size/digest type')
    p = Path(ref['path'])
    need(p.is_absolute() and p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)), 'reference path')
    need(p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256'], 'reference size/SHA: ' + str(p))
    return read(p)


def reference(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def profile(c):
    return ({'epochs': 3, 'windows': 1, 'engineering_epochs': 2, 'images': 6, 'fixed': 4}
            if c['mode'] == CPU else {'epochs': 24, 'windows': 304, 'engineering_epochs': 2, 'images': 548, 'fixed': 64})


def new_forwards(c):
    p = profile(c)
    return 3 * p['epochs'] * p['windows'] * 2 + 3 * ((p['images'] + 3) // 4) + 3 * 8 * (p['fixed'] // 4)


def validate(c):
    need(set(c) == {'schema','mode','policy','caps','prior','arms','N_checkpoint_SHA','manifest_sha256',
                   'attempt_root','metadata','acceptance_cost','gpu_uuid','fixture','planned_new_forwards'}, 'contract fields')
    need(c['schema'] == SCOPE and c['mode'] in (CPU, GPU), 'formal scope')
    need(digest(c['policy']) == digest(POLICY) and digest(c['caps']) == digest(CAPS) and digest(c['prior']) == digest(PRIOR), 'frozen policy/budget inheritance')
    need(c['arms'] == list(ARMS) and c['N_checkpoint_SHA'] == N_SHA, 'formal arms/parent')
    need(c['planned_new_forwards'] == new_forwards(c), 'profile forward budget')
    need(c['mode'] != GPU or new_forwards(c) + PRIOR['detector_forward'] == CAPS['detector_forward'], 'full budget')
    need(c['mode'] != GPU or not c.get('fixture'), 'production fixture forbidden')
    need(c['mode'] != CPU or (c['metadata'] is None and c['acceptance_cost'] is None and c['gpu_uuid'] is None), 'CPU cannot admit real material')
    need(c['mode'] != CPU or c['fixture'].keys() == {'fault'}, 'CPU fixture schema')
    need(c['mode'] != CPU or c['fixture']['fault'] in ('none', 'worker_exception', 'forward_exhausted', 'bad_endpoint', 'bad_input', 'nonfinite', 'missing_close'), 'unknown CPU fault')
    return c


def verify_acceptance(ref):
    cost = exact(ref)
    need(ref['sha256'] == ACCEPT_COST_SHA and cost['commit'] == ACCEPT_COMMIT and cost['manifest_SHA'] == ACCEPT_MANIFEST, 'accepted cost identity')
    need(cost['used_forward'] == 18 and cost['used_actual_updates'] == 9 and cost['used_GPU_stage_seconds'] == PRIOR['GPU_stage_seconds'], 'accepted cost changed')
    need(not cost['formal_training_authorized'] and not cost['acceptance_states_for_formal'], 'acceptance scope')
    root = Path(ref['path']).parent.parent
    need(sha(root / 'metadata-001/contract.json') == ACCEPT_CONTRACT_SHA, 'short contract file SHA')
    closed = read(root / 'attempt-acceptance-001/CLOSED.json')
    result = read(root / 'attempt-acceptance-001/slim/RESULT.json')
    need(closed['status'] == result['status'] == 'PASS_ENGINEERING', 'short acceptance closure')
    need(closed['ledger'] == result['ledger'] == {'detector_forward': 18, 'actual_updates': 9}, 'short ledger')
    need(read(root / 'attempt-acceptance-001/EXIT.json')['returncode'] == 0, 'short exit')
    for arm in ARMS:
        r = result['arms'][arm]
        need(r['replay_full_state_exact'] and r['logits_dtype'] == 'torch.bfloat16', 'short precision/restore')
    return result


def accepted_storage(ref):
    root = Path(ref['path']).parent
    audit = read(root / 'TERMINAL_AUDIT.json')
    need(audit['status'] == 'VERIFIED_PASS_ENGINEERING' and audit['used_forward'] == 18
         and audit['charged_GPU_stage_seconds_conservative'] == PRIOR['GPU_stage_seconds'], 'accepted storage audit')
    return audit['task_bytes_at_audit']
