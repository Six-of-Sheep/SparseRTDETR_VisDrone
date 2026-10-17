"""Read-only Windows observer; never launches, resumes, or changes remote jobs.

Only bounded status files are replaced during polling. Terminal export uses the
server's existing immutable references, not newly invented file digests.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent
MODES = ('gpu-acceptance', 'formal', 'infer-A', 'accept-A', 'accept-B',
         'accept-C', 'full-C', 'evaluation')
MAX_BYTES = 32 * 1024 ** 2
MAX_FILES = 200
FRESH_SECONDS = 180


class EvidenceError(RuntimeError):
    pass


class NetworkUnavailable(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return dt.datetime.now().astimezone().isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('wb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def exclusive_json(path, value):
    data = json_bytes(value)
    if Path(path).exists():
        require(Path(path).read_bytes() == data, 'different immutable local receipt exists')
        return
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def reference(path):
    p = Path(path)
    data = p.read_bytes()
    return {'path': str(p), 'bytes': len(data), 'sha256': sha(data)}


def local_execution(path):
    path = Path(path)
    execution = json.loads(path.read_text('utf8'))
    require(execution['seeds'] == [1, 2], 'local observer seeds differ')
    require(execution['execution_sha256'] == sha(json.dumps(
        {k: v for k, v in execution.items() if k != 'execution_sha256'}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()),
        'local execution canonical digest differs')
    require(execution['no_retry'] is True and execution['holdout'] is False,
            'execution policy differs')
    return execution, {'path': execution['job_root'] + '/EXECUTION.json',
                       'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}


REMOTE_HEADER = r'''
from pathlib import Path
import base64, hashlib, json, math, os, sys, time
root = Path(CONFIG['root'])
def fail(message): raise RuntimeError(message)
def require(condition,message):
 if not condition: fail(message)
def exact(ref):
 p=Path(ref['path'])
 if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()): fail('reference path escaped')
 data=p.read_bytes()
 if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']: fail('frozen reference bytes/SHA differ: '+str(p))
 return data
execution=json.loads(exact(CONFIG['execution_ref']))
if execution['task_id']!=CONFIG['task_id'] or execution['execution_sha256']!=CONFIG['execution_sha256']: fail('remote execution identity differs')
def ref(path):
 data=path.read_bytes()
 return {'path':path.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
'''


def read_tail(path, expected_identity, digest):
    """Authenticate one complete tail record; no full-chain/sidecar/audit scan."""
    path = Path(path)
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        offset = max(0, info.st_size - 2 * 1024 ** 2)
        stream.seek(offset)
        block = stream.read(2 * 1024 ** 2)
    cursor = {'device': info.st_dev, 'inode': info.st_ino, 'size': info.st_size,
              'seq': None, 'record_sha256': None, 'monotonic': None}
    end = block.rfind(b'\n')
    require(len(block) - end - 1 <= 1024 ** 2, 'partial progress record exceeds 1MiB')
    if end < 0:
        require(offset == 0, 'bounded tail has no complete record delimiter')
        return None, cursor
    start = block.rfind(b'\n', 0, end) + 1
    raw = block[start:end]
    require(0 < len(raw) <= 1024 ** 2 and (offset == 0 or start > 0),
            'complete bounded tail record absent/oversized')
    record = json.loads(raw)
    proof = record.pop('sha256')
    require(proof == digest(record), 'tail record envelope digest differs')
    require(record['identity'] == expected_identity and record['stream'] == 'progress',
            'tail record bound identity/stream differs')
    require(type(record['seq']) is int and record['seq'] > 0
            and type(record['writer_pid']) is int and record['writer_pid'] > 0,
            'tail sequence/writer invalid')
    payload = record['payload']
    require(type(payload.get('monotonic')) in (int, float) and math.isfinite(payload['monotonic']),
            'tail monotonic clock invalid')
    cursor.update(seq=record['seq'], record_sha256=proof, monotonic=payload['monotonic'])
    return payload, cursor


REMOTE_STATUS = r'''
sys.path[:0]=[execution['source_root'], execution['source_root']+'/src']
import new_runtime as rt
rt.install()
from sparse_rtdetr.localization import channel, guard
from sparse_rtdetr.localization.runio import read, digest
launch=json.loads(exact(CONFIG['launch_ref']))
live=guard.identity(launch['pid'])
if live is not None and live['start']!=launch['process_identity']['start']: fail('chain PID reused')
result={'sampled_epoch':time.time(),'execution_sha256':execution['execution_sha256'],
        'chain_alive':live is not None and live['state'] not in ('X','Z'),'chain_process':live,
        'seeds':[],'terminal':None,'slim_export':None,'export_error':None}
for name in ('STOP.json','CHAIN_RESULT.json'):
 p=root/name
 if p.is_file():
  if result['terminal'] is not None: fail('both STOP and completion exist')
  result['terminal']={'name':name,'reference':ref(p),'value':read(p)}
for seedref in execution['contracts']:
 c=json.loads(exact(seedref));row={'seed':c['seed'],'completed_modes':[],'latest':None}
 for mode in ('gpu-acceptance','formal','infer-A','accept-A','accept-B','accept-C','full-C','evaluation'):
  d=Path(c['job_root'])/mode
  if (d/'CLOSED.json').is_file(): row['completed_modes'].append(mode)
  runtime=Path(c['runtime_root'])/mode
  if (runtime/'identity.json').is_file():
   b=read(runtime/'identity.json')
   for k in ('task_id','seed','source_sha256','contract_sha256'):
    if b[k]!=c[k]: fail('progress identity differs: '+k)
   if not (runtime/'progress.jsonl').exists():
    row['latest']={'mode':mode,'progress':None,'tail_scope':'stream not created yet'}
    continue
   event,cursor=read_tail(runtime/'progress.jsonl',digest(b),digest)
   selected=None if event is None else {k:event[k] for k in
     ('event','update','completed_updates','completed_epochs','epoch','finite','loss','wall_epoch','images','forward_calls','pid') if k in event}
   row['latest']={'mode':mode,'progress':selected,'cursor':cursor,
                  'tail_scope':'bounded final complete record digest/identity; not full chain or scientific audit recertification'}
   row.setdefault('stream_cursors',{})[mode]=cursor
   if (runtime/'supervisor-exit.json').is_file():
    ex=read(runtime/'supervisor-exit.json')
    row['latest']['supervisor_exit']={k:ex[k] for k in ('status','returncode') if k in ex}
 result['seeds'].append(row)
if (root/'SLIM_EXPORT.json').is_file(): result['slim_export']=ref(root/'SLIM_EXPORT.json')
if (root/'EXPORT_ERROR.json').is_file(): result['export_error']=read(root/'EXPORT_ERROR.json')
print(json.dumps(result,allow_nan=False))
'''

REMOTE_FETCH = r'''
manifest_ref=CONFIG['manifest_ref']
manifest=json.loads(exact(manifest_ref))
if manifest['task_id']!=execution['task_id'] or manifest['status']!='READY_FOR_WINDOWS_SLIM_RETURN': fail('export identity differs')
by_path={r['path']:r for r in manifest['files']}
if len(by_path)!=len(manifest['files']): fail('duplicate export path')
rows=[]
for expected in CONFIG['requested']:
 if expected!=manifest_ref and by_path.get(expected['path'])!=expected: fail('requested ref not exact export member')
 data=exact(expected)
 rows.append({'reference':expected,'base64':base64.b64encode(data).decode('ascii')})
print(json.dumps({'files':rows}))
'''


def ssh_process_options(platform=None):
    """A detached Windows observer must not create a console for each SSH poll."""
    if (os.name if platform is None else platform) == 'nt':
        return {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)}
    return {}


def remote(config, body, timeout=120):
    command = ('nice -n 19 ionice -c 3 env CUDA_VISIBLE_DEVICES= '
               'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
               'OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ' + config['python'] + ' -B -')
    code = 'CONFIG=' + repr(config) + '\n' + REMOTE_HEADER + inspect.getsource(read_tail) + body
    try:
        process = subprocess.run(['ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                  '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
                                  'visdrone-gpu', command], input=code.encode('utf8'),
                                 capture_output=True, timeout=timeout, **ssh_process_options())
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise NetworkUnavailable(repr(exc)) from exc
    if process.returncode == 255:
        raise NetworkUnavailable(process.stderr.decode('utf8', 'replace')[-2000:])
    require(process.returncode == 0, 'remote read failed: ' + process.stderr.decode('utf8', 'replace')[-4000:])
    require(len(process.stdout) <= 48 * 1024 ** 2, 'remote small-file response exceeded budget')
    try:
        return json.loads(process.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvidenceError('invalid remote response') from exc


def config_for(execution_path, root):
    execution, ref = local_execution(execution_path)
    launch_path = Path(root) / 'LAUNCH_RECEIPT.json'
    if not launch_path.is_file():
        launch_path = Path(root) / 'returned-small/LAUNCH_RECEIPT.json'
    launch_value = json.loads(launch_path.read_text('utf8'))
    require(launch_value['execution_sha256'] == execution['execution_sha256'],
            'local launch receipt execution differs')
    launch = reference(launch_path)
    launch['path'] = execution['job_root'] + '/LAUNCH_RECEIPT.json'
    return {'root': execution['job_root'], 'task_id': execution['task_id'],
            'execution_ref': ref, 'execution_sha256': execution['execution_sha256'],
            'launch_ref': launch, 'python': execution['python']}


def allowed_relative(ref, remote_root):
    path = PurePosixPath(ref['path'])
    require('\\' not in str(path) and '..' not in path.parts, 'unsafe export path')
    try:
        rel = path.relative_to(PurePosixPath(remote_root))
    except ValueError as exc:
        raise EvidenceError('export path outside task') from exc
    name = rel.as_posix()
    root_names = {'PLAN-v002.json', 'PLAN.json', 'EXECUTION.json', 'INVENTORY.json', 'CPU_GATE.json',
                  'CHAIN_STARTED.json', 'CHAIN_RESULT.json', 'STOP.json', 'LAUNCH_RECEIPT.json',
                  'SCIENTIFIC_SERVER_INDEX.json', 'SLIM_EXPORT.json', 'PARENT_REUSE.json',
                  'PROTOCOL_DELTA.json', 'COST.json', 'COST_LEDGER.json', 'SOURCE_DELTA.json',
                  'SOURCE_BINDING.json', 'PARENT_REFERENCES.json', 'CPU_INTEGRATION_GATE.json',
                  'OBSERVER_CPU_GATE.json', 'CPU_CLOSEOUT_GATE.json', 'RESUME_CONTRACT.json',
                  'PREPARED.json', 'CPU_INTEGRATION.json', 'CPU_UNIT_GATE.json', 'SOURCE_GIT.json'}
    allowed = name in root_names or name in {'summary/' + n for n in
              ('AGGREGATE.json', 'PAIRED_SUMMARY.csv', 'REPORT.md', 'RESULT.json')}
    allowed |= len(rel.parts) == 2 and rel.parts[0].startswith('release-') and rel.parts[1] == 'source-manifest.json'
    allowed |= len(rel.parts) == 1 and any(name == f'seed{s}-{m}-{suffix}.json'
        for s in (1, 2) for m in MODES for suffix in ('exit', 'admission', 'complete'))
    for seed in (1, 2):
        prefix = f'seed{seed}/'
        allowed |= name in {prefix + n for n in ('execution-contract.json', 'gpu-execution-contract.json', 'CPU_GATE.json',
                      'FORMAL_PREFLIGHT.json', 'PARENT_REUSE.json', 'formal/binding.json', 'formal/endpoint-lock.json',
                      'evaluation/COMPARISON.json', 'evaluation/TABLE.csv',
                      'formal/checkpoint-epoch-000.pth.complete.json',
                      'formal/checkpoint-epoch-120.pth.complete.json')}
        for mode in MODES:
            allowed |= name in {prefix + mode + '/' + n for n in
                         ('RESULT.json', 'CLOSED.json', 'failure.json',
                          'runtime-archive-index.json', 'runtime-archive/supervisor-exit.json')}
    require(allowed, 'not an explicit local small-file whitelist member: ' + name)
    require(type(ref['bytes']) is int and 0 <= ref['bytes'] <= 4 * 1024 ** 2,
            'individual small-file size exceeded')
    require(isinstance(ref['sha256'], str) and len(ref['sha256']) == 64
            and all(c in '0123456789abcdef' for c in ref['sha256']), 'invalid SHA reference')
    return name


def validate_manifest(manifest, manifest_ref, config):
    require(manifest['task_id'] == config['task_id']
            and manifest['status'] == 'READY_FOR_WINDOWS_SLIM_RETURN', 'manifest identity/status differs')
    require(manifest['not_full_audit_delivery'] is True, 'not a slim export')
    rows = manifest['files']
    require(len(rows) == manifest['files_count'] <= MAX_FILES, 'file count differs/exceeded')
    require(sum(r['bytes'] for r in rows) == manifest['total_bytes'] <= MAX_BYTES, 'byte budget differs/exceeded')
    require(len({r['path'] for r in rows}) == len(rows), 'duplicate export references')
    for ref in [*rows, manifest_ref]:
        allowed_relative(ref, config['root'])
    return [*rows, manifest_ref]


def decode_rows(payload, requested):
    rows = payload.get('files', [])
    require(len(rows) == len(requested), 'transport returned missing/extra files')
    values = {}
    for row, expected in zip(rows, requested):
        require(row['reference'] == expected, 'transport ref differs from frozen export')
        try:
            data = base64.b64decode(row['base64'], validate=True)
        except ValueError as exc:
            raise EvidenceError('invalid base64 transport') from exc
        require(len(data) == expected['bytes'] and sha(data) == expected['sha256'],
                'transport bytes/SHA differ')
        values[expected['path']] = data
    return values


def export_terminal(root, config, snapshot, fetch=remote, receipt_root=None):
    """No remote writes; missing small files are read once against original refs."""
    root = Path(root)
    manifest_ref = snapshot['slim_export']
    require(manifest_ref is not None, 'terminal manifest absent')
    require(manifest_ref['path'] == config['root'] + '/SLIM_EXPORT.json', 'manifest path differs')
    allowed_relative(manifest_ref, config['root'])
    args = {**config, 'manifest_ref': manifest_ref, 'requested': [manifest_ref]}
    data = decode_rows(fetch(args, REMOTE_FETCH), [manifest_ref])[manifest_ref['path']]
    manifest = json.loads(data)
    refs = validate_manifest(manifest, manifest_ref, config)
    terminal_ref = snapshot['terminal']['reference']
    require(terminal_ref in refs, 'terminal receipt absent from frozen slim export')
    require(shutil.disk_usage(root).free >= 1024 ** 3 + 3 * manifest['total_bytes'],
            'D-drive local reserve insufficient')
    out = root / 'returned-small'
    out.mkdir(exist_ok=True)
    pending, local, reused = [], {}, {}
    for ref in refs:
        rel = allowed_relative(ref, config['root'])
        p = out / rel
        require(p.resolve().is_relative_to(out.resolve()), 'local return path escaped')
        local[ref['path']] = p
        if p.exists():
            b = p.read_bytes()
            require(len(b) == ref['bytes'] and sha(b) == ref['sha256'],
                    'would overwrite changed local evidence: ' + str(p))
            reused[ref['path']] = True
        else:
            pending.append(ref)
            reused[ref['path']] = False
    received = {manifest_ref['path']: data}
    needed = [ref for ref in pending if ref != manifest_ref]
    if needed:
        received.update(decode_rows(fetch({**args, 'requested': needed}, REMOTE_FETCH, timeout=180), needed))
    # All transport bytes are verified before any new scientific small-file write.
    for ref in pending:
        p = local[ref['path']]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            stream.write(received[ref['path']])
            stream.flush()
            os.fsync(stream.fileno())
    rows = []
    for ref in refs:
        p = local[ref['path']]
        b = p.read_bytes()
        require(len(b) == ref['bytes'] and sha(b) == ref['sha256'], 'final local SHA mismatch')
        rows.append({'reference': ref, 'local': str(p), 'SHA_PASS': True, 'reused': reused[ref['path']]})
    receipt = {'status': 'PASS', 'task_id': config['task_id'], 'files': rows,
               'files_count': len(rows), 'total_bytes': sum(r['bytes'] for r in refs),
               'new_files': len(pending), 'new_bytes': sum(r['bytes'] for r in pending),
               'original_manifest_reference': manifest_ref, 'terminal_reference': terminal_ref,
               'not_full_delivery': True, 'large_scientific_originals': 'SERVER_ONLY',
               'remote_large_SHA_recomputed': False}
    exclusive_json(Path(receipt_root or root) / 'OBSERVER_TERMINAL_RETURN.json', receipt)
    return receipt


def costs_from_small_files(root):
    root = Path(root) / 'returned-small'
    records = []
    for seed in (1, 2):
        for mode in MODES:
            p = root / f'seed{seed}' / mode / 'RESULT.json'
            if not p.is_file():
                continue
            value = json.loads(p.read_text('utf8'))
            fields = ('status', 'actual_updates', 'updates', 'training_wall_seconds', 'elapsed_seconds',
                      'complete_update_seconds', 'model_batch_forwards', 'forward_calls',
                      'new_score_budget_calls', 'peak_memory_allocated_bytes', 'peak_memory_reserved_bytes',
                      'peak_allocated_bytes', 'peak_reserved_bytes')
            row = {'seed': seed, 'mode': mode, 'fields': {k: value[k] for k in fields if k in value},
                   'reference_source': str(p)}
            if 'timing' in value:
                row['fields']['e2e_total_seconds'] = value['timing']['e2e_total_seconds']
            records.append(row)
    return {'scope': 'existing authenticated mode result summaries; no scientific recomputation',
            'modes': records, 'missing_mode_results': 16 - len(records),
            'not_total_historical_project_cost': True, 'previous_costs_preserved': True}


def terminal_text(root, snapshot, receipt):
    complete = snapshot['terminal']['name'] == 'CHAIN_RESULT.json'
    aggregate_path = Path(root) / 'returned-small/summary/AGGREGATE.json'
    decision = '未形成三seed裁决'
    if complete:
        require(aggregate_path.is_file(), 'completed chain lacks aggregate')
        aggregate = json.loads(aggregate_path.read_text('utf8'))
        require(aggregate['seeds'] == [0, 1, 2], 'aggregate seed identities differ')
        decision = ('跨seed冻结裁决' + aggregate['decision'] + '；七项收益/保护逐seed要求及C/A≤6未改，'
                    '均值不抵消失败；逐项、mean/sampleSD(ddof1)/worst和C−B/APL见summary。')
    else:
        stop = snapshot['terminal']['value']
        decision = ('未形成三seed裁决；STOP留证，不自动重试；seed=' + str(stop.get('seed')) + '，阶段=' + str(stop.get('mode'))
                    + '，错误=' + str(stop.get('error'))[:600] + '。已完成阶段保留，未完成不冒充终点。')
    return (f'固定SAHI式seed1/2执行链{"完成" if complete else "停止"}。{decision}'
            f'本机{receipt["files_count"]}必要小文件/{receipt["total_bytes"]}bytes逐SHA通过，'
            '完整checkpoint、预测、逐窗审计与日志仅实验室保存；索引不是独立备份或全量回传。'
            '未追加seed/参数/holdout，旧结果和失败成本保持。终态及成本见任务目录observers/本次身份下的'
            'OBSERVER_TERMINAL_RETURN.json、OBSERVER_COST_SUMMARY.json及returned-small/summary。AutoDL建议无需开机，实际模式/计费本轮未核验。')


def append_once_fresh(path, marker, text):
    """Append after fresh read. Never replace another executor's current text."""
    path = Path(path)
    previous = path.read_bytes() if path.exists() else b''
    if marker.encode('utf8') in previous:
        return False
    data = ('\n\n' + text + '\n').encode('utf8')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, 'O_BINARY', 0), 0o666)
    try:
        written = os.write(fd, data)
        require(written == len(data), 'handoff append incomplete')
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def prepend_fresh(path, marker, text, before_commit=None):
    """New-read optimistic merge. A concurrent content change is reread, never lost."""
    path = Path(path)
    for attempt in range(8):
        old = path.read_bytes()
        if marker.encode('utf8') in old:
            return False
        line_end = old.find(b'\n')
        require(line_end >= 0, 'CURRENT lacks its title line')
        data = old[:line_end + 1] + ('\n\n' + text + '\n\n').encode('utf8') + old[line_end + 1:]
        temp = path.with_name(path.name + '.' + str(os.getpid()) + '.terminal-observer.tmp')
        with temp.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if before_commit:
            before_commit(attempt)
        if path.read_bytes() != old:
            continue
        os.replace(temp, path)
        return True
    raise EvidenceError('CURRENT changed repeatedly; stop local merge without stale replacement')


def merge_handoff(project, root, snapshot, receipt):
    marker = '<!-- terminal-observer:' + Path(root).name + ' -->'
    at = now()
    text = terminal_text(root, snapshot, receipt)
    title = f'## {at} · Codex 固定SAHI式三seed自动收尾\n\n{marker}\n\n{text}'
    current = Path(project) / 'collaboration/CURRENT.md'
    history = Path(project) / 'collaboration/HISTORY.md'
    # Keep the live summary newest-first. Re-read on a concurrent change rather
    # than replacing the document using the observer's launch-time copy.
    prepend_fresh(current, marker, title)
    append_once_fresh(history, marker, f'## {at} · Codex\n\n{marker}\n\n{text}\n\n证据：{root}')


def classify(snapshot):
    if snapshot['terminal']:
        if snapshot['slim_export']:
            return 'TERMINAL_READY'
        if snapshot['chain_alive']:
            return 'TERMINAL_EXPORT_CLOSING'
        return 'TERMINAL_EXPORT_MISSING'
    if not snapshot['chain_alive']:
        return 'CHAIN_ABSENT_WITHOUT_TERMINAL'
    return 'REMOTE_CHAIN_RUNNING'


def check_cursors(previous, snapshot):
    updated = dict(previous)
    for seed in snapshot['seeds']:
        for mode, current in seed.get('stream_cursors', {}).items():
            key = str(seed['seed']) + ':' + mode
            old = previous.get(key)
            if old:
                require(current['device'] == old['device'] and current['inode'] == old['inode']
                        and current['size'] >= old['size'], 'observed stream replaced/truncated')
                if old['seq'] is not None:
                    require(current['seq'] is not None and current['seq'] >= old['seq']
                            and current['monotonic'] >= old['monotonic'], 'observed stream regressed')
                    if current['seq'] == old['seq']:
                        require(current['record_sha256'] == old['record_sha256'], 'observed tail mutated')
            updated[key] = current
    return updated


def hold_singleton(root):
    stream = (Path(root) / 'OBSERVER.singleton').open('a+b')
    if stream.tell() == 0:
        stream.write(b'1')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise EvidenceError('local observer already holds the singleton lock')
    return stream


def snapshot_age(snapshot, checked_epoch=None):
    """Age is local receipt age, never inferred from a remote machine clock."""
    if not snapshot or type(snapshot.get("received_epoch")) not in (int, float):
        return None
    at = time.time() if checked_epoch is None else checked_epoch
    age = at - snapshot["received_epoch"]
    return max(0.0, age) if math.isfinite(age) else None


def cached_state(snapshot, checked_epoch=None):
    age = snapshot_age(snapshot, checked_epoch)
    return {"freshness": "unknown" if age is None else ("stale" if age > FRESH_SECONDS else "fresh"),
            "age_seconds": age, "cached_only": True,
            "remote_status": "UNKNOWN_UNTIL_FRESH_READ"}


def fresh_status(config, previous=None, fetch=None):
    """Always performs a new read. A network error never returns old RUNNING."""
    started = time.time()
    fetch = remote if fetch is None else fetch
    try:
        snapshot = fetch(config, REMOTE_STATUS)
    except NetworkUnavailable as exc:
        return {"checked_at": now(), "checked_epoch": time.time(), "freshness": "unknown",
                "status": "NETWORK_UNAVAILABLE_REMOTE_STATUS_UNKNOWN", "age_seconds": None,
                "fresh_until_epoch": None,
                "last_success": cached_state(previous), "network_error": str(exc)[-2000:],
                "remote_training_failure_not_inferred": True, "snapshot": None}
    received = time.time()
    require(type(snapshot.get("sampled_epoch")) in (int, float)
            and math.isfinite(snapshot["sampled_epoch"]), "remote sample clock invalid")
    require(snapshot["execution_sha256"] == config["execution_sha256"], "snapshot execution differs")
    return {"checked_at": now(), "checked_epoch": received, "freshness": "fresh",
            "age_seconds": 0.0, "received_epoch": received,
            "fresh_until_epoch": received + FRESH_SECONDS,
            "request_started_epoch": started, "request_seconds": received - started,
            "status": classify(snapshot), "snapshot": snapshot,
            "scope": "new read and bounded progress-tail identity; not full audit recertification"}


def process_identity():
    result = {"pid": os.getpid(), "parent_pid": os.getppid()}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(created),
                ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        result["creation_100ns"] = (created.dwHighDateTime << 32) | created.dwLowDateTime
    else:
        result["proc_start_ticks"] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    return result


def run_observer(config, local_root, project, observer_id, interval=120, fetch=None):
    local_root = Path(local_root)
    fetch = remote if fetch is None else fetch
    local_root.mkdir(parents=True, exist_ok=True)
    singleton = hold_singleton(local_root)
    folder = local_root / "observers" / observer_id
    try:
        folder.mkdir(parents=True, exist_ok=False)
        exclusive_json(folder / "OBSERVER_STARTED.json", {
            "status": "READ_ONLY_OBSERVER_STARTED", "started_at": now(),
            "process_identity": process_identity(), "observer_id": observer_id,
            "interval_seconds": interval, "task_id": config["task_id"],
            "execution_reference": config["execution_ref"], "remote_launcher_calls": 0,
            "remote_writes": 0, "server_chain_independent": True})
        state = {"polls": 0, "network_errors": 0, "last_success_at": None}
        cursors, previous, prior_label = {}, None, None
        try:
            while True:
                view = fresh_status(config, previous, fetch)
                state["polls"] += 1
                if view["freshness"] == "fresh":
                    snapshot = view["snapshot"]
                    cursors = check_cursors(cursors, snapshot)
                    previous = {"received_epoch": view["received_epoch"]}
                    state["last_success_at"] = view["checked_at"]
                else:
                    state["network_errors"] += 1
                # This file changes on *every* poll, including a failed read.
                # A stopped observer is identified by its timestamp age.
                atomic_json(folder / "OBSERVER_LATEST.json", view)
                atomic_json(folder / "OBSERVER_STATE.json", {**state, **{
                    k: view[k] for k in ("checked_at", "checked_epoch", "freshness", "age_seconds", "fresh_until_epoch", "status")}})
                label = view["status"]
                if label != prior_label:
                    print(json.dumps({"at": now(), "observer_status": label,
                                      "freshness": view["freshness"]}), flush=True)
                    prior_label = label
                if view["freshness"] == "fresh":
                    if label == "TERMINAL_READY":
                        receipt = export_terminal(local_root, config, snapshot, fetch, folder)
                        exclusive_json(folder / "OBSERVER_COST_SUMMARY.json", costs_from_small_files(local_root))
                        merge_handoff(project, local_root, snapshot, receipt)
                        exclusive_json(folder / "OBSERVER_COMPLETE.json", {
                            "status": "SLIM_RETURN_AND_HANDOFF_COMPLETE", "finished_at": now(),
                            "terminal": snapshot["terminal"]["name"],
                            "return_receipt": reference(folder / "OBSERVER_TERMINAL_RETURN.json"),
                            "remote_large_materials": "SERVER_ONLY", "server_chain_restarted": False})
                        return 0
                    require(label not in ("TERMINAL_EXPORT_MISSING", "CHAIN_ABSENT_WITHOUT_TERMINAL"),
                            label + "; observer does not repair/relaunch the server chain")
                time.sleep(interval)
        except BaseException as exc:
            exclusive_json(folder / "OBSERVER_STOP.json", {
                "status": "LOCAL_OBSERVER_STOP_NO_REMOTE_ACTION", "at": now(),
                "error": repr(exc), "last_state": state,
                "remote_training_failure_not_inferred": True, "remote_retry_or_restart": False})
            raise
    finally:
        singleton.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("watch", "status"), nargs="?", default="watch")
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--observer-id", default=None)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args(argv)
    require(90 <= args.interval <= 120, "poll interval outside fixed low-load range")
    require(args.local_root.resolve().is_relative_to(args.project.resolve()), "local output outside project")
    config = config_for(args.execution, args.local_root)
    if args.mode == "status":
        # Deliberately does not read OBSERVER_LATEST as a live source.
        view = fresh_status(config)
        print(json.dumps(view, ensure_ascii=False, allow_nan=False))
        return 0 if view["freshness"] == "fresh" else 2
    observer_id = args.observer_id or (dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    require(observer_id and all(c.isalnum() or c in "-_" for c in observer_id), "invalid observer identity")
    return run_observer(config, args.local_root, args.project, observer_id, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
