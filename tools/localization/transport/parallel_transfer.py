"""At most four immutable byte requests; commit in order under the original SHA."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time

from local_ops import BOOTSTRAP, REMOTE, SSH_OPTIONS, remote_python, sha, write
import streaming_transfer as S

WORKERS = 4


def _phase_hook(phase, index):
    """No-op seam for deterministic local concurrency checks."""
    pass


def download(remote, expected_sha, dest, receipts, *, expected_size=None):
    S.require(remote.startswith(REMOTE + '/') and '..' not in PurePosixPath(remote).parts, 'remote scope')
    dest, receipts = Path(dest), Path(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return S.download(remote, expected_sha, dest, receipts, expected_size=expected_size)
    started = time.time()
    tag = str(time.time_ns())
    code = f'''import hashlib,json,pathlib
p=pathlib.Path({remote!r});assert p.is_file() and not p.is_symlink()
h=hashlib.sha256();chunks=[];offset=0
with p.open('rb') as f:
 while True:
  data=f.read({S.BLOCK})
  if not data:break
  h.update(data);chunks.append(dict(offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));offset+=len(data)
assert h.hexdigest()=={expected_sha!r},'original file SHA differs'
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),chunks=chunks)),flush=True)
'''
    manifest = S.validate_manifest(json.loads(remote_python(code, receipts / (tag + '-manifest.json'), 90)),
        remote, expected_sha, expected_size)
    part = dest.with_name(dest.name + '.stream-v1.partial')
    done = part.stat().st_size if part.exists() else 0
    S.require(done <= manifest['bytes'], 'partial larger than original file')
    if done:
        cursor = 0
        with part.open('rb') as stream:
            for chunk in manifest['chunks']:
                if cursor == done:
                    break
                S.require(cursor + chunk['bytes'] <= done, 'partial ends inside an unverified block')
                data = stream.read(chunk['bytes'])
                S.require(len(data) == chunk['bytes'] and hashlib.sha256(data).hexdigest() == chunk['sha256'], 'partial prefix SHA differs')
                cursor += len(data)
        S.require(cursor == done, 'partial prefix coverage')
    start_record = dict(remote_path=remote, expected_sha256=expected_sha, expected_bytes=manifest['bytes'],
        verified_existing_prefix_bytes=done, block_bytes=S.BLOCK, maximum_concurrent_requests=WORKERS,
        started_epoch=started, training_control=False, original_partial_retained=str(dest.with_name(dest.name + '.partial')))
    write(receipts / (tag + '-start.json'), start_record)
    stop = threading.Event()
    failure_lock = threading.Lock()
    first_failure = {}

    def receive(*args):
        S.require(not stop.is_set(), 'transport group cancelled; no new request')
        value = S.receive(*args)
        S.require(not stop.is_set(), 'transport group cancelled; received bytes retained but not committed')
        return value

    def retry_delay(seconds):
        _phase_hook('network_retry_wait', None)
        S.require(not stop.wait(seconds), 'transport group cancelled during network retry delay')

    def fetch(index, chunk):
        folder = receipts / (tag + '-block-%04d' % index)
        code = f'''import pathlib,hashlib,sys
p=pathlib.Path({remote!r});assert p.is_file() and not p.is_symlink() and p.stat().st_size=={manifest['bytes']}
with p.open('rb') as f:f.seek({chunk['offset']});data=f.read({chunk['bytes']})
assert len(data)=={chunk['bytes']} and hashlib.sha256(data).hexdigest()=={chunk['sha256']!r},'original block SHA differs'
sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''
        try:
            payload, result = S.receive_retried(['ssh.exe', *SSH_OPTIONS, 'autodl-c2', BOOTSTRAP + ' -B -'],
                code, folder, chunk['bytes'], chunk['sha256'], receiver=receive, sleep=retry_delay)
            return payload, folder
        except BaseException as error:
            with failure_lock:
                if not first_failure:
                    first_failure.update(block=index, error=repr(error), error_type=type(error).__name__, epoch=time.time())
                stop.set()
            _phase_hook('failure_published', index)
            raise

    remaining = [(i, c) for i, c in enumerate(manifest['chunks']) if c['offset'] >= done]
    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix='immutable-transfer')
    pending = {}
    next_submit = 0
    try:
        with failure_lock:
            for next_submit in range(min(WORKERS, len(remaining))):
                index, chunk = remaining[next_submit]
                pending[index] = pool.submit(fetch, index, chunk)
        next_submit = min(WORKERS, len(remaining))
        with part.open('ab') as output:
            for index, chunk in remaining:
                payload, folder = pending.pop(index).result()
                data = payload.read_bytes()
                S.require(chunk['offset'] == done and len(data) == chunk['bytes']
                    and hashlib.sha256(data).hexdigest() == chunk['sha256'], 'received block changed before ordered commit')
                _phase_hook('before_commit', index)
                # Failure publication and commit/schedule admission share this lock.
                # No network operation is awaited while it is held.
                with failure_lock:
                    S.require(not stop.is_set(), 'transport group stopped before commit')
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                    done += len(data)
                    write(folder / 'commit.json', dict(status='VERIFIED_BLOCK_COMMITTED', remote_path=remote,
                        offset=chunk['offset'], committed_bytes=len(data), total_verified_bytes=done,
                        sha256=chunk['sha256'], received_result=str(payload.parent / 'result.json'), epoch=time.time()))
                _phase_hook('before_schedule', index)
                with failure_lock:
                    S.require(not stop.is_set(), 'transport group stopped before scheduling')
                    if next_submit < len(remaining):
                        index, chunk = remaining[next_submit]
                        pending[index] = pool.submit(fetch, index, chunk)
                        next_submit += 1
    except BaseException as error:
        with failure_lock:
            stop.set()
        write(receipts / (tag + '-abort.json'), dict(status='TRANSFER_STOP_PRESERVE', error=repr(error),
            first_request_failure=first_failure, verified_partial_bytes=done, committed_target=False,
            epoch=time.time(), training_control=False))
        raise
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        exits, unconfirmed = [], []
        for request in sorted(receipts.glob(tag + '-block-*/request-*')):
            receipt = request / 'subprocess-exit.json'
            try:
                value = json.loads(receipt.read_text(encoding='utf-8'))
                S.require(value['exited'] is True and type(value['returncode']) is int and type(value['pid']) is int,
                    'invalid subprocess exit proof')
                exits.append(dict(path=str(receipt), sha256=sha(receipt), pid=value['pid'], returncode=value['returncode']))
            except (OSError, ValueError, KeyError, TypeError, S.TransferIntegrityError) as error:
                unconfirmed.append(dict(request=str(request), error=repr(error)))
        write(receipts / (tag + '-requests-exited.json'), dict(
            status='ALL_OWN_SUBPROCESS_EXITS_CONFIRMED' if not unconfirmed else 'THREADS_JOINED_SUBPROCESS_CLEANUP_UNCONFIRMED',
            threads_joined=True, subprocess_cleanup_confirmed=not unconfirmed, subprocess_exit_receipts=exits,
            unconfirmed_requests=unconfirmed, finished_epoch=time.time(), maximum_concurrency=WORKERS, training_control=False))
        S.require(not unconfirmed, 'subprocess cleanup unconfirmed; do not continue transport')
    S.require(done == manifest['bytes'] and part.stat().st_size == manifest['bytes'] and sha(part) == expected_sha,
        'complete original Windows SHA differs')
    S.require(not dest.exists(), 'immutable target appeared concurrently')
    part.rename(dest)
    result = dict(status='WINDOWS_SHA_PASS', remote_path=remote, path=str(dest), bytes=done, sha256=expected_sha,
        verified_epoch=time.time(), started_epoch=started, elapsed_seconds=time.time() - started,
        transport='4x1MiB_STDIN_SCRIPT_DIRECT_FILE_BYTE_PROGRESS_ORDERED_COMMIT', maximum_concurrent_requests=WORKERS,
        request_cap_seconds=120, byte_idle_cap_seconds=45)
    write(receipts / (tag + '-complete.json'), result)
    return result
