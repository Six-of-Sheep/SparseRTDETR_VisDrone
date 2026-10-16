"""Bounded immutable transport with byte-progress evidence; never loads a model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

from local_ops import BOOTSTRAP, REMOTE, SSH_OPTIONS, remote_python, sha, write

BLOCK = 1024 * 1024


class TransferIntegrityError(RuntimeError):
    pass


class NetworkRequestError(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise TransferIntegrityError(reason)


def validate_manifest(value, remote, expected_sha, expected_size=None):
    require(isinstance(value, dict), 'manifest must be an object')
    require(value.get('path') == remote and value.get('sha256') == expected_sha, 'manifest identity changed')
    total = value.get('bytes')
    require(type(total) is int and total >= 0, 'manifest size')
    require(expected_size is None or total == expected_size, 'published file size changed')
    require(isinstance(value.get('chunks'), list), 'manifest chunks')
    cursor = 0
    for row in value['chunks']:
        require(isinstance(row, dict), 'manifest chunk object')
        require(type(row.get('offset')) is int and row['offset'] == cursor, 'chunk offset continuity')
        length = min(BLOCK, total - cursor)
        require(length > 0 and type(row.get('bytes')) is int and row['bytes'] == length, 'chunk size')
        digest = row.get('sha256')
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'chunk SHA syntax')
        cursor += length
    require(cursor == total, 'manifest coverage')
    return value


def known_network_stderr(text):
    value = text.lower()
    if 'traceback (most recent call last)' in value:
        return False
    return any(token in value for token in ('banner exchange', 'connection aborted', 'connection reset',
        'connection timed out', 'connection refused', 'connection closed', 'broken pipe',
        'no route to host', 'network is unreachable', 'could not resolve hostname'))


def receive(command, code, folder, expected_bytes, expected_sha, *, request_cap=120, idle_cap=45):
    """One subprocess request. Only complete, matching bytes plus rc0 are accepted."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'remote-read.py').write_text(code, encoding='utf-8')
    started = time.time()
    tick = time.monotonic()
    last_change = tick
    last_bytes = 0
    cap = None
    with (folder / 'remote-read.py').open('rb') as script, (folder / 'received.bin').open('xb', buffering=0) as received, (folder / 'stderr.log').open('xb', buffering=0) as stderr:
        proc = subprocess.Popen(command, stdin=script, stdout=received, stderr=stderr)
        try:
            write(folder / 'process.json', dict(pid=proc.pid, started_epoch=time.time(), training_control=False,
                requested_bytes=expected_bytes, request_cap_seconds=request_cap, byte_idle_cap_seconds=idle_cap,
                stdin_source='saved_script_file'))
            with (folder / 'byte-progress.jsonl').open('x', encoding='utf-8') as progress:
                while proc.poll() is None:
                    size = os.fstat(received.fileno()).st_size
                    current = time.monotonic()
                    if size != last_bytes:
                        last_bytes = size
                        last_change = current
                        progress.write(json.dumps(dict(epoch=time.time(), received_bytes=size)) + '\n')
                        progress.flush()
                    if size > expected_bytes:
                        cap = 'OVERSIZE_INTEGRITY_FAILURE'
                        break
                    if current - tick >= request_cap or current - last_change >= idle_cap:
                        cap = 'REQUEST_CAP' if current - tick >= request_cap else 'BYTE_IDLE_CAP'
                        break
                    time.sleep(0.2)
        finally:
            killed = False
            try:
                if proc.poll() is None:
                    proc.kill()
                    killed = True
                returncode = proc.wait(timeout=10)
            except BaseException as error:
                raise TransferIntegrityError('owned transport subprocess cleanup unconfirmed; no retry') from error
            write(folder / 'subprocess-exit.json', dict(pid=proc.pid, returncode=returncode,
                exited=True, killed=killed, observed_epoch=time.time(), training_control=False))
        received_bytes = os.fstat(received.fileno()).st_size
    data_sha = sha(folder / 'received.bin')
    error_text = (folder / 'stderr.log').read_text(encoding='utf-8', errors='replace')
    complete_match = received_bytes == expected_bytes and data_sha == expected_sha
    if received_bytes > expected_bytes or (received_bytes == expected_bytes and data_sha != expected_sha):
        status = 'INTEGRITY_FAILURE_NO_RETRY'
    elif cap:
        status = 'NETWORK_REQUEST_TIMEOUT'
    elif returncode != 0:
        status = 'NETWORK_PROCESS_FAILURE' if known_network_stderr(error_text) else 'REMOTE_PROCESS_FAILURE_NO_RETRY'
    elif not complete_match:
        status = 'INTEGRITY_FAILURE_NO_RETRY'
    else:
        status = 'COMPLETE_BLOCK_SHA_PASS'
    result = dict(status=status, started_epoch=started, finished_epoch=time.time(), elapsed_seconds=time.time() - started,
        requested_bytes=expected_bytes, received_bytes=received_bytes, committed_bytes=0,
        expected_sha256=expected_sha, received_sha256=data_sha, returncode=returncode, timeout_kind=cap,
        stdin_source='saved_script_file', stderr_sha256=sha(folder / 'stderr.log'), subprocess_exited=True,
        last_observed_byte_change_elapsed_seconds=last_change - tick, training_control=False)
    write(folder / 'result.json', result)
    if status in ('NETWORK_REQUEST_TIMEOUT', 'NETWORK_PROCESS_FAILURE'):
        raise NetworkRequestError(str(folder / 'result.json'))
    require(status == 'COMPLETE_BLOCK_SHA_PASS', status + ': ' + str(folder / 'result.json'))
    return result


def receive_retried(command, code, folder, expected_bytes, expected_sha, *, receiver=receive, sleep=time.sleep):
    for attempt in range(3):
        request = Path(folder) / ('request-%d' % attempt)
        try:
            result = receiver(command, code, request, expected_bytes, expected_sha)
            return request / 'received.bin', result
        except NetworkRequestError:
            if attempt == 2:
                raise
            sleep((10, 30)[attempt])


def download(remote, expected_sha, dest, receipts, *, expected_size=None):
    require(remote.startswith(REMOTE + '/') and '..' not in PurePosixPath(remote).parts, 'remote scope')
    dest, receipts = Path(dest), Path(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        require(expected_size is None or dest.stat().st_size == expected_size, 'existing Windows size differs')
        require(sha(dest) == expected_sha, 'existing Windows SHA differs')
        return dict(status='EXISTING_SHA_PASS', path=str(dest), bytes=dest.stat().st_size, sha256=expected_sha)
    started = time.time()
    tag = str(time.time_ns())
    code = f'''import hashlib,json,pathlib
p=pathlib.Path({remote!r});assert p.is_file() and not p.is_symlink()
h=hashlib.sha256();chunks=[];offset=0
with p.open('rb') as f:
 while True:
  data=f.read({BLOCK})
  if not data:break
  h.update(data);chunks.append(dict(offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));offset+=len(data)
assert h.hexdigest()=={expected_sha!r},'original file SHA differs'
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),chunks=chunks)),flush=True)
'''
    manifest = validate_manifest(json.loads(remote_python(code, receipts / (tag + '-manifest.json'), 90)),
        remote, expected_sha, expected_size)
    part = dest.with_name(dest.name + '.stream-v1.partial')
    done = part.stat().st_size if part.exists() else 0
    require(done <= manifest['bytes'], 'partial larger than original file')
    if done:
        cursor = 0
        with part.open('rb') as stream:
            for chunk in manifest['chunks']:
                if cursor == done:
                    break
                require(cursor + chunk['bytes'] <= done, 'partial ends inside an unverified block')
                data = stream.read(chunk['bytes'])
                require(len(data) == chunk['bytes'] and hashlib.sha256(data).hexdigest() == chunk['sha256'], 'partial prefix SHA differs')
                cursor += len(data)
        require(cursor == done, 'partial prefix coverage')
    write(receipts / (tag + '-start.json'), dict(remote_path=remote, expected_sha256=expected_sha,
        expected_bytes=manifest['bytes'], verified_existing_prefix_bytes=done, block_bytes=BLOCK,
        original_partial_retained=str(dest.with_name(dest.name + '.partial')), started_epoch=started, training_control=False))
    with part.open('ab') as output:
        for index, chunk in enumerate(manifest['chunks']):
            if chunk['offset'] < done:
                continue
            require(chunk['offset'] == done, 'commit offset continuity')
            code = f'''import pathlib,hashlib,sys
p=pathlib.Path({remote!r});assert p.is_file() and not p.is_symlink() and p.stat().st_size=={manifest['bytes']}
with p.open('rb') as f:f.seek({chunk['offset']});data=f.read({chunk['bytes']})
assert len(data)=={chunk['bytes']} and hashlib.sha256(data).hexdigest()=={chunk['sha256']!r},'original block SHA differs'
sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''
            block_folder = receipts / (tag + '-block-%04d' % index)
            received, request = receive_retried(['ssh.exe', *SSH_OPTIONS, 'autodl-c2', BOOTSTRAP + ' -B -'],
                code, block_folder, chunk['bytes'], chunk['sha256'])
            with received.open('rb') as source:
                data = source.read()
            require(len(data) == chunk['bytes'] and hashlib.sha256(data).hexdigest() == chunk['sha256'], 'received block changed before commit')
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            done += len(data)
            write(block_folder / 'commit.json', dict(status='VERIFIED_BLOCK_COMMITTED', remote_path=remote,
                offset=chunk['offset'], committed_bytes=len(data), total_verified_bytes=done,
                sha256=chunk['sha256'], received_result=str(received.parent / 'result.json'), epoch=time.time()))
    require(done == manifest['bytes'] and part.stat().st_size == manifest['bytes'] and sha(part) == expected_sha,
        'complete original Windows SHA differs')
    require(not dest.exists(), 'immutable target appeared concurrently')
    part.rename(dest)
    result = dict(status='WINDOWS_SHA_PASS', remote_path=remote, path=str(dest), bytes=done, sha256=expected_sha,
        verified_epoch=time.time(), started_epoch=started, elapsed_seconds=time.time() - started,
        transport='1MiB_STDIN_SCRIPT_DIRECT_FILE_BYTE_PROGRESS', request_cap_seconds=120, byte_idle_cap_seconds=45)
    write(receipts / (tag + '-complete.json'), result)
    return result
