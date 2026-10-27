"""Strict one-shot binary stdin transport, outside the frozen experiment release."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def validate(payload):
    if not isinstance(payload, bytes):
        raise TypeError('payload must be bytes')
    if payload.startswith(b'\xef\xbb\xbf') or b'\r' in payload or b'\x00' in payload:
        raise ValueError('payload must be UTF-8 without BOM, CR or NUL')
    text = payload.decode('utf-8', errors='strict')
    if not text or not text.endswith('\n') or '\ufeff' in text:
        raise ValueError('payload must be nonempty UTF-8/LF without BOM and end with LF')
    return hashlib.sha256(payload).hexdigest()


def encode(text):
    payload = text.encode('utf-8', errors='strict')
    validate(payload)
    return payload


def shell(cwd, argv, stdin_file=None):
    """Generate LF bytes directly, never via platform text-mode output."""
    if not Path(cwd).is_absolute() or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError('absolute cwd and nonempty string argv required')
    command = shlex.join(argv)
    if stdin_file is not None:
        command += ' < ' + shlex.quote(str(stdin_file))
    return encode('set -euo pipefail\ncd -- ' + shlex.quote(str(cwd)) + '\nexec ' + command + '\n')


def write_payload(path, payload):
    digest = validate(payload)
    with Path(path).open('xb') as f:
        f.write(payload)
    if Path(path).read_bytes() != payload:
        raise ValueError('payload readback mismatch')
    return digest


def _json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as f:
        f.write(data)


def send(path, receiver, receipt, cwd=None):
    """Exactly one subprocess call; reject invalid bytes BEFORE any invocation.

    receiver is ['bash','-s'] in integration tests or configured SSH argv in
    production. No decoding/re-encoding, newline conversion, or retries.
    """
    payload = Path(path).read_bytes()
    digest = validate(payload)
    receipt = Path(receipt)
    paths = {name:Path(str(receipt) + suffix) for name, suffix in
             [('begin','.begin.json'),('out','.stdout.bin'),('err','.stderr.bin'),('result','.result.json')]}
    if any(p.exists() or p.is_symlink() for p in paths.values()):
        raise FileExistsError('existing transport receipt; no replay')
    _json(paths['begin'], {'argv':list(receiver),'cwd':None if cwd is None else str(cwd),
                           'payload_path':str(path),'payload_bytes':len(payload),
                           'payload_sha256':digest,'binary_stdin':True,'retry':False})
    started = time.monotonic()
    result = subprocess.run(receiver, input=payload, cwd=cwd, capture_output=True, text=False)
    for name, blob in [('out',result.stdout),('err',result.stderr)]:
        with paths[name].open('xb') as f:
            f.write(blob)
    value = {'returncode':result.returncode,'elapsed_seconds':time.monotonic()-started,
             'sent_payload_sha256':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload),
             'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
             'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),
             'stdout':result.stdout.decode('utf-8', errors='replace'),
             'stderr':result.stderr.decode('utf-8', errors='replace'),'retry':False}
    _json(paths['result'], value)
    return value
