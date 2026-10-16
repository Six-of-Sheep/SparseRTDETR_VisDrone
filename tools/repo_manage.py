"""Small, standard-library-only Git checks. Never launches an experiment."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

MAX_BYTES = 2 * 1024 * 1024
ARTIFACT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.onnx', '.engine', '.pkl', '.npy', '.npz', '.log', '.tar', '.gz', '.zip', '.pem', '.key'}
ROOT_ARTIFACT_DIRS = {'data', 'datasets', 'runs', 'outputs', 'artifacts', 'weights', 'checkpoints', 'logs', 'local', 'tmp', 'scratch'}
SECRET_PATTERNS = [
    rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    rb'gh[pousr]_[A-Za-z0-9]{30,}',
    rb'github_pat_[A-Za-z0-9_]{50,}',
    rb'AKIA[0-9A-Z]{16}',
]

def git(repo, *args, data=None):
    p = subprocess.run(['git', '-C', str(repo), *args], input=data, capture_output=True)
    if p.returncode:
        raise ValueError(p.stderr.decode('utf-8', 'replace').strip() or 'Git command failed')
    return p.stdout

def relative_path(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or '\n' in name or '\r' in name or p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('Unsafe relative path: ' + repr(name))
    return p

def objects(repo, specs):
    if not specs:
        return []
    raw = git(repo, 'cat-file', '--batch', data=('\n'.join(specs) + '\n').encode('utf-8'))
    out, offset = [], 0
    for spec in specs:
        end = raw.index(b'\n', offset)
        parts = raw[offset:end].split()
        if len(parts) != 3 or parts[1] != b'blob':
            raise ValueError('Missing or non-blob object: ' + spec)
        size = int(parts[2]); start = end + 1
        out.append(raw[start:start + size]); offset = start + size + 1
    return out

def file_problems(name, blob, mode='100644'):
    p = relative_path(name); issues = []
    if mode not in ('100644', '100755'):
        issues.append('symlinks/submodules need an explicit dependency review')
    if p.parts[0].lower() in ROOT_ARTIFACT_DIRS or p.suffix.lower() in ARTIFACT_SUFFIXES:
        issues.append('runtime artifact or private key is outside the source allowlist')
    if any(part.lower() in {'.ssh', '.venv', '__pycache__', '.claude', '.codex'} for part in p.parts):
        issues.append('machine-local directory')
    if p.name == '.env' or p.name.startswith('.env.') or p.name in {'id_rsa', 'id_ed25519'}:
        issues.append('environment/credential file')
    if len(blob) > MAX_BYTES:
        issues.append('file exceeds the 2 MiB source review limit')
    if any(re.search(pattern, blob) for pattern in SECRET_PATTERNS):
        issues.append('credential-shaped content; value deliberately not printed')
    return issues

def check(repo, staged=False):
    entries = []
    for record in git(repo, 'ls-files', '--stage', '-z').split(b'\0'):
        if record:
            meta, name = record.split(b'\t', 1)
            mode, oid, stage = meta.decode().split()
            if stage != '0':
                raise ValueError('Unresolved index conflicts')
            entries.append((name.decode('utf-8'), mode, oid))
    if staged:
        changed = set(git(repo, 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').decode().split('\0'))
        entries = [e for e in entries if e[0] in changed]
    blobs = objects(repo, [e[2] for e in entries])
    failures = [{'path': e[0], 'issues': problems} for e, blob in zip(entries, blobs)
                if (problems := file_problems(e[0], blob, e[1]))]
    if failures:
        raise ValueError(json.dumps(failures, ensure_ascii=False))
    return {'status': 'PASS', 'scope': 'staged changes' if staged else 'all indexed files', 'files': len(entries),
            'bytes': sum(map(len, blobs)), 'credential_check': 'limited pattern scan, not a comprehensive secret audit'}

def verify_import(repo, working_tree=False):
    index = json.loads((repo / 'manifests/experiments.json').read_text(encoding='utf-8'))
    binding = index['source_import']
    commit = binding['commit']
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Source import must name a complete immutable commit')
    manifest_path = relative_path(binding['manifest'])
    raw_manifest = (repo / manifest_path).read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != binding['manifest_sha256']:
        raise ValueError('Imported manifest SHA mismatch')
    rows = json.loads(raw_manifest)['files']
    names = [str(relative_path(r['path'])) for r in rows]
    if len(set(names)) != len(names) or len(rows) != binding['files']:
        raise ValueError('Duplicate paths or wrong source count')
    blobs = objects(repo, [commit + ':' + p for p in names])
    for row, blob in zip(rows, blobs):
        if len(blob) != row['bytes'] or hashlib.sha256(blob).hexdigest() != row['sha256']:
            raise ValueError('Imported Git source differs: ' + row['path'])
        if working_tree:
            p = repo / row['path']
            if p.is_symlink() or p.read_bytes() != blob:
                raise ValueError('Working source differs from original import: ' + row['path'])
    return {'status': 'PASS', 'files': len(rows), 'commit': commit, 'working_tree_compared': working_tree,
            'scope': 'retrospective source identity; no scientific validation or launch'}

def freeze(repo, experiment, config, output):
    if not re.fullmatch('[a-z0-9][a-z0-9-]{2,100}', experiment):
        raise ValueError('Use a lowercase experiment ID with hyphens')
    if git(repo, 'status', '--porcelain'):
        raise ValueError('Commit or isolate pending changes/untracked files before freezing')
    rel = relative_path(config)
    tracked = set(git(repo, 'ls-files', '-z').decode().split('\0'))
    if str(rel) not in tracked:
        raise ValueError('Configuration must be tracked in this commit')
    if (repo / rel).is_symlink():
        raise ValueError('Configuration cannot be a symlink')
    out = Path(output).resolve()
    if out.is_relative_to(repo.resolve()):
        raise ValueError('Write the run receipt outside the source checkout')
    check(repo)
    commit = git(repo, 'rev-parse', 'HEAD').decode().strip()
    records = []
    for record in git(repo, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
        if record:
            meta, name = record.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            if kind != 'blob':
                raise ValueError('External Git objects require a separate pinned dependency manifest')
            records.append((name.decode(), mode, oid))
    blobs = objects(repo, [r[2] for r in records])
    files = [{'path': r[0], 'mode': r[1], 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
             for r, b in zip(records, blobs)]
    result = {'schema_version': 1, 'experiment_id': experiment, 'commit': commit,
              'branch': git(repo, 'branch', '--show-current').decode().strip(), 'config': str(rel),
              'config_sha256': next(r['sha256'] for r in files if r['path'] == str(rel)),
              'source_files': files, 'execution_authorized': False,
              'scope': 'code identity only; existing resource, data, environment and run acceptance remain required'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False); stream.write('\n')
    return {'status': 'PASS', 'receipt': str(out), 'commit': commit, 'files': len(files)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status')
    c = commands.add_parser('check'); c.add_argument('--staged', action='store_true')
    v = commands.add_parser('verify-import'); v.add_argument('--working-tree', action='store_true')
    f = commands.add_parser('freeze')
    for field in ('experiment', 'config', 'output'):
        f.add_argument('--' + field, required=True)
    args = parser.parse_args(); repo = args.repo.resolve()
    try:
        if args.command == 'status':
            result = {'branch': git(repo, 'branch', '--show-current').decode().strip(),
                      'commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
                      'changes': git(repo, 'status', '--porcelain').decode().splitlines()}
        elif args.command == 'check': result = check(repo, args.staged)
        elif args.command == 'verify-import': result = verify_import(repo, args.working_tree)
        else: result = freeze(repo, args.experiment, args.config, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print('FAIL: ' + str(error), file=sys.stderr); return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
