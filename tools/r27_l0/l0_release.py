"""Freeze the committed HEAD of the L0 work tree into <round>/<name>/source with MANIFEST.json (file SHA256 per path).

usage: l0_release.py <repo> <round> <name> [--dev-allow-dirty]
Never launches an experiment.  Refuses an existing release directory.  The export is always `git archive HEAD`
(uncommitted or untracked files are never part of a release).  A work tree with uncommitted/untracked paths is
refused, except with --dev-allow-dirty for a dev round (round directory name starting with "dev-round"); the
MANIFEST then records dirty=true with the porcelain lines, and prepare refuses such a release for the formal round.
"""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def git(repo, *args, text=True):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=text, check=True).stdout


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    allow_dirty = "--dev-allow-dirty" in argv
    argv = [a for a in argv if a != "--dev-allow-dirty"]
    if len(argv) != 3:
        raise SystemExit(__doc__)
    repo, round_, name = argv
    out = Path(round_) / name
    if out.exists():
        raise SystemExit(f"release exists: {out}")
    dirty = [line for line in git(repo, "status", "--porcelain").splitlines() if line.strip()]
    if dirty:
        if not allow_dirty:
            raise SystemExit(f"work tree has uncommitted/untracked paths; release refused: {dirty[:8]}")
        if not Path(round_).resolve().name.startswith("dev-round"):
            raise SystemExit("--dev-allow-dirty is only for a dev round (directory name dev-round*)")
    full = git(repo, "rev-parse", "HEAD^{commit}").strip()
    branch = git(repo, "branch", "--show-current").strip()
    raw = git(repo, "archive", "--format=tar", full, text=False)
    src = out / "source"
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(src)
    files = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(src))] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"version": "R27-002-L0-release-v1", "commit": full, "branch": branch, "dirty": bool(dirty),
                "dirty_paths": dirty, "files": files}
    with (out / "MANIFEST.json").open("x") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(json.dumps({"release": str(out), "commit": full, "files": len(files), "dirty": bool(dirty),
                      "manifest_sha256": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
