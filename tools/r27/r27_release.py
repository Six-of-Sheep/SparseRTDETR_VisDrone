"""Freeze a committed R27 source tree into <round>/<name>/source with MANIFEST.json (file SHA256 per path).

usage: r27_release.py <repo> <commit> <round> <name>
Never launches an experiment.  Refuses an existing release directory or a dirty commit reference.
"""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    repo, commit, round_, name = sys.argv[1:5]
    out = Path(round_) / name
    if out.exists():
        raise SystemExit(f"release exists: {out}")
    full = subprocess.run(["git", "-C", repo, "rev-parse", f"{commit}^{{commit}}"], capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "-C", repo, "branch", "--show-current"], capture_output=True, text=True, check=True).stdout.strip()
    raw = subprocess.run(["git", "-C", repo, "archive", "--format=tar", full], capture_output=True, check=True).stdout
    src = out / "source"
    src.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(src)
    files = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(src))] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"version": "R27-001-release-v1", "commit": full, "branch": branch, "files": files}
    with (out / "MANIFEST.json").open("x") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(json.dumps({"release": str(out), "commit": full, "files": len(files),
                      "manifest_sha256": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
