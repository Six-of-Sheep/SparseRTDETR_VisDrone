"""Build the r22 execution release: frozen release-008 source plus the r22 expand600 tools.

usage: build_release_r22.py <release008_dir> <worktree_root> <out_dir> <plan_json>

The base source is copied file by file and verified against the release-008 manifest before use;
only the four r22 tool files are added under tools/expand600/.  The emitted manifest follows the
release-008 schema so the existing freeze/bind readers keep working.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

R22_TOOLS = ("expand600.py", "test_expand600.py", "run_expand600.py")
EXPERIMENT_ID = "r22-expand600-20261102-001"
BASE_MANIFEST_NAME = "execution-source-008.json"
OUT_MANIFEST_NAME = "execution-source-r22-001.json"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    base_dir, worktree, out_dir, plan_path = (Path(a) for a in sys.argv[1:5])
    base_manifest_path = base_dir / BASE_MANIFEST_NAME
    base_manifest = json.loads(base_manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_src = out_dir / "source"
    if out_src.exists():
        raise SystemExit("release output already exists; refusing to overwrite")
    shutil.copytree(base_dir / "source", out_src)

    verified = 0
    for row in base_manifest["source_files"]:
        path = out_src / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise SystemExit("base source file differs: %s" % row["path"])
        verified += 1

    overlay = []
    target = out_src / "tools/expand600"
    target.mkdir(parents=True, exist_ok=True)
    for name in R22_TOOLS:
        source = worktree / "tools/expand600" / name
        if not source.is_file():
            raise SystemExit("missing r22 tool %s" % name)
        shutil.copy2(source, target / name)
        overlay.append({"path": "tools/expand600/%s" % name, "sha256": sha(target / name),
                        "bytes": (target / name).stat().st_size})
    (target / "__init__.py").write_text("")
    overlay.append({"path": "tools/expand600/__init__.py", "sha256": sha(target / "__init__.py"),
                    "bytes": (target / "__init__.py").stat().st_size})

    files = sorted(p for p in out_src.rglob("*") if p.is_file())
    entries = [{"path": str(p.relative_to(out_src)),
                "mode": "100644" if p.stat().st_mode & 0o200 else "100444",
                "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    commit = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    branch = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip(),
        "base_manifest": {"path": str(base_manifest_path), "sha256": sha(base_manifest_path)},
        "base_note": ("base source is the frozen r18-tiny-positives execution-release-008 source, "
                      "verified file by file against its own manifest before the copy"),
        "overlay": overlay,
        "patches": {},
        "plan": {"path": str(plan_path), "sha256": sha(plan_path), "bytes": plan_path.stat().st_size},
        "source_files": entries,
        "note": ("r22 adds inference-only tools under tools/expand600; no training, matcher, criterion, "
                 "evaluation or decoder source is modified"),
    }
    manifest_path = out_dir / OUT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    build = {"status": "BUILT", "release": str(out_dir), "files": len(entries),
             "base_verified_files": verified, "overlay_files": len(overlay), "patched_files": 0,
             "base_manifest_sha256": sha(base_manifest_path), "manifest": str(manifest_path),
             "manifest_sha256": sha(manifest_path), "plan_sha256": sha(plan_path),
             "commit": commit.stdout.strip(), "branch": branch.stdout.strip()}
    (out_dir / "BUILD.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(build, indent=1))


if __name__ == "__main__":
    main()
