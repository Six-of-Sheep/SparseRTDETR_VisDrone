"""Build the r20 execution release: frozen release-008 source plus the r20 zoom-refine tools.

usage: build_release_r20.py <release008_dir> <worktree_root> <out_dir> <plan_json>

The base source is copied file by file and verified against the release-008 manifest before
use; only the four r20 tool files are added under tools/zoom_refine/.  The emitted manifest
follows the release-008 schema so the existing freeze/bind readers keep working.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

R20_TOOLS = ("zoom_refine.py", "test_zoom_refine.py", "check_substitution.py", "run_zoom_refine.py")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    base_dir, worktree, out_dir, plan_path = (Path(a) for a in sys.argv[1:5])
    base_manifest_path = base_dir / "execution-source-008.json"
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
            raise SystemExit(f"base source file differs: {row['path']}")
        verified += 1

    overlay = []
    target = out_src / "tools/zoom_refine"
    target.mkdir(parents=True, exist_ok=True)
    for name in R20_TOOLS:
        source = worktree / "tools/zoom_refine" / name
        if not source.is_file():
            raise SystemExit(f"missing r20 tool {name}")
        shutil.copy2(source, target / name)
        overlay.append({"path": f"tools/zoom_refine/{name}", "sha256": sha(target / name),
                        "bytes": (target / name).stat().st_size})
    (target / "__init__.py").write_text("")
    overlay.append({"path": "tools/zoom_refine/__init__.py", "sha256": sha(target / "__init__.py"),
                    "bytes": (target / "__init__.py").stat().st_size})

    files = sorted(p for p in out_src.rglob("*") if p.is_file())
    entries = [{"path": str(p.relative_to(out_src)), "mode": "100644" if p.stat().st_mode & 0o200 else "100444",
                "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    commit = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True, text=True)
    branch = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": "r20-zoom-refine-20261102-001",
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip(),
        "base_manifest": {"path": str(base_manifest_path), "sha256": sha(base_manifest_path)},
        "base_note": "base source is the frozen r18-tiny-positives execution-release-008 source, verified file by file against its own manifest before the copy",
        "overlay": overlay,
        "patches": {},
        "plan": {"path": str(plan_path), "sha256": sha(plan_path), "bytes": plan_path.stat().st_size},
        "source_files": entries,
        "note": "r20 adds inference-only tools; no training, matcher, criterion or evaluation source is modified",
    }
    manifest_path = out_dir / "execution-source-r20-001.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    build = {"status": "BUILT", "release": str(out_dir), "files": len(entries),
             "base_verified_files": verified, "overlay_files": len(overlay), "patched_files": 0,
             "base_manifest_sha256": sha(base_manifest_path), "manifest": str(manifest_path),
             "manifest_sha256": sha(manifest_path), "plan_sha256": sha(plan_path),
             "commit": commit.stdout.strip()}
    (out_dir / "BUILD.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(build, indent=1))


if __name__ == "__main__":
    main()
