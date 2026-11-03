"""Build the r24 execution release: frozen release-008 source, the r24 roi-ld overlay and patches.

usage: build_release_r24.py <release008_dir> <worktree_root> <out_dir> <plan_rev2_path>

The base source is copied and re-verified file by file against the release-008 manifest, then the
r24 round is applied on top of it in two auditable layers:

* anchored patches to the three frozen protocol files (criterion, checkpoint, slice_finetune) via
  ``roi_ld_patch_r24`` - every anchor must occur exactly once, and each file's before/after digest
  is recorded;
* whole-file overlays for the round's own engine modules (slice_ft_job, slice_ft_execution,
  slice_ft_progress), the three roi modules, the round tools and the amended science plan.

Nothing of the retired r19 item is removed: its modules stay present and inert, so the frozen
files keep their base bytes everywhere the round does not change them.  The emitted manifest
follows the release-008 schema so the existing freeze/bind readers keep working.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_ld_patch_r24  # noqa: E402

EXPERIMENT_ID = "r24-roi-ld-20261103-001"
BASE_MANIFEST_NAME = "execution-source-008.json"
OUT_MANIFEST_NAME = "execution-source-r24-001.json"
PROTOCOL_REL = "src/sparse_rtdetr/protocol_r18"

# Whole-file overlays: the round's engine modules and the three roi modules land in protocol_r18.
ENGINE_OVERLAY = {
    "engine_r24/slice_ft_job.py": PROTOCOL_REL + "/slice_ft_job.py",
    "engine_r24/slice_ft_execution.py": PROTOCOL_REL + "/slice_ft_execution.py",
    "engine_r24/slice_ft_progress.py": PROTOCOL_REL + "/slice_ft_progress.py",
    "roi_ld.py": PROTOCOL_REL + "/roi_ld.py",
    "roi_ld_arm.py": PROTOCOL_REL + "/roi_ld_arm.py",
    "roi_ld_data.py": PROTOCOL_REL + "/roi_ld_data.py",
}
# The round tools ship under tools/roi_ld so the release carries its own patch/check/test kit.
TOOL_OVERLAY = ("__init__.py", "roi_ld.py", "roi_ld_arm.py", "roi_ld_data.py", "roi_ld_patch.py",
                "roi_ld_patch_r24.py", "roi_ld_check.py", "test_roi_ld.py", "test_roi_ld_engine.py",
                "build_release_r24.py", "prepare_contracts_r24.py", "amend_plan_r24.py", "run_roi_ld.py")
PLAN_REL = "configs/research/r24_roi_ld_v1.rev2.json"


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

    # Layer 1: anchored patches to the frozen protocol files.
    patches = roi_ld_patch_r24.apply(out_src)

    # Layer 2: whole-file overlays.
    worktree_tools = worktree / "tools/roi_ld"
    overlay = []
    for source_name, relative in sorted(ENGINE_OVERLAY.items()):
        source = worktree_tools / source_name
        if not source.is_file():
            raise SystemExit("missing r24 overlay %s" % source_name)
        target = out_src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        overlay.append({"path": relative, "sha256": sha(target), "bytes": target.stat().st_size,
                        "from": "tools/roi_ld/" + source_name})
    tools_target = out_src / "tools/roi_ld"
    tools_target.mkdir(parents=True, exist_ok=True)
    for name in TOOL_OVERLAY:
        source = worktree_tools / name
        if not source.is_file():
            raise SystemExit("missing r24 tool %s" % name)
        shutil.copy2(source, tools_target / name)
        overlay.append({"path": "tools/roi_ld/%s" % name, "sha256": sha(tools_target / name),
                        "bytes": (tools_target / name).stat().st_size, "from": "tools/roi_ld/" + name})
    (tools_target / "__pycache__").is_dir() and shutil.rmtree(tools_target / "__pycache__")
    plan_target = out_src / PLAN_REL
    plan_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, plan_target)
    overlay.append({"path": PLAN_REL, "sha256": sha(plan_target), "bytes": plan_target.stat().st_size,
                    "from": str(plan_path)})

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
        "derived_head": commit.stdout.strip(),
        "branch": branch.stdout.strip(),
        "base_manifest": {"path": str(base_manifest_path), "sha256": sha(base_manifest_path)},
        "base_note": ("base source is the frozen r18-tiny-positives execution-release-008 source, "
                      "verified file by file against its own manifest before the copy"),
        "overlay": overlay,
        "patches": patches,
        "plan": {"path": str(plan_target), "sha256": sha(plan_target), "bytes": plan_target.stat().st_size},
        "source_files": entries,
        "note": ("r24 installs the band ROI-resolution box distillation term into the frozen chain: "
                 "anchored patches to criterion/checkpoint/slice_finetune plus whole-file overlays for "
                 "the round engine modules, the three roi modules, the round tools and the amended "
                 "science plan.  The retired r19 tiny-positives item keeps its modules and is installed "
                 "by no arm.  No scientific constant, threshold, schedule or evaluation budget changes."),
    }
    manifest_path = out_dir / OUT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    build = {"status": "BUILT", "release": str(out_dir), "files": len(entries),
             "base_verified_files": verified, "overlay_files": len(overlay),
             "patched_files": len(patches), "base_manifest_sha256": sha(base_manifest_path),
             "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
             "plan_sha256": sha(plan_target), "plan": str(plan_target),
             "derived_head": commit.stdout.strip(), "commit": commit.stdout.strip(),
             "branch": branch.stdout.strip(), "patches": patches}
    (out_dir / "BUILD.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in build.items() if k != "patches"}, indent=1))


if __name__ == "__main__":
    main()
