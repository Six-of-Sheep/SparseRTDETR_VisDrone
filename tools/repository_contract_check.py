"""Standard-library-only checks for the initial P3 repository contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_UPSTREAM_REPOSITORY = "https://github.com/lyuwenyu/RT-DETR.git"
EXPECTED_UPSTREAM_BRANCH = "main"
EXPECTED_UPSTREAM_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"
EXPECTED_UPSTREAM_COMMIT_TIME = "2026-06-15T13:50:44+09:00"
EXPECTED_UPSTREAM_ROOT_TREE = "b6de37e186373fc59b91d23c846bb0eda35b6986"
EXPECTED_UPSTREAM_SUBTREE = "96a3b3e7e015d5e548e2917df2fec9641375e96e"
EXPECTED_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
VENDOR_PREFIX = "vendor/rtdetrv2_pytorch/"
VENDOR_ADDITIONAL_FILES = {
    "vendor/rtdetrv2_pytorch/LICENSE",
    "vendor/rtdetrv2_pytorch/UPSTREAM.md",
}


ALLOWED_FILES = {
    "README.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "src/sparse_rtdetr/__init__.py",
    "src/sparse_rtdetr/data_protocol/__init__.py",
    "src/sparse_rtdetr/data_protocol/categories.py",
    "src/sparse_rtdetr/data_protocol/cli.py",
    "src/sparse_rtdetr/data_protocol/converter.py",
    "src/sparse_rtdetr/data_protocol/evaluation.py",
    "src/sparse_rtdetr/data_protocol/lineage.py",
    "src/sparse_rtdetr/data_protocol/parser.py",
    "src/sparse_rtdetr/data_protocol/process_launcher.py",
    "src/sparse_rtdetr/data_protocol/protocol.py",
    "src/sparse_rtdetr/data_protocol/schema.py",
    "src/sparse_rtdetr/data_protocol/split.py",
    "configs/README.md",
    "configs/visdrone_protocol_v1.json",
    "configs/visdrone_protocol_v2.json",
    "tests/test_repository_contract.py",
    "tests/test_environment_contract.py",
    "tools/repository_contract_check.py",
    "scripts/README.md",
    "docs/contracts/P3_RESEARCH_CONTRACT.md",
    "docs/contracts/DATA_PROTOCOL.md",
    "docs/contracts/EVALUATION_CONTRACT.md",
    "docs/contracts/ARTIFACT_POLICY.md",
    "docs/contracts/ENVIRONMENT_POLICY.md",
    "docs/contracts/AUTODL_MIGRATION.md",
    "docs/contracts/VISDRONE_PROTOCOL_V1.md",
    "docs/contracts/VISDRONE_PROTOCOL_V2.md",
    "docs/contracts/VISDRONE_CONVERTER_V1.md",
    "docs/contracts/TEST_ACCESS_INCIDENT.md",
    "docs/upstream/RTDETRV2_SELECTION.md",
    "docs/legacy_p2/P2_FINAL_CLOSURE.md",
    "manifests/p2_legacy_manifest.json",
    "manifests/legacy_source_identity.json",
    "manifests/rtdetrv2_upstream.json",
    "environment/README.md",
    "environment/CPU_CONTRACT.md",
    "environment/REBUILD_R2.md",
    "environment/conda-linux-64.explicit.txt",
    "environment/conda-packages.json",
    "environment/pip-constraints.txt",
    "environment/pip-packages.json",
    "environment/manifest.json",
    "tests/test_visdrone_protocol.py",
    "tests/test_visdrone_converter.py",
    "tests/test_process_launcher.py",
    "tests/test_rtdetr_baseline_adapter.py",
    "tests/test_rtdetr_baseline_smoke.py",
    *VENDOR_ADDITIONAL_FILES,
}

BASELINE_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json",
    "docs/contracts/RTDETR_BASELINE_ADAPTER_V1.md",
    "src/sparse_rtdetr/baseline/__init__.py",
    "src/sparse_rtdetr/baseline/artifacts.py",
    "src/sparse_rtdetr/baseline/categories.py",
    "src/sparse_rtdetr/baseline/config.py",
    "src/sparse_rtdetr/baseline/contract.py",
    "src/sparse_rtdetr/baseline/dataset.py",
    "src/sparse_rtdetr/baseline/postprocessor.py",
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_evidence.py",
    "src/sparse_rtdetr/baseline/smoke_launcher.py",
    "src/sparse_rtdetr/baseline/smoke_outer_launcher.py",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v3.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json",
    "docs/contracts/RTDETR_BASELINE_SMOKE_V1.md",
    "docs/contracts/RTDETR_BASELINE_SMOKE_OUTER_LAUNCH_V2.md",
    "tests/test_rtdetr_baseline_smoke_outer_launcher.py",
}

BASELINE_MODEL_IMPORT_FILES = {
    "src/sparse_rtdetr/baseline/categories.py",
    "src/sparse_rtdetr/baseline/postprocessor.py",
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_evidence.py",
}

LEGACY_FILES = {
    "docs/legacy_p2/P2_FINAL_CLOSURE.md",
    "manifests/p2_legacy_manifest.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink()) and ".git" not in p.relative_to(root).parts
    }


def _git_tracked_files(root: Path) -> set[str]:
    """Read the index without treating ignored tracked files as runtime data."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {item for item in result.stdout.decode("utf-8").split("\0") if item}


def _gitignore_patterns(root: Path) -> list[tuple[bool, str]]:
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    patterns: list[tuple[bool, str]] = []
    for raw in lines:
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        negated = value.startswith("!")
        if negated:
            value = value[1:]
        if value.endswith("/"):
            value = value[:-1]
        if value:
            patterns.append((negated, value.lstrip("/")))
    return patterns


def _gitignored(relative: str, patterns: list[tuple[bool, str]]) -> bool:
    """Cover the repository's Gitignore grammar without walking runtime data."""

    from fnmatch import fnmatchcase

    parts = relative.split("/")
    ignored = False
    for negated, pattern in patterns:
        if "/" in pattern:
            matched = fnmatchcase(relative, pattern) or any(
                fnmatchcase("/".join(parts[index:]), pattern)
                for index in range(len(parts))
            )
        else:
            matched = any(fnmatchcase(part, pattern) for part in parts)
        if matched:
            ignored = not negated
    return ignored


def _file_policy(root: Path) -> tuple[set[str], list[str]]:
    """Separate ignored runtime files from source while still auditing them."""

    failures: list[str] = []
    all_files = _relative_files(root)
    tracked_files = _git_tracked_files(root)
    ignore_patterns = _gitignore_patterns(root)
    files: set[str] = set()
    for relative in all_files:
        path = root / relative
        ignored = _gitignored(relative, ignore_patterns)
        if path.is_symlink():
            failures.append(f"symlink: {relative}")
        if ignored:
            if relative in tracked_files:
                if relative == "artifacts" or relative.startswith("artifacts/"):
                    failures.append(f"tracked runtime artifact: {relative}")
                else:
                    files.add(relative)
            continue
        files.add(relative)
    artifacts_root = root / "artifacts"
    if artifacts_root.exists() and (artifacts_root.is_symlink() or not artifacts_root.is_dir()):
        failures.append("artifacts must be a real directory")
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts or path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _gitignored(relative, ignore_patterns):
            continue
        if path.stat().st_size == 0:
            failures.append(f"empty file: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"large file: {relative}")
    return files, failures


def _vendor_source_role(relative: str) -> str:
    if relative in VENDOR_ADDITIONAL_FILES:
        return "p3_additional"
    inner = relative[len(VENDOR_PREFIX):]
    if inner.endswith((".yml", ".yaml")):
        return "config"
    if inner == "Dockerfile" or inner.endswith(("requirements.txt", "docker-compose.yml")):
        return "environment"
    if inner.startswith("references/deploy/"):
        return "deployment"
    if inner.startswith("tools/"):
        return "tool"
    if inner.startswith("src/data/"):
        return "data_adapter"
    if inner.startswith(("src/nn/", "src/zoo/")):
        return "model"
    if inner.startswith("src/optim/"):
        return "optimization"
    if inner.startswith("src/solver/"):
        return "training"
    if inner.startswith(("src/core/", "src/misc/")):
        return "runtime"
    return "package_or_documentation"


def _source_policy_failures(root: Path, files: set[str]) -> list[str]:
    """Reject machine-specific paths and model imports outside explicit files."""

    failures: list[str] = []
    media_path = "/" + "media/"
    home_path = "/" + "home/"
    mount_path = "/" + "mnt/"
    file_scheme = "file" + "://"
    forbidden_path = re.compile("(?:" + re.escape(media_path) + "|" + re.escape(home_path) + "|" + re.escape(mount_path) + "|" + re.escape(file_scheme) + r"|https?://[^\s]+@)")
    forbidden_import = re.compile(r"(?m)^\s*(?:from|import)\s+(?:torch|ultralytics|rtdetr)\b")
    for relative in sorted(files - LEGACY_FILES):
        if relative.startswith(VENDOR_PREFIX):
            continue
        path = root / relative
        if path.suffix not in {".py", ".md", ".toml", ".yml", ".json", ".gitattributes", ""}:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden_path.search(text):
            failures.append(f"machine-specific path: {relative}")
        if path.suffix == ".py" and forbidden_import.search(text) and relative not in BASELINE_MODEL_IMPORT_FILES:
            failures.append(f"model import: {relative}")
    return failures


def _vendor_inventory(root: Path) -> list[dict[str, object]]:
    vendor_root = root / "vendor" / "rtdetrv2_pytorch"
    inventory = []
    for path in sorted(vendor_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        inventory.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "executable": bool(path.stat().st_mode & 0o111),
            "source_role": _vendor_source_role(relative),
        })
    return inventory


def _canonical_inventory_sha256(inventory: list[dict[str, object]]) -> str:
    payload = json.dumps(
        inventory,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_vendor(root: Path, failures: list[str]) -> None:
    vendor_root = root / "vendor" / "rtdetrv2_pytorch"
    manifest_path = root / "manifests" / "rtdetrv2_upstream.json"
    if not vendor_root.is_dir():
        failures.append("missing vendor/rtdetrv2_pytorch directory")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"upstream manifest parse failure: {type(exc).__name__}")
        return

    expected_fields = {
        "schema_version",
        "upstream_repository",
        "upstream_branch",
        "upstream_commit",
        "upstream_commit_time",
        "upstream_root_tree",
        "upstream_subtree",
        "license_name",
        "license_sha256",
        "implementation",
        "vendor_relative_path",
        "file_count",
        "total_size_bytes",
        "inventory_algorithm",
        "canonical_inventory_sha256",
        "files",
    }
    if manifest.get("schema_version") != 1:
        failures.append("upstream manifest schema mismatch")
    if manifest.get("upstream_repository") != EXPECTED_UPSTREAM_REPOSITORY:
        failures.append("upstream repository mismatch")
    if manifest.get("upstream_branch") != EXPECTED_UPSTREAM_BRANCH:
        failures.append("upstream branch mismatch")
    if manifest.get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
        failures.append("upstream commit mismatch")
    if manifest.get("upstream_commit_time") != EXPECTED_UPSTREAM_COMMIT_TIME:
        failures.append("upstream commit time mismatch")
    if manifest.get("upstream_root_tree") != EXPECTED_UPSTREAM_ROOT_TREE:
        failures.append("upstream root tree mismatch")
    if manifest.get("upstream_subtree") != EXPECTED_UPSTREAM_SUBTREE:
        failures.append("upstream subtree mismatch")
    if manifest.get("license_name") != "Apache-2.0":
        failures.append("upstream license name mismatch")
    if manifest.get("license_sha256") != EXPECTED_LICENSE_SHA256:
        failures.append("upstream license sha mismatch")
    if manifest.get("implementation") != "rtdetrv2_pytorch":
        failures.append("upstream implementation mismatch")
    if manifest.get("vendor_relative_path") != "vendor/rtdetrv2_pytorch":
        failures.append("vendor path mismatch")
    if manifest.get("inventory_algorithm") != "sha256(canonical_json(files))":
        failures.append("inventory algorithm mismatch")
    if set(manifest) != expected_fields:
        failures.append("upstream manifest field set mismatch")

    actual = _vendor_inventory(root)
    declared = manifest.get("files")
    if not isinstance(declared, list) or declared != sorted(declared, key=lambda item: item.get("relative_path", "")):
        failures.append("upstream manifest files are not sorted")
    elif declared != actual:
        failures.append("vendor inventory mismatch")
    if manifest.get("file_count") != len(actual):
        failures.append("vendor file count mismatch")
    if manifest.get("total_size_bytes") != sum(item["size_bytes"] for item in actual):
        failures.append("vendor total size mismatch")
    if manifest.get("canonical_inventory_sha256") != _canonical_inventory_sha256(actual):
        failures.append("canonical vendor inventory sha mismatch")

    license_path = vendor_root / "LICENSE"
    if not license_path.is_file() or _sha256(license_path) != EXPECTED_LICENSE_SHA256:
        failures.append("vendor LICENSE mismatch")
    upstream_path = vendor_root / "UPSTREAM.md"
    try:
        upstream_text = upstream_path.read_text(encoding="utf-8")
        for needle in (
            "project: RT-DETR",
            "implementation: RT-DETRv2 PyTorch",
            f"upstream_commit: {EXPECTED_UPSTREAM_COMMIT}",
            "license: Apache-2.0",
            "source_modified: false",
            "weights_included: false",
            "datasets_included: false",
            "upstream_git_history_included: false",
        ):
            if needle not in upstream_text:
                failures.append(f"missing vendor identity text: {needle}")
    except OSError:
        failures.append("missing vendor UPSTREAM.md")

    forbidden_suffixes = {".zip", ".tar", ".gz", ".tgz", ".pth", ".pt", ".ckpt"}
    for path in vendor_root.rglob("*"):
        if path.is_symlink():
            failures.append(f"vendor symlink: {path.relative_to(root)}")
        elif path.is_file():
            if path.stat().st_size > 10 * 1024 * 1024:
                failures.append(f"vendor large file: {path.relative_to(root)}")
            if path.suffix.lower() in forbidden_suffixes:
                failures.append(f"vendor archive/weight: {path.relative_to(root)}")
            try:
                with path.open("rb") as handle:
                    is_lfs_pointer = handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
                if is_lfs_pointer:
                    failures.append(f"vendor LFS pointer: {path.relative_to(root)}")
            except OSError:
                failures.append(f"vendor unreadable file: {path.relative_to(root)}")


def check_repository(root: Path) -> bool:
    failures: list[str] = []
    files, file_policy_failures = _file_policy(root)
    failures.extend(file_policy_failures)

    vendor_files = {relative for relative in files if relative.startswith(VENDOR_PREFIX)}
    allowed_files = ALLOWED_FILES | vendor_files
    allowed_files |= BASELINE_FILES
    if files != allowed_files:
        failures.append(f"file set mismatch: extra={sorted(files - allowed_files)} missing={sorted(ALLOWED_FILES - files)}")

    _check_vendor(root, failures)

    protocol_path = root / "configs/visdrone_protocol_v1.json"
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("schema_version") != 1:
            failures.append("VisDrone protocol schema mismatch")
        if protocol.get("protocol_id") != "P3-VISDRONE-DATA-PROTOCOL-V1":
            failures.append("VisDrone protocol ID mismatch")
        if protocol.get("test_access_allowed") is not False:
            failures.append("VisDrone test access is not disabled")
        if protocol.get("real_conversion_outputs_generated") is not False:
            failures.append("real conversion output marker is not false")
        if protocol.get("production_split_manifest_generated") is not False:
            failures.append("production split marker is not false")
        split = protocol.get("split", {})
        if split.get("seed") != 20260808 or split.get("salt") != "P3-confirmatory-v1":
            failures.append("VisDrone split seed/salt mismatch")
        if split.get("test") != "disabled":
            failures.append("VisDrone test split is not disabled")
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        failures.append(f"VisDrone protocol parse failure: {type(exc).__name__}")

    identity_path = root / "manifests/legacy_source_identity.json"
    manifest_path = root / "manifests/p2_legacy_manifest.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("copied_byte_exact") is not True:
            failures.append("legacy copy is not marked byte exact")
        if identity.get("legacy_runs_copied") is not False:
            failures.append("legacy runs copied flag is not false")
        if identity.get("legacy_checkpoints_copied") is not False:
            failures.append("legacy checkpoints copied flag is not false")
        if identity.get("legacy_environment_reused") is not False:
            failures.append("legacy environment reused flag is not false")
        for item in identity.get("files", []):
            path = root / item["new_relative_path"]
            if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
                failures.append(f"legacy identity mismatch: {path}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        failures.append(f"identity parse failure: {type(exc).__name__}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("legacy_git_parent_head") != "a5508d63b503eacb98d5f9b812ffed04f97747f7":
            failures.append("legacy manifest parent commit mismatch")
    except (OSError, json.JSONDecodeError):
        failures.append("legacy manifest parse failure")

    failures.extend(_source_policy_failures(root, files))

    required_text = {
        "README.md": ["P2 YOLO", "RT-DETRv2", "test split", "vendor"],
        "docs/contracts/DATA_PROTOCOL.md": ["TEST_ACCESS_ALLOWED=false", "post-hoc"],
        "docs/contracts/EVALUATION_CONTRACT.md": ["MODEL_SELECTION_READY=false", "ACCURACY_ACCEPTANCE_READY=false"],
        "docs/contracts/ENVIRONMENT_POLICY.md": ["rtdetrv2_pytorch", "P3_ENVIRONMENT_READY=false"],
        "docs/upstream/RTDETRV2_SELECTION.md": ["Apache-2.0", "300 Queries", "1024"],
    }
    for relative, needles in required_text.items():
        text = (root / relative).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"missing contract text {needle!r}: {relative}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return False
    print("repository_contract_check: PASS")
    return True


def main() -> int:
    return 0 if check_repository(Path(__file__).resolve().parents[1]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
