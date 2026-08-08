"""Standard-library-only checks for the initial P3 repository contract."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ALLOWED_FILES = {
    "README.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "src/sparse_rtdetr/__init__.py",
    "configs/README.md",
    "tests/test_repository_contract.py",
    "tools/repository_contract_check.py",
    "scripts/README.md",
    "docs/contracts/P3_RESEARCH_CONTRACT.md",
    "docs/contracts/DATA_PROTOCOL.md",
    "docs/contracts/EVALUATION_CONTRACT.md",
    "docs/contracts/ARTIFACT_POLICY.md",
    "docs/contracts/ENVIRONMENT_POLICY.md",
    "docs/contracts/AUTODL_MIGRATION.md",
    "docs/legacy_p2/P2_FINAL_CLOSURE.md",
    "manifests/p2_legacy_manifest.json",
    "manifests/legacy_source_identity.json",
    "environment/README.md",
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
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def check_repository(root: Path) -> bool:
    failures: list[str] = []
    files = _relative_files(root)
    if files != ALLOWED_FILES:
        failures.append(f"file set mismatch: extra={sorted(files - ALLOWED_FILES)} missing={sorted(ALLOWED_FILES - files)}")

    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            failures.append(f"symlink: {path.relative_to(root)}")
        elif path.is_file():
            if path.stat().st_size == 0:
                failures.append(f"empty file: {path.relative_to(root)}")
            if path.stat().st_size >= 1024 * 1024:
                failures.append(f"large file: {path.relative_to(root)}")

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

    media_path = "/" + "media/"
    home_path = "/" + "home/"
    mount_path = "/" + "mnt/"
    file_scheme = "file" + "://"
    forbidden_path = re.compile("(?:" + re.escape(media_path) + "|" + re.escape(home_path) + "|" + re.escape(mount_path) + "|" + re.escape(file_scheme) + r"|https?://[^\\s]+@)")
    forbidden_import = re.compile(r"(?m)^\\s*(?:from|import)\\s+(?:torch|ultralytics|rtdetr)\\b")
    for relative in sorted(files - LEGACY_FILES):
        path = root / relative
        if path.suffix not in {".py", ".md", ".toml", ".yml", ".json", ".gitattributes", ""}:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden_path.search(text):
            failures.append(f"machine-specific path: {relative}")
        if path.suffix == ".py" and forbidden_import.search(text):
            failures.append(f"model import: {relative}")

    required_text = {
        "README.md": ["P2 YOLO", "No RT-DETR", "test split"],
        "docs/contracts/DATA_PROTOCOL.md": ["TEST_ACCESS_ALLOWED=false", "post-hoc"],
        "docs/contracts/EVALUATION_CONTRACT.md": ["MODEL_SELECTION_READY=false", "ACCURACY_ACCEPTANCE_READY=false"],
        "docs/contracts/ENVIRONMENT_POLICY.md": ["P3_ENVIRONMENT_READY=false"],
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
