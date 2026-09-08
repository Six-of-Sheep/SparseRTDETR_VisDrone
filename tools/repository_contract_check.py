"""Standard-library-only checks for the initial P3 repository contract."""

from __future__ import annotations

import hashlib
import ast
import json
import re
import stat
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
    "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
    "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_V1.md",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_RUNTIME_V1.md",
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
    "src/sparse_rtdetr/baseline/training_contract.py",
    "src/sparse_rtdetr/baseline/training_runtime.py",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v3.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v6.json",
    "configs/baseline/rtdetrv2_r18_visdrone_smoke_v7.json",
    "docs/contracts/RTDETR_BASELINE_SMOKE_V1.md",
    "docs/contracts/RTDETR_BASELINE_SMOKE_OUTER_LAUNCH_V2.md",
    "tests/test_rtdetr_baseline_smoke_outer_launcher.py",
    "tests/test_rtdetr_baseline_training_contract.py",
    "tests/test_rtdetr_baseline_training_runtime.py",
}

TRAINING_EVIDENCE_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_training_evidence_v1.json",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_EVIDENCE_V1.md",
    "src/sparse_rtdetr/baseline/training_evidence.py",
    "tests/test_rtdetr_baseline_training_evidence.py",
}
TRAINING_EVIDENCE_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_evidence_v1.json"
TRAINING_EVIDENCE_CONFIG_RAW_SIZE_BYTES = 7139
TRAINING_EVIDENCE_CONFIG_RAW_SHA256 = "4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e"
TRAINING_EVIDENCE_CONFIG_CANONICAL_SIZE_BYTES = 6069
TRAINING_EVIDENCE_CONFIG_CANONICAL_SHA256 = "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6"
TRAINING_EVIDENCE_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_evidence.py"
TRAINING_EVIDENCE_MODULE_SHA256 = "5efad5ea963f2b08e52cc7b7811ca64472dd36a5186cd5c476c7657adb5bc1b0"
TRAINING_EVIDENCE_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_evidence.py"
TRAINING_EVIDENCE_TEST_SHA256 = "84e3de1d80f636d9b02b75c7de17752e2972c5c25e69f95277157b0d33d176b7"
TRAINING_EVIDENCE_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_EVIDENCE_V1.md"
TRAINING_EVIDENCE_DOCUMENT_SHA256 = "8bd7ecabb2940901ca508092108959209c1b3a8c6305f96cc3d01da9b2eda931"
TRAINING_EVIDENCE_PUBLIC_APIS = {
    "load_training_evidence_contract",
    "validate_training_evidence_contract",
    "canonical_training_evidence_contract_bytes",
    "training_evidence_contract_binding",
    "TrainingEvidenceWriter",
    "write_atomic_checkpoint",
    "validate_training_evidence",
    "validate_training_checkpoint",
    "classify_training_evidence",
    "TrainingEvidenceError",
}
TRAINING_EVIDENCE_ALLOWED_IMPORTS = {
    "copy",
    "errno",
    "hashlib",
    "json",
    "math",
    "os",
    "stat",
    "pathlib",
    "typing",
    "sparse_rtdetr.baseline.training_contract",
    "sparse_rtdetr.baseline.training_runtime",
}
TRAINING_EVIDENCE_ROOT_RELATIVE_PATH = "artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1"

TRAINING_RUNTIME_PLAN_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json"
TRAINING_RUNTIME_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_runtime.py"
TRAINING_RUNTIME_PLAN_ID = "rtdetrv2_r18_visdrone_baseline_training_runtime_v1"
TRAINING_RUNTIME_PLAN_SCHEMA_VERSION = 1
TRAINING_RUNTIME_PLAN_RAW_SIZE_BYTES = 12957
TRAINING_RUNTIME_PLAN_RAW_SHA256 = "cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef"
TRAINING_RUNTIME_PUBLIC_APIS = {
    "load_training_runtime_plan",
    "validate_training_runtime_plan",
    "canonical_training_runtime_plan_bytes",
    "training_runtime_plan_binding",
}
TRAINING_RUNTIME_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "json",
    "math",
    "os",
    "typing",
    "sparse_rtdetr.baseline.training_contract",
}

PRIMARY_EVALUATOR_FILES = {
    "configs/baseline/visdrone_official_evaluator_v1.json",
    "manifests/visdrone_det_toolkit_005445.json",
    "src/sparse_rtdetr/baseline/primary_evaluator.py",
    "tests/test_rtdetr_baseline_primary_evaluator.py",
    "docs/contracts/RTDETR_BASELINE_PRIMARY_EVALUATOR_V1.md",
}

PREPARED_ADAPTER_FILES = {
    "src/sparse_rtdetr/baseline/training_adapter.py",
    "tests/test_rtdetr_baseline_training_adapter.py",
    "docs/contracts/RTDETR_BASELINE_PREPARED_TRAINER_ADAPTER_V1.md",
}
PREPARED_ADAPTER_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_adapter.py"
PREPARED_ADAPTER_MODULE_SHA256 = "3dca63c67fc2263b9f3f3006b05c8667e4457a5d1344db41f08e720e85016fd3"
PREPARED_ADAPTER_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_adapter.py"
PREPARED_ADAPTER_TEST_SHA256 = "6df4b0d1b57407b768472dfacb6b5ef946d8adfc319f65690b1cf0b919e9e988"
PREPARED_ADAPTER_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_PREPARED_TRAINER_ADAPTER_V1.md"
PREPARED_ADAPTER_DOCUMENT_SHA256 = "a5e9dafe6d4bdd608e4a83e82dd5d47a320e25c08a6e725d2f073ac3d9a01100"
PREPARED_ADAPTER_PUBLIC_NAMES = {
    "PreparedTrainerError",
    "prepare_training_adapter",
    "validate_prepared_training_adapter",
    "run_synthetic_prepared_batch",
}
PREPARED_ADAPTER_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "inspect",
    "json",
    "math",
    "os",
    "pathlib",
    "typing",
    "sparse_rtdetr.baseline.primary_evaluator",
    "sparse_rtdetr.baseline.training_contract",
    "sparse_rtdetr.baseline.training_evidence",
    "sparse_rtdetr.baseline.training_runtime",
}
PREPARED_ADAPTER_FORBIDDEN_IMPORT_ROOTS = {
    "asyncio",
    "cupy",
    "dask",
    "http",
    "multiprocessing",
    "numpy",
    "pandas",
    "requests",
    "socket",
    "subprocess",
    "threading",
    "torch",
    "urllib",
}
PREPARED_ADAPTER_FORBIDDEN_OPERATION_NAMES = {
    "connect",
    "cuda",
    "dataloader",
    "dataset",
    "device",
    "eval",
    "fork",
    "is_available",
    "nvidia",
    "open",
    "popen",
    "post",
    "process",
    "request",
    "run",
    "sleep",
    "spawn",
    "start",
    "system",
    "thread",
    "urlopen",
}

TRAINING_PROCESS_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_training_process_v1.json",
    "src/sparse_rtdetr/baseline/training_entry.py",
    "src/sparse_rtdetr/baseline/training_process_launcher.py",
    "tests/test_rtdetr_baseline_training_entry.py",
    "tests/test_rtdetr_baseline_training_process_launcher.py",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_PROCESS_LAUNCH_V1.md",
}
TRAINING_PROCESS_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_process_v1.json"
TRAINING_PROCESS_CONFIG_RAW_SIZE_BYTES = 10468
TRAINING_PROCESS_CONFIG_RAW_SHA256 = "a16ff1c31e0a935f0ab6ab9224b7c3f70c377006d00562dc5b2bc1b44f70ed89"
TRAINING_PROCESS_CONFIG_CANONICAL_SIZE_BYTES = 9330
TRAINING_PROCESS_CONFIG_CANONICAL_SHA256 = "8469bde5b8e87e47ef8df0b9674aa1e685bf993f90ef3bc99aa3354bbe95f19c"
TRAINING_ENTRY_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_entry.py"
TRAINING_ENTRY_SHA256 = "6990a1a0f2d47b034db5a0aa3928fdb1820819f8272c85f2b86c78832d464892"
TRAINING_ENTRY_PUBLIC_NAMES = {
    "TrainingEntryError",
    "build_synthetic_entry_descriptor",
    "validate_entry_descriptor",
    "run_synthetic_entry",
    "validate_entry_result",
}
TRAINING_ENTRY_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "json",
    "math",
    "os",
    "re",
    "stat",
    "sys",
    "pathlib",
    "typing",
    "sparse_rtdetr.baseline",
}
TRAINING_LAUNCHER_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_process_launcher.py"
TRAINING_LAUNCHER_SHA256 = "7543e20ea23e9e9b526d4ca89c6c3db6f4d77127a9ba48e48cc926edb5ea54de"
TRAINING_LAUNCHER_PUBLIC_NAMES = {
    "TrainingProcessLauncherError",
    "build_synthetic_launch_descriptor",
    "validate_launch_descriptor",
    "run_synthetic_launch",
    "validate_launch_result",
    "classify_launch",
}
TRAINING_LAUNCHER_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "inspect",
    "json",
    "math",
    "os",
    "stat",
    "pathlib",
    "typing",
    "sparse_rtdetr.baseline",
}
TRAINING_PROCESS_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_PROCESS_LAUNCH_V1.md"
TRAINING_PROCESS_DOCUMENT_SHA256 = "ff1d3c8b170b63e20fb62e0737a7167a87864a7b3b0e67fb49b874d3a5736523"
TRAINING_ENTRY_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_entry.py"
TRAINING_PROCESS_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_process_launcher.py"

T6A_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_training_launch_t6_v1.json",
    "src/sparse_rtdetr/baseline/training_t6_authorization.py",
    "tests/test_rtdetr_baseline_training_t6_authorization.py",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_LAUNCH_T6_V1.md",
}
T6A_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_launch_t6_v1.json"
T6A_CONFIG_RAW_SIZE_BYTES = 11730
T6A_CONFIG_RAW_SHA256 = "539abe556edd00a5bb0ecf0ed350195ee28e55a0af0b804404a0d92ea9830ff9"
T6A_CONFIG_CANONICAL_SIZE_BYTES = 11729
T6A_CONFIG_CANONICAL_SHA256 = "13a5923aa62f3048b2baefe32aaa163b9aa44d3b3e11d2a374ebceec5034ae94"
T6A_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_authorization.py"
T6A_MODULE_SHA256 = "7b8d0ce7cf1edfd545e35b9165c5eb0ba67d16eb7db5434c5c218dbe46dbb0b5"
T6A_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_t6_authorization.py"
T6A_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_LAUNCH_T6_V1.md"
T6A_PUBLIC_NAMES = {
    "TrainingLaunchContractError",
    "canonical_training_launch_contract_bytes",
    "load_training_launch_contract",
    "validate_training_launch_contract",
    "training_launch_contract_binding",
    "canonical_owner_authorization_bytes",
    "validate_owner_authorization",
    "owner_authorization_binding",
}
T6A_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "json",
    "math",
    "os",
    "re",
    "stat",
    "pathlib",
    "typing",
}

T6B_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_training_t6b_v1.json",
    "src/sparse_rtdetr/baseline/training_t6_engine.py",
    "src/sparse_rtdetr/baseline/training_t6_entry.py",
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py",
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py",
    "tests/test_rtdetr_baseline_training_t6b.py",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_T6B_PRODUCTION_BOUNDARY_V1.md",
}
T6B_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_t6b_v1.json"
T6B_CONFIG_RAW_SIZE_BYTES = 6925
T6B_CONFIG_RAW_SHA256 = "a152ccb7caefd42531ee2126ca194495e4af8d4a660d907c5a9acf796fefcc85"
T6B_CONFIG_CANONICAL_SIZE_BYTES = 5565
T6B_CONFIG_CANONICAL_SHA256 = "dfeecc7b002db9ce10b33afa9d166fab4c7f88a5f922b4862fa0d1d99ff755a1"
T6B_SOURCE_IDENTITIES = {
    "src/sparse_rtdetr/baseline/training_t6_engine.py": "99b40f4838f05ec5e3aabb4df5a56a6f7e826ad4c3ffbe3dfcf9cfd5508454a3",
    "src/sparse_rtdetr/baseline/training_t6_entry.py": "d433c5fb91f933eb731ae08e3c43b7cffae9071a9c3d22d227e01791b28a16c1",
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py": "0dd192b1137223e19baa23ec10611767428d4f422b297d5442281b6834d89aeb",
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py": "723fc69abe1bfb3a8660134e5477b228708ea9941acfa5021213f1f67530ed67",
}
T6B_SUPPORT_IDENTITIES = {
    "tests/test_rtdetr_baseline_training_t6b.py": "b7ce80ea8ab9b4756eded2ab163b74e5fd16250bc814a9266e228f3e0d345fcb",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_T6B_PRODUCTION_BOUNDARY_V1.md": "9a9edd1300f132601d14156e0ec25f01c65db2116611a28061858f0cd9935a67",
}
T6B_PUBLIC_APIS = {
    "src/sparse_rtdetr/baseline/training_t6_engine.py": {
        "TrainingEngineError",
        "validate_training_policy",
        "resolve_production_ports",
        "run_training_engine",
    },
    "src/sparse_rtdetr/baseline/training_t6_entry.py": {
        "TrainingEntryError",
        "load_t6b_config",
        "t6b_config_binding",
        "current_source_bindings",
        "authorization_file_identity",
        "validate_detached_authorization",
        "consume_owner_authorization",
        "validate_consumed_authorization_receipt",
        "build_production_entry_descriptor",
        "validate_entry_descriptor",
        "run_production_entry",
        "validate_entry_result",
        "classify_entry",
    },
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py": {
        "TrainingProcessError",
        "build_production_process_descriptor",
        "validate_process_descriptor",
        "run_process_once",
        "validate_process_result",
        "classify_process",
    },
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py": {
        "TrainingOuterError",
        "build_production_outer_descriptor",
        "validate_outer_descriptor",
        "run_outer_once",
        "validate_outer_result",
        "classify_outer",
    },
}
T6B_ALLOWED_IMPORTS = {
    "src/sparse_rtdetr/baseline/training_t6_engine.py": {"copy", "hashlib", "importlib", "json", "math", "os", "pathlib", "random", "stat", "typing", "sparse_rtdetr.baseline"},
    "src/sparse_rtdetr/baseline/training_t6_entry.py": {"copy", "datetime", "hashlib", "json", "math", "os", "stat", "subprocess", "sys", "pathlib", "typing", "sparse_rtdetr.baseline"},
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py": {"copy", "hashlib", "inspect", "json", "math", "os", "stat", "subprocess", "sys", "pathlib", "typing", "sparse_rtdetr.baseline"},
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py": {"copy", "datetime", "hashlib", "inspect", "json", "math", "os", "stat", "subprocess", "sys", "time", "pathlib", "typing", "sparse_rtdetr.baseline"},
}
T6B_MODULE_RELATIVE_PATHS = (
    "src/sparse_rtdetr/baseline/training_t6_entry.py",
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py",
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py",
    "src/sparse_rtdetr/baseline/training_t6_engine.py",
)

BASELINE_MODEL_IMPORT_FILES = {
    "src/sparse_rtdetr/baseline/categories.py",
    "src/sparse_rtdetr/baseline/postprocessor.py",
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_evidence.py",
    "tests/test_rtdetr_baseline_smoke.py",
}

SCIENTIFIC_ENTRY_ROOTS = (
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_launcher.py",
    "src/sparse_rtdetr/baseline/smoke_outer_launcher.py",
)
SCIENTIFIC_SOURCE_ALLOWLIST = (
    "src/sparse_rtdetr/__init__.py",
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
    "src/sparse_rtdetr/data_protocol/__init__.py",
    "src/sparse_rtdetr/data_protocol/categories.py",
    "src/sparse_rtdetr/data_protocol/converter.py",
    "src/sparse_rtdetr/data_protocol/evaluation.py",
    "src/sparse_rtdetr/data_protocol/lineage.py",
    "src/sparse_rtdetr/data_protocol/parser.py",
    "src/sparse_rtdetr/data_protocol/protocol.py",
    "src/sparse_rtdetr/data_protocol/schema.py",
    "src/sparse_rtdetr/data_protocol/split.py",
)

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


def _local_import_targets(root: Path, source: Path) -> set[Path]:
    """Resolve AST-visible imports inside the repository's own package."""

    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError):
        return set()
    package_parts = source.relative_to(root / "src").with_suffix("").parts[:-1]
    targets: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                prefix = package_parts[:keep]
                module_parts = tuple(node.module.split(".")) if node.module else ()
                base = (*prefix, *module_parts)
                modules = [
                    ".".join((*base, alias.name)) for alias in node.names
                ] + ([".".join(base)] if base else [])
            else:
                base = tuple(node.module.split(".")) if node.module else ()
                modules = [
                    ".".join((*base, alias.name)) for alias in node.names
                ] + ([".".join(base)] if base else [])
        else:
            continue
        for module in modules:
            if not module or not module.startswith("sparse_rtdetr"):
                continue
            candidate = root / "src" / Path(*module.split("."))
            file_candidate = candidate.with_suffix(".py")
            package_candidate = candidate / "__init__.py"
            if file_candidate.is_file():
                targets.add(file_candidate)
            elif package_candidate.is_file():
                targets.add(package_candidate)
    return targets


def _scientific_dependency_closure(root: Path) -> set[str]:
    pending = [root / relative for relative in SCIENTIFIC_ENTRY_ROOTS]
    seen: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        package = source.parent
        while package != root / "src" and package.is_relative_to(root / "src"):
            init = package / "__init__.py"
            if init.is_file() and init not in seen:
                pending.append(init)
            package = package.parent
        pending.extend(_local_import_targets(root, source) - seen)
    return {path.relative_to(root).as_posix() for path in seen}


def _scientific_dependency_failures(root: Path) -> list[str]:
    closure = _scientific_dependency_closure(root)
    allowlist = set(SCIENTIFIC_SOURCE_ALLOWLIST)
    failures = []
    for relative in sorted(closure - allowlist):
        failures.append(f"scientific source dependency missing from allowlist: {relative}")
    for relative in sorted(allowlist - closure):
        failures.append(f"scientific source allowlist contains non-closure file: {relative}")
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


def _check_training_runtime_plan(root: Path, failures: list[str]) -> None:
    """Check only the frozen T5A config/module identity and public surface."""

    config_path = root / TRAINING_RUNTIME_PLAN_RELATIVE_PATH
    module_path = root / TRAINING_RUNTIME_MODULE_RELATIVE_PATH
    if config_path.is_symlink() or not config_path.is_file():
        failures.append("missing or symlinked training runtime plan")
        return
    if module_path.is_symlink() or not module_path.is_file():
        failures.append("missing or symlinked training runtime module")
        return
    try:
        raw = config_path.read_bytes()
        if len(raw) != TRAINING_RUNTIME_PLAN_RAW_SIZE_BYTES:
            failures.append("training runtime plan raw size mismatch")
        if _sha256(config_path) != TRAINING_RUNTIME_PLAN_RAW_SHA256:
            failures.append("training runtime plan raw SHA mismatch")
        document = json.loads(raw.decode("utf-8"))
        if type(document) is not dict:
            failures.append("training runtime plan root is not an object")
        else:
            if document.get("schema_version") != TRAINING_RUNTIME_PLAN_SCHEMA_VERSION:
                failures.append("training runtime plan schema version mismatch")
            if document.get("runtime_plan_id") != TRAINING_RUNTIME_PLAN_ID:
                failures.append("training runtime plan ID mismatch")
            if document.get("training_contract_id") != "rtdetrv2_r18_visdrone_baseline_training_v1":
                failures.append("training runtime plan training contract ID mismatch")
            if document.get("runtime_stage") != "pre_cuda_plan_only":
                failures.append("training runtime plan stage mismatch")
            source = document.get("source_bindings")
            if not isinstance(source, dict) or source.get("training_contract", {}).get("module_sha256") != _sha256(
                root / "src/sparse_rtdetr/baseline/training_contract.py"
            ):
                failures.append("training runtime plan training-contract module identity mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        failures.append(f"training runtime plan parse failure: {type(exc).__name__}")

    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != TRAINING_RUNTIME_ALLOWED_IMPORTS:
            failures.append(
                "training runtime import set mismatch: "
                f"extra={sorted(imported - TRAINING_RUNTIME_ALLOWED_IMPORTS)} "
                f"missing={sorted(TRAINING_RUNTIME_ALLOWED_IMPORTS - imported)}"
            )
        public_functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        }
        if public_functions != TRAINING_RUNTIME_PUBLIC_APIS:
            failures.append(
                "training runtime public API mismatch: "
                f"extra={sorted(public_functions - TRAINING_RUNTIME_PUBLIC_APIS)} "
                f"missing={sorted(TRAINING_RUNTIME_PUBLIC_APIS - public_functions)}"
            )
        all_values = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                all_values = ast.literal_eval(node.value)
        if set(all_values) != TRAINING_RUNTIME_PUBLIC_APIS:
            failures.append("training runtime __all__ mismatch")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"training runtime source audit failure: {type(exc).__name__}")


def _check_training_evidence(root: Path, failures: list[str]) -> None:
    """Check the frozen T5B schema surface without importing execution code."""

    expected_hashes = {
        TRAINING_EVIDENCE_MODULE_RELATIVE_PATH: TRAINING_EVIDENCE_MODULE_SHA256,
        TRAINING_EVIDENCE_TEST_RELATIVE_PATH: TRAINING_EVIDENCE_TEST_SHA256,
        TRAINING_EVIDENCE_DOCUMENT_RELATIVE_PATH: TRAINING_EVIDENCE_DOCUMENT_SHA256,
    }
    for relative in sorted(TRAINING_EVIDENCE_FILES):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing or symlinked T5B file: {relative}")
            continue
        if relative in expected_hashes and _sha256(path) != expected_hashes[relative]:
            failures.append(f"T5B file identity mismatch: {relative}")

    config_path = root / TRAINING_EVIDENCE_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != TRAINING_EVIDENCE_CONFIG_RAW_SIZE_BYTES:
            failures.append("training evidence config raw size mismatch")
        if _sha256(config_path) != TRAINING_EVIDENCE_CONFIG_RAW_SHA256:
            failures.append("training evidence config raw SHA mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("training evidence config portable bytes mismatch")
        value = json.loads(raw.decode("utf-8"))
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != TRAINING_EVIDENCE_CONFIG_CANONICAL_SIZE_BYTES:
            failures.append("training evidence config canonical size mismatch")
        if hashlib.sha256(canonical).hexdigest() != TRAINING_EVIDENCE_CONFIG_CANONICAL_SHA256:
            failures.append("training evidence config canonical SHA mismatch")
        if value.get("schema_version") != 1 or value.get("training_evidence_contract_id") != "rtdetrv2_r18_visdrone_baseline_training_evidence_v1":
            failures.append("training evidence config semantic identity mismatch")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"training evidence config parse failure: {type(exc).__name__}")

    module_path = root / TRAINING_EVIDENCE_MODULE_RELATIVE_PATH
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != TRAINING_EVIDENCE_ALLOWED_IMPORTS:
            failures.append(
                "training evidence import set mismatch: "
                f"extra={sorted(imported - TRAINING_EVIDENCE_ALLOWED_IMPORTS)} "
                f"missing={sorted(TRAINING_EVIDENCE_ALLOWED_IMPORTS - imported)}"
            )
        public_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public_names != TRAINING_EVIDENCE_PUBLIC_APIS:
            failures.append(
                "training evidence public API mismatch: "
                f"extra={sorted(public_names - TRAINING_EVIDENCE_PUBLIC_APIS)} "
                f"missing={sorted(TRAINING_EVIDENCE_PUBLIC_APIS - public_names)}"
            )
        all_values: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                all_values = ast.literal_eval(node.value)
        if set(all_values) != TRAINING_EVIDENCE_PUBLIC_APIS:
            failures.append("training evidence __all__ mismatch")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"training evidence source audit failure: {type(exc).__name__}")

    production_root = root / TRAINING_EVIDENCE_ROOT_RELATIVE_PATH
    if production_root.is_symlink() or production_root.exists():
        failures.append("T5B production evidence root must not exist")


def _check_prepared_adapter(root: Path, failures: list[str]) -> None:
    """Check the T5C source boundary without importing or executing it."""

    expected_hashes = {
        PREPARED_ADAPTER_MODULE_RELATIVE_PATH: PREPARED_ADAPTER_MODULE_SHA256,
        PREPARED_ADAPTER_TEST_RELATIVE_PATH: PREPARED_ADAPTER_TEST_SHA256,
        PREPARED_ADAPTER_DOCUMENT_RELATIVE_PATH: PREPARED_ADAPTER_DOCUMENT_SHA256,
    }
    for relative in sorted(PREPARED_ADAPTER_FILES):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing or symlinked T5C file: {relative}")
            continue
        if _sha256(path) != expected_hashes[relative]:
            failures.append(f"T5C file identity mismatch: {relative}")

    module_path = root / PREPARED_ADAPTER_MODULE_RELATIVE_PATH
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
                if any(alias.name == "*" for alias in node.names):
                    failures.append("T5C wildcard import is forbidden")
        if imported != PREPARED_ADAPTER_ALLOWED_IMPORTS:
            failures.append(
                "T5C import set mismatch: "
                f"extra={sorted(imported - PREPARED_ADAPTER_ALLOWED_IMPORTS)} "
                f"missing={sorted(PREPARED_ADAPTER_ALLOWED_IMPORTS - imported)}"
            )
        forbidden_imports = set()
        for module in imported:
            root_name = module.split(".", 1)[0].casefold()
            if root_name in PREPARED_ADAPTER_FORBIDDEN_IMPORT_ROOTS:
                forbidden_imports.add(module)
        if forbidden_imports:
            failures.append(f"T5C forbidden import: {sorted(forbidden_imports)}")

        public_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        }
        if public_names != PREPARED_ADAPTER_PUBLIC_NAMES:
            failures.append(
                "T5C public API mismatch: "
                f"extra={sorted(public_names - PREPARED_ADAPTER_PUBLIC_NAMES)} "
                f"missing={sorted(PREPARED_ADAPTER_PUBLIC_NAMES - public_names)}"
            )
        all_values: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                all_values = ast.literal_eval(node.value)
        if (
            type(all_values) is not list
            and type(all_values) is not tuple
        ) or len(all_values) != len(set(all_values)) or set(all_values) != PREPARED_ADAPTER_PUBLIC_NAMES:
            failures.append("T5C __all__ mismatch")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                symbol = None
                if isinstance(node.func, ast.Name):
                    symbol = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    symbol = node.func.attr
                if symbol is not None and symbol.casefold() in PREPARED_ADAPTER_FORBIDDEN_OPERATION_NAMES:
                    failures.append(f"T5C forbidden operation call: {symbol}")
            elif isinstance(node, ast.Attribute) and node.attr.casefold() in {
                "cuda",
                "dataloader",
                "dataset",
                "device",
                "nvidia",
            }:
                failures.append(f"T5C forbidden operation attribute: {node.attr}")
            elif isinstance(node, ast.Name) and node.id.casefold() in {
                "torch",
                "cupy",
                "cuda",
                "nvidia",
                "dataloader",
                "dataset",
            }:
                failures.append(f"T5C forbidden operation name: {node.id}")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"T5C source audit failure: {type(exc).__name__}")


def _check_training_process(root: Path, failures: list[str]) -> None:
    """Check the T5D process/entry boundary without executing it."""

    expected_hashes = {
        TRAINING_PROCESS_CONFIG_RELATIVE_PATH: TRAINING_PROCESS_CONFIG_RAW_SHA256,
        TRAINING_ENTRY_RELATIVE_PATH: TRAINING_ENTRY_SHA256,
        TRAINING_LAUNCHER_RELATIVE_PATH: TRAINING_LAUNCHER_SHA256,
        TRAINING_ENTRY_TEST_RELATIVE_PATH: "",
        TRAINING_PROCESS_TEST_RELATIVE_PATH: "",
        TRAINING_PROCESS_DOCUMENT_RELATIVE_PATH: TRAINING_PROCESS_DOCUMENT_SHA256,
    }
    for relative, expected in expected_hashes.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing or symlinked T5D file: {relative}")
            continue
        if expected and _sha256(path) != expected:
            failures.append(f"T5D file identity mismatch: {relative}")

    config_path = root / TRAINING_PROCESS_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != TRAINING_PROCESS_CONFIG_RAW_SIZE_BYTES or _sha256(config_path) != TRAINING_PROCESS_CONFIG_RAW_SHA256:
            failures.append("T5D process config raw identity mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("T5D process config portable bytes mismatch")
        value = json.loads(raw.decode("utf-8"))
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != TRAINING_PROCESS_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != TRAINING_PROCESS_CONFIG_CANONICAL_SHA256:
            failures.append("T5D process config canonical identity mismatch")
        if type(value) is not dict or value.get("schema_version") != 1 or value.get("process_contract_id") != "rtdetrv2_r18_visdrone_baseline_training_process_v1" or value.get("mode") != "synthetic":
            failures.append("T5D process config identity mismatch")
        readiness = value.get("readiness")
        if not isinstance(readiness, dict) or any(readiness.get(field) is not False for field in ("production_training_authorized", "real_entry_execution_authorized", "real_process_launch_authorized", "training_implementation_ready", "training_ready", "model_selection_certified", "confirmatory_access", "test_access", "speed_measurement")):
            failures.append("T5D readiness is not fail-closed")
        source = value.get("source_bindings")
        modules = source.get("repository_modules") if isinstance(source, dict) else None
        if modules != [{"role": role, "relative_path": relative} for role, relative in _training_process_module_rows()]:
            failures.append("T5D source module closure mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        failures.append("T5D process config parse failure")

    def source_surface(relative: str, expected_imports: set[str], expected_public: set[str]) -> None:
        path = root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    imported.add(node.module or "")
            if imported != expected_imports:
                failures.append(f"T5D import set mismatch: {relative}")
            public = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_") and node.name != "main"
            }
            if public != expected_public:
                failures.append(f"T5D public API mismatch: {relative}")
            all_values = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    all_values = ast.literal_eval(node.value)
            if set(all_values) != expected_public:
                failures.append(f"T5D __all__ mismatch: {relative}")
            forbidden = {"torch", "subprocess", "multiprocessing", "socket", "requests", "urllib", "vendor", "dataloader", "dataset"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id.casefold() in forbidden:
                    failures.append(f"T5D forbidden source name: {relative}:{node.id}")
        except (OSError, UnicodeError, SyntaxError, ValueError, TypeError):
            failures.append(f"T5D source audit failure: {relative}")

    source_surface(TRAINING_ENTRY_RELATIVE_PATH, TRAINING_ENTRY_ALLOWED_IMPORTS, TRAINING_ENTRY_PUBLIC_NAMES)
    source_surface(TRAINING_LAUNCHER_RELATIVE_PATH, TRAINING_LAUNCHER_ALLOWED_IMPORTS, TRAINING_LAUNCHER_PUBLIC_NAMES)


def _check_t6a_owner_authorization(root: Path, failures: list[str]) -> None:
    """Opt-in, versioned static check for the detached T6A boundary."""

    for relative in sorted(T6A_FILES):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing or symlinked T6A file: {relative}")

    config_path = root / T6A_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != T6A_CONFIG_RAW_SIZE_BYTES or _sha256(config_path) != T6A_CONFIG_RAW_SHA256:
            failures.append("T6A config raw identity mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("T6A config portable bytes mismatch")

        def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result

        value = json.loads(raw[:-1].decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != T6A_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != T6A_CONFIG_CANONICAL_SHA256:
            failures.append("T6A config canonical identity mismatch")
        if type(value) is not dict:
            failures.append("T6A config root is not an object")
        else:
            if value.get("schema_version") != 1 or value.get("contract_id") != "rtdetrv2_r18_visdrone_baseline_training_launch_t6_v1":
                failures.append("T6A config semantic identity mismatch")
            if value.get("stage") != "T6A" or value.get("status") != "DETACHED_AUTHORIZATION_REQUIRED":
                failures.append("T6A config stage/status mismatch")
            readiness = value.get("readiness")
            if not isinstance(readiness, dict) or readiness.get("static_config_authorizes_production") is not False:
                failures.append("T6A static config authorizes production")

            def contains_key(item: object, key: str) -> bool:
                if isinstance(item, dict):
                    return key in item or any(contains_key(child, key) for child in item.values())
                if isinstance(item, list):
                    return any(contains_key(child, key) for child in item)
                return False

            if contains_key(value, "authorized") or contains_key(value, "nonce"):
                failures.append("T6A static config contains detached-only field")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"T6A config parse failure: {type(exc).__name__}")

    module_path = root / T6A_MODULE_RELATIVE_PATH
    try:
        if T6A_MODULE_SHA256.startswith("PENDING_") or _sha256(module_path) != T6A_MODULE_SHA256:
            failures.append("T6A module identity mismatch")
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != T6A_ALLOWED_IMPORTS:
            failures.append(
                "T6A import set mismatch: "
                f"extra={sorted(imported - T6A_ALLOWED_IMPORTS)} "
                f"missing={sorted(T6A_ALLOWED_IMPORTS - imported)}"
            )
        public_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public_names != T6A_PUBLIC_NAMES:
            failures.append(
                "T6A public API mismatch: "
                f"extra={sorted(public_names - T6A_PUBLIC_NAMES)} "
                f"missing={sorted(T6A_PUBLIC_NAMES - public_names)}"
            )
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or set(all_values) != T6A_PUBLIC_NAMES or len(all_values) != len(set(all_values)):
            failures.append("T6A __all__ mismatch")
        forbidden = {"torch", "subprocess", "tmux", "dataloader", "dataset", "nvidia"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.casefold() in forbidden:
                failures.append(f"T6A forbidden source name: {node.id}")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"T6A source audit failure: {type(exc).__name__}")

    document_path = root / T6A_DOCUMENT_RELATIVE_PATH
    test_path = root / T6A_TEST_RELATIVE_PATH
    for path, label in ((document_path, "document"), (test_path, "test")):
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                failures.append(f"T6A {label} identity is invalid")
        except OSError:
            failures.append(f"T6A {label} is unavailable")


def _check_t6b_production_boundary(root: Path, failures: list[str]) -> None:
    """Check the versioned T6B boundary without importing its launch code."""

    expected_hashes = {
        T6B_CONFIG_RELATIVE_PATH: T6B_CONFIG_RAW_SHA256,
        **T6B_SOURCE_IDENTITIES,
        **T6B_SUPPORT_IDENTITIES,
    }
    for relative in sorted(T6B_FILES):
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"T6B file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"T6B file nlink drift: {relative}")
            expected = expected_hashes.get(relative)
            if expected and not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"T6B file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing T6B file: {relative}")

    config_path = root / T6B_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != T6B_CONFIG_RAW_SIZE_BYTES or _sha256(config_path) != T6B_CONFIG_RAW_SHA256:
            failures.append("T6B config raw identity mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("T6B config portable bytes mismatch")

        def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result

        value = json.loads(raw[:-1].decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != T6B_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != T6B_CONFIG_CANONICAL_SHA256:
            failures.append("T6B config canonical identity mismatch")
        if type(value) is not dict:
            failures.append("T6B config root is not an object")
        else:
            required = {"schema_version", "contract_id", "stage", "status", "t6a_binding", "repository", "source_policy", "run_identity", "targets", "environment_identity", "data_roles", "training_policy", "evidence_policy", "state_machine", "failure_policy", "invocation_policy", "readiness"}
            if set(value) != required:
                failures.append("T6B config key set drift")
            if value.get("schema_version") != 1 or value.get("contract_id") != "rtdetrv2_r18_visdrone_baseline_training_t6b_v1" or value.get("stage") != "T6B" or value.get("status") != "PRODUCTION_BOUNDARY_IMPLEMENTED_DETACHED_AUTH_REQUIRED":
                failures.append("T6B config semantic identity mismatch")
            if value.get("t6a_binding") != {"contract_id": "rtdetrv2_r18_visdrone_baseline_training_launch_t6_v1", "config_relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_launch_t6_v1.json", "authorization_source": "external_detached_owner_artifact", "validation_api": "owner_authorization_binding", "binding_required": True}:
                failures.append("T6B T6A binding policy mismatch")
            if value.get("source_policy", {}).get("t6b_modules") != list(T6B_MODULE_RELATIVE_PATHS) or value.get("source_policy", {}).get("t6a_binding_required") is not True or value.get("source_policy", {}).get("launch_time_sha_required") is not True:
                failures.append("T6B source policy mismatch")
            if value.get("run_identity") != {"run_id_source": "detached_owner_authorization", "nonce_source": "detached_owner_authorization", "session_source": "detached_owner_authorization", "all_unique": True, "targets_absent_before_launch": True}:
                failures.append("T6B run identity policy mismatch")
            targets = value.get("targets")
            expected_targets = {"training_evidence_root": "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1", "process_evidence_root": "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_t6b_v1", "outer_evidence_root": "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_t6b_v1", "receipt_policy": "adjacent_exclusive_mode_0600", "lock_policy": "adjacent_exclusive_mode_0600"}
            if targets != expected_targets:
                failures.append("T6B target policy mismatch")
            data_roles = value.get("data_roles")
            if type(data_roles) is not dict or data_roles.get("test") != {"role": "test", "access": "forbidden", "identity_read": "forbidden"} or data_roles.get("confirmatory") != {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}:
                failures.append("T6B sealed data policy mismatch")
            readiness = value.get("readiness")
            if readiness != {"static_config_authorizes_production": False, "owner_authorization_required": True, "launch_acceptance_separate": True, "terminal_completion_separate": True, "independent_audit_required": True, "training_certification_separate": True}:
                failures.append("T6B readiness is not fail-closed")
            invocation = value.get("invocation_policy")
            if invocation != {"outer_calls": 1, "tmux_new_session_calls": 1, "child_calls": 1, "shell": False, "argv_sequence": True, "network": False, "speed_measurement": False}:
                failures.append("T6B invocation policy mismatch")
            state_machine = value.get("state_machine")
            if type(state_machine) is not dict or state_machine.get("terminal_success") != "TERMINAL_COMPLETE" or state_machine.get("terminal_failure") != "PERMANENT_FAIL" or state_machine.get("no_resume_after") != "LAUNCH_ACCEPTED" or state_machine.get("no_retry_after") != "LAUNCH_ACCEPTED" or state_machine.get("no_overwrite_after") != "LAUNCH_ACCEPTED" or state_machine.get("audit_required") is not True or state_machine.get("certification_separate") is not True:
                failures.append("T6B state-machine policy mismatch")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"T6B config parse failure: {type(exc).__name__}")

    def source_surface(relative: str) -> None:
        path = root / relative
        try:
            source_text = path.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    imported.add(node.module or "")
            if imported != T6B_ALLOWED_IMPORTS[relative]:
                failures.append(f"T6B import set mismatch: {relative}")
            public = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_") and node.name != "main"
            }
            if public != T6B_PUBLIC_APIS[relative]:
                failures.append(f"T6B public API mismatch: {relative}")
            all_values: object = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    all_values = ast.literal_eval(node.value)
            if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != T6B_PUBLIC_APIS[relative]:
                failures.append(f"T6B __all__ mismatch: {relative}")

            if relative == "src/sparse_rtdetr/baseline/training_t6_engine.py":
                forbidden_capability_names = {"_ProductionCapability", "_PRODUCTION_CAPABILITY", "_authorization_capability"}
                if any(name in source_text for name in forbidden_capability_names):
                    failures.append("T6B engine exposes legacy capability authorization")
                engine_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training_engine"]
                if len(engine_functions) != 1:
                    failures.append("T6B engine run_training_engine definition count mismatch")
                elif not any(argument.arg == "authorization_context" for argument in engine_functions[0].args.args + engine_functions[0].args.kwonlyargs):
                    failures.append("T6B engine lacks bound authorization_context API")
                if "_claim_engine_execution" not in source_text or "_validate_execution_context" not in source_text:
                    failures.append("T6B engine lacks execution context and exclusive claim gates")
                if "production_authorized" in source_text or "authorize_production" in source_text:
                    failures.append("T6B engine retains boolean/callback production authorization")
                claim_calls = [
                    node
                    for node in ast.walk(engine_functions[0])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_claim_engine_execution"
                ] if engine_functions else []
                if not claim_calls:
                    failures.append("T6B engine does not claim execution")
            if relative == "src/sparse_rtdetr/baseline/training_t6_entry.py":
                entry_source_forbidden = {"_authorization_capability", "_PRODUCTION_CAPABILITY", "production_authorized", "authorize_production"}
                if any(name in source_text for name in entry_source_forbidden):
                    failures.append("T6B entry retains legacy production injection authorization")
                if "authorization_context" not in source_text or "injected production engine ports are forbidden" not in source_text:
                    failures.append("T6B entry does not bind and close production ports")

            forbidden_roots = {"torch", "cupy", "numpy", "pandas", "tensorflow", "ultralytics", "dataloader", "dataset", "vendor"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                        failures.append(f"T6B forbidden import: {relative}")
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                    failures.append(f"T6B forbidden import: {relative}")
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "Popen":
                        shell = [keyword.value for keyword in node.keywords if keyword.arg == "shell"]
                        if len(shell) != 1 or not isinstance(shell[0], ast.Constant) or shell[0].value is not False:
                            failures.append(f"T6B Popen must use shell=False: {relative}")
                    if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                        failures.append(f"T6B dynamic execution: {relative}")
                    if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen"}:
                        failures.append(f"T6B shell operation: {relative}")

            for statement in tree.body:
                if isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Name) and statement.test.left.id == "__name__":
                    continue
                if not isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                    failures.append(f"T6B import-time call: {relative}")
        except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
            failures.append(f"T6B source audit failure: {relative}: {type(exc).__name__}")

    for relative in T6B_MODULE_RELATIVE_PATHS:
        source_surface(relative)

    for relative in ("tests/test_rtdetr_baseline_training_t6b.py", "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_T6B_PRODUCTION_BOUNDARY_V1.md"):
        path = root / relative
        try:
            if path.stat().st_size <= 0:
                failures.append(f"empty T6B support file: {relative}")
            expected = expected_hashes.get(relative)
            if expected and not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"T6B support file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing T6B support file: {relative}")

    for relative in (
        "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
    ):
        target = root / relative
        if target.exists() or target.is_symlink():
            failures.append(f"T6B production target must remain absent: {relative}")


def _training_process_module_rows() -> tuple[tuple[str, str], ...]:
    return (
        ("package", "src/sparse_rtdetr/__init__.py"),
        ("package", "src/sparse_rtdetr/baseline/__init__.py"),
        ("entry", "src/sparse_rtdetr/baseline/training_entry.py"),
        ("launcher", "src/sparse_rtdetr/baseline/training_process_launcher.py"),
        ("prepared_adapter", "src/sparse_rtdetr/baseline/training_adapter.py"),
        ("training_contract", "src/sparse_rtdetr/baseline/training_contract.py"),
        ("runtime_plan", "src/sparse_rtdetr/baseline/training_runtime.py"),
        ("training_evidence", "src/sparse_rtdetr/baseline/training_evidence.py"),
        ("primary_evaluator", "src/sparse_rtdetr/baseline/primary_evaluator.py"),
        ("package", "src/sparse_rtdetr/data_protocol/__init__.py"),
        ("evaluation_protocol", "src/sparse_rtdetr/data_protocol/evaluation.py"),
        ("protocol_schema", "src/sparse_rtdetr/data_protocol/schema.py"),
    )


def check_repository(root: Path) -> bool:
    failures: list[str] = []
    files, file_policy_failures = _file_policy(root)
    failures.extend(file_policy_failures)

    vendor_files = {relative for relative in files if relative.startswith(VENDOR_PREFIX)}
    allowed_files = ALLOWED_FILES | vendor_files
    allowed_files |= BASELINE_FILES
    allowed_files |= TRAINING_EVIDENCE_FILES
    allowed_files |= PRIMARY_EVALUATOR_FILES
    allowed_files |= PREPARED_ADAPTER_FILES
    allowed_files |= TRAINING_PROCESS_FILES
    allowed_files |= T6A_FILES
    allowed_files |= T6B_FILES
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
    failures.extend(_scientific_dependency_failures(root))

    try:
        import importlib.util

        contract_path = root / "src/sparse_rtdetr/baseline/training_contract.py"
        spec = importlib.util.spec_from_file_location("p3_training_contract_check", contract_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("training contract module spec unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        binding = module.training_contract_binding(root)
        if binding["canonical_sha256"] != module.TRAINING_CONTRACT_CANONICAL_SHA256:
            failures.append("formal training contract canonical identity mismatch")
    except Exception as exc:
        failures.append(f"formal training contract validation failure: {type(exc).__name__}: {exc}")

    _check_training_runtime_plan(root, failures)
    _check_training_evidence(root, failures)
    _check_prepared_adapter(root, failures)
    _check_training_process(root, failures)
    _check_t6a_owner_authorization(root, failures)
    _check_t6b_production_boundary(root, failures)

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
