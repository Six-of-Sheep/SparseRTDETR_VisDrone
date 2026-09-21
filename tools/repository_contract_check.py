"""Standard-library-only checks for the initial P3 repository contract."""

from __future__ import annotations

import hashlib
import argparse
import ast
import json
import os
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

V2B_ENGINEERING_FILES = {
    'src/sparse_rtdetr/baseline/training_v2b.py',
    'src/sparse_rtdetr/baseline/training_v2b_engine.py',
    'src/sparse_rtdetr/baseline/training_v2b_checkpoint.py',
    'src/sparse_rtdetr/baseline/training_v2b_evidence.py',
    'tests/test_rtdetr_baseline_training_v2b.py',
    'tests/test_rtdetr_baseline_training_v2b_engine.py',
    'tests/test_rtdetr_baseline_training_v2b_checkpoint.py',
    'tests/test_rtdetr_baseline_training_v2b_evidence.py',
    'tools/verify_training_v2b_cpu.py',
    'configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_engineering.json',
    'docs/contracts/RTDETR_BASELINE_V2B_ENGINEERING.md',
}

V2B_SAMPLING_FILES = {
    'src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py',
    'tests/test_rtdetr_baseline_training_v2b_deterministic_sampling.py',
}

# Keep the earlier foundation archive valid under the current checker. The
# runtime extension is one additional explicit file group, never a glob permit.
V2B_RUNTIME_FILES = {
    'tests/conftest.py',
    'src/sparse_rtdetr/baseline/training_v2b_data.py',
    'src/sparse_rtdetr/baseline/training_v2b_device.py',
    'src/sparse_rtdetr/baseline/training_v2b_hardware.py',
    'src/sparse_rtdetr/baseline/training_v2b_runtime.py',
    'tests/test_rtdetr_baseline_training_v2b_data.py',
    'tests/test_rtdetr_baseline_training_v2b_device.py',
    'tests/test_rtdetr_baseline_training_v2b_hardware.py',
    'tests/test_rtdetr_baseline_training_v2b_runtime.py',
    'tests/test_rtdetr_baseline_training_v2b_prerun.py',
    'tools/verify_training_v2b_prerun.py',
    'configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_runtime.json',
    'docs/contracts/RTDETR_BASELINE_V2B_RUNTIME.md',
}

V2B_PAIRED_CONTROL_FILES = {
    'docs/contracts/RTDETR_BASELINE_V2B_PAIRED_CONTROL.md',
    'src/sparse_rtdetr/baseline/training_v2b_admission.py',
    'src/sparse_rtdetr/baseline/training_v2b_campaign.py',
    'src/sparse_rtdetr/baseline/training_v2b_control.py',
    'src/sparse_rtdetr/baseline/training_v2b_development.py',
    'tests/test_rtdetr_baseline_training_v2b_admission.py',
    'tests/test_rtdetr_baseline_training_v2b_campaign.py',
    'tests/test_rtdetr_baseline_training_v2b_control.py',
    'tests/test_rtdetr_baseline_training_v2b_development.py',
    'tools/run_training_v2b_campaign.py',
    'tools/run_training_v2b_control.py',
}

V2B_896_MATCHED_FILES = {
    'docs/contracts/RTDETR_BASELINE_V2B_896_MATCHED_CONTROL.md',
    'src/sparse_rtdetr/baseline/training_v2b_geometry.py',
    'src/sparse_rtdetr/baseline/training_v2b_resolution.py',
    'src/sparse_rtdetr/baseline/training_v2b_resolution_campaign.py',
    'tests/test_training_v2b_resolution_contract.py',
    'tests/test_training_v2b_resolution_development.py',
    'tests/test_training_v2b_resolution_training.py',
    'tests/test_training_v2b_resolution_campaign.py',
    'tools/run_training_v2b_resolution_campaign.py',
}

# Exact development-evidence extension; older engineering archives remain valid.
V2B_RESOLUTION_EVIDENCE_FILES = {
    'docs/contracts/RTDETR_BASELINE_V2B_RESOLUTION_EVIDENCE.md',
    'src/sparse_rtdetr/baseline/training_v2b_cross_eval.py',
    'src/sparse_rtdetr/baseline/training_v2b_evidence_campaign.py',
    'src/sparse_rtdetr/baseline/training_v2b_official_gt.py',
    'src/sparse_rtdetr/baseline/training_v2b_primary_cache.py',
    'src/sparse_rtdetr/baseline/training_v2b_resolution_analysis.py',
    'src/sparse_rtdetr/baseline/training_v2b_resolution_diagnostics.py',
    'tests/test_rtdetr_baseline_training_v2b_cross_eval.py',
    'tests/test_rtdetr_baseline_training_v2b_development_admission.py',
    'tests/test_training_v2b_evidence_campaign.py',
    'tests/test_training_v2b_primary_cache.py',
    'tests/test_training_v2b_resolution_analysis.py',
    'tests/test_training_v2b_resolution_diagnostics.py',
    'tools/analyze_training_v2b_resolution.py',
    'tools/run_training_v2b_cross_eval.py',
    'tools/run_training_v2b_evidence_campaign.py',
}


# Explicit paired-seed execution and fixed-epoch evidence extension.
V2B_PAIRED_SEED_FILES = {
    'docs/contracts/RTDETR_BASELINE_V2B_PAIRED_SEED_REPLICATION.md',
    'src/sparse_rtdetr/baseline/training_v2b_replication_campaign.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_capacity.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_gate.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_worker.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_summary.py',
    'tests/test_training_v2b_replication_campaign.py',
    'tests/test_training_v2b_replication_capacity.py',
    'tests/test_training_v2b_replication_gate.py',
    'tests/test_training_v2b_replication_worker.py',
    'tests/test_training_v2b_replication_summary.py',
    'tools/run_training_v2b_replication_campaign.py',
    'tools/run_training_v2b_replication_worker.py',
    'tools/summarize_training_v2b_replication.py',
}

V2B_SEED2_COMPLETION_FILES = {
    'src/sparse_rtdetr/baseline/training_v2b_seed2_completion.py',
    'src/sparse_rtdetr/baseline/training_v2b_seed2_completion_campaign.py',
    'src/sparse_rtdetr/baseline/training_v2b_seed2_completion_summary.py',
    'src/sparse_rtdetr/baseline/training_v2b_seed2_completion_worker.py',
    'src/sparse_rtdetr/baseline/training_v2b_seed2_process_logs.py',
    'tests/test_training_v2b_seed2_completion.py',
    'tests/test_training_v2b_seed2_completion_summary.py',
    'tests/test_training_v2b_seed2_process_logs.py',
    'tools/run_training_v2b_seed2_completion_campaign.py',
    'tools/run_training_v2b_seed2_completion_worker.py',
    'tools/summarize_training_v2b_seed2_completion.py',
}

# Truthful cross-campaign continuation from four certified epoch-30 boundaries.
# Training-core bytes stay frozen; these files own only lineage, orchestration,
# fail-closed stage execution, and independently recomputable evidence.
V2B_EPOCH60_CONTINUATION_FILES = {
    'docs/contracts/RTDETR_BASELINE_V2B_EPOCH60_CONTINUATION_T8A.md',
    'src/sparse_rtdetr/baseline/training_v2b_epoch60_continuation.py',
    'tests/test_training_v2b_epoch60_continuation.py',
    'tests/test_training_v2b_epoch60_worker.py',
    'tools/run_training_v2b_epoch60_continuation_campaign.py',
    'tools/run_training_v2b_epoch60_continuation_worker.py',
}

# REV1 is an explicit, read-only contract graph.  Its JSON files intentionally
# carry runtime locators, which are operational metadata and not source
# identity.  Keep the file set explicit; do not permit a repository-wide glob.
V2B_CONTRACT_REV1_FILES = {
    'contracts/v2b/rev001/external_authorities.json',
    'contracts/v2b/rev001/scientific_source.json',
    'contracts/v2b/rev001/evaluation_contract.json',
    'contracts/v2b/rev001/revision.json',
    'tools/seal_v2b_contract_revision.py',
    'tools/verify_v2b_contract_revision.py',
    'tools/evaluate_v2b_checkpoint.py',
    'tests/test_v2b_contract_revision.py',
    'contracts/v2b/rev001/training_contract.json',
    'contracts/v2b/rev001/execution_source.json',
    'contracts/v2b/rev001/lineage.json',
    'contracts/v2b/rev001/execution_contract.json',
    'contracts/v2b/rev001/campaign_authorization.json',
    'contracts/v2b/rev001/smoke_authorization.json',
    'contracts/v2b/rev001/gpu_smoke_authorization.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r2.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r3.json',
    'contracts/v2b/rev001/execution_source_r2.json',
    'contracts/v2b/rev001/execution_contract_r2.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r4.json',
    'contracts/v2b/rev001/execution_source_r3.json',
    'contracts/v2b/rev001/execution_contract_r3.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r5.json',
    'tools/verify_v2b_smoke_launcher_r3.py',
    'tools/bridge_v2b_checkpoint_revision.py',
    'tools/verify_v2b_checkpoint_revision.py',
    'tests/test_v2b_checkpoint_revision.py',
    'tools/v2b_training_execution_contracts.py',
    'tools/seal_v2b_training_execution.py',
    'tools/verify_v2b_training_execution.py',
    'tests/test_v2b_training_execution_contracts.py',
    'src/sparse_rtdetr/baseline/training_v2b_smoke_controller.py',
    'tools/verify_v2b_smoke_launcher.py',
    'tests/test_v2b_smoke_controller.py',
}

# REV1 runtime-locator execution is an explicit extension of the contract
# graph.  Keep these files enumerated: a repository-wide glob would make an
# unrelated runtime artifact silently part of the execution surface.
V2B_RUNTIME_LOCATOR_FILES = {
    'contracts/v2b/rev001/execution_source_r4.json',
    'contracts/v2b/rev001/execution_contract_r4.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r6.json',
    'contracts/v2b/rev001/execution_source_r5.json',
    'contracts/v2b/rev001/execution_contract_r5.json',
    'contracts/v2b/rev001/gpu_smoke_authorization_r7.json',
    'contracts/v2b/rev001/runtime_policy_authority_r1.json',
    'src/sparse_rtdetr/baseline/training_v2b_runtime_locator.py',
    'tests/test_v2b_runtime_locator.py',
    'tools/run_v2b_rev1_gpu_smoke.py',
    'tools/run_v2b_rev1_smoke_controller.py',
    'tools/verify_v2b_smoke_launcher_r4.py',
    'tools/verify_v2b_smoke_launcher_r5.py',
}

V2A_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json",
    "src/sparse_rtdetr/baseline/training_v2a_contract.py",
    "tests/test_rtdetr_baseline_training_v2a_contract.py",
    "docs/contracts/RTDETR_BASELINE_V2A_FORMAL_TRAINING_V1.md",
}
V2A_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json"
V2A_CONFIG_RAW_SIZE_BYTES = 12751
V2A_CONFIG_RAW_SHA256 = "40800725fa0409ac440f463d1df08ce647714aa7fa715a05f7e98878403ae407"
V2A_CONFIG_CANONICAL_SIZE_BYTES = 10198
V2A_CONFIG_CANONICAL_SHA256 = "96db5c69286775fb2b2b6a1a6997e58fe9d6019040eeca7328c02c7c2426ab0e"
V2A_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_contract.py"
V2A_MODULE_SHA256 = "5c5d5647f85100b301d9de533e92b1725a7e7b101d84cf657b9ab275cae35a96"
V2A_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_v2a_contract.py"
V2A_TEST_SHA256 = "5dba59cca8b917ab3eb6c615b42baab9016f7dc3e8a2b1320ae9cdb2bcf4ef95"
V2A_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_V2A_FORMAL_TRAINING_V1.md"
V2A_DOCUMENT_SHA256 = "3dae6d85a7180d30aaaf58993a6c71568589ba2cc34a9d7b32133cced6b2998d"
V2A_PUBLIC_NAMES = {
    "TrainingV2AContractError",
    "load_training_v2a_contract",
    "validate_training_v2a_contract",
    "canonical_training_v2a_contract_bytes",
    "training_v2a_contract_binding",
    "audit_optimizer_parameter_groups",
    "resolve_optimizer_parameter_group",
}
V2A_ALLOWED_IMPORTS = {
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

T7C_FILES = {
    "src/sparse_rtdetr/baseline/training_v2a_runtime.py",
    "tests/test_rtdetr_baseline_training_v2a_runtime.py",
    "docs/contracts/RTDETR_BASELINE_V2A_RUNTIME_INTEGRATION_T7C.md",
}
T7C_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_runtime.py"
T7C_MODULE_SHA256 = "31b3880c1aef8fa09dafaf430999df574d01549579a4c8a19fb8a8fb1527812d"
T7C_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_v2a_runtime.py"
T7C_TEST_SHA256 = "04dd063de8b2867d95f63f3effd8b8766d406f86f4d086a837d76062209e3602"
T7C_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_V2A_RUNTIME_INTEGRATION_T7C.md"
T7C_DOCUMENT_SHA256 = "656a3161f492485bf4462d35ce25cc72bd1912a15e8641881754d713ada91b09"
T7C_PUBLIC_NAMES = {
    "V2ARuntimeError",
    "build_v2a_runtime_policy",
    "bind_training_parent_identity",
    "bind_authorized_data_roots",
    "bind_environment_identity",
    "load_v2a_pretrained_backbone",
    "audit_v2a_optimizer_parameters",
    "build_v2a_optimizer",
    "build_v2a_runtime_components",
    "v2a_bf16_autocast_context",
    "select_v2a_development_candidate",
    "build_v2a_checkpoint",
    "validate_v2a_checkpoint",
    "EpochProgressWriter",
    "parse_v2a_terminal_stdout",
    "validate_v2a_runtime_closure",
    "run_v2a_fake_runtime",
}
T7C_ALLOWED_IMPORTS = {
    "contextlib",
    "copy",
    "hashlib",
    "importlib",
    "inspect",
    "json",
    "math",
    "os",
    "pathlib",
    "stat",
    "sys",
    "typing",
    "sparse_rtdetr.baseline",
}

T7D_FILES = {
    "src/sparse_rtdetr/baseline/training_v2a_production.py",
    "tests/test_rtdetr_baseline_training_v2a_production.py",
    "docs/contracts/RTDETR_BASELINE_V2A_PRODUCTION_BOUNDARY_T7D.md",
}
T7D_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_production.py"
T7D_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_v2a_production.py"
T7D_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_V2A_PRODUCTION_BOUNDARY_T7D.md"
T7D_MODULE_SHA256 = "c3bbc7738e2db12c6f47b93d8d87d80d6d0766b711033382b8ce28a6933edf10"
T7D_TEST_SHA256 = "3880b71f50f013003b8d85384b7e456f17b676c7cee52fced485e1aef6e447ab"
T7D_DOCUMENT_SHA256 = "4d487ae89dc382ba5b12f55d1980572a0f5b06d3010a1d4e9630fc48c3aa5d23"
T7D_PUBLIC_NAMES = {
    "V2AProductionError",
    "build_v2a_production_policy",
    "build_v2a_detached_launch_descriptor",
    "validate_v2a_detached_launch_descriptor",
    "validate_v2a_authorization_context",
    "current_v2a_source_identity",
    "parse_v2a_production_stdout",
    "classify_v2a_terminal",
    "validate_v2a_production_result",
    "run_v2a_production_entry",
    "run_v2a_cpu_fake",
}
T7D_ALLOWED_IMPORTS = {
    "contextlib",
    "copy",
    "hashlib",
    "importlib",
    "inspect",
    "json",
    "math",
    "os",
    "pathlib",
    "signal",
    "sys",
    "typing",
    "sparse_rtdetr.baseline",
}

T7E_FILES = {
    "src/sparse_rtdetr/baseline/training_v2a_launcher.py",
    "tests/test_rtdetr_baseline_training_v2a_launcher.py",
    "docs/contracts/RTDETR_BASELINE_V2A_EXACTLY_ONCE_LAUNCH_T7E.md",
}
T7E_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_launcher.py"
T7E_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_v2a_launcher.py"
T7E_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_V2A_EXACTLY_ONCE_LAUNCH_T7E.md"
T7E_MODULE_SHA256 = "060501d7ff19b255885c8141c5961ee669537f21d23f8cdc72d23d3f1c09a292"
T7E_TEST_SHA256 = "77c363a734992d92b47bc043c271bea5e37efb84325ed6ddecfc9c75a146ef23"
T7E_DOCUMENT_SHA256 = "761851975da8b1c662ae70500398aa37256c304a73a17ecaf446e5643e13bf35"
T7E_PUBLIC_NAMES = {
    "V2ALauncherError",
    "canonical_v2a_launcher_bytes",
    "build_v2a_owner_authorization",
    "validate_v2a_owner_authorization",
    "build_v2a_launch_plan",
    "validate_v2a_launch_plan",
    "consume_v2a_authorization",
    "validate_v2a_consumption_receipt",
    "validate_v2a_immediate_launch_result",
    "run_v2a_cpu_fake_launch",
    "run_v2a_production_launch",
}
T7E_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "json",
    "math",
    "os",
    "pathlib",
    "stat",
    "subprocess",
    "sys",
    "typing",
    "sparse_rtdetr.baseline",
}

T7H_FILES = {
    "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a_t7h_pilot.json",
    "src/sparse_rtdetr/baseline/training_v2a_pilot.py",
    "tests/test_rtdetr_baseline_training_v2a_pilot.py",
    "docs/contracts/RTDETR_BASELINE_V2A_LOSS_REPAIR_PILOT_T7H.md",
}
T7H_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a_t7h_pilot.json"
T7H_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_pilot.py"
T7H_TEST_RELATIVE_PATH = "tests/test_rtdetr_baseline_training_v2a_pilot.py"
T7H_DOCUMENT_RELATIVE_PATH = "docs/contracts/RTDETR_BASELINE_V2A_LOSS_REPAIR_PILOT_T7H.md"
T7H_CONFIG_RAW_SIZE_BYTES = 4890
T7H_CONFIG_RAW_SHA256 = "5135e7f90f971fd6bf700996a12a165fcc40a8d3f157a259fae783a19e758206"
T7H_CONFIG_CANONICAL_SIZE_BYTES = 4889
T7H_CONFIG_CANONICAL_SHA256 = "2c6f10db3248d523df65102dea21fa0595fe059db03a7289efdc3af68fb506ab"
T7H_MODULE_SHA256 = "3e35806b9b4d6baddae446ae89da512aa217eea0b638d063fbafdefbf791d9a7"
T7H_TEST_SHA256 = "23a40b2f7f2960460750f9fa3cd44db81e7020cc0013a843e01cfeb1335050ca"
T7H_DOCUMENT_SHA256 = "1ec2b599a78a731138605da56f5970127281d0427a96b0632cce8d96707765c9"
T7H_PUBLIC_NAMES = {
    "T7HPilotError",
    "canonical_t7h_pilot_bytes",
    "load_t7h_pilot_contract",
    "validate_t7h_pilot_contract",
    "build_t7h_pilot_policy",
    "validate_t7h_pilot_policy",
    "build_t7h_pilot_descriptor",
    "validate_t7h_pilot_descriptor",
    "validate_t7h_pilot_result",
    "run_t7h_cpu_fake",
}
T7H_ALLOWED_IMPORTS = {
    "copy",
    "hashlib",
    "importlib",
    "inspect",
    "json",
    "math",
    "os",
    "pathlib",
    "shutil",
    "stat",
    "tempfile",
    "typing",
    "sparse_rtdetr.baseline",
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
T6B_CONFIG_RAW_SIZE_BYTES = 7325
T6B_CONFIG_RAW_SHA256 = "87467339ec759abe89ab162c67395d0b34e50ffe1033adcd6dc9a4bc4413951d"
T6B_CONFIG_CANONICAL_SIZE_BYTES = 5930
T6B_CONFIG_CANONICAL_SHA256 = "ed71970ff3e14476763ca8c72d89bff5c58e3848ce76ad98cc209b62e66d312a"
T6B_SOURCE_IDENTITIES = {
    "src/sparse_rtdetr/baseline/training_t6_engine.py": "6bc7fa8c6a97004f0d698143af8189cf5db1a9bb6c0b4a631ec307532d7d37de",
    "src/sparse_rtdetr/baseline/training_t6_entry.py": "0ec5fb1990c53e444da2ba975f8722f9a329764b136e2264788cb46736dd5c57",
    "src/sparse_rtdetr/baseline/training_t6_process_launcher.py": "8d36f01acd05c89e9f8c4ee916484d7750eb21e0f6e5f3c5a12071e369ea6da3",
    "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py": "5139cd124b1bd8460a16ad02c98c9045594df408a015e8f3f26f8a9cd6079590",
}
T6B_SUPPORT_IDENTITIES = {
    "tests/test_rtdetr_baseline_training_t6b.py": "bea60a3b3959e6975cb334e0b1d2f0e81a5ca65caf56d5a635a54c56adb47c65",
    "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_T6B_PRODUCTION_BOUNDARY_V1.md": "4c888731104c918fa118fda1d458c3ca5838350c985e25f1f40284a40b758423",
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

# Explicitly allow only the new CPU engineering implementation and tests.
BASELINE_MODEL_IMPORT_FILES |= {
    path for path in V2B_ENGINEERING_FILES | V2B_RUNTIME_FILES | V2B_SAMPLING_FILES
    if path.endswith(".py")
}

BASELINE_MODEL_IMPORT_FILES |= {
    "src/sparse_rtdetr/baseline/training_v2b_campaign.py",
    "src/sparse_rtdetr/baseline/training_v2b_control.py",
    "src/sparse_rtdetr/baseline/training_v2b_development.py",
    "tests/test_rtdetr_baseline_training_v2b_control.py",
    "tests/test_rtdetr_baseline_training_v2b_development.py",
}

# Explicit 896 experiment modules use the same native admitted runtime.
BASELINE_MODEL_IMPORT_FILES |= {path for path in V2B_896_MATCHED_FILES if path.endswith(".py")}

# Only the inference worker and its CPU test need additional model imports.
BASELINE_MODEL_IMPORT_FILES |= {
    'src/sparse_rtdetr/baseline/training_v2b_cross_eval.py',
    'tests/test_rtdetr_baseline_training_v2b_cross_eval.py',
    'tests/test_training_v2b_resolution_diagnostics.py',
}

# Only the paired-seed runtime and actual CPU model fixtures import torch.
BASELINE_MODEL_IMPORT_FILES |= {
    'src/sparse_rtdetr/baseline/training_v2b_replication_campaign.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_capacity.py',
    'src/sparse_rtdetr/baseline/training_v2b_replication_worker.py',
    'tests/test_training_v2b_replication_capacity.py',
    'tests/test_training_v2b_replication_gate.py',
    'tests/test_training_v2b_replication_worker.py',
    'tests/test_training_v2b_replication_summary.py',
}

# The completion summary performs a bounded CPU checkpoint read. Its tests and
# CLI import that same explicitly reviewed bridge; no other new file receives
# a model-import exemption.
BASELINE_MODEL_IMPORT_FILES |= {
    'src/sparse_rtdetr/baseline/training_v2b_seed2_completion_summary.py',
    'tests/test_training_v2b_seed2_completion.py',
    'tests/test_training_v2b_seed2_completion_summary.py',
    'tools/summarize_training_v2b_seed2_completion.py',
}

# The epoch60 worker imports torch only after authenticating its contract and
# selecting the historical source checkout.  The controller and contract
# validator remain model-import free.
BASELINE_MODEL_IMPORT_FILES |= {
    'tools/run_training_v2b_epoch60_continuation_worker.py',
    'tools/evaluate_v2b_checkpoint.py',
}

# The explicit REV1 checkpoint bridge and its tests authenticate a real
# torch payload; they are bounded by the bridge verifier and are not generic
# model-import entry points.
BASELINE_MODEL_IMPORT_FILES |= {
    'tools/bridge_v2b_checkpoint_revision.py',
    'tools/verify_v2b_checkpoint_revision.py',
    'tests/test_v2b_checkpoint_revision.py',
}

# The REV1 smoke runner is a deliberately bounded, identity-gated torch
# entry point.  Its import is allowed explicitly; the controller and verifier
# remain covered by the ordinary source policy.
BASELINE_MODEL_IMPORT_FILES |= {
    'tools/run_v2b_rev1_gpu_smoke.py',
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


def _relative_files(root: Path, *, source_only: bool = False) -> set[str]:
    if source_only:
        # A source audit must not enter runtime evidence or resolve its links.
        # Retain source directory symlinks in the inventory so they still fail.
        result = set()
        for directory, subdirectories, filenames in os.walk(root, followlinks=False):
            parent = Path(directory)
            subdirectories[:] = [name for name in subdirectories
                                 if name != ".git" and not (parent == root and name == "artifacts")]
            for name in subdirectories + filenames:
                if name == ".git" or (parent == root and name == "artifacts"):
                    continue
                path = parent / name
                if path.is_symlink() or path.is_file():
                    result.add(path.relative_to(root).as_posix())
        return result
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


def _file_policy(root: Path, *, source_only: bool = False) -> tuple[set[str], list[str]]:
    """Audit source; historical mode additionally inventories runtime paths."""

    failures: list[str] = []
    all_files = _relative_files(root, source_only=source_only)
    tracked_files = _git_tracked_files(root)
    if source_only:
        # Git's index names suffice to reject tracked data without opening it.
        failures.extend(f"tracked runtime artifact: {relative}" for relative in sorted(tracked_files)
                        if relative == "artifacts" or relative.startswith("artifacts/"))
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
    if artifacts_root.is_symlink() or (artifacts_root.exists() and not artifacts_root.is_dir()):
        failures.append("artifacts must be a real directory")
    for relative in all_files:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
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
        if relative.startswith("contracts/v2b/rev001/"):
            # Runtime locators are deliberately present in these contracts;
            # their canonical identity is checked by the REV1 verifier.
            continue
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


def _vendor_inventory(
    root: Path,
    *,
    declared: list[dict[str, object]] | None = None,
    failures: list[str] | None = None,
) -> list[dict[str, object]]:
    """Read vendor identity rows without treating generated cache as source.

    The manifest is the immutable source of truth. When it is available, only
    declared files are hashed, while an additional scan rejects tracked or
    non-ignored unknown files. Ignored generated Python cache is deliberately
    outside the identity boundary. declared=None retains the raw-scan behavior
    for generic callers.
    """

    vendor_root = root / "vendor" / "rtdetrv2_pytorch"
    if declared is None:
        paths = [
            path
            for path in sorted(vendor_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
    else:
        declared_paths = {
            item.get("relative_path")
            for item in declared
            if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
        }
        paths = []
        for relative in sorted(declared_paths):
            if not relative.startswith(VENDOR_PREFIX):
                continue
            path = root / relative
            if path.is_symlink():
                if failures is not None:
                    failures.append(f"vendor symlink: {relative}")
                continue
            if not path.is_file():
                if failures is not None:
                    failures.append(f"missing vendor file: {relative}")
                continue
            paths.append(path)

        if failures is not None:
            tracked_files = _git_tracked_files(root)
            ignore_patterns = _gitignore_patterns(root)
            for path in sorted(vendor_root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                if relative in declared_paths:
                    continue
                if relative in tracked_files or not _gitignored(relative, ignore_patterns):
                    failures.append(f"unexpected vendor file: {relative}")

    inventory = []
    for path in paths:
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

    declared = manifest.get("files")
    actual = _vendor_inventory(
        root,
        declared=declared if isinstance(declared, list) else None,
        failures=failures,
    )
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
            if invocation != {"outer_calls": 1, "tmux_new_session_calls": 1, "child_calls": 1, "shell": False, "argv_sequence": True, "network": False, "speed_measurement": False, "entry_module": "sparse_rtdetr.baseline.training_t6_entry", "process_module": "sparse_rtdetr.baseline.training_t6_process_launcher", "entry_descriptor_filename": "entry_descriptor.json", "process_descriptor_filename": "process_descriptor.json", "descriptor_persistence_order": ["process_descriptor", "outer_invocation", "tmux", "entry_descriptor", "process_invocation", "child"]}:
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
                claim_validators = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_validate_engine_claim"
                ]
                if len(claim_validators) != 1:
                    failures.append("T6B engine claim validator definition count mismatch")
                else:
                    validator = claim_validators[0]

                    def field_ref(node: ast.AST, base: str) -> str | None:
                        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name) or node.value.id != base:
                            return None
                        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str):
                            return None
                        return node.slice.value

                    checked_context_assignments = [
                        node
                        for node in ast.walk(validator)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "checked_context" for target in node.targets)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_validate_execution_context"
                        and len(node.value.args) == 1
                        and field_ref(node.value.args[0], "claim") == "context"
                    ]
                    if len(checked_context_assignments) != 1:
                        failures.append("T6B claim validator does not bind checked execution context")

                    bound_fields: set[str] = set()
                    for node in ast.walk(validator):
                        if not isinstance(node, ast.Compare):
                            continue
                        operands = [node.left, *node.comparators]
                        for left, right in zip(operands, operands[1:]):
                            left_field = field_ref(left, "claim")
                            right_field = field_ref(right, "checked_context")
                            if left_field in {"descriptor_sha256", "training_run_id", "nonce", "evidence_root"} and right_field == left_field:
                                bound_fields.add(left_field)
                            if right_field in {"descriptor_sha256", "training_run_id", "nonce", "evidence_root"} and left_field == right_field:
                                bound_fields.add(right_field)
                    if bound_fields != {"descriptor_sha256", "training_run_id", "nonce", "evidence_root"}:
                        failures.append("T6B claim validator missing duplicated identity cross-binding")
                    digest_checks = [
                        node
                        for node in ast.walk(validator)
                        if isinstance(node, ast.Compare)
                        and any(field_ref(operand, "claim") == "context_sha256" for operand in [node.left, *node.comparators])
                        and any(
                            isinstance(operand, ast.Call)
                            and isinstance(operand.func, ast.Name)
                            and operand.func.id == "_digest"
                            and len(operand.args) == 1
                            and isinstance(operand.args[0], ast.Name)
                            and operand.args[0].id == "checked_context"
                            for operand in [node.left, *node.comparators]
                        )
                    ]
                    if len(digest_checks) != 1:
                        failures.append("T6B claim validator missing checked-context digest binding")
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

def _check_training_v2a_contract(root: Path, failures: list[str]) -> None:
    """Check the detached, pre-runtime baseline-v2a contract surface."""

    expected_hashes = {
        V2A_MODULE_RELATIVE_PATH: V2A_MODULE_SHA256,
        V2A_TEST_RELATIVE_PATH: V2A_TEST_SHA256,
        V2A_DOCUMENT_RELATIVE_PATH: V2A_DOCUMENT_SHA256,
    }
    for relative in sorted(V2A_FILES):
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"v2a file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"v2a file nlink drift: {relative}")
            expected = expected_hashes.get(relative)
            if expected and not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"v2a file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing v2a file: {relative}")

    config_path = root / V2A_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != V2A_CONFIG_RAW_SIZE_BYTES or _sha256(config_path) != V2A_CONFIG_RAW_SHA256:
            failures.append("v2a config raw identity mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("v2a config portable bytes mismatch")

        def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result

        value = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)),
        )
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != V2A_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != V2A_CONFIG_CANONICAL_SHA256:
            failures.append("v2a config canonical identity mismatch")
        if type(value) is not dict:
            failures.append("v2a config root is not an object")
        else:
            if value.get("schema_version") != 1 or value.get("contract_id") != "rtdetrv2_r18_visdrone_baseline_training_v2a" or value.get("baseline_id") != "rtdetrv2_r18_visdrone_baseline_v2a":
                failures.append("v2a config identity mismatch")
            if value.get("stage") != "T7B_BASELINE_V2A_FROZEN_CONTRACT":
                failures.append("v2a config stage mismatch")
            if value.get("model", {}).get("input_size") != [640, 640] or value.get("topology", {}).get("train_micro_batch") != 16 or value.get("topology", {}).get("development_batch") != 32:
                failures.append("v2a B1 input/batch binding mismatch")
            authority = value.get("source_bindings", {}).get("pretrained_authority", {})
            weight = authority.get("weight", {})
            manifest = authority.get("manifest", {})
            if weight.get("size_bytes") != 44878642 or weight.get("sha256") != "911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720":
                failures.append("v2a pretrained weight binding mismatch")
            if manifest.get("size_bytes") != 1608 or manifest.get("sha256") != "f290281d7c9f1589e2f6da3aca2251e351df48dbad82317ee59e4381ecb01dd0":
                failures.append("v2a pretrained manifest binding mismatch")
            amp = value.get("amp", {})
            if amp.get("autocast_dtype") != "bfloat16" or amp.get("grad_scaler_enabled") is not False:
                failures.append("v2a AMP/scaler policy mismatch")
            if any(key in amp for key in ("init_scale", "growth_factor", "backoff_factor", "growth_interval", "scaler_type")):
                failures.append("v2a config retains forbidden scaler parameters")
            groups = value.get("optimizer", {}).get("parameter_groups", [])
            if [group.get("name") for group in groups] != ["norm_or_bn", "default"]:
                failures.append("v2a optimizer group order mismatch")
            if value.get("source_bindings", {}).get("data_roles", {}).get("confirmatory", {}).get("status") != "sealed_and_forbidden" or value.get("source_bindings", {}).get("data_roles", {}).get("test", {}).get("status") != "forbidden":
                failures.append("v2a sealed/test data policy mismatch")
            if any(
                type(requirement) is not dict
                or requirement.get("required") is not True
                or requirement.get("implemented") is not False
                or requirement.get("independently_certified") is not False
                for requirement in value.get("runtime_closure", {}).values()
            ) or len(value.get("runtime_closure", {})) != 5:
                failures.append("v2a runtime closure state mismatch")
            readiness = value.get("readiness", {})
            if readiness.get("training_implementation_ready") is not False or readiness.get("training_ready") is not False:
                failures.append("v2a readiness is not fail-closed")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"v2a config parse failure: {type(exc).__name__}")

    parent_path = root / "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json"
    try:
        parent_raw = parent_path.read_bytes()
        if len(parent_raw) != 10907 or _sha256(parent_path) != "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297":
            failures.append("v2a parent v1 raw identity mismatch")
    except OSError:
        failures.append("v2a parent v1 config unavailable")

    module_path = root / V2A_MODULE_RELATIVE_PATH
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != V2A_ALLOWED_IMPORTS:
            failures.append(
                "v2a import set mismatch: "
                f"extra={sorted(imported - V2A_ALLOWED_IMPORTS)} "
                f"missing={sorted(V2A_ALLOWED_IMPORTS - imported)}"
            )
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public != V2A_PUBLIC_NAMES:
            failures.append(
                "v2a public API mismatch: "
                f"extra={sorted(public - V2A_PUBLIC_NAMES)} missing={sorted(V2A_PUBLIC_NAMES - public)}"
            )
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != V2A_PUBLIC_NAMES:
            failures.append("v2a __all__ mismatch")
        forbidden_roots = {"torch", "numpy", "pandas", "tensorflow", "cupy", "dataloader", "dataset", "subprocess", "requests"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                failures.append("v2a forbidden import")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                failures.append("v2a forbidden import")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"v2a source audit failure: {type(exc).__name__}")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("p3_training_v2a_contract_check", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("v2a module spec unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.V2A_CONFIG_RAW_SIZE_BYTES != V2A_CONFIG_RAW_SIZE_BYTES or module.V2A_CONFIG_CANONICAL_SHA256 != V2A_CONFIG_CANONICAL_SHA256:
            failures.append("v2a module/config identity constants mismatch")
        validated = module.validate_training_v2a_contract(value)
        if validated != value:
            failures.append("v2a detached validation is not identity-preserving")
    except Exception as exc:
        failures.append(f"v2a detached validation failure: {type(exc).__name__}: {exc}")


def _check_t7c_runtime(root: Path, failures: list[str]) -> None:
    """Check the independent CPU/fake v2a runtime surface."""

    expected_hashes = {
        T7C_MODULE_RELATIVE_PATH: T7C_MODULE_SHA256,
        T7C_TEST_RELATIVE_PATH: T7C_TEST_SHA256,
        T7C_DOCUMENT_RELATIVE_PATH: T7C_DOCUMENT_SHA256,
    }
    for relative in sorted(T7C_FILES):
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"t7c file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"t7c file nlink drift: {relative}")
            expected = expected_hashes[relative]
            if not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"t7c file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing t7c file: {relative}")

    module_path = root / T7C_MODULE_RELATIVE_PATH
    try:
        source_text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != T7C_ALLOWED_IMPORTS:
            failures.append(
                "t7c import set mismatch: "
                f"extra={sorted(imported - T7C_ALLOWED_IMPORTS)} "
                f"missing={sorted(T7C_ALLOWED_IMPORTS - imported)}"
            )
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public != T7C_PUBLIC_NAMES:
            failures.append(
                "t7c public API mismatch: "
                f"extra={sorted(public - T7C_PUBLIC_NAMES)} missing={sorted(T7C_PUBLIC_NAMES - public)}"
            )
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != T7C_PUBLIC_NAMES:
            failures.append("t7c __all__ mismatch")
        forbidden_roots = {"torch", "numpy", "pandas", "tensorflow", "cupy", "dataloader", "dataset", "subprocess", "requests"}
        if "GradScaler" in source_text or "training_t6_engine" in source_text:
            failures.append("t7c source retains forbidden scaler or T6 engine surface")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                failures.append("t7c forbidden import")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                failures.append("t7c forbidden import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                failures.append("t7c dynamic execution")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Assign) and any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                failures.append("t7c import-time call")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"t7c source audit failure: {type(exc).__name__}")


def _check_t7d_production_boundary(root: Path, failures: list[str]) -> None:
    """Check the detached T7D boundary without importing production code."""

    expected_hashes = {
        T7D_MODULE_RELATIVE_PATH: T7D_MODULE_SHA256,
        T7D_TEST_RELATIVE_PATH: T7D_TEST_SHA256,
        T7D_DOCUMENT_RELATIVE_PATH: T7D_DOCUMENT_SHA256,
    }
    for relative in sorted(T7D_FILES):
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"t7d file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"t7d file nlink drift: {relative}")
            expected = expected_hashes[relative]
            if not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"t7d file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing t7d file: {relative}")

    module_path = root / T7D_MODULE_RELATIVE_PATH
    try:
        source_text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != T7D_ALLOWED_IMPORTS:
            failures.append(
                "t7d import set mismatch: "
                f"extra={sorted(imported - T7D_ALLOWED_IMPORTS)} "
                f"missing={sorted(T7D_ALLOWED_IMPORTS - imported)}"
            )
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public != T7D_PUBLIC_NAMES:
            failures.append(
                "t7d public API mismatch: "
                f"extra={sorted(public - T7D_PUBLIC_NAMES)} missing={sorted(T7D_PUBLIC_NAMES - public)}"
            )
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != T7D_PUBLIC_NAMES:
            failures.append("t7d __all__ mismatch")

        forbidden_roots = {"torch", "numpy", "pandas", "tensorflow", "cupy", "subprocess", "requests"}
        if "GradScaler" in source_text or "training_t6_engine" in source_text or "training_t6_entry" in source_text:
            failures.append("t7d source retains forbidden scaler or legacy production surface")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                failures.append("t7d forbidden import")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                failures.append("t7d forbidden import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                failures.append("t7d dynamic execution")
        production_entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_v2a_production_entry"]
        if len(production_entries) != 1:
            failures.append("t7d production entry definition count mismatch")
        else:
            arguments = production_entries[0].args.args + production_entries[0].args.kwonlyargs
            if [argument.arg for argument in arguments] != ["descriptor", "authorization_context"]:
                failures.append("t7d production entry exposes injected ports")
        for statement in tree.body:
            if isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Name) and statement.test.left.id == "__name__":
                continue
            if not isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                failures.append("t7d import-time call")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"t7d source audit failure: {type(exc).__name__}")

    for relative in (T7D_TEST_RELATIVE_PATH, T7D_DOCUMENT_RELATIVE_PATH):
        path = root / relative
        try:
            if path.stat().st_size <= 0:
                failures.append(f"empty T7D support file: {relative}")
        except OSError:
            failures.append(f"missing T7D support file: {relative}")


def _check_t7e_exactly_once_launcher(root: Path, failures: list[str]) -> None:
    """Check the T7E schema and keep real subprocess calls behind private owners."""

    expected_hashes = {
        T7E_MODULE_RELATIVE_PATH: T7E_MODULE_SHA256,
        T7E_TEST_RELATIVE_PATH: T7E_TEST_SHA256,
        T7E_DOCUMENT_RELATIVE_PATH: T7E_DOCUMENT_SHA256,
    }
    for relative in sorted(T7E_FILES):
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"t7e file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"t7e file nlink drift: {relative}")
            expected = expected_hashes[relative]
            if not expected.startswith("PENDING_") and _sha256(path) != expected:
                failures.append(f"t7e file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing t7e file: {relative}")

    module_path = root / T7E_MODULE_RELATIVE_PATH
    try:
        source_text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != T7E_ALLOWED_IMPORTS:
            failures.append(
                "t7e import set mismatch: "
                f"extra={sorted(imported - T7E_ALLOWED_IMPORTS)} "
                f"missing={sorted(T7E_ALLOWED_IMPORTS - imported)}"
            )
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public != T7E_PUBLIC_NAMES:
            failures.append(
                "t7e public API mismatch: "
                f"extra={sorted(public - T7E_PUBLIC_NAMES)} missing={sorted(T7E_PUBLIC_NAMES - public)}"
            )
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != T7E_PUBLIC_NAMES:
            failures.append("t7e __all__ mismatch")
        forbidden_roots = {"torch", "numpy", "pandas", "tensorflow", "cupy", "requests", "training_t6", "training_v1"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                failures.append("t7e forbidden import")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                failures.append("t7e forbidden import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                failures.append("t7e dynamic execution")
        if "O_EXCL" not in source_text or "os.fsync" not in source_text or "shell=False" not in source_text:
            failures.append("t7e durability or shell boundary is incomplete")
        if "TMUX_ACCEPTED" not in source_text or "training_ready" not in source_text:
            failures.append("t7e readiness boundary is incomplete")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Name) and statement.test.left.id == "__name__":
                continue
            if any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                failures.append("t7e import-time call")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"t7e source audit failure: {type(exc).__name__}")

    for relative in (T7E_TEST_RELATIVE_PATH, T7E_DOCUMENT_RELATIVE_PATH):
        path = root / relative
        try:
            if path.stat().st_size <= 0:
                failures.append(f"empty T7E support file: {relative}")
        except OSError:
            failures.append(f"missing T7E file: {relative}")


def _check_t7h_loss_repair_pilot(root: Path, failures: list[str]) -> None:
    """Check the detached T7H contract and inert CPU/fake adapter surface."""

    expected_hashes = {
        T7H_MODULE_RELATIVE_PATH: T7H_MODULE_SHA256,
        T7H_TEST_RELATIVE_PATH: T7H_TEST_SHA256,
        T7H_DOCUMENT_RELATIVE_PATH: T7H_DOCUMENT_SHA256,
    }
    for relative, expected in expected_hashes.items():
        path = root / relative
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file() or not stat.S_ISREG(observed.st_mode):
                failures.append(f"t7h file is not a regular non-symlink file: {relative}")
                continue
            if observed.st_nlink != 1:
                failures.append(f"t7h file nlink drift: {relative}")
            if expected.startswith("PENDING_") or _sha256(path) != expected:
                failures.append(f"t7h file identity mismatch: {relative}")
        except OSError:
            failures.append(f"missing t7h file: {relative}")

    config_path = root / T7H_CONFIG_RELATIVE_PATH
    try:
        raw = config_path.read_bytes()
        if len(raw) != T7H_CONFIG_RAW_SIZE_BYTES or _sha256(config_path) != T7H_CONFIG_RAW_SHA256:
            failures.append("T7H config raw identity mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
            failures.append("T7H config portable bytes mismatch")
        value = json.loads(raw[:-1].decode("utf-8"))
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(canonical) != T7H_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != T7H_CONFIG_CANONICAL_SHA256:
            failures.append("T7H config canonical identity mismatch")
        if value.get("schema_version") != 1 or value.get("contract_id") != "rtdetrv2_r18_visdrone_baseline_v2a_t7h_loss_repair_pilot_r1" or value.get("stage") != "T7H_LOSS_REPAIR_DIAGNOSTIC_PILOT":
            failures.append("T7H config semantic identity mismatch")
        overlay = value.get("overlay")
        if not isinstance(overlay, dict) or overlay.get("executed_epochs") != 10 or overlay.get("required_complete_epochs") != 10 or overlay.get("checkpoint_epochs") != list(range(1, 11)) or overlay.get("development_evaluation_epochs") != list(range(1, 11)) or overlay.get("augmentation_stop_epoch") != 117 or overlay.get("terminal_classification") != "DIAGNOSTIC_ONLY":
            failures.append("T7H ten-epoch overlay mismatch")
        t7g = value.get("t7g_binding")
        if not isinstance(t7g, dict) or t7g.get("head_sha") != "476e224cadebf8787a10c38dd9a0823808b56d48" or t7g.get("tree_sha") != "cf85724186e9c7180f2f6f00cdc400d31c145ee9" or t7g.get("weighted_loss_source_sha256") != "c50be6f2ed177c8c0a97a1d1bde8cb2675ac5a60d293abf3607b37fb4d8d0bf2" or t7g.get("independent_audit_pass") is not True:
            failures.append("T7H T7G binding mismatch")
        readiness = value.get("readiness")
        if not isinstance(readiness, dict) or readiness.get("contract_layer_implemented") is not True or any(readiness.get(field) is not False for field in ("cpu_fake_verified", "archive_verified", "published", "independent_audit_pass", "owner_authorization_created", "production_launch_executed", "diagnostic_complete", "training_ready", "model_selection_certified", "test_access_ready", "confirmatory_metrics_accessed", "dataset_test_split_accessed_by_this_stage")):
            failures.append("T7H readiness is not fail-closed")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        failures.append("T7H config parse failure")

    module_path = root / T7H_MODULE_RELATIVE_PATH
    try:
        source_text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(module_path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add(node.module or "")
        if imported != T7H_ALLOWED_IMPORTS:
            failures.append(f"t7h import set mismatch: extra={sorted(imported - T7H_ALLOWED_IMPORTS)} missing={sorted(T7H_ALLOWED_IMPORTS - imported)}")
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        if public != T7H_PUBLIC_NAMES:
            failures.append(f"t7h public API mismatch: extra={sorted(public - T7H_PUBLIC_NAMES)} missing={sorted(T7H_PUBLIC_NAMES - public)}")
        all_values: object = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                all_values = ast.literal_eval(node.value)
        if type(all_values) not in {list, tuple} or len(all_values) != len(set(all_values)) or set(all_values) != T7H_PUBLIC_NAMES:
            failures.append("t7h __all__ mismatch")
        forbidden_roots = {"cupy", "numpy", "pandas", "requests", "socket", "subprocess", "tensorflow", "torch", "urllib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0].casefold() in forbidden_roots for alias in node.names):
                failures.append("t7h forbidden import")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0].casefold() in forbidden_roots:
                failures.append("t7h forbidden import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                failures.append("t7h dynamic execution")
            if isinstance(node, ast.FunctionDef) and node.name == "_weighted_loss":
                failures.append("t7h duplicates weighted-loss implementation")
        if "GradScaler" in source_text:
            failures.append("t7h source retains forbidden scaler surface")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Name) and statement.test.left.id == "__name__":
                continue
            if any(isinstance(node, ast.Call) for node in ast.walk(statement)):
                failures.append("t7h import-time call")
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError):
        failures.append("t7h source audit failure")

    for relative in (T7H_TEST_RELATIVE_PATH, T7H_DOCUMENT_RELATIVE_PATH):
        path = root / relative
        try:
            if path.stat().st_size <= 0:
                failures.append(f"empty T7H support file: {relative}")
        except OSError:
            failures.append(f"missing T7H file: {relative}")


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


def check_repository(root: Path, *, source_only: bool = False) -> bool:
    """Validate frozen source; default also requires the historical runtime artifacts.

    source_only is an explicit engineering scope, never a production admission.
    """
    failures: list[str] = []
    files, file_policy_failures = _file_policy(root, source_only=source_only)
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
    allowed_files |= V2A_FILES
    if files & V2B_ENGINEERING_FILES:
        allowed_files |= V2B_ENGINEERING_FILES
    if files & V2B_SAMPLING_FILES:
        allowed_files |= V2B_SAMPLING_FILES
    if files & V2B_RUNTIME_FILES:
        allowed_files |= V2B_RUNTIME_FILES
    if files & V2B_PAIRED_CONTROL_FILES:
        allowed_files |= V2B_PAIRED_CONTROL_FILES
    if files & V2B_896_MATCHED_FILES:
        allowed_files |= V2B_896_MATCHED_FILES
    if files & V2B_RESOLUTION_EVIDENCE_FILES:
        allowed_files |= V2B_RESOLUTION_EVIDENCE_FILES
    if files & V2B_PAIRED_SEED_FILES:
        allowed_files |= V2B_PAIRED_SEED_FILES
    if files & V2B_SEED2_COMPLETION_FILES:
        allowed_files |= V2B_SEED2_COMPLETION_FILES
    if files & V2B_EPOCH60_CONTINUATION_FILES:
        allowed_files |= V2B_EPOCH60_CONTINUATION_FILES
    if files & V2B_CONTRACT_REV1_FILES:
        allowed_files |= V2B_CONTRACT_REV1_FILES
    if files & V2B_RUNTIME_LOCATOR_FILES:
        allowed_files |= V2B_RUNTIME_LOCATOR_FILES
    allowed_files |= T7C_FILES
    t7d_files_present = files & T7D_FILES
    if t7d_files_present:
        allowed_files |= T7D_FILES
    t7e_files_present = files & T7E_FILES
    if t7e_files_present:
        allowed_files |= T7E_FILES
    t7h_files_present = files & T7H_FILES
    if t7h_files_present:
        allowed_files |= T7H_FILES
    if files != allowed_files:
        failures.append(f"file set mismatch: extra={sorted(files - allowed_files)} missing={sorted(allowed_files - files)}")

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
        if source_only:
            config = module.load_training_contract(root)
            digest = hashlib.sha256(module.canonical_training_contract_bytes(config)).hexdigest()
        else:
            digest = module.training_contract_binding(root)["canonical_sha256"]
        if digest != module.TRAINING_CONTRACT_CANONICAL_SHA256:
            failures.append("formal training contract canonical identity mismatch")
    except Exception as exc:
        failures.append(f"formal training contract validation failure: {type(exc).__name__}: {exc}")

    _check_training_runtime_plan(root, failures)
    _check_training_evidence(root, failures)
    _check_prepared_adapter(root, failures)
    _check_training_process(root, failures)
    _check_t6a_owner_authorization(root, failures)
    _check_t6b_production_boundary(root, failures)
    _check_training_v2a_contract(root, failures)
    _check_t7c_runtime(root, failures)
    if t7d_files_present:
        _check_t7d_production_boundary(root, failures)
    if t7e_files_present:
        _check_t7e_exactly_once_launcher(root, failures)
    if t7h_files_present:
        _check_t7h_loss_repair_pilot(root, failures)

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
    if source_only:
        print("repository_contract_check: SOURCE_ONLY_PASS; runtime_artifacts_checked=false")
    else:
        print("repository_contract_check: PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true",
                        help="check source without historical runtime/data artifacts")
    args = parser.parse_args()
    return 0 if check_repository(Path(__file__).resolve().parents[1],
                                source_only=args.source_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
