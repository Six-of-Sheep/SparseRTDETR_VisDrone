"""Explicit engineering regression scope and isolated legacy fake-port fixtures.

The fifteen CPU fake tests retain their frozen source bytes. Their ROOT is
redirected to a temporary source checkout with clearly synthetic annotations,
so they still exercise canonical paths, regular files, identity, and fake ports.
No production binder is patched and no historical success target is fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_FAKE_CASES = {
    "tests/test_rtdetr_baseline_training_v2a_runtime.py": {
        "test_policy_reads_binding_and_binds_all_input_closures",
        "test_checkpoint_records_contract_authority_groups_and_no_scaler",
        "test_fake_runtime_closes_process_parent_data_environment_and_progress",
        "test_fake_runtime_rejects_secondary_evaluator_and_batch_adaptation",
    },
    "tests/test_rtdetr_baseline_training_v2a_production.py": {
        "test_policy_binds_t7c_and_uses_independent_target_names",
        "test_cpu_fake_call_order_step_ema_primary_and_terminal",
        "test_tensor_loss_is_recorded_as_epoch_mean",
        "test_cpu_fake_duplicate_and_production_port_injection_are_rejected",
        "test_detached_descriptor_context_digest_and_cross_field_mutations",
        "test_production_entry_rejects_unconsumed_artifact_drift_and_existing_target",
        "test_bf16_context_has_no_scaler_construction_or_source_surface",
    },
    "tests/test_rtdetr_baseline_training_v2a_pilot.py": {
        "test_cpu_fake_records_ten_epochs_loss_families_and_encoder_state",
        "test_cpu_fake_delegates_corrected_weighted_loss_and_rejects_injection",
        "test_cpu_fake_rejects_replay_and_missing_encoder_evidence",
        "test_fake_result_is_canonical_and_round_trips",
    },
}
ADDITIONAL_FAKE_CASES = {
    "tests/test_rtdetr_baseline_training_v2a_launcher.py": {
        "test_closed_layers_bind_t7d_authority_and_fixed_argv",
        "test_cpu_fake_exactly_once_call_graph_and_terminal_evidence",
        "test_duplicate_consumption_is_rejected_without_second_fake_call",
        "test_preflight_failure_has_zero_writes",
        "test_post_consumption_launcher_failure_publishes_permanent_result",
        "test_authorization_mutations_fail_closed",
        "test_plan_mutations_fail_closed",
        "test_public_fake_ports_cannot_inject_production_capabilities",
        "test_production_authorization_identity_matches_receipt_schema",
    },
    "tests/test_rtdetr_baseline_training_t6b.py": {
        "test_runtime_data_role_binding_uses_exact_manifest_identity",
    },
    "tests/test_rtdetr_baseline_smoke.py": {
        "test_real_child_argv_roundtrip_dispatches_from_v3_receipt",
    },
}
HISTORICAL_RUNTIME_CASES = {
    "tests/test_repository_contract.py::RepositoryContractTests::test_contract_passes":
        "default repository admission requires complete historical runtime artifacts",
    "tests/test_rtdetr_baseline_adapter.py::test_repository_contract_checker_passes_after_allowlist_update":
        "default repository admission requires complete historical runtime artifacts",
    "tests/test_rtdetr_baseline_training_v2a_contract.py::test_repository_checker_accepts_checkout_with_completed_v1_targets":
        "test explicitly requires completed historical v1 training targets",
}

# This frozen test explicitly implements a source-only refusal branch. Select
# that branch only in engineering scope with an isolated source tree; leave
# default historical runs to select their branch from the actual checkout.
SOURCE_ONLY_CASES = {
    "tests/test_rtdetr_baseline_training_runtime.py": {
        "test_training_runtime_plan_binding_returns_complete_detached_observed_identity",
    },
}


# Broader repository regression also contains frozen success/mutation tests
# whose baseline is the complete 25-file historical R3 conversion, including
# sealed partitions. These are distinct from the original nineteen cases.
DATA_CLOSURE_CASES = {
    "tests/test_rtdetr_baseline_training_contract.py": (
        "test_positive_load_validate_canonical_and_binding",
        "test_primary_evaluator_runtime_binding_is_detached_and_does_not_read_external_provenance",
        "test_vendor_runtime_binding_returns_full_observed_identity_on_portable_fixture",
        "test_conversion_r3_completion_missing_is_rejected",
        "test_every_conversion_r3_file_byte_mutation_is_bound",
        "test_conversion_r3_real_structure_mutations_are_rejected",
        "test_conversion_r3_sandbox_safe_object_mutations_are_rejected",
        "test_conversion_r3_authority_mutations_are_rejected",
        "test_conversion_r3_enumeration_drift_is_rejected",
        "test_conversion_r3_eintr_is_recovered",
        "test_conversion_r3_fd_leak_success_and_failure",
    ),
    "tests/test_rtdetr_baseline_training_evidence.py": (
        "test_contract_binding_is_detached_and_does_not_create_files_or_change_environment",
        "test_contract_authority_binding_and_detachment",
        "test_synthetic_writer_orders_records_checkpoints_and_terminalizes_once",
        "test_writer_rejects_duplicate_and_out_of_order_epochs",
        "test_writer_rejects_nonfinite_and_forbidden_epoch_observations",
        "test_writer_rejects_evaluator_checkpoint_and_identity_input_drift",
        "test_classifier_absent_in_progress_and_unknown_are_conservative",
        "test_permanent_failure_is_immutable_and_terminal_failed",
        "test_real_terminal_success_requires_120_epochs",
        "test_validator_rejects_repacked_records_and_terminal_files",
        "test_validator_rejects_unstable_prepared_objects",
        "test_validator_rejects_socket_extra_file_and_root_symlink",
        "test_readonly_validation_does_not_change_evidence_identity",
        "test_atomic_checkpoint_publishes_exactly_once_and_is_loadable",
        "test_atomic_checkpoint_rejects_missing_parent_bad_states_and_bad_loader",
        "test_checkpoint_state_schema_mutations_fail_closed",
        "test_checkpoint_authority_identity_mutations_fail_closed",
        "test_checkpoint_authority_identity_rejects_coordinated_expected_repack",
        "test_checkpoint_file_mutation_and_wrong_objects_are_rejected",
        "test_checkpoint_loader_and_input_are_detached",
        "test_writer_rejects_symlinked_checkpoint_root_and_noncanonical_paths",
    ),
    "tests/test_rtdetr_baseline_training_adapter.py": (
        "test_positive_prepared_binding_is_detached_and_authorities_are_immutable",
        "test_prepared_descriptor_rejects_missing_top_level_keys",
        "test_prepared_descriptor_rejects_nested_extra_and_retyped_values",
        "test_prepared_descriptor_rejects_extra_object_roles",
        "test_authority_mutations_are_rejected_before_prepared_descriptor_creation",
        "test_component_roles_call_exactly_once_and_state_predecessors_are_live",
        "test_second_run_retry_resume_overwrite_and_second_batch_are_rejected",
        "test_data_role_and_path_allowlist_fail_closed",
        "test_production_evidence_root_and_existing_root_are_rejected",
        "test_each_late_transition_mutation_fails_closed",
        "test_failed_transition_stops_downstream_calls",
        "test_optimizer_group_coverage_exclusivity_and_numeric_mutations_fail",
        "test_optimizer_group_observation_has_exact_counts_numel_and_lr",
        "test_amp_nonfinite_skip_overflow_growth_and_scale_mutations_fail",
        "test_ema_requires_policy_order_count_and_distinct_checkpoint_states",
        "test_evaluator_is_development_only_and_binds_run_ema_and_predecessor",
        "test_checkpoint_and_evidence_handoff_is_atomic_and_authority_bound",
        "test_checkpoint_serializer_rejects_raw_ema_alias_and_missing_state",
        "test_result_and_component_identity_outputs_are_detached",
        "test_no_production_artifact_or_external_execution_is_used",
    ),
    "tests/test_rtdetr_baseline_training_entry.py": (
        "test_entry_descriptor_binds_authorities_and_source_closure",
        "test_entry_process_config_mode_is_bound_and_revalidated",
        "test_entry_descriptor_mutations_fail_closed",
        "test_entry_synthetic_transaction_is_detached_and_exactly_once",
        "test_entry_rejects_production_target_without_creating_it",
        "test_entry_receipt_blocks_module_reload_and_root_replay",
        "test_entry_public_file_writer_handles_partial_writes",
    ),
    "tests/test_rtdetr_baseline_training_process_launcher.py": (
        "test_launcher_descriptor_has_bound_root_and_detaches",
        "test_launcher_process_config_mode_is_cross_file_bound",
        "test_fake_runner_launch_is_durable_and_exactly_once",
        "test_same_descriptor_replay_after_module_reload_and_different_root_is_refused",
        "test_runner_failure_classes_are_terminal_and_fail_closed",
        "test_runner_signature_and_input_mutation_are_rejected",
        "test_launch_descriptor_mutations_fail_closed",
        "test_classifier_truth_table_and_real_success_refusal",
        "test_classifier_rejects_anchor_object_and_metadata_drift",
        "test_coordinated_json_repack_without_binary_change_is_rejected",
        "test_published_root_final_name_replacement_is_rejected",
        "test_binary_anchor_mutation_and_inventory_repack_are_rejected",
        "test_descriptor_expected_binding_rejects_independently_repacked_result",
        "test_bound_root_mismatch_is_rejected_before_runner_call",
        "test_receipt_without_root_is_not_absent",
        "test_classifier_retries_interrupted_directory_enumeration",
    ),
}


def pytest_addoption(parser):
    parser.addoption("--v2b-engineering-scope", action="store_true", default=False,
                     help="Report explicitly listed historical-data cases as inapplicable; never mark them passed")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--v2b-engineering-scope"):
        for item in items:
            node = item.nodeid.split("[", 1)[0]
            reason = HISTORICAL_RUNTIME_CASES.get(node)
            if any(node == module + "::" + name for module, names in DATA_CLOSURE_CASES.items()
                   for name in names):
                reason = "requires complete historical R3 closure, including sealed partitions; missing-file failure is not this test's proof"
            if reason:
                item.add_marker(pytest.mark.skip(reason="historical-only: " + reason))


def _copy_source_tree(root):
    """Copy only source/configuration trees; never copy an artifacts subtree."""
    for name in ("src", "configs", "manifests", "vendor", "docs", "tests", "tools",
                 "environment", "scripts"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, root / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in (".gitignore", ".gitattributes", "README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, root / name)


@pytest.fixture(scope="session")
def _legacy_source_only_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy-source-only")
    _copy_source_tree(root)
    assert not (root / "artifacts").exists()
    return root


@pytest.fixture(autouse=True)
def _isolate_selected_legacy_source_only_cases(request, monkeypatch):
    if not request.config.getoption("--v2b-engineering-scope"):
        return
    relative = request.node.nodeid.split("::", 1)[0]
    name = getattr(request.node, "originalname", None) or request.node.name.split("[", 1)[0]
    if name in SOURCE_ONLY_CASES.get(relative, set()):
        root = request.getfixturevalue("_legacy_source_only_root")
        monkeypatch.setattr(request.module, "ROOT", root)


@pytest.fixture(scope="session")
def _legacy_fake_source_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2a-fake-source")
    _copy_source_tree(root)
    authority = "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1"
    shutil.copytree(ROOT / authority, root / authority)
    conversion = root / "artifacts/data/visdrone_protocol_v2_conversion_r3"
    conversion.mkdir(parents=True)
    # Exact metadata supports frozen config/authority checks; no actual data
    # or historical training output is copied into this synthetic fixture.
    for name in ("completion.json", "artifact_inventory.json", "config.json",
                 "category_contract.json", "source_identity.json"):
        shutil.copy2(ROOT / conversion.relative_to(root) / name, conversion / name)
    for role in ("train_core", "development"):
        (conversion / (role + "_coco.json")).write_text(json.dumps({
            "images": [], "annotations": [], "categories": [],
            "fixture_scope": "synthetic_cpu_fake_ports_only",
        }) + "\n")
        (conversion / (role + "_manifest.json")).write_text(json.dumps({
            "records": [], "role": role, "fixture_scope": "synthetic_cpu_fake_ports_only",
        }) + "\n")
    (root / "artifacts/training").mkdir()
    (root / "artifacts/SYNTHETIC_FIXTURE_ONLY.json").write_text(json.dumps({
        "real_data_accessed": False, "historical_runtime_certified": False,
        "purpose": "isolated legacy CPU fake-port file dependencies",
    }) + "\n")
    return root


@pytest.fixture(autouse=True)
def _isolate_selected_legacy_fake_ports(request, monkeypatch):
    relative = request.node.nodeid.split("::", 1)[0]
    name = getattr(request.node, "originalname", None) or request.node.name.split("[", 1)[0]
    cases = LEGACY_FAKE_CASES.get(relative, set()) | ADDITIONAL_FAKE_CASES.get(relative, set())
    if name in cases:
        root = request.getfixturevalue("_legacy_fake_source_root")
        if relative == "tests/test_rtdetr_baseline_smoke.py":
            isolated = request.getfixturevalue("tmp_path") / "smoke-routing-repository"
            shutil.copytree(root, isolated)
            root = isolated
            # Frozen smoke selection validates exact manifest entries, without
            # opening their raw images. Keep this allowed metadata separate
            # from the explicitly empty manifests used by fake T6B ports.
            manifest = "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json"
            shutil.copy2(ROOT / manifest, root / manifest)
        monkeypatch.setattr(request.module, "ROOT", root)
        if relative == "tests/test_rtdetr_baseline_smoke.py":
            smoke = request.module
            monkeypatch.setattr(smoke, "V3_CONFIG", root / smoke.SMOKE_V3_CONFIG_RELATIVE)
            # This frozen test replaces both directory creators with fake
            # ports. Supply its pre-existing process directory explicitly;
            # the assertion proves argv/receipt routing, not launcher creation.
            (root / smoke.SMOKE_V3_PROCESS_RELATIVE).mkdir(parents=True, exist_ok=True)
            assert not (root / smoke.SMOKE_V3_OUTPUT_RELATIVE).exists()
