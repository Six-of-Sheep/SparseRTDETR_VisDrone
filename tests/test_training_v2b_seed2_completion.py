"""CPU-only tests of the explicitly authorized historical-control bridge."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_seed2_completion as gate
from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256


def ref(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


def snapshot(root, files, commit):
    return {"repo_root": str(root), "commit": commit, "tree": "b" * 40,
            "branch": "fixture", "code_files": files, "inventory_sha256": canonical_sha256(files)}


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("historical admission CPU tests must not initialize or observe CUDA")
    for name in ("init", "_lazy_init", "is_available", "device_count", "get_rng_state"):
        monkeypatch.setattr(torch.cuda, name, blocked)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


@pytest.fixture
def definition(tmp_path):
    old_root, new_root = tmp_path / "original", tmp_path / "completion"
    old_files = {"src/training.py": ref(old_root / "src/training.py")}
    new_files = {"src/training.py": ref(new_root / "src/training.py")}
    checker = "tools/repository_contract_check.py"
    old_files[checker] = ref(old_root / checker, "d" * 64, 2)
    new_files[checker] = ref(new_root / checker, "e" * 64, 3)
    module = "src/sparse_rtdetr/baseline/training_v2b_seed2_completion.py"
    new_files[module] = ref(new_root / module, "c" * 64)
    old = snapshot(old_root, old_files, gate.BASE_COMMIT)
    new = snapshot(new_root, new_files, "d" * 40)
    old_output = old_root / "artifacts/training/old-campaign"
    output = new_root / "artifacts/training/new-campaign"
    history = {
        key: ref(old_output / (key + ".json"))
        for key in gate.HISTORY_KEYS - {"preformal_completions"}
    }
    for key, sha in (
        ("freeze_reference", gate.HISTORICAL_FREEZE_SHA256),
        ("control_completion_reference", gate.HISTORICAL_CONTROL_COMPLETION_SHA256),
        ("failed_campaign_reference", gate.HISTORICAL_FAILURE_SHA256),
        ("partial_summary_reference", gate.HISTORICAL_PARTIAL_SUMMARY_SHA256),
        ("control_ema_audit_reference", gate.HISTORICAL_EMA_AUDIT_SHA256),
    ):
        history[key]["sha256"] = sha
    history["preformal_completions"] = [
        {"cell_id": cell, "stage": stage,
         "worker_result_reference": ref(old_output / f"{cell}-{stage}/worker-result.json"),
         "completion_reference": ref(old_output / f"{cell}-{stage}.completion.json")}
        for cell in ("s1_r640", "s1_r896", "s2_r640", "s2_r896")
        for stage in ("capacity", "smoke", "smoke_replay")
    ]
    authority = ref(tmp_path / "original-authorization.txt")
    development = {
        "repo_root": str(new_root), "binding_sha256": "a" * 64,
        "annotation": {k: v for k, v in ref(tmp_path / "development_coco.json").items() if k != "sha256_scope"},
        "manifest": {k: v for k, v in ref(tmp_path / "development_manifest.json").items() if k != "sha256_scope"},
        "image_root": str(tmp_path / "development-images"), "policy": {"input_size": [896, 896]},
    }
    from sparse_rtdetr.baseline.training_v2b_replication_worker import frozen_seed_orders
    value = {
        "schema_version": 1, "kind": gate.KIND, "campaign_id": "new-campaign",
        "repo_root": str(new_root), "output_root": str(output), "source": new,
        "clean_archive_reference": ref(tmp_path / "clean-archive.json"),
        "qualification_reference": ref(tmp_path / "qualification.json"),
        "authorization_reference": authority,
        "supplemental_authorization_reference": ref(
            tmp_path / "supplemental-authorization.txt", gate.SUPPLEMENTAL_AUTHORIZATION_SHA256,
            len(gate.SUPPLEMENTAL_AUTHORIZATION_TEXT.encode("utf-8"))),
        "foundation_reference": ref(tmp_path / "foundation.json"),
        "common": {
            "torch_version": str(torch.__version__), "code_files": new_files,
            "train_core": {"fixture_only": True}, "pretrained": ref(tmp_path / "weights.pth"),
            "policy_bundle": {"authorization_reference": authority},
            "expected_gpu_uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
            "num_workers": 2, "prefetch_factor": 2,
        },
        "cells": {"s2_r896": {
            "config": gate.fixed_config(), "development_binding": development,
            "epoch_order_sha256": frozen_seed_orders(2), "capacity_plan": ref(tmp_path / "capacity.json"),
            "invocations": gate.invocation_plan("new-campaign", output),
        }},
        "stages": list(gate.STAGES), "historical": history,
        "source_equivalence": gate.source_extension_proof(old, new),
        "execution_policy": copy.deepcopy(gate.EXECUTION_POLICY),
    }
    return value


def contract(freeze, stage="control30"):
    output = Path(freeze["output_root"])
    replay = (ref(output / "execution/s2_r896-smoke/worker-result.json")
              if stage == "smoke_replay" else None)
    return gate.seed2_completion_worker_contract(
        freeze, ref(output / "freeze.json"), stage,
        stage_gate_reference=ref(output / "stage-gates" / (stage + ".json")), replay_source=replay)


def test_single_new_arm_keeps_old_control_as_a_real_external_reference(definition):
    checked = gate.validate_seed2_completion_freeze(definition, verify_files=False)
    formal = contract(checked)
    assert set(checked["cells"]) == {"s2_r896"}
    assert formal["matched_reference"] == checked["historical"]["control_result_reference"]
    assert Path(formal["matched_reference"]["path"]).is_relative_to(
        Path(checked["source_equivalence"]["old_source"]["repo_root"]))
    assert formal["kind"] != "v2b_seed_resolution_replication_worker"
    assert checked["execution_policy"]["historical_control_is_new_campaign_product"] is False
    assert checked["execution_policy"]["old_campaign_status"] == "STOP_NO_RETRY"


def test_three_unique_process_identities_share_only_new_smoke_replay_logical_run(definition):
    rows = definition["cells"]["s2_r896"]["invocations"]
    assert len({v["invocation_id"] for v in rows.values()}) == 3
    assert len({v["output_dir"] for v in rows.values()}) == 3
    assert rows["smoke"]["run_id"] == rows["smoke_replay"]["run_id"]
    assert rows["control30"]["run_id"] != rows["smoke"]["run_id"]
    assert contract(definition, "smoke")["replay_source"] is None
    assert contract(definition, "smoke_replay")["replay_source"] is not None
    assert contract(definition)["replay_source"] is None


@pytest.mark.parametrize("field,value", [
    ("seed", 1), ("input_size", 640), ("input_size", 960),
    ("physical_batch_size", 16), ("accumulation_steps", 1),
    ("amp_dtype", "float16"), ("bn_statistics", "frozen"),
    ("num_denoising", 0), ("learning_rate", 0.0002),
    ("warmup_steps", 1000), ("ema_warmups", 1000),
    ("sampling_backend", "native"), ("physical_batch_size", 8.0),
    ("accumulation_steps", True),
])
def test_any_training_semantics_change_rejects_missing_cell_design(definition, field, value):
    definition["cells"]["s2_r896"]["config"][field] = value
    with pytest.raises(gate.Seed2CompletionError, match="configuration"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


@pytest.mark.parametrize("field,value", [
    ("formal_checkpoint_resume", True), ("automatic_retry_or_batch_fallback", True),
    ("primary_endpoint_epoch", 29), ("optimizer_updates", 9005),
    ("training_extension_authorized", True), ("historical_control_is_new_campaign_product", True),
    ("old_campaign_status", "PASS"), ("clock_setting_or_reset_allowed", True),
    ("input_roles", ["train_core", "test"]),
])
def test_execution_policy_cannot_relax_the_user_authority(definition, field, value):
    definition["execution_policy"][field] = value
    with pytest.raises(gate.Seed2CompletionError, match="execution policy"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


def test_history_and_new_output_cannot_overlap(definition):
    definition["historical"]["freeze_reference"]["path"] = str(Path(definition["output_root"]) / "freeze.json")
    with pytest.raises(gate.Seed2CompletionError, match="disjoint"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


def test_original_campaign_id_cannot_be_reused(definition):
    definition["campaign_id"] = gate.HISTORICAL_CAMPAIGN_ID
    with pytest.raises(gate.Seed2CompletionError, match="old campaign ID"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


@pytest.mark.parametrize("key", [
    "freeze_reference", "control_completion_reference", "failed_campaign_reference",
    "partial_summary_reference", "control_ema_audit_reference",
])
def test_pinned_old_evidence_cannot_be_replaced_after_observing_results(definition, key):
    definition["historical"][key]["sha256"] = "f" * 64
    with pytest.raises(gate.Seed2CompletionError, match="pinned historical"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_cell", "formal_instead"])
def test_twelve_real_historical_qualification_stages_are_mandatory(definition, mutation):
    rows = definition["historical"]["preformal_completions"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_cell":
        rows[-1]["cell_id"] = "s3_r896"
    else:
        rows[-1]["stage"] = "control30"
    with pytest.raises(gate.Seed2CompletionError, match="historical engineering"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


def test_all_thirty_epoch_orders_are_fixed_before_launch(definition):
    definition["cells"]["s2_r896"]["epoch_order_sha256"]["30"] = "f" * 64
    with pytest.raises(gate.Seed2CompletionError, match="epoch orders"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


def test_formal_and_smoke_cannot_restore_old_or_new_checkpoints(definition):
    for stage in ("smoke", "control30"):
        value = contract(definition, stage)
        value["replay_source"] = definition["historical"]["control_checkpoint_reference"]
        with pytest.raises(gate.Seed2CompletionError, match="resume forbidden"):
            gate.validate_seed2_completion_worker_contract(value, definition)


def test_replay_cannot_consume_original_failed_b_or_old_smoke(definition):
    value = contract(definition, "smoke_replay")
    value["replay_source"] = definition["historical"]["historical_smoke_reference"]
    with pytest.raises(gate.Seed2CompletionError, match="only the new smoke"):
        gate.validate_seed2_completion_worker_contract(value, definition)


def test_matched_source_cannot_be_relabelled_into_new_output(definition):
    value = contract(definition)
    value["matched_reference"]["path"] = str(Path(definition["output_root"]) / "fake-old-640.json")
    with pytest.raises(gate.Seed2CompletionError, match="actual historical matched source"):
        gate.validate_seed2_completion_worker_contract(value, definition)


@pytest.mark.parametrize("field,value", [("stage", "capacity"), ("cell_id", "s2_r640"),
                                        ("kind", "v2b_seed_resolution_replication_worker")])
def test_new_worker_is_not_a_generic_original_gate_bypass(definition, field, value):
    selected = contract(definition)
    selected[field] = value
    with pytest.raises(gate.Seed2CompletionError):
        gate.validate_seed2_completion_worker_contract(selected, definition)


def test_supplemental_authorization_is_bound_separately(definition):
    definition["supplemental_authorization_reference"]["sha256"] = definition["authorization_reference"]["sha256"]
    with pytest.raises(gate.Seed2CompletionError, match="supplemental grant"):
        gate.validate_seed2_completion_freeze(definition, verify_files=False)


@pytest.mark.parametrize("change", ["bytes", "size", "removed", "new_core", "base", "inventory"])
def test_source_delta_rejects_any_unapproved_original_change(definition, change):
    old = copy.deepcopy(definition["source_equivalence"]["old_source"])
    new = copy.deepcopy(definition["source"])
    if change == "bytes":
        new["code_files"]["src/training.py"]["sha256"] = "e" * 64
    elif change == "size":
        new["code_files"]["src/training.py"]["size_bytes"] += 1
    elif change == "removed":
        del new["code_files"]["src/training.py"]
    elif change == "new_core":
        name = "src/sparse_rtdetr/baseline/changed_training_core.py"
        new["code_files"][name] = ref(Path(new["repo_root"]) / name)
    elif change == "base":
        old["commit"] = "e" * 40
    new["inventory_sha256"] = canonical_sha256(new["code_files"])
    if change == "inventory":
        new["inventory_sha256"] = "f" * 64
    with pytest.raises(gate.Seed2CompletionError):
        gate.source_extension_proof(old, new)


def test_source_proof_does_not_claim_equal_cross_checkout_inventory_hashes(definition):
    proof = definition["source_equivalence"]
    assert proof["original_tracked_files_byte_identical"] is False
    assert proof["original_scientific_tracked_files_byte_identical"] is True
    assert proof["modified_files"] == ["tools/repository_contract_check.py"]
    assert proof["permitted_modified_files"] == ["tools/repository_contract_check.py"]
    assert proof["modified_files_are_evidence_only"] is True
    assert proof["old_source"]["inventory_sha256"] != proof["new_inventory_sha256"]
    assert proof["training_mathematics_changed"] is False


def test_qualification_runs_the_unmodified_verifier_in_its_actual_old_checkout(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    reference = ref(old / "qualification.json")
    observed = {}
    def run(argv, **kwargs):
        observed.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="HISTORICAL_QUALIFICATION_PASS:" + reference["sha256"] + "\n")
    monkeypatch.setattr(gate.subprocess, "run", run)
    monkeypatch.setattr(gate, "_json", lambda value: {"actual_reference": value})
    result = gate.verify_historical_qualification(reference, str(old))
    assert result["actual_reference"] == reference
    assert observed["cwd"] == old
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert observed["env"]["PYTHONPATH"] == str(old / "src")
    assert "_qualified_document" in observed["argv"][2]
    assert observed["argv"][3] == str(old)
    assert "__file__ =" not in observed["argv"][2]
    assert "monkeypatch" not in observed["argv"][2]


def test_historical_qualification_subprocess_failure_is_not_converted_to_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=3, stderr="source mismatch", stdout=""))
    with pytest.raises(gate.Seed2CompletionError, match="original historical qualification verifier failed"):
        gate.verify_historical_qualification(ref(tmp_path / "qualification.json"), str(tmp_path))


def test_reference_validation_rejects_forbidden_role_and_truncated_hash(tmp_path):
    with pytest.raises(gate.Seed2CompletionError, match="prohibited data role"):
        gate._reference(ref(tmp_path / "confirmatory" / "annotations.json"))
    with pytest.raises(gate.Seed2CompletionError, match="complete SHA"):
        gate._reference(ref(tmp_path / "report.json", "abcd"))


def test_history_builder_normalizes_real_checkpoint_metadata_without_inventing_a_scope(definition, monkeypatch):
    """Saved checkpoint receipts use extra fields and omit sha256_scope."""
    h = definition["historical"]
    result_ref = h["control_result_reference"]
    raw_checkpoint = {k: v for k, v in h["control_checkpoint_reference"].items() if k != "sha256_scope"}
    raw_checkpoint.update(epoch=30, optimizer_updates=9120, binding_sha256="a" * 64)
    result = {
        "initial_state_reference": h["control_initial_state_reference"],
        "final_state_reference": h["control_final_state_reference"],
        "checkpoints": {"epoch_30": {"checkpoint": raw_checkpoint}},
    }
    documents = {
        h["freeze_reference"]["path"]: {"source": definition["source_equivalence"]["old_source"]},
        h["control_completion_reference"]["path"]: {"result_reference": result_ref},
        result_ref["path"]: result,
        h["partial_summary_reference"]["path"]: {
            "failed_controller_reference": h["failed_controller_reference"],
            "failed_monitor_reference": h["failed_monitor_reference"],
        },
        h["failed_campaign_reference"]["path"]: {"completed": h["preformal_completions"]},
    }
    monkeypatch.setattr(gate, "_json", lambda reference: copy.deepcopy(documents[reference["path"]]))
    built, old = gate._build_history(
        historical_freeze_reference=h["freeze_reference"],
        historical_control_completion_reference=h["control_completion_reference"],
        historical_control_ema_audit_reference=h["control_ema_audit_reference"],
        failed_campaign_reference=h["failed_campaign_reference"],
        partial_summary_reference=h["partial_summary_reference"])
    assert built["control_checkpoint_reference"] == h["control_checkpoint_reference"]
    assert built["control_checkpoint_reference"]["sha256_scope"] == "complete_file_bytes"
    assert "epoch" not in built["control_checkpoint_reference"]
    assert raw_checkpoint["optimizer_updates"] == 9120
