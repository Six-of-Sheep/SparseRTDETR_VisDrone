"""Synthetic CPU adversaries for qualification, never a GPU or data authorization."""
from __future__ import annotations

import copy
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from sparse_rtdetr.baseline import training_v2b_replication_gate as gate
from sparse_rtdetr.baseline import training_v2b_resolution_diagnostics as diagnostic
from sparse_rtdetr.baseline.training_v2b_campaign import _write_json, file_reference

ERROR = gate.ReplicationQualificationError


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("qualification fixtures cannot read weights or initialize CUDA")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.fixture
def directory(tmp_path_factory):
    # No protected-data-role spelling is needed for synthetic filesystem fixtures.
    return tmp_path_factory.mktemp("replication_qualification")


def inputs():
    return {
        "primary_ap_percent": dict(zip(gate.CELLS, (20., 21., 22., 23.))),
        "coco_ap_small_percent": dict(zip(gate.CELLS, (15., 16., 17., 18.))),
        "small_gt_count": dict.fromkeys(gate.CELLS, 100),
        "small_unmatched_iou50": dict(zip(gate.CELLS, (80, 78, 76, 70))),
        "cluster_diagonal_small_ci95_pp": [.5, 1.5],
    }


@pytest.fixture(scope="module")
def bootstrap_fixture():
    policy = diagnostic.FROZEN_POLICY
    records = [
        {"coco_image_id": i, "split": "val", "sequence_key": "synthetic_" + str((i - 1) % 75),
         "image_sha256": hashlib.sha256(("fixture-image-" + str(i)).encode()).hexdigest()}
        for i in range(1, 549)
    ]
    ids = tuple(range(1, 549))
    clusters = diagnostic.build_development_clusters(records, ids)
    points = {
        cell: {metric: float(20 + index + j)
               for j, metric in enumerate(gate.BOOTSTRAP_METRICS)}
        for index, cell in enumerate(gate.CELLS)
    }
    identities = {cell: {"synthetic_cache_identity": cell} for cell in gate.CELLS}
    coefficients = np.asarray(list(diagnostic.CONTRASTS.values()))
    point = np.asarray([[points[cell][metric] for metric in gate.BOOTSTRAP_METRICS]
                        for cell in gate.CELLS])
    contrasts = coefficients @ point
    resampling = {}
    for kind, units, count, seed in (
        ("cluster", clusters, policy.cluster_replicates, policy.rng_seed),
        ("image_sensitivity", tuple((i,) for i in ids), policy.image_replicates, policy.rng_seed + 1),
    ):
        draws = np.random.Generator(np.random.PCG64(seed)).integers(
            0, len(units), size=(count, len(units)), dtype=np.int64)
        distribution = (np.broadcast_to(contrasts, (count, *contrasts.shape))
                        + np.linspace(-.2, .2, count)[:, None, None])
        ci = np.quantile(distribution, [.025, .975], axis=0, method="linear")
        resampling[kind] = {
            "replicates": count, "unit_count": len(units), "seed": seed,
            "draws_sha256": hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest(),
            "distribution_sha256": hashlib.sha256(distribution.astype("<f8").tobytes()).hexdigest(),
            "distribution_axes": ["replicate", "contrast", "metric"],
            "distribution": distribution.tolist(),
            "contrasts": {
                name: {metric: {"point_pp": float(contrasts[j, k]),
                                "ci95_pp": [float(ci[0, j, k]), float(ci[1, j, k])]}
                       for k, metric in enumerate(gate.BOOTSTRAP_METRICS)}
                for j, name in enumerate(diagnostic.CONTRASTS)
            },
        }
    report = {
        "schema_version": 1, "method": diagnostic.frozen_method(),
        "method_sha256": gate._digest(diagnostic.frozen_method()),
        "effective_policy": asdict(policy), "fixture_only": False,
        "cells": list(gate.CELLS), "metrics": list(gate.BOOTSTRAP_METRICS),
        "contrasts": list(diagnostic.CONTRASTS), "point_cells": points,
        "clusters": [list(x) for x in clusters],
        "manifest_canonical_sha256": gate._digest(records),
        "cache_identities": identities, "resampling": resampling,
    }
    return report, records, ids, points, identities


def audit_bootstrap(fixture, report=None):
    saved, records, ids, points, identities = fixture
    return gate._audit_bootstrap(saved if report is None else report,
                                 records, ids, points, identities)


def test_saved_distribution_sha_rng_draws_and_every_percentile_are_recomputed(bootstrap_fixture):
    result = audit_bootstrap(bootstrap_fixture)
    assert result["cluster"]["replicates"] == 2000
    assert result["image_sensitivity"]["replicates"] == 200
    assert result["cluster"]["unit_count"] == 75
    assert result["image_sensitivity"]["unit_count"] == 548
    assert result["cluster"]["contrasts"]["diagonal_total"]["coco_AP_small"]["ci95_pp"] == pytest.approx([2.81, 3.19])


@pytest.mark.parametrize("kind", ["cluster", "image_sensitivity"])
@pytest.mark.parametrize("field", ["draws_sha256", "distribution_sha256"])
def test_fabricated_resampling_digest_is_rejected(bootstrap_fixture, kind, field):
    changed = copy.deepcopy(bootstrap_fixture[0])
    changed["resampling"][kind][field] = "0" * 64
    with pytest.raises(ERROR, match="SHA differs"):
        audit_bootstrap(bootstrap_fixture, changed)


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf"), float("-inf"), "3.0"])
def test_distribution_rejects_nonreal_or_nonfinite_entries(bootstrap_fixture, value):
    changed = copy.deepcopy(bootstrap_fixture[0])
    changed["resampling"]["cluster"]["distribution"][0][0][0] = value
    with pytest.raises(ERROR, match="finite real"):
        audit_bootstrap(bootstrap_fixture, changed)


def test_repacked_distribution_cannot_preserve_an_invented_positive_ci(bootstrap_fixture):
    changed = copy.deepcopy(bootstrap_fixture[0])
    row = changed["resampling"]["cluster"]
    values = np.asarray(row["distribution"])
    values[:, 0, 3] -= 10
    row["distribution"] = values.tolist()
    row["distribution_sha256"] = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
    with pytest.raises(ERROR, match="percentile CI"):
        audit_bootstrap(bootstrap_fixture, changed)


@pytest.mark.parametrize("field,value", [
    ("fixture_only", True), ("fixture_only", 0), ("schema_version", True),
    ("cells", ["t640_e640", "t896_e896"]),
    ("metrics", ["primary_AP", "coco_AP_small"]),
    ("method_sha256", "f" * 64),
])
def test_fixture_or_changed_bootstrap_design_cannot_qualify(bootstrap_fixture, field, value):
    changed = copy.deepcopy(bootstrap_fixture[0])
    changed[field] = value
    with pytest.raises(ERROR):
        audit_bootstrap(bootstrap_fixture, changed)


@pytest.mark.parametrize("field,value", [
    ("replicates", True), ("replicates", 200), ("unit_count", 548),
    ("seed", 20260916), ("distribution_axes", ["metric", "contrast", "replicate"]),
])
def test_resampling_axes_counts_and_seed_are_frozen(bootstrap_fixture, field, value):
    changed = copy.deepcopy(bootstrap_fixture[0])
    changed["resampling"]["cluster"][field] = value
    with pytest.raises(ERROR):
        audit_bootstrap(bootstrap_fixture, changed)


def test_full_evaluator_point_cannot_be_replaced_by_the_analysis_copy(bootstrap_fixture):
    changed = copy.deepcopy(bootstrap_fixture[0])
    changed["point_cells"]["t896_e896"]["primary_AP"] += .01
    with pytest.raises(ERROR, match="verified full evaluator"):
        audit_bootstrap(bootstrap_fixture, changed)


@pytest.mark.parametrize("where", ["point_pp", "ci95_pp"])
def test_boolean_is_not_a_contrast_number(bootstrap_fixture, where):
    changed = copy.deepcopy(bootstrap_fixture[0])
    value = changed["resampling"]["cluster"]["contrasts"]["diagonal_total"]["coco_AP_small"]
    value[where] = True if where == "point_pp" else [True, 3.19]
    with pytest.raises(ERROR, match="finite real"):
        audit_bootstrap(bootstrap_fixture, changed)


def test_cluster_identity_is_rebuilt_from_sequence_and_content_links(bootstrap_fixture):
    report, records, ids, points, identities = bootstrap_fixture
    changed = copy.deepcopy(records)
    changed[0]["image_sha256"] = changed[1]["image_sha256"]
    with pytest.raises(ERROR, match="cardinality"):
        gate._audit_bootstrap(report, changed, ids, points, identities)


def test_exactly_four_independent_strict_inequalities_determine_eligibility():
    positive = gate._conditions(inputs())
    assert positive["eligible_for_fixed_paired_seed1_and_seed2"] is True
    assert all(positive["checks"].values())
    for name in ("primary_ap_percent", "coco_ap_small_percent"):
        value = inputs()
        value[name]["t896_e896"] = value[name]["t640_e640"]
        assert gate._conditions(value)["eligible_for_fixed_paired_seed1_and_seed2"] is False
    value = inputs()
    value["cluster_diagonal_small_ci95_pp"] = [0., 10.]
    assert gate._conditions(value)["checks"]["cluster_small_ci_lower_positive"] is False
    value = inputs()
    value["small_unmatched_iou50"]["t896_e896"] = value["small_unmatched_iou50"]["t640_e640"]
    assert gate._conditions(value)["checks"]["small_unmatched_at_iou50_decreases"] is False


@pytest.mark.parametrize("field,value", [
    ("small_gt_count", 0), ("small_gt_count", True), ("small_gt_count", 100.),
    ("small_unmatched_iou50", -1), ("small_unmatched_iou50", True),
    ("small_unmatched_iou50", 101), ("primary_ap_percent", True),
    ("primary_ap_percent", float("nan")), ("coco_ap_small_percent", 101),
])
def test_invalid_threshold_units_counts_and_numerical_types_fail(field, value):
    witness = inputs()
    witness[field]["t640_e640"] = value
    with pytest.raises(ERROR):
        gate._conditions(witness)


def test_mismatched_denominators_or_reversed_ci_do_not_become_a_scientific_win():
    witness = inputs()
    witness["small_gt_count"]["t896_e896"] = 101
    with pytest.raises(ERROR, match="denominator differs"):
        gate._conditions(witness)
    witness = inputs()
    witness["cluster_diagonal_small_ci95_pp"] = [2., 1.]
    with pytest.raises(ERROR, match="reversed"):
        gate._conditions(witness)


@pytest.mark.parametrize("raw", [
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e309}',
    b'{"value":1,"value":2}',
])
def test_json_numbers_and_duplicate_keys_are_strict(raw):
    with pytest.raises(ValueError):
        gate._json(raw)


def test_reader_authenticates_complete_bytes_and_refuses_changed_artifact(directory):
    reference = _write_json(directory / "evidence.json", {"value": 1})
    assert gate._Reader().json(reference) == {"value": 1}
    (directory / "evidence.json").write_bytes(b'{"value":2}\n')
    with pytest.raises(ERROR, match="SHA"):
        gate._Reader().json(reference)


def test_reader_refuses_symlink_parent_and_leaf(directory):
    actual = directory / "actual"
    actual.mkdir()
    reference = _write_json(actual / "evidence.json", {"value": 1})
    alias = directory / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    for path in (alias / "evidence.json", directory / "linked.json"):
        if path.name == "linked.json":
            path.symlink_to(actual / "evidence.json")
        with pytest.raises((OSError, ERROR)):
            gate._Reader().read({**reference, "path": str(path)})


@pytest.mark.parametrize("change", [
    {"size_bytes": True}, {"sha256": "abc"}, {"sha256_scope": "prefix"},
    {"path": "relative.json"}, {"path": "/tmp/../tmp/evidence.json"},
])
def test_incomplete_or_ambiguous_file_reference_is_rejected(directory, change):
    reference = _write_json(directory / "evidence.json", {})
    reference.update(change)
    with pytest.raises(ERROR):
        gate._Reader().read(reference)


@pytest.mark.parametrize("role", ["confirmatory", "test", "confirmatory_coco.json", "test_coco.json"])
def test_protected_data_paths_are_rejected_before_open(directory, role):
    reference = {"path": str(directory / role / "unopened.json"), "sha256": "a" * 64,
                 "size_bytes": 1, "sha256_scope": "complete_file_bytes"}
    with pytest.raises(ERROR, match="protected data role"):
        gate._Reader().read(reference)


def test_three_field_historical_metadata_requires_explicit_embedded_context(directory):
    reference = _write_json(directory / "development_manifest.json", {})
    del reference["sha256_scope"]
    with pytest.raises(ERROR, match="schema"):
        gate._Reader().json(reference)
    assert gate._Reader().json(reference, embedded=True) == {}


def _resign(document):
    document["leaf_references"] = sorted(document["leaf_references"], key=lambda ref: ref["path"])
    document["leaf_inventory_sha256"] = gate._digest(document["leaf_references"])
    document["qualification_sha256"] = gate._digest(
        {key: value for key, value in document.items() if key != "qualification_sha256"})
    return document


@pytest.fixture
def sealed(directory, bootstrap_fixture, monkeypatch):
    """Synthetic graph for the fast binding layer; full audit is tested separately."""
    bootstrap, records, ids, points, identities = copy.deepcopy(bootstrap_fixture)
    authority = _write_json(directory / "user-authorization.json", {"synthetic": True})
    monkeypatch.setattr(gate, "AUTHORIZATION_SHA256", authority["sha256"])
    manifest = _write_json(directory / "development_manifest.json", {"records": records})
    foundation = _write_json(directory / "foundation.json", {"predeclared_additional_seeds": [1, 2]})
    evaluator = _write_json(directory / "evaluator-evidence.json", {"synthetic": True})
    source_root = Path(gate.__file__).resolve().parents[3]
    source_files = {name: file_reference(source_root / name) for name in gate._HELPERS}
    consumer_identity = {"repo_root": str(source_root), "commit": "1" * 40, "tree": "2" * 40,
                         "branch": "synthetic-no-Git-mutation", "inventory_sha256": "3" * 64}
    git_values = {("rev-parse", "HEAD"): consumer_identity["commit"],
                  ("rev-parse", "HEAD^{tree}"): consumer_identity["tree"],
                  ("branch", "--show-current"): consumer_identity["branch"]}
    monkeypatch.setattr(gate, "_git", lambda root, *args: git_values[args])
    helper_correspondence = {name: {"producer": ref, "consumer": ref}
                             for name, ref in source_files.items()}
    backend_ref = _write_json(directory / "synthetic-backend.json", {"synthetic": True})
    campaign = {
        "campaign_id": "synthetic", "output_root": str(directory),
        "source": {"repo_root": str(source_root), "code_files": source_files},
        "authorization_reference": authority, "foundation_reference": foundation,
        "evaluator_evidence_reference": evaluator,
        "checkpoints": {"640": {"synthetic_checkpoint": 640}, "896": {"synthetic_checkpoint": 896}},
        "development_bindings": {"640": {"manifest": manifest}, "896": {"manifest": manifest}},
        "official_gt_bindings": {"640": {}, "896": {}}, "policy_bundle": {},
        "expected_gpu_uuid": "synthetic-no-GPU", "cache_policy": "synthetic-no-inference",
    }
    campaign_ref = _write_json(directory / "campaign.json", campaign)
    witness = inputs()
    witness["primary_ap_percent"] = {cell: points[cell]["primary_AP"] for cell in gate.CELLS}
    witness["coco_ap_small_percent"] = {cell: points[cell]["coco_AP_small"] for cell in gate.CELLS}
    witness["cluster_diagonal_small_ci95_pp"] = bootstrap["resampling"]["cluster"]["contrasts"]["diagonal_total"]["coco_AP_small"]["ci95_pp"]
    conditions = gate._conditions(witness)
    seed_gate = {
        "available": True, "checks": conditions["checks"],
        "eligible_for_fixed_paired_seed1_and_seed2": True,
        "not_model_certification": True, "not_training_seed_uncertainty": True,
    }
    bootstrap["seed_gate"] = seed_gate
    bootstrap_ref = _write_json(directory / "paired-bootstrap.json", bootstrap)
    analysis_cells, summary_cells, completed, extra_documents = {}, {}, {}, []
    for cell in gate.CELLS:
        home = directory / cell
        home.mkdir()
        contract = gate.cross_worker_contract(campaign, cell)
        contract_ref = _write_json(home / "contract.json", contract)
        binding_ref = _write_json(home / "binding.json", {"synthetic": True})
        sample_ref = _write_json(home / "sample.json", {"synthetic_native_sample": True})
        native = {"status": "PASS", "worker_exited": True,
                  "sampled_clock_compliance": True, "samples": sample_ref}
        native_ref = _write_json(home / "native-final.json", native)
        worker = {
            "status": "PASS", "cell_key": cell, "run_id": contract["run_id"],
            "contract_reference": contract_ref, "binding_reference": binding_ref,
            "primary_official_gt": {"metrics": {"AP": points[cell]["primary_AP"]}},
            "coco_secondary": {"metrics": {"AP_small": {"value": points[cell]["coco_AP_small"]}}},
        }
        worker_ref = _write_json(home / "worker-result.json", worker)
        launch_ref = _write_json(home / "launch.json", {"contract": contract_ref})
        completion = {
            "result_reference": worker_ref, "launch_reference": launch_ref,
            "monitor": {**native, "final_report_reference": native_ref},
        }
        completion_ref = _write_json(home / "completion.json", completion)
        completed[cell] = {"contract_reference": contract_ref, "result_reference": worker_ref,
                           "completion_reference": completion_ref}
        errors = _write_json(home / "errors.json", {"small_unmatched": witness["small_unmatched_iou50"][cell]})
        raw = _write_json(home / "raw-coverage.json", {"synthetic": True})
        prediction = _write_json(home / "prediction.json", {"synthetic": True})
        queries = _write_json(home / "queries.json", {"synthetic": True})
        analysis_cells[cell] = {
            "worker_result": worker_ref, "errors": errors, "raw_query_coverage": raw,
            "prediction_artifact": prediction, "raw_query_artifact": queries,
        }
        summary_cells[cell] = {
            "worker_result": worker_ref, "errors_reference": errors, "raw_query_reference": raw,
        }
        extra_documents.extend((worker, completion, {"contract": contract_ref}, {"synthetic": True}))
    result = {"status": "PASS", "training_started": False, "campaign_reference": campaign_ref,
              "completed": completed}
    result_ref = _write_json(directory / "campaign-result.json", result)
    analysis = {
        "status": "PASS", "method": diagnostic.frozen_method(), "campaign_reference": campaign_ref,
        "campaign_result_reference": result_ref, "cells": analysis_cells,
        "paired_bootstrap": bootstrap_ref, "seed_gate": seed_gate,
    }
    analysis_ref = _write_json(directory / "analysis-result.json", analysis)
    verifier_ref = file_reference(Path(gate.__file__).resolve())
    reader = gate._Reader()
    reader.source_paths.add(verifier_ref["path"])
    reader.source_paths.update(ref["path"] for ref in source_files.values())
    for value in (campaign, result, analysis, *extra_documents):
        reader.visit(value)
    for reference in (campaign_ref, result_ref, analysis_ref, verifier_ref, backend_ref):
        reader.read(reference, retain=False)
    summary = {
        "status": "PASS", "conditions": conditions, "point_cells": points, "cells": summary_cells,
        "bootstrap": gate._audit_bootstrap(bootstrap, records, ids, points, identities),
        "progress": {"status": "PASS"}, "backend_references": [backend_ref],
        "official_gt_provenance": {"producer_adapter": verifier_ref, "verifier_adapter": verifier_ref},
        "scope": {"gpu_work_launched": False, "training_launched": False,
                  "bootstrap_ap_rerun": False, "image_bytes_opened": False,
                  "training_seed_uncertainty_estimated": False, "model_certification": False},
        "development": {"image_ids": list(ids), "clusters": bootstrap["clusters"],
                        "manifest_records": records, "cache_identities": identities},
    }
    document = _resign({
        "schema_version": 1, "kind": gate.KIND, "status": "PASS",
        "eligible_for_fixed_paired_seed1_and_seed2": True, "analysis_reference": analysis_ref,
        "campaign_reference": campaign_ref, "campaign_result_reference": result_ref,
        "conditional_authorization_reference": authority,
        "method_sha256": gate._digest(diagnostic.frozen_method()),
        "allowed_experiments": list(copy.deepcopy(gate.ALLOWED_EXPERIMENTS)),
        "gpu_admission_granted": False, "requires_fresh_native_admission": True, "fixture_only": False,
        "producer_source": campaign["source"], "consumer_source_identity": consumer_identity,
        "verifier_reference": verifier_ref,
        "helper_correspondence": helper_correspondence, "audit_summary": summary,
        "leaf_references": reader.inventory(),
    })
    reference = _write_json(directory / "qualification.json", document)
    return document, reference, directory


def test_fast_worker_binding_never_rebuilds_caches_or_touches_a_gpu(sealed, monkeypatch):
    document, reference, _ = sealed
    from sparse_rtdetr.baseline import training_v2b_primary_cache as primary

    def forbidden(*args, **kwargs):
        raise AssertionError("fast binding must not rebuild nonlinear caches")

    monkeypatch.setattr(gate, "_full_audit", forbidden)
    monkeypatch.setattr(diagnostic, "prepare_coco_cache", forbidden)
    monkeypatch.setattr(primary, "prepare_primary_cache", forbidden)
    checked = gate.verify_replication_qualification_binding(reference)
    assert checked == document
    assert checked["gpu_admission_granted"] is False
    assert checked["requires_fresh_native_admission"] is True


@pytest.mark.parametrize("field,value", [
    ("gpu_admission_granted", True), ("requires_fresh_native_admission", False),
    ("fixture_only", True), ("eligible_for_fixed_paired_seed1_and_seed2", 1),
    ("status", "RUNNING"),
])
def test_repacked_qualification_cannot_expand_scope_or_boolean_eligibility(sealed, field, value):
    document, _, directory = sealed
    document[field] = value
    changed = _write_json(directory / "altered.json", _resign(document))
    with pytest.raises(ERROR):
        gate.verify_replication_qualification_binding(changed)


@pytest.mark.parametrize("field,value", [
    ("seed", 0), ("seed", 3), ("input_size", 960), ("physical_batch_size", 16),
    ("accumulation_steps", 1), ("logical_batch_size", 32), ("epochs", 120),
])
def test_repacked_scope_is_fixed_to_two_seeds_two_resolutions_eight_by_two_thirty_epochs(sealed, field, value):
    document, _, directory = sealed
    document["allowed_experiments"][0][field] = value
    changed = _write_json(directory / "altered.json", _resign(document))
    with pytest.raises(ERROR, match="fixed seed/resolution"):
        gate.verify_replication_qualification_binding(changed)


def test_missing_native_leaf_is_discovered_even_with_recomputed_qualification_digest(sealed):
    document, _, directory = sealed
    omitted = next(ref for ref in document["leaf_references"] if ref["path"].endswith("/sample.json"))
    document["leaf_references"].remove(omitted)
    changed = _write_json(directory / "altered.json", _resign(document))
    with pytest.raises(ERROR, match="leaf omitted"):
        gate.verify_replication_qualification_binding(changed)


def test_after_seal_modified_raw_evidence_is_detected_without_running_diagnostics(sealed):
    document, reference, _ = sealed
    raw = Path(document["audit_summary"]["cells"]["t896_e896"]["raw_query_reference"]["path"])
    raw.write_bytes(raw.read_bytes() + b"changed")
    with pytest.raises(ERROR):
        gate.verify_replication_qualification_binding(reference)


def test_swapped_analysis_or_worker_reference_is_not_accepted(sealed):
    document, _, directory = sealed
    document["analysis_reference"] = document["campaign_result_reference"]
    changed = _write_json(directory / "altered.json", _resign(document))
    with pytest.raises((ERROR, KeyError)):
        gate.verify_replication_qualification_binding(changed)


def test_full_validation_repeats_independent_audit_and_rejects_false_saved_summary(sealed, monkeypatch):
    document, reference, _ = sealed
    called = []

    def audit(ref):
        called.append(ref)
        changed = copy.deepcopy(document["audit_summary"])
        changed["conditions"]["inputs"]["small_unmatched_iou50"]["t896_e896"] += 1
        return {"helpers": {}}, changed, document["leaf_references"], document["verifier_reference"]

    monkeypatch.setattr(gate, "_full_audit", audit)
    with pytest.raises(ERROR, match="no longer reproduces"):
        gate.validate_replication_qualification(reference)
    assert called == [document["analysis_reference"]]


def test_build_persists_ineligible_audit_without_granting_gpu_admission(directory, monkeypatch):
    reference = _write_json(directory / "analysis.json", {"synthetic": True})
    witness = inputs()
    witness["cluster_diagonal_small_ci95_pp"] = [0., 1.]
    conditions = gate._conditions(witness)
    authority = _write_json(directory / "authorization.json", {"synthetic": True})
    context = {
        "analysis": {"campaign_reference": reference, "campaign_result_reference": reference},
        "campaign": {"authorization_reference": authority, "source": {"synthetic": True}}, "helpers": {},
        "consumer_source_identity": {"synthetic": True},
    }
    monkeypatch.setattr(gate, "_full_audit", lambda ref: (
        context, {"conditions": conditions}, [reference], reference))
    output = directory / "qualification.json"
    result = gate.build_replication_qualification(reference, output)
    assert result["qualification"]["status"] == "PASS"
    assert result["qualification"]["eligible_for_fixed_paired_seed1_and_seed2"] is False
    assert result["qualification"]["gpu_admission_granted"] is False
    assert result["qualification_reference"] == file_reference(output)
    assert output.stat().st_mode & 0o777 == 0o600
    before = output.read_bytes()
    with pytest.raises(ERROR, match="new canonical"):
        gate.build_replication_qualification(reference, output)
    assert output.read_bytes() == before


def test_full_audit_requires_cpu_visibility_before_any_input_or_model_work(directory, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-GPU")
    reference = {"path": str(directory / "absent.json"), "sha256": "a" * 64,
                 "size_bytes": 1, "sha256_scope": "complete_file_bytes"}
    with pytest.raises(ERROR, match="empty CUDA_VISIBLE_DEVICES"):
        gate.build_replication_qualification(reference, directory / "qualification.json")
    with pytest.raises(ERROR, match="empty CUDA_VISIBLE_DEVICES"):
        gate.validate_replication_qualification(reference)


def _native_report(directory, records):
    rows, previous = [], None
    for index, record in enumerate(records):
        entry = {"sequence": index, "previous_sha256": previous, "record": record}
        entry["sha256"] = gate.evidence_digest(entry)
        previous = entry["sha256"]
        rows.append(entry)
    raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
    path = directory / "native-samples.jsonl.gz"
    path.write_bytes(gzip.compress(raw))
    return {"samples": file_reference(path),
            "sample_hash_chain": {"records": len(rows), "last_sha256": previous}}


def _native_records():
    return [
        *({"kind": "clock", "graphics_clock_mhz": 1500., "sm_clock_mhz": 1500.} for _ in range(3)),
        {"kind": "health", "gpu": {"current_graphics_clock_mhz": 1500.},
         "kernel_health": {"readable": True, "findings": []}},
    ]


def test_native_sample_chain_is_authenticated_without_querying_current_hardware(directory):
    report = _native_report(directory, _native_records())
    result = gate._audit_native_samples(gate._Reader(), report)
    assert result["status"] == "PASS"
    assert result["records"] == 4
    assert result["clock_samples"] == 3
    assert result["maximum_sampled_mhz"] == 1500.


@pytest.mark.parametrize("key,value", [
    ("graphics_clock_mhz", 1501.), ("sm_clock_mhz", 1501.),
    ("graphics_clock_mhz", True), ("sm_clock_mhz", float("inf")),
])
def test_native_clock_ceiling_covers_both_graphics_and_sm_samples(directory, key, value):
    rows = _native_records()
    rows[0][key] = value
    if math_is_finite(value):
        report = _native_report(directory, rows)
        with pytest.raises(ERROR):
            gate._audit_native_samples(gate._Reader(), report)
    else:
        with pytest.raises(ValueError):
            _native_report(directory, rows)


def math_is_finite(value):
    import math
    return math.isfinite(value)


def test_health_sample_above_ceiling_or_new_kernel_error_is_rejected(directory):
    rows = _native_records()
    rows[-1]["gpu"]["current_graphics_clock_mhz"] = 1501.
    report = _native_report(directory, rows)
    with pytest.raises(ERROR, match="allowed range"):
        gate._audit_native_samples(gate._Reader(), report)
    rows = _native_records()
    rows[-1]["kernel_health"]["findings"] = [{"code": "Xid"}]
    report = _native_report(directory, rows)
    with pytest.raises(ERROR, match="newly faulty"):
        gate._audit_native_samples(gate._Reader(), report)


def test_native_final_chain_and_minimum_samples_are_mandatory(directory):
    report = _native_report(directory, _native_records())
    report["sample_hash_chain"]["last_sha256"] = "0" * 64
    with pytest.raises(ERROR, match="final chain"):
        gate._audit_native_samples(gate._Reader(), report)
    report = _native_report(directory, _native_records()[1:])
    with pytest.raises(ERROR, match="lacks complete clock"):
        gate._audit_native_samples(gate._Reader(), report)


def test_native_reencoded_line_cannot_forge_the_saved_hash_chain(directory):
    report = _native_report(directory, _native_records())
    path = Path(report["samples"]["path"])
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    rows[0]["record"]["graphics_clock_mhz"] = 1490.
    path.write_bytes(gzip.compress(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"))
    report["samples"] = file_reference(path)
    with pytest.raises(ERROR, match="chain hash"):
        gate._audit_native_samples(gate._Reader(), report)


@pytest.fixture
def tensor_worker(directory):
    from types import SimpleNamespace
    from sparse_rtdetr.baseline.training_v2b_development import _coco_metrics

    precision = np.full((10, 101, 10, 4, 3), .25, dtype=np.float64)
    recall = np.full((10, 10, 4, 3), .5, dtype=np.float64)
    precision[:, :, :, 1, 2] = .125
    params = SimpleNamespace(
        iouType="bbox", useCats=1, iouThrs=np.linspace(.5, .95, 10),
        recThrs=np.linspace(0., 1., 101), catIds=list(range(1, 11)),
        areaRng=[[0., 1e10], [0., 1024.], [1024., 9216.], [9216., 1e10]],
        areaRngLbl=["all", "small", "medium", "large"], maxDets=[1, 10, 100],
    )
    metrics, axes, _, _ = _coco_metrics(
        SimpleNamespace(params=params, eval={"precision": precision, "recall": recall}))
    path = directory / "coco-tensors.npz"
    np.savez_compressed(path, precision=precision, recall=recall)
    return {"coco_secondary": {"metrics": metrics, "axes": axes,
                              "tensor_artifact": file_reference(path)}}, path, precision, recall


def test_full_coco_ap_small_is_recalculated_from_the_saved_accumulator(tensor_worker):
    worker, _, _, _ = tensor_worker
    observed = gate._tensor_metrics(gate._Reader(), worker)
    assert observed["AP"] == 25.
    assert observed["AP_small"] == 12.5
    assert observed["AR_small"] == 50.


@pytest.mark.parametrize("mutant", ["dtype", "shape", "nan", "range", "copied_metric", "units", "max_dets"])
def test_coco_tensors_units_axes_and_summary_cannot_be_replaced_by_pass_flags(tensor_worker, mutant):
    worker, path, precision, recall = tensor_worker
    if mutant == "dtype":
        precision = precision.astype(np.float32)
    elif mutant == "shape":
        precision = precision[:, :, :, :, :2]
    elif mutant == "nan":
        precision[0, 0, 0, 0, 0] = np.nan
    elif mutant == "range":
        recall[0, 0, 0, 0] = 1.5
    elif mutant == "copied_metric":
        worker["coco_secondary"]["metrics"]["AP_small"]["value"] = 99.
    elif mutant == "units":
        worker["coco_secondary"]["metrics"]["AP_small"]["units"] = "fraction"
    elif mutant == "max_dets":
        worker["coco_secondary"]["axes"]["maxDets"] = [1, 10, 300]
    np.savez_compressed(path, precision=precision, recall=recall)
    worker["coco_secondary"]["tensor_artifact"] = file_reference(path)
    with pytest.raises((ValueError, ERROR)):
        gate._tensor_metrics(gate._Reader(), worker)


def test_missing_helper_or_backend_provenance_cannot_be_repacked_as_complete(sealed):
    document, _, directory = sealed
    document["helper_correspondence"].pop(next(iter(document["helper_correspondence"])))
    ref = _write_json(directory / "missing-helper.json", _resign(document))
    with pytest.raises(ERROR, match="helper provenance"):
        gate.verify_replication_qualification_binding(ref)
    document["helper_correspondence"] = {
        name: {"producer": value, "consumer": value}
        for name, value in document["producer_source"]["code_files"].items()
    }
    document["audit_summary"]["backend_references"] = []
    ref = _write_json(directory / "missing-backend.json", _resign(document))
    with pytest.raises(ERROR, match="backend leaf"):
        gate.verify_replication_qualification_binding(ref)


@pytest.mark.parametrize("axis,index", [("maxDets", 0), ("catIds", 0), ("recThrs", 0)])
def test_coco_boolean_axis_entries_are_not_accepted_as_zero_or_one(tensor_worker, axis, index):
    worker = tensor_worker[0]
    worker["coco_secondary"]["axes"][axis][index] = bool(worker["coco_secondary"]["axes"][axis][index])
    with pytest.raises(ERROR, match="builtin integer|finite real"):
        gate._tensor_metrics(gate._Reader(), worker)


def test_named_pipe_is_rejected_without_blocking_or_reading_a_device(directory):
    import os
    path = directory / "not-a-regular-evidence-file"
    os.mkfifo(path)
    reference = {"path": str(path), "sha256": hashlib.sha256(b"").hexdigest(),
                 "size_bytes": 0, "sha256_scope": "complete_file_bytes"}
    with pytest.raises(ERROR, match="not a regular file"):
        gate._Reader().read(reference)


@pytest.mark.parametrize("field", ["commit", "tree", "branch"])
def test_consumer_git_identity_is_reauthenticated_separately_from_producer_source(sealed, field):
    document, _, directory = sealed
    document["consumer_source_identity"][field] = ("f" * 40 if field != "branch" else "changed-branch")
    ref = _write_json(directory / "changed-consumer.json", _resign(document))
    with pytest.raises(ERROR, match="consumer source Git identity changed"):
        gate.verify_replication_qualification_binding(ref)


def test_full_consumer_identity_requires_a_clean_source_snapshot(monkeypatch):
    seen = []
    expected = {"repo_root": "/synthetic", "commit": "a" * 40, "tree": "b" * 40,
                "branch": "synthetic", "inventory_sha256": "c" * 64, "code_files": {}}

    def snapshot(root):
        seen.append(root)
        return expected

    monkeypatch.setattr(gate, "source_snapshot", snapshot)
    assert gate._consumer_source_identity() == {key: value for key, value in expected.items() if key != "code_files"}
    assert seen == [Path(gate.__file__).resolve().parents[3]]


@pytest.fixture
def development_rebinding_fixture(directory, monkeypatch):
    """Synthetic metadata only; the separate real integration uses the validator unmocked."""
    from sparse_rtdetr.baseline import training_v2b_development as development
    producer, consumer = directory / "producer", directory / "consumer"
    relative = "src/metadata_helper.py"
    data = b"synthetic metadata helper, no model\n"
    for root in (producer, consumer):
        (root / "src").mkdir(parents=True)
        (root / relative).write_bytes(data)
    gate_path = consumer / "src/sparse_rtdetr/baseline/training_v2b_replication_gate.py"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(b"# synthetic verifier identity\n")
    monkeypatch.setattr(gate, "__file__", str(gate_path))
    backend = directory / "synthetic_backend.py"
    backend.write_bytes(b"synthetic evaluator identity\n")
    annotation = _write_json(directory / "development_coco.json", {"synthetic": "metadata"})
    manifest = _write_json(directory / "development_manifest.json", {"synthetic": "membership"})
    source = {
        "sources": [{"path": relative, "sha256": hashlib.sha256(data).hexdigest()}],
        "faster_coco_eval_sources": [
            {"path": str(backend), "sha256": file_reference(backend)["sha256"]}],
        "versions": {"synthetic": "1"}, "primary_contract": {"synthetic": "fixed"},
    }
    binding = {
        "repo_root": str(producer), "source": source, "role": "development",
        "annotation": {k: v for k, v in annotation.items() if k != "sha256_scope"},
        "manifest": {k: v for k, v in manifest.items() if k != "sha256_scope"},
        "policy": {"input_size": [640, 640]}, "image_count": 2,
    }
    binding["binding_sha256"] = gate._digest(binding)
    calls = []

    def validate(value, *, verify_files=True):
        expected = gate._digest({k: v for k, v in value.items() if k != "binding_sha256"})
        if value.get("binding_sha256") != expected:
            raise RuntimeError("synthetic metadata digest differs")
        calls.append(verify_files)
        if verify_files:
            assert value["repo_root"] == str(consumer)
        return copy.deepcopy(value)

    monkeypatch.setattr(development, "validate_development_binding", validate)
    monkeypatch.setattr(development, "_source_binding", lambda root: copy.deepcopy(source))
    return {"binding": binding, "producer": producer, "consumer": consumer, "relative": relative,
            "backend": backend, "source": source, "calls": calls, "development": development}


def test_rebinding_keeps_original_and_changes_only_checkout_and_derived_digest(development_rebinding_fixture):
    f = development_rebinding_fixture
    original = copy.deepcopy(f["binding"])
    consumer, provenance = gate.rebind_development_for_consumer(f["binding"])
    assert f["binding"] == original
    assert f["calls"] == [False, True]
    assert consumer["repo_root"] == str(f["consumer"])
    assert consumer["binding_sha256"] != original["binding_sha256"]
    assert provenance["changed_fields"] == ["binding_sha256", "repo_root"]
    assert provenance["producer_binding_sha256"] == original["binding_sha256"]
    assert provenance["consumer_binding_sha256"] == consumer["binding_sha256"]
    assert provenance["original_binding_preserved"] is True
    assert provenance["image_bytes_opened"] is False
    assert provenance["source_correspondence"][0]["producer"]["sha256"] == provenance["source_correspondence"][0]["consumer"]["sha256"]


@pytest.mark.parametrize("where", ["producer", "consumer", "backend"])
def test_rebinding_rejects_changed_helper_or_installed_backend_before_reconstruction(
        development_rebinding_fixture, where):
    f = development_rebinding_fixture
    path = f["backend"] if where == "backend" else f[where] / f["relative"]
    path.write_bytes(b"changed source bytes\n")
    with pytest.raises(ERROR, match="SHA or size differs|not the bound regular file"):
        gate.rebind_development_for_consumer(f["binding"])
    assert f["calls"] == [False]


@pytest.mark.parametrize("field", ["versions", "primary_contract", "sources"])
def test_rebinding_requires_full_source_config_and_version_identity(development_rebinding_fixture, field):
    f = development_rebinding_fixture
    original = copy.deepcopy(f["binding"])
    if field == "sources":
        original["source"][field][0]["sha256"] = "a" * 64
    else:
        original["source"][field]["synthetic"] = "changed"
    original["binding_sha256"] = gate._digest(
        {k: v for k, v in original.items() if k != "binding_sha256"})
    with pytest.raises(ERROR, match="source/config/backend/version"):
        gate.rebind_development_for_consumer(original)
    assert f["calls"] == [False]


def test_rebinding_exports_all_complete_file_leaves_to_a_reader_without_source_paths(development_rebinding_fixture):
    class Recorder:
        def __init__(self):
            self.references = []

        def read(self, reference, *, retain=True):
            assert retain is False
            assert reference["sha256_scope"] == "complete_file_bytes"
            self.references.append(reference)

    recorder = Recorder()
    _, provenance = gate.rebind_development_for_consumer(
        development_rebinding_fixture["binding"], reader=recorder)
    assert len(recorder.references) == 5  # producer/consumer helper, backend, annotation, manifest
    assert set(provenance["metadata_references"]) == {"annotation", "manifest"}


def test_rebinding_rejects_extra_rebuilt_metadata_change(development_rebinding_fixture, monkeypatch):
    f = development_rebinding_fixture
    original_validate = f["development"].validate_development_binding

    def changed(value, *, verify_files=True):
        result = original_validate(value, verify_files=verify_files)
        if verify_files:
            result["image_count"] += 1
            result["binding_sha256"] = gate._digest(
                {k: v for k, v in result.items() if k != "binding_sha256"})
        return result

    monkeypatch.setattr(f["development"], "validate_development_binding", changed)
    with pytest.raises(ERROR, match="beyond checkout"):
        gate.rebind_development_for_consumer(f["binding"])


def test_rebinding_requires_cpu_before_any_metadata_validation(development_rebinding_fixture, monkeypatch):
    f = development_rebinding_fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ERROR, match="empty CUDA_VISIBLE_DEVICES"):
        gate.rebind_development_for_consumer(f["binding"])
    assert f["calls"] == []


def test_rebinding_reauthenticates_source_after_metadata_reconstruction(
        development_rebinding_fixture, monkeypatch):
    f = development_rebinding_fixture
    original_validate = f["development"].validate_development_binding

    def changed(value, *, verify_files=True):
        result = original_validate(value, verify_files=verify_files)
        if verify_files:
            (f["producer"] / f["relative"]).write_bytes(b"source changed during metadata reconstruction\n")
        return result

    monkeypatch.setattr(f["development"], "validate_development_binding", changed)
    with pytest.raises(ERROR, match="SHA or size differs|not the bound regular file"):
        gate.rebind_development_for_consumer(f["binding"])
