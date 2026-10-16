"""Authenticate the fixed development evidence before paired seed replication.

This is a CPU evidence gate, not a GPU capability.  The full audit replays point
metrics and diagnostics, but never reruns bootstrap AP or opens model tensors.
A worker may verify the sealed audit and every leaf hash without rebuilding
COCO/primary caches inside the native monitor's startup deadline.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import gzip
import hashlib
import io
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

from . import training_v2b_resolution_diagnostics as diagnostic
from .training_v2b_campaign import (
    CampaignError, _canonical, _git, _write_json, source_snapshot, validate_frozen_source,
)
from .training_v2b_evidence import (
    canonical_sha256 as evidence_digest, strict_json_loads,
)
from .training_v2b_evidence_campaign import (
    AUTHORIZATION_SHA256, CELL_SIZES, cross_worker_contract,
    validate_evidence_campaign,
)

KIND = "v2b_seed_resolution_replication_qualification"
CELLS = tuple(CELL_SIZES)
PRIMARY_METRICS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
COCO_POINT_METRICS = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR100", "AR_small", "AR_medium", "AR_large",
)
BOOTSTRAP_METRICS = (
    "coco_AP", "coco_AP50", "coco_AP75", "coco_AP_small", "coco_AR_small", "primary_AP",
)
REFERENCE_KEYS = {"path", "sha256", "size_bytes", "sha256_scope"}
ALLOWED_EXPERIMENTS = tuple(
    {"seed": seed, "input_size": size, "physical_batch_size": 8,
     "accumulation_steps": 2, "logical_batch_size": 16, "epochs": 30}
    for seed in (1, 2) for size in (640, 896)
)
_HELPERS = tuple("src/sparse_rtdetr/baseline/" + name + ".py" for name in (
    "training_v2b_campaign", "training_v2b_evidence_campaign",
    "training_v2b_resolution_analysis", "training_v2b_resolution_diagnostics",
    "training_v2b_official_gt", "training_v2b_primary_cache",
    "training_v2b_development", "training_v2b_admission", "training_v2b_evidence",
    "training_v2b_cross_eval", "primary_evaluator",
)) + tuple("src/sparse_rtdetr/data_protocol/" + name + ".py"
           for name in ("evaluation", "parser", "schema"))
_QUALIFICATION_KEYS = {
    "schema_version", "kind", "status", "eligible_for_fixed_paired_seed1_and_seed2",
    "analysis_reference", "campaign_reference", "campaign_result_reference",
    "conditional_authorization_reference", "method_sha256", "allowed_experiments",
    "gpu_admission_granted", "requires_fresh_native_admission", "fixture_only",
    "producer_source", "consumer_source_identity", "verifier_reference", "helper_correspondence",
    "audit_summary", "leaf_references", "leaf_inventory_sha256",
    "qualification_sha256",
}


class ReplicationQualificationError(CampaignError):
    """Unbound, incomplete, altered or scientifically ineligible evidence."""


def _require(condition, message):
    if not condition:
        raise ReplicationQualificationError(message)


def _same(actual, expected, message):
    _require(_canonical(actual) == _canonical(expected), message)


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value, name, *, lower=None, upper=None):
    _require(type(value) in (int, float) and math.isfinite(value),
             name + " must be a finite real number, not a boolean")
    _require(lower is None or value >= lower, name + " is below its allowed range")
    _require(upper is None or value <= upper, name + " exceeds its allowed range")
    return float(value)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name + " must be a builtin integer")
    return value


def _json(raw):
    value = strict_json_loads(raw)

    def finite(item):
        if type(item) is float:
            _require(math.isfinite(item), "JSON contains a nonfinite number")
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)

    finite(value)
    return value


def _reference(value, *, embedded=False):
    keys = REFERENCE_KEYS - {"sha256_scope"} if embedded else REFERENCE_KEYS
    _require(type(value) is dict and set(value) in (keys, REFERENCE_KEYS),
             "complete file reference schema differs")
    result = copy.deepcopy(value)
    result.setdefault("sha256_scope", "complete_file_bytes")
    _require(result["sha256_scope"] == "complete_file_bytes",
             "file reference does not cover complete bytes")
    _require(type(result["path"]) is str, "reference path must be a string")
    path = Path(result["path"])
    _require(path.is_absolute() and ".." not in path.parts and str(path) == result["path"],
             "reference path must be absolute and canonical")
    _require(type(result["sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", result["sha256"]) is not None,
             "reference SHA256 is malformed")
    _integer(result["size_bytes"], "reference size")
    return result


def _path_role(path, source_paths):
    if str(path) in source_paths:
        return
    forbidden = {"test", "confirmatory", "visdrone2019-det-test-dev",
                 "visdrone2019-det-test-challenge", "train_core", "train"}
    _require(not any(part.casefold() in forbidden
                     or part.casefold().startswith(("test_", "confirmatory_"))
                     for part in path.parts),
             "evidence reference crosses a protected data role")


class _Reader:
    """Non-following, streaming, complete-file authentication with a leaf ledger."""

    def __init__(self):
        self.references = {}
        self.source_paths = set()
        self._small_bytes = {}

    def read(self, reference, *, embedded=False, retain=True):
        ref = _reference(reference, embedded=embedded)
        path = Path(ref["path"])
        _path_role(path, self.source_paths)
        previous = self.references.get(ref["path"])
        if previous is not None:
            _same(ref, previous, "one evidence path has conflicting identities")
            if not retain:
                return None
            if ref["path"] in self._small_bytes:
                return self._small_bytes[ref["path"]]
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parent.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            leaf = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            _require(stat.S_ISREG(leaf.st_mode), "evidence reference is not a regular file")
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
        finally:
            os.close(directory)
        hasher, chunks, length = hashlib.sha256(), [], 0
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size == ref["size_bytes"],
                     "evidence reference is not the bound regular file")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                length += len(chunk)
                if retain:
                    chunks.append(chunk)
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(all(getattr(before, k) == getattr(after, k) for k in fields),
                 "evidence changed during authentication")
        _require(length == ref["size_bytes"] and hasher.hexdigest() == ref["sha256"],
                 "evidence full-file SHA or size differs: " + path.name)
        self.references[ref["path"]] = ref
        if not retain:
            return None
        raw = b"".join(chunks)
        if len(raw) <= 4 * 1024 * 1024:
            self._small_bytes[ref["path"]] = raw
        return raw

    def json(self, reference, *, embedded=False):
        return _json(self.read(reference, embedded=embedded))

    def visit(self, value):
        if type(value) is dict:
            if (set(value) in (REFERENCE_KEYS, REFERENCE_KEYS - {"sha256_scope"})
                    and value.get("sha256_scope", "complete_file_bytes") == "complete_file_bytes"):
                self.read(value, embedded=True, retain=False)
            else:
                for child in value.values():
                    self.visit(child)
        elif type(value) is list:
            for child in value:
                self.visit(child)

    def inventory(self):
        return [self.references[key] for key in sorted(self.references)]


def _owned(reference, directory, name=None):
    ref = _reference(reference)
    path, root = Path(ref["path"]), Path(directory)
    _require(path.is_relative_to(root) and path != root
             and (name is None or path.name == name),
             "evidence artifact escaped its declared owner")
    return ref


def _source_correspondence(reader, source):
    root = Path(__file__).resolve().parents[3]
    correspondences = {}
    for name in _HELPERS:
        _require(name in source["code_files"], "recomputation helper absent from producer source")
        producer = _reference(source["code_files"][name])
        current_path = root / name
        consumer = {**producer, "path": str(current_path)}
        reader.source_paths.update((producer["path"], consumer["path"]))
        reader.read(producer, retain=False)
        reader.read(consumer, retain=False)
        correspondences[name] = {"producer": producer, "consumer": consumer}
    return correspondences


def _register_evaluator_source_proofs(reader, evidence, producer_source):
    """Authenticate explicit cross-checkout certification source files.

    Only a source proof with the same relative name, full SHA and size as the
    authenticated producer inventory can acquire a per-file source role.  A
    direct tests/test_*.py leaf is engineering source; its ancestors still pass
    the ordinary role guard.  No directory or arbitrary test-prefixed file is
    exempted, and the unchanged reader authenticates both physical copies.
    """
    _require(evidence.get("status") == "PASS"
             and evidence.get("certification_record_verified") is True
             and evidence.get("source_match") is True,
             "evaluator certification source evidence is not verified")
    proofs, inputs = evidence.get("source_proofs"), evidence.get("inputs")
    _require(type(proofs) is list and proofs and type(inputs) is dict,
             "evaluator certification source proofs are missing")
    roots = []
    for value in (evidence.get("repo_root"), producer_source.get("repo_root")):
        _require(type(value) is str, "evaluator source checkout must be a string")
        root = Path(value)
        _path_role(root, set())
        _require(root.is_absolute() and ".." not in root.parts and str(root) == value
                 and root.resolve(strict=True) == root and root.is_dir(),
                 "evaluator source checkout is not canonical")
        roots.append(root)
    certificate_root, producer_root = roots
    code_files = producer_source.get("code_files")
    _require(type(code_files) is dict and code_files,
             "evaluator producer source inventory is missing")
    forbidden = {"artifacts", "data", "train_core", "train", "test", "confirmatory",
                 "visdrone2019-det-test-dev", "visdrone2019-det-test-challenge"}
    correspondences = {}
    # Validate every descriptor before opening any of the declared source leaves.
    for row in proofs:
        _require(type(row) is dict and set(row) == {
            "relative_path", "reference", "matches_formal_full_audit_frozen_input"}
            and row["matches_formal_full_audit_frozen_input"] is True,
            "evaluator certification source proof schema differs")
        name = row["relative_path"]
        _require(type(name) is str, "evaluator source relative path must be a string")
        relative = Path(name)
        _require(not relative.is_absolute() and relative.parts and ".." not in relative.parts
                 and str(relative) == name and name not in correspondences,
                 "evaluator source relative path is noncanonical or duplicated")
        _require(not any(part.casefold() in forbidden
                         or part.casefold().startswith("confirmatory_")
                         for part in relative.parts),
                 "evaluator source proof crosses a protected data role")
        engineering_test = (len(relative.parts) == 2 and relative.parts[0] == "tests"
                            and relative.name.startswith("test_") and relative.suffix == ".py")
        ordinary_source = ((relative.parts[0] == "src" and relative.suffix == ".py")
                           or (relative.parts[0] == "docs" and relative.suffix == ".md"))
        _require(engineering_test or ordinary_source,
                 "evaluator source proof is not an authorized source file")
        _path_role(certificate_root / (relative.parent if engineering_test else relative), set())
        _path_role(producer_root / (relative.parent if engineering_test else relative), set())
        historical = _reference(row["reference"])
        _same(historical["path"], str(certificate_root / relative),
              "evaluator source proof path differs from its checkout")
        _same(_reference(inputs.get(historical["path"])), historical,
              "evaluator source proof differs from certification inputs")
        producer = _reference(code_files.get(name))
        _same(producer["path"], str(producer_root / relative),
              "evaluator producer source path differs from its checkout")
        _same((historical["sha256"], historical["size_bytes"]),
              (producer["sha256"], producer["size_bytes"]),
              "evaluator source proof differs from authenticated producer bytes")
        _require(producer["path"] in reader.source_paths,
                 "evaluator producer source was not classified by its inventory")
        correspondences[name] = {"producer": producer, "historical": historical}
    for pair in correspondences.values():
        reader.read(pair["producer"], retain=False)
        reader.source_paths.add(pair["historical"]["path"])
        reader.read(pair["historical"], retain=False)
    return correspondences


def rebind_development_for_consumer(producer_binding, *, reader=None):
    """Derive this checkout's metadata binding without changing producer evidence.

    Full source/config/backend identities are proved before replacing only the
    checkout path and its derived digest.  The original development validator
    then rebuilds the consumer binding from the same metadata, without images.
    This helper is CPU-only; fast worker verification never calls it.
    """
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "development provenance rebinding requires empty CUDA_VISIBLE_DEVICES")
    from . import training_v2b_development as development

    original = development.validate_development_binding(producer_binding, verify_files=False)
    root = Path(__file__).resolve().parents[3]
    producer_root = Path(original["repo_root"])
    _require(producer_root.is_absolute() and ".." not in producer_root.parts
             and producer_root.resolve(strict=True) == producer_root,
             "producer development checkout is not canonical")
    _path_role(producer_root, set())
    current_source = development._source_binding(root)
    _same(current_source, original["source"],
          "development source/config/backend/version identity differs before rebinding")

    proof = _Reader()
    correspondences, installed = [], []
    source_rows = current_source["sources"]
    _require(type(source_rows) is list and source_rows,
             "development source correspondence is empty")
    for row in source_rows:
        relative = Path(row["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and str(relative) == row["path"],
                 "development helper path is not checkout-relative")
        old_path, new_path = producer_root / relative, root / relative
        producer = _reference({
            "path": str(old_path), "sha256": row["sha256"],
            "size_bytes": old_path.lstat().st_size, "sha256_scope": "complete_file_bytes"})
        consumer = {**producer, "path": str(new_path)}
        proof.source_paths.update((producer["path"], consumer["path"]))
        proof.read(producer, retain=False)
        proof.read(consumer, retain=False)
        correspondences.append({"relative_path": row["path"],
                                "producer": producer, "consumer": consumer})
    for row in current_source["faster_coco_eval_sources"]:
        path = Path(row["path"])
        reference = _reference({
            "path": str(path), "sha256": row["sha256"],
            "size_bytes": path.lstat().st_size, "sha256_scope": "complete_file_bytes"})
        proof.source_paths.add(reference["path"])
        proof.read(reference, retain=False)
        installed.append(reference)
    metadata = {key: _reference(original[key], embedded=True)
                for key in ("annotation", "manifest")}
    for reference in metadata.values():
        proof.read(reference, retain=False)

    candidate = copy.deepcopy(original)
    candidate["repo_root"] = str(root)
    candidate["binding_sha256"] = evidence_digest(
        {key: value for key, value in candidate.items() if key != "binding_sha256"})
    consumer = development.validate_development_binding(candidate, verify_files=True)
    restored = copy.deepcopy(consumer)
    restored["repo_root"] = original["repo_root"]
    restored["binding_sha256"] = evidence_digest(
        {key: value for key, value in restored.items() if key != "binding_sha256"})
    _same(restored, original,
          "development reconstruction differs beyond checkout and derived binding digest")
    _same(producer_binding, original, "producer development evidence changed while rebinding")

    # Reauthenticate after metadata reconstruction and export complete leaves to
    # either the qualification reader or a summary reader with the same read API.
    final_proof = _Reader()
    final_proof.source_paths.update(proof.source_paths)
    for reference in proof.inventory():
        final_proof.read(reference, retain=False)
        if reader is not None:
            if hasattr(reader, "source_paths"):
                reader.source_paths.update(proof.source_paths)
            reader.read(reference, retain=False)
    changed = [key for key in original if original[key] != consumer[key]]
    _require(set(changed).issubset({"repo_root", "binding_sha256"}),
             "development rebinding changed scientific fields")
    provenance = {
        "kind": "development_consumer_provenance_rebinding",
        "producer_repo_root": original["repo_root"], "consumer_repo_root": consumer["repo_root"],
        "producer_binding_sha256": original["binding_sha256"],
        "consumer_binding_sha256": consumer["binding_sha256"],
        "source_binding_sha256": evidence_digest(current_source),
        "changed_fields": sorted(changed), "source_correspondence": correspondences,
        "installed_backend_references": installed, "metadata_references": metadata,
        "original_binding_preserved": True, "consumer_metadata_rebuilt_and_verified": True,
        "image_bytes_opened": False, "gpu_work_launched": False,
    }
    return copy.deepcopy(consumer), provenance


def _consumer_source_identity():
    """Bind the clean committed verifier checkout without equating it with N2."""
    source = source_snapshot(Path(__file__).resolve().parents[3])
    return {key: source[key] for key in
            ("repo_root", "commit", "tree", "branch", "inventory_sha256")}


def _verify_consumer_git(identity):
    expected_keys = {"repo_root", "commit", "tree", "branch", "inventory_sha256"}
    _require(type(identity) is dict and set(identity) == expected_keys,
             "consumer source identity schema differs")
    root = Path(__file__).resolve().parents[3]
    _same(identity["repo_root"], str(root), "consumer source checkout differs")
    for field, length in (("commit", 40), ("tree", 40), ("inventory_sha256", 64)):
        _require(type(identity[field]) is str
                 and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", identity[field]) is not None,
                 "consumer source Git or inventory identity is malformed")
    for field, args in (("commit", ("rev-parse", "HEAD")),
                        ("tree", ("rev-parse", "HEAD^{tree}")),
                        ("branch", ("branch", "--show-current"))):
        _same(_git(root, *args), identity[field], "consumer source Git identity changed: " + field)


def _load_context(reader, analysis_reference):
    from .training_v2b_resolution_analysis import _load_completed_cells

    analysis = reader.json(analysis_reference)
    _require(type(analysis) is dict and type(analysis.get("schema_version")) is int
             and analysis["schema_version"] == 1 and analysis.get("status") == "PASS",
             "resolution analysis did not complete")
    for name, expected in (
        ("primary_points_from_full_evaluator", True), ("faster_coco_points_replayed", True),
        ("gpu_work_launched", False), ("training_launched", False),
        ("confirmatory_or_test_accessed", False),
    ):
        _require(analysis.get(name) is expected, "analysis scope or evaluator evidence differs: " + name)
    _same(analysis.get("method"), diagnostic.frozen_method(), "frozen analysis method differs")
    _require(set(analysis.get("cells", {})) == set(CELLS), "analysis must contain all four cells")
    campaign_ref, result_ref = analysis["campaign_reference"], analysis["campaign_result_reference"]
    campaign, result = reader.json(campaign_ref), reader.json(result_ref)
    source = campaign["source"]
    _require(type(source.get("code_files")) is dict and source["code_files"],
             "producer source inventory is missing")
    _path_role(Path(source["repo_root"]), set())
    for name, ref in source["code_files"].items():
        _require(not any(part.casefold() in {"artifacts", "test", "confirmatory",
                         "visdrone2019-det-test-dev", "visdrone2019-det-test-challenge"}
                         for part in Path(name).parts), "source inventory crosses a data role")
        _require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
                 and ref["path"] == str(Path(source["repo_root"]) / name),
                 "producer source inventory path differs")
        reader.source_paths.add(ref["path"])
    reader.visit(campaign)
    _require(campaign["authorization_reference"]["sha256"] == AUTHORIZATION_SHA256,
             "paired seed authority is not the authenticated user authorization")
    validate_evidence_campaign(campaign, verify_files=False)
    validate_frozen_source(source)
    helpers = _source_correspondence(reader, source)
    _owned(analysis_reference, campaign["output_root"], "analysis-result.json")
    _owned(campaign_ref, campaign["output_root"], "campaign.json")
    _owned(result_ref, Path(campaign["output_root"]) / "execution", "campaign-result.json")
    foundation = reader.json(campaign["foundation_reference"])
    _same(foundation.get("predeclared_additional_seeds"), [1, 2],
          "foundation did not predeclare both replication seeds")
    if "user_authorization" in foundation:
        _same(foundation["user_authorization"], campaign["authorization_reference"],
              "foundation authorization differs")
    reader.visit(foundation)
    evaluator_evidence = reader.json(campaign["evaluator_evidence_reference"])
    _require(evaluator_evidence.get("status") == "PASS"
             and evaluator_evidence.get("certification_record_verified") is True
             and evaluator_evidence.get("source_match") is True,
             "existing primary evaluator certification evidence is not closed")
    _register_evaluator_source_proofs(reader, evaluator_evidence, source)
    reader.visit(evaluator_evidence)
    workers, completions = _load_completed_cells(campaign, campaign_ref, result_ref)
    for cell in CELLS:
        item, row, worker = result["completed"][cell], analysis["cells"][cell], workers[cell]
        contract = cross_worker_contract(campaign, cell)
        cell_root = Path(contract["output_dir"])
        expected_paths = {
            "contract_reference": Path(campaign["output_root"]) / "execution/contracts" / (cell + ".json"),
            "result_reference": cell_root / "worker-result.json",
            "completion_reference": Path(campaign["output_root"]) / "execution" / (cell + ".complete.json"),
        }
        for name, expected_path in expected_paths.items():
            _same(item[name]["path"], str(expected_path), "cell terminal artifact owner differs")
            reader.read(item[name], retain=False)
        for name, expected in (
            ("cell_key", cell), ("source_training_size", CELL_SIZES[cell][0]),
            ("evaluation_size", CELL_SIZES[cell][1]), ("campaign_id", campaign["campaign_id"]),
        ):
            _same(worker.get(name), expected, "worker cell identity differs: " + name)
        _same(row.get("worker_result"), item["result_reference"], "analysis worker reference swapped")
        for name in ("prediction_artifact", "raw_query_artifact", "primary_official_gt",
                     "primary_legacy_formal_gt", "coco_secondary", "memory",
                     "inference_elapsed_seconds", "execution_elapsed_seconds"):
            _same(row.get(name), worker.get(name), "analysis copied worker evidence differs: " + name)
        _same(worker.get("full_development_coverage"), {
            "images": 548, "predictions": 548 * 300,
            "image_order_sha256": contract["development_binding"]["image_order_sha256"],
        }, "worker development coverage differs")
        for name in ("prediction_artifact", "raw_query_artifact", "binding_reference",
                     "cpu_preparation_reference", "official_gt_binding_reference",
                     "diagnostic_attributes_reference", "image_receipts_reference",
                     "batch_timings_reference"):
            _owned(worker[name], cell_root)
        for name in ("errors", "raw_query_coverage"):
            _owned(row[name], Path(analysis_reference["path"]).parent,
                   cell + (".errors.json" if name == "errors" else ".raw-query-coverage.json"))
        reader.visit(worker)
        reader.visit(completions[cell])
        monitor = completions[cell]["monitor"]
        _require(monitor.get("status") == "PASS" and monitor.get("worker_exited") is True
                 and monitor.get("sampled_clock_compliance") is True
                 and monitor.get("settings_after_exit") == "keep_1500_1500",
                 "native monitor has no successful complete workload closure")
        _integer(monitor.get("loaded_clock_samples"), "loaded native clock samples", 3)
        _integer(monitor.get("post_exit_health_observations"), "post-exit native health observations", 1)
        shutdown = monitor.get("sampler_shutdown", {})
        _same(shutdown.get("sample_errors"), [], "native sampler errors were recorded")
        _same(shutdown.get("unfinished_threads"), [], "native sampling threads did not close")
        binding = reader.json(worker["binding_reference"])
        reader.visit(binding)
        _same(binding.get("binding_sha256"), monitor["run_binding_sha256"],
              "native worker run-binding SHA differs")
        for role in ("launch_reference",):
            reader.visit(reader.json(completions[cell][role]))
    analysis_names = {
        "paired_bootstrap": "paired-bootstrap.json",
        "diagonal_routing_oracle": "diagonal.routing-oracle.json",
        "same_training640_routing_oracle": "same_training640.routing-oracle.json",
        "progress_reference": "progress.jsonl",
    }
    for name, filename in analysis_names.items():
        _owned(analysis[name], Path(analysis_reference["path"]).parent, filename)
        reader.read(analysis[name], retain=False)
    return {"analysis": analysis, "campaign": campaign, "result": result,
            "workers": workers, "completions": completions, "helpers": helpers}


def _point_check(saved, calculated, reference, names):
    from .training_v2b_resolution_analysis import compare_point_metrics
    expected = compare_point_metrics(calculated, reference, names)
    _same(saved, expected, "saved full evaluator/cache point arithmetic differs")
    return expected


def _tensor_metrics(reader, worker):
    from types import SimpleNamespace
    from .training_v2b_development import _coco_metrics

    report = worker["coco_secondary"]
    with np.load(io.BytesIO(reader.read(report["tensor_artifact"])), allow_pickle=False) as archive:
        _require(set(archive.files) == {"precision", "recall"}, "COCO tensor archive schema differs")
        precision, recall = archive["precision"].copy(), archive["recall"].copy()
    _require(precision.dtype == np.float64 and recall.dtype == np.float64,
             "COCO tensor dtype differs")
    axes = report["axes"]
    _integer(axes.get("useCats"), "COCO useCats", 1)
    for name in ("catIds", "maxDets"):
        _require(type(axes.get(name)) is list, "COCO integer axis must be a list")
        for value in axes[name]:
            _integer(value, "COCO " + name, 1)
    for name in ("iouThrs", "recThrs"):
        _require(type(axes.get(name)) is list, "COCO threshold axis must be a list")
        for value in axes[name]:
            _number(value, "COCO " + name, lower=0, upper=1)
    for pair in axes.get("areaRng", []):
        _require(type(pair) is list and len(pair) == 2, "COCO area range shape differs")
        for value in pair:
            _number(value, "COCO area range", lower=0)
    params = SimpleNamespace(**{k: axes[k] for k in
                             ("iouType", "useCats", "iouThrs", "recThrs", "catIds",
                              "areaRng", "areaRngLbl", "maxDets")})
    metrics, checked_axes, _, _ = _coco_metrics(
        SimpleNamespace(params=params, eval={"precision": precision, "recall": recall}))
    _same(axes, checked_axes, "full COCO axes differ")
    _same(report["metrics"], metrics, "COCO report metrics do not reproduce its tensors")
    for name in COCO_POINT_METRICS:
        _number(metrics[name]["value"], "COCO " + name, lower=0, upper=100)
    return {key: value["value"] for key, value in metrics.items()}


def _audit_bootstrap(bootstrap, manifest_records, image_ids, point_cells, cache_identities):
    """Verify saved distributions and RNG draws; do not repeat AP resampling."""
    method, policy = diagnostic.frozen_method(), diagnostic.FROZEN_POLICY
    _require(type(bootstrap.get("schema_version")) is int and bootstrap["schema_version"] == 1
             and bootstrap.get("fixture_only") is False, "fixture bootstrap cannot qualify replication")
    for key, expected in (("method", method), ("method_sha256", _digest(method)),
                          ("effective_policy", asdict(policy)), ("cells", list(CELLS)),
                          ("metrics", list(BOOTSTRAP_METRICS)),
                          ("contrasts", list(diagnostic.CONTRASTS)),
                          ("point_cells", point_cells), ("cache_identities", cache_identities)):
        if key == "point_cells":
            _require(set(bootstrap.get(key, {})) == set(CELLS), "bootstrap cell schema differs")
            for cell in CELLS:
                _require(set(bootstrap[key][cell]) == set(BOOTSTRAP_METRICS),
                         "bootstrap point metric schema differs")
                for metric in BOOTSTRAP_METRICS:
                    actual = _number(bootstrap[key][cell][metric], "bootstrap point", lower=0, upper=100)
                    _require(math.isclose(actual, point_cells[cell][metric], rel_tol=0., abs_tol=1e-10),
                             "bootstrap point is not the verified full evaluator/cache point")
        else:
            _same(bootstrap.get(key), expected, "bootstrap frozen identity differs: " + key)
    clusters = diagnostic.build_development_clusters(manifest_records, image_ids)
    _require(len(image_ids) == policy.expected_images and len(clusters) == policy.expected_clusters,
             "bootstrap development cluster cardinality differs")
    _same(bootstrap.get("clusters"), [list(x) for x in clusters], "bootstrap clusters differ")
    _same(bootstrap.get("manifest_canonical_sha256"), _digest(manifest_records),
          "bootstrap manifest identity differs")
    output = bootstrap.get("resampling")
    _require(type(output) is dict and set(output) == {"cluster", "image_sensitivity"},
             "both declared bootstrap distributions must be present")
    point = np.asarray([[point_cells[cell][metric] for metric in BOOTSTRAP_METRICS] for cell in CELLS])
    coefficients = np.asarray(list(diagnostic.CONTRASTS.values()))
    contrasts = coefficients @ point
    summary = {}
    for kind, units, count, seed in (
        ("cluster", clusters, policy.cluster_replicates, policy.rng_seed),
        ("image_sensitivity", tuple((i,) for i in image_ids), policy.image_replicates, policy.rng_seed + 1),
    ):
        row = output[kind]
        for key, expected in (("replicates", count), ("unit_count", len(units)), ("seed", seed),
                              ("distribution_axes", ["replicate", "contrast", "metric"])):
            _same(row.get(key), expected, "bootstrap resampling definition differs: " + key)
        raw = row.get("distribution")
        _require(type(raw) is list and len(raw) == count, "bootstrap distribution length differs")
        for replicate in raw:
            _require(type(replicate) is list and len(replicate) == len(diagnostic.CONTRASTS),
                     "bootstrap contrast axis differs")
            for metrics in replicate:
                _require(type(metrics) is list and len(metrics) == len(BOOTSTRAP_METRICS),
                         "bootstrap metric axis differs")
                for value in metrics:
                    _number(value, "bootstrap distribution entry", lower=-200, upper=200)
        values = np.asarray(raw, dtype=np.float64)
        draws = np.random.Generator(np.random.PCG64(seed)).integers(
            0, len(units), size=(count, len(units)), dtype=np.int64)
        draws_sha = hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
        distribution_sha = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
        _same(row.get("draws_sha256"), draws_sha, "bootstrap PCG64 draws SHA differs")
        _same(row.get("distribution_sha256"), distribution_sha, "bootstrap distribution SHA differs")
        ci = np.quantile(values, [.025, .975], axis=0, method="linear")
        _require(set(row.get("contrasts", {})) == set(diagnostic.CONTRASTS),
                 "bootstrap reported contrast set differs")
        for j, contrast in enumerate(diagnostic.CONTRASTS):
            _require(set(row["contrasts"][contrast]) == set(BOOTSTRAP_METRICS),
                     "bootstrap reported metric set differs")
            for k, metric in enumerate(BOOTSTRAP_METRICS):
                item = row["contrasts"][contrast][metric]
                _require(set(item) == {"point_pp", "ci95_pp"}, "bootstrap contrast value schema differs")
                actual = _number(item["point_pp"], "bootstrap contrast point", lower=-200, upper=200)
                _require(math.isclose(actual, float(contrasts[j, k]), rel_tol=0., abs_tol=1e-10),
                         "bootstrap contrast point differs from verified four cells")
                _require(type(item["ci95_pp"]) is list and len(item["ci95_pp"]) == 2,
                         "bootstrap CI must contain two endpoints")
                for endpoint in item["ci95_pp"]:
                    _number(endpoint, "bootstrap CI endpoint", lower=-200, upper=200)
                _same(item["ci95_pp"], [float(ci[0, j, k]), float(ci[1, j, k])],
                      "bootstrap percentile CI does not reproduce the saved distribution")
        summary[kind] = {
            "replicates": count, "unit_count": len(units), "seed": seed,
            "draws_sha256": draws_sha, "distribution_sha256": distribution_sha,
            "contrasts": copy.deepcopy(row["contrasts"]),
        }
    return summary


def _conditions(inputs):
    _require(type(inputs) is dict and set(inputs) == {
        "primary_ap_percent", "coco_ap_small_percent", "small_gt_count",
        "small_unmatched_iou50", "cluster_diagonal_small_ci95_pp",
    }, "scientific threshold witness schema differs")
    values = {}
    for name in ("primary_ap_percent", "coco_ap_small_percent", "small_gt_count", "small_unmatched_iou50"):
        _require(type(inputs[name]) is dict and set(inputs[name]) == set(CELLS),
                 "threshold witness must cover the exact four cells")
    for cell in CELLS:
        denominator = _integer(inputs["small_gt_count"][cell], "small GT denominator", 1)
        unmatched = _integer(inputs["small_unmatched_iou50"][cell], "small unmatched count")
        _require(unmatched <= denominator, "small unmatched count exceeds the denominator")
        for name in ("primary_ap_percent", "coco_ap_small_percent"):
            _number(inputs[name][cell], name, lower=0, upper=100)
    _require(len(set(inputs["small_gt_count"].values())) == 1, "small-GT denominator differs across cells")
    ci = inputs["cluster_diagonal_small_ci95_pp"]
    _require(type(ci) is list and len(ci) == 2, "cluster small CI shape differs")
    lower, upper = (_number(x, "cluster small CI", lower=-100, upper=100) for x in ci)
    _require(lower <= upper, "cluster small CI endpoints are reversed")
    for key, source in (("primary_diagonal_delta_pp", "primary_ap_percent"),
                        ("coco_small_diagonal_delta_pp", "coco_ap_small_percent")):
        values[key] = float(inputs[source]["t896_e896"] - inputs[source]["t640_e640"])
    values["cluster_small_ci95_pp"] = [lower, upper]
    values["small_unmatched_before"] = inputs["small_unmatched_iou50"]["t640_e640"]
    values["small_unmatched_after"] = inputs["small_unmatched_iou50"]["t896_e896"]
    values["small_gt_denominator"] = inputs["small_gt_count"]["t640_e640"]
    checks = {
        "uniform_primary_diagonal_positive": values["primary_diagonal_delta_pp"] > 0.,
        "coco_small_diagonal_positive": values["coco_small_diagonal_delta_pp"] > 0.,
        "cluster_small_ci_lower_positive": lower > 0.,
        "small_unmatched_at_iou50_decreases": values["small_unmatched_after"] < values["small_unmatched_before"],
    }
    return {"inputs": copy.deepcopy(inputs), "values": values, "checks": checks,
            "eligible_for_fixed_paired_seed1_and_seed2": all(checks.values())}


def _audit_native_samples(reader, monitor):
    """Authenticate persisted sampled compliance; no fresh hardware query occurs."""
    raw = reader.read(monitor["samples"])
    previous, count, clocks, maximum = None, 0, 0, 0.
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        for line in stream:
            row = _json(line)
            _require(set(row) == {"sequence", "previous_sha256", "record", "sha256"},
                     "native sample chain schema differs")
            _same(row["sequence"], count, "native sample sequence differs")
            _same(row["previous_sha256"], previous, "native sample chain is discontinuous")
            body = {k: v for k, v in row.items() if k != "sha256"}
            _same(row["sha256"], evidence_digest(body), "native sample chain hash differs")
            record = row["record"]
            if record.get("kind") == "clock":
                for key in ("graphics_clock_mhz", "sm_clock_mhz"):
                    clock = _number(record.get(key), "native " + key, lower=0, upper=1500)
                    maximum = max(maximum, clock)
                clocks += 1
            elif record.get("kind") == "health":
                _number(record.get("gpu", {}).get("current_graphics_clock_mhz"),
                        "native health graphics clock", lower=0, upper=1500)
                health = record.get("kernel_health", {})
                _require(health.get("readable") is True and health.get("findings") == [],
                         "native sample recorded unreadable or newly faulty kernel health")
            else:
                raise ReplicationQualificationError("native sample kind differs")
            previous, count = row["sha256"], count + 1
    _same(monitor.get("sample_hash_chain"), {"records": count, "last_sha256": previous},
          "native monitor final chain closure differs")
    _require(clocks >= 3, "native monitor lacks complete clock evidence")
    return {"records": count, "clock_samples": clocks, "maximum_sampled_mhz": maximum,
            "last_sha256": previous, "status": "PASS"}


def _progress(reader, analysis, bootstrap):
    rows = [_json(line) for line in reader.read(analysis["progress_reference"]).splitlines()]
    _require(rows and rows[0].get("stage") == "authorized_metadata_loaded"
             and rows[-1].get("stage") == "finished", "analysis progress lacks terminal closure")
    elapsed = [_number(row.get("elapsed_seconds"), "progress elapsed", lower=0) for row in rows]
    _require(elapsed == sorted(elapsed), "analysis progress clock went backwards")
    for stage in ("reference_points_reproduced", "cell_diagnostics_completed"):
        cells = [row.get("cell") for row in rows if row.get("stage") == stage]
        _same(cells, list(CELLS), "analysis progress cell order or completion differs")
    _require(sum(row.get("stage") == "routing_completed" for row in rows) == 1,
             "routing progress did not close exactly once")
    for kind, expected in (("cluster", 2000), ("image_sensitivity", 200)):
        selected = [row for row in rows if row.get("stage") == "bootstrap" and row.get("kind") == kind]
        counts = [1, *range(25, expected + 1, 25)]
        _same([row.get("completed") for row in selected], counts, "bootstrap progress is incomplete")
        _require(all(type(row.get("total")) is int and row["total"] == expected for row in selected),
                 "bootstrap progress total differs")
    _require(not any(row.get("stage") == "failed" for row in rows), "analysis recorded a failure")
    _same(rows[-1].get("seed_gate"), bootstrap["seed_gate"]["eligible_for_fixed_paired_seed1_and_seed2"],
          "terminal progress eligibility differs")
    return {"events": len(rows), "terminal_elapsed_seconds": elapsed[-1], "status": "PASS"}


def _full_audit(analysis_reference):
    from .training_v2b_official_gt import _make_binding, with_predictions
    from .training_v2b_primary_cache import prepare_primary_cache
    from .training_v2b_resolution_analysis import load_raw_queries

    reader = _Reader()
    consumer_identity = _consumer_source_identity()
    context = _load_context(reader, analysis_reference)
    context["consumer_source_identity"] = consumer_identity
    analysis, campaign = context["analysis"], context["campaign"]
    dev = campaign["development_bindings"]["640"]
    coco, manifest = reader.json(dev["annotation"], embedded=True), reader.json(dev["manifest"], embedded=True)
    lineage_ref = campaign["official_gt_bindings"]["640"]["lineage"]
    lineage = [_json(row) for row in reader.read(lineage_ref, embedded=True).splitlines()]
    image_ids = tuple(sorted(row["id"] for row in coco["images"]))
    _require(len(image_ids) == 548 and len(coco["annotations"]) == 38759,
             "development cardinality differs from the authorized dataset")
    original_binding = campaign["official_gt_bindings"]["640"]
    if original_binding["raw_annotation_root"] is not None:
        for record in manifest["records"]:
            relative = Path(record["annotation_relative_path"])
            _require(not relative.is_absolute() and ".." not in relative.parts
                     and relative.parts[:2] == ("val", "annotations"),
                     "raw annotation reference crosses the development role")
            reader.read({"path": str(Path(original_binding["raw_annotation_root"]) / relative),
                         "sha256": record["annotation_sha256"],
                         "size_bytes": record["annotation_size_bytes"],
                         "sha256_scope": "complete_file_bytes"}, retain=False)
    consumer_dev, development_provenance = rebind_development_for_consumer(dev, reader=reader)
    rebuilt, official, image_map, attributes = _make_binding(
        consumer_dev, original_binding["lineage"], original_binding["raw_annotation_root"])
    consumer_official_binding_sha256 = rebuilt["binding_sha256"]
    adapter_original, adapter_current = original_binding["adapter_source"], rebuilt["adapter_source"]
    _require((adapter_original["sha256"], adapter_original["size_bytes"]) ==
             (adapter_current["sha256"], adapter_current["size_bytes"]),
             "official GT adapter bytes differ between producer and verifier")
    # Normalize only this explicit provenance path after proving helper byte equality.
    rebuilt["adapter_source"] = copy.deepcopy(adapter_original)
    rebuilt["binding_sha256"] = evidence_digest({k: v for k, v in rebuilt.items() if k != "binding_sha256"})
    _same(rebuilt, original_binding, "official GT reconstruction differs beyond verifier provenance")
    caches, predictions, raw_queries, points, errors, cells = {}, {}, {}, {}, {}, {}
    backend_references = {}
    for cell in CELLS:
        worker, row = context["workers"][cell], analysis["cells"][cell]
        predictions[cell] = reader.json(worker["prediction_artifact"])
        _require(len(predictions[cell]) == 548 * 300, "prediction count is incomplete")
        cached = diagnostic.prepare_coco_cache(coco, predictions[cell], max_dets=100)
        caches[cell] = cached
        for item in cached.describe()["matching_backend_identity"]["files"].values():
            backend = {**item, "size_bytes": Path(item["path"]).lstat().st_size,
                       "sha256_scope": "complete_file_bytes"}
            reader.read(backend, retain=False)
            backend_references[backend["path"]] = backend
        coco_metrics = _tensor_metrics(reader, worker)
        coco_check = _point_check(row["coco_cache_check"], cached.score(image_ids),
                                  coco_metrics, COCO_POINT_METRICS)
        primary_report = worker["primary_official_gt"]
        _same(primary_report.get("units"), "percent", "official primary units differ")
        _same(primary_report.get("ground_truth_binding_sha256"), original_binding["binding_sha256"],
              "official primary GT binding differs")
        primary = reader.json(primary_report["result_artifact"])
        _require(set(primary_report["metrics"]) == set(PRIMARY_METRICS), "primary endpoint set differs")
        for name in PRIMARY_METRICS:
            _number(primary_report["metrics"][name], "official primary " + name, lower=0, upper=100)
            _same(primary.get(name), primary_report["metrics"][name], "official primary artifact metric differs")
        primary_input = with_predictions(official, image_map, predictions[cell])
        primary_cache = prepare_primary_cache(primary_input, dev["source"]["primary_contract"], image_map)
        _same(primary.get("canonical_input_sha256"), primary_cache.input_sha256,
              "full official primary input identity differs")
        primary_check = _point_check(row["primary_cache_check"], primary_cache.score(image_ids),
                                    primary_report["metrics"], ("AP", "AP50", "AP75", "AR500"))
        worker_point = worker.get("primary_cache_point_check", {})
        _same(worker_point.get("metrics"), primary_cache.score(image_ids), "worker primary cache point differs")
        _same(worker_point.get("description"), primary_cache.describe(), "worker primary cache identity differs")
        _require(worker_point.get("full_primary_entry_used_for_reported_point_estimate") is True,
                 "worker primary point did not use full evaluator")
        legacy = reader.json(worker["primary_legacy_formal_gt"]["result_artifact"])
        _same({key: legacy.get(key) for key in PRIMARY_METRICS},
              worker["primary_legacy_formal_gt"]["metrics"], "legacy primary artifact differs")
        _same(reader.json(worker["diagnostic_attributes_reference"]), attributes,
              "worker diagnostic raw attributes differ")
        error_cache = diagnostic.prepare_coco_cache(coco, predictions[cell], max_dets=300)
        errors[cell] = reader.json(row["errors"])
        recomputed = diagnostic.decompose_errors(
            coco, predictions[cell], lineage_rows=lineage, cache=error_cache,
            source_references={"annotation": dev["annotation"], "lineage": lineage_ref,
                               "predictions": worker["prediction_artifact"]})
        _same(errors[cell], recomputed, "small-object error ledger does not replay: " + cell)
        raw_queries[cell] = load_raw_queries(worker["raw_query_artifact"], image_ids)
        for image in coco["images"]:
            original = raw_queries[cell][image["id"]]["original_sizes"]
            _require(original.dtype == np.int64
                     and original.tolist() == [image["width"], image["height"]],
                     "raw query original dimensions differ from the development image")
        raw_report = reader.json(row["raw_query_coverage"])
        _same(raw_report, diagnostic.raw_query_coverage(coco, predictions[cell], raw_queries[cell]),
              "raw query coverage does not replay: " + cell)
        points[cell] = {**{"coco_" + key: coco_metrics[key]
                          for key in ("AP", "AP50", "AP75", "AP_small", "AR_small")},
                        "primary_AP": primary_report["metrics"]["AP"]}
        cells[cell] = {
            "worker_result": row["worker_result"], "cache_identity": cached.describe(),
            "coco_cache_check": coco_check, "primary_cache_check": primary_check,
            "errors_reference": row["errors"], "raw_query_reference": row["raw_query_coverage"],
            "native_monitor": _audit_native_samples(reader, context["completions"][cell]["monitor"]),
        }
        del error_cache, primary_cache, recomputed, raw_report
    for label, coarse, fine in (
        ("diagonal", "t640_e640", "t896_e896"),
        ("same_training640", "t640_e640", "t640_e896"),
    ):
        report = reader.json(analysis[label + "_routing_oracle"])
        _same(report, diagnostic.routing_oracle_diagnostic(
            coco, predictions[coarse], predictions[fine],
            raw_coarse_by_image=raw_queries[coarse], coarse_cache=caches[coarse], fine_cache=caches[fine]),
            "routing oracle diagnostic does not replay: " + label)
    bootstrap = reader.json(analysis["paired_bootstrap"])
    identities = {cell: caches[cell].describe() for cell in CELLS}
    resampling = _audit_bootstrap(bootstrap, manifest["records"], image_ids, points, identities)
    inputs = {
        "primary_ap_percent": {cell: points[cell]["primary_AP"] for cell in CELLS},
        "coco_ap_small_percent": {cell: points[cell]["coco_AP_small"] for cell in CELLS},
        "small_gt_count": {cell: errors[cell]["by_iou"]["0.5"]["small_gt_count"] for cell in CELLS},
        "small_unmatched_iou50": {cell: errors[cell]["by_iou"]["0.5"]["small_unmatched"] for cell in CELLS},
        "cluster_diagonal_small_ci95_pp":
            resampling["cluster"]["contrasts"]["diagonal_total"]["coco_AP_small"]["ci95_pp"],
    }
    conditions = _conditions(inputs)
    gate = {"available": True, "eligible_for_fixed_paired_seed1_and_seed2":
            conditions["eligible_for_fixed_paired_seed1_and_seed2"],
            "not_model_certification": True, "not_training_seed_uncertainty": True,
            "checks": conditions["checks"]}
    _same(bootstrap.get("seed_gate"), gate, "bootstrap copied qualification disagrees with recomputation")
    _same(analysis.get("seed_gate"), {
        **gate, "engineering_and_hardware_campaign_passed": True,
        "conditional_execution_authority": campaign["authorization_reference"],
        "predeclared_additional_seeds": [1, 2],
    }, "analysis copied qualification disagrees with recomputation")
    progress = _progress(reader, analysis, bootstrap)
    summary = {
        "status": "PASS", "conditions": conditions, "point_cells": points,
        "bootstrap": resampling, "cells": cells, "progress": progress,
        "backend_references": [backend_references[key] for key in sorted(backend_references)],
        "development": {"image_ids": list(image_ids), "clusters": bootstrap["clusters"],
                        "manifest_records": manifest["records"], "cache_identities": identities},
        "official_gt_provenance": {
            "producer_adapter": adapter_original, "verifier_adapter": adapter_current,
            "development_rebinding": development_provenance,
            "producer_official_binding_sha256": original_binding["binding_sha256"],
            "consumer_official_binding_sha256_before_provenance_normalization": consumer_official_binding_sha256,
        },
        "scope": {"gpu_work_launched": False, "training_launched": False,
                  "bootstrap_ap_rerun": False, "image_bytes_opened": False,
                  "training_seed_uncertainty_estimated": False, "model_certification": False},
    }
    verifier = Path(__file__).resolve()
    raw = verifier.read_bytes()
    verifier_ref = {"path": str(verifier), "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    reader.source_paths.add(str(verifier))
    reader.read(verifier_ref, retain=False)
    validate_frozen_source(campaign["source"])
    _same(_consumer_source_identity(), consumer_identity, "consumer source changed during full audit")
    # The expensive replay must not leave a cached, stale evidence admission.
    final_reader = _Reader()
    final_reader.source_paths.update(reader.source_paths)
    for ref in reader.inventory():
        final_reader.read(ref, retain=False)
    return context, summary, final_reader.inventory(), verifier_ref


def build_replication_qualification(analysis_reference, output_path):
    """Persist one exclusive full CPU audit; PASS alone does not mean eligible."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "building replication qualification requires empty CUDA_VISIBLE_DEVICES")
    reference = _reference(analysis_reference)
    output = Path(output_path)
    _require(output.is_absolute() and ".." not in output.parts
             and output.resolve() == output and not output.exists(),
             "qualification must be a new canonical evidence file")
    _path_role(output, set())
    context, summary, leaves, verifier_ref = _full_audit(reference)
    analysis, campaign = context["analysis"], context["campaign"]
    document = {
        "schema_version": 1, "kind": KIND, "status": "PASS",
        "eligible_for_fixed_paired_seed1_and_seed2":
            summary["conditions"]["eligible_for_fixed_paired_seed1_and_seed2"],
        "analysis_reference": reference, "campaign_reference": analysis["campaign_reference"],
        "campaign_result_reference": analysis["campaign_result_reference"],
        "conditional_authorization_reference": campaign["authorization_reference"],
        "method_sha256": _digest(diagnostic.frozen_method()),
        "allowed_experiments": copy.deepcopy(list(ALLOWED_EXPERIMENTS)),
        "gpu_admission_granted": False, "requires_fresh_native_admission": True, "fixture_only": False,
        "producer_source": campaign["source"],
        "consumer_source_identity": context["consumer_source_identity"], "verifier_reference": verifier_ref,
        "helper_correspondence": context["helpers"], "audit_summary": summary,
        "leaf_references": leaves, "leaf_inventory_sha256": _digest(leaves),
    }
    document["qualification_sha256"] = _digest(document)
    result = _write_json(output, document)
    return {"qualification": document, "qualification_reference": result}


def _qualification_document(reader, reference):
    document = reader.json(_reference(reference))
    _require(type(document) is dict and set(document) == _QUALIFICATION_KEYS,
             "qualification schema differs")
    _same(document.get("schema_version"), 1, "qualification schema version differs")
    _same(document.get("kind"), KIND, "qualification kind differs")
    _require(document.get("status") == "PASS", "qualification audit did not pass")
    _require(document.get("fixture_only") is False and document.get("gpu_admission_granted") is False
             and document.get("requires_fresh_native_admission") is True,
             "qualification cannot grant GPU admission or use a fixture policy")
    _same(document.get("allowed_experiments"), list(ALLOWED_EXPERIMENTS),
          "qualification expanded the fixed seed/resolution/batch/epoch scope")
    _same(document.get("method_sha256"), _digest(diagnostic.frozen_method()), "qualification method differs")
    _same(document["qualification_sha256"],
          _digest({k: v for k, v in document.items() if k != "qualification_sha256"}),
          "qualification canonical digest differs")
    summary = document["audit_summary"]
    _require(type(summary) is dict and set(summary) == {
        "status", "conditions", "point_cells", "bootstrap", "cells", "progress",
        "backend_references", "development", "official_gt_provenance", "scope",
    }, "full qualification audit summary is incomplete")
    _same(summary["scope"], {
        "gpu_work_launched": False, "training_launched": False,
        "bootstrap_ap_rerun": False, "image_bytes_opened": False,
        "training_seed_uncertainty_estimated": False, "model_certification": False,
    }, "full qualification audit scope differs")
    _require(type(summary["backend_references"]) is list and summary["backend_references"],
             "recomputation backend leaf evidence is missing")
    _require(summary["progress"].get("status") == "PASS", "full audit progress is incomplete")
    _require(set(document["helper_correspondence"]) == set(_HELPERS),
             "recomputation helper provenance is incomplete")
    for name, pair in document["helper_correspondence"].items():
        _same(pair["producer"], document["producer_source"]["code_files"].get(name),
              "recomputation helper producer binding differs")
        _same(pair["consumer"], {**pair["producer"],
              "path": str(Path(__file__).resolve().parents[3] / name)},
              "recomputation helper consumer binding differs")
    leaves = document["leaf_references"]
    _require(type(leaves) is list and leaves, "qualification leaf inventory is empty")
    normalized = [_reference(ref) for ref in leaves]
    _require(len({ref["path"] for ref in normalized}) == len(normalized)
             and normalized == sorted(normalized, key=lambda ref: ref["path"]),
             "qualification leaf inventory is duplicated or unsorted")
    _same(document["leaf_inventory_sha256"], _digest(leaves), "qualification leaf inventory digest differs")
    return document


def verify_replication_qualification_binding(reference):
    """Fast worker check: sealed full-audit witnesses plus every current leaf SHA.

    This deliberately does not rebuild a model, integral image, COCO cache, raw
    diagnostic or bootstrap AP.  Its qualification reference must separately be
    bound by the worker's frozen campaign; it grants no native GPU capability.
    """
    reader = _Reader()
    document = _qualification_document(reader, reference)
    source = document["producer_source"]
    _verify_consumer_git(document["consumer_source_identity"])
    reader.source_paths.update(ref["path"] for ref in source["code_files"].values())
    reader.source_paths.add(document["verifier_reference"]["path"])
    for pair in document["helper_correspondence"].values():
        reader.source_paths.update((pair["producer"]["path"], pair["consumer"]["path"]))
    leaves = {ref["path"]: ref for ref in document["leaf_references"]}
    campaign = reader.json(document["campaign_reference"])
    _same(campaign["source"], source, "qualification producer source differs")
    evaluator_ref = _reference(campaign["evaluator_evidence_reference"])
    _same(leaves.get(evaluator_ref["path"]), evaluator_ref,
          "evaluator certification leaf omitted from sealed inventory")
    _register_evaluator_source_proofs(reader, reader.json(evaluator_ref), source)
    for ref in document["leaf_references"]:
        reader.read(ref, retain=False)
    for field in ("analysis_reference", "campaign_reference", "campaign_result_reference",
                  "conditional_authorization_reference", "verifier_reference"):
        _same(leaves.get(document[field]["path"]), document[field],
              "qualification mandatory reference absent from leaf inventory: " + field)
    actual_verifier = Path(__file__).resolve()
    _require(document["verifier_reference"]["path"] == str(actual_verifier),
             "qualification verifier imported from a different checkout")
    analysis = reader.json(document["analysis_reference"])
    _same(analysis["campaign_reference"], document["campaign_reference"], "qualification analysis campaign swapped")
    _same(analysis["campaign_result_reference"], document["campaign_result_reference"],
          "qualification analysis result swapped")
    _same(campaign["source"], source, "qualification producer source differs")
    _same(campaign["authorization_reference"], document["conditional_authorization_reference"],
          "qualification authority differs")
    _require(campaign["authorization_reference"]["sha256"] == AUTHORIZATION_SHA256,
             "qualification authority is not the authenticated user request")
    _same(analysis.get("method"), diagnostic.frozen_method(), "analysis method changed")
    _require(analysis.get("status") == "PASS" and set(analysis.get("cells", {})) == set(CELLS),
             "qualification analysis is incomplete")
    result = reader.json(document["campaign_result_reference"])
    _require(result.get("status") == "PASS" and set(result.get("completed", {})) == set(CELLS)
             and result.get("training_started") is False, "four-cell terminal result is incomplete")
    _same(result["campaign_reference"], document["campaign_reference"], "terminal campaign reference differs")
    reader.visit(analysis)
    reader.visit(campaign)
    reader.visit(document["helper_correspondence"])
    for key in ("foundation_reference", "evaluator_evidence_reference"):
        reader.visit(reader.json(campaign[key]))
    summary = document["audit_summary"]
    reader.visit(summary["backend_references"])
    _require(summary.get("status") == "PASS" and set(summary.get("cells", {})) == set(CELLS),
             "full audit summary is missing a cell")
    conditions = _conditions(summary["conditions"]["inputs"])
    _same(summary["conditions"], conditions, "sealed scientific threshold arithmetic differs")
    _same(document["eligible_for_fixed_paired_seed1_and_seed2"],
          conditions["eligible_for_fixed_paired_seed1_and_seed2"], "qualification eligibility differs")
    bootstrap_ref = _reference(analysis["paired_bootstrap"])
    _same(leaves.get(bootstrap_ref["path"]), bootstrap_ref, "bootstrap leaf omitted")
    bootstrap = reader.json(bootstrap_ref)
    dev = summary["development"]
    actual_manifest = reader.json(campaign["development_bindings"]["640"]["manifest"], embedded=True)
    _same(dev["manifest_records"], actual_manifest["records"], "sealed development manifest differs")
    resampling = _audit_bootstrap(
        bootstrap, dev["manifest_records"], tuple(dev["image_ids"]),
        summary["point_cells"], dev["cache_identities"])
    _same(summary["bootstrap"], resampling, "sealed bootstrap audit summary differs")
    _same(conditions["inputs"]["cluster_diagonal_small_ci95_pp"],
          resampling["cluster"]["contrasts"]["diagonal_total"]["coco_AP_small"]["ci95_pp"],
          "threshold witness uses a different bootstrap CI")
    for cell in CELLS:
        row, checked = analysis["cells"][cell], summary["cells"][cell]
        _same(row["worker_result"], checked["worker_result"], "sealed worker identity differs")
        worker = reader.json(row["worker_result"])
        item = result["completed"][cell]
        _same(worker["contract_reference"], item["contract_reference"], "worker contract reference differs")
        contract = reader.json(item["contract_reference"])
        _same(contract, cross_worker_contract(campaign, cell), "worker contract body differs")
        completion = reader.json(item["completion_reference"])
        reader.visit(worker)
        reader.visit(completion)
        reader.visit(reader.json(worker["binding_reference"]))
        reader.visit(reader.json(completion["launch_reference"]))
        _same(completion["result_reference"], row["worker_result"], "completion worker reference differs")
        _require(worker.get("status") == "PASS" and worker.get("cell_key") == cell
                 and worker.get("run_id") == contract["run_id"]
                 and completion["monitor"].get("status") == "PASS"
                 and completion["monitor"].get("worker_exited") is True
                 and completion["monitor"].get("sampled_clock_compliance") is True,
                 "sealed worker or native terminal closure differs")
        native_final = reader.json(completion["monitor"]["final_report_reference"])
        _same({k: v for k, v in completion["monitor"].items() if k != "final_report_reference"},
              native_final, "current native final differs from sealed completion")
        for metric, name in (("primary_AP", "primary_ap_percent"),
                             ("coco_AP_small", "coco_ap_small_percent")):
            _same(summary["point_cells"][cell][metric], conditions["inputs"][name][cell],
                  "threshold metric witness differs")
        _same(worker["primary_official_gt"]["metrics"]["AP"], conditions["inputs"]["primary_ap_percent"][cell],
              "official primary AP witness differs from worker")
        _same(worker["coco_secondary"]["metrics"]["AP_small"]["value"],
              conditions["inputs"]["coco_ap_small_percent"][cell],
              "COCO small AP witness differs from worker")
        for field, target in (("errors", "errors_reference"), ("raw_query_coverage", "raw_query_reference")):
            _same(row[field], checked[target], "diagnostic witness reference differs")
        for ref in (row["worker_result"], row["errors"], row["raw_query_coverage"],
                    row["prediction_artifact"], row["raw_query_artifact"]):
            _same(leaves.get(ref["path"]), ref, "required cell leaf omitted")
    _same(bootstrap["seed_gate"]["checks"], conditions["checks"], "bootstrap threshold checks differ")
    _same(analysis["seed_gate"]["checks"], conditions["checks"], "analysis threshold checks differ")
    observed_leaves = [ref for ref in reader.inventory() if ref["path"] != reference["path"]]
    _same(observed_leaves, document["leaf_references"], "required evidence leaf omitted from sealed inventory")
    return copy.deepcopy(document)


def validate_replication_qualification(reference):
    """Repeat the full independent CPU audit of an existing qualification."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "full replication qualification audit requires empty CUDA_VISIBLE_DEVICES")
    document = verify_replication_qualification_binding(reference)
    context, summary, leaves, verifier_ref = _full_audit(document["analysis_reference"])
    _same(summary, document["audit_summary"], "full qualification audit no longer reproduces")
    _same(leaves, document["leaf_references"], "full qualification leaf closure differs")
    _same(verifier_ref, document["verifier_reference"], "qualification verifier identity differs")
    _same(context["helpers"], document["helper_correspondence"], "qualification helper correspondence differs")
    _same(context["consumer_source_identity"], document["consumer_source_identity"],
          "qualification consumer source identity differs")
    return document
