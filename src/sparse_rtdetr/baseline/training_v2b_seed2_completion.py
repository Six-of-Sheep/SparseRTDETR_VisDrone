"""A separate seed-2 missing-cell contract; historical runs retain their identities.

This module changes orchestration and evidence admission only.  It does not
train, construct a model, change a clock, or turn the failed old campaign PASS.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

from .training_v2b_evidence import canonical_sha256, strict_json_loads, write_exclusive_json
from .training_v2b_campaign import source_snapshot, validate_frozen_source
from .training_v2b_replication_summary import _Reader, _checkpoint_reference

BASE_COMMIT = "747565458d41bd021de8fb42d44cf8ce197a945e"
HISTORICAL_CAMPAIGN_ID = "v2bseeds-20260914t211244z-065fe7a4"
HISTORICAL_FREEZE_SHA256 = "9be19c48ad445d7cc1027fdc3015c88068894aff94bedfc8e8a7b37e6c1360ab"
HISTORICAL_CONTROL_COMPLETION_SHA256 = "c0735f0df8bd80a6938bf0dfc4220da2009d0ab5e5739ffca91bada8b3475cfd"
HISTORICAL_FAILURE_SHA256 = "99c1b3c21856d8ae7889ced73c5b51ee9180f4406dd35e4924dfeb891df8e048"
HISTORICAL_PARTIAL_SUMMARY_SHA256 = "699d8396ad0da95b8fdb0d706ef8f91387052e937780d7ec9fa4b024f5291438"
HISTORICAL_EMA_AUDIT_SHA256 = "b715160694b15ec6308d539d071e417271b233781c9ec72bf5656dadabbf91c3"
KIND = "v2b_seed2_historical_control_completion_freeze"
WORKER_KIND = "v2b_seed2_historical_control_completion_worker"
CELL_ID = "s2_r896"
STAGES = ("smoke", "smoke_replay", "control30")
REF_KEYS = {"path", "sha256", "size_bytes", "sha256_scope"}
SOURCE_KEYS = {"repo_root", "commit", "tree", "branch", "code_files", "inventory_sha256"}
ALLOWED_ADDED_FILES = frozenset({
    "src/sparse_rtdetr/baseline/training_v2b_seed2_completion.py",
    "src/sparse_rtdetr/baseline/training_v2b_seed2_completion_worker.py",
    "src/sparse_rtdetr/baseline/training_v2b_seed2_completion_campaign.py",
    "src/sparse_rtdetr/baseline/training_v2b_seed2_completion_summary.py",
    "src/sparse_rtdetr/baseline/training_v2b_seed2_process_logs.py",
    "tools/run_training_v2b_seed2_completion_worker.py",
    "tools/run_training_v2b_seed2_completion_campaign.py",
    "tools/summarize_training_v2b_seed2_completion.py",
    "tests/test_training_v2b_seed2_completion.py",
    "tests/test_training_v2b_seed2_completion_worker.py",
    "tests/test_training_v2b_seed2_completion_campaign.py",
    "tests/test_training_v2b_seed2_completion_summary.py",
    "tests/test_training_v2b_seed2_process_logs.py",
})
ALLOWED_MODIFIED_FILES = frozenset({
    "tools/repository_contract_check.py",
})
SUPPLEMENTAL_AUTHORIZATION_TEXT = (
    "授权继续完成第三个可信的 seed 2 配对。\n"
    "优先采用“缺失单元补全”：保留旧 seed 2／640 的认证终点作为不可变历史对照，保留失败的 seed 2／896 STOP_NO_RETRY 现场；建立新的跨 campaign 证据绑定，并从 update 0 重新运行一次 896×8×2、30 epoch。不得续跑失败的 epoch 29，也不得把旧 640 伪装成新 campaign 产物。\n"
    "请自行判断这种方案能否在不改变训练科学语义的情况下严谨实现。需要绑定初始化、预训练权重、seed、样本与增强顺序、优化器、BN/DN、loss、EMA、更新次数和评估合同。若只能修改编排与证据层且训练核心语义保持一致，则只运行 896；若无法证明等价，或必须修改训练语义，则退回完整重跑 640×8×2 与 896×8×2 两臂。\n"
    "运行前重新完成 GPU、锁频、外部进程及硬件日志门禁。外部 GPU 作业、身份漂移、非有限值、OOM、恢复不一致或硬件异常仍应立即停止并保留现场。\n"
    "本阶段只闭合 seed 2，不进入 60 epoch、960、sparse、confirmatory 或 test。完成后报告三个 seed 的逐对差值、均值与样本 SD，并明确 seed 0 为探索性、seed 1/2 为前瞻复现；不要把 n=3 表述为统计显著。\n"
)
SUPPLEMENTAL_AUTHORIZATION_SHA256 = hashlib.sha256(SUPPLEMENTAL_AUTHORIZATION_TEXT.encode("utf-8")).hexdigest()
EXECUTION_POLICY = {
    "design": "one_missing_cell_with_immutable_historical_control",
    "historical_control_is_new_campaign_product": False,
    "old_campaign_status": "STOP_NO_RETRY",
    "new_formal_cells": ["s2_r896"],
    "formal_start": "pretrained_update_zero",
    "formal_checkpoint_resume": False,
    "automatic_retry_or_batch_fallback": False,
    "input_roles": ["train_core", "development"],
    "health_only_epochs": [10, 20],
    "primary_endpoint_epoch": 30,
    "optimizer_updates": 9120,
    "physical_microsteps": 18240,
    "logical_sample_draws": 145920,
    "historical_capacity_reused": True,
    "fresh_smoke_and_new_process_replay_required": True,
    "fresh_native_admission_required_per_invocation": True,
    "clock_setting_or_reset_allowed": False,
    "training_extension_authorized": False,
}
HISTORY_KEYS = {
    "freeze_reference", "control_completion_reference", "control_result_reference",
    "control_initial_state_reference", "control_final_state_reference",
    "control_checkpoint_reference", "control_ema_audit_reference",
    "failed_campaign_reference", "failed_controller_reference", "failed_monitor_reference",
    "partial_summary_reference", "preformal_completions", "historical_smoke_reference",
    "historical_capacity_reference",
}
FREEZE_KEYS = {
    "schema_version", "kind", "campaign_id", "repo_root", "output_root", "source",
    "clean_archive_reference", "qualification_reference", "authorization_reference",
    "supplemental_authorization_reference", "foundation_reference", "common", "cells",
    "stages", "historical", "source_equivalence", "execution_policy",
}
CELL_KEYS = {"config", "development_binding", "epoch_order_sha256", "capacity_plan", "invocations"}
COMMON_KEYS = {
    "torch_version", "code_files", "train_core", "pretrained", "policy_bundle",
    "expected_gpu_uuid", "num_workers", "prefetch_factor",
}
WORKER_KEYS = {
    "schema_version", "kind", "campaign_id", "cell_id", "stage", "run_id",
    "invocation_id", "output_dir", "repo_root", "freeze_reference",
    "qualification_reference", "supplemental_authorization_reference", "stage_gate_reference",
    "config", "development_binding", "epoch_order_sha256", "capacity_plan",
    "matched_reference", "replay_source", *COMMON_KEYS,
}


class Seed2CompletionError(RuntimeError):
    """The explicitly frozen historical-control design was not established."""


def _require(value, message):
    if not value:
        raise Seed2CompletionError(message)


def _typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_typed_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _same(actual, expected, label):
    _require(_typed_equal(actual, expected), label + " differs")


def _schema(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys), label + " schema differs")


def _path(value, label):
    _require(type(value) is str, label + " path is not text")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts and str(path) == value,
             label + " path must be canonical and absolute")
    _require(not any(part.casefold() in {"confirmatory", "test"} for part in path.parts),
             label + " crosses a prohibited data role")
    return path


def _identifier(value, label):
    _require(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value),
             label + " is not a safe identifier")
    return value


def _reference(value, label="evidence"):
    _schema(value, REF_KEYS, label + " reference")
    _path(value["path"], label)
    _require(type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
             label + " requires a complete SHA256")
    _require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0,
             label + " byte size is invalid")
    _same(value["sha256_scope"], "complete_file_bytes", label + " hash scope")
    return copy.deepcopy(value)


def _json(reference):
    return _Reader().json(_reference(reference))


def fixed_config():
    from .training_v2b import V2BConfig
    return asdict(V2BConfig(seed=2, input_size=896, physical_batch_size=8,
                           accumulation_steps=2, sampling_backend="deterministic_gather"))


def invocation_plan(campaign_id, output_root):
    _identifier(campaign_id, "campaign")
    root = _path(str(output_root), "output")
    return {
        stage: {
            "run_id": f"{campaign_id}-{CELL_ID}-{'smoke' if stage == 'smoke_replay' else stage}",
            "invocation_id": f"{campaign_id}-{CELL_ID}-{stage}",
            "output_dir": str(root / "execution" / f"{CELL_ID}-{stage}"),
        }
        for stage in STAGES
    }


def _source_schema(source):
    _schema(source, SOURCE_KEYS, "source")
    root = _path(source["repo_root"], "source")
    for key in ("commit", "tree"):
        _require(type(source[key]) is str and re.fullmatch(r"[0-9a-f]{40}", source[key]),
                 "complete source " + key + " is required")
    _require(type(source["branch"]) is str, "source branch is required")
    files = source["code_files"]
    _require(type(files) is dict and files, "complete tracked source inventory is required")
    for name, reference in files.items():
        path = Path(name)
        _require(type(name) is str and not path.is_absolute() and ".." not in path.parts
                 and str(path) == name and not name.startswith("artifacts/"),
                 "invalid tracked relative source path")
        _same(_reference(reference)["path"], str(root / path), "tracked source location")
    _same(source["inventory_sha256"], canonical_sha256(files), "tracked source inventory digest")
    return source


def source_extension_proof(old_source, new_source):
    """Freeze scientific bytes; allow only the exact additive/evidence delta."""
    _source_schema(old_source)
    _source_schema(new_source)
    _same(old_source["commit"], BASE_COMMIT, "historical baseline commit")
    _require(old_source["repo_root"] != new_source["repo_root"], "an isolated new checkout is required")
    old, new = old_source["code_files"], new_source["code_files"]
    _require(set(old) <= set(new), "an original tracked file was removed")
    modified = set()
    for name in old:
        if any(new[name][field] != old[name][field]
               for field in ("sha256", "size_bytes", "sha256_scope")):
            modified.add(name)
            continue
        for field in ("sha256", "size_bytes", "sha256_scope"):
            _same(new[name][field], old[name][field], "unchanged tracked bytes " + name)
    _same(modified, set(ALLOWED_MODIFIED_FILES), "exact evidence-only modified file set")
    additions = set(new) - set(old)
    _require(additions <= ALLOWED_ADDED_FILES, "source addition is outside the frozen orchestration allowlist")
    _require("src/sparse_rtdetr/baseline/training_v2b_seed2_completion.py" in additions,
             "the independent completion validator must be in the frozen source")
    return {
        "base_commit": BASE_COMMIT,
        "old_source": copy.deepcopy(old_source),
        "new_inventory_sha256": new_source["inventory_sha256"],
        "original_tracked_file_count": len(old),
        "original_tracked_files_byte_identical": False,
        "original_scientific_tracked_files_byte_identical": True,
        "modified_files": sorted(modified),
        "permitted_modified_files": sorted(ALLOWED_MODIFIED_FILES),
        "modified_files_are_evidence_only": True,
        "added_files": sorted(additions),
        "permitted_additions": sorted(ALLOWED_ADDED_FILES),
        "training_mathematics_changed": False,
    }


def verify_historical_qualification(reference, old_repo_root):
    """Run the original verifier in its real original checkout, without rebinding it.

    Its imported-file and consumer-Git assertions remain active. The new
    checkout never masquerades as the original qualification consumer.
    """
    ref = _reference(reference)
    root = _path(old_repo_root, "historical qualification checkout")
    code = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(sys.argv[1])/'src'))\n"
        "from sparse_rtdetr.baseline.training_v2b_replication_worker import _qualified_document\n"
        "reference=json.loads(sys.argv[2])\n"
        "document=_qualified_document(reference)\n"
        "print('HISTORICAL_QUALIFICATION_PASS:'+reference['sha256'])\n"
    )
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1", PYTHONPATH=str(root / "src"),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MKL_THREADING_LAYER="GNU")
    import json
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), json.dumps(ref, sort_keys=True)],
        cwd=root, env=environment, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=600, check=False,
    )
    _require(completed.returncode == 0,
             "original historical qualification verifier failed: " + completed.stderr[-2000:])
    _require(("HISTORICAL_QUALIFICATION_PASS:" + ref["sha256"]) in completed.stdout.splitlines(),
             "original historical qualification verifier did not return its bound witness")
    return _json(ref)


def _history_structure(history):
    _schema(history, HISTORY_KEYS, "historical evidence")
    for key in HISTORY_KEYS - {"preformal_completions"}:
        _reference(history[key], "historical " + key)
    for key, sha in (
        ("freeze_reference", HISTORICAL_FREEZE_SHA256),
        ("control_completion_reference", HISTORICAL_CONTROL_COMPLETION_SHA256),
        ("failed_campaign_reference", HISTORICAL_FAILURE_SHA256),
        ("partial_summary_reference", HISTORICAL_PARTIAL_SUMMARY_SHA256),
        ("control_ema_audit_reference", HISTORICAL_EMA_AUDIT_SHA256),
    ):
        _same(history[key]["sha256"], sha, "pinned historical " + key)
    rows = history["preformal_completions"]
    _require(type(rows) is list and len(rows) == 12, "all twelve historical engineering stages are required")
    expected = {(cell, stage) for cell in ("s1_r640", "s1_r896", "s2_r640", "s2_r896")
                for stage in ("capacity", "smoke", "smoke_replay")}
    positions = []
    for row in rows:
        _schema(row, {"cell_id", "stage", "worker_result_reference", "completion_reference"},
                "historical prerequisite")
        positions.append((row["cell_id"], row["stage"]))
        _reference(row["worker_result_reference"])
        _reference(row["completion_reference"])
    _require(set(positions) == expected and len(set(positions)) == 12,
             "historical engineering identities are missing or duplicated")


def _build_history(*, historical_freeze_reference, historical_control_completion_reference,
                   historical_control_ema_audit_reference, failed_campaign_reference,
                   partial_summary_reference):
    old = _json(historical_freeze_reference)
    partial = _json(partial_summary_reference)
    failure = _json(failed_campaign_reference)
    completion = _json(historical_control_completion_reference)
    result = _json(completion["result_reference"])
    rows = failure["completed"]
    _require(type(rows) is list, "historical completed-stage sequence is missing")
    preformal = [copy.deepcopy(row) for row in rows if row.get("stage") != "control30"]
    indexed = {(row["cell_id"], row["stage"]): row for row in rows}
    history = {
        "freeze_reference": historical_freeze_reference,
        "control_completion_reference": historical_control_completion_reference,
        "control_result_reference": completion["result_reference"],
        "control_initial_state_reference": result["initial_state_reference"],
        "control_final_state_reference": result["final_state_reference"],
        "control_checkpoint_reference": _checkpoint_reference(result["checkpoints"]["epoch_30"]["checkpoint"]),
        "control_ema_audit_reference": historical_control_ema_audit_reference,
        "failed_campaign_reference": failed_campaign_reference,
        "failed_controller_reference": partial["failed_controller_reference"],
        "failed_monitor_reference": partial["failed_monitor_reference"],
        "partial_summary_reference": partial_summary_reference,
        "preformal_completions": preformal,
        "historical_smoke_reference": indexed["s2_r640", "smoke"]["worker_result_reference"],
        "historical_capacity_reference": indexed["s2_r896", "capacity"]["worker_result_reference"],
    }
    _history_structure(history)
    return history, old


def _verify_ema_audit(reader, history, source, partial):
    reference = history["control_ema_audit_reference"]
    _same(partial["ema_binding_audits"]["s2_r640"]["report_reference"], reference,
          "partial closure EMA audit")
    report = reader.json(reference)
    _same(report.get("status"), "PASS_CHECKPOINT_EMA_EVALUATION_BINDING", "historical EMA audit status")
    _same(report.get("kind"), "completed_checkpoint_ema_evaluation_binding_cpu", "historical EMA audit kind")
    _same(report.get("digest_domain"), "payload['ema']['module']; canonical_tensor_inventory_v1, all tensors",
          "EMA module digest domain")
    _same(report.get("ema_wrapper_digest_compared"), False, "EMA wrapper/module distinction")
    rows = report.get("cells")
    _require(type(rows) is list and len(rows) == 1, "historical EMA audit must contain exactly one cell")
    row = rows[0]
    for field, expected in (("cell_id", "s2_r640"), ("input_size", 640), ("status", "PASS"),
                            ("checkpoint_ema_updates", 9120), ("both_development_digests_match", True),
                            ("checkpoint_bytes_verified_before_and_after_load", True)):
        _same(row.get(field), expected, "historical EMA " + field)
    _same(row["checkpoint_reference"], source["checkpoint_reference"], "historical EMA checkpoint")
    _same(row["development_result_reference"], source["evaluation"]["result_artifact"], "historical EMA evaluation")
    _same(row["weights_state_sha256"], source["evaluation"]["weights"]["state_sha256"], "evaluated EMA module")
    _same(row["training_state_unchanged_ema_sha256"], row["weights_state_sha256"], "preserved EMA module")
    for ref in report["input_references"]:
        reader.read(ref, retain=False)
    return report


def _audit_control_receipts(reader, source):
    from . import training_v2b_replication_summary as summary
    resources = summary._resources()
    digest = hashlib.sha256()
    for epoch in range(1, 31):
        indices = []
        for index in range(304):
            ref = {key: source["result"]["receipts"][(epoch - 1) * 304 + index][key] for key in REF_KEYS}
            record = reader.json(ref)
            indices.append(summary._consume_window(record, source, epoch, index, resources))
            digest.update((ref["sha256"] + "\n").encode("ascii"))
        _same(canonical_sha256(indices), source["contract"]["epoch_order_sha256"][str(epoch)],
              "historical control actual complete epoch order")
        _require(len({i for batch in indices for i in batch}) == 4864,
                 "historical control epoch does not contain 4864 distinct images")
    return {"logical_windows": 9120, "logical_sample_draws": 145920,
            "physical_microsteps": 18240, "receipt_sha256_sequence": digest.hexdigest(),
            "original_receipt_validator_used": True}


def admit_historical_seed2_control(freeze):
    """Authenticate the real old campaign and source, never manufacture a PASS.

    Reads JSON and full-byte hashes, including the old completed epoch-30
    checkpoint. No checkpoint deserialization, model, image decode, or GPU API.
    """
    validate_seed2_completion_freeze(freeze, verify_files=False)
    history = freeze["historical"]
    reader = _Reader()
    old = reader.json(history["freeze_reference"])
    _same(old["campaign_id"], HISTORICAL_CAMPAIGN_ID, "historical campaign")
    _same(old["source"], freeze["source_equivalence"]["old_source"], "historical source")
    _same(old["qualification_reference"], freeze["qualification_reference"], "original qualification")
    _same(old["authorization_reference"], freeze["authorization_reference"], "original authorization")
    _same(old["foundation_reference"], freeze["foundation_reference"], "original foundation")
    validate_frozen_source(old["source"])
    validate_frozen_source(freeze["source"])
    _same(source_extension_proof(old["source"], freeze["source"]), freeze["source_equivalence"],
          "source extension proof")
    qualification = verify_historical_qualification(old["qualification_reference"], old["repo_root"])
    _same(qualification["conditional_authorization_reference"], old["authorization_reference"],
          "historical qualification authorization")

    from . import training_v2b_replication_worker as original
    from . import training_v2b_replication_summary as summary
    partial = reader.json(history["partial_summary_reference"])
    _same(partial.get("status"), "PARTIAL", "partial closure scope")
    _same(partial.get("cpu_validation_status"), "PASS", "partial CPU evidence closure")
    _same(partial.get("experiment_status"), "STOP_NO_RETRY", "partial experiment status")
    _same(partial.get("campaign_complete"), False, "partial campaign completeness")
    _same(partial.get("three_pairs_complete"), False, "partial paired completeness")
    _same(partial.get("freeze_reference"), history["freeze_reference"], "partial freeze")
    for partial_key, history_key in (
        ("campaign_result_reference", "failed_campaign_reference"),
        ("failed_controller_reference", "failed_controller_reference"),
        ("failed_monitor_reference", "failed_monitor_reference"),
    ):
        _same(partial.get(partial_key), history[history_key], "partial pinned " + partial_key)
    _same(partial.get("qualification_reference"), freeze["qualification_reference"], "partial qualification")
    failure = reader.json(history["failed_campaign_reference"])
    _same(failure.get("campaign_id"), old["campaign_id"], "failed campaign identity")
    _same(failure.get("freeze_reference"), history["freeze_reference"], "failed freeze")
    _same(failure.get("status"), "STOP_NO_RETRY", "preserved failure status")
    _same(failure.get("automatic_retry"), False, "preserved no-retry failure")
    controller = reader.json(history["failed_controller_reference"])
    _same(controller.get("status"), "STOP_NO_RETRY", "failed controller")
    _same((controller.get("cell_id"), controller.get("stage")), ("s2_r896", "control30"),
          "failed missing cell")
    monitor = reader.json(history["failed_monitor_reference"])
    _same(monitor.get("status"), "FAIL", "preserved failed native monitor")
    _same(monitor.get("worker_exited"), True, "failed worker exited")
    failed_contract = reader.json(controller["contract_reference"])
    original.validate_replication_worker_contract(failed_contract, verify_files=False)
    original.validate_replication_freeze(old, failed_contract)
    _same(failed_contract["freeze_reference"], history["freeze_reference"], "failed contract freeze")
    _same(failed_contract["replay_source"], None, "failed formal resume")
    old_gate = reader.json(failed_contract["stage_gate_reference"])
    original.validate_replication_stage_gate(old_gate, failed_contract, old, verify_files=True)
    old_rows = {(row["cell_id"], row["stage"]): row for row in old_gate["prerequisites"]}
    _same([row for row in failure["completed"] if row["stage"] != "control30"],
          history["preformal_completions"], "all historical preformal completions")
    _same({key for key in old_rows if key[1] != "control30"},
          {(row["cell_id"], row["stage"]) for row in history["preformal_completions"]},
          "qualified historical engineering set")
    for row in history["preformal_completions"]:
        _same(old_rows[row["cell_id"], row["stage"]], row, "old stage evidence lineage")

    contexts = {}
    for cell, stage in (("s2_r640", "control30"), ("s2_r640", "smoke"), ("s2_r896", "capacity")):
        row = old_rows[cell, stage]
        result, contract = original._successful_cell_source(
            row["worker_result_reference"], cell_id=cell, stage=stage, contract=failed_contract, freeze=old)
        completion = reader.json(row["completion_reference"])
        contexts[cell, stage] = {
            "result": result, "result_reference": row["worker_result_reference"],
            "contract": contract, "completion": completion, "completion_reference": row["completion_reference"],
        }
    control = contexts["s2_r640", "control30"]
    _same(control["completion_reference"], history["control_completion_reference"], "historical control completion")
    _same(control["result_reference"], history["control_result_reference"], "historical control result")
    source = summary._training_source(
        reader, control["result_reference"], seed=2, size=640,
        expected_config=old["cells"]["s2_r640"]["config"],
        expected_checkpoint=history["control_checkpoint_reference"], completion=control["completion"])
    source.update(completion_reference=control["completion_reference"])
    _same(source["result"]["formal_started_from_pretrained_update_zero"], True, "old A started at update zero")
    _same(source["result"]["formal_checkpoint_resume_used"], False, "old A was not resumed")
    for key, field in (("control_initial_state_reference", "initial_state_reference"),
                       ("control_final_state_reference", "final_state_reference")):
        _same(history[key], source["result"][field], "historical control " + field)
    _same(history["historical_smoke_reference"], contexts["s2_r640", "smoke"]["result_reference"],
          "historical matched smoke")
    _same(history["historical_capacity_reference"], contexts["s2_r896", "capacity"]["result_reference"],
          "historical capacity")
    source["ema_audit"] = _verify_ema_audit(reader, history, source, partial)
    source["ema_audit_reference"] = history["control_ema_audit_reference"]
    source["receipt_audit"] = _audit_control_receipts(reader, source)
    expected_common = copy.deepcopy(old["common"])
    expected_common["code_files"] = freeze["source"]["code_files"]
    _same(freeze["common"], expected_common, "unchanged common scientific/environment/hardware contract")
    cell = freeze["cells"][CELL_ID]
    old_cell = old["cells"][CELL_ID]
    for key in ("config", "capacity_plan", "epoch_order_sha256"):
        _same(cell[key], old_cell[key], "same seed2/896 " + key)
    restored = copy.deepcopy(cell["development_binding"])
    restored["repo_root"] = old_cell["development_binding"]["repo_root"]
    restored["binding_sha256"] = canonical_sha256({k: v for k, v in restored.items() if k != "binding_sha256"})
    _same(restored, old_cell["development_binding"], "development rebinding may only change source checkout identity")
    # Reuse the original input validators in the actual new consumer before any
    # GPU admission. Historical metadata PASS is not a live-file exemption.
    from .training_v2b_replication_capacity import validate_replication_capacity_plan
    from .training_v2b_development import validate_development_binding
    plan = reader.json(cell["capacity_plan"])
    validate_replication_capacity_plan(
        plan, seed=2, input_size=896, train_core=freeze["common"]["train_core"], verify_files=True)
    validate_development_binding(cell["development_binding"], verify_files=True)
    for field in ("annotation", "manifest"):
        reader.read(freeze["common"]["train_core"][field], retain=False)
    reader.read(freeze["common"]["pretrained"], retain=False)
    forbidden = Path(old["cells"][CELL_ID]["invocations"]["control30"]["output_dir"])
    for name in ("worker-result.json", "checkpoint-epoch-030.pt", "epoch-030.json"):
        _require(not (forbidden / name).exists(), "a failed historical B terminal artifact appeared")
    _require(not (forbidden.parent / (forbidden.name + ".completion.json")).exists(),
             "the failed historical B completion appeared")
    return {
        "freeze": old, "qualification": qualification, "control": source,
        "smoke": contexts["s2_r640", "smoke"], "capacity": contexts["s2_r896", "capacity"],
        "preformal": copy.deepcopy(history["preformal_completions"]),
        "failed_campaign": failure, "failed_controller": controller, "failed_monitor": monitor,
        "partial_summary": partial, "partial_summary_reference": history["partial_summary_reference"],
        "verified_leaf_references": reader.inventory(),
    }


def validate_seed2_completion_freeze(freeze, *, verify_files=True):
    _schema(freeze, FREEZE_KEYS, "seed2 completion freeze")
    _same(freeze["schema_version"], 1, "completion schema version")
    _require(type(freeze["schema_version"]) is int, "completion version must be an integer")
    _same(freeze["kind"], KIND, "independent completion kind")
    _identifier(freeze["campaign_id"], "new campaign")
    _require(freeze["campaign_id"] != HISTORICAL_CAMPAIGN_ID, "the old campaign ID cannot be reused")
    root, output = _path(freeze["repo_root"], "new checkout"), _path(freeze["output_root"], "new output")
    _require(output.is_relative_to(root / "artifacts" / "training") and output != root / "artifacts" / "training",
             "new output must belong to the isolated training artifact root")
    _source_schema(freeze["source"])
    _same(freeze["source"]["repo_root"], str(root), "new source root")
    _schema(freeze["common"], COMMON_KEYS, "common")
    _same(freeze["source"]["code_files"], freeze["common"]["code_files"], "actual new code inventory")
    _same(freeze["stages"], list(STAGES), "exact new 896 smoke/replay/formal stages")
    _same(freeze["execution_policy"], EXECUTION_POLICY, "fixed execution policy")
    _require(type(freeze["cells"]) is dict, "missing-cell inventory must be a mapping")
    _same(set(freeze["cells"]), {CELL_ID}, "exact one new missing cell")
    _history_structure(freeze["historical"])
    history = freeze["historical"]
    old_output = _path(history["freeze_reference"]["path"], "old freeze").parent
    _require(not output.is_relative_to(old_output) and not old_output.is_relative_to(output),
             "historical and new output roots must be disjoint")
    for key in ("clean_archive_reference", "qualification_reference", "authorization_reference",
                "supplemental_authorization_reference", "foundation_reference"):
        _reference(freeze[key], key)
    _same(freeze["supplemental_authorization_reference"]["sha256"], SUPPLEMENTAL_AUTHORIZATION_SHA256,
          "explicit missing-cell supplemental grant")
    _require(freeze["authorization_reference"] != freeze["supplemental_authorization_reference"],
             "supplemental grant must not replace original authorization")
    proof = freeze["source_equivalence"]
    _require(type(proof) is dict and "old_source" in proof, "explicit source equivalence evidence is required")
    _same(source_extension_proof(proof["old_source"], freeze["source"]), proof, "source equivalence proof")
    _require(Path(proof["old_source"]["repo_root"]) != root, "historical checkout must remain separate")
    cell = freeze["cells"][CELL_ID]
    _schema(cell, CELL_KEYS, "missing cell")
    _same(cell["config"], fixed_config(), "unchanged seed2/896 training configuration")
    from .training_v2b_replication_worker import frozen_seed_orders, _development_metadata
    _same(cell["epoch_order_sha256"], frozen_seed_orders(2), "all thirty seed2 epoch orders")
    _reference(cell["capacity_plan"], "historical capacity plan")
    _development_metadata(cell["development_binding"], 896)
    _same(cell["development_binding"]["repo_root"], str(root), "development consumer source")
    _same(cell["invocations"], invocation_plan(freeze["campaign_id"], output), "new invocation identities")
    _same(freeze["common"]["num_workers"], 2, "physical loader workers")
    _same(freeze["common"]["prefetch_factor"], 2, "loader prefetch")
    _same(freeze["common"]["policy_bundle"]["authorization_reference"], freeze["authorization_reference"],
          "original hardware authorization")
    if verify_files:
        reader = _Reader()
        _same(reader.read(freeze["supplemental_authorization_reference"]),
              SUPPLEMENTAL_AUTHORIZATION_TEXT.encode("utf-8"), "supplemental authorization bytes")
        from .training_v2b_replication_campaign import _external_policy
        _external_policy(freeze["common"]["policy_bundle"], freeze["authorization_reference"])
        archive = reader.json(freeze["clean_archive_reference"])
        _same(archive.get("status"), "PASS", "clean archive status")
        _same(archive.get("candidate_tree"), freeze["source"]["tree"], "clean archive tree")
        admit_historical_seed2_control(freeze)
    return copy.deepcopy(freeze)


def seed2_completion_worker_contract(freeze, freeze_reference, stage, *, stage_gate_reference, replay_source=None):
    _require(stage in STAGES, "only the three frozen seed2/896 invocations are allowed")
    value = freeze["cells"][CELL_ID]
    matched = freeze["historical"]["control_result_reference" if stage == "control30" else "historical_smoke_reference"]
    contract = {
        "schema_version": 1, "kind": WORKER_KIND, "campaign_id": freeze["campaign_id"],
        "cell_id": CELL_ID, "stage": stage, "repo_root": freeze["repo_root"],
        **copy.deepcopy(value["invocations"][stage]), **copy.deepcopy(freeze["common"]),
        **{key: copy.deepcopy(value[key]) for key in CELL_KEYS - {"invocations"}},
        "freeze_reference": copy.deepcopy(freeze_reference),
        "qualification_reference": copy.deepcopy(freeze["qualification_reference"]),
        "supplemental_authorization_reference": copy.deepcopy(freeze["supplemental_authorization_reference"]),
        "stage_gate_reference": copy.deepcopy(stage_gate_reference),
        "matched_reference": copy.deepcopy(matched), "replay_source": copy.deepcopy(replay_source),
    }
    return validate_seed2_completion_worker_contract(contract, freeze, verify_files=False)


def validate_seed2_completion_worker_contract(contract, freeze, *, verify_files=False):
    _schema(contract, WORKER_KEYS, "new completion worker")
    validate_seed2_completion_freeze(freeze, verify_files=False)
    _same(contract["schema_version"], 1, "worker version")
    _require(type(contract["schema_version"]) is int, "worker version must be integer")
    _same(contract["kind"], WORKER_KIND, "worker independent kind")
    _same(contract["cell_id"], CELL_ID, "single authorized missing cell")
    stage = contract["stage"]
    _require(stage in STAGES, "worker stage is outside the new fixed sequence")
    for key in ("campaign_id", "repo_root", "qualification_reference", "supplemental_authorization_reference"):
        _same(contract[key], freeze[key], "worker freeze " + key)
    for key, value in freeze["common"].items():
        _same(contract[key], value, "worker common " + key)
    cell = freeze["cells"][CELL_ID]
    for key in CELL_KEYS - {"invocations"}:
        _same(contract[key], cell[key], "worker cell " + key)
    for key, value in cell["invocations"][stage].items():
        _same(contract[key], value, "worker invocation " + key)
    ref = _reference(contract["freeze_reference"], "new freeze")
    _same(ref["path"], str(Path(freeze["output_root"]) / "freeze.json"), "owned new freeze")
    _reference(contract["stage_gate_reference"], "new stage gate")
    matched = freeze["historical"]["control_result_reference" if stage == "control30" else "historical_smoke_reference"]
    _same(contract["matched_reference"], matched, "actual historical matched source")
    if stage == "smoke_replay":
        replay = _reference(contract["replay_source"], "same-campaign new smoke")
        _same(replay["path"], str(Path(cell["invocations"]["smoke"]["output_dir"]) / "worker-result.json"),
              "only the new smoke may supply a replay checkpoint")
    else:
        _same(contract["replay_source"], None, "formal/smoke checkpoint resume forbidden")
    if verify_files:
        _same(_json(ref), freeze, "new freeze bytes")
        validate_seed2_completion_freeze(freeze, verify_files=True)
    return copy.deepcopy(contract)


def make_seed2_completion_freeze(*, repo_root, campaign_id, output_root,
                                supplemental_authorization_reference, clean_archive_reference,
                                historical_freeze_reference, historical_control_completion_reference,
                                historical_control_ema_audit_reference, failed_campaign_reference,
                                partial_summary_reference):
    """CPU preparation; create one new frozen evidence namespace, never old output."""
    _same(os.environ.get("CUDA_VISIBLE_DEVICES"), "", "CPU preparation CUDA visibility")
    root = Path(repo_root).resolve(strict=True)
    _same(Path(__file__).resolve(), root / "src/sparse_rtdetr/baseline/training_v2b_seed2_completion.py",
          "completion validator checkout")
    output = Path(output_root).absolute()
    _require(output.resolve() == output and not output.exists(), "a unique unused output root is required")
    history, old = _build_history(
        historical_freeze_reference=historical_freeze_reference,
        historical_control_completion_reference=historical_control_completion_reference,
        historical_control_ema_audit_reference=historical_control_ema_audit_reference,
        failed_campaign_reference=failed_campaign_reference, partial_summary_reference=partial_summary_reference)
    source = source_snapshot(root)
    proof = source_extension_proof(old["source"], source)
    from .training_v2b_replication_gate import rebind_development_for_consumer
    development, provenance = rebind_development_for_consumer(old["cells"][CELL_ID]["development_binding"])
    common = copy.deepcopy(old["common"])
    common["code_files"] = source["code_files"]
    cell = copy.deepcopy(old["cells"][CELL_ID])
    cell["development_binding"] = development
    cell["invocations"] = invocation_plan(campaign_id, output)
    freeze = {
        "schema_version": 1, "kind": KIND, "campaign_id": campaign_id,
        "repo_root": str(root), "output_root": str(output), "source": source,
        "clean_archive_reference": copy.deepcopy(clean_archive_reference),
        "qualification_reference": copy.deepcopy(old["qualification_reference"]),
        "authorization_reference": copy.deepcopy(old["authorization_reference"]),
        "supplemental_authorization_reference": copy.deepcopy(supplemental_authorization_reference),
        "foundation_reference": copy.deepcopy(old["foundation_reference"]),
        "common": common, "cells": {CELL_ID: cell}, "stages": list(STAGES),
        "historical": history, "source_equivalence": proof, "execution_policy": copy.deepcopy(EXECUTION_POLICY),
    }
    validate_seed2_completion_freeze(freeze, verify_files=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    provenance_reference = write_exclusive_json(output / "development-provenance.json", provenance)
    reference = write_exclusive_json(output / "freeze.json", freeze)
    return {"freeze": freeze, "freeze_reference": reference,
            "development_provenance_reference": provenance_reference}
