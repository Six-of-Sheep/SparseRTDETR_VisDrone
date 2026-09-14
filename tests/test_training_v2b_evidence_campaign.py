"""CPU-only fail-closed tests for the resolution evidence controller."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from sparse_rtdetr.baseline import training_v2b_evidence_campaign as campaign
from sparse_rtdetr.baseline.training_v2b_campaign import CampaignError, _write_json


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "source"
    output = root / "artifacts" / "evidence" / "fixture"
    output.mkdir(parents=True)
    value = {
        "campaign_id": "fixture",
        "output_root": str(output),
        "source": {"repo_root": str(root)},
        "checkpoints": {"640": {"name": "640-ema30"}, "896": {"name": "896-ema30"}},
        "development_bindings": {"640": {"size": 640}, "896": {"size": 896}},
        "official_gt_bindings": {"640": {"raw_gt": "same"}, "896": {"raw_gt": "same"}},
        "policy_bundle": {"clock_max_mhz": 1500},
        "expected_gpu_uuid": "fixture-uuid",
        "authorization_reference": {"sha256": campaign.AUTHORIZATION_SHA256},
        "cache_policy": "ema_geometry_replay_9120_v1",
        "evaluator_evidence_reference": {"scope": "fixture"},
    }
    value["source"]["code_files"] = {"fixture.py": {"sha256": "fixture"}}
    reference = _write_json(output / "campaign.json", value)
    monkeypatch.setattr(campaign, "validate_evidence_campaign", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda *a, **k: None)
    return value, reference


def test_cross_template_uses_training_checkpoint_and_evaluation_geometry_independently(frozen):
    value, _ = frozen
    for cell, (train, evaluate) in campaign.CELL_SIZES.items():
        contract = campaign.cross_worker_contract(value, cell)
        assert contract["checkpoint"] == {"name": f"{train}-ema30"}
        assert contract["development_binding"] == {"size": evaluate}
        assert (contract["source_training_size"], contract["evaluation_size"]) == (train, evaluate)
        assert contract["official_gt_binding"] == {"raw_gt": "same"}
        contract["checkpoint"]["name"] = "mutated"
        assert value["checkpoints"][str(train)]["name"] == f"{train}-ema30"
    with pytest.raises(CampaignError, match="four resolution cells"):
        campaign.cross_worker_contract(value, "t896_e960")


def _fake_completion(value, contract, contract_ref, execution):
    path = Path(contract["output_dir"])
    path.mkdir()
    result_ref = _write_json(path / "worker-result.json", {
        "status": "PASS", "run_id": contract["run_id"], "contract_reference": contract_ref,
    })
    return {"result_reference": result_ref, "monitor": {"status": "PASS"}}


def test_controller_executes_all_four_once_and_rejects_reentry(frozen, monkeypatch):
    value, reference = frozen
    cells = []

    def worker(*args):
        cells.append(args[1]["cell_key"])
        return _fake_completion(*args)

    monkeypatch.setattr(campaign, "_run_evaluation", worker)
    result = campaign.run_evidence_campaign(value, campaign_reference=reference)
    assert result["status"] == "PASS"
    assert cells == list(campaign.CELL_SIZES)
    assert list(result["completed"]) == list(campaign.CELL_SIZES)
    assert result["training_started"] is False
    with pytest.raises(FileExistsError):
        campaign.run_evidence_campaign(value, campaign_reference=reference)
    assert cells == list(campaign.CELL_SIZES)


def test_second_cell_failure_preserves_first_and_never_starts_remaining(frozen, monkeypatch):
    value, reference = frozen
    cells = []

    def worker(*args):
        cell = args[1]["cell_key"]
        cells.append(cell)
        if cell == "t640_e896":
            raise CampaignError("synthetic hardware stop")
        return _fake_completion(*args)

    monkeypatch.setattr(campaign, "_run_evaluation", worker)
    with pytest.raises(CampaignError, match="synthetic hardware stop"):
        campaign.run_evidence_campaign(value, campaign_reference=reference)
    result = json.loads((Path(value["output_root"]) / "execution/campaign-result.json").read_text())
    assert result["status"] == "STOP_NO_RETRY"
    assert cells == ["t640_e640", "t640_e896"]
    assert list(result["completed"]) == ["t640_e640"]
    assert result["automatic_retry"] is False


def test_source_drift_after_first_cell_prevents_second_launch(frozen, monkeypatch):
    value, reference = frozen
    cells = []
    changed = False

    def source_check(*args):
        if changed:
            raise CampaignError("source drift")

    def worker(*args):
        nonlocal changed
        cells.append(args[1]["cell_key"])
        answer = _fake_completion(*args)
        changed = True
        return answer

    monkeypatch.setattr(campaign, "validate_frozen_source", source_check)
    monkeypatch.setattr(campaign, "_run_evaluation", worker)
    with pytest.raises(CampaignError, match="source drift"):
        campaign.run_evidence_campaign(value, campaign_reference=reference)
    assert cells == ["t640_e640"]
    result = json.loads((Path(value["output_root"]) / "execution/campaign-result.json").read_text())
    assert list(result["completed"]) == cells


def test_campaign_memory_mutation_fails_before_creating_execution(frozen):
    value, reference = frozen
    changed = copy.deepcopy(value)
    changed["cache_policy"] = "canonical_geometry_all_cells_v1"
    with pytest.raises(CampaignError, match="campaign bytes"):
        campaign.run_evidence_campaign(changed, campaign_reference=reference)
    assert not (Path(value["output_root"]) / "execution").exists()


def test_output_cannot_escape_or_follow_symlink(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    allowed = root / "artifacts/evidence"
    allowed.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (allowed / "alias").symlink_to(outside, target_is_directory=True)
    for output in (outside / "run", allowed, allowed / "alias/run"):
        with pytest.raises(CampaignError, match="canonical"):
            campaign._output_path(root, output)


def test_uniform_development_rejects_different_images_gt_or_non_resize_policy():
    left = {"annotation": {"sha256": "a"}, "manifest": {"sha256": "b"},
            "image_manifest_sha256": "images", "image_order_sha256": "order",
            "source": {"evaluator": "same"},
            "policy": {"input_size": [640, 640], "transform": [{"type": "Resize", "size": [640, 640]}],
                       "batch_size": 4},
            "binding_sha256": "left"}
    right = copy.deepcopy(left)
    right["policy"]["input_size"] = [896, 896]
    right["policy"]["transform"][0]["size"] = [896, 896]
    right["binding_sha256"] = "right"
    value = {"development_bindings": {"640": left, "896": right},
             "official_gt_bindings": {"640": {"gt": "same"}, "896": {"gt": "same"}}}
    campaign.validate_uniform_development(value)
    for field in ("image_manifest_sha256", "image_order_sha256", "manifest"):
        changed = copy.deepcopy(value)
        changed["development_bindings"]["896"][field] = "different"
        with pytest.raises(CampaignError, match="cross-cell development"):
            campaign.validate_uniform_development(changed)
    changed = copy.deepcopy(value)
    changed["official_gt_bindings"]["896"]["gt"] = "different"
    with pytest.raises(CampaignError, match="complete primary GT"):
        campaign.validate_uniform_development(changed)
    changed = copy.deepcopy(value)
    changed["development_bindings"]["896"]["policy"]["batch_size"] = 8
    with pytest.raises(CampaignError, match="changes only"):
        campaign.validate_uniform_development(changed)
