"""One frozen 640 comparison: smoke/replay first, then sequential fixed epochs.

This controller never initializes CUDA or changes hardware itself. Every GPU
worker obtains its own native monitored admission. Results are immutable, and
any worker/monitor/identity failure ends this invocation without a retry.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any


class CampaignError(RuntimeError):
    """A frozen source, result, or sequential transition is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_reference(path: str | Path) -> dict:
    actual = Path(path).resolve(strict=True)
    raw = actual.read_bytes()
    return {"path": str(actual), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}


def read_reference(reference: dict) -> bytes:
    if type(reference) is not dict or set(reference) != {
            "path", "sha256", "size_bytes", "sha256_scope"}:
        raise CampaignError("file reference schema differs")
    path = Path(reference["path"]).resolve(strict=True)
    raw = path.read_bytes()
    observed = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    if observed != reference:
        raise CampaignError("bound file changed: " + reference["path"])
    return raw


def _json_reference(reference: dict) -> dict:
    from .training_v2b_evidence import strict_json_loads
    return strict_json_loads(read_reference(reference))


def _write_json(path: Path, payload: dict) -> dict:
    raw = _canonical(payload) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return file_reference(path)


def _git(root: Path, *args: str) -> str:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    proc = subprocess.run(["git", "-C", str(root), *args], env=env,
                          capture_output=True, text=True, timeout=30, check=False)
    if proc.returncode:
        raise CampaignError("read-only git observation failed: " + proc.stderr.strip())
    return proc.stdout.strip()


def source_snapshot(root: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    if _git(root, "status", "--porcelain=v1"):
        raise CampaignError("campaign source must be a clean committed checkout")
    files = {}
    names = _git(root, "ls-files", "-z").split("\0")
    for name in filter(None, names):
        path = root / name
        if name.startswith("artifacts/") or path.is_symlink():
            raise CampaignError("tracked source inventory crosses an artifact or symlink")
        if not path.resolve(strict=True).is_relative_to(root):
            raise CampaignError("tracked source escapes the checkout")
        files[name] = file_reference(path)
    return {"repo_root": str(root), "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "branch": _git(root, "branch", "--show-current"),
            "code_files": files, "inventory_sha256": _digest(files)}


def validate_frozen_source(expected: dict) -> None:
    observed = source_snapshot(expected["repo_root"])
    if observed != expected:
        raise CampaignError("source changed; the paired campaign cannot continue")


def _safe_id(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,55}", value) is None:
        raise CampaignError("campaign ID must be short, lowercase and unique")
    return value


def _data_path(path: str, *, expected_name: str | None = None) -> Path:
    candidate = Path(path).absolute()
    def protected(value: Path) -> bool:
        return any(part.casefold() in {"test", "confirmatory", "visdrone2019-det-test-dev",
                                      "visdrone2019-det-test-challenge"}
                   or part.casefold().startswith(("test_", "confirmatory_")) for part in value.parts)
    if protected(candidate) or (expected_name is not None and candidate.name != expected_name):
        raise CampaignError("data path is outside the authorized role")
    resolved = candidate.resolve(strict=True)
    if protected(resolved) or (expected_name is not None and resolved.name != expected_name):
        raise CampaignError("resolved data path is outside the authorized role")
    return resolved


def make_campaign(*, repo_root: str | Path, campaign_id: str, output_root: str | Path,
                  input_spec: dict, policy_bundle: dict,
                  authorization_reference: dict,
                  sampling_backend: str = "deterministic_gather") -> dict:
    """Prepare the immutable campaign without loading images or starting a GPU."""
    from .training_v2b import V2BConfig, logical_batch_indices
    from .training_v2b_development import build_development_binding
    import torch

    common_config = asdict(V2BConfig(sampling_backend=sampling_backend))
    root = Path(repo_root).resolve(strict=True)
    campaign_id = _safe_id(campaign_id)
    output = Path(output_root).absolute()
    if not output.is_relative_to(root / "artifacts" / "training"):
        raise CampaignError("campaign output must be a new directory under artifacts/training")
    if output.exists() or output.is_symlink():
        raise CampaignError("campaign output already exists")
    read_reference(authorization_reference)
    required = {"train_core", "development", "pretrained"}
    if type(input_spec) is not dict or set(input_spec) != required:
        raise CampaignError("exact train_core/development/pretrained input spec required")
    for role in ("train_core", "development"):
        value = input_spec[role]
        if set(value) != {"annotation", "manifest", "image_root"}:
            raise CampaignError("unexpected data input fields")
        for key in ("annotation", "manifest"):
            name = role + ("_coco.json" if key == "annotation" else "_manifest.json")
            _data_path(value[key]["path"], expected_name=name)
            read_reference(value[key])
        _data_path(value["image_root"])
    _data_path(input_spec["pretrained"]["path"], expected_name="ResNet18_vd_pretrained_from_paddle.pth")
    read_reference(input_spec["pretrained"])
    dev = input_spec["development"]
    development = build_development_binding(
        repo_root=root, annotation_file=dev["annotation"]["path"],
        annotation_sha256=dev["annotation"]["sha256"],
        manifest_file=dev["manifest"]["path"],
        manifest_sha256=dev["manifest"]["sha256"], image_root=dev["image_root"],
    )
    if development["image_count"] != 548:
        raise CampaignError("development cardinality differs from the reviewed split")
    source = source_snapshot(root)
    plans = {str(epoch): _digest([list(batch) for batch in logical_batch_indices(
        4869, seed=0, epoch=epoch, logical_batch_size=16)]) for epoch in range(1, 31)}
    value = {
        "schema_version": 1, "kind": "v2b_640_paired_campaign",
        "campaign_id": campaign_id, "output_root": str(output),
        "authorization_reference": authorization_reference,
        "source": source, "torch_version": str(torch.__version__),
        "common_config": common_config,
        "arms": {"A": {"physical_batch_size": 16, "accumulation_steps": 1},
                 "B": {"physical_batch_size": 8, "accumulation_steps": 2}},
        "num_workers": 2, "prefetch_factor": 2,
        "train_core": {**input_spec["train_core"], "expected_samples": 4869,
                       "expected_batches_per_epoch": 304},
        "pretrained": input_spec["pretrained"], "development_binding": development,
        "epoch_order_sha256": plans, "policy_bundle": policy_bundle,
        "expected_gpu_uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
        "smoke": {"logical_windows": 4, "checkpoint_after_window": 2,
                  "replay_windows": 2, "development_preview_images": 4},
        "control": {"epochs": 30, "fixed_evaluation_epochs": [10, 20, 30],
                    "primary_endpoint": 30, "nominal_recipe_epochs": 120,
                    "augmentation_stop_internal_epoch": 117,
                    "checkpoint_each_completed_epoch": True},
        "sequence": ["smoke-A", "replay-A", "smoke-B", "replay-B", "control-A", "control-B"],
        "formal_freeze": "before control-A; only B's reference to A's completed result is filled later",
        "on_any_failure": "STOP_NO_RETRY; no batch/config adaptation or automatic resume",
        "no_model_selection_certification": True,
        "forbidden": ["896", "960", "confirmatory", "test", "sparse", "data_protocol_changes"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    reference = _write_json(output / "campaign.json", value)
    return {"campaign_reference": reference, "campaign": value}


def worker_contract(campaign: dict, *, stage: str, arm: str, output_dir: Path,
                    paired_reference: dict | None = None,
                    replay_source: dict | None = None) -> dict:
    if stage not in {"smoke", "smoke_replay", "control30"} or arm not in {"A", "B"}:
        raise CampaignError("invalid fixed worker stage or arm")
    config = {**campaign["common_config"], **campaign["arms"][arm]}
    suffix = "control" if stage == "control30" else "smoke"
    return {
        "schema_version": 1, "kind": "v2b_640_control_worker",
        "campaign_id": campaign["campaign_id"], "stage": stage, "arm": arm,
        "run_id": campaign["campaign_id"] + "-" + suffix + "-" + arm.lower(),
        "repo_root": campaign["source"]["repo_root"], "output_dir": str(output_dir),
        "config": config, "num_workers": campaign["num_workers"],
        "prefetch_factor": campaign["prefetch_factor"], "torch_version": campaign["torch_version"],
        "code_files": campaign["source"]["code_files"], "train_core": campaign["train_core"],
        "pretrained": campaign["pretrained"], "development_binding": campaign["development_binding"],
        "epoch_order_sha256": campaign["epoch_order_sha256"], "policy_bundle": campaign["policy_bundle"],
        "expected_gpu_uuid": campaign["expected_gpu_uuid"],
        "paired_reference": paired_reference, "replay_source": replay_source,
    }


def scientific_template(contract: dict) -> dict:
    """A's future result hash is evidence, not a mutable scientific setting."""
    return json.loads(_canonical({key: value for key, value in contract.items()
                                  if key != "paired_reference"}))


def _worker_environment(campaign: dict) -> dict:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=campaign["expected_gpu_uuid"],
               MKL_THREADING_LAYER="GNU", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
    return env


class GuardianSupervisor:
    """The controller watches the watchdog even when CUDA blocks its worker.

    No sampling thread writes this heartbeat. Native sample ages and the
    guardian's own main-loop age are independently bounded by this process.
    This object can only stop the pidfd of the Popen child supplied at creation.
    """

    def __init__(self, worker_pid: int, hardware_dir: Path, *, run_id: str,
                 gpu_uuid: str, policy_sha256: str):
        from . import training_v2b_admission as admission
        from .training_v2b_hardware import _process_identity
        self._admission = admission
        self.pidfd = admission._open_pidfd(worker_pid)
        try:
            self.owner = _process_identity(worker_pid)
            if self.owner.get("readable") is not True:
                raise CampaignError("controller cannot bind its worker process")
        except BaseException:
            admission._stop_owner(self.pidfd)
            os.close(self.pidfd)
            raise
        self.guardian_pidfd = None
        self.root = Path(hardware_dir)
        self.expected = {"run_id": run_id, "gpu_uuid": gpu_uuid, "policy_sha256": policy_sha256}
        self.descriptor = None
        self.stream = None
        self.buffer = b""
        self.latest = None
        self.last_heartbeat_ns = None

    def check(self) -> None:
        from .training_v2b_evidence import strict_json_loads
        from .training_v2b_hardware import _process_identity
        if self.descriptor is None:
            path = self.root / "admission-start.json"
            if not path.exists():
                return  # Native code permits no CUDA before this atomic publication.
            desc = strict_json_loads(path.read_bytes())
            if desc.get("owner") != self.owner or any(desc.get(k) != v for k, v in self.expected.items()):
                raise CampaignError("guardian descriptor belongs to a different worker or policy")
            guardian = desc.get("guardian_identity")
            if (type(guardian) is not dict or guardian.get("readable") is not True
                    or guardian.get("pid") != desc.get("monitor_pid")
                    or _process_identity(guardian["pid"]) != guardian):
                raise CampaignError("guardian process identity is unavailable or changed")
            expected_path = self.root / "guardian-heartbeat.jsonl"
            if desc.get("heartbeat_path") != str(expected_path):
                raise CampaignError("guardian heartbeat escaped its owned evidence directory")
            for key, value in (("heartbeat_max_gap_seconds", 1.),
                               ("heartbeat_period_seconds", .1),
                               ("clock_max_gap_seconds", 1.), ("health_max_gap_seconds", 2.)):
                if type(desc.get(key)) not in (int, float) or desc[key] != value:
                    raise CampaignError("guardian supervision deadline differs from the frozen policy")
            self.guardian_pidfd = self._admission._open_pidfd(guardian["pid"])
            fd = os.open(expected_path, os.O_RDONLY | os.O_NOFOLLOW)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise CampaignError("guardian heartbeat is not a regular stream")
            self.stream = os.fdopen(fd, "rb", buffering=0)
            self.descriptor = desc
        if self._admission._pidfd_dead(self.guardian_pidfd):
            raise CampaignError("guardian process exited while its GPU worker is alive")
        raw = self.stream.read(256 * 1024)
        self.buffer += raw
        if len(self.buffer) > 2 * 1024 * 1024:
            raise CampaignError("guardian heartbeat frame is unbounded")
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            row = strict_json_loads(line)
            for key in ("run_id", "run_binding_sha256", "policy_sha256", "gpu_uuid", "owner"):
                if row.get(key) != self.descriptor.get(key):
                    raise CampaignError("guardian heartbeat identity differs")
            if row.get("guardian_identity") != self.descriptor["guardian_identity"]:
                raise CampaignError("guardian heartbeat emitter differs")
            stamp = row.get("monotonic_ns")
            if (type(stamp) is not int or
                    (self.last_heartbeat_ns is not None and stamp <= self.last_heartbeat_ns)):
                raise CampaignError("guardian heartbeat time is invalid or replayed")
            if self.last_heartbeat_ns is not None and stamp - self.last_heartbeat_ns > 1e9:
                raise CampaignError("guardian heartbeat contains a historical sampling gap")
            if (row.get("schema_version") != 1 or row.get("phase") not in
                    {"preload", "running", "finishing", "owner_exited"}):
                raise CampaignError("guardian heartbeat schema or phase differs")
            for key, maximum in (("last_clock_started_ns", 1.), ("last_health_started_ns", 2.)):
                sample_time = row.get(key)
                if type(sample_time) is not int or not 0 <= stamp - sample_time <= maximum * 1e9:
                    raise CampaignError("guardian heartbeat contains a native observation gap: " + key)
            self.last_heartbeat_ns = stamp
            self.latest = row
        if self.latest is None:
            raise CampaignError("guardian had no heartbeat before CUDA admission")
        now = time.monotonic_ns()
        for key, maximum in (("monotonic_ns", 1.), ("last_clock_started_ns", 1.),
                             ("last_health_started_ns", 2.)):
            stamp = self.latest.get(key)
            if type(stamp) is not int or not 0 <= now - stamp <= maximum * 1e9:
                raise CampaignError("independent controller detected guardian/native observation gap: " + key)

    def stop_owned_worker(self) -> None:
        self._admission._stop_owner(self.pidfd)

    def close_failed_guardian(self, *, grace_seconds: float = 3.) -> dict:
        """After owner exit, close only the guardian whose pidfd we bound."""
        if not self._admission._pidfd_dead(self.pidfd):
            raise CampaignError("guardian cleanup requires confirmed worker exit")
        if self.guardian_pidfd is None:
            return {"bound_guardian": False, "guardian_signalled": False}
        end = time.monotonic() + grace_seconds
        while not self._admission._pidfd_dead(self.guardian_pidfd) and time.monotonic() < end:
            time.sleep(.025)
        signalled = not self._admission._pidfd_dead(self.guardian_pidfd)
        if signalled:
            self._admission._stop_owner(self.guardian_pidfd)
            end = time.monotonic() + 1.
            while not self._admission._pidfd_dead(self.guardian_pidfd) and time.monotonic() < end:
                time.sleep(.01)
        return {"bound_guardian": True, "guardian_signalled": signalled,
                "guardian_identity": self.descriptor["guardian_identity"],
                "guardian_exited": self._admission._pidfd_dead(self.guardian_pidfd),
                "owner_exit_confirmed_before_guardian_cleanup": True}

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        for name in ("guardian_pidfd", "pidfd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)


def _run_worker(campaign: dict, contract: dict, contract_ref: dict, execution: Path) -> dict:
    from .training_v2b_admission import wait_for_monitored_finish, _validate_policy
    label = contract["stage"] + "-" + contract["arm"]
    root = Path(campaign["source"]["repo_root"])
    argv = [sys.executable, str(root / "tools/run_training_v2b_control.py"),
            "--contract", contract_ref["path"], "--contract-sha256", contract_ref["sha256"]]
    started = time.monotonic()
    scope = "train_core_30epoch" if contract["stage"] == "control30" else "paired_smoke"
    policy = _validate_policy(contract["policy_bundle"], scope, 43200 if scope == "train_core_30epoch" else 600)
    with open(execution / (label + ".stdout.log"), "xb", buffering=0) as stdout, \
            open(execution / (label + ".stderr.log"), "xb", buffering=0) as stderr:
        proc = subprocess.Popen(argv, cwd=root, env=_worker_environment(campaign),
                                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        supervisor = None
        try:
            supervisor = GuardianSupervisor(
                proc.pid, Path(contract["output_dir"]) / "native-hardware",
                run_id=contract["run_id"], gpu_uuid=contract["expected_gpu_uuid"],
                policy_sha256=policy["policy_sha256"],
            )
            launch_ref = _write_json(execution / (label + ".launch.json"), {
                "argv": argv, "pid": proc.pid, "owner": supervisor.owner,
                "contract": contract_ref, "monotonic_started": started,
                "stage": contract["stage"], "arm": contract["arm"],
            })
            deadline = started + (43500 if contract["stage"] == "control30" else 900)
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    raise CampaignError("owned worker exceeded its fixed wall limit; no retry")
                supervisor.check()
                time.sleep(.05)
            code = proc.returncode
            result_path = Path(contract["output_dir"]) / "worker-result.json"
            if code != 0 or not result_path.is_file():
                raise CampaignError(f"worker {label} failed (exit {code}); no retry")
            result_ref = file_reference(result_path)
            result = _json_reference(result_ref)
            if (result.get("status") != "PASS" or result.get("contract_reference") != contract_ref
                    or result.get("run_id") != contract["run_id"]):
                raise CampaignError("worker result does not match its exact completed invocation")
            monitor = wait_for_monitored_finish(result["monitor_reference"])
            if monitor.get("status") != "PASS":
                raise CampaignError("monitor did not close successfully after the CUDA worker exited")
            validate_frozen_source(campaign["source"])
            return {"result_reference": result_ref, "monitor": monitor,
                    "launch_reference": launch_ref, "elapsed_seconds": time.monotonic() - started}
        except BaseException as exc:
            if supervisor is not None:
                supervisor.stop_owned_worker()
            elif proc.poll() is None:
                # Construction failed before supervision. This is exclusively
                # our unreaped Popen child: its PID cannot have been reused.
                proc.kill()
            proc.wait(timeout=5.)
            guardian_cleanup = supervisor.close_failed_guardian() if supervisor is not None else None
            _write_json(execution / (label + ".supervisor-stop.json"), {
                "error_type": type(exc).__name__, "error": str(exc),
                "owner": supervisor.owner if supervisor is not None else {"pid": proc.pid},
                "only_owned_campaign_processes_signalled": True,
                "pidfd_supervision_established": supervisor is not None,
                "guardian_cleanup": guardian_cleanup,
                "no_retry": True, "monotonic_ns": time.monotonic_ns(),
            })
            raise
        finally:
            if supervisor is not None:
                supervisor.close()
            _write_json(execution / (label + ".exit.json"), {
                "returncode": proc.poll(), "pid": proc.pid, "contract": contract_ref,
                "stdout": file_reference(execution / (label + ".stdout.log")),
                "stderr": file_reference(execution / (label + ".stderr.log")),
            })


def run_campaign(campaign_reference: dict) -> dict:
    """Run once. No public fake ports, resume, automatic retry, or batch fallback."""
    from .training_v2b_control import compare_smoke_outputs
    campaign = _json_reference(campaign_reference)
    if campaign.get("kind") != "v2b_640_paired_campaign" or campaign.get("schema_version") != 1:
        raise CampaignError("unknown campaign schema")
    if (type(campaign.get("common_config")) is not dict or
            campaign["common_config"].get("sampling_backend") != "deterministic_gather"):
        raise CampaignError("paired formal campaign requires deterministic_gather; native is diagnostic only")
    validate_frozen_source(campaign["source"])
    execution = Path(campaign["output_root"]) / "execution"
    execution.mkdir(mode=0o700, exist_ok=False)
    (execution / "contracts").mkdir()
    finished = {}
    formal_started = False
    began = time.monotonic()
    try:
        for stage, arm, key in (("smoke", "A", "smoke-A"), ("smoke_replay", "A", "replay-A"),
                                ("smoke", "B", "smoke-B"), ("smoke_replay", "B", "replay-B")):
            validate_frozen_source(campaign["source"])
            pair = finished["smoke-A"]["result_reference"] if arm == "B" else None
            replay = finished["smoke-" + arm]["result_reference"] if stage == "smoke_replay" else None
            contract = worker_contract(campaign, stage=stage, arm=arm, output_dir=execution / key,
                                       paired_reference=pair, replay_source=replay)
            contract_ref = _write_json(execution / "contracts" / (key + ".json"), contract)
            finished[key] = _run_worker(campaign, contract, contract_ref, execution)
            _write_json(execution / (key + ".complete.json"), finished[key])
        comparison = compare_smoke_outputs(*(execution / name for name in
                                             ("smoke-A", "replay-A", "smoke-B", "replay-B")))
        comparison_ref = _write_json(execution / "smoke-pair-comparison.json", comparison)
        if comparison.get("status") != "PASS":
            raise CampaignError("paired smoke comparison failed; formal training is prohibited")
        a = worker_contract(campaign, stage="control30", arm="A", output_dir=execution / "control-A")
        b_template = worker_contract(campaign, stage="control30", arm="B", output_dir=execution / "control-B")
        frozen_b_template = scientific_template(b_template)
        validate_frozen_source(campaign["source"])
        freeze_ref = _write_json(execution / "formal-freeze.json", {
            "schema_version": 1, "campaign_reference": campaign_reference,
            "source": campaign["source"], "smoke_comparison": comparison_ref,
            "completed_smoke_workers": finished,
            "formal_A_template": a, "formal_B_template": frozen_b_template,
            "only_late_binding": "formal B paired_reference must equal completed formal A result_reference",
            "fixed_epochs": [10, 20, 30], "primary_endpoint": 30,
            "source_or_scientific_configuration_change_requires_stop": True,
        })
        formal_started = True
        a_ref = _write_json(execution / "contracts" / "control-A.json", a)
        finished["control-A"] = _run_worker(campaign, a, a_ref, execution)
        _write_json(execution / "control-A.complete.json", finished["control-A"])
        validate_frozen_source(campaign["source"])
        b = worker_contract(campaign, stage="control30", arm="B", output_dir=execution / "control-B",
                            paired_reference=finished["control-A"]["result_reference"])
        if (scientific_template(b) != frozen_b_template
                or _json_reference(freeze_ref)["formal_B_template"] != frozen_b_template):
            raise CampaignError("formal B template changed after A began")
        b_ref = _write_json(execution / "contracts" / "control-B.json", b)
        finished["control-B"] = _run_worker(campaign, b, b_ref, execution)
        _write_json(execution / "control-B.complete.json", finished["control-B"])
        validate_frozen_source(campaign["source"])
        result = {"schema_version": 1, "status": "PASS", "campaign_reference": campaign_reference,
                  "formal_freeze": freeze_ref, "workers": finished,
                  "elapsed_seconds": time.monotonic() - began, "formal_started": True,
                  "primary_endpoint": 30, "fixed_evaluation_epochs": [10, 20, 30],
                  "model_selection_certified": False, "high_resolution_authorized": False}
        _write_json(execution / "campaign-result.json", result)
        return result
    except BaseException as exc:
        result = {"schema_version": 1, "status": "STOP_NO_RETRY",
                  "campaign_reference": campaign_reference, "workers": finished,
                  "formal_started": formal_started, "elapsed_seconds": time.monotonic() - began,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "continue_B_or_splice_comparison_allowed": False}
        _write_json(execution / "campaign-result.json", result)
        raise
