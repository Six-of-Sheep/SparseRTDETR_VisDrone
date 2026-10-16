"""Minimal fresh-seed runtime; never calls the historical seed0 dispatcher."""
import os
import sys
from pathlib import Path


def _bootstrap(c=None):
    root = Path(c["source_root"]) if c else Path(__file__).resolve().parent
    path = str(root / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def material_binding(c):
    from sparse_rtdetr.localization.runio import digest
    return digest({"data_sha256": c["data_sha256"], "data": c["data"],
                   "pretrained": c["pretrained"], "materials": c.get("materials", [])})


def install(c=None):
    _bootstrap(c)
    # All fresh workers, supervisors, observers and archive readers install the
    # real laboratory resource channel before they read any progress record.
    from sparse_rtdetr.protocol_r18.control import install_resources
    install_resources()
    from sparse_rtdetr.protocol_r18 import transport
    from seed_protocol import PROTOCOL, install as install_protocol, save_audit, validate_audit
    install_protocol()
    if not getattr(transport, "_paired_audit_installed", False):
        previous_save = transport.save_audit; previous_validate = transport.validate_audit

        def save(out, identity, result, role):
            if identity.get("progress_protocol") == PROTOCOL:
                return save_audit(out, identity, result, role)
            return previous_save(out, identity, result, role)

        def validate(payload, identity):
            if identity.get("progress_protocol") == PROTOCOL:
                return validate_audit(payload, identity)
            return previous_validate(payload, identity)

        transport.save_audit = save; transport.validate_audit = validate
        transport._paired_audit_installed = True


def build_binding(seed):
    from sparse_rtdetr.protocol_r18.model import Config
    return Config(seed=seed).binding()


def load(path):
    import hashlib
    import json
    path = Path(path)
    c = json.loads(path.read_text(encoding="utf8"))
    canonical = json.dumps({k: v for k, v in c.items() if k != "contract_sha256"}, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != c.get("contract_sha256"):
        raise RuntimeError("paired execution contract digest changed")
    install(c)
    from sparse_rtdetr.localization.runio import digest, read, reference, require, sha
    from seed_protocol import TASK, ARM
    seed = c.get("seed")
    require(type(seed) is int and seed in (1, 2) and c.get("task_id") == TASK + "-seed" + str(seed)
            and c.get("arm") == ARM, "paired execution task/seed/arm differs")
    root = Path(c["source_root"])
    require(root == Path(__file__).resolve().parent and root.is_absolute() and not root.is_symlink(), "paired executable source root differs")
    require(sha(c["source_manifest"]) == c["source_sha256"], "paired source manifest SHA differs")
    manifest = read(c["source_manifest"])
    for row in manifest["files"]:
        p = Path(row["path"])
        require(not p.is_absolute() and ".." not in p.parts, "paired source manifest path escapes")
        target = root / p
        require(target.resolve().is_relative_to(root.resolve()) and not target.is_symlink(), "paired source symlink/path escaped")
        require(reference(target) == {"path": str(target), "bytes": row["bytes"], "sha256": row["sha256"]}, "paired source changed: " + row["path"])
    require(c["config"] == build_binding(seed) and c.get("scope", f"formal_R18_seed{seed}_e120") == f"formal_R18_seed{seed}_e120", "paired common recipe/seed scope differs")
    require(Path(c["job_root"]).name == "seed" + str(seed) and Path(c["runtime_root"]).name == "seed" + str(seed), "paired output/runtime seed directory differs")
    require(Path(c["job_root"]).is_absolute() and Path(c["runtime_root"]).is_absolute()
            and c["job_root"] != c["runtime_root"], "paired SSD/output isolation differs")
    require(c.get("workload") == {"unique_updates": 2, "actual_updates": 3, "sequence": [1, 2, 2]}, "paired fixed3 acceptance workload differs")
    require(isinstance(c.get("fixed_ids"), list) and len(c["fixed_ids"]) == 2
            and all(len(i) == len(set(i)) == 16 for i in c["fixed_ids"]), "paired fixed input windows absent")
    ids = c.get("development_ids", c.get("full_image_ids"))
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 548, "paired development identities absent")
    require(isinstance(c.get("fixed64"), list) and len(c["fixed64"]) == len(set(c["fixed64"])) == 64
            and set(c["fixed64"]) <= set(ids), "paired fixed64 identities differ")
    from sparse_rtdetr.protocol_r18.model import PRETRAINED_SHA
    require(sha(c["pretrained"]["path"]) == PRETRAINED_SHA, "paired certified ImageNet bytes changed")
    for ref in [*c.get("materials", []), c["plan"]]:
        # Pretrained records may carry a descriptive identity in addition to the
        # immutable path/byte/SHA tuple; no large original is copied locally.
        actual = reference(ref["path"])
        require(actual["sha256"] == ref["sha256"] and actual["bytes"] == ref.get("bytes", ref.get("size_bytes")), "paired frozen material bytes changed: " + ref["path"])
    require(sha(c["data_verification"]) == c["data_sha256"] and read(c["data_verification"])["status"] == "PASS", "paired data verification closure changed")
    if c.get("material_binding_sha256") is not None:
        require(c["material_binding_sha256"] == material_binding(c), "paired material binding digest differs")
    if c.get("GPU_execution_authorized"):
        require(c.get("formal_training_authorized") is True, "paired formal authority missing")
        proof = c["CPU_gate"]; require(reference(proof["path"]) == proof, "paired CPU gate bytes changed")
        gate = read(proof["path"])
        require(gate.get("status") == "PASS" and gate.get("source_sha256") == c["source_sha256"], "paired source-bound CPU gate absent")
    return c


def identity(c, mode):
    install(c)
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.runio import read, reference, require
    from seed_protocol import PROTOCOL, INFERENCE, SAHI
    result = {**{k: c[k] for k in ("task_id", "seed", "arm", "source_sha256", "contract_sha256", "data_sha256")},
              "plan_sha256": c["plan"]["sha256"], "material_binding_sha256": material_binding(c),
              "progress_protocol": PROTOCOL, "mode": mode, "output_root": str(Path(c["job_root"]) / mode),
              "fixed_ids": c["fixed_ids"], "fixed64": c["fixed64"], "supervisor_pid": os.getpid(),
              "supervisor_start": guard.identity(os.getpid())["start"],
              "starting_cursor": 36480 if mode in (*INFERENCE, *SAHI, "evaluation") else 0}
    if mode == "cpu-protocol" and c.get("CPU_protocol"):
        spec = c["CPU_protocol"]
        result.update(CPU_protocol_only=True, protocol_mode=spec["mode"], starting_cursor=spec["starting_cursor"])
    if mode in (*INFERENCE, *SAHI, "evaluation"):
        lock = read(Path(c["job_root"]) / "formal/endpoint-lock.json")
        require(lock["status"] == "R18_COMMON_E120_EMA_LOCKED" and lock.get("seed") == c["seed"]
                and lock["epoch"] == 120 and lock["updates"] == 36480
                and lock["source_sha256"] == c["source_sha256"] and lock["contract_sha256"] == c["contract_sha256"], "paired endpoint identity/seed differs")
        require(reference(Path(c["job_root"]) / "formal/binding.json") == lock["binding"], "paired endpoint binding bytes changed")
        result["evaluation_authorization"] = {"checkpoint_sha256": lock["checkpoint"]["sha256"],
                                               "image_ids": c.get("development_ids", c.get("full_image_ids")), "budgets": [600, 300]}
    return result


def make_engine(c, runtime, scope=None):
    install(c)
    from sparse_rtdetr.protocol_r18.worker import make_engine as native
    return native(c, runtime, scope or f"formal_R18_seed{c['seed']}_e120")
