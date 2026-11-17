"""Compare the r17 retrained native arm A0 against the retained r10 original-G0 arm.

Read-only control for the packaging: the native arm of this round must reproduce the r10
G0 trajectory bitwise (same parent, data order, augmentation seeds, optimizer, dual EMA and
budget; no ranking term is installed for A0).  Discovers the formal directories by glob,
requires the same checkpoint and audit name sets, compares the deterministic fields of every
per-update audit and every tensor of every checkpoint, and compares the behaviour identity
only on the keys both rounds carry (r17 adds the ranking declaration).

usage: <frozen python> -B rank_verify.py <r17_job_root> <r10_round_root> <out.json> [ref_arm]
"""
import json
import sys
from pathlib import Path

import torch

DETERMINISTIC = ("update", "microbatch_losses", "loss", "normalizers", "gradient_norm_before_clip",
                 "gradient_sha256_before_clip", "output_sha256", "dn_noise_sha256", "input_sha256",
                 "finite", "optimizer_steps", "ema_updates", "coefficients")
STATE_KEYS = ("model", "ema", "ema_alt", "optimizer", "engine", "rng")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def formal_dir(root, arm):
    hits = sorted(Path(root).glob("formal-%s" % arm))
    assert len(hits) == 1, ("formal dir", str(root), arm, [str(h) for h in hits])
    return hits[0]


def compare_checkpoint(new_path, ref_path, report, label):
    new = torch.load(new_path, map_location="cpu", weights_only=True)
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    assert set(new) == set(ref), ("checkpoint schema differs", label, sorted(set(new) ^ set(ref)))
    differences = []
    for key in sorted(new):
        if key == "behavior":
            common = set(new[key]) & set(ref[key])
            if any(new[key][name] != ref[key][name] for name in common):
                differences.append("behavior")
            report.setdefault("behaviour_keys_only_in_new", sorted(set(new[key]) - set(ref[key])))
            continue
        if key == "selection_rank":
            continue
        a, b = new[key], ref[key]
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append(key + ":keys")
                continue
            for name in sorted(a):
                if isinstance(a[name], torch.Tensor):
                    if not torch.equal(a[name], b[name]):
                        differences.append("%s.%s" % (key, name))
                elif a[name] != b[name]:
                    differences.append("%s.%s" % (key, name))
        elif isinstance(a, torch.Tensor):
            if not torch.equal(a, b):
                differences.append(key)
        elif a != b:
            differences.append(key)
    report["checkpoints"][label] = {"path": str(new_path), "reference": str(ref_path),
                                    "differences": differences, "exact": not differences}
    return not differences


def compare_audits(new_dir, ref_dir, report):
    new_names = sorted(p.name for p in (new_dir / "audits").glob("update-*.json"))
    ref_names = sorted(p.name for p in (ref_dir / "audits").glob("update-*.json"))
    assert new_names == ref_names, ("audit sets differ", len(new_names), len(ref_names))
    mismatches = []
    for name in new_names:
        a = load(new_dir / "audits" / name)["result"]
        b = load(ref_dir / "audits" / name)["result"]
        for key in DETERMINISTIC:
            if key not in a or key not in b:
                continue
            if a[key] != b[key]:
                mismatches.append("%s:%s" % (name, key))
    report["audits"] = {"files": len(new_names), "compared_fields": list(DETERMINISTIC),
                        "mismatches": mismatches[:40], "mismatch_count": len(mismatches),
                        "exact": not mismatches}
    return not mismatches


def main():
    job = Path(sys.argv[1]).resolve()
    ref_root = Path(sys.argv[2]).resolve()
    out = Path(sys.argv[3])
    ref_arm = sys.argv[4] if len(sys.argv) > 4 else "G0"
    new_dir = formal_dir(job, "A0")
    hits = sorted(ref_root.glob("production-*/formal-%s" % ref_arm))
    assert len(hits) == 1, ("reference formal dir", ref_arm, [str(h) for h in hits])
    ref_dir = hits[0]
    report = {"schema": "r18-selection-rank-reproduction-v1", "new": str(new_dir), "reference": str(ref_dir),
              "reference_arm": ref_arm, "checkpoints": {}, "behaviour_keys_only_in_new": []}
    new_ckpts = sorted(p.name for p in new_dir.glob("checkpoint-FT*.pth"))
    ref_ckpts = sorted(p.name for p in ref_dir.glob("checkpoint-FT*.pth"))
    assert new_ckpts == ref_ckpts and new_ckpts, ("checkpoint sets differ", new_ckpts, ref_ckpts)
    exact = all(compare_checkpoint(new_dir / name, ref_dir / name, report, name) for name in new_ckpts)
    exact = compare_audits(new_dir, ref_dir, report) and exact
    report["reproduction_holds"] = bool(exact)
    report["note"] = ("A0 installs no ranking term; identical parent, sampler, augmentation, optimizer, dual EMA "
                      "and budget are expected to reproduce the r10 G0 trajectory bitwise. The behaviour identity "
                      "is compared on the keys both rounds carry, because r17 adds the ranking declaration.")
    out.write_bytes(json.dumps(report, indent=1, ensure_ascii=False).encode("utf-8"))
    print(json.dumps({k: v for k, v in report.items() if k in
                      ("reproduction_holds", "checkpoints", "audits", "behaviour_keys_only_in_new")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
