"""Read-only scan of the frozen audit stream for the tiny-band geometry round.

usage: python3 tools/tiny_verify.py <job_root> <out_json>

Scans every published update audit of the three formal arms, aggregates the declared
observation fields per arm and per family class (final / aux / enc / dn), checks internal
consistency (the native arm records applied == native and no shifts; A2 records no shifts;
A1 records at least one shifted call) and writes the summary the closeout receipt embeds.
"""
import hashlib
import json
import sys
from pathlib import Path

ARMS = ("A0", "A1", "A2")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def family(head):
    if head == "final":
        return "final"
    for prefix in ("aux", "enc", "dn"):
        if str(head).startswith(prefix):
            return prefix
    return "other"


def main():
    job = Path(sys.argv[1])
    out = Path(sys.argv[2])
    summary = {}
    for arm in ARMS:
        audits = sorted((job / ("formal-%s" % arm) / "audits").glob("update-*.json"))
        assert audits, "no audits for " + arm
        rows = 0
        totals = {"pairs": 0, "tiny": 0, "zero_iou": 0, "native_sum": 0.0, "applied_sum": 0.0, "shifted": 0}
        per_family = {}
        for path in audits:
            result = load(path)["result"]
            for diag in result.get("normalizers", ()):
                for row in diag:
                    if "geo_pairs" not in row:
                        continue
                    rows += 1
                    fam = family(row.get("head"))
                    agg = per_family.setdefault(fam, {"rows": 0, "pairs": 0, "tiny": 0,
                                                      "native_sum": 0.0, "applied_sum": 0.0, "shifted": 0})
                    shifted = 1 if row.get("geo_shift") is not None else 0
                    agg["rows"] += 1
                    pairs = int(row["geo_pairs"])
                    tiny = int(row["geo_tiny_pairs"])
                    agg["pairs"] += pairs
                    agg["tiny"] += tiny
                    agg["native_sum"] += float(row["geo_tiny_native_sum"])
                    agg["applied_sum"] += float(row["geo_tiny_applied_sum"])
                    agg["shifted"] += shifted
                    totals["pairs"] += pairs
                    totals["tiny"] += tiny
                    totals["zero_iou"] += int(row["geo_tiny_zero_iou"])
                    totals["native_sum"] += float(row["geo_tiny_native_sum"])
                    totals["applied_sum"] += float(row["geo_tiny_applied_sum"])
                    totals["shifted"] += shifted
        if arm == "A0":
            assert totals["applied_sum"] == totals["native_sum"], "native arm applied != native"
            assert totals["shifted"] == 0, "native arm recorded a shift"
        elif arm == "A1":
            assert totals["shifted"] >= 1, "A1 shift ledger empty"
        else:
            assert totals["shifted"] == 0, "A2 recorded a shift"
        totals["tiny_fraction"] = (totals["tiny"] / totals["pairs"]) if totals["pairs"] else None
        summary[arm] = {"audits_scanned": len(audits), "geo_rows": rows,
                        "totals": totals, "per_family": per_family}
    body = json.dumps({"schema": "r18-tiny-geometry-training-signal-v1", "arms": summary,
                       "note": ("observation-only fields from the frozen audit stream; the native arm records "
                                "applied == native, A1 additionally records its shift ledger, and the DN families "
                                "carry no observations by declaration")},
                      indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "VERIFY-OK", "path": str(out), "bytes": len(body),
                      "sha256": hashlib.sha256(body.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
