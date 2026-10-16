"""Read-only Linux cloud snapshot; stdlib only, no CUDA compute or acceptance.

Usage: python -I cloud_probe.py --output NEW.json --disk /root/autodl-tmp
Optional --min-ram-gib / --min-cpus are operator screening thresholds, NOT
certified training minima. Exit 0 means receipt written, never phase acceptance.
No settings are changed. Output is exclusive-create; existing evidence is kept.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import time

GIB = 1024 ** 3


def unknown(reason, status="UNKNOWN"):
    return {"status": status, "reason": reason}


def observed(value, source):
    return {"status": "OBSERVED", "value": value, "source": str(source)}


def read_text(path):
    try:
        with open(path, encoding="utf-8") as stream:
            text = stream.read(262145)
        if len(text) > 262144:
            return None
        return text.strip()
    except (OSError, UnicodeError):
        return None


def positive(text):
    try:
        value = float(text)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def cpuset_count(text):
    if not text:
        return None
    intervals = []
    try:
        for piece in text.split(","):
            if not re.fullmatch(r"\d+(?:-\d+)?", piece):
                return None
            pair = list(map(int, piece.split("-")))
            a, b = pair[0], pair[-1]
            if a > b:
                return None
            intervals.append((a, b))
        intervals.sort()
        total, end = 0, -1
        for a, b in intervals:
            total += max(0, b - max(a, end + 1) + 1)
            end = max(end, b)
        return total
    except ValueError:
        return None


def unescape_mount(text):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), text)


def resolve_groups(cgroup, mountinfo):
    """Resolve only unambiguous visible mount roots, never assume host root."""
    memberships = []
    for line in (cgroup or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            memberships.append((set(parts[1].split(",")), parts[2]))
    groups = {}
    for line in (mountinfo or "").splitlines():
        halves = line.split(" - ", 1)
        if len(halves) != 2:
            continue
        left, right = halves[0].split(), halves[1].split()
        if len(left) < 6 or len(right) < 3 or right[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = map(unescape_mount, (left[3], left[4]))
        version = 2 if right[0] == "cgroup2" else 1
        controllers = {"memory", "cpu", "cpuset", "pids"} if version == 2 else set(right[2].split(","))
        for member_controllers, member in memberships:
            if version == 2 and member_controllers != {""}:
                continue
            if version == 1 and not member_controllers.intersection(controllers):
                continue
            if not member.startswith("/") or ".." in PurePosixPath(member).parts:
                continue
            try:
                relative = PurePosixPath(member).relative_to(PurePosixPath(root))
            except ValueError:
                continue
            leaf = Path(mount).joinpath(*relative.parts)
            for controller in controllers.intersection({"memory", "cpu", "cpuset", "pids"}):
                if version == 2 or controller in member_controllers:
                    groups[controller] = (version, Path(mount), leaf)
    return groups


def walk_limits(group, filenames, reader=read_text):
    version, mount, leaf = group
    rows = []
    current = leaf
    while True:
        rows.append({"path": str(current), "values": {name: reader(current / name) for name in filenames}})
        if current == mount:
            break
        if mount not in current.parents:
            return []
        current = current.parent
    return rows


def resource_snapshot(proc=Path("/proc"), reader=read_text, affinity=None):
    groups = resolve_groups(reader(proc / "self/cgroup"), reader(proc / "self/mountinfo"))
    result = {
        "scope": "visible cgroup ancestors plus process affinity, NOT host free/CPU count",
        "hidden_ancestor_limits": unknown("Cgroup namespace/mount may hide stricter ancestors; host-wide visibility not proven"),
        "effective_resources_certified": False,
    }
    for controller in ("memory", "cpu", "cpuset", "pids"):
        if controller not in groups:
            result[controller] = unknown("No resolvable cgroup mount/membership")
            continue
        version = groups[controller][0]
        names = {
            "memory": ("memory.max", "memory.current") if version == 2 else ("memory.limit_in_bytes", "memory.usage_in_bytes"),
            "cpu": ("cpu.max",) if version == 2 else ("cpu.cfs_quota_us", "cpu.cfs_period_us"),
            "cpuset": ("cpuset.cpus.effective",) if version == 2 else ("cpuset.cpus",),
            "pids": ("pids.max", "pids.current"),
        }[controller]
        rows = walk_limits(groups[controller], names, reader)
        bounds, remaining, missing = [], [], False
        for row in rows:
            values = row["values"]
            raw = values[names[0]]
            if controller == "cpuset":
                value = cpuset_count(raw)
                # v1 empty cpuset inherits from a visible ancestor.
                if value is None and not (version == 1 and raw == ""):
                    missing = True
            elif controller == "cpu":
                parts = raw.split() if raw else []
                quota, period = (parts if len(parts) == 2 else (None, None)) if version == 2 else (raw, values[names[1]])
                value = None
                period_value = positive(period)
                if period_value is None or period_value <= 0:
                    missing = True
                elif quota not in ("max", "-1"):
                    q = positive(quota)
                    if q is None or q <= 0:
                        missing = True
                    else:
                        value = q / period_value
            else:
                value = positive(raw)
                if raw == "max" or (version == 1 and value is not None and value >= 2 ** 60):
                    value = None
                elif value is None:
                    missing = True
                usage = positive(values[names[1]])
                if value is not None and usage is not None:
                    remaining.append(max(0, value - usage))
            if value is not None:
                bounds.append(value)
        result[controller] = {
            "status": "UNKNOWN" if missing or not rows else "OBSERVED",
            "version": version,
            "visible_upper_bound": min(bounds) if bounds else None,
            "visible_headroom_upper_bound": min(remaining) if remaining else None,
            "ancestors": rows,
            "note": "Null bound is unlimited-or-unknown, NOT infinite available resources. Visible bounds may overestimate hidden-parent capacity; sibling usage changes headroom.",
        }
    if affinity is None:
        try:
            affinity = len(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            affinity = None
    result["affinity_logical_cpus"] = affinity
    cpu_bounds = [result[c].get("visible_upper_bound") for c in ("cpu", "cpuset")]
    cpu_bounds.append(affinity)
    cpu_bounds = [v for v in cpu_bounds if v is not None]
    result["cpu_visible_upper_bound"] = min(cpu_bounds) if cpu_bounds else None
    result["memory_visible_upper_bound_bytes"] = result["memory"].get("visible_upper_bound")
    return result


GPU_FIELDS = ("uuid", "name", "driver_version", "memory.total", "memory.free", "memory.used", "temperature.gpu", "utilization.gpu", "clocks.current.graphics", "clocks.current.memory", "power.draw", "power.limit")


def bounded_command(argv, timeout=5, runner=subprocess.run):
    try:
        completed = runner(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)
        if completed.returncode != 0:
            # Do not copy arbitrary diagnostics into the receipt.
            return {"status": "UNAVAILABLE", "reason": "command nonzero", "returncode": completed.returncode}
        if len(completed.stdout) > 65536:
            return unknown("command output exceeds accepted size")
        return {"status": "OBSERVED", "stdout": completed.stdout}
    except FileNotFoundError:
        return unknown("command missing", "UNAVAILABLE")
    except subprocess.TimeoutExpired:
        return unknown("command timeout")
    except (OSError, UnicodeError):
        return unknown("command execution/read failure")


def parse_gpu(text):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if len(fields) != len(GPU_FIELDS):
            return unknown("Unexpected nvidia-smi CSV schema")
        row = {}
        for index, (name, raw) in enumerate(zip(GPU_FIELDS, fields)):
            raw = raw.strip()
            if index < 3:
                valid = bool(raw) and raw.lower() not in ("n/a", "[n/a]", "unknown", "not supported", "[not supported]")
                if name == "uuid":
                    valid = valid and raw.startswith("GPU-")
                value = raw if valid else None
            else:
                value = positive(raw)
                if name == "memory.total" and value == 0:
                    value = None
                if name == "utilization.gpu" and value is not None and value > 100:
                    value = None
            row[name] = observed(value, "nvidia-smi query") if value is not None else unknown("Invalid/N/A/not supported", "UNAVAILABLE")
        rows.append(row)
    return {"status": "OBSERVED" if rows else "UNAVAILABLE", "devices": rows,
            "units": "memory MiB; temperature C; utilization percent; clocks MHz; power W",
            "scope": "idle point observation, no compute/health/acceptance proof"}


def gpu_snapshot(runner=subprocess.run):
    result = bounded_command(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"], runner=runner)
    return parse_gpu(result["stdout"]) if result["status"] == "OBSERVED" else result


def filesystem(path):
    try:
        usage = shutil.disk_usage(path)
        return observed({"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}, path)
    except OSError:
        return unknown("Path inaccessible", "UNAVAILABLE")


def screen(resources, min_ram_gib=None, min_cpus=None):
    ram = resources.get("memory_visible_upper_bound_bytes")
    cpu = resources.get("cpu_visible_upper_bound")
    blockers = []
    if ram is not None and ram <= 2 * GIB:
        blockers.append("At most 2GiB visible RAM: no heavy deployment/P2")
    if cpu is not None and cpu <= 0.5:
        blockers.append("At most 0.5 CPU visible quota: no heavy deployment/P2")
    if min_ram_gib is not None and ram is not None and ram < min_ram_gib * GIB:
        blockers.append("Below explicit operator RAM screening threshold")
    if min_cpus is not None and cpu is not None and cpu < min_cpus:
        blockers.append("Below explicit operator CPU screening threshold")
    return {"status": "BLOCKED" if blockers else "REVIEW_REQUIRED", "blockers": blockers,
            "operator_min_ram_gib": min_ram_gib, "operator_min_cpus": min_cpus,
            "certified_training_minimum": None,
            "note": "No automatic admission: visible upper bounds cannot certify hidden quotas, workload peaks, disk requirements, budget or phase prerequisites."}


def snapshot(disks, min_ram_gib=None, min_cpus=None):
    resources = resource_snapshot()
    gpu = gpu_snapshot()
    mandatory = ("uuid", "name", "memory.total", "memory.free", "temperature.gpu", "utilization.gpu")
    missing_gpu = ["GPU device inventory"] if not gpu.get("devices") else []
    for index, row in enumerate(gpu.get("devices", [])):
        missing_gpu.extend("gpu%d.%s" % (index, key) for key in mandatory if row[key]["status"] != "OBSERVED")
    return {
        "schema": "cloud-capability-snapshot-v1", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": platform.python_version(), "os": platform.system(), "sources": ["/proc/self/cgroup", "/proc/self/mountinfo", "visible cgroup files", "sched_getaffinity", "disk_usage", "bounded nvidia-smi query"], "environment_variables_collected": False},
        "performance_policy": {"mode": "PLATFORM_DEFAULT", "settings_written": False, "old_gpu_1500MHz": "NOT_APPLICABLE", "old_cpu_PL1_PL2": "NOT_APPLICABLE", "old_no_turbo": "NOT_APPLICABLE", "default_configuration_verified": False},
        "resources": resources, "resource_screen": screen(resources, min_ram_gib, min_cpus),
        "disks": {str(path): filesystem(path) for path in disks}, "shm": filesystem("/dev/shm"),
        "gpu": gpu,
        "optional_cloud_telemetry": {key: unknown("Not collected by this minimal probe; platform exposure/permission not established") for key in ("RAPL", "cpu_temperature", "ECC", "MCE", "kernel_log_completeness")},
        "future_gpu_phase": {"status": "BLOCKED" if missing_gpu else "NOT_RUN", "mandatory_missing_telemetry": missing_gpu, "note": "Missing required identity/free-memory/temperature observations block future GPU guard acceptance; observations alone never authorize compute."},
        "phase_acceptance": {"P1": "NOT_EVALUATED", "P2": "NOT_RUN", "GPU": "NOT_RUN", "formal_training": "NOT_AUTHORIZED_BY_PROBE", "native_monitor": "NOT_APPLICABLE"},
        "retained_protections": ["unchanged recipe", "integrity", "full recovery", "GPU memory margin", "disk margin", "budget/deadline"],
        "performance_and_cost": {"status": "NOT_MEASURED", "note": "Remeasure actual cloud complete update cycles and full billable lifecycle; no extrapolation from old capped clocks."},
    }


def write_exclusive(path, document):
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--disk", action="append", default=[])
    parser.add_argument("--min-ram-gib", type=float)
    parser.add_argument("--min-cpus", type=float)
    args = parser.parse_args()
    for value in (args.min_ram_gib, args.min_cpus):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("screening thresholds must be finite and positive")
    start = time.monotonic()
    document = snapshot(args.disk or ["/", "/root/autodl-tmp"], args.min_ram_gib, args.min_cpus)
    document["probe_wall_seconds"] = time.monotonic() - start
    write_exclusive(args.output, document)


if __name__ == "__main__":
    main()
