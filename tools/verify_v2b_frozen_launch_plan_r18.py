#!/usr/bin/env python3
"""Read-only verifier for the concrete REV1 bridge-018 frozen launch plan."""
from __future__ import annotations
import argparse, hashlib, json, re, shlex, stat, subprocess
from pathlib import Path
from typing import Any

PLAN_EXCLUDED = {"frozen_launch_plan_sha256", "authorization_sha256"}
RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
PLACEHOLDERS = {"TBD", "AUTO", "latest", "current", "infer-at-launch", "generate-at-runtime"}
PLAN_AUTH_MARKER = "__BOUND_AUTHORIZATION_SHA256__"
UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")

def canonical(v: Any) -> bytes:
    return (json.dumps(v, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def load(path: Path) -> dict[str,Any]:
    raw=path.read_bytes(); v=json.loads(raw.decode())
    if raw != canonical(v): raise ValueError(f"non-canonical JSON: {path}")
    if not isinstance(v,dict): raise ValueError(f"object required: {path}")
    return v

def regular(path: Path) -> None:
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode): raise ValueError(f"regular non-symlink required: {path}")

def without_runtime(v: Any) -> Any:
    if isinstance(v,dict): return {k:without_runtime(x) for k,x in v.items() if k not in RUNTIME_KEYS}
    if isinstance(v,list): return [without_runtime(x) for x in v]
    return v

def walk_values(v: Any):
    if isinstance(v,dict):
        for k,x in v.items():
            yield k; yield from walk_values(x)
    elif isinstance(v,list):
        for x in v: yield from walk_values(x)
    elif isinstance(v,str):
        yield v

def plan_identity_view(plan: dict[str,Any]) -> dict[str,Any]:
    body={k:v for k,v in plan.items() if k not in PLAN_EXCLUDED}
    argv=list(body.get("exact_argv") or [])
    if "--auth-sha" not in argv:
        raise ValueError("exact argv missing --auth-sha")
    index=argv.index("--auth-sha")
    if index + 1 >= len(argv):
        raise ValueError("exact argv missing auth SHA value")
    argv[index+1]=PLAN_AUTH_MARKER
    body["exact_argv"]=argv
    # Normalize the repeated auth digest in the display command as well.
    body["exact_launch_command"]=shlex.join(argv)
    return body

def verify_plan(plan_path: Path, auth_path: Path, repo: Path) -> dict[str,Any]:
    plan=load(plan_path); auth=load(auth_path)
    for value in walk_values(plan):
        if value in PLACEHOLDERS: raise ValueError(f"placeholder value: {value}")
    if plan.get("schema_version") != 2 or plan.get("plan_id") != "P3-V2B-REV1-FROZEN-LAUNCH-BRIDGE018":
        raise ValueError("plan identity/schema mismatch")
    if plan.get("authorization_id") != "smoke-v2b-rev1-s2-r896-bridge-018":
        raise ValueError("plan authorization ID mismatch")
    if plan.get("canonicalization",{}).get("digest_excluded_fields") != ["authorization_sha256","frozen_launch_plan_sha256"] or plan.get("canonicalization",{}).get("argv_auth_sha_substitution") != PLAN_AUTH_MARKER:
        raise ValueError("plan digest exclusion contract mismatch")
    if sha_bytes(canonical(plan_identity_view(plan))) != plan.get("frozen_launch_plan_sha256"):
        raise ValueError("frozen launch plan digest mismatch")
    if plan.get("authorization_sha256") != auth.get("authorization_sha256"):
        raise ValueError("plan/auth SHA mismatch")
    if auth.get("smoke_authorization_id") != plan["authorization_id"] or auth.get("status") != "PENDING_NOT_EXECUTED":
        raise ValueError("authorization status mismatch")
    if auth.get("consumed") is not False or auth.get("formal_launch_permitted") is not False:
        raise ValueError("authorization is not pending fail-closed")
    auth_body={k:v for k,v in auth.items() if k != "authorization_sha256"}
    if sha_bytes(canonical(without_runtime(auth_body))) != auth["authorization_sha256"]:
        raise ValueError("authorization digest mismatch")
    if auth.get("frozen_launch_plan_sha256") != plan["frozen_launch_plan_sha256"]:
        raise ValueError("authorization/plan binding mismatch")
    fields=("repository_root","working_directory","interpreter","launcher","tmux_session_name","local_output_root","external_evidence_root")
    if any(not isinstance(plan.get(k),str) or not plan[k] for k in fields): raise ValueError("concrete locator missing")
    if plan["repository_root"] != plan["working_directory"]: raise ValueError("repo/cwd mismatch")
    if not SESSION_RE.fullmatch(plan["tmux_session_name"]): raise ValueError("invalid tmux session name")
    if UUID_RE.fullmatch(plan["cuda_mapping"]["cuda_visible_devices"]) is None: raise ValueError("GPU UUID mapping invalid")
    for key in ("repository_root","working_directory","interpreter","launcher","resolved_checkpoint_locator","resolved_manifest_locator","policy_authority_path","execution_contract_path","authorization_path","plan_path"):
        path=Path(plan[key])
        if key not in ("local_output_root","external_evidence_root") and not path.exists(): raise ValueError(f"locator unavailable: {key}")
    if plan["repository_root"] != str(repo.resolve()): raise ValueError("repository root drift")
    git=lambda *args: subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
    if git("branch","--show-current") != plan["git"]["branch"] or git("status","--porcelain"): raise ValueError("git identity/worktree drift")
    head=git("rev-parse","HEAD")
    source_tree=git("rev-parse",plan["git"]["commit"]+"^{tree}")
    if source_tree != plan["git"]["tree"]: raise ValueError("frozen source tree drift")
    if head != plan["git"]["commit"]:
        parent=git("rev-parse","HEAD^")
        allowed=sorted(plan.get("postseal_data_paths") or [])
        changed=git("diff","--name-only",plan["git"]["commit"],"HEAD").splitlines()
        if parent != plan["git"]["commit"] or changed != allowed: raise ValueError("git identity/worktree drift")
    for target_key in ("local_output_root","external_evidence_root"):
        if Path(plan[target_key]).exists(): raise ValueError(f"target already exists: {target_key}")
        if not Path(plan[target_key]).parent.is_dir(): raise ValueError(f"target parent unavailable: {target_key}")
    session=subprocess.run(["tmux","has-session","-t",plan["tmux_session_name"]],capture_output=True)
    if session.returncode==0: raise ValueError("tmux session already exists")
    argv=plan["exact_argv"]
    if not isinstance(argv,list) or argv[:2] != [plan["interpreter"],plan["launcher"]]: raise ValueError("exact argv prefix mismatch")
    expected_pairs={"--root":plan["external_evidence_root"],"--session":plan["tmux_session_name"],"--repo":plan["repository_root"],"--python":plan["interpreter"],"--runner":plan["runner"],"--derived":plan["resolved_checkpoint_locator"],"--manifest":plan["resolved_manifest_locator"],"--auth":plan["authorization_path"],"--auth-id":plan["authorization_id"],"--auth-sha":plan["authorization_sha256"],"--exec-source-sha":plan["execution_source_sha256"],"--exec-contract-sha":plan["execution_contract_sha256"],"--exec-contract":plan["execution_contract_path"],"--policy-authority":plan["policy_authority_path"],"--policy-authority-id":plan["policy_authority_id"],"--policy-authority-identity-sha":plan["policy_authority_identity_sha"],"--bridge-binding-sha":plan["bridge_binding_sha256"],"--bridge-manifest-sha":plan["bridge_manifest_sha256"],"--training-contract-sha":plan["training_contract_sha256"],"--gpu-uuid":plan["cuda_mapping"]["cuda_visible_devices"]}
    for flag,val in expected_pairs.items():
        if flag not in argv or argv[argv.index(flag)+1] != val: raise ValueError(f"argv mismatch: {flag}")
    if plan.get("exact_launch_command") != shlex.join(argv): raise ValueError("display command mismatch")
    if plan.get("tmux_argv") != ["tmux","new-session","-d","-s",plan["tmux_session_name"],"-c",plan["working_directory"],plan["tmux_wrapper_path"]]: raise ValueError("tmux argv mismatch")
    if not plan["startup_environment"].get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" or plan["startup_environment"].get("PYTHONHASHSEED") != "0": raise ValueError("startup environment mismatch")
    for key in ("checkpoint_sha256","bridge_manifest_sha256","execution_source_sha256","execution_contract_sha256","training_contract_sha256","bridge_binding_sha256","policy_authority_identity_sha"):
        if not isinstance(plan.get(key),str) or len(plan[key]) != 64: raise ValueError(f"identity missing: {key}")
    if sha_file(Path(plan["resolved_checkpoint_locator"])) != plan["checkpoint_sha256"] or sha_file(Path(plan["resolved_manifest_locator"])) != plan["bridge_manifest_sha256"]: raise ValueError("bridge identity drift")
    if plan["plan_path"] != str(plan_path.resolve()): raise ValueError("plan path mismatch")
    return {"status":"FROZEN_LAUNCH_PLAN_MATCH","plan_id":plan["plan_id"],"frozen_launch_plan_sha256":plan["frozen_launch_plan_sha256"],"authorization_id":auth["smoke_authorization_id"],"authorization_sha256":auth["authorization_sha256"],"targets_absent":True,"tmux_absent":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",type=Path,required=True); ap.add_argument("--authorization",type=Path,required=True); ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    try: print(json.dumps(verify_plan(args.plan.resolve(strict=True),args.authorization.resolve(strict=True),args.repo_root.resolve(strict=True)),sort_keys=True,indent=2)); return 0
    except Exception as exc: print(json.dumps({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"},sort_keys=True)); return 2
if __name__=="__main__": raise SystemExit(main())
