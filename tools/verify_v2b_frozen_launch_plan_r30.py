#!/usr/bin/env python3
"""Strict read-only verifier for the REV1 bridge-033 frozen launch plan."""
from __future__ import annotations
import argparse, hashlib, json, re, shlex, stat, subprocess
from pathlib import Path
from typing import Any

PLAN_EXCLUDED = {"frozen_launch_plan_sha256", "authorization_sha256"}
RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
PLAN_AUTH_MARKER = "__BOUND_AUTHORIZATION_SHA256__"
PLACEHOLDERS = {"TBD", "AUTO", "latest", "current", "infer-at-launch", "generate-at-runtime"}
UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")
AUTH_ID = "smoke-v2b-rev1-s2-r896-bridge-033"
PLAN_ID = "P3-V2B-REV1-FROZEN-LAUNCH-BRIDGE033"
BRIDGE_SHA = "3809d35df513eb2923fd91518c013dff4ddf9800b4c1a1196011bd20a30c0b89"
BINDING_SHA = "99a3f44d7077b1768522010f92aabf7992e169017c1028ace1502655d27abc57"
MANIFEST_SHA = "95ec833926ee64cfab421b69c1592eb23d12fe95e988763670fe06ece5b59da3"
TRAINING_SHA = "bafdcbfbd219de288c3947595902d47bf37031bfdc2e9b01d36cd9f1ee2177c4"
SCIENTIFIC_SHA = "15f4ce16e373f08cf192b73adbb09b2de90f64ef640a5d8f3b1df44c586e92f8"
GPU_UUID = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"

def canonical(v: Any) -> bytes:
    return (json.dumps(v, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
def load(p: Path) -> dict[str, Any]:
    raw=p.read_bytes(); v=json.loads(raw.decode("utf-8"))
    if raw != canonical(v) or not isinstance(v,dict): raise ValueError(f"non-canonical object: {p}")
    return v
def without_runtime(v: Any) -> Any:
    if isinstance(v,dict): return {k:without_runtime(x) for k,x in v.items() if k not in RUNTIME_KEYS}
    if isinstance(v,list): return [without_runtime(x) for x in v]
    return v
def identity_view(plan: dict[str,Any]) -> dict[str,Any]:
    body={k:v for k,v in plan.items() if k not in PLAN_EXCLUDED}
    for field in ("exact_argv", "controller_argv", "runner_argv"):
        argv=list(body.get(field) or [])
        if "--auth-sha" not in argv: raise ValueError(f"{field} missing --auth-sha")
        argv[argv.index("--auth-sha")+1]=PLAN_AUTH_MARKER
        body[field]=argv
    body["exact_launch_command"]=shlex.join(body["exact_argv"])
    return body
def regular(p: Path) -> None:
    st=p.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode): raise ValueError(f"regular non-symlink required: {p}")
def verify(args):
    repo=args.repo_root.resolve(strict=True); plan=load(args.plan.resolve(strict=True)); auth=load(args.authorization.resolve(strict=True))
    if plan.get("schema_version") != 2 or plan.get("plan_id") != PLAN_ID: raise ValueError("plan identity/schema mismatch")
    if plan.get("authorization_id") != AUTH_ID or auth.get("smoke_authorization_id") != AUTH_ID: raise ValueError("authorization id mismatch")
    if plan.get("canonicalization",{}).get("digest_excluded_fields") != ["authorization_sha256","frozen_launch_plan_sha256"]: raise ValueError("plan exclusion mismatch")
    if plan.get("canonicalization",{}).get("argv_auth_substitution") != PLAN_AUTH_MARKER: raise ValueError("plan auth substitution mismatch")
    if hashlib.sha256(canonical(identity_view(plan))).hexdigest() != plan.get("frozen_launch_plan_sha256"): raise ValueError("frozen launch plan digest mismatch")
    if plan.get("authorization_sha256") != auth.get("authorization_sha256"): raise ValueError("plan/auth SHA mismatch")
    if auth.get("status") != "PENDING_NOT_EXECUTED" or auth.get("consumed") is not False or auth.get("formal_launch_permitted") is not False: raise ValueError("authorization not pending fail-closed")
    auth_body={k:v for k,v in auth.items() if k != "authorization_sha256"}
    if hashlib.sha256(canonical(without_runtime(auth_body))).hexdigest() != auth.get("authorization_sha256"): raise ValueError("authorization digest mismatch")
    if auth.get("frozen_launch_plan_sha256") != plan.get("frozen_launch_plan_sha256"): raise ValueError("authorization plan binding mismatch")
    for value in plan.values():
        if isinstance(value,str) and value in PLACEHOLDERS: raise ValueError(f"placeholder value: {value}")
    if plan.get("repository_root") != plan.get("working_directory") or plan.get("repository_root") != str(repo): raise ValueError("repository/cwd mismatch")
    if not SESSION_RE.fullmatch(str(plan.get("tmux_session_name"))): raise ValueError("invalid tmux session")
    if UUID_RE.fullmatch(str(plan.get("cuda_mapping",{}).get("cuda_visible_devices"))) is None: raise ValueError("GPU mapping invalid")
    if plan.get("exact_argv") != plan.get("controller_argv"): raise ValueError("exact argv/controller argv mismatch")
    argv=plan["exact_argv"]
    if not isinstance(argv,list) or len(argv)<2: raise ValueError("exact argv invalid")
    expected={"--root":plan.get("external_evidence_root"),"--session":plan.get("tmux_session_name"),"--repo":plan.get("repository_root"),"--python":plan.get("interpreter"),"--runner":plan.get("runner"),"--derived":plan.get("resolved_checkpoint_locator"),"--manifest":plan.get("resolved_manifest_locator"),"--policy-authority":plan.get("policy_authority_path"),"--auth":plan.get("authorization_path"),"--auth-id":plan.get("authorization_id"),"--auth-sha":plan.get("authorization_sha256"),"--exec-source-sha":plan.get("execution_source_sha256"),"--exec-contract-sha":plan.get("execution_contract_sha256"),"--exec-contract":plan.get("execution_contract_path"),"--bridge-binding-sha":plan.get("bridge_binding_sha256"),"--bridge-manifest-sha":plan.get("bridge_manifest_sha256"),"--training-contract-sha":plan.get("training_contract_sha256"),"--gpu-uuid":plan.get("cuda_mapping",{}).get("cuda_visible_devices"),"--expected-execution-contract-id":plan.get("execution_contract_id"),"--execution-source":plan.get("execution_source_path"),"--revision-verifier":plan.get("revision_verifier")}
    for key,val in expected.items():
        if key not in argv or argv[argv.index(key)+1] != str(val): raise ValueError(f"argv mismatch: {key}")
    if plan.get("exact_launch_command") != shlex.join(argv): raise ValueError("launch command mismatch")
    if plan.get("tmux_argv") != ["tmux","new-session","-d","-s",plan.get("tmux_session_name"),"-c",plan.get("working_directory"),plan.get("tmux_wrapper_path")]: raise ValueError("tmux argv mismatch")
    startup=plan.get("startup_environment",{}).get("static") or {}
    if startup.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or startup.get("PYTHONHASHSEED") != "0": raise ValueError("startup environment mismatch")
    for key in ("repository_root","working_directory","interpreter","launcher","runner","resolved_checkpoint_locator","resolved_manifest_locator","policy_authority_path","execution_contract_path","authorization_path","plan_path","structured_invocation"):
        p=Path(str(plan.get(key))); p.resolve(strict=True)
    if Path(plan["plan_path"]).resolve() != args.plan.resolve(): raise ValueError("plan path mismatch")
    for key in ("local_output_root","external_evidence_root"):
        p=Path(plan[key]);
        if p.exists(): raise ValueError(f"target already exists: {key}")
        if not p.parent.is_dir(): raise ValueError(f"target parent unavailable: {key}")
    if subprocess.run(["tmux","has-session","-t",plan["tmux_session_name"]],capture_output=True).returncode==0: raise ValueError("tmux session already exists")
    if sha_file(Path(plan["resolved_checkpoint_locator"])) != str(plan.get("checkpoint_sha256")) or sha_file(Path(plan["resolved_manifest_locator"])) != str(plan.get("bridge_manifest_sha256")): raise ValueError("bridge identity drift")
    git=lambda *x: subprocess.check_output(["git",*x],cwd=repo,text=True).strip()
    if git("status","--porcelain"): raise ValueError("worktree dirty")
    if git("rev-parse","HEAD") != plan["git"]["commit"]:
        if git("rev-parse","HEAD^") != plan["git"]["commit"] or git("diff","--name-only",plan["git"]["commit"],"HEAD").splitlines() != sorted(plan.get("postseal_data_paths") or []): raise ValueError("postseal git identity mismatch")
    return {"status":"FROZEN_LAUNCH_PLAN_MATCH","plan_id":plan["plan_id"],"frozen_launch_plan_sha256":plan["frozen_launch_plan_sha256"],"authorization_id":auth["smoke_authorization_id"],"authorization_sha256":auth["authorization_sha256"],"targets_absent":True,"tmux_absent":True}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",type=Path,required=True); ap.add_argument("--authorization",type=Path,required=True); ap.add_argument("--repo-root",type=Path,required=True)
    a=ap.parse_args()
    try: print(json.dumps(verify(a),sort_keys=True,indent=2)); return 0
    except Exception as e: print(json.dumps({"status":"FAIL","error":f"{type(e).__name__}:{e}"},sort_keys=True)); return 2
if __name__=='__main__': raise SystemExit(main())
