"""Freeze committed PR31 bytes and a pending, non-authorizing real contract on server."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
from sparse_rtdetr.dn_block_runtime import write,ref,need
from dn_block_explore import PY,DEFINITION
def main():
    dest=Path(sys.argv[1]).resolve()
    need(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT),"dirty worktree")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    original=subprocess.check_output(["git","show","6228e20f1229f70e85334e8fdb248c1afd54228e:docs/research/r18_dn_block_replay/PLAN.md"],cwd=ROOT)
    need(hashlib.sha256(original).hexdigest()==DEFINITION,"original definition")
    need((ROOT/"docs/research/r18_dn_block_replay/PLAN.md").read_bytes().startswith(original),"scientific definition changed")
    dest.mkdir(parents=True,exist_ok=False);release=dest/"release";release.mkdir()
    rows=[]
    for name in subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines():
        raw=subprocess.check_output(["git","show",head+":"+name],cwd=ROOT)
        need(raw==(ROOT/name).read_bytes(),"Git working bytes")
        p=release/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        rows.append({"path":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    write(dest/"MANIFEST.json",{"commit":head,"files":rows,"definition_SHA":DEFINITION,
                              "scope":"prepared_execution_not_authorized"})
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=""
    attempt="/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/r18-dn-block-exploration-20261030-001/attempt-001"
    args=[PY,"-B",str(release/"tools/dn_block_explore.py"),"prepare","--manifest",str(dest/"MANIFEST.json"),
          "--attempt",attempt,"--out",str(dest/"CONTRACT.json")]
    with (dest/"prepare.stdout").open("xb") as out,(dest/"prepare.stderr").open("xb") as err:
        result=subprocess.run(args,cwd=release,env=env,stdout=out,stderr=err)
    need(result.returncode==0,"frozen actual preparation failed")
    contract=ref(dest/"CONTRACT.json");manifest=ref(dest/"MANIFEST.json")
    write(dest/"AUTHORIZATION.template.json",{"contract_SHA":contract["sha256"],"commit":head,"manifest_SHA":manifest["sha256"],
                                            "scope":"ONE_DN_BLOCK_648F_300U","approved":False})
    cmd=PY+" -B "+str(release/"tools/dn_block_explore.py")+" launch --contract "+str(dest/"CONTRACT.json")+" --authorization "+str(dest/"AUTHORIZATION.json")
    (dest/"COMMANDS.txt").write_text("# Pending separate GPU authorization. Template is NOT approval.\n"+cmd+"\n")
    write(dest/"INDEX.json",{"commit":head,"manifest":manifest,"contract":contract,"definition_SHA":DEFINITION,
        "authorization_created":False,"attempt_created":False,"GPU_executed":False,"unique_command":cmd})
    print((dest/"INDEX.json").read_text())
if __name__=="__main__":main()
