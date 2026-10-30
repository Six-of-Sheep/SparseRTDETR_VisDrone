"""One read-only library reuse/precheck. No detector, optimizer, probe or learning command."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="2"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys,json,time,hashlib,subprocess,argparse,traceback,copy,tempfile,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import torch
from sparse_rtdetr.dn_block_runtime import (write,ref,exact,need,digest,loader,input_hash,seed,
    capture_rng,restore_rng,rng_hash,make_library,_move)
from sparse_rtdetr.dn_block_replay import (distribution,semantic_audit,Replay,dn_entry,
    effective,corners,encode,scale_group,control_audit)
from dn_block_explore import checked_contract,verify_source,admitted_resources
WS=Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
OLD=WS/"research_cpu/dn-block-replay-20261030/frozen-full-aa3213ab7148"
OLD_COMMIT="aa3213ab7148550e7081f693647a5c93f58addf4"
OLD_MANIFEST="108a9cc060cec3ae2be84d23d48129fbd9621e12578093ed5b21347ba9a86a10"
OLD_CONTRACT="72ddbbbf3a775113274593ebbddbfe2951ace488ae6f9e1731eb2b02bcb09ba9"
LIB_SHA="3cabea9aad68c64aec6cabcab4f0e362228fb68c72922d391e44e50dd703c246"
CAL_SHA="3addf6f9f80d4ea852f995ccff262c3d4aa5e95a9907911269ed9c508764ca76"
COST_SHA="37e4f7cbd4b0b627062383039babae84c131f5e4e62543360b8704bf00ada290"
PY="/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def size(p):return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
def native():
    import types,importlib
    pkg=types.ModuleType("native_recheck")
    pkg.__path__=[str(ROOT/"vendor/rtdetrv2_pytorch/src/zoo/rtdetr")]
    sys.modules["native_recheck"]=pkg
    return importlib.import_module("native_recheck.denoising").get_contrastive_denoising_training_group
def verify_materials():
    need(sha(OLD/"CONTRACT.json")==OLD_CONTRACT and sha(OLD/"MANIFEST.json")==OLD_MANIFEST,"old identities")
    c=checked_contract(OLD/"CONTRACT.json");r=Path(c["attempt"])
    m=json.loads((OLD/"MANIFEST.json").read_text())
    need(m["commit"]==OLD_COMMIT,"old commit")
    for row in m["files"]:
        f=OLD/"release"/row["path"]
        need(f.stat().st_size==row["bytes"] and sha(f)==row["sha256"],"old frozen bytes")
    exact(c["parent"]);exact(c["data"]["manifest"])
    started=json.loads((r/"STARTED.json").read_text());exact(started["authorization"])
    need(started["contract_SHA"]==OLD_CONTRACT,"old started binding")
    closed=json.loads((r/"LIBRARY_CLOSED.json").read_text())
    need(closed["library"]["sha256"]==LIB_SHA and closed["calibration"]["sha256"]==CAL_SHA,"closed library identities")
    lib=torch.load(exact(closed["library"]),map_location="cpu",weights_only=True)
    rows=torch.load(exact(closed["calibration"]),map_location="cpu",weights_only=True)
    need([v["image"] for v in rows]==c["selection"]["calibration"] and len(rows)==64,"64 image order")
    rebuilt=make_library(rows)
    for k in ("errors","groups"):need(torch.equal(rebuilt[k],lib[k]),"library/calibration exact "+k)
    for row in rows:
        need(row["stage"]=="decoder_3_final_ordinary_Q300_before_topK","stage")
        qi=torch.tensor(row["query"]);gi=torch.tensor(row["gt_index"])
        need(torch.equal(row["prediction"],row["ordinary_boxes"][qi]),"query binding")
        need(torch.equal(row["gt"],row["all_GT_boxes"][gi]),"GT binding")
        need(torch.equal(encode(row["prediction"],row["gt"]),row["error"]),"source residual")
        need(torch.equal(scale_group(row["gt"]),row["group"]),"source scales")
    need(sha(r/"COST_ADDITION.json")==COST_SHA,"prior cost")
    cost=json.loads((r/"COST_ADDITION.json").read_text())
    need(cost["new_forward"]==16 and cost["new_updates"]==0,"old counts")
    old_failure=json.loads((r/"precheck.pt.failure.json").read_text())
    need(old_failure["microbatch"]==0 and old_failure["audit"]["invalid_J"]==1,"old failure")
    protected={str(f):sha(f) for f in [OLD/"CONTRACT.json",OLD/"MANIFEST.json",r/"STOP.json",
        r/"COST_ADDITION.json",r/"LIBRARY_CLOSED.json",r/"library.pt",r/"calibration.pt",r/"precheck.pt.failure.json"]}
    return c,lib,rows,cost,protected
def evidence(ctx,targets,meta,rows):
    """Read existing replay observation; independently associate recorded slot/GT IDs."""
    obs=ctx.observation;gt=obs["gt"].to(targets[0]["boxes"])
    allidx=[]
    for i,t in enumerate(targets):
        for slot in meta["dn_positive_idx"][i].tolist():allidx.append((i,slot))
    chosen=sorted(allidx)[::2]
    need(len(chosen)==len(gt),"observation/slot count")
    donor_identity=[]
    for row in rows:
        for q,gi in zip(row["query"],row["gt_index"]):
            donor_identity.append({"image":row["image"],"query":q,"gt_id":row["gt_ids"][gi]})
    found=[]
    for arm in ("J","B"):
        before,clipped,final,unact,wasclipped,invalid=effective(gt,obs[arm].to(gt))
        for k in invalid.nonzero().flatten().tolist():
            bi,slot=chosen[k];ti=slot%len(targets[bi]["boxes"])
            need(torch.equal(gt[k],targets[bi]["boxes"][ti]),"recipient identity")
            donor=ctx.audit["donors"][k];size_source=ctx.audit["donors"][ctx.audit["size_permutation"][k]] if arm=="B" else donor
            found.append({"arm":arm,"replacement_index":k,"batch_row":bi,"DN_slot":slot,
              "recipient_image":targets[bi]["image_id"],"recipient_gt_id":targets[bi]["gt_ids"][ti],
              "recipient_GT_cxcywh":gt[k].tolist(),"donor_index":donor,"donor_identity":donor_identity[donor],
              "size_donor_index":size_source,"error":obs[arm][k].tolist(),"before_cxcywh":before[k].tolist(),
              "before_xyxy":corners(before[k:k+1])[0].tolist(),
              "clipped_xyxy":corners(before[k:k+1]).clamp(0,1)[0].tolist(),
              "clipped_cxcywh":clipped[k].tolist(),"final_reference":final[k].tolist(),
              "unact":unact[k].tolist(),"invalid_rule":"any clipped width/height <= 0 before inverse-sigmoid",
              "invalid":bool(invalid[k]),"would_STOP_original_rule":True})
    return {"invalid_samples":found,"coordinate":"normalized cxcywh / xyxy on augmented input",
            "final_reference_is_prospective":True,"injection_was_rejected":True,
            "no_repair_filter_resample":True}
def stats_check(device):
    torch.set_num_threads(2)
    for dtype in (torch.float32,torch.float64,torch.bfloat16):
        x=torch.tensor([[0.,1.,2.,3.],[1.,2.,4.,6.],[-2.,3.,1.,9.]],dtype=dtype,device=device)
        actual=distribution(x);expected=distribution(x.cpu())
        for key in ("mean","quantiles_0_25_50_75_100","cross_covariance"):
            need(torch.allclose(torch.tensor(actual[key],dtype=torch.float64),torch.tensor(expected[key],dtype=torch.float64),rtol=1e-12,atol=1e-12),"statistics FP64 reference")
    need(distribution(torch.empty(0,4,device=device))=={"n":0},"empty")
    g=torch.tensor([[.5,.5,.1,.2],[.4,.4,.2,.1]],device=device)
    need(semantic_audit(g,g,g,g)["admissible"],"semantic statistics path")
    need(control_audit(g,torch.zeros_like(g),torch.zeros_like(g),torch.arange(2,device=device))["joint_only_admissible"],"valid control")
    # Strict invalid counterexample: center outside image -> clipped zero width.
    e=torch.tensor([[20.,0.,0.,0.],[0.,0.,0.,0.]],device=device)
    a=control_audit(g,e,e,torch.arange(2,device=device))
    need(a["invalid_J"]==1 and not a["joint_only_admissible"],"invalid original guard")
    distribution(e)
    return {"statistics_device":str(device),"dtypes":["float32","float64","bfloat16"],"compute_dtype":"float64",
            "CPU_reference_rtol_atol":1e-12,"empty_semantics_invalid_counterexample":True,"detector_forward":0,"updates":0}
def gpu_resource(c):
    need(not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"other GPU process")
    return admitted_resources(c)
def cpu_check(out):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    r=stats_check(torch.device("cpu"))
    fn=native();g=torch.tensor([[.5,.5,.1,.1]])
    t=[{"boxes":g,"labels":torch.tensor([0]),"image_id":"CPU_CONSTRUCTED","gt_ids":["g"]}]
    embed=torch.nn.Embedding(11,256);seed()
    a=dn_entry(fn,t,10,300,embed)
    need(a[3] is not None,"actual native metadata")
    # Actual injection entry rejects fixed invalid donor, records observations.
    ctx=Replay(torch.tensor([[20.,0.,0.,0.]]),torch.tensor([0]),"J",20261030)
    # Input-area stratum for .1*.1 is L (>=9216).
    ctx.groups=torch.tensor([2])
    captured={}
    def observed(*args,**kwargs):
        out=fn(*args,**kwargs);captured["meta"]=out[3];return out
    try:dn_entry(observed,t,10,300,embed,replay=ctx)
    except ValueError as ex:
        need(str(ex)=="post-injection block control failed; no filtering/retry","exact original rejection")
        need(ctx.audit["invalid_J"]>0 and hasattr(ctx,"observation"),"invalid evidence")
        rows=[{"image":"CPU_DONOR","query":[3],"gt_index":[0],"gt_ids":["donor_gt"]}]
        ev=evidence(ctx,t,captured["meta"],rows)
        need(len(ev["invalid_samples"])>0,"evidence samples")
        for row in ev["invalid_samples"]:
            need(row["recipient_gt_id"]=="g" and row["clipped_cxcywh"][2]==0,"independent clipped width")
            need(abs(row["before_cxcywh"][0]-2.5)<1e-6,"independent GT + error center")
            need(row["final_reference"][2]>0 and row["would_STOP_original_rule"],"epsilon does not rescue invalid")
        r["independent_invalid_geometry_and_identity"]=True
    else:raise ValueError("invalid accepted")
    r.update(actual_native_DN_entry=True,invalid_observation_retained=True,CUDA_initialized=torch.cuda.is_initialized())
    write(out,r);print(json.dumps(r))
def cuda_check(out):
    start=time.monotonic();c=checked_contract(OLD/"CONTRACT.json")
    gpu_resource(c)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    r=stats_check(torch.device("cuda:0"));torch.cuda.synchronize()
    r.update(wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),
             scope="minimum_synthetic_statistics_only",source_sha=sha(ROOT/"src/sparse_rtdetr/dn_block_replay.py"))
    need(r["wall_seconds"]<60,"minimal CUDA duration")
    write(out,r);print(json.dumps(r))
def freeze(dest,cuda_receipt):
    need(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT),"clean commit")
    c,lib,rows,cost,protected=verify_materials()
    receipt=json.loads(Path(cuda_receipt).read_text())
    need(receipt["source_sha"]==sha(ROOT/"src/sparse_rtdetr/dn_block_replay.py"),"CUDA checked bytes")
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);release=dest/"release";release.mkdir()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    files=[]
    for name in subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines():
        b=subprocess.check_output(["git","show",head+":"+name],cwd=ROOT)
        need(b==(ROOT/name).read_bytes(),"Git bytes");f=release/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
        files.append({"path":name,"bytes":len(b),"sha256":sha(f)})
    write(dest/"MANIFEST.json",{"commit":head,"files":files})
    contract={"scope":"READ_ONLY_LIBRARY_PRECHECK_ONLY","commit":head,"manifest":ref(dest/"MANIFEST.json"),
      "old_contract":ref(OLD/"CONTRACT.json"),"library":ref(Path(c["attempt"])/"library.pt"),
      "calibration":ref(Path(c["attempt"])/"calibration.pt"),"protected":protected,
      "minimal_cuda":ref(cuda_receipt),"minimal_cuda_seconds":receipt["wall_seconds"],
      "prior_cost":cost,"forward":0,"updates":0,"GPU_seconds":300,"wall_seconds":600,
      "allocated":21<<30,"cumulative_storage":2<<30,"cumulative_return":1<<20,"prior_return_upper":100000,
      "attempt":str(Path(c["attempt"]).parent/"precheck-reuse-001")}
    write(dest/"CONTRACT.json",contract)
    cmd=PY+" -B "+str(release/"tools/dn_block_recheck.py")+" run --package "+str(dest)
    (dest/"COMMANDS.txt").write_text(cmd+"\n")
    write(dest/"INDEX.json",{"commit":head,"manifest":ref(dest/"MANIFEST.json"),"contract":ref(dest/"CONTRACT.json"),
                           "command":cmd,"no_learning_path":True})
    print((dest/"INDEX.json").read_text())
def package_check(package):
    p=Path(package);k=json.loads((p/"CONTRACT.json").read_text())
    m=json.loads(exact(k["manifest"]).read_text());need(m["commit"]==k["commit"],"new commit")
    for row in m["files"]:
        f=ROOT/row["path"];need(f.stat().st_size==row["bytes"] and sha(f)==row["sha256"],"new frozen bytes")
    exact(k["minimal_cuda"]);exact(k["library"]);exact(k["calibration"]);exact(k["old_contract"])
    for name,h in k["protected"].items():need(sha(name)==h,"protected material changed")
    return p,k
def worker(package):
    p,k=package_check(package);root=Path(k["attempt"]);start=time.monotonic()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    status="ENGINEERING_STOP";result={};peak=None;processed=0
    try:
        c,lib,rows,cost,protected=verify_materials()
        resources=gpu_resource(c);write(root/"RESOURCES.json",resources)
        # Future parent is absent by design. No detector is instantiated here.
        fn=native();embedding=torch.nn.Embedding.from_pretrained(torch.zeros(11,256,device="cuda"))
        seed();moved=total=0
        for wi,(images,targets,traces) in enumerate(loader(c,"learning")):
            for micro in range(2):
                need(time.monotonic()-start+k["minimal_cuda_seconds"]<295,"time budget")
                need(size(root)+size(Path(c["attempt"]))+size(str(c["attempt"])+"-launcher")+size(p)+size(os.environ["TMPDIR"])<k["cumulative_storage"],"cumulative storage")
                original=_move(targets[micro*c["batch"]:(micro+1)*c["batch"]],torch.device("cuda"))
                rng=capture_rng(torch.device("cuda"));rng_before=rng_hash(torch.device("cuda"));native_end=None
                for arm in ("C","J","B"):
                    restore_rng(rng,torch.device("cuda"))
                    ctx=None if arm=="C" else Replay(lib["errors"],lib["groups"],arm,20261030+2*wi+micro)
                    captured={}
                    def wrapped(*args,**kwargs):
                        v=fn(*args,**kwargs);captured["meta"]=v[3];return v
                    try:dn_entry(wrapped,original,10,300,embedding,replay=ctx)
                    except ValueError as ex:
                        if str(ex) not in ("post-injection block control failed; no filtering/retry","positive-negative semantic guard failed; no filtering/retry"):raise
                        detail=evidence(ctx,original,captured["meta"],rows)
                        old=json.loads((Path(c["attempt"])/"precheck.pt.failure.json").read_text())
                        compared={q:ctx.audit[q] for q in old["audit"]}
                        if wi==0 and micro==0:need(compared==old["audit"],"old partial audit disagreement")
                        result={"status":"CONTROL_FAIL","error":str(ex),"window":wi,"microbatch":2*wi+micro,
                                "arm":arm,"audit":ctx.audit,"evidence":detail,"input_SHA":input_hash(images,targets,traces),
                                "RNG_before_SHA":rng_before,"old_partial_audit_exact":compared==old["audit"],
                                "historical_full_input_RNG_digest":"NOT_SAVED; deterministic original reconstruction only",
                                "learning_benefit":"NOT_TESTED"}
                        status="CONTROL_FAIL";write(root/"RESULT.json",result);return
                    now=rng_hash(torch.device("cuda"))
                    if native_end is None:native_end=now
                    need(now==native_end,"J/B changed native RNG")
                    if arm=="J":
                        audit=ctx.audit
                        donors=torch.tensor(audit["donors"]);perm=torch.tensor(audit["size_permutation"])
                        total+=len(donors);moved+=int((lib["errors"][donors,2:]!=lib["errors"][donors[perm],2:]).any(-1).sum())
                processed+=1
        need(processed==200 and total>0 and moved/total>=.8,"all-window coverage/change gate")
        status="CONTROL_PASS";write(root/"RESULT.json",{"status":status,"microbatches":processed,"moved_fraction":moved/total,"learning_benefit":"NOT_TESTED"})
    except BaseException as ex:
        (root/"stderr.txt").write_text(traceback.format_exc())
        write(root/"STOP.json",{"status":"ENGINEERING_STOP","error":str(ex),"no_retry":True})
        raise
    finally:
        try:
            if torch.cuda.is_initialized():peak=torch.cuda.max_memory_allocated()
        except Exception:peak=None
        write(root/"WORKER_EXIT.json",{"status":status,"peak_allocated":peak,"wall_seconds":time.monotonic()-start,
                                     "completed_microbatches":processed,"forward":0,"updates":0})
def run(package):
    p,k=package_check(package);root=Path(k["attempt"]);root.mkdir(parents=True,exist_ok=False)
    temp=Path("/tmp")/("dn31r-"+sha(p/"CONTRACT.json")[:10])
    temp.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(TMPDIR=str(temp),TORCHINDUCTOR_CACHE_DIR=str(temp/"inductor"),TRITON_CACHE_DIR=str(temp/"triton"))
    write(root/"STARTED.json",{"contract":ref(p/"CONTRACT.json"),"scope":k["scope"],"temp":str(temp)})
    start=time.monotonic();timedout=False
    with (root/"stdout.txt").open("xb") as out,(root/"process.stderr").open("xb") as err:
        proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"worker","--package",str(p)],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        try:code=proc.wait(timeout=max(1,290-k["minimal_cuda_seconds"]))
        except subprocess.TimeoutExpired:
            timedout=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait(timeout=2)
    seconds=time.monotonic()-start+k["minimal_cuda_seconds"]
    status=json.loads((root/"RESULT.json").read_text())["status"] if (root/"RESULT.json").exists() else "ENGINEERING_STOP"
    if code or timedout:status="ENGINEERING_STOP"
    write(root/"RUN_EXIT.json",{"status":status,"returncode":code,"timeout":timedout,"GPU_seconds_conservative":seconds,"no_retry":True})
    prior=k["prior_cost"]["cumulative"]
    write(root/"COST_ADDITION.json",{"prior":prior,"new_forward":0,"new_updates":0,"new_GPU_seconds":seconds,
      "includes_minimal_CUDA":k["minimal_cuda_seconds"],"cumulative":{"forward":prior["forward"],"GPU_stage_seconds":prior["GPU_stage_seconds"]+seconds},
      "original_exploration_remaining_GPU_seconds":1800-k["prior_cost"]["new_GPU_seconds"]-seconds})
    for name,h in k["protected"].items():need(sha(name)==h,"protected changed on exit")
    write(root/"CLOSED.json",{"engineering":"PASS" if status!="ENGINEERING_STOP" else "STOP","control":status,
                              "learning_benefit":"NOT_TESTED","protected_unchanged":True,"no_learning_forward":True})
    files=[ref(f) for f in root.iterdir() if f.is_file()]
    need(sum(x["bytes"] for x in files)+k["prior_return_upper"]+32768<k["cumulative_return"],"return cap")
    write(root/"EXPORT.json",{"files":files,"bytes":sum(x["bytes"] for x in files),"prior_upper":k["prior_return_upper"]})
    print(json.dumps({"status":status,"attempt":str(root),"GPU_seconds":seconds,"forward":0,"updates":0}))
    raise SystemExit(0 if status!="ENGINEERING_STOP" else 1)
def main():
    a=argparse.ArgumentParser();a.add_argument("mode",choices=["verify","cpu-check","cuda-check","freeze","run","worker"])
    a.add_argument("--out");a.add_argument("--dest");a.add_argument("--cuda-receipt");a.add_argument("--package")
    q=a.parse_args()
    if q.mode=="verify":
        c,lib,rows,cost,protected=verify_materials()
        write(q.out,{"status":"PASS","images":len(rows),"donors":len(lib["errors"]),"small":int((lib["groups"]==0).sum()),
                     "protected":protected,"parent":c["parent"],"no_detector_forward":True})
    elif q.mode=="cpu-check":cpu_check(q.out)
    elif q.mode=="cuda-check":cuda_check(q.out)
    elif q.mode=="freeze":freeze(q.dest,q.cuda_receipt)
    elif q.mode=="run":run(q.package)
    else:worker(q.package)
if __name__=="__main__":main()
