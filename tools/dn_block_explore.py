"""PR31 actual entry: prepare, isolated precheck, execute, synthetic fixtures. No GPU default."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="2"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys, argparse, subprocess, tempfile, traceback, time, gc, json, copy, hashlib, shutil, platform, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
# A short, task-exclusive directory is chosen before DataLoader/torch imports.
if "--contract" in sys.argv:
    _contract_path=Path(sys.argv[sys.argv.index("--contract")+1])
    if _contract_path.is_file() and "DN_BLOCK_TMP" not in os.environ:
        _tmp="/tmp/dn31-"+hashlib.sha256(_contract_path.read_bytes()).hexdigest()[:10]+"-"+str(os.getpid())
        Path(_tmp).mkdir()
        os.environ["DN_BLOCK_TMP"]=_tmp
        os.environ["TMPDIR"]=_tmp
        os.environ["TORCHINDUCTOR_CACHE_DIR"]=_tmp+"/inductor"
        os.environ["TRITON_CACHE_DIR"]=_tmp+"/triton"
        tempfile.tempdir=None
import torch
from sparse_rtdetr.dn_block_runtime import *
from sparse_rtdetr.protocol_r18.geometry import forward_guards
from sparse_rtdetr.baseline.training_v2b_engine import _move
PY="/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
WS=Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
BASE_CONTRACT=WS/"experiments/localization-distribution/r18-sabl-formal24-20261027-001/contract.json"
BASE_SHA="01ca71cd7931ed121553830c4794bdf5df772114b8fe416e2552663e1487afbb"
COST=WS/"experiments/localization-distribution/r18-sabl-paired-diagnostic-20261029-001/COST_ADDITION.json"
DEFINITION="a1b21726b31d24b216d289e82e4f6ffc95e01c20e064c67aa531dcb09f379fd6"

def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())

def checked_contract(path):
    c=json.loads(Path(path).read_text())
    need(c["schema"]==SCHEMA and c["policy"]==POLICY and c["caps"]==CAPS,"contract policy")
    need(c["gpu_execution_authorized"] is False and c["definition_SHA"]==DEFINITION,"authorization separation")
    if not c["synthetic"]:
        need(c["input_hw"]==[768,1344] and c["steps"]==100 and c["batch"]==8 and c["eval_batch"]==4,"real recipe")
        need({k:len(v) for k,v in c["selection"].items()}=={"calibration":64,"learning":128,"probe":32},"real counts")
        need(c["parent"]["sha256"]==N_SHA,"parent identity")
    ids=sum(c["selection"].values(),[]);need(len(ids)==len(set(ids)),"disjoint IDs")
    return c

def verify_source(c):
    m=json.loads(exact(c["manifest"]).read_text())
    need(m["commit"]==c["commit"],"source commit")
    for r in m["files"]:
        p=ROOT/r["path"];need(p.is_file() and p.stat().st_size==r["bytes"] and file_sha(p)==r["sha256"],"source bytes "+r["path"])

def prepare(a):
    need(file_sha(BASE_CONTRACT)==BASE_SHA,"historical metadata source")
    base=json.loads(BASE_CONTRACT.read_text());meta=base["metadata"]
    data=meta["data"]["train_core"];rows=json.loads(exact(data["manifest"]).read_text())["records"]
    selected=select_ids(rows)
    exact(meta["checkpoint"])
    import scipy,importlib.metadata
    packages={k:importlib.metadata.version(k) for k in ("torch","torchvision","numpy","scipy","Pillow","PyYAML","faster-coco-eval")}
    prior=json.loads(COST.read_text())
    manifest=ref(a.manifest);m=json.loads(Path(a.manifest).read_text())
    c={"schema":SCHEMA,"policy":POLICY,"caps":CAPS,"definition_SHA":DEFINITION,
       "synthetic":False,"commit":m["commit"],"manifest":manifest,"parent":meta["checkpoint"],
       "data":data,"selection":selected,"selection_SHA":digest(selected),"input_hw":[768,1344],
       "steps":100,"batch":8,"eval_batch":4,"gpu_execution_authorized":False,
       "attempt":str(Path(a.attempt).resolve()),"python":PY,"historical_metadata":ref(BASE_CONTRACT),
       "environment":meta["environment"],"gpu_uuid":base["gpu_uuid"],
       "python_version":platform.python_version(),"scipy_version":scipy.__version__,"package_versions":packages,
       "historical_cost":ref(COST),"historical_cumulative":prior["cumulative_all_research"],
       "budget_scope":"NEW_UNAUTHORIZED_648F_300U; historical ledgers remain read-only"}
    write(a.out,c)
    checked_contract(a.out);verify_source(c)
    print(json.dumps({"contract_SHA":file_sha(a.out),"selection_SHA":c["selection_SHA"],
                      "real_images_read":False,"GPU_authorized":False}))

class Budget:
    def __init__(self,c,root):
        self.c,self.root=c,Path(root);self.start=time.monotonic()
        self.count={"forward":0,"updates":0};self.peak=0
        self.limit=CAPS if not c["synthetic"] else {**CAPS,"forward":200,"updates":30,"wall_seconds":600,"storage_bytes":4<<30}
    def persist(self):
        value={**self.count,"wall_seconds":time.monotonic()-self.start,"peak_allocated":self.peak,
               "synthetic":self.c["synthetic"]}
        tmp=self.root/"ledger.tmp";tmp.write_text(json.dumps(value));os.replace(tmp,self.root/"LEDGER.json")
    def check(self):
        self.persist()
        need(time.monotonic()-self.start<=self.limit["wall_seconds"],"wall cap")
        if not self.c["synthetic"]:
            need(time.monotonic()-self.start<=CAPS["GPU_seconds"],"GPU stage conservative full runtime cap")
            self.peak=max(self.peak,torch.cuda.max_memory_allocated())
            need(self.peak<=CAPS["allocated_bytes"],"allocated cap")
        need(size(self.root)+size(str(self.root)+"-launcher")+size(os.environ.get("DN_BLOCK_TMP",self.root/"nonexistent"))<=self.limit["storage_bytes"],"storage cap")
    def forward(self,*args):
        need(self.count["forward"]<self.limit["forward"],"forward cap")
        self.count["forward"]+=1;self.check()
    def updated(self,*args):
        self.count["updates"]+=1;need(self.count["updates"]<=self.limit["updates"],"update cap");self.check()
    def reserve_forward(self):
        # collect uses explicit reservation; learning uses the hook below.
        self.forward()

def precheck(a):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    c=checked_contract(a.contract)
    if not c["synthetic"]:
        need(a.parent_receipt is not None,"precheck only from authorized parent")
        parent=json.loads(Path(a.parent_receipt).read_text())
        need(Path(a.parent_receipt)==Path(c["attempt"])/"STARTED.json" and parent["pid"]==os.getppid()
             and parent["contract_SHA"]==file_sha(a.contract),"precheck parent identity")
        exact(parent["authorization"])
        torch.cuda.set_per_process_memory_fraction(CAPS["allocated_bytes"]/torch.cuda.get_device_properties(0).total_memory,0)
    device=torch.device("cpu" if c["synthetic"] else "cuda:0")
    # This subprocess builds no detector; real mode uses CUDA DN random operations only.
    import types,importlib
    pkg=types.ModuleType("native_precheck");pkg.__path__=[str(ROOT/"vendor/rtdetrv2_pytorch/src/zoo/rtdetr")]
    sys.modules["native_precheck"]=pkg
    fn=importlib.import_module("native_precheck.denoising").get_contrastive_denoising_training_group
    lib=torch.load(a.library,map_location="cpu",weights_only=True)
    controllers={arm:Controller(fn,arm,lib,device=device) for arm in POLICY["arms"]}
    embedding=torch.nn.Embedding.from_pretrained(torch.zeros(11,256,device=device))
    seed();rows=[];moved=total=0
    for images,targets,traces in loader(c,"learning"):
        rows.append(input_hash(images,targets,traces))
        for micro in range(2):
            targets1=_move(targets[micro*c["batch"]:(micro+1)*c["batch"]],device)
            start=capture_rng(device);end=None
            for arm,ctrl in controllers.items():
                restore_rng(start,device)
                try:
                    ctrl(targets1,10,300,embedding,num_denoising=100,label_noise_ratio=.5,box_noise_scale=1.)
                except Exception:
                    write(str(a.out)+".failure.json",getattr(ctrl,"failure",{"stage":"native_dn"}))
                    raise
                if end is None:end=rng_hash(device)
                need(rng_hash(device)==end,"precheck RNG arm drift")
            audit=controllers["J"].records[-1]["audit"]
            if audit:
                donor=torch.tensor(audit["donors"]);perm=torch.tensor(audit["size_permutation"])
                total+=len(donor);moved+=int((lib["errors"][donor,2:]!=lib["errors"][donor[perm],2:]).any(-1).sum())
    need(len(rows)==c["steps"] and all(v.index==2*c["steps"] for v in controllers.values()),"all-window coverage")
    need(total>0 and moved/total>=.8,"insufficient changed size blocks")
    result={"input_SHA":rows,"arms":{k:v.records for k,v in controllers.items()},
            "total_replaced":total,"changed_size_blocks":moved,"changed_fraction":moved/total,
            "detector_instantiated":False,"detector_forward":0,"separate_process":True,
            "library_SHA":file_sha(a.library),"contract_SHA":file_sha(a.contract),
            "peak_allocated":0 if c["synthetic"] else torch.cuda.max_memory_allocated()}
    torch.save(result,a.out)
    write(str(a.out)+".closed.json",{"sha256":file_sha(a.out),"windows":len(rows),"status":"PASS"})

def admitted_resources(c):
    import torchvision,scipy,importlib.metadata
    need({k:importlib.metadata.version(k) for k in c["package_versions"]}==c["package_versions"],"bound package versions")
    need(sys.executable==c["python"] and platform.python_version()==c["python_version"],"actual interpreter")
    need({"torch":torch.__version__.split("+")[0],"torchvision":torchvision.__version__.split("+")[0],
          "cuda_build":torch.version.cuda,"cudnn":torch.backends.cudnn.version()}==c["environment"],"actual environment")
    need(scipy.__version__==c["scipy_version"],"matcher scipy version")
    need(shutil.disk_usage(Path(c["attempt"]).parent).free>=CAPS["storage_bytes"]+(1<<30),"disk headroom")
    exact(c["historical_cost"])
    need(json.loads(Path(c["historical_cost"]["path"]).read_text())["cumulative_all_research"]==c["historical_cumulative"],"cost lineage")
    tasks=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    need(not tasks,"GPU compute task exists")
    query=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.free,memory.total","--format=csv,noheader,nounits"],text=True).strip().splitlines()
    need(len(query)==1,"one fixed GPU required")
    uuid,free,total=[s.strip() for s in query[0].split(",")]
    need(uuid==c["gpu_uuid"] and int(free)*(1<<20)>=CAPS["allocated_bytes"]+(1<<30),"fresh GPU/free headroom")
    torch.cuda.set_per_process_memory_fraction(CAPS["allocated_bytes"]/(int(total)*(1<<20)),0)
    need(os.environ.get("PYTORCH_CUDA_ALLOC_CONF","") in ("","backend:native"),"allocator policy")
    return {"uuid":uuid,"free_MiB":int(free),"total_MiB":int(total)}

def execute(a):
    c=checked_contract(a.contract);verify_source(c)
    if not c["synthetic"]:
        auth=json.loads(Path(a.authorization).read_text())
        need(auth=={"contract_SHA":file_sha(a.contract),"commit":c["commit"],"manifest_SHA":c["manifest"]["sha256"],
                    "scope":"ONE_DN_BLOCK_648F_300U","approved":True},"one-time authorization")
        exact(c["parent"]);exact(c["data"]["manifest"])
        need(select_ids(json.loads(Path(c["data"]["manifest"]["path"]).read_text())["records"])==c["selection"],"selection fixed rule")
    root=Path(c["attempt"]);root.mkdir(parents=True,exist_ok=False)
    write(root/"STARTED.json",{"contract_SHA":file_sha(a.contract),"synthetic":c["synthetic"],"pid":os.getpid(),
          "authorization":None if c["synthetic"] else ref(a.authorization)})
    budget=Budget(c,root);device=torch.device("cpu" if c["synthetic"] else "cuda:0")
    try:
        if not c["synthetic"]:write(root/"RESOURCES.json",admitted_resources(c))
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        initial=torch.load(c["synthetic_parent"],map_location="cpu",weights_only=True) if c["synthetic"] else None
        engine=build(c,device,initial);initial_digest=tensor_digest(sorted(engine.model.state_dict().items()))
        # Calibration and initial probe share the same unchanged parent. No training before precheck.
        calibration=collect(engine,c,"calibration",budget)
        torch.save(calibration,root/"calibration.pt");lib=make_library(calibration)
        torch.save(lib,root/"library.pt")
        write(root/"LIBRARY_CLOSED.json",{"library":ref(root/"library.pt"),"calibration":ref(root/"calibration.pt"),"small_donors":int((lib["groups"]==0).sum())})
        del engine;gc.collect()
        if device.type=="cuda":torch.cuda.empty_cache()
        env=os.environ.copy()
        if c["synthetic"]:env["CUDA_VISIBLE_DEVICES"]=""
        cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"precheck","--contract",str(Path(a.contract).resolve()),
             "--library",str(root/"library.pt"),"--out",str(root/"precheck.pt"),"--parent-receipt",str(root/"STARTED.json")]
        with (root/"precheck.stdout").open("xb") as out,(root/"precheck.stderr").open("xb") as err:
            result=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=max(1,budget.limit["wall_seconds"]-(time.monotonic()-budget.start)))
        need(result.returncode==0,"precheck subprocess STOP before learning")
        closed=json.loads((root/"precheck.pt.closed.json").read_text())
        need(closed["sha256"]==file_sha(root/"precheck.pt"),"precheck closed SHA")
        checked=torch.load(root/"precheck.pt",map_location="cpu",weights_only=True)
        budget.peak=max(budget.peak,checked["peak_allocated"])
        need(checked["contract_SHA"]==file_sha(a.contract) and checked["library_SHA"]==file_sha(root/"library.pt"),"precheck binding")
        engine=build(c,device,initial)
        probe=collect(engine,c,"probe",budget)
        need(sum(int((r["raw_area"]<1024).sum()) for r in probe)>=100,"small probe count")
        torch.save(probe,root/"probe_initial.pt");del engine;gc.collect()
        assoc={r["image"]:r for r in probe};endpoints={};initials={}
        for arm in POLICY["arms"]:
            engine=build(c,device,initial)
            need(engine.completed_updates==0 and not engine.optimizer.state and engine.ema.updates==0,"fresh optimizer/EMA/cursor")
            initials[arm]=tensor_digest(sorted(engine.model.state_dict().items()))
            need(initials[arm]==initial_digest,"common initial model/buffers")
            ctrl=Controller(native_fn(engine.model),arm,lib,checked["arms"][arm],device)
            obs=attach(engine,ctrl);cost=engine.model.register_forward_pre_hook(budget.forward)
            update_hook=engine.optimizer.register_step_post_hook(budget.updated)
            seed()
            for i,(images,targets,traces) in enumerate(loader(c,"learning")):
                need(input_hash(images,targets,traces)==checked["input_SHA"][i],"training/precheck input differs")
                if c["synthetic"] and arm=="J" and i==0:
                    # Synthetic-only extra forward/backward proves the DN supervision path.
                    before=complete_digest(engine);state=copy.deepcopy(engine.model.state_dict())
                    rng=capture_rng(device);engine.model.train()
                    t=_move(targets[:c["batch"]],device)
                    outputs=engine.model(images[:c["batch"]].to(device),targets=t)
                    losses=engine.criterion(common.fp32_outputs(outputs),t)
                    dn=[v for k,v in losses.items() if "dn" in k]
                    need(len(dn)>0 and len(losses)==21,"actual criterion DN outputs")
                    denominator=max(1,sum(len(v["labels"]) for v in t))*outputs["dn_meta"]["dn_num_group"]
                    diags=[v for v in engine.criterion.last_diagnostics if v["head"].startswith("dn")]
                    need(len(diags)==3 and all(v["denominator"]==denominator for v in diags),"DN loss denominator")
                    sum(dn).backward()
                    parameters=dict(engine.model.named_parameters())
                    emb=parameters["decoder.denoising_class_embed.weight"].grad
                    need(emb is not None and torch.isfinite(emb).all() and bool(emb.abs().sum()>0),"DN label gradient")
                    need(any(n.startswith("backbone") and p.grad is not None and torch.isfinite(p.grad).all()
                             and bool(p.grad.abs().sum()>0) for n,p in parameters.items()),"DN-only shared feature gradient")
                    engine.model.load_state_dict(state,strict=True);engine.optimizer.zero_grad(set_to_none=True)
                    restore_rng(rng,device);ctrl.index=0;ctrl.records=[]
                    del outputs,losses,dn,state,t,parameters,emb
                    need(complete_digest(engine)==before,"DN gradient check changed training state")
                    write(root/"CPU_DN_GRADIENT.json",{"DN_only_backbone_and_embedding_nonzero_finite":True,"loss_terms":21,"DN_denominator_verified":True,"extra_CPU_forward":1,"optimizer_updates":0})
                r=engine.train_window(images,targets,capture=True)
                need(r["lr"]==1e-5 and len(r["microbatch_losses"])==2,"fixed schedule")
                with (root/("train-"+arm+".jsonl")).open("a") as f:
                    f.write(json.dumps({k:r[k] for k in ("update","lr","loss","gradient_norm_before_clip","dn_noise_sha256")})+"\n")
                if c["synthetic"] and arm=="J" and i==0:
                    save_state(root/"synthetic-J-u1.pt",engine,c,arm,ctrl)
                if c["synthetic"] and arm=="J" and i==1:
                    expected=complete_digest(engine)
                    restore_state(root/"synthetic-J-u1.pt",engine,c,arm,ctrl)
                    replay_result=engine.train_window(images,targets,capture=True)
                    need(complete_digest(engine)==expected and replay_result["output_sha256"]==r["output_sha256"],"full restore/replay mismatch")
                    write(root/"CPU_RESTORE.json",{"model_optimizer_EMA_RNG_cursor_next_window_exact":True,"extra_CPU_forward":2,"extra_CPU_updates":1})
                budget.check()
            need(engine.completed_updates==c["steps"] and ctrl.index==2*c["steps"],"endpoint clock")
            save_state(root/("state-"+arm+".pt"),engine,c,arm,ctrl)
            obs.remove();cost.remove();update_hook.remove();del engine.model.decoder._dn_generator
            endpoints[arm]=collect(engine,c,"probe",budget,assoc)
            torch.save(endpoints[arm],root/("probe-"+arm+".pt"))
            del engine;gc.collect()
        result=summarize(probe,endpoints)
        result.update(initial_state_SHA=initials,synthetic=c["synthetic"],engineering="PASS",
                      ordinary_query_identity_limit=True,extra_GPU_acceptance_performed=False)
        write(root/"RESULT.json",result)
        budget.check();need(budget.count==({"forward":648,"updates":300} if not c["synthetic"] else
             {"forward":len(plan(c,"calibration"))+4*len(plan(c,"probe"))+6*c["steps"]+3,"updates":3*c["steps"]+1}),"closed count")
        export(root,c)
        write(root/"CLOSED.json",{"engineering":"PASS","scientific":result["scientific_status"],
                                  "counts":budget.count,"export_SHA":file_sha(root/"EXPORT.json")})
    except BaseException as e:
        try:
            if torch.cuda.is_initialized():budget.peak=max(budget.peak,torch.cuda.max_memory_allocated())
            budget.persist();final_resource="SAMPLED"
        except Exception:final_resource="UNKNOWN"
        failure=root/"precheck.pt.failure.json"
        if failure.exists():
            audit=json.loads(failure.read_text());details=audit.get("audit") or {}
            write(root/"CONTROL_FAILURE.json",{"arm":audit.get("arm"),"microbatch":audit.get("microbatch"),
                 "statistics":{k:v for k,v in details.items() if k not in ("donors","size_permutation")}})
        (root/"stderr.txt").write_text(traceback.format_exc())
        write(root/"STOP.json",{"error":str(e),"counts":budget.count,"automatic_retry":False,"final_resource_sampling":final_resource})
        export(root,c)
        raise

def export(root,c):
    # Only bounded scalar/aggregate reports. No checkpoints, features or raw matches.
    names=[n for n in ("RESULT.json","LEDGER.json","STOP.json","stderr.txt","RESOURCES.json","precheck.stderr","CONTROL_FAILURE.json") if (root/n).exists()]
    total=sum((root/n).stat().st_size for n in names)
    need(total<=(1<<20)-16384,"return whitelist cap")
    write(root/"EXPORT.json",{"files":[ref(root/n) for n in names],"bytes":total,
          "stdout_stderr_remaining_allowance":(1<<20)-total,"checkpoint_download":False})

def supervise(a):
    c=checked_contract(a.contract);verify_source(c)
    directory=Path(c["attempt"]+"-launcher")
    need((directory/"DISPATCH.json").exists() or directory.is_dir(),"launcher scope")
    started=time.monotonic();root=Path(c["attempt"])
    cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"execute","--contract",str(Path(a.contract).resolve())]
    if a.authorization:cmd+=["--authorization",str(Path(a.authorization).resolve())]
    with (directory/"child.stdout").open("xb") as out,(directory/"child.stderr").open("xb") as err:
        child=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True)
    timeout=c.get("CPU_supervisor_timeout",600) if c["synthetic"] else min(CAPS["GPU_seconds"],CAPS["wall_seconds"])-5
    timed_out=False
    try:code=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:code=child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=child.wait(timeout=2)
    elapsed=time.monotonic()-started
    receipt={"returncode":code,"timed_out":timed_out,"wall_seconds":elapsed,"no_retry":True,
             "synthetic":c["synthetic"],"GPU_seconds_conservative":0 if c["synthetic"] else elapsed}
    write(directory/"RUN_EXIT.json",receipt)
    if root.exists():
        ledger=json.loads((root/"LEDGER.json").read_text()) if (root/"LEDGER.json").exists() else {"forward":0,"updates":0}
        if timed_out and not (root/"STOP.json").exists():
            write(root/"STOP.json",{"error":"supervisor timeout","counts":ledger,"final_resource_sampling":"UNKNOWN","no_retry":True})
        prior={"forward":0,"GPU_stage_seconds":0} if c["synthetic"] else c["historical_cumulative"]
        cost={"prior":prior,"new_forward":ledger["forward"],"new_updates":ledger["updates"],
              "new_GPU_seconds":receipt["GPU_seconds_conservative"],
              "cumulative":{"forward":prior["forward"]+(0 if c["synthetic"] else ledger["forward"]),
                            "GPU_stage_seconds":prior["GPU_stage_seconds"]+receipt["GPU_seconds_conservative"]},
              "synthetic_not_charged_to_GPU_budget":c["synthetic"],"no_refund":True}
        write(root/"COST_ADDITION.json",cost)
        files=[ref(root/n) for n in ("RESULT.json","LEDGER.json","STOP.json","stderr.txt","RESOURCES.json",
                                      "CONTROL_FAILURE.json","COST_ADDITION.json","CLOSED.json") if (root/n).exists()]
        files.append(ref(directory/"RUN_EXIT.json"))
        need(sum(f["bytes"] for f in files)<CAPS["return_bytes"]-16384,"final export cap")
        write(root/"FINAL_EXPORT.json",{"files":files,"bytes":sum(f["bytes"] for f in files),
                                       "reserved_stdout_receipts_bytes":16384})
    raise SystemExit(code if code>=0 else 1)

def launch(a):
    c=checked_contract(a.contract);verify_source(c)
    if not c["synthetic"]:
        need(not a.wait,"GPU launch must detach; no local polling")
        need(a.authorization is not None,"separate GPU authorization required")
        approval=json.loads(Path(a.authorization).read_text())
        need(approval=={"contract_SHA":file_sha(a.contract),"commit":c["commit"],"manifest_SHA":c["manifest"]["sha256"],
                        "scope":"ONE_DN_BLOCK_648F_300U","approved":True},"fixed approval")
    directory=Path(c["attempt"]+"-launcher");directory.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"supervise","--contract",str(Path(a.contract).resolve())]
    if a.authorization:cmd+=["--authorization",str(Path(a.authorization).resolve())]
    with (directory/"stdout.txt").open("xb") as out,(directory/"stderr.txt").open("xb") as err:
        child=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True)
    write(directory/"DISPATCH.json",{"pid":child.pid,"contract_SHA":file_sha(a.contract),"command":cmd,"single_launch":True})
    if a.wait:
        code=child.wait(timeout=600)
        need(code==0,"synthetic launch child failed; see "+str(directory/"stderr.txt"))
    print(json.dumps({"pid":child.pid,"status":"CPU_CLOSED" if a.wait else "DISPATCHED_NOT_COMPLETED",
                      "receipt":str(directory/"DISPATCH.json")}))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare");q.add_argument("--manifest",required=True);q.add_argument("--attempt",required=True);q.add_argument("--out",required=True)
    q=sub.add_parser("precheck");q.add_argument("--contract",required=True);q.add_argument("--library",required=True);q.add_argument("--out",required=True);q.add_argument("--parent-receipt")
    q=sub.add_parser("execute");q.add_argument("--contract",required=True);q.add_argument("--authorization")
    q=sub.add_parser("supervise");q.add_argument("--contract",required=True);q.add_argument("--authorization")
    q=sub.add_parser("launch");q.add_argument("--contract",required=True);q.add_argument("--authorization");q.add_argument("--wait",action="store_true")
    a=p.parse_args()
    {"prepare":prepare,"precheck":precheck,"execute":execute,"launch":launch,"supervise":supervise}[a.command](a)
if __name__=="__main__":main()
