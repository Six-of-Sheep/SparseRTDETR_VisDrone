"""Fixed DN block experiment components. No automatic execution at import."""
import copy, gc, hashlib, importlib, io, json, os, random, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .protocol_r18 import model as common
from .protocol_r18.engine import Engine, output_digest, tensor_digest, assert_healthy
from .protocol_r18.checkpoint import capture_rng, restore_rng, file_sha
from .baseline.training_v2b import _optimizer, seed_cpu_sources, seed_worker, logical_batch_indices
from .baseline.training_v2b_engine import _move
from .localization.data import RawRoleDataset, RawAugmentation, parse_raw, collate_window
from .dn_block_replay import Replay, dn_entry, encode, scale_group, aligned_iou

SCHEMA="DN_BLOCK_EXPLORATION_V2"
N_SHA="be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83"
CAPS={"forward":648,"updates":300,"GPU_seconds":1800,"wall_seconds":3600,
      "allocated_bytes":21<<30,"storage_bytes":2<<30,"return_bytes":1<<20}
POLICY={"arms":["C","J","B"],"calibration":64,"learning":128,"probe":32,
        "steps":100,"batch":8,"accumulation":2,"eval_batch":4,"seed":0,
        "lr":1e-5,"endpoint":"u100_raw","definition_base":"6228e20f1229f70e85334e8fdb248c1afd54228e",
        "min_small":100,"min_protection":30,"min_moved":.8,
        "IoU_margin":.005,"harm_margin":.02,"protection_margin":-.005}

def need(ok,msg):
    if not ok: raise ValueError(msg)

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def exact(ref):
    p=Path(ref["path"])
    sizes=[ref[k] for k in ("bytes","size_bytes") if k in ref]
    need(sizes and all(type(v) is int and v>0 for v in sizes) and len(set(sizes))==1,"size fields")
    need(p.is_absolute() and p.is_file() and not p.is_symlink(),"reference path")
    need(p.stat().st_size==sizes[0] and file_sha(p)==ref["sha256"],"reference SHA")
    return p

def ref(path):
    p=Path(path).resolve()
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":file_sha(p)}

def seed():
    seed_cpu_sources(0)
    if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(0)

def rng_hash(device):
    return output_digest(capture_rng(device))

def select_ids(records):
    rows=sorted(records,key=lambda r:(hashlib.sha256(("dn-block-v1:"+r["stable_image_id"]).encode()).hexdigest(),r["stable_image_id"]))
    need(len(rows)>=224 and len({r["stable_image_id"] for r in rows})==len(rows),"selection membership")
    return {k:[r["stable_image_id"] for r in rows[a:b]] for k,a,b in (("calibration",0,64),("learning",64,192),("probe",192,224))}

class View(Dataset):
    def __init__(self,c,part,training):
        self.c,self.part,self.training=c,part,training
        self.ids=c["selection"][part]
        if c["synthetic"]:
            self.raw=None;self.records=c["fixture_records"]
            self.by_id={r["stable_image_id"]:r for r in self.records}
        else:
            d=c["data"];self.raw=RawRoleDataset(role="train_core",manifest_path=d["manifest"]["path"],
                images_path=d["roots"]["images"],annotations_path=d["roots"]["annotations"],
                root_binding=d["roots"],input_hw=(768,1344))
            self.by_id=self.raw.by_id
        self.transform=RawAugmentation(tuple(c["input_hw"]))
    def __len__(self):return len(self.ids)
    def __getitem__(self,key):
        epoch,pos,index=key
        identity=self.ids[index];row=self.by_id[identity]
        if self.raw:
            image_bytes,anno=self.raw.bound_bytes(row)
        else:
            image_bytes=exact(row["image"]).read_bytes();anno=exact(row["annotation"]).read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as im: image=im.convert("RGB")
        raw=parse_raw(anno,identity)
        pixels,target,trace=self.transform(image,raw,epoch=epoch,seed=0,position=pos,image_id=identity,training=self.training)
        trace["worker_pid"]=os.getpid();trace["cuda_initialized"]=torch.cuda.is_initialized()
        return pixels,target,trace

def plan(c,part):
    n=len(c["selection"][part])
    if part!="learning":
        b=c["eval_batch"]
        return [[(0,i,i) for i in range(k,min(k+b,n))] for k in range(0,n,b)]
    b=2*c["batch"];steps=c["steps"];result=[];epoch=1
    while len(result)<steps:
        for k,indices in enumerate(logical_batch_indices(n,seed=0,epoch=epoch,logical_batch_size=b,completed_batches=0)):
            result.append([(epoch,k*b+i,j) for i,j in enumerate(indices)])
            if len(result)==steps:break
        epoch+=1
    return result

def loader(c,part,start=0):
    return DataLoader(View(c,part,part=="learning"),batch_sampler=plan(c,part)[start:],
        num_workers=2,collate_fn=collate_window,worker_init_fn=seed_worker,
        multiprocessing_context="spawn",prefetch_factor=2,persistent_workers=False,
        generator=torch.Generator().manual_seed(0),timeout=120)

def input_hash(images,targets,traces):
    need(all(not x["cuda_initialized"] for x in traces),"worker CUDA")
    return output_digest((images,targets,[{k:v for k,v in x.items() if k!="worker_pid"} for x in traces]))

@dataclass(frozen=True)
class ShortConfig(common.Config):
    learning_rate:float=1e-5
    windows_per_epoch:int=8
    def binding(self):
        return {**super().binding(),"experiment":SCHEMA,"scheduler":"constant_1e-5_100_updates",
                "endpoint":"u100_raw","warmup_steps":0,"epochs":None,"training_updates":100}

class ShortEngine(Engine):
    def state_dict(self):
        out=super().state_dict();n=self.config.windows_per_epoch;u=self.completed_updates
        out.update(completed_epochs=u//n,next_epoch=u//n+1,next_window=u%n,
                   sampler="fixed_selected_subset_v2b_epoch_order",epoch_windows=n)
        return out
    def validate_state_dict(self,state):
        u=state.get("completed_updates");need(type(u) is int and 0<=u<=100,"short cursor")
        expected=self.state_dict();n=self.config.windows_per_epoch
        expected.update(completed_updates=u,microsteps=2*u,completed_epochs=u//n,next_epoch=u//n+1,next_window=u%n)
        counts=state.get("parameter_optimizer_steps")
        need(isinstance(counts,dict) and counts.keys()==self.parameter_steps.keys()
             and all(type(x) is int and 0<=x<=u for x in counts.values()),"parameter clock")
        expected["parameter_optimizer_steps"]=counts
        need(state==expected,"complete short state binding")
    def train_window(self,*args,**kwargs):
        need(self.completed_updates<100,"short terminal")
        result=super().train_window(*args,**kwargs)
        result.update(epoch=(self.completed_updates-1)//self.config.windows_per_epoch+1,
                      window=(self.completed_updates-1)%self.config.windows_per_epoch)
        return result
    def rate(self,update):
        need(1<=update<=100,"short update bound");return 1e-5

def build(c,device,initial=None):
    cfg=common.Config(input_hw=tuple(c["input_hw"]),physical_batch_size=c["batch"],
                      amp_dtype="float32" if c["synthetic"] else "bfloat16")
    model,criterion,_=common.build(cfg,synthetic=True)
    if initial is not None: model.load_state_dict(initial,strict=True)
    elif not c["synthetic"]:
        p=torch.load(exact(c["parent"]),map_location="cpu",weights_only=True)
        need(c["parent"]["sha256"]==N_SHA and p["format"]=="sparse_rtdetr.sampling_span_fullstate_v1","certified parent")
        need(p["binding"]["arm"]=="r18-sampling-span-v1/B0" and p["engine"]["completed_updates"]==36480
             and p["engine"]["completed_epochs"]==120 and p["ema"]["updates"]==36480,"parent terminal")
        need(p["binding_sha256"]==c["parent"]["binding_sha256"],"parent binding")
        model.load_state_dict(p["ema"]["module"],strict=True)
    model.to(device)
    for m in model.modules():
        if isinstance(m,(torch.nn.Dropout,torch.nn.MultiheadAttention)):
            need(getattr(m,"p",getattr(m,"dropout",0))==0,"unplanned model RNG")
    train=ShortConfig(input_hw=cfg.input_hw,physical_batch_size=cfg.physical_batch_size,amp_dtype=cfg.amp_dtype,windows_per_epoch=len(c["selection"]["learning"])//(2*c["batch"]))
    opt=_optimizer(model,train);ema=common.CommonEMA(model)
    return ShortEngine(model,criterion,opt,ema,train)

def native_fn(model):
    return importlib.import_module(type(model.decoder).__module__).get_contrastive_denoising_training_group

class Controller:
    def __init__(self,native,arm,library,expected=None,device="cpu"):
        self.native,self.arm,self.lib,self.expected=native,arm,library,expected
        self.index=0;self.records=[];self.device=torch.device(device)
    def __call__(self,*args,**kwargs):
        before=rng_hash(self.device);classes=[]
        a=list(args);embedding=a[3]
        def record_embed(ids):
            classes.append(ids.detach().cpu());return embedding(ids)
        a[3]=record_embed
        ctx=None if self.arm=="C" else Replay(self.lib["errors"],self.lib["groups"],self.arm,20261030+self.index)
        try:
            out=dn_entry(self.native,*a,**kwargs,replay=ctx)
        except Exception:
            self.failure={"microbatch":self.index,"arm":self.arm,"audit":None if ctx is None else ctx.audit}
            raise
        after=rng_hash(self.device)
        record={"before_rng":before,"after_rng":after,"classes":output_digest(classes),
                "unact":output_digest(out[1]),"mask":output_digest(out[2]),"meta":output_digest(out[3]),
                "audit":None if ctx is None else ctx.audit}
        if self.expected is not None:
            need(self.index<len(self.expected) and record==self.expected[self.index],"precheck/native DN identity mismatch")
        self.records.append(record);self.index+=1
        return out
    def after_forward(self,module,args,out):
        need(self.records and rng_hash(self.device)==self.records[-1]["after_rng"],"model consumes RNG outside DN")

def attach(engine,controller):
    engine.model.decoder._dn_generator=controller
    return engine.model.register_forward_hook(controller.after_forward)

def complete_digest(engine):
    def normalize(x):
        if isinstance(x,dict):return {type(k).__name__+":"+str(k):normalize(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
        return x
    return output_digest(normalize({"model":engine.model.state_dict(),"optimizer":engine.optimizer.state_dict(),
                         "ema":engine.ema.state_dict(),"engine":engine.state_dict(),"rng":capture_rng(engine.device)}))

def save_state(path,engine,c,arm,controller):
    need(engine.phase==0 and not engine.failed and all(p.grad is None for p in engine.model.parameters()),"save boundary")
    payload={"format":SCHEMA,"contract":digest(c),"arm":arm,"model":engine.model.state_dict(),
       "optimizer":engine.optimizer.state_dict(),"ema":engine.ema.state_dict(),"engine":engine.state_dict(),
       "rng":capture_rng(engine.device),"cursor":engine.completed_updates,"dn_cursor":controller.index,
       "modes":{n:m.training for n,m in engine.model.named_modules()}}
    path=Path(path);need(not path.exists(),"checkpoint overwrite")
    torch.save(payload,path);write(str(path)+".sha.json",ref(path))
    return payload

def restore_state(path,engine,c,arm,controller):
    p=torch.load(exact(json.loads(Path(str(path)+".sha.json").read_text())),map_location=engine.device,weights_only=True)
    need(p["format"]==SCHEMA and p["contract"]==digest(c) and p["arm"]==arm,"state identity")
    engine.model.load_state_dict(p["model"],strict=True);engine.optimizer.load_state_dict(p["optimizer"])
    engine.ema.load_state_dict(p["ema"]);engine.load_state_dict(p["engine"])
    need(p["cursor"]==engine.completed_updates and p["dn_cursor"]==2*p["cursor"],"state cursor")
    for n,m in engine.model.named_modules():m.training=p["modes"][n]
    controller.index=p["dn_cursor"];restore_rng(p["rng"],engine.device)
    return p

def collect(engine,c,part,budget,association=None):
    model=engine.model;before=tensor_digest(sorted(model.state_dict().items()));before_rng=rng_hash(engine.device)
    modes={n:m.training for n,m in model.named_modules()};u=engine.completed_updates
    model.eval();rows=[]
    with torch.no_grad():
        for images,targets,traces in loader(c,part):
            # Library/probe matching has stable GT-ID order; training retains native order.
            for target in targets:
                order=sorted(range(len(target["gt_ids"])),key=lambda k:target["gt_ids"][k])
                for key in ("boxes","labels","raw_area"):target[key]=target[key][order]
                target["gt_ids"]=[target["gt_ids"][k] for k in order]
            budget.reserve_forward()
            out=model(images.to(engine.device))
            need("dn_meta" not in out and out["pred_boxes"].shape[1]==300,"ordinary final stage")
            pred=common.fp32_outputs(out);tg=_move(targets,engine.device)
            if association is None:
                match=engine.criterion.matcher(pred,tg)["indices"]
                again=engine.criterion.matcher(pred,tg)["indices"]
                need(output_digest(match)==output_digest(again),"matching nondeterminism")
            for i,t in enumerate(targets):
                b=pred["pred_boxes"][i].cpu();logits=pred["pred_logits"][i].cpu()
                need(torch.isfinite(logits).all(),"nonfinite probe logits")
                if association is None:
                    need(torch.isfinite(b).all() and (b[:,2:]>0).all(),"invalid source boxes")
                if association is None:
                    qi,gi=match[i];qi=qi.cpu();gi=gi.cpu()
                else:
                    old=association[t["image_id"]];qi=torch.tensor(old["query"],dtype=torch.long);gi=torch.tensor(old["gt_index"],dtype=torch.long)
                    need(old["gt_ids"]==t["gt_ids"],"probe GT identity changed")
                gt=t["boxes"][gi]
                valid=torch.isfinite(b[qi]).all(-1)&(b[qi,2:]>0).all(-1)
                val=torch.zeros(len(qi))
                val[valid]=aligned_iou(b[qi][valid],gt[valid])
                row={"image":t["image_id"],"gt_ids":t["gt_ids"],"query":qi.tolist(),"gt_index":gi.tolist(),
                     "all_GT":len(t["boxes"]),"unmatched_GT":len(t["boxes"])-len(gi),
                     "unmatched_queries":300-len(qi),"prediction":b[qi],"gt":gt,
                     "labels":t["labels"][gi],"raw_area":t["raw_area"][gi],"all_raw_area":t["raw_area"],"IoU":val,"invalid":(~valid).tolist(),
                     "input_sha":input_hash(images[i:i+1],[t],[traces[i]])}
                if part=="calibration":
                    row["error"]=encode(b[qi],gt);row["group"]=scale_group(gt)
                    row.update(ordinary_boxes=b,ordinary_logits=logits,all_GT_boxes=t["boxes"],
                               all_GT_labels=t["labels"],raw_counts=traces[i]["raw_counts"],
                               stage="decoder_3_final_ordinary_Q300_before_topK")
                rows.append(row)
            budget.check()
    need(before_rng==rng_hash(engine.device) and before==tensor_digest(sorted(model.state_dict().items())) and u==engine.completed_updates
         and all(p.grad is None for p in model.parameters()),"probe mutated training")
    for n,m in model.named_modules():m.training=modes[n]
    return rows

def make_library(rows):
    lib={"errors":torch.cat([r["error"] for r in rows]),"groups":torch.cat([r["group"] for r in rows])}
    need(int((lib["groups"]==0).sum())>=100,"fewer than 100 small donors")
    return lib

def summarize(initial,arms):
    by0={r["image"]:r for r in initial}
    result={}
    for scale in range(3):
        groups={a:[] for a in arms};invalid={a:0 for a in arms};base=[]
        denominator=0;unmatched=0
        for row in initial:
            area=row["raw_area"];g=(area>=1024).long()+(area>=9216).long();ix=g==scale
            base.extend(row["IoU"][ix].tolist());all_area=row["all_raw_area"];all_group=(all_area>=1024).long()+(all_area>=9216).long()
            denominator+=int((all_group==scale).sum());unmatched+=int((all_group==scale).sum())-int(ix.sum())
            for arm,rows in arms.items():
                rr={r["image"]:r for r in rows}[row["image"]]
                need(rr["query"]==row["query"] and rr["gt_ids"]==row["gt_ids"],"fixed probe association")
                groups[arm].extend(rr["IoU"][ix].tolist())
                invalid[arm]+=int(torch.tensor(rr.get("invalid",[False]*len(rr["IoU"])),dtype=torch.bool)[ix].sum())
        n=len(base);base=np.array(base);a={k:np.array(v) for k,v in groups.items()}
        table={"paired":n,"all_GT":denominator,"unmatched_GT":unmatched,
               "initial_mean":float(base.mean()) if n else None,"arms":{}}
        for arm,v in a.items():
            d=v-base;table["arms"][arm]={"invalid":invalid[arm],"mean":float(v.mean()) if n else None,
              "quantiles":np.quantile(v,[0,.25,.5,.75,1]).tolist() if n else None,
              "delta_mean":float(d.mean()) if n else None,"improved":int((d>1e-6).sum()),
              "degraded":int((d< -1e-6).sum()),"unchanged":int((abs(d)<=1e-6).sum())}
        if n:
            table["J_minus_B"]=float((a["J"]-a["B"]).mean());table["J_minus_C"]=float((a["J"]-a["C"]).mean())
            table["harm_J_minus_B"]=float(((a["J"]-base)<-1e-6).mean()-((a["B"]-base)<-1e-6).mean())
        result["SML"[scale]]=table
    s=result["S"]
    main=s["paired"]>=100 and s["J_minus_B"]>=.005 and s["J_minus_C"]>=0 and s["harm_J_minus_B"]<=.02
    protection={k:("UNKNOWN" if result[k]["paired"]<30 else ("PASS" if result[k]["J_minus_C"]>=-.005 else "FAIL")) for k in "ML"}
    status="FAIL" if not main or "FAIL" in protection.values() else ("UNKNOWN" if "UNKNOWN" in protection.values() else "PASS")
    return {"statistics":result,"main_small":"PASS" if main else "FAIL","protection":protection,
            "scientific_status":status,"AP_AR":False,"cross_seed_claim":False}
