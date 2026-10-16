"""F0-only workers dispatched by deploy.py; no implicit restart or other arms."""
from __future__ import annotations
import gc,gzip,io,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from . import checkpoint
from .model import F0Config,F0EMA,build_f0
from .engine import F0Engine,assert_healthy,tensor_digest
from .distribution import logits_probabilities
from .data import RawRoleDataset,parse_raw,training_loader
from .runio import (TASK_ID,PRETRAINED_SHA,SELECTION_SHA,require,read,write,publish,event,sha,digest,reference,execution_binding)

def dataset(contract,role):
    spec=contract["data"][role]
    return RawRoleDataset(role=role,manifest_path=spec["manifest"],images_path=spec["roots"]["images"],
                          annotations_path=spec["roots"]["annotations"],root_binding=spec["roots"],
                          selection_path=contract["selection_path"] if role=="development" else None,
                          selection_sha256=SELECTION_SHA if role=="development" else None)

def plan_epoch(data,epoch):
    if type(data.seed) is not int or data.seed not in (0,1,2):raise ValueError('epoch plan seed differs')
    generator=torch.Generator(device="cpu")
    import hashlib
    generator.manual_seed(int.from_bytes(hashlib.sha256(f"v2b-order:{data.seed}:{epoch}".encode()).digest()[:8],"big")&((1<<63)-1))
    order=torch.randperm(len(data),generator=generator).tolist()
    return {"epoch":epoch,"order":order,"ids":[data.records[i]["stable_image_id"] for i in order],
            "complete_windows":304,"dropped_indices":order[4864:]}

def check_inputs(images,targets,traces,plan,epoch,window):
    require(len(traces)==16 and len(targets)==16,"logical input size")
    ids=plan["ids"][window*16:(window+1)*16]
    require([t["image_id"] for t in traces]==ids,"input sampler identity/order mismatch")
    require(all(t["epoch"]==epoch and t["position"]==window*16+i for i,t in enumerate(traces)),"input augmentation cursor mismatch")
    return {"epoch":epoch,"window":window,"ids":ids,"augmented_sha256":[t["augmented_tensor_sha256"] for t in traces],
            "metadata_sha256":digest(traces),"valid_gt_per_sample":[len(t["labels"]) for t in targets],
            "raw_counts":[t["raw_counts"] for t in traces]}

def live_admission(out,contract):
    from .guard import screen
    from . import channel
    if contract.get('runtime_root'):
        sample=channel.resource()
    else:
        path=Path(out)/"monitor.json"
        sample=read(path if path.exists() else Path(out)/"resource-entry.json")
    require(0<=time.monotonic()-sample["monotonic"]<25,"resource monitor stale before model work")
    require(sample["boot_id"]==contract["boot_id"],"boot changed before model work")
    from .storage import identity as storage_identity
    require(storage_identity(sample['storage'])==contract['storage']['identity'],'storage mount changed before model work')
    screen(sample,gpu_uuid=contract["gpu_uuid"],gpu=True)

def prepare_cuda(contract,out):
    live_admission(out,contract)
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==contract["gpu_uuid"],"GPU UUID visibility not frozen")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8","deterministic workspace not set at process startup")
    require(not torch.cuda.is_initialized(),"unexpected prior CUDA initialization")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
    torch.cuda.init();require(torch.cuda.device_count()==1,"CUDA inventory differs from admission")
    torch.cuda.set_device(0);torch.cuda.default_generators[0].manual_seed(0)
    p=torch.cuda.get_device_properties(0)
    if getattr(p,"uuid",None) is not None:require(str(p.uuid).lower()==contract["gpu_uuid"].lower(),"CUDA/NVML UUID mismatch")
    actual={"gpu_uuid":contract["gpu_uuid"],"torch":str(torch.__version__),"torchvision":__import__("torchvision").__version__,
            "cuda_build":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu_name":p.name,
            "capability":[p.major,p.minor],"memory_bytes":p.total_memory,"cpu_threads":2,
            "deterministic_algorithms":True,"cudnn_benchmark":False,"cudnn_deterministic":True,
            "cudnn_tf32":False,"matmul_tf32":False,"cublas_workspace":":4096:8","power_clock_settings_written":False}
    for key in ("torch","torchvision","cuda_build","cudnn"):
        require(actual[key]==contract["environment"][key],"GPU worker runtime differs from frozen environment: "+key)
    event(out,"cuda_initialized",runtime=actual)
    return actual

def make_engine(contract,runtime,scope):
    from ..baseline.training_v2b import _optimizer
    config=F0Config();require(config.binding()==contract["config"],"F0 recipe differs from contract")
    model,criterion,info=build_f0(config,pretrained_path=contract["pretrained"]["path"],pretrained_sha256=PRETRAINED_SHA)
    model.to("cuda:0");criterion.to("cuda:0")
    optimizer=_optimizer(model,config);ema=F0EMA(model)
    engine=F0Engine(model,criterion,optimizer,ema,config)
    binding={"scope":scope,"run_id":TASK_ID,"config":config.binding(),"model":info,"runtime":runtime,
             "source_sha256":contract["source_sha256"],"contract_sha256":contract["contract_sha256"],
             "data_sha256":contract["data_sha256"],"pretrained_sha256":PRETRAINED_SHA}
    if contract.get('numerical_implementation'):
        from .numerics import identity
        require(contract['numerical_implementation']==identity(),'numerical implementation differs from frozen contract')
        binding['numerical_implementation']=identity()
        if contract.get('consumer_geometry_contract'):
            from .consumer_geometry import identity as geometry_identity
            require(contract['consumer_geometry_contract']==geometry_identity(),'consumer rule differs')
            binding['consumer_geometry_contract']=geometry_identity()
        binding['lineage']={**(contract['diagnostic_lineage'] if contract.get('diagnostic_lineage') and scope=='gpu_short_window' else contract['parent_lineage']),'acceptance_state':scope=='gpu_short_window'}
    if contract.get('recovery'):
        binding['lineage']={'attempt':contract.get('attempt','recovery-001'),'parent_checkpoint_sha256':contract['parent_checkpoint']['sha256'],
                            'parent_source_sha256':contract['parent_source_sha256'],
                            'parent_contract_sha256':contract['parent_contract_sha256'],
                            'acceptance_state':scope=='gpu_short_window'}
    return engine,binding

def save_state(engine,binding,out,name):
    event(out,"checkpoint_begin",name=name,update=engine.completed_updates,completed_epochs=engine.completed_updates//304)
    started=time.monotonic();receipt=checkpoint.save(engine,binding,Path(out)/name)
    event(out,"checkpoint_complete",name=name,update=engine.completed_updates,completed_epochs=engine.completed_updates//304,
          checkpoint=receipt,save_wall_seconds=time.monotonic()-started)
    return receipt

def run_short(contract,out,state_holder):
    runtime=prepare_cuda(contract,out);data=dataset(contract,"train_core");plan=plan_epoch(data,1)
    write(Path(out)/"input-plan.json",{"ids":plan["ids"][:96],"positions":list(range(96)),"unique_windows":6,"replayed_window_zero_based":5})
    engine,binding=make_engine(contract,runtime,"gpu_short_window");state_holder["engine"]=engine
    write(Path(out)/"binding.json",binding)
    initial_hash=tensor_digest(engine.model.state_dict().items())
    initial=save_state(engine,binding,out,"initial.pth");results=[];input_rows=[];saved=None
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();last=started
    for window,(images,targets,traces) in enumerate(training_loader(data,epoch=1,window_limit=6)):
        metadata=check_inputs(images,targets,traces,plan,1,window)
        state_holder["input"]=metadata;live_admission(out,contract)
        result=engine.train_window(images,targets,capture=True)
        result["input"]=metadata;result["complete_cycle_seconds"]=time.monotonic()-last
        results.append(result);input_rows.append(traces);last=time.monotonic()
        event(out,"update_complete",**result,completed_epochs=0)
        if window==4:saved=save_state(engine,binding,out,"C5.pth")
    require(len(results)==6 and engine.completed_updates==6,"short workload incomplete or expanded")
    direct=save_state(engine,binding,out,"C6-direct.pth")
    expected=checkpoint.read(direct["path"],direct["sha256"],binding)
    expected_result=results[-1];peak_direct={"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()}
    state_holder.pop("engine");del engine;gc.collect();torch.cuda.empty_cache()
    # Reconstruction is fresh; only the short-scope C5 is restored for the single replay.
    clone,clone_binding=make_engine(contract,runtime,"gpu_short_window");state_holder["engine"]=clone
    require(clone_binding==binding,"short reconstruction binding differs")
    restored=checkpoint.restore(clone,binding,saved["path"],saved["sha256"])
    event(out,"short_restore_exact",**restored)
    replay_count=0
    for images,targets,traces in training_loader(data,epoch=1,completed_windows=5,window_limit=1):
        metadata=check_inputs(images,targets,traces,plan,1,5);state_holder["input"]=metadata;live_admission(out,contract)
        replay=clone.train_window(images,targets,capture=True);replay["input"]=metadata;replay_count+=1
    require(replay_count==1,"replay loader range changed")
    comparable={k:v for k,v in expected_result.items() if k not in ("elapsed_seconds","complete_cycle_seconds")}
    actual={k:v for k,v in replay.items() if k!="elapsed_seconds"}
    require(actual==comparable,"short restored next-window output/loss/gradient/input mismatch")
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,binding)),"short restored next-window complete state mismatch")
    replay_state=save_state(clone,binding,out,"C6-replay.pth")
    write(Path(out)/"inputs.json",input_rows)
    times=[r["elapsed_seconds"] for r in results]
    measured=float(np.mean(times[1:]));checkpoint_time_size=max(saved["size_bytes"],direct["size_bytes"],replay_state["size_bytes"])
    result={"status":"PASS","scope":"gpu_short_window","unique_updates":6,"actual_updates":7,"replay_count":1,
            "initial_model_sha256":initial_hash,"initial_checkpoint":initial,"saved_checkpoint":saved,"direct_checkpoint":direct,"replay_checkpoint":replay_state,
            "full_restore":restored,"same_next_window_exact":True,"same_next_state_exact":True,
            "complete_update_seconds":times,"steady_update_mean_seconds":measured,
            "steady_update_p50_seconds":float(np.quantile(times[1:],.5)),"steady_update_p95_seconds":float(np.quantile(times[1:],.95)),
            "input_and_checkpoint_inclusive_seconds":time.monotonic()-started,"observed_peak_memory":peak_direct,
            "checkpoint_bytes":checkpoint_time_size,"formal_estimate_training_seconds":measured*36480,
            "formal_estimate_training_cny":measured*36480/3600*1.88,"estimate_excludes":"checkpoint/IO/trace/transfer/idle; only 5 steady windows",
            "bill_cny":None,"rate_cny_per_gpu_hour":1.88,"formal_state_usable":False,"additional_arms":[]}
    publish(Path(out)/"RESULT.json",result)

def run_formal(contract,out,state_holder):
    short=read(Path(contract["job_root"])/"short/RESULT.json")
    require(short["status"]=="PASS" and short["actual_updates"]==7 and short["same_next_state_exact"],"short gate absent")
    runtime=prepare_cuda(contract,out);data=dataset(contract,"train_core")
    engine,binding=make_engine(contract,runtime,"formal_F0_seed0_e120");state_holder["engine"]=engine
    write(Path(out)/"binding.json",binding)
    initial_hash=tensor_digest(engine.model.state_dict().items())
    require(initial_hash==short["initial_model_sha256"],"independent formal initialization differs from admitted recipe")
    require(engine.completed_updates==0 and not engine.optimizer.state and engine.ema.updates==0,"formal initialization was not independent")
    save_state(engine,binding,out,"checkpoint-epoch-000.pth")
    started=time.monotonic();epoch_times=[];last_receipt=None
    with gzip.open(Path(out)/"input-acknowledgements.jsonl.gz","xt",encoding="utf-8") as inputs:
        for epoch in range(1,121):
            from .control import verify_source
            verify_source(contract)
            require(engine.completed_updates==(epoch-1)*304,"formal epoch start clock")
            plan=plan_epoch(data,epoch);write(Path(out)/f"input-plan-epoch-{epoch:03d}.json",plan)
            event(out,"epoch_begin",epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan["ids"][4864:])
            epoch_start=time.monotonic();count=0
            for window,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
                require(engine.completed_updates==(epoch-1)*304+window,"formal input/update clock mismatch")
                metadata=check_inputs(images,targets,traces,plan,epoch,window);state_holder["input"]=metadata
                live_admission(out,contract);result=engine.train_window(images,targets)
                inputs.write(json.dumps(metadata,allow_nan=False)+"\n");inputs.flush()
                event(out,"update_complete",**result,completed_epochs=engine.completed_updates//304,input_sha256=digest(metadata))
                count+=1
            require(count==304 and engine.completed_updates==epoch*304,"formal epoch workload mismatch")
            last_receipt=save_state(engine,binding,out,f"checkpoint-epoch-{epoch:03d}.pth")
            elapsed=time.monotonic()-epoch_start;epoch_times.append(elapsed)
            publish(Path(out)/f"epoch-{epoch:03d}-complete.json",{"epoch":epoch,"updates":engine.completed_updates,
                    "checkpoint":last_receipt,"epoch_wall_seconds":elapsed,"strong_augmentation":epoch<118})
        saved=checkpoint.read(last_receipt["path"],last_receipt["sha256"],binding)
        require(checkpoint.exact_equal(saved,checkpoint.capture(engine,binding)),"e120 disk state differs from live complete state")
    require(engine.completed_updates==36480,"fixed formal endpoint not reached")
    elapsed=time.monotonic()-started
    endpoint={"status":"F0_E120_EMA_LOCKED","arm":"F0","seed":0,"epoch":120,"updates":36480,"checkpoint":last_receipt,
              "binding":reference(Path(out)/"binding.json"),"source_sha256":contract["source_sha256"],"formal_initial_model_sha256":initial_hash,
              "used_short_checkpoint":False,"training_wall_seconds":elapsed,"epoch_wall_seconds":epoch_times,
              "task_time_cost_estimate_cny":elapsed/3600*1.88,"rate_cny_per_gpu_hour":1.88,"actual_bill_cny":None,
              "evaluation_performed":False,"additional_arms":[],"holdout_accessed":False}
    publish(Path(out)/"endpoint-lock.json",endpoint);publish(Path(out)/"RESULT.json",endpoint)

def run_trace_cache(contract,out,state_holder):
    from .trace_contract import SCHEMA,validate_cache,save_cache,validate_collection_identities
    base=Path(contract["job_root"]);endpoint=read(base/"formal/endpoint-lock.json");binding=read(base/"formal/binding.json")
    require(endpoint["status"]=="F0_E120_EMA_LOCKED" and endpoint["epoch"]==120 and endpoint["updates"]==36480,"no fixed e120 endpoint")
    cp=endpoint["checkpoint"]
    require(cp["completed_updates"]==36480 and cp["scope"]=="formal_F0_seed0_e120" and Path(cp["path"])==base/"formal/checkpoint-epoch-120.pth","locked checkpoint is not the formal e120 state")
    require(sha(base/"formal/binding.json")==endpoint["binding"]["sha256"],"formal binding changed")
    runtime=prepare_cuda(contract,out)
    engine,new_binding=make_engine(contract,runtime,"formal_F0_seed0_e120");state_holder["engine"]=engine
    require(new_binding==binding,"trace reconstruction differs from formal binding")
    from .numerics import restore_active
    restore_active(engine,binding,cp["path"],cp["sha256"],contract=contract)
    require(engine.completed_updates==36480 and engine.ema.updates==36480,"restored trace state is not e120")
    model=engine.ema.module;model.eval();model.decoder.trace_enabled=True;data=dataset(contract,"development")
    require(len(data)==64,"trace data cardinality changed")
    before=tensor_digest(model.state_dict().items());rng_before=checkpoint.capture_rng(engine.device)
    entries=[];metadata=[];started=time.monotonic();forward_count=0
    with torch.no_grad():
        for start in range(0,64,4):
            pixels=[];raw_targets=[];rows=data.records[start:start+4]
            for position,row in enumerate(rows,start):
                image_bytes,annotation_bytes=data.bound_bytes(row)
                raw=parse_raw(annotation_bytes,row["stable_image_id"],expected_rows=row["raw_annotation_rows"])
                with Image.open(io.BytesIO(image_bytes)) as im:image=im.convert("RGB")
                require((image.width,image.height)==(row["width"],row["height"]),"trace image dimensions differ")
                image_tensor,_,_=data.transform(image,raw,epoch=120,seed=0,position=position,image_id=row["stable_image_id"],training=False)
                pixels.append(image_tensor);raw_targets.append(raw)
            live_admission(out,contract)
            with torch.autocast("cuda",dtype=torch.bfloat16):output=model(torch.stack(pixels).to("cuda:0"))
            forward_count+=1;layers=output["trace_outputs"]
            require(len(layers)==3 and "dn_outputs" not in output,"trace decoder scope changed")
            actual_probabilities=torch.stack([logits_probabilities(h["pred_corners"]).detach().cpu() for h in layers])
            actual_teacher=logits_probabilities(layers[-1]["pred_corners"],5.).detach().cpu()
            for j,(row,raw) in enumerate(zip(rows,raw_targets)):
                cache={"schema":SCHEMA,"arm":"F0","seed":0,"epoch":120,"ema":True,
                       "numerical_implementation":contract['numerical_implementation']['name'],
                       "geometry_contract":contract['consumer_geometry_contract']['name'],
                       "image_id":row["stable_image_id"],"image_sha256":row["image_sha256"],"annotation_sha256":row["annotation_sha256"],
                       "original_hw":[row["height"],row["width"]],"input_hw":[768,1344],
                       "checkpoint_sha256":cp["sha256"],"selection_sha256":SELECTION_SHA}
                for key,source in (("boxes","pred_boxes"),("references","ref_points"),("class_logits","pred_logits"),("effective_logits","pred_corners")):
                    value=torch.stack([h[source][j].detach().float().cpu() for h in layers])
                    cache[key]=value.reshape(3,300,4,33).numpy() if key=="effective_logits" else value.numpy()
                cache["probabilities"]=actual_probabilities[:,j].numpy()
                cache["teacher_probabilities"]=actual_teacher[j].numpy()
                validate_cache(cache,row,checkpoint_sha256=cp["sha256"],selection_sha256=SELECTION_SHA,raw_target=raw)
                target=Path(out)/f"trace-{start+j:03d}.npz";save_cache(target,cache)
                entries.append({**reference(target),"image_id":row["stable_image_id"]})
                metadata.append({"image_id":cache["image_id"],"checkpoint_sha256":cp["sha256"],"selection_sha256":SELECTION_SHA})
                event(out,"trace_image_saved",image_number=start+j+1,image_id=row["stable_image_id"],cache_sha256=entries[-1]["sha256"])
            del output,layers,pixels
    validate_collection_identities(metadata,data.records,checkpoint_sha256=cp["sha256"],selection_sha256=SELECTION_SHA)
    model.decoder.trace_enabled=False
    require(forward_count==16 and len(entries)==64,"trace fixed workload mismatch")
    require(before==tensor_digest(model.state_dict().items()),"trace modified EMA weights/buffers")
    require(checkpoint.exact_equal(rng_before,checkpoint.capture_rng(engine.device)),"trace mutated saved RNG state")
    require(sha(cp["path"])==cp["sha256"],"locked endpoint changed")
    publish(Path(out)/"RESULT.json",{"status":"PASS","images":64,"model_batch_forwards":16,"batch_size":4,
            "checkpoint_sha256":cp["sha256"],"selection_sha256":SELECTION_SHA,"files":entries,
            "ema_and_rng_unchanged":True,"elapsed_seconds":time.monotonic()-started,"ap_evaluation":False,"additional_arms":[]})

def run_trace_analysis(contract,out,state_holder):
    from .trace_contract import read_cache,validate_cache,validate_collection_identities
    from .trace import diagnose_image,decision
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_initialized(),"analysis must be CPU only")
    torch.set_num_threads(16);data=dataset(contract,"development");base=Path(contract["job_root"])
    collection=read(base/"trace-cache/RESULT.json")
    require(collection["status"]=="PASS" and collection["images"]==64,"complete fixed64 cache absent")
    endpoint=read(base/"formal/endpoint-lock.json")
    require(endpoint["status"]=="F0_E120_EMA_LOCKED" and endpoint["epoch"]==120 and endpoint["updates"]==36480,"analysis has no fixed F0 endpoint")
    require(endpoint["checkpoint"]["completed_updates"]==36480 and endpoint["checkpoint"]["scope"]=="formal_F0_seed0_e120" and Path(endpoint["checkpoint"]["path"])==base/"formal/checkpoint-epoch-120.pth","analysis checkpoint receipt is not the formal e120 state")
    require(sha(base/"formal/binding.json")==endpoint["binding"]["sha256"],"analysis formal binding changed")
    endpoint_sha=endpoint["checkpoint"]["sha256"]
    require(collection["checkpoint_sha256"]==endpoint_sha and collection["selection_sha256"]==SELECTION_SHA,"cache collection belongs to another endpoint/selection")
    require([x["image_id"] for x in collection["files"]]==[r["stable_image_id"] for r in data.records],"cache/raw ordered set mismatch")
    reports=[];metadata=[];report_files=[];started=time.monotonic()
    for index,(row,item) in enumerate(zip(data.records,collection["files"])):
        p=Path(item["path"]).resolve();require(p.parent==(base/"trace-cache").resolve(),"cache path outside fixed collection")
        require(sha(p)==item["sha256"] and p.stat().st_size==item["bytes"],"cache SHA/size mismatch")
        cache=read_cache(p);_,annotation=data.bound_bytes(row)
        raw=parse_raw(annotation,row["stable_image_id"],expected_rows=row["raw_annotation_rows"])
        validate_cache(cache,row,checkpoint_sha256=endpoint_sha,selection_sha256=SELECTION_SHA,raw_target=raw)
        metadata.append({"image_id":cache["image_id"],"checkpoint_sha256":cache["checkpoint_sha256"],"selection_sha256":SELECTION_SHA})
        result=diagnose_image(cache,raw,row["height"],row["width"]);result["image_id"]=row["stable_image_id"]
        target=Path(out)/f"image-{index:03d}.json"
        publish(target,result);reports.append(result);report_files.append(reference(target))
        event(out,"trace_image_analyzed",image_number=index+1,E=len(result["E"]),E_clip=len(result["E_clip"]))
    validate_collection_identities(metadata,data.records,checkpoint_sha256=endpoint_sha,selection_sha256=SELECTION_SHA)
    result=decision(reports);result.update(status="PASS",images=64,elapsed_seconds=time.monotonic()-started,
        checkpoint_sha256=collection["checkpoint_sha256"],selection_sha256=SELECTION_SHA,model_inference_calls=0,holdout_accessed=False,
        image_reports=report_files)
    publish(Path(out)/"RESULT.json",result)

def execute(contract,mode,out,*,actions=None):
    holder={};started=time.time()
    from .channel import receipt_root
    telemetry=receipt_root(out)
    try:
        publish(Path(out)/"execution-binding.json",execution_binding(contract,mode))
        (actions or {"short":run_short,"formal":run_formal,"trace-cache":run_trace_cache,"trace-analysis":run_trace_analysis})[mode](contract,out,holder)
    except BaseException as error:
        failure={"status":"STOP_PRESERVE_NO_RETRY","error":repr(error),"type":type(error).__name__,
                 "traceback":traceback.format_exc(),"input":holder.get("input"),"started_epoch":started,"finished_epoch":time.time()}
        if contract.get('runtime_root'):publish(telemetry/'failure.json',failure)
        engine=holder.get("engine")
        if engine is not None:
            engine.failed=True
            try:
                payload={"format":"F0_QUARANTINED_NONRESUMABLE","phase":engine.phase,"completed_updates":engine.completed_updates,
                         "model":engine.model.state_dict(),"optimizer":engine.optimizer.state_dict(),"ema":engine.ema.state_dict(),
                         "gradients":{n:p.grad for n,p in engine.model.named_parameters() if p.grad is not None},
                         "rng":checkpoint.capture_rng(engine.device),"numerical_error_evidence":getattr(error,"evidence",None)}
                with (Path(out)/"failure-state.pth").open("xb") as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
                failure["quarantine"]=reference(Path(out)/"failure-state.pth")
            except BaseException as save_error:failure["quarantine_save_error"]=repr(save_error)
        publish(telemetry/("failure-evidence-final.json" if contract.get('runtime_root') else 'failure.json'),failure)
        raise
    finally:
        write(telemetry/"worker-exit.json",{"pid":os.getpid(),"mode":mode,"body_returned":not (telemetry/"failure.json").exists(),
              "started_epoch":started,"finished_epoch":time.time(),"elapsed_seconds":time.time()-started,
              "process_exit_confirmed_by":"supervisor-exit.json only","total_wallclock_cap":None})


def storage_restore(source_sha256,out):
    """One synthetic CPU save/restore next-window gate on the actual new volume."""
    from .storage import path_on_mount,observe
    from ..baseline.training_v2b import _optimizer
    from .engine import learning_rate
    out=path_on_mount(out);out.mkdir(exist_ok=False);start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'storage gate must remain CPU-only')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    config=F0Config(input_hw=(128,192),physical_batch_size=1,amp_dtype='bfloat16')
    def make():
        model,criterion,info=build_f0(config,synthetic=True)
        return F0Engine(model,criterion,_optimizer(model,config),F0EMA(model),config),info
    try:
        engine,info=make()
        binding={'scope':'cpu_synthetic','run_id':TASK_ID+'/storage-restore','config':config.binding(),'model':info}
        images=torch.rand(2,3,128,192,generator=torch.Generator().manual_seed(24))
        targets=[{'boxes':torch.tensor([[.5,.5,.15,.1]],dtype=torch.float32),'labels':torch.tensor([0]),'gt_ids':['synthetic0']},
                 {'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'gt_ids':[]}]
        engine.train_window(images,targets,capture=True)
        saved=checkpoint.save(engine,binding,out/'C1.pth')
        direct=engine.train_window(images,targets,capture=True);expected=checkpoint.capture(engine,binding)
        clone,_=make();restored=checkpoint.restore(clone,binding,saved['path'],saved['sha256'])
        replay=clone.train_window(images,targets,capture=True)
        require({k:v for k,v in direct.items() if k!='elapsed_seconds'}=={k:v for k,v in replay.items() if k!='elapsed_seconds'},'storage restored next window differs')
        require(checkpoint.exact_equal(expected,checkpoint.capture(clone,binding)),'storage restored complete next state differs')
        replay_state=checkpoint.save(clone,binding,out/'C2-replay.pth')
        result={'status':'PASS','source_sha256':source_sha256,'full_state_exact':restored['full_state_exact'],
                'same_next_window_exact':True,'same_next_state_exact':True,'state':saved,'replay_state':replay_state,
                'restore':restored,'storage':observe(out),'cpu_synthetic_updates':3,'gpu_calls':0,'real_image_reads':0,
                'forbidden_as_formal_initialization':True,'elapsed_seconds':time.monotonic()-start}
    except BaseException as error:
        result={'status':'FAIL_PRESERVE','source_sha256':source_sha256,'error':repr(error),'traceback':traceback.format_exc(),
                'elapsed_seconds':time.monotonic()-start,'gpu_calls':0}
        write(out/'RESULT.json',result);raise
    publish(out/'RESULT.json',result);return reference(out/'RESULT.json')
