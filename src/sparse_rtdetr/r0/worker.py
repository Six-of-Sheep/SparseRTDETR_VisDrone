"""R0 workers: fixed3 acceptance, independent e120, one endpoint FP32 cache/readout."""
import copy,gc,gzip,json,random,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..baseline.training_v2b import _optimizer
from ..localization import worker as base,channel,guard
from ..localization.data import training_loader
from ..localization.distribution import require_finite
from .model import R0Config,R0EMA,build_r0
from .engine import R0Engine,tensor_digest,output_digest,assert_healthy
from . import checkpoint
from .runio import require,read,write,publish,event,reference,sha,digest,TASK_ID,PRETRAINED_SHA
SOURCE=Path(__file__).resolve().parents[3]
def reset_training_rng(device):
 random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
 if device.type=="cuda":torch.cuda.default_generators[0].manual_seed(0)
def make_engine(c,runtime,scope):
 config=R0Config();require(config.binding()==c["config"],"R0 config differs")
 model,criterion,info=build_r0(config,pretrained_path=c["pretrained"]["path"],pretrained_sha256=PRETRAINED_SHA)
 initial=tensor_digest(model.state_dict().items())
 model.to("cuda:0");criterion.to("cuda:0");optimizer=_optimizer(model,config);ema=R0EMA(model)
 e=R0Engine(model,criterion,optimizer,ema,config)
 b={"arm":"R0","scope":scope,"run_id":TASK_ID,"config":config.binding(),"model":info,"runtime":runtime,"source_sha256":c["source_sha256"],
    "contract_sha256":c["contract_sha256"],"data_sha256":c["data_sha256"],"pretrained_sha256":PRETRAINED_SHA,"geometry":c["geometry"]}
 reset_training_rng(e.device)
 return e,b,initial
def save_state(e,b,out,name):
 event(out,"checkpoint_begin",name=name,update=e.completed_updates,completed_epochs=e.completed_updates//304);started=time.monotonic()
 r=checkpoint.save(e,b,Path(out)/name)
 event(out,"checkpoint_complete",name=name,update=e.completed_updates,completed_epochs=e.completed_updates//304,checkpoint=r,save_wall_seconds=time.monotonic()-started)
 return r
def input_check(c,images,targets,traces,meta,win):
 import hashlib
 expected=read(c["fixed_inputs"])["inputs"][str(win)]
 require(meta==expected["metadata"] and [t["gt_ids"] for t in targets]==expected["gt_ids"],"R0 frozen input/GT differs")
 require(len(images)==len(targets)==len(traces)==16,"R0 physical input expanded")
 for image,target,trace in zip(images,targets,traces):
  h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
  for key in ("boxes","labels","raw_area","ignore_spatial_boxes","ignore_union","ignore_class_boxes","ignore_class_labels"):
   h.update(key.encode());h.update(target[key].contiguous().numpy().tobytes())
  h.update(json.dumps([target["gt_ids"],target["ignore_ids"]],separators=(",",":")).encode())
  require(h.hexdigest()==trace["augmented_tensor_sha256"],"R0 actual augmented input bytes differ from trace")
def clock(e,r,u):
 require(e.completed_updates==e.ema.updates==r["update"]==u and e.phase==0 and r["optimizer_steps"]==r["ema_updates"]==1 and r["finite"],"R0 full update clock mismatch")
 require(all(v["loss_count"] in (15,27) and v["classification_denominator"]==max(v["gt_count"],1) and v["regression_denominator"]==max(v["go_count"],1) for v in r["normalizers"]),"R0 loss/denominator mismatch")
 require(all(not any("fgl" in n or "ddf" in n for n in d) for d in r["microbatch_losses"]),"R0 distribution loss encountered")
def acceptance(c,out,holder):
 runtime=base.prepare_cuda(c,out);data=base.dataset(c,"train_core");plan=base.plan_epoch(data,1)
 e,b,initial=make_engine(c,runtime,"gpu_R0_acceptance");holder["engine"]=e;write(out/"binding.json",b)
 require(not e.optimizer.state and e.completed_updates==e.ema.updates==0,"R0 acceptance not fresh")
 write(out/"INITIAL_IDENTITY.json",{"initial_model_sha256":initial,"optimizer_empty":True,"EMA_updates":0,"training_rng":reference(c["fixed_inputs"]),"state_source":"new untrained ImageNet"})
 zero=save_state(e,b,out,"initial-R0.pth");rows=[]
 for win,(images,targets,traces) in enumerate(training_loader(data,epoch=1,window_limit=2)):
  meta=base.check_inputs(images,targets,traces,plan,1,win);input_check(c,images,targets,traces,meta,win);holder["input"]=meta;base.live_admission(out,c)
  r=e.train_window(images,targets,capture=True);clock(e,r,win+1);r["input"]=meta;rows.append(r);event(out,"update_complete",**r,completed_epochs=0)
  if win==0:saved=save_state(e,b,out,"C1-R0.pth")
 require(len(rows)==2 and e.completed_updates==2,"R0 acceptance fixed work differs")
 direct=save_state(e,b,out,"C2-R0-direct.pth");expected=checkpoint.read(direct["path"],direct["sha256"],b)
 holder.pop("engine");del e;gc.collect();torch.cuda.empty_cache()
 clone,new,new_initial=make_engine(c,runtime,"gpu_R0_acceptance");holder["engine"]=clone;require(new==b and initial==new_initial,"fresh R0 reconstruction differs")
 restored=checkpoint.restore(clone,b,saved["path"],saved["sha256"]);write(out/"RESTORE_C1.json",restored);count=0
 for images,targets,traces in training_loader(data,epoch=1,completed_windows=1,window_limit=1):
  meta=base.check_inputs(images,targets,traces,plan,1,1);input_check(c,images,targets,traces,meta,1);holder["input"]=meta;base.live_admission(out,c)
  replay=clone.train_window(images,targets,capture=True);clock(clone,replay,2);replay["input"]=meta;count+=1
 require(count==1 and {k:v for k,v in rows[-1].items() if k!="elapsed_seconds"}=={k:v for k,v in replay.items() if k!="elapsed_seconds"},"R0 input/DN/output/loss/gradient replay mismatch")
 require(checkpoint.exact_equal(expected,checkpoint.capture(clone,b)),"R0 full state/RNG replay mismatch")
 replay_cp=save_state(clone,b,out,"C2-R0-replay.pth")
 publish(out/"RESULT.json",{"status":"PASS","arm":"R0","actual_updates":3,"unique_updates":2,"updates":[1,2,2],"initial_model_sha256":initial,"initial_checkpoint":zero,"checkpoints":[saved,direct,replay_cp],
    "restore":restored,"input_DN_noise_result_gradient_exact":True,"full_state_RNG_exact":True,"state_for_formal":False,
    "results":rows,"replay_result":replay,"complete_update_seconds":[x["elapsed_seconds"] for x in rows]+[replay["elapsed_seconds"]],
    "peak_memory_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_memory_reserved_bytes":torch.cuda.max_memory_reserved()})
def formal(c,out,holder):
 runtime=base.prepare_cuda(c,out);data=base.dataset(c,"train_core");e,b,initial=make_engine(c,runtime,"formal_R0_seed0_e120");holder["engine"]=e;write(out/"binding.json",b)
 short=read(Path(c["job_root"])/"acceptance/RESULT.json")
 require(short["status"]=="PASS" and initial==short["initial_model_sha256"] and e.completed_updates==e.ema.updates==0 and not e.optimizer.state,"R0 formal fresh initialization differs")
 zero=save_state(e,b,out,"checkpoint-epoch-000.pth")
 event(out,"epoch_begin",epoch=0,completed_epochs=0,initial_model_sha256=initial,state_source="independent new untrained ImageNet, no restored state")
 start=time.monotonic();times=[];updates=0
 from .control import verify_source
 with gzip.open(out/"input-acknowledgements.jsonl.gz","xt",encoding="utf-8") as ack:
  for epoch in range(1,121):
   verify_source(c);require(e.completed_updates==(epoch-1)*304,"R0 epoch initial clock");plan=base.plan_epoch(data,epoch);write(out/f"input-plan-epoch-{epoch:03d}.json",plan)
   event(out,"epoch_begin",epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan["ids"][4864:]);t=time.monotonic();count=0
   for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
    meta=base.check_inputs(images,targets,traces,plan,epoch,win);holder["input"]=meta
    if epoch==1 and win<2:input_check(c,images,targets,traces,meta,win)
    base.live_admission(out,c);r=e.train_window(images,targets);clock(e,r,(epoch-1)*304+win+1);ack.write(json.dumps(meta,allow_nan=False)+"\n");ack.flush()
    event(out,"update_complete",**r,completed_epochs=e.completed_updates//304,input_sha256=digest(meta));count+=1;updates+=1
   require(count==304 and e.completed_updates==epoch*304,"R0 epoch work incomplete")
   last=save_state(e,b,out,f"checkpoint-epoch-{epoch:03d}.pth");elapsed=time.monotonic()-t;times.append(elapsed)
   publish(out/f"epoch-{epoch:03d}-complete.json",{"epoch":epoch,"updates":e.completed_updates,"checkpoint":last,"epoch_wall_seconds":elapsed,"strong_augmentation":epoch<118})
 require(updates==36480 and e.completed_updates==e.ema.updates==36480 and checkpoint.exact_equal(checkpoint.read(last["path"],last["sha256"],b),checkpoint.capture(e,b)),"R0 e120 full endpoint differs")
 endpoint={"status":"R0_E120_EMA_LOCKED","arm":"R0","seed":0,"epoch":120,"updates":36480,"actual_updates":updates,"checkpoint":last,"binding":reference(out/"binding.json"),
           "source_sha256":c["source_sha256"],"contract_sha256":c["contract_sha256"],"initial_model_sha256":initial,"initial_checkpoint":zero,"used_acceptance_state":False,"used_e050_state":False,
           "training_wall_seconds":time.monotonic()-start,"epoch_wall_seconds":times,"evaluation_before_endpoint":False,"other_arms":[],"holdout_accessed":False}
 publish(out/"endpoint-lock.json",endpoint);publish(out/"RESULT.json",endpoint)
def inference(c,out,holder):
 from .evaluation_data import Development548
 from ..localization.consumer_geometry import positive_box
 runtime=base.prepare_cuda(c,out);endpoint=read(Path(c["job_root"])/"formal/endpoint-lock.json");binding=read(Path(c["job_root"])/"formal/binding.json")
 e,b,_=make_engine(c,runtime,"formal_R0_seed0_e120");holder["engine"]=e;require(b==binding,"R0 endpoint reconstruction binding differs")
 cp=endpoint["checkpoint"];restored=checkpoint.restore(e,b,cp["path"],cp["sha256"]);require(e.completed_updates==e.ema.updates==36480,"R0 inference not e120")
 model=e.ema.module;model.eval();before=tensor_digest(model.state_dict().items());rng=checkpoint.capture_rng(e.device)
 data=Development548(c["data"]["development"]);logits=[];boxes=[];ids=[];calls=0
 with torch.inference_mode():
  for start in range(0,548,4):
   rows=data.records[start:start+4];pixels=[];meta=[]
   for position,row in enumerate(rows,start):
    image,_=data.item(row,position);pixels.append(image);meta.append({"image_id":row["stable_image_id"],"image_sha256":row["image_sha256"],"annotation_sha256":row["annotation_sha256"]})
   holder["input"]=meta;base.live_admission(out,c)
   with torch.autocast("cuda",enabled=False):result=model(torch.stack(pixels).to("cuda"))
   require(set(result)=={"pred_logits","pred_boxes"} and result["pred_logits"].shape==(4,300,10) and result["pred_boxes"].shape==(4,300,4),"R0 fixed inference schema")
   require_finite(result["pred_logits"],"endpoint logits");positive_box("R0/e120/boxes",result["pred_boxes"]);calls+=1
   logits.append(result["pred_logits"].cpu());boxes.append(result["pred_boxes"].cpu())
   for position,row in enumerate(rows,start):ids.append(row["stable_image_id"]);event(out,"evaluation_image_saved",evaluation_image_number=position+1,image_id=row["stable_image_id"],checkpoint_sha256=cp["sha256"])
 require(calls==137 and len(ids)==len(set(ids))==548 and before==tensor_digest(model.state_dict().items()) and checkpoint.exact_equal(rng,checkpoint.capture_rng(e.device)),"R0 inference count/state/RNG changed")
 raw={"schema":"r0_raw_endpoint_queries_v1","arm":"R0","epoch":120,"EMA":True,"image_ids":ids,"pred_logits":torch.cat(logits),"pred_boxes":torch.cat(boxes),"query_ids":torch.arange(300),
      "checkpoint_sha256":cp["sha256"],"source_sha256":c["source_sha256"],"evaluation_protocol_sha256":c["evaluation_protocol_sha256"]}
 with (out/"raw-queries-R0-e120.pth").open("xb") as f:torch.save(raw,f)
 publish(out/"RESULT.json",{"status":"PASS","images":548,"model_batch_forwards":137,"batch_size":4,"inference_FP32":True,"EMA_and_RNG_unchanged":True,"checkpoint_sha256":cp["sha256"],
                          "raw":reference(out/"raw-queries-R0-e120.pth"),"restore":restored,"budgets":[600,300],"holdout_accessed":False})
def evaluate(c,out,holder):
 from .evaluation_data import Development548
 from .evaluation import rank_candidates,score_budget
 rawref=read(Path(c["job_root"])/"inference/RESULT.json")["raw"];require(reference(rawref["path"])==rawref,"R0 raw cache changed")
 raw=torch.load(rawref["path"],map_location="cpu",weights_only=True);data=Development548(c["data"]["development"])
 endpoint=read(Path(c["job_root"])/"formal/endpoint-lock.json")
 require(raw["image_ids"]==[r["stable_image_id"] for r in data.records] and raw["checkpoint_sha256"]==endpoint["checkpoint"]["sha256"] and raw["evaluation_protocol_sha256"]==c["evaluation_protocol_sha256"],"R0 readout cache identity differs")
 parsed=[data.raw(row) for row in data.records];results={}
 for budget in (600,300):
  predictions=[p for i,row in enumerate(data.records) for p in rank_candidates(raw["pred_logits"][i],raw["pred_boxes"][i],row,budget)]
  write(out/f"top{budget}-predictions.json",predictions);r=score_budget(data.records,parsed,predictions,SOURCE);write(out/f"top{budget}-metrics.json",r);results[str(budget)]=r
  event(out,"evaluation_budget_complete",name=f"top{budget}",budget=budget,checkpoint_sha256=raw["checkpoint_sha256"],images=548)
 publish(out/"RESULT.json",{"status":"PASS","arm":"R0","epoch":120,"images":548,"budgets":[600,300],"results":results,"checkpoint_sha256":raw["checkpoint_sha256"],
                          "model_forward_calls":0,"standalone_points":True,"paired_comparator_present":False,"holdout_accessed":False})

