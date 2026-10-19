"""Fixed endpoint scoring; both budgets from the same FP32 raw query cache."""
import contextlib,copy,json
from dataclasses import asdict
import numpy as np
import torch
from ..baseline import primary_evaluator as primary
from ..data_protocol.evaluation import PrimaryImageV2,PrimaryGroundTruth,PrimaryEvaluatorInputV2,Detection
from ..localization.consumer_geometry import corners
POLICY={"id":"r0_endpoint_development_raw_ignore_v1","endpoint":"fixed e120 EMA","images":548,"batch_size":4,"inference_dtype":"FP32","canvas":[768,1344],
        "raw_queries":300,"classes":10,"budgets":[600,300],"main":"top600","secondary":"top300","ranking":"descending FP32 sigmoid score, query ID, class ID",
        "threshold":None,"nms":False,"tta":False,"prediction_clip":False,"truncate_before_ignore_no_refill":True,
        "official_toolkit_commit":"005445782213e20cb91bc50a597db3dd949e749a","official_APS":False,
        "COCO_bridge":"same official category0 raster prefilter; class score0 iscrowd1; macro classes IoU.50:.05:.95/101 recall/maxDets100",
        "area":"original raw width*height; native COCO small/medium overlapping endpoints retained",
        "statistics":"standalone endpoint point metrics; no paired difference without a new same-protocol comparator"}
def rank_candidates(logits,boxes,row,budget):
 if budget not in (600,300) or logits.shape!=(300,10) or boxes.shape!=(300,4) or logits.dtype!=torch.float32 or boxes.dtype!=torch.float32:raise ValueError("R0 cache/ranking schema")
 if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all() or (boxes[:,2:]<=0).any():raise ValueError("R0 invalid endpoint cache")
 xy=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
 corners("R0/evaluation/normalized_corners",xy)
 xy=xy*torch.tensor([row["width"],row["height"]]*2,dtype=torch.float32)
 corners("R0/evaluation/original_pixel_corners",xy)
 scores=logits.sigmoid().numpy().reshape(-1);q=np.repeat(np.arange(300),10);cl=np.tile(np.arange(10),300)
 order=np.lexsort((cl,q,-scores))[:budget]
 return [{"image_id":row["stable_image_id"],"query_id":int(q[i]),"category_id":int(cl[i])+1,"score":float(scores[i]),
          "bbox_xyxy":[float(x) for x in xy[q[i]].tolist()]} for i in order]
def truth(row,parsed):
 out=[]
 for obj in parsed["rows"]:
  box=tuple(float(x) for x in obj["box"]);area=(box[2]-box[0])*(box[3]-box[1])
  out.append(PrimaryGroundTruth(obj["id"],row["stable_image_id"],int(obj["category"]),box,area,obj["category"]==0,obj["score"]==0))
 return out
def primary_inputs(rows,parsed,predictions):
 images=tuple(PrimaryImageV2(r["stable_image_id"],int(r["width"]),int(r["height"])) for r in rows)
 gt=tuple(g for r,p in zip(rows,parsed) for g in truth(r,p))
 detections=tuple(Detection(p["image_id"],p["category_id"],tuple(p["bbox_xyxy"]),p["score"]) for p in predictions)
 return PrimaryEvaluatorInputV2(images,detections,gt)
def bridge_inputs(rows,parsed,predictions,*,gt_extension=None):
 if gt_extension is not None:primary._validate_input(primary_inputs(rows,parsed,predictions),gt_extension=gt_extension)
 # Reuse the exact frozen official raster primitive. Do not replace with
 # continuous training IoA or move budget truncation after this filter.
 ids={r["stable_image_id"]:i+1 for i,r in enumerate(rows)};gt=[];det=[];counts={"GT_before":0,"GT_after":0,"detections_before":len(predictions),"detections_after":0}
 pred_by={r["stable_image_id"]:[] for r in rows}
 for p in predictions:pred_by[p["image_id"]].append(p)
 for row,raw in zip(rows,parsed):
  image=PrimaryImageV2(row["stable_image_id"],row["width"],row["height"]);regions=[g for g in truth(row,raw) if g.category_id==0];integral=primary._build_integral(image,regions,gt_extension=gt_extension)
  for obj in raw["rows"]:
   if obj["category"] not in range(1,11):continue
   counts["GT_before"]+=1;box=tuple(obj["box"])
   if primary._ignored_fraction(integral,image,box,gt_extension=gt_extension)>=.5:continue
   gt.append({"id":len(gt)+1,"image_id":ids[row["stable_image_id"]],"category_id":int(obj["category"]),"bbox":[box[0],box[1],box[2]-box[0],box[3]-box[1]],
              "area":float(obj["raw_area"]),"iscrowd":int(obj["score"]==0),"raw_annotation_id":obj["id"],"raw_score":obj["score"]})
   counts["GT_after"]+=1
  for p in pred_by[row["stable_image_id"]]:
   b=tuple(p["bbox_xyxy"])
   if primary._ignored_fraction(integral,image,b)>=.5:continue
   det.append({"image_id":ids[p["image_id"]],"category_id":p["category_id"],"bbox":[b[0],b[1],b[2]-b[0],b[3]-b[1]],"score":p["score"]})
   counts["detections_after"]+=1
 dataset={"info":{"description":"R0 raw-ignore COCO bridge"},"images":[{"id":ids[r["stable_image_id"]],"width":r["width"],"height":r["height"]} for r in rows],
          "categories":[{"id":i,"name":str(i)} for i in range(1,11)],"annotations":gt}
 return dataset,det,counts
def bridge_metrics(dataset,predictions):
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 coco=COCO();coco.dataset=dataset;coco.createIndex()
 if predictions:dt=coco.loadRes(predictions)
 else:
  dt=COCO();dt.dataset={"info":dataset["info"],"images":dataset["images"],"categories":dataset["categories"],"annotations":[]};dt.createIndex()
 evaluator=COCOeval(coco,dt,"bbox");evaluator.params.imgIds=sorted(r["id"] for r in dataset["images"]);evaluator.params.catIds=list(range(1,11));evaluator.params.maxDets=[1,10,100]
 evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
 names=("AP","AP50","AP75","APS","APM","APL","AR1","AR10","AR100","AR_small","AR_medium","AR_large")
 values={k:float(v) for k,v in zip(names,evaluator.stats)}
 # Explicit AP75-small from the same actual COCO small area/maxDets100 cells.
 precision=evaluator.eval["precision"];small=precision[5,:,:,1,2];valid=small[small>-1]
 values["AP75_small"]=float(valid.mean()) if len(valid) else -1.
 if not all(np.isfinite(v) for v in values.values()):raise ValueError("nonfinite endpoint metrics")
 return {"protocol":POLICY["id"],"units":"0..1 AP, -1 means no GT in an area/class aggregation","metrics":values,
         "maxDets":[1,10,100],"iouThrs":evaluator.params.iouThrs.tolist(),"areaRng":evaluator.params.areaRng,"category_macro":True}
def score_budget(rows,parsed,predictions,repo_root,*,gt_extension=None):
 binding=primary.primary_evaluator_contract_binding(repo_root);inputs=primary_inputs(rows,parsed,predictions)
 official=primary.evaluate_primary_v1(inputs,binding,gt_extension=gt_extension);primary.validate_primary_evaluator_result(official,inputs,binding,gt_extension=gt_extension)
 dataset,det,counts=bridge_inputs(rows,parsed,predictions,gt_extension=gt_extension);bridge=bridge_metrics(dataset,det)
 result={"official":asdict(official),"COCO_bridge":bridge,"spatial_filter_counts":counts,"official_APS":None}
 if gt_extension is not None:
  from ..data_protocol.supplemental_zero_gt import specification
  result["evaluation_input_extension"]=specification()
 return result

