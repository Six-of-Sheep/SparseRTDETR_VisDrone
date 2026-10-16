"""Frozen 64-image trace identity, matching and offline transport diagnostics.

This module never evaluates AP and never constructs another trained arm.
All moving-support operations are counterfactuals of saved F0 distributions.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..baseline.primary_evaluator import _ignored_fraction
from .operators import edge_order, physical_support, positive_prior, project
from .moment_prior import maximum_entropy


def official_spatial_filter(raw_target, height, width):
    """Literal toolkit region raster + existing verified detection/GT fraction.

Toolkit region indices are NOT clamped down to the image extent. A region
starting beyond the canvas is empty. Non-integer region indices are invalid in
the pinned MATLAB source, so this adapter does not silently round them.
    """
    canvas = np.zeros((height,width), dtype=np.uint8)
    for row in raw_target["rows"]:
        if row["category"] != 0: continue
        x1,y1,x2,y2 = row["box"]
        xywh = [max(1.,v) for v in (x1,y1,x2-x1,y2-y1)]
        if any(v != int(v) for v in xywh):
            raise ValueError("pinned toolkit requires integral ignored-region indices")
        x,y,w,h = map(int,xywh)
        if x <= width and y <= height:
            canvas[y-1:min(height,y+h), x-1:min(width,x+w)] = 1
    integral = canvas.cumsum(axis=0,dtype=np.int64).cumsum(axis=1,dtype=np.int64)
    image = SimpleNamespace(width=width, height=height)
    fractions = {row["id"]: _ignored_fraction(integral, image, tuple(row["box"])) for row in raw_target["gt"]}
    normal = [row for row in raw_target["gt"] if fractions[row["id"]] < .5]
    return normal, fractions, integral


def iou_matrix(gt, pred):
    gt, pred = np.asarray(gt,dtype=np.float64).reshape(-1,4), np.asarray(pred,dtype=np.float64).reshape(-1,4)
    low = np.maximum(gt[:,None,:2], pred[None,:,:2]); high = np.minimum(gt[:,None,2:],pred[None,:,2:])
    overlap = np.maximum(0,high-low).prod(-1)
    a = (gt[:,2:]-gt[:,:2]).prod(-1); b=(pred[:,2:]-pred[:,:2]).prod(-1)
    if not np.isfinite(gt).all() or not np.isfinite(pred).all() or np.any(a<=0) or np.any(b<=0):
        raise ValueError("invalid trace boxes")
    return overlap / (a[:,None]+b[None,:]-overlap)


def lexicographic_assignment(weights, gt_ids):
    """Maximum FP64 IoU sum; exact fsum ties use GT-ID then query-ID order.

Unmatched is a sentinel after real query IDs. Re-solving only feasible smaller
query alternatives avoids perturbing genuine close but unequal IoU objectives.
    """
    weights = np.asarray(weights,dtype=np.float64)
    n,q = weights.shape
    if len(gt_ids) != n or len(set(gt_ids)) != n or not np.isfinite(weights).all():
        raise ValueError("invalid assignment identity/weights")
    order = sorted(range(n),key=lambda i:gt_ids[i]); w=weights[order]
    def solve(rows, available):
        if not rows: return {}, 0.
        costs = np.concatenate((-w[np.ix_(rows,available)], np.zeros((len(rows),len(rows)))),axis=1)
        costs[:,:len(available)][w[np.ix_(rows,available)]<=0] = 1.
        rr,cc = linear_sum_assignment(costs)
        mapping = {rows[r]:(available[c] if c<len(available) and w[rows[r],available[c]]>0 else q) for r,c in zip(rr,cc)}
        return mapping, math.fsum(float(w[r,c]) for r,c in mapping.items() if c<q)
    remaining=list(range(q)); mapping,optimum=solve(list(range(n)),remaining)
    prefix={}
    for r in range(n):
        chosen=mapping[r]
        for c in remaining:
            if c>=chosen: break
            if w[r,c]<=0: continue
            suffix,_=solve(list(range(r+1,n)),[a for a in remaining if a!=c])
            candidate={**prefix,r:c,**suffix}
            score=math.fsum(float(w[a,b]) for a,b in candidate.items() if b<q)
            if score == optimum:
                chosen=c; mapping=candidate; break
        prefix[r]=chosen
        if chosen<q: remaining.remove(chosen)
    result=np.full(n,-1,dtype=np.int64)
    for row,col in prefix.items(): result[order[row]]=col if col<q else -1
    score=math.fsum(float(weights[r,c]) for r,c in enumerate(result) if c>=0)
    if score!=optimum: raise RuntimeError("lexicographic assignment changed maximum IoU objective")
    return result


def match_layers(boxes_xyxy, logits, normal_gt):
    gt=np.array([g["box"] for g in normal_gt],dtype=np.float64).reshape(-1,4)
    labels=np.array([g["category"]-1 for g in normal_gt]); ids=[g["id"] for g in normal_gt]
    matches,ious=[],[]
    for boxes, scores in zip(boxes_xyxy,logits):
        overlap=iou_matrix(gt,boxes)
        weights=np.where((labels[:,None]==np.argmax(scores,axis=-1)[None,:]) & (overlap>=.1),overlap,0.)
        indices=lexicographic_assignment(weights,ids)
        values=np.array([overlap[i,q] if q>=0 else 0. for i,q in enumerate(indices)])
        matches.append(indices); ious.append(values)
    return np.array(matches),np.array(ious)


def diagnose_image(cache, raw_target, height, width):
    normal, fractions, _=official_spatial_filter(raw_target,height,width)
    scale=np.array([width,height,width,height])
    boxes=np.asarray(cache["boxes"],dtype=np.float64)
    corners=np.concatenate((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),axis=-1)*scale
    matches,ious=match_layers(corners,cache["class_logits"],normal)
    stable=(matches[0]>=0)&(matches[0]==matches[1])&(matches[1]==matches[2])
    if not np.array_equal(cache["references"][0],cache["references"][1]) or not np.array_equal(cache["references"][0],cache["references"][2]):
        raise ValueError("F0 trace does not have a fixed first-layer support")
    report={"normal_gt":len(normal),"raw_gt":len(raw_target["gt"]),"category0_filtered_gt":len(raw_target["gt"])-len(normal),
            "stable_gt":int(stable.sum()),"all_layers_unmatched":int((matches<0).all(0).sum()),
            "switched_or_partially_unmatched":int((~stable & ~(matches<0).all(0)).sum()),
            "E":[],"E_clip":[],"reference_iou_ge_075":[],"events":[],"gt":[]}
    for index,g in enumerate(normal):
        details={"gt_id":g["id"],"raw_area":g["raw_area"],"layer_query_ids":matches[:,index].tolist(),
                 "layer_iou":ious[:,index].tolist(),"stable":bool(stable[index]),"category0_fraction":fractions[g["id"]]}
        report["gt"].append(details)
        if not stable[index] or g["raw_area"]>1024: continue
        if ious[-1,index]>=.75: report["reference_iou_ge_075"].append(g["id"])
        if not .5<=ious[-1,index]<.75: continue
        report["E"].append(g["id"])
        query=int(matches[0,index]); old=physical_support(cache["references"][0,query])
        gt_edges=np.array(g["box"],dtype=np.float64)/scale
        clipped=(gt_edges<old[:,0])|(gt_edges>old[:,-1])
        if not clipped.any(): continue
        report["E_clip"].append(g["id"])
        gt_sizes=(np.array(g["box"])[2:]-np.array(g["box"])[:2])/scale[:2]
        teacher=edge_order(cache["teacher_probabilities"][query])
        for transition in range(2):
            new=physical_support(boxes[transition,query])
            previous=edge_order(cache["probabilities"][transition,query])
            for edge in range(4):
                source=project(previous[edge],old[edge],new[edge])
                pi,eps=positive_prior(source,new[edge])
                try: mm=maximum_entropy(pi,new[edge])
                except Exception as exc:
                    if hasattr(exc,"evidence"):
                        exc.evidence.update(image_id=cache["image_id"],gt_id=g["id"],query=query,
                                            transition=transition+1,edge=edge)
                    raise
                teacher_p=project(teacher[edge],old[edge],new[edge])
                back=project(source.probability,new[edge],old[edge])
                size=float(gt_sizes[edge%2])
                over=max(old[edge,0]-gt_edges[edge],gt_edges[edge]-old[edge,-1],0.)
                report["events"].append({"gt_id":g["id"],"query":query,"transition":transition+1,"edge":edge,
                    "gt_old_clipped":bool(clipped[edge]),"gt_old_overflow_relative":float(over/size),
                    "gt_new_clipped":bool(gt_edges[edge]<new[edge,0] or gt_edges[edge]>new[edge,-1]),
                    "tv_pi_mm":float(.5*np.abs(pi-mm.probability).sum()),
                    "history":source.diagnostics,"teacher":teacher_p.diagnostics,"epsilon":eps,
                    "history_relative_mean_shift":abs(source.diagnostics["mean_shift"])/size,
                    "teacher_relative_mean_shift":abs(teacher_p.diagnostics["mean_shift"])/size,
                    "epsilon_relative_mean_shift":abs(eps["epsilon_mean_shift"])/size,
                    "roundtrip":back.diagnostics,"roundtrip_tv":float(.5*np.abs(back.probability-previous[edge].astype(np.float64)/previous[edge].sum(dtype=np.float64)).sum()),
                    "pi":pi.tolist(),"mm":mm.probability.tolist(),"mm_certificate":mm.certificate})
    return report


def decision(images):
    events=[e for image in images for e in image["events"]]
    n=sum(len(i["E"]) for i in images); clipped=sum(len(i["E_clip"]) for i in images)
    nimages=sum(bool(i["E_clip"]) for i in images)
    def quantile(values,q): return float(np.quantile(values,q,method="linear")) if values else None
    old=sum(e["gt_old_clipped"] for e in events); new=sum(e["gt_new_clipped"] for e in events)
    reduction=(old-new)/old if old else None
    tv=quantile([e["tv_pi_mm"] for e in events],.5)
    readings={"E":n,"E_clip":clipped,"E_clip_images":nimages,"E_clip_fraction":clipped/n if n else None,
              "events":len(events),"old_clipped_edge_events":old,"new_clipped_edge_events":new,
              "clipped_event_reduction":reduction,"tv_pi_mm_median":tv}
    gates={"E_ge_50":n>=50,"E_clip_ge_20":clipped>=20,"E_clip_images_ge_8":nimages>=8,
           "E_clip_fraction_ge_005":n>0 and clipped/n>=.05,"tv_median_ge_005":tv is not None and tv>=.05,
           "new_support_clipping_reduction_ge_020":reduction is not None and reduction>=.20}
    for kind in ("history","teacher"):
        tail=sum(e[kind]["tail_mass"]>.10 for e in events)/len(events) if events else None
        shift=quantile([e[kind+"_relative_mean_shift"] for e in events],.95)
        readings[kind+"_tail_over_010_fraction"]=tail; readings[kind+"_relative_mean_shift_p95"]=shift
        gates[kind+"_tail_fraction_le_010"]=tail is not None and tail<=.10
        gates[kind+"_shift_p95_le_002"]=shift is not None and shift<=.02
    shift=quantile([e["epsilon_relative_mean_shift"] for e in events],.95)
    readings["epsilon_relative_mean_shift_p95"]=shift
    gates["epsilon_shift_p95_le_0005"]=shift is not None and shift<=.005
    assert len(events)==8*clipped
    return {"readings":readings,"gates":gates,
            "recommendation":"CONSIDER_REMAINING_THREE_ARMS" if all(gates.values()) else "DO_NOT_INVEST_CURRENT_VERSION",
            "sample_evidence":"SUFFICIENT" if n>=50 and clipped>=20 and nimages>=8 else "INSUFFICIENT",
            "probability_tv_is_not_localization_gain":True,"training_or_sample_expansion_authorized":False}
