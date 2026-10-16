"""Authenticated terminal DN-negative decoded output domain; no computation masks."""
import torch
from .operators import NumericalIntegrityError

NAME = "terminal_valid_dn_negative_finite_aux_v1"

def check(ok, reason):
    if not ok:
        raise NumericalIntegrityError("terminal role metadata: " + reason)

def dn_roles(meta, targets, normal_queries, device):
    check(type(normal_queries) is int and normal_queries == 300, "normal query count")
    counts = [len(t["labels"]) for t in targets]
    check(bool(counts), "empty batch")
    check(all(len(t["gt_ids"]) == n and t["boxes"].shape == (n,4) for t,n in zip(targets,counts)), "GT identity alignment")
    if meta is None:
        return (torch.ones((len(counts),normal_queries),dtype=torch.bool,device=device),
                torch.zeros((len(counts),normal_queries),dtype=torch.bool,device=device))
    check(isinstance(meta,dict) and set(meta)=={"dn_positive_idx","dn_num_group","dn_num_split"}, "DN metadata fields")
    maximum=max(counts);check(maximum>0, "DN for empty GT batch")
    groups=max(1,100//maximum)
    check(type(meta["dn_num_group"]) is int and meta["dn_num_group"]==groups, "DN group count")
    check(type(meta["dn_num_split"]) is list and all(type(v) is int for v in meta["dn_num_split"]) and meta["dn_num_split"]==[2*maximum*groups,normal_queries], "DN split")
    positives=meta["dn_positive_idx"]
    check(isinstance(positives,(tuple,list)) and len(positives)==len(counts), "positive index batch")
    for got,n in zip(positives,counts):
        expected=torch.cat([torch.arange(n,device=device)+2*maximum*g for g in range(groups)])
        check(isinstance(got,torch.Tensor) and got.device==device and got.dtype==torch.int64 and torch.equal(got,expected), "positive index identity/order")
    slots=torch.arange(2*maximum*groups,device=device)
    base=(slots%maximum)[None,:]<torch.tensor(counts,device=device)[:,None]
    negative=base & ((slots//maximum)%2==1)[None,:]
    valid=torch.cat((base,torch.ones((len(counts),normal_queries),dtype=torch.bool,device=device)),dim=1)
    negative=torch.cat((negative,torch.zeros((len(counts),normal_queries),dtype=torch.bool,device=device)),dim=1)
    return valid,negative

def decoded_positive_mask(reference, valid, role):
    if role is None:
        return valid
    check(isinstance(role,dict) and set(role)=={"meta","targets","layer_index","num_layers","normal_queries"}, "role fields")
    check(type(role["num_layers"]) is int and role["num_layers"]==3 and type(role["layer_index"]) is int and 0<=role["layer_index"]<3, "layer identity")
    expected,negative=dn_roles(role["meta"],role["targets"],role["normal_queries"],reference.device)
    check(reference.shape==(*expected.shape,4) and isinstance(valid,torch.Tensor) and valid.dtype==torch.bool and valid.device==reference.device and torch.equal(valid,expected), "original query_valid differs")
    # This is a distinct decoded-output semantic selection only. The caller
    # passes the ORIGINAL valid to reference/support and keeps output query_valid.
    return valid & ~negative if role["layer_index"]==2 else valid

def criterion_roles(output,targets):
    if "dn_outputs" not in output:
        check("dn_meta" not in output, "orphan DN metadata")
        return
    check("dn_meta" in output and len(output["dn_outputs"])==3, "DN layer count")
    normal=output["pred_boxes"]
    valid,negative=dn_roles(output["dn_meta"],targets,normal.shape[1],normal.device)
    d=output["dn_meta"]["dn_num_split"][0]
    check(normal.shape==(*valid[:,d:].shape,4), "normal batch/query layout")
    check(torch.equal(output["query_valid"],valid[:,d:]), "normal validity changed")
    for index,h in enumerate(output["dn_outputs"]):
        check(h["pred_boxes"].shape==(*valid[:,:d].shape,4) and h["pred_logits"].shape==(*valid[:,:d].shape,10) and h["pred_corners"].shape==(*valid[:,:d].shape,132), "DN output layout")
        check(h["query_valid"].dtype==torch.bool and torch.equal(h["query_valid"],valid[:,:d]), "DN validity changed")
        check(h["ref_points"].shape==h["pred_boxes"].shape and torch.equal(h["ref_points"],h["teacher_ref_points"]), "original initial/reference differs")
