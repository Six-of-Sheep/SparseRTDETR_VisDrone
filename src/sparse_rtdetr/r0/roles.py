"""Read-only authentication of R0 head/DN roles; no mask transformation."""
import torch
from ..localization.terminal_geometry import dn_roles,check
def criterion_roles(output,targets):
    def walk(value):
        if isinstance(value,dict):
            check(not set(value)&{"pred_corners","teacher_corners","teacher_logits","support_parameters"},"distribution fields forbidden in R0")
            for v in value.values():walk(v)
        elif isinstance(value,(list,tuple)):
            for v in value:walk(v)
    walk(output)
    box=output["pred_boxes"];meta=output.get("dn_meta");v,_=dn_roles(meta,targets,box.shape[1],box.device)
    d=meta["dn_num_split"][0] if meta else 0
    check(box.shape==(*v[:,d:].shape,4) and output["pred_logits"].shape==(*v[:,d:].shape,10),"normal R0 head shape")
    check(torch.equal(output["query_valid"],v[:,d:]),"normal R0 validity drift")
    check(len(output["aux_outputs"])==2 and len(output["enc_aux_outputs"])==1,"R0 five GO head count")
    for h in output["aux_outputs"]:
        check(h["pred_boxes"].shape==box.shape and torch.equal(h["query_valid"],v[:,d:]),"R0 auxiliary role drift")
    if meta is not None:
        check(len(output["dn_outputs"])==3,"R0 DN head count")
        for h in output["dn_outputs"]:
            check(h["pred_boxes"].shape==(*v[:,:d].shape,4) and h["pred_logits"].shape==(*v[:,:d].shape,10) and torch.equal(h["query_valid"],v[:,:d]),"R0 DN role shape/validity")
        check(output["dn_pre_outputs"]["pred_boxes"].shape==(*v[:,:d].shape,4),"R0 DN pre head")
        noise=output["dn_noise"]
        check(noise["embedding_calls"]==1 and noise["classes"].shape==v[:,:d].shape and noise["boxes_unact"].shape==(*v[:,:d].shape,4) and noise["attention_mask"].shape==(v.shape[1],v.shape[1]),"R0 actual DN noise capture")
        check(noise["metadata"] is meta or all(torch.equal(a,b) for a,b in zip(noise["metadata"]["dn_positive_idx"],meta["dn_positive_idx"])),"R0 DN capture metadata")
    else:
        check("dn_outputs" not in output and "dn_pre_outputs" not in output,"R0 orphan DN output")
        check(output["dn_noise"]["embedding_calls"]==0 and output["dn_noise"]["classes"] is None and output["dn_noise"]["boxes_unact"] is None,"R0 unexpected no-DN noise")
