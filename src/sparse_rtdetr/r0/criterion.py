"""FGL/GO-LSD with explicit per-family physical-microbatch normalizers."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou

from ..localization.distribution import box_to_corners, require_finite
from ..localization.ignore import negative_keep_mask
from ..localization.consumer_geometry import tensor,tree,overlap,matching


def source_indices(indices):
    return (torch.cat([torch.full_like(q, b) for b, (q, _) in enumerate(indices)]),
            torch.cat([q for q, _ in indices]))


def go_union(votes, targets):
    results = []
    for batch, target in enumerate(targets):
        counts = Counter((int(q), int(g)) for vote in votes for q, g in zip(vote[batch][0].tolist(), vote[batch][1].tolist()))
        by_query = defaultdict(list)
        ids = target["gt_ids"]
        for (q, g), frequency in counts.items():
            by_query[q].append((frequency, ids[g], g))
        pairs = [(q, sorted(choices, key=lambda row: (-row[0], row[1]))[0][2]) for q, choices in sorted(by_query.items())]
        device = target["labels"].device
        results.append((torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device),
                        torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)))
    return results


def matched_boxes(output, targets, indices):
    idx = source_indices(indices)
    selected = output["pred_boxes"][idx].float()
    gt = torch.cat([t["boxes"][g] for t, (_, g) in zip(targets, indices)]).float()
    iou = overlap("loss/iou",box_iou,box_to_corners(selected),box_to_corners(gt)).diag().detach()
    return idx, selected, gt, iou


class R0Criterion(nn.Module):
    weights = {"vfl":1.,"bbox":5.,"giou":2.}

    def __init__(self, matcher, alpha=.75, gamma=2.):
        super().__init__()
        self.matcher, self.alpha, self.gamma = matcher, alpha, gamma
        self.last_diagnostics = None

    def recipe(self):
        return {"weights":dict(self.weights),"alpha":self.alpha,"gamma":self.gamma,
                "representation":"r0_scalar_sigmoid_go_v1","go_heads":5,"go_tie":"stable_gt_id",
                "normalization":"per_family_physical_batch", "ignore":"raw_role_v1",
                "matcher":{k:getattr(self.matcher,k) for k in
                           ("cost_class","cost_bbox","cost_giou","use_focal_loss","alpha","gamma")}}

    def classification(self, output, targets, indices, denominator, *, ignore=True):
        idx, _, _, iou = matched_boxes(output, targets, indices)
        z = output["pred_logits"].float()
        labels = torch.cat([t["labels"][g] for t, (_, g) in zip(targets, indices)])
        classes = torch.full(z.shape[:2], 10, dtype=torch.long, device=z.device)
        classes[idx] = labels
        one_hot = F.one_hot(classes, 11)[..., :10]
        quality = torch.zeros(z.shape[:2], dtype=z.dtype, device=z.device)
        quality[idx] = iou
        score = quality[:, :, None] * one_hot
        weight = self.alpha * z.sigmoid().detach().pow(self.gamma) * (1-one_hot) + score
        tensor("classification/logits",z);tensor("classification/target_score",score,False);tensor("classification/weight",weight,False)
        per_class = F.binary_cross_entropy_with_logits(z, score, weight=weight, reduction="none")
        tensor("classification/per_class",per_class)
        if ignore:
            keep = negative_keep_mask(output["pred_boxes"], indices, targets)
            per_class = per_class * keep
            masked = int((~keep).sum())
        else:
            masked = 0
        return per_class.sum() / denominator, masked

    def regression(self, output, targets, indices, denominator):
        idx, pred, gt, iou = matched_boxes(output, targets, indices)
        losses = {"bbox": F.l1_loss(pred, gt, reduction="sum") / denominator,
                  "giou": (1.-overlap("loss/giou",generalized_box_iou,box_to_corners(pred),box_to_corners(gt),True).diag()).sum() / denominator}
        return losses

    def forward(self, output, targets):
        from .roles import criterion_roles
        criterion_roles(output, targets)
        if len(output["aux_outputs"]) != 2 or len(output["enc_aux_outputs"]) != 1:
            raise ValueError("R0 criterion requires all five GO voting heads")
        heads = [("", output), ("_aux_0", output["aux_outputs"][0]),
                 ("_aux_1", output["aux_outputs"][1]), ("_pre", output["pre_outputs"]),
                 ("_enc_0", output["enc_aux_outputs"][0])]
        tree("loss/outputs",output)
        with matching(self.matcher):
            votes = [self.matcher({**h,"pred_logits":h["pred_logits"].float(),"pred_boxes":h["pred_boxes"].float()}, targets)["indices"] for _, h in heads]
        go = go_union(votes, targets)
        ngt = sum(len(t["labels"]) for t in targets)
        ngo = sum(len(q) for q, _ in go)
        losses,masked={},{}
        for (suffix,h),own_indices in zip(heads,votes):
            values={}
            values["vfl"],masked[suffix]=self.classification(h,targets,own_indices,max(ngt,1))
            values.update(self.regression(h,targets,go,max(ngo,1)))
            losses.update({"loss_"+k+suffix:v*self.weights[k] for k,v in values.items()})
        dn_denom = None
        if "dn_outputs" in output:
            meta = output["dn_meta"]; groups = meta["dn_num_group"]
            dn = [(q, torch.arange(len(t["labels"]), device=q.device).tile((groups,)))
                  for q, t in zip(meta["dn_positive_idx"], targets)]
            dn_denom = max(ngt * groups, 1)
            for index, h in enumerate(output["dn_outputs"] + [output["dn_pre_outputs"]]):
                suffix = f"_dn_{index}" if index < 3 else "_dn_pre"
                values = {}
                values["vfl"], _ = self.classification(h, targets, dn, dn_denom, ignore=False)
                values.update(self.regression(h,targets,dn,dn_denom))
                losses.update({"loss_"+k+suffix: v*self.weights[k] for k, v in values.items()})
        for name, value in losses.items():
            tensor("loss/"+name,value)
            require_finite(value, name)
        self.last_diagnostics = {"gt_count": ngt, "go_count": ngo, "classification_denominator": max(ngt, 1),
                                 "regression_denominator": max(ngo, 1), "dn_denominator": dn_denom,
                                 "ignore_negative_components": masked,
                                 "loss_count":len(losses),"go_pairs":[{"query":q.tolist(),"gt":g.tolist()} for q,g in go]}
        return losses
