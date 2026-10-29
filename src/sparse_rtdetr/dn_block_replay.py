"""CPU research prototype: no detector, optimizer, data reader or execution launcher."""
import torch

EPS = 1e-5
TOL = 2e-5

def valid_boxes(b):
    if b.ndim != 2 or b.shape[1] != 4 or not b.is_floating_point():
        raise ValueError("boxes must be floating Nx4 cxcywh")
    if not torch.isfinite(b).all() or (b[:, 2:] <= 0).any():
        raise ValueError("nonfinite or nonpositive box")

def encode(pred, gt):
    valid_boxes(pred); valid_boxes(gt)
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch")
    return torch.cat(((pred[:, :2]-gt[:, :2])/gt[:, 2:],
                      (pred[:, 2:]/gt[:, 2:]).log()), -1).detach()

def reconstruct(gt, error):
    valid_boxes(gt)
    if error.shape != gt.shape or not torch.isfinite(error).all():
        raise ValueError("bad error")
    b = torch.cat((gt[:, :2]+error[:, :2]*gt[:, 2:],
                   gt[:, 2:]*error[:, 2:].exp()), -1)
    valid_boxes(b)
    return b

def corners(b):
    return torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), -1)

def effective(gt, error):
    before = reconstruct(gt, error)
    xy = corners(before)
    clipped = xy.clamp(0, 1)
    b = torch.cat(((clipped[:, :2]+clipped[:, 2:])/2,
                   clipped[:, 2:]-clipped[:, :2]), -1)
    # Invalids are retained as evidence, never resampled, sorted or repaired.
    invalid = (b[:, 2:] <= 0).any(-1)
    unact = (b.clamp(min=EPS)/(1-b).clamp(min=EPS)).log()
    return before, b, unact.sigmoid(), unact, (xy != clipped).any(-1), invalid

def scale_group(gt, height=768, width=1344):
    area = gt[:, 2]*width*gt[:, 3]*height
    return (area >= 32**2).long()+(area >= 96**2).long()

def select(library, library_groups, recipient_groups, seed):
    if library.ndim != 2 or library.shape[1] != 4 or not torch.isfinite(library).all():
        raise ValueError("bad library")
    if library.device.type != "cpu" or library_groups.device.type != "cpu":
        raise ValueError("library must be CPU")
    if library_groups.shape != (len(library),) or library_groups.dtype != torch.int64:
        raise ValueError("bad library groups")
    if recipient_groups.ndim != 1 or recipient_groups.dtype != torch.int64:
        raise ValueError("bad recipient groups")
    if ((library_groups < 0)|(library_groups > 2)).any() or ((recipient_groups < 0)|(recipient_groups > 2)).any():
        raise ValueError("unknown group")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    donors = torch.empty(len(recipient_groups), dtype=torch.long)
    perm = torch.arange(len(recipient_groups))
    # Sampling with replacement; B is a nonzero cyclic shift of the sampled
    # size blocks inside each scale group, never a componentwise shuffle.
    for s in range(3):
        slots = (recipient_groups == s).nonzero().flatten()
        if not len(slots): continue
        pool = (library_groups == s).nonzero().flatten()
        if not len(pool): raise ValueError("empty donor stratum")
        donors[slots] = pool[torch.randint(len(pool), (len(slots),), generator=rng)]
        if len(slots) > 1:
            shift = int(torch.randint(1, len(slots), (1,), generator=rng))
            perm[slots] = slots.roll(shift)
    joint = library[donors].clone()
    block = joint.clone()
    block[:, 2:] = joint[perm, 2:]
    return joint, block, donors, perm

def control_audit(gt, joint, block, perm):
    a, b = effective(gt, joint), effective(gt, block)
    invalid = a[5] | b[5]
    out = {"count": len(gt), "clipped_J": int(a[4].sum()),
           "clipped_B": int(b[4].sum()), "invalid_J": int(a[5].sum()),
           "invalid_B": int(b[5].sum()),
           "pre_center_equal": torch.equal(joint[:, :2], block[:, :2]),
           "pre_size_block_equal": torch.equal(joint[perm, 2:], block[:, 2:])}
    for name, idx in [("clipped",1), ("injected",2)]:
        if invalid.any():
            out[name+"_max_block_error"] = None
        else:
            ea, eb = encode(a[idx],gt), encode(b[idx],gt)
            delta = torch.cat((ea[:, :2]-eb[:, :2],ea[perm, 2:]-eb[:, 2:]),-1)
            out[name+"_max_block_error"] = float(delta.abs().max()) if delta.numel() else 0.
    out["joint_only_admissible"] = bool(
        out["pre_center_equal"] and out["pre_size_block_equal"] and not invalid.any()
        and out["injected_max_block_error"] <= TOL)
    return out

def aligned_iou(a, b):
    aa,bb=corners(a),corners(b)
    intersection=(torch.minimum(aa[:,2:],bb[:,2:])-torch.maximum(aa[:,:2],bb[:,:2])).clamp(min=0).prod(-1)
    union=a[:,2:].prod(-1)+b[:,2:].prod(-1)-intersection
    return intersection/union

def distribution(x):
    if not len(x): return {"n":0}
    x=x.double()
    centered=x-x.mean(0)
    cov=centered.T@centered/max(1,len(x)-1)
    return {"n":len(x),"mean":x.mean(0).tolist(),
            "quantiles_0_25_50_75_100":torch.quantile(x,torch.tensor([0.,.25,.5,.75,1.],dtype=x.dtype),dim=0).tolist(),
            "cross_covariance":cov[:2,2:].tolist()}

def semantic_audit(gt, native_positive, native_negative, empirical):
    pi=aligned_iou(native_positive,gt)
    ni=aligned_iou(native_negative,gt)
    ei=aligned_iou(empirical,gt)
    rate=lambda mask: float(mask.double().mean()) if len(mask) else 0.
    result={"n":len(gt),"native_positive_below_negative":rate(pi<ni),
            "replay_positive_below_negative":rate(ei<ni),
            "native_positive_zero_iou":rate(pi==0),
            "replay_positive_zero_iou":rate(ei==0),
            "native_positive_iou":distribution(pi[:,None].repeat(1,4)),
            "replay_positive_iou":distribution(ei[:,None].repeat(1,4))}
    result["admissible"]=(result["replay_positive_below_negative"]-result["native_positive_below_negative"] <= .05
                         and result["replay_positive_zero_iou"]-result["native_positive_zero_iou"] <= .02)
    return result

class Replay:
    """Optional real-DN-entry callback; no train-loop integration in this prototype."""
    def __init__(self, library, groups, arm, seed):
        if arm not in ("J","B"): raise ValueError("unknown arm")
        self.library=library.detach().cpu().clone()
        self.groups=groups.detach().cpu().clone()
        self.arm,self.seed=arm,seed
        self.audit=None

    @torch.no_grad()
    def __call__(self, native_unact, gt, positive_mask):
        idx=positive_mask.bool().nonzero()
        flat_gt=gt[idx[:,0],idx[:,1]].detach()
        j,b,donors,perm=select(self.library,self.groups,scale_group(flat_gt).cpu(),self.seed)
        j,b=j.to(flat_gt),b.to(flat_gt)
        self.audit=control_audit(flat_gt,j,b,perm.to(flat_gt.device))
        self.audit["pre_J"]=distribution(j)
        self.audit["pre_B"]=distribution(b)
        for name,error in (("J",j),("B",b)):
            v=effective(flat_gt,error)
            self.audit["clipped_"+name+"_distribution"]=None if v[5].any() else distribution(encode(v[1],flat_gt))
            self.audit["post_"+name]=None if v[5].any() else distribution(encode(v[2],flat_gt))
        self.audit["donors"]=donors.tolist()
        self.audit["size_permutation"]=perm.tolist()
        # Keep both raw and effective audit tensors for synthetic inspection.
        self.observation={"gt":flat_gt.cpu(), "J":j.cpu(), "B":b.cpu(),
                          "native":native_unact[idx[:,0],idx[:,1]].sigmoid().cpu()}
        if not self.audit["joint_only_admissible"]:
            raise ValueError("post-injection block control failed; no filtering/retry")
        chosen=j if self.arm=="J" else b
        out=native_unact.clone()
        out[idx[:,0],idx[:,1]]=effective(flat_gt,chosen)[3]
        return out

def dn_entry(native_fn, targets, num_classes, num_queries, class_embed,
             num_denoising=100, label_noise_ratio=.5, box_noise_scale=1., replay=None):
    """Call the unmodified production DN generator, then replace its positive coordinates."""
    if replay is not None and box_noise_scale <= 0:
        raise ValueError("positive native scale required")
    out = native_fn(targets, num_classes, num_queries, class_embed,
                    num_denoising, label_noise_ratio, box_noise_scale)
    if replay is None or out[3] is None:
        return out
    logits, unact, mask, meta = out
    gt = torch.zeros_like(unact)
    positive = torch.zeros(unact.shape[:2], dtype=torch.bool, device=unact.device)
    for i, target in enumerate(targets):
        ids = meta["dn_positive_idx"][i]
        gt[i, ids] = target["boxes"].repeat(meta["dn_num_group"], 1)
        positive[i, ids] = True
    # Frozen proposal: replace ceil(P/2) valid positives in lexicographic
    # (batch row, DN slot) order; no dependence on GT size, error or IoU.
    valid_indices=positive.nonzero()
    positive.zero_()
    chosen=valid_indices[::2]
    positive[chosen[:,0],chosen[:,1]]=True
    injected=replay(unact, gt, positive)
    replay.audit["all_valid_positives"]=len(valid_indices)
    replay.audit["replaced_positives"]=len(chosen)
    replay.audit["untouched_positives"]=len(valid_indices)-len(chosen)
    replay.audit["valid_negatives"]=len(valid_indices)
    replay.audit["padding_slots"]=positive.numel()-2*len(valid_indices)
    max_gt=max(len(t["boxes"]) for t in targets)
    idx=positive.nonzero()
    pos=unact[idx[:,0],idx[:,1]].sigmoid()
    neg=unact[idx[:,0],idx[:,1]+max_gt].sigmoid()
    changed=injected[idx[:,0],idx[:,1]].sigmoid()
    replay.audit["positive_negative"]=semantic_audit(gt[idx[:,0],idx[:,1]],pos,neg,changed)
    if not replay.audit["positive_negative"]["admissible"]:
        raise ValueError("positive-negative semantic guard failed; no filtering/retry")
    return logits, injected, mask, meta
