"""Use original ranking/score interfaces; remove invalid selected boxes without refill."""
import torch
from ..r0.evaluation import rank_candidates,score_budget
from .sabl_control import need

def rank_refined(logits,boxes,valid,row,budget):
    need(valid.shape==(300,) and valid.dtype==torch.bool,'invalid refined mask')
    if bool(valid.all()):return rank_candidates(logits,boxes,row,budget)
    # The legacy ranker rejects zero area. A private placeholder only carries ranks;
    # it is never a returned prediction, training target, or fallback detection.
    safe=boxes.clone();safe[~valid]=safe.new_tensor([.5,.5,1.,1.])
    ranked=rank_candidates(logits,safe,row,budget)
    return [p for p in ranked if bool(valid[p['query_id']])]
