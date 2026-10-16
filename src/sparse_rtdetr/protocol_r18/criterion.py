"""Delegate native matching/box losses/weights; mask only raw-ignore negatives."""
import copy
import torch
from torch.nn import functional as F
from ..localization.ignore import negative_keep_mask
from ..localization import consumer_geometry as geo

def adapt(base):
    Native=type(base)
    class CommonCriterion(Native):
        def recipe(self):
            return {'weights':dict(self.weight_dict),'alpha':self.alpha,'gamma':self.gamma,
                    'losses':list(self.losses),'share_matched_indices':self.share_matched_indices,
                    'boxes_weight_format':self.boxes_weight_format,
                    'matcher':{'cost_class':self.matcher.cost_class,'cost_bbox':self.matcher.cost_bbox,'cost_giou':self.matcher.cost_giou,
                               'alpha':self.matcher.alpha,'gamma':self.matcher.gamma,'use_focal_loss':self.matcher.use_focal_loss},
                    'ignore':'raw_unmatched_negatives_normal_aux_encoder_only; native_DN_unchanged',
                    'normalization':'native_GT_count_per_physical_batch; DN_times_groups; empty_min1',
                    'loss_weight_application':'native_criterion_once; engine_sum_all'}
        def forward(self,outputs,targets,**kwargs):
            self._dn_ids={id(x) for x in outputs.get('dn_aux_outputs',[])}
            self._families={id(outputs):'final',**{id(x):f'aux{i}' for i,x in enumerate(outputs.get('aux_outputs',[]))},
                **{id(x):f'enc{i}' for i,x in enumerate(outputs.get('enc_aux_outputs',[]))},
                **{id(x):f'dn{i}' for i,x in enumerate(outputs.get('dn_aux_outputs',[]))}}
            self.last_diagnostics=[]
            try:
                with geo.matching(self.matcher):return super().forward(outputs,targets,**kwargs)
            finally:self._dn_ids=set();self._families={}
        def get_loss(self,loss,outputs,targets,indices,num_boxes,**kwargs):
            if loss=='vfl':
                self.last_diagnostics.append({'head':self._families.get(id(outputs),'final'),'GT_count':sum(len(t['labels']) for t in targets),
                    'denominator':num_boxes,'matched_queries':[x.tolist() for x,y in indices],
                    'matched_GT_ids':[[t['gt_ids'][int(j)] for j in y] for t,(x,y) in zip(targets,indices)]})
            geo.positive_box('native/consumer_boxes',outputs['pred_boxes'])
            geo.tensor('native/consumer_logits',outputs['pred_logits'])
            return super().get_loss(loss,outputs,targets,indices,num_boxes,**kwargs)
        def loss_labels_vfl(self,outputs,targets,indices,num_boxes,values=None):
            if id(outputs) in getattr(self,'_dn_ids',set()):
                return super().loss_labels_vfl(outputs,targets,indices,num_boxes,values)
            keep=negative_keep_mask(outputs['pred_boxes'],indices,targets,num_classes=self.num_classes)
            if bool(keep.all()):return super().loss_labels_vfl(outputs,targets,indices,num_boxes,values)
            module=__import__(Native.__module__,fromlist=['box_iou'])
            idx=self._get_src_permutation_idx(indices)
            if values is None:
                gt=torch.cat([t['boxes'][j] for t,(_,j) in zip(targets,indices)])
                iou,_=geo.overlap('native/vfl_iou',module.box_iou,module.box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),module.box_cxcywh_to_xyxy(gt))
                values=iou.diag().detach()
            logits=outputs['pred_logits']
            classes=torch.full(logits.shape[:2],self.num_classes,dtype=torch.int64,device=logits.device)
            classes[idx]=torch.cat([t['labels'][j] for t,(_,j) in zip(targets,indices)])
            target=F.one_hot(classes,num_classes=self.num_classes+1)[...,:-1]
            quality=torch.zeros_like(classes,dtype=logits.dtype);quality[idx]=values.to(quality.dtype)
            score=quality.unsqueeze(-1)*target
            weight=self.alpha*logits.sigmoid().detach().pow(self.gamma)*(1-target)+score
            loss=F.binary_cross_entropy_with_logits(logits,score,weight=weight,reduction='none')*keep
            return {'loss_vfl':loss.mean(1).sum()*logits.shape[1]/num_boxes}
        def loss_boxes(self,outputs,targets,indices,num_boxes,boxes_weight=None):
            module=__import__(Native.__module__,fromlist=['generalized_box_iou'])
            idx=self._get_src_permutation_idx(indices)
            gt=torch.cat([t['boxes'][j] for t,(_,j) in zip(targets,indices)])
            # Admission duplicate only. The original native loss performs its original arithmetic.
            with torch.no_grad():geo.overlap('native/giou',module.generalized_box_iou,module.box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),module.box_cxcywh_to_xyxy(gt),True)
            return super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)
    result=CommonCriterion(base.matcher,base.weight_dict,base.losses,alpha=base.alpha,gamma=base.gamma,
                          num_classes=base.num_classes,boxes_weight_format=base.boxes_weight_format,share_matched_indices=base.share_matched_indices)
    result.last_diagnostics=[]
    return result
