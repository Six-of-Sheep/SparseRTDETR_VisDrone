"""Bounded bad-frame evidence; invoked only after original geometry gate fails."""
import torch
from .operators import weighting_function

def box_evidence(logits,reference,valid,p,x,edges,box):
 bad=(box[...,2:]<=0)
 if valid is not None:bad=bad & valid[...,None]
 indices=torch.nonzero(bad.any(-1),as_tuple=False);records=[]
 z=logits.float().reshape_as(p)
 w=torch.as_tensor(weighting_function(),device=p.device,dtype=torch.float64)
 for idx in indices[:16]:
  b,q=[int(i) for i in idx];ref=reference[b,q].detach().double();means=(p[b,q].detach().double()*w).sum(-1);sums=p[b,q].detach().double().sum(-1)
  exact=ref[:2]*(sums[2:]-sums[:2])+ref[2:]*(.5*(sums[:2]+sums[2:])+(means[:2]+means[2:])/4)
  records.append({'sample':b,'query':q,'box_fp32':box[b,q].detach().cpu().tolist(),
    'edges_fp64':edges[b,q].detach().cpu().tolist(),'edge_difference_wh_fp64':(edges[b,q,2:]-edges[b,q,:2]).detach().cpu().tolist(),
    'decode_reference':ref.cpu().tolist(),'mean_W':means.cpu().tolist(),
    'idealized_unit_mass_formula_wh':(ref[2:]*(1+(means[:2]+means[2:])/4)).cpu().tolist(),
    'exact_sum_corrected_formula_wh':exact.cpu().tolist(),
    'sum_corrected_formula_vs_edges_residual':(exact-(edges[b,q,2:]-edges[b,q,:2]).detach()).cpu().tolist(),
    'normalized_probabilities_counterfactual_wh':(ref[2:]*(1+(means[:2]/sums[:2]+means[2:]/sums[2:])/4)).cpu().tolist(),
    'logits':z[b,q].detach().cpu().tolist(),'probabilities':p[b,q].detach().cpu().tolist(),
    'probability_sums':p[b,q].detach().double().sum(-1).cpu().tolist(),
    'support_endpoints':x[b,q].detach()[...,[0,-1]].cpu().tolist(),'valid':True})
 return {'kind':'FINITE_DECODED_BOX_BELOW_FROZEN_GEOMETRY_BOUND','frozen_threshold':'consumer_geometry_contract_v1:valid_wh>0',
         'bad_queries':int(indices.shape[0]),'stored_queries':len(records),'limit':16,'records':records}

def centered_box_evidence(logits,reference,valid,p):
 from .distribution import NUMERICAL_IMPLEMENTATION
 r=reference.detach().double();p64=p.detach().double();mass=p64.sum(-1)
 w=torch.as_tensor(weighting_function(),device=p.device,dtype=torch.float64)
 relative=(p64*(.5+w/4.)).sum(-1)/mass
 delta=r.new_tensor([-1.,-1.,1.,1.])*r[...,[2,3,2,3]]*relative
 center=r[...,:2]+(delta[...,:2]+delta[...,2:])/2.;wh=delta[...,2:]-delta[...,:2]
 box=torch.cat((center,wh),-1).float();bad=(box[...,2:]<=0)
 if valid is not None:bad=bad & valid[...,None]
 indices=torch.nonzero(bad.any(-1),as_tuple=False);z=logits.detach().float().reshape_as(p);records=[]
 for idx in indices[:16]:
  b,q=[int(i) for i in idx]
  records.append({'sample':b,'query':q,'box_fp32':box[b,q].cpu().tolist(),'width_height_fp64':wh[b,q].cpu().tolist(),
    'decode_reference':r[b,q].cpu().tolist(),'probability_sums':mass[b,q].cpu().tolist(),'relative_displacements':relative[b,q].cpu().tolist(),
    'signed_displacements_fp64':delta[b,q].cpu().tolist(),'logits':z[b,q].cpu().tolist(),'probabilities':p[b,q].detach().cpu().tolist(),'valid':True})
 return {'kind':'FINITE_DECODED_BOX_BELOW_FROZEN_GEOMETRY_BOUND','frozen_threshold':'consumer_geometry_contract_v1:valid_wh>0','bad_queries':int(indices.shape[0]),
         'stored_queries':len(records),'limit':16,'numerical_implementation':NUMERICAL_IMPLEMENTATION,'records':records}

def label_evidence(error,layer,attention_reference,targets,normal_count,dn_meta):
 evidence=getattr(error,'evidence',None)
 if not isinstance(evidence,dict) or evidence.get('kind')!='FINITE_DECODED_BOX_BELOW_FROZEN_GEOMETRY_BOUND':return
 dn=0 if dn_meta is None else int(dn_meta['dn_num_split'][0]);maxgt=max((len(t['labels']) for t in targets),default=0) if targets else 0
 evidence.update(decoder_layer=layer,dn_queries=dn,normal_queries=normal_count,max_gt=maxgt)
 for r in evidence['records']:
  b,q=r['sample'],r['query'];r['attention_reference']=attention_reference[b,q].detach().cpu().tolist()
  r['image_id']=targets[b].get('image_id') if targets else None
  r['query_kind']='DN' if q<dn else 'NORMAL';r['query_local']=q if q<dn else q-dn
  if q<dn:
   slot=q%(2*maxgt);gt_slot=slot%maxgt
   r.update(dn_group=q//(2*maxgt),dn_positive=slot<maxgt,dn_gt_slot=gt_slot,
            dn_gt_id=targets[b]['gt_ids'][gt_slot] if gt_slot<len(targets[b]['gt_ids']) else None)
