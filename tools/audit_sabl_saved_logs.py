from pathlib import Path
import hashlib,json,time,math,resource,signal
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace');T=WS/'experiments/localization-distribution/r18-sabl-s-path-audit-20261028-001';F=WS/'experiments/localization-distribution/r18-sabl-formal24-20261027-001/attempt-formal-001/train-S';resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.alarm(300);start=time.monotonic();files=[];count=0;micro_count=0;finite_count=0;den_ok=0;bad=[];extrema={k:[float('inf'),float('-inf')] for k in ('loss_bucket','loss_offset','gradient_norm_before_clip')};first=None;last=None;missing={'output_tensor_snapshots':0,'gradient_tensor_SHA_present':0,'boundary_edges_or_logits':0};ordered_updates=[]
for name in ('engineering-events.jsonl','remaining-events.jsonl'):
 p=F/name;before=p.stat();h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for line in f:
   h.update(line);v=json.loads(line);r=v['result'];count+=1;n+=1;ordered_updates.append(r['update']);finite_count+=r['finite'] is True
   if not math.isfinite(r['gradient_norm_before_clip']):bad.append(['grad_nonfinite',r['update']])
   extrema['gradient_norm_before_clip'][0]=min(extrema['gradient_norm_before_clip'][0],r['gradient_norm_before_clip']);extrema['gradient_norm_before_clip'][1]=max(extrema['gradient_norm_before_clip'][1],r['gradient_norm_before_clip'])
   snap={'update':r['update'],'epoch':r['epoch'],'loss_bucket':[x['loss_bucket'] for x in r['microbatch_losses']],'loss_offset':[x['loss_offset'] for x in r['microbatch_losses']],'gradient_norm_before_clip':r['gradient_norm_before_clip']}
   if first is None:first=snap
   last=snap
   for losses,diags in zip(r['microbatch_losses'],r['normalizers']):
    micro_count+=1;assert len(losses)==23 and 'loss_bucket' in losses and 'loss_offset' in losses
    final=[x for x in diags if x['head']=='final'];extra=[x for x in diags if x['head']=='SABL_ordinary_only'];assert len(final)==len(extra)==1
    m=sum(len(x) for x in final[0]['matched_queries']);matches=extra[0]['matches'];valid=m==matches and extra[0]['bucket_denominator']==(4*m if m else 1) and extra[0]['offset_denominator']==(m if m else 1);den_ok+=valid
    if not valid:bad.append(['normalizer',r['update']])
    for key in ('loss_bucket','loss_offset'):
     x=losses[key];assert math.isfinite(x);extrema[key][0]=min(extrema[key][0],x);extrema[key][1]=max(extrema[key][1],x)
   missing['output_tensor_snapshots']+=bool(r.get('output_sha256'))
   missing['gradient_tensor_SHA_present']+=r.get('gradient_sha256_before_clip') is not None
   missing['boundary_edges_or_logits']+=any(k in r for k in ('refined_edges','sabl_logits','sabl_offsets','pre_valid_edges'))
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);files.append({'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest(),'records':n,'size_mtime_unchanged':True})
assert ordered_updates==list(range(1,7297)) and not bad
summary={'status':'EXISTING_S_LOGS_READ_ONLY_REVIEWED','files':files,'complete_updates':count,'microbatch_loss_and_diagnostic_records':micro_count,'finite_flag_records':finite_count,'new_loss_names_present_23_total_losses':True,'ordinary_matches_and_4M_M_denominators_confirmed':den_ok,'first':first,'last':last,'minmax':extrema,'saved_field_counts':missing,'evidence_limit':'loss/grad finite or decreasing do not verify correct side semantics; logs lack pre-valid edges and per-query logits/offsets','CPU_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_forward_backward_training_AP_AR':0}
with (T/'EXISTING_LOG_REVIEW.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))