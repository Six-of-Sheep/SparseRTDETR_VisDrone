"""Observe certified N e120 EMA without replacing its forward computation."""
import argparse,contextlib,hashlib,io,json,os,signal,sys,time
from pathlib import Path
from r18_readout_cli import ROOT,DOC,N_SHA,MEMBER_SHA,sha,read,write,need,digest,torch_setup
sys.path.insert(0,str(ROOT/'src'))


def state_digest(model):
    h=hashlib.sha256()
    rows=list(model.state_dict().items())
    for i in model.encoder.use_encoder_idx:rows.append(('unregistered_pos'+str(i),getattr(model.encoder,'pos_embed'+str(i))))
    for name,value in rows:
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def capture(model):
    import torch
    d=model.decoder.decoder;seen={'layers':[]};handles=[]
    need(not model.training and d.eval_idx==2 and len(d.layers)==3,'eval three-layer decoder required')
    def initial(module,args,kwargs):
        need('r' not in seen,'duplicate decoder call')
        value=args[1] if len(args)>1 else kwargs['ref_points_unact']
        mask=kwargs.get('attn_mask',args[7] if len(args)>7 else None)
        need(mask is None and value.shape[1:]==(300,4),'ordinary query identity/DN')
        seen['r']=value.detach().sigmoid().clone()
    handles.append(d.register_forward_pre_hook(initial,with_kwargs=True))
    def layer(i):
        def hook(module,args,output):
            need(i==len(seen['layers']),'wrong layer order')
            need(output.shape[1:]==(300,256) and not output.requires_grad,'wrong query layout/gradient')
            if i==0:need(torch.equal(args[1].squeeze(2),seen['r']),'initial reference differs')
            seen['layers'].append(output.detach().clone())
        return hook
    for i,l in enumerate(d.layers):handles.append(l.register_forward_hook(layer(i)))
    try:yield seen
    finally:
        for h in handles:h.remove()


def construct(synthetic,checkpoint=None):
    import torch
    from sparse_rtdetr.protocol_r18.model import Config,native_objects,vendor
    config=Config(input_hw=(128,128),physical_batch_size=4,amp_dtype='float32') if synthetic else Config()
    model,criterion,_,_,_=native_objects(config)
    vendor._configure_model_sampling(model,config)
    if not synthetic:
        need(sha(checkpoint)==N_SHA,'N checkpoint SHA')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(payload['format']=='sparse_rtdetr.sampling_span_fullstate_v1','N format')
        need(payload['binding']['arm']=='r18-sampling-span-v1/B0','N B0 identity')
        need(payload['engine']['completed_updates']==36480 and payload['engine']['completed_epochs']==120,'N endpoint')
        need(payload['ema']['updates']==36480,'EMA endpoint')
        model.load_state_dict(payload['ema']['module'],strict=True)
    from sparse_rtdetr.protocol_r18.geometry import validate_geometry
    validate_geometry(model)
    model.eval().requires_grad_(False)
    need(all(not m.training for m in model.modules()),'eval mode')
    need(criterion.matcher.cost_class==2 and criterion.matcher.cost_bbox==5 and criterion.matcher.cost_giou==2 and criterion.matcher.alpha==.25 and criterion.matcher.gamma==2 and criterion.matcher.use_focal_loss,'matcher recipe')
    return model,criterion.matcher,config


def pack(model,matcher,pixels,targets):
    import torch
    from sparse_rtdetr.localization.consumer_geometry import matching
    from sparse_rtdetr.localization.ignore import negative_keep_mask
    with torch.no_grad(),capture(model) as observed:
        outputs=model(pixels)
    need(len(observed['layers'])==3,'missing layer')
    need(outputs['pred_boxes'].shape[1:]==(300,4) and outputs['pred_logits'].shape[1:]==(300,10),'final query order')
    need(torch.isfinite(outputs['pred_boxes']).all() and torch.isfinite(outputs['pred_logits']).all(),'nonfinite outputs')
    with torch.no_grad(),matching(matcher):indices=matcher(outputs,targets)['indices']
    # Same training rule: ignore affects unmatched negative classification only.
    # No class loss is optimized here; matched positives are never removed.
    ignored=negative_keep_mask(outputs['pred_boxes'],indices,targets,num_classes=10)
    b=len(targets);q=300
    gt=torch.tensor([.5,.5,1.,1.]).repeat(b,q,1)
    mask=torch.zeros(b,q,dtype=torch.bool);small=mask.clone()
    gid=torch.full((b,q),-1,dtype=torch.int64);cat=gid.clone();tables=[]
    for i,(src,tgt) in enumerate(indices):
        src=src.cpu();tgt=tgt.cpu();target=targets[i]
        need(len(src)==len(src.unique()) and len(tgt)==len(tgt.unique()),'one-to-one matching')
        gt[i,src]=target['boxes'][tgt].cpu();mask[i,src]=True
        need(bool(ignored[i,src.to(ignored.device),target['labels'][tgt].to(ignored.device)].all()),'raw-ignore removed matched positive')
        # Original annotation area, before clipping/resizing; inclusive COCO small.
        small[i,src]=((target['raw_area'][tgt]>0)&(target['raw_area'][tgt]<=1024)).cpu()
        gid[i,src]=tgt;cat[i,src]=target['labels'][tgt].cpu()
        tables.append(list(target['gt_ids']))
    return dict(q=torch.stack(observed['layers'],1).float().cpu(),r=observed['r'].float().cpu(),gt=gt,mask=mask,small=small,gt_ids=gid,category=cat,
                GT_identity_table=tables),outputs,{'unmatched_negative_ignore_count':int((~ignored).sum()),'matched':int(mask.sum())}


def synthetic_sample(index):
    from PIL import Image
    from sparse_rtdetr.localization.data import RawAugmentation,parse_raw
    # Native parser/resize, not a substitute target pipeline. No file I/O.
    raw=b'10,11,16,16,1,1,0,0\n40,30,32,32,1,2,0,0\n0,0,100,80,0,0,0,0\n70,50,10,10,0,3,0,0\n20,20,5,5,1,11,0,0\n'
    image=Image.new('RGB',(160,96),(index%255,40,80))
    parsed=parse_raw(raw,'SYNTHETIC-%03d'%index)
    pixels,target,trace=RawAugmentation((128,128))(image,parsed,epoch=120,seed=0,position=index,image_id='SYNTHETIC-%03d'%index,training=False)
    need(len(target['gt_ids'])==2 and target['raw_area'].tolist()==[256.,1024.],'raw parsing/small boundary')
    import torch
    expected=torch.tensor([[18/160,19/96,16/160,16/96],[56/160,46/96,32/160,32/96]])
    need(torch.allclose(target['boxes'],expected,atol=2e-7,rtol=0) and target['labels'].tolist()==[0,1],'coordinate/category conversion')
    need(len(target['ignore_ids'])==2,'ignore/category11 semantics')
    need(trace['geometry_steps']==[],'unexpected augmentation')
    return pixels,target


def real_samples(c):
    from PIL import Image
    from sparse_rtdetr.localization.data import RawRoleDataset,parse_raw
    roots=c['roots'];need(roots['role']=='train_core','role')
    d=RawRoleDataset(role='train_core',manifest_path=c['members'],images_path=roots['images'],annotations_path=roots['annotations'],root_binding=roots)
    for i,id in enumerate(c['ids']):
        row=d.by_id[id];ib,ab=d.bound_bytes(row)
        with Image.open(io.BytesIO(ib)) as im:image=im.convert('RGB')
        need((image.width,image.height)==(row['width'],row['height']),'image geometry')
        raw=parse_raw(ab,id,expected_rows=row['raw_annotation_rows'])
        pixels,target,trace=d.transform(image,raw,epoch=120,seed=0,position=i,image_id=id,training=False)
        need(trace['geometry_steps']==[],'unexpected augmentation')
        yield pixels,target


def execute(c,out,authorization=None):
    out=Path(out);out.mkdir();start=time.monotonic();used=0
    synthetic=c['mode']=='SYNTHETIC_NOT_SCIENTIFIC';identity=digest(c)
    def guard():
        need(time.monotonic()-start<c['wall_seconds'],'extraction wall budget')
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=512<<20,'task storage')
        if not synthetic:
            need(time.monotonic()-start<c['gpu_seconds'],'GPU stage budget')
            need(torch.cuda.max_memory_allocated()<=18<<30,'allocated cap')
    try:
        write(out/'CONSUMED.json',{'contract':identity,'no_retry':True})
        need(c['semantics']=='pr25-native-N-EMA-layer012-initial-v1','extraction semantics')
        need(c['coordinate']=='normalized_canvas_cxcywh' and c['layer_order']==[0,1,2],'wrong coordinate/layer')
        need(0<c['forward_cap']<=48 and c['wall_seconds']<=3600 and c['gpu_seconds']<=1800,'resource contract')
        if synthetic:
            need(c['ids']==['SYNTHETIC-%03d'%i for i in range(24)],'synthetic IDs')
        else:
            need(c['mode']=='REAL_N_EMA','mode')
            from r18_deploy import verify_release
            verify_release(ROOT,c['release_manifest_sha256'])
            need(authorization=={'chain_contract':identity,'authorize_once':True,'scope':'PR25_FULL_FROZEN_EXPLORATION'},'full-chain authorization')
            s=read(DOC/'SELECTED_IDS.json');need(c['ids']==s['fit']+s['probe'],'wrong real ID/order')
            need(c['forward_cap']==48 and sha(c['members'])==MEMBER_SHA,'real budget/members')
            need(sha(c['checkpoint'])==N_SHA,'checkpoint')
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('extraction wall timeout')))
        signal.alarm(int(c['wall_seconds'] if synthetic else min(c['wall_seconds'],c['gpu_seconds'])))
        torch=torch_setup(synthetic)
        model,matcher,config=construct(synthetic,c.get('checkpoint'));device='cpu' if synthetic else 'cuda'
        model.to(device);before=state_digest(model)
        iterator=(synthetic_sample(i) for i in range(24)) if synthetic else real_samples(c)
        chunks=[];stats=[]
        for start_index in range(0,len(c['ids']),4):
            items=[next(iterator) for _ in range(4)];pixels=torch.stack([x[0] for x in items]).to(device)
            targets=[{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in x[1].items()} for x in items]
            guard();need(used<c['forward_cap'],'detector forward budget')
            used+=1
            tmp=out/'ledger.tmp'
            with tmp.open('w') as f:json.dump({'detector_forward':used},f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,out/'ledger.json')
            chunk,outputs,info=pack(model,matcher,pixels,targets);chunks.append(chunk);stats.append(info);guard()
        need(state_digest(model)==before,'parameters/buffers changed')
        need(all(p.grad is None and not p.requires_grad for p in model.parameters()),'detector gradient')
        data={k:torch.cat([v[k] for v in chunks]) for k in ('q','r','gt','mask','small','gt_ids','category')}
        data['ids']=c['ids'];data['GT_identity_table']=[r for chunk in chunks for r in chunk['GT_identity_table']]
        data['query_ids']=torch.arange(300);data['layer_order']=[0,1,2];data['coordinate']=c['coordinate']
        torch.save(data,out/'features.pt');guard()
        elapsed=time.monotonic()-start
        receipt={'status':'CLOSED','mode':c['mode'],'semantics':c['semantics'],'N_checkpoint_sha256':N_SHA if not synthetic else 'SYNTHETIC_RANDOM_NATIVE_NO_REAL_WEIGHTS',
                 'ids_sha256':digest({'fit':c['ids'][:16 if synthetic else 128],'probe':c['ids'][16 if synthetic else 128:]}),
                 'detector_forward':used,'features_sha256':sha(out/'features.pt'),'frozen_parameters':True,'fixed_final_Hungarian':True,'ordinary_only':True,
                 'gpu_seconds':0 if synthetic else elapsed,'wall_seconds':elapsed,'parameters_and_buffers_unchanged':True,'stats':stats,
                 'real_weight_forward_checked':not synthetic,'real_weight_capture_on_off_comparison':False,'CPU_constructed_behavior_is_not_real_weight_evidence':True}
        write(out/'CLOSED.json',receipt);return receipt
    except Exception as e:
        write(out/'STOP.json',{'error':repr(e),'detector_forward':used,'seconds':time.monotonic()-start,'no_retry':True});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--out',required=True);p.add_argument('--authorization')
    a=p.parse_args();execute(read(a.contract),a.out,read(a.authorization) if a.authorization else None)
