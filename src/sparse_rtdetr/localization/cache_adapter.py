"""Import authenticated serialized non-learning geometry, keep original expected layout.

No saved object is mutated and no comparison is relaxed. Preflight both sides
completely before writing any fresh buffer; original strict restore follows.
"""
import torch
from sparse_rtdetr.localization import checkpoint
from sparse_rtdetr.localization.runio import require,digest

ALLOW={'decoder:persistent_cache:anchors':'decoder.anchors',
       'decoder:persistent_cache:valid_mask':'decoder.valid_mask',
       **{f'decoder.decoder.layers.{i}.cross_attn:persistent_cache:num_points_scale':f'decoder.decoder.layers.{i}.cross_attn.num_points_scale' for i in range(3)}}

def install(engine,payload):
    planned=[];ledgers=[]
    for side,module in (('model',engine.model),('ema',engine.ema.module)):
        saved=payload['model_layout' if side=='model' else 'ema_layout'];current=checkpoint.old._model_layout(module)
        state=payload['model'] if side=='model' else payload['ema']['module']
        for field in ('modules','state','parameters'):require(saved[field]==current[field],'non-cache layout difference: '+side+'/'+field)
        require(set(saved['caches'])==set(current['caches']),'cache inventory differs: '+side)
        checkpoint.old._check_model_state(state,saved,side)
        parameters={x['name'] for x in saved['parameters']}
        for key in saved['caches']:
            old,new=saved['caches'][key],current['caches'][key]
            if old==new:continue
            require(key in ALLOW,'non-whitelisted cache difference: '+side+'/'+key)
            name=ALLOW[key];require(old is not None and new is not None and old['shape']==new['shape'] and old['dtype']==new['dtype'],'cache shape/dtype difference: '+name)
            require(set(old)==set(new)=={'shape','dtype','sha256'},'cache spec schema differs')
            require(name in state and name not in parameters,'missing original cache bytes or learned parameter: '+name)
            require(checkpoint.old._tensor_spec(state[name],digest=True)==old,'original cache bytes not original saved SHA')
            owner_name,leaf=name.rsplit('.',1);owner=module.get_submodule(owner_name)
            require(leaf in owner._buffers and leaf not in owner._non_persistent_buffers_set and leaf not in owner._parameters,'cache registration is not non-learning persistent buffer')
            value=owner._buffers[leaf];require(not value.requires_grad and not state[name].requires_grad,'geometry must not be learned')
            require(checkpoint.old._tensor_spec(value)=={k:v for k,v in old.items() if k!='sha256'},'fresh buffer shape/dtype differs')
            planned.append((side,name,value,state[name]));ledgers.append({'component':side,'state_key':name,'cache_key':key,'before':new,'original':old})
    # Original online and EMA immutable geometry must agree too.
    for key,name in ALLOW.items():
        if name in payload['model'] or name in payload['ema']['module']:
            require(name in payload['model'] and name in payload['ema']['module'] and checkpoint.old._equal(payload['model'][name],payload['ema']['module'][name]),'original model/EMA geometry differs: '+name)
    original_layout_digest=digest({'model':payload['model_layout'],'ema':payload['ema_layout']})
    with torch.no_grad():
        for side,name,target,original in planned:target.copy_(original.to(device=target.device))
    require(checkpoint.old._model_layout(engine.model)==payload['model_layout'],'model layout still differs after exact cache import')
    require(checkpoint.old._model_layout(engine.ema.module)==payload['ema_layout'],'EMA layout still differs after exact cache import')
    require(digest({'model':payload['model_layout'],'ema':payload['ema_layout']})==original_layout_digest,'saved expected layout changed')
    return {'status':'EXACT_ORIGINAL_CACHE_IMPORT_PASS','changed':ledgers,'mutated_saved_payload':False,'post_layout_exact':True,'full_restore_performed':False}

def strict_restore(engine,binding,path,expected_sha,ledger_path=None):
    from sparse_rtdetr.localization.runio import publish
    try:
        payload=checkpoint.read(path,expected_sha,binding)
        ledger=install(engine,payload)
        if ledger_path is not None:publish(ledger_path,ledger)
        # No modification of old payload/binding/layout/hash/expected_sha.
        result=checkpoint.restore(engine,binding,path,expected_sha)
        return {**result,'cache_import':ledger}
    except BaseException:
        engine.failed=True
        raise
