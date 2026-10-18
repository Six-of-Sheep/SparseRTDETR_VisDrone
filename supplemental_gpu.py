"""Frozen A/B/C inference only. No backward or optimizer dispatch."""
import gc
import io
from pathlib import Path
import time
from supplemental_runtime import require,read,exact,reference,publish,ROOT,live

def restore(c,seed,out):
    import torch
    from sparse_rtdetr.localization import worker as base
    from sparse_rtdetr.protocol_r18.worker import make_engine
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.model import Config
    p=c['parents'][seed]
    for r in (p['binding'],p['endpoint'],p['checkpoint_sidecar']):require(reference(r['path'])==r,'endpoint/binding/sidecar changed')
    b=read(p['binding']['path']);side=read(p['checkpoint_sidecar']['path']);cp=p['checkpoint']
    require(side['sha256']==cp['sha256'] and side['completed_updates']==36480 and side['scope']==f'formal_R18_seed{seed}_e120','checkpoint endpoint scope')
    require(b['config']==Config(seed=seed).binding() and b['scope']==f'formal_R18_seed{seed}_e120','real seed model recipe differs')
    # Build only the unchanged native model, then strict original full restore.
    # Original binding is preserved; new execution lineage is a separate receipt.
    from supplemental_admission import adapt_resources
    require(p['construction']['environment']==c['environment'],'parent/resource environment conflict')
    cfg=adapt_resources({**c,**p['construction'],'seed':seed,'runtime_root':c['runtime_root']})
    runtime=base.prepare_cuda(cfg,out)
    engine,constructed,_=make_engine(cfg,runtime,f'formal_R18_seed{seed}_e120')
    require(constructed['config']==b['config'] and constructed['model']==b['model'],'model construction differs from certified endpoint')
    result=checkpoint.restore(engine,b,cp['path'],cp['sha256'])
    require(result['full_state_exact'] and engine.completed_updates==engine.ema.updates==36480,'strict own e120 restore')
    engine.ema.module.eval()
    publish(out/'RESTORE.json',{'status':'PASS','seed':seed,'parent_checkpoint':cp,'parent_binding':p['binding'],
        'new_source_sha256':c['source_sha256'],'new_contract_sha256':c['contract_sha256'],'strict_restore':result,
        'original_binding_not_rewritten':True,'optimizer_updates':0})
    return engine,b


def decode_merge(records, rows, mode):
    """Identical original FP32 decode and hard NMS, shared by CPU acceptance."""
    import torch
    from slice_core import tile_windows, map_crop_xyxy, select_view_candidates, merge_candidates
    from sparse_rtdetr.localization.consumer_geometry import corners, positive_box
    from infer_seed import prediction
    pools = [[] for _ in rows]
    for record in records:
        vid, logits, boxes = record['view_id'], record['pred_logits'], record['pred_boxes']
        require(logits.shape == (len(rows), 300, 10) and boxes.shape == (len(rows), 300, 4), 'dynamic final batch output shape')
        require(logits.dtype == boxes.dtype == torch.float32 and torch.isfinite(logits).all(), 'FP32 finite outputs')
        positive_box('supplemental/output', boxes)
        for j, row in enumerate(rows):
            tile = tile_windows(row['height'], row['width'])[vid - 1] if vid else None
            w, h = (tile.width, tile.height) if tile else (row['width'], row['height'])
            xy = torch.cat((boxes[j, :, :2] - boxes[j, :, 2:] / 2, boxes[j, :, :2] + boxes[j, :, 2:] / 2), -1)
            xy *= torch.tensor([w, h] * 2, dtype=torch.float32)
            corners('supplemental/decode', xy)
            mapped = map_crop_xyxy(xy.tolist(), tile) if tile else xy.tolist()
            pools[j].extend(select_view_candidates(mapped, logits[j].sigmoid().numpy(), view_id=vid))
    output = {a: {'600': [], '300': []} for a in ('ABC' if mode == 'full-C' else mode[-1])}
    for j, row in enumerate(rows):
        whole = pools[j][:600]
        for arm in output:
            if arm == 'A':
                merged = {'top600': whole, 'top300': whole[:300]}
            else:
                merged = merge_candidates(whole if arm == 'B' else pools[j])
            for budget in ('600', '300'):
                output[arm][budget].extend(prediction(row['stable_image_id'], p, whole=arm == 'A') for p in merged['top' + budget])
    return output


def run(c, seed, mode, out, emit):
    import torch
    from infer_seed import make_view_pixels, save_tensor
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.localization import worker as base
    from sparse_rtdetr.r0.evaluation_data import Development548
    from supplemental_data import dataset
    previous_event = base.event

    def cuda_event(path, kind, **kw):
        require(kind == 'cuda_initialized', 'unexpected restore event')
        emit.emit('cuda_initialized', **kw)

    base.event = cuda_event
    try:
        e, b = restore(c, seed, out)
    finally:
        base.event = previous_event
    emit.emit('restored', restore=reference(out / 'RESTORE.json'))
    parent = c['parents'][seed]
    full = mode == 'full-C'
    if full:
        data, rows, _ = dataset(c)
        work = rows
    else:
        data = Development548(parent['construction']['data']['development'])
        work = [data.by_id[i] for i in parent['fixed64']]
    expected_count = 1602 if full else 64
    require(len(work) == expected_count and len({r['stable_image_id'] for r in work}) == expected_count, 'inference member count')
    own_a = None
    if not full:
        ref = parent['A_raw']; require(reference(ref['path']) == ref, 'certified own A raw SHA')
        own_a = torch.load(ref['path'], map_location='cpu', weights_only=True)
        require(own_a['checkpoint_sha256'] == parent['checkpoint']['sha256'] and own_a['image_ids'] == [r['stable_image_id'] for r in data.records], 'own A identity')
        index = {r['stable_image_id']: i for i, r in enumerate(data.records)}
    model = e.ema.module
    before = checkpoint.capture(e, b)
    mutations = []
    for owner, name in ((torch.autograd, 'backward'), (e.optimizer, 'step'), (e.ema, 'update')):
        mutations.append((owner, name, getattr(owner, name)))
        def forbidden(*a, **kw):
            raise RuntimeError('detector mutation forbidden')
        setattr(owner, name, forbidden)
    sliced = mode in ('accept-C', 'full-C')
    predictions = {a: {'600': [], '300': []} for a in ('ABC' if full else mode[-1])}
    calls, raws, timings = 0, [], []

    def batch(br, number, save):
        nonlocal calls
        torch.cuda.synchronize(); start = time.perf_counter()
        views, _ = make_view_pixels(data, br, sliced)
        prep = time.perf_counter(); records = []
        for vid, tensors in enumerate(views):
            x = torch.stack(tensors).to('cuda:0')
            with torch.autocast('cuda', enabled=False), forward_guards(model):
                y = model(x)
            torch.cuda.synchronize(); calls += 1
            require(set(y) == {'pred_logits', 'pred_boxes'}, 'detector output fields')
            records.append({'view_id': vid, 'pred_logits': y['pred_logits'].detach().cpu(), 'pred_boxes': y['pred_boxes'].detach().cpu()})
            del x, y
        finals = decode_merge(records, br, mode)
        torch.cuda.synchronize(); end = time.perf_counter()
        if not full:
            ix = [index[r['stable_image_id']] for r in br]
            for key in ('pred_logits', 'pred_boxes'):
                require(torch.equal(records[0][key], own_a[key][ix]), 'certified whole branch differs')
        if save:
            if full:
                raws.append(save_tensor(out / 'raw' / f'batch-{number:03d}.pth', {'schema': 'supplemental-original-five-view-v1',
                    'task_id': c['task_id'], 'purpose': c['purpose'], 'seed': seed, 'source_sha256': c['source_sha256'],
                    'contract_sha256': c['contract_sha256'], 'checkpoint_sha256': parent['checkpoint']['sha256'],
                    'image_ids': [r['stable_image_id'] for r in br], 'views': records}))
            for arm in predictions:
                for budget in ('600', '300'):
                    predictions[arm][budget].extend(finals[arm][budget])
        return {'e2e_seconds': end - start, 'preprocess_seconds': prep - start}

    try:
        with torch.inference_mode():
            for k in range(2):
                live(c, out, True); batch(work[k * 4:k * 4 + 4], k, False)
            emit.emit('warmup', forward_calls=calls)
            torch.cuda.reset_peak_memory_stats()
            for k in range((len(work) + 3) // 4):
                live(c, out, True)
                br = work[k * 4:k * 4 + 4]
                timings.append(batch(br, k, True))
                emit.emit(f'batch-{k + 1:03d}', forward_calls=calls, images=min((k + 1) * 4, len(work)), actual_batch=len(br))
        require(checkpoint.exact_equal(before, checkpoint.capture(e, b)), 'model/optimizer/EMA/BN/RNG changed')
        require(all(p.grad is None for m in (e.model, e.ema.module) for p in m.parameters()), 'detector gradients not empty')
        emit.emit('state_exact', complete_state_RNG_exact=True)
        if mode == 'accept-C':
            old = exact(parent['accept_C_result'])
            for budget in ('600', '300'):
                require(predictions['C'][budget] == exact(old['predictions'][budget]), 'original certified C64 predictions differ')
        expected_calls = 2015 if full else 90 if sliced else 18
        require(calls == expected_calls, 'fixed forward budget')
        pred_refs = {}
        for arm in predictions:
            pred_refs[arm] = {}
            for budget in ('600', '300'):
                path = out / f'{arm}-top{budget}-predictions.json'
                publish(path, predictions[arm][budget]); pred_refs[arm][budget] = reference(path)
        result = {'status': 'PASS', 'seed': seed, 'mode': mode, 'image_ids': [r['stable_image_id'] for r in work],
                  'forward_calls': calls, 'predictions': pred_refs, 'raw_refs': raws, 'state_RNG_exact': True,
                  'e2e_seconds': sum(t['e2e_seconds'] for t in timings), 'timings': timings,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'detector_updates': 0, 'no_AP_on_fixed64': not full, 'last_batch': len(work[-(len(work) % 4 or 4):])}
        publish(out / 'RESULT.json', result); emit.emit('complete', result=reference(out / 'RESULT.json'))
    finally:
        for owner, name, value in mutations:
            setattr(owner, name, value)
        del e
        gc.collect(); torch.cuda.empty_cache()

