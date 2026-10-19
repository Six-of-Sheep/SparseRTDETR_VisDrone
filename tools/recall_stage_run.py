"""One immutable production attempt, fixed scope and counterfactuals."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import numpy as np
from recall_stage_core import cap, iou_matrix, pair_match, maximum_match, transition
from recall_stage_job import dependency, publish, read, sha, PARENT_SHA, TASK


def raw_geometry(raw, rows, j, torch):
    result = [[] for _ in rows]
    for view in raw['views']:
        v = view['view_id']
        for k, row in enumerate(rows):
            tile = j.tile_windows(row['height'],row['width'])[v-1] if v else None
            w,h = (tile.width,tile.height) if tile else (row['width'],row['height'])
            b = view['pred_boxes'][k]
            xy = torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),-1)
            xy *= torch.tensor([w,h]*2,dtype=torch.float32)
            mapped = j.map_crop_xyxy(xy.tolist(),tile) if tile else xy.tolist()
            result[k].extend((v,q,box) for q,box in enumerate(mapped))
    return result


def analyze_set(cands, gts, area, threshold, geometry=None, upper_bound=False):
    areas = {'small':(0,1024),'medium':(1024,9216),'large':(9216,1e10)}
    lo,hi = areas[area]
    crowd = np.array([bool(g['iscrowd']) for g in gts],dtype=bool)
    eligible = np.array([not g['iscrowd'] and lo<=g['area']<=hi for g in gts],dtype=bool)
    if any(g['area']<=0 for g in gts): raise ValueError('nonpositive development GT')
    ignored = ~eligible
    boxes = [[g['bbox'][0],g['bbox'][1],g['bbox'][0]+g['bbox'][2],g['bbox'][1]+g['bbox'][3]] for g in gts]
    gc = np.array([g['category_id'] for g in gts],dtype=int)
    pc = np.array([p.category_id for p in cands],dtype=int)
    v,cv = geometry if geometry is not None else (
        iou_matrix([p.box_xyxy for p in cands],boxes),
        iou_matrix([p.box_xyxy for p in cands],boxes,crowd))
    geom_best = v.argmax(axis=0) if len(cands) else np.full(len(gts),-1)
    geom = v.max(axis=0,initial=0.)
    correct = np.where(pc[:,None] == gc[None,:],v,-1.)
    correct_best = correct.argmax(axis=0) if len(cands) else np.full(len(gts),-1)
    correct_best[correct.max(axis=0,initial=-1.)<0] = -1
    correct_max = correct.max(axis=0,initial=0.)
    matched = np.full(len(gts),-1,dtype=int)
    upper = 0
    for cat in range(1,11):
        gi = np.flatnonzero(gc==cat); di = np.flatnonzero(pc==cat)
        if not len(gi): continue
        pairs = pair_match(cv[np.ix_(di,gi)],ignored[gi],crowd[gi],threshold)
        hit = pairs>=0
        matched[gi[hit]] = di[pairs[hit]]
        # Cardinality bound only on final sets (caller can ignore this value elsewhere).
        if upper_bound: upper += maximum_match(v[np.ix_(di,gi[eligible[gi]])]>=threshold)
    matched[ignored] = -1
    return {'eligible':eligible,'geometry':geom>=threshold,'correct_geometry':correct_max>=threshold,
            'match':matched>=0,'matched_index':matched,'geometry_index':geom_best,
            'correct_index':correct_best,'geometry_iou':geom,'correct_iou':correct_max,
            'upper_match_count':upper if upper_bound else None}


def run(parent, contract, out):
    start = time.monotonic()
    c = read(contract)
    j,torch,spec,rows,parsed = dependency(parent)
    root = Path(__file__).parent.parent
    j.require(c['task_id']==TASK and c['parent_sha256']==PARENT_SHA, 'execution identity differs')
    j.require(c['commit'] == read(root/'RUN_SOURCE.json')['commit'], 'release commit differs')
    for r in c['own_files']:
        j.require(sha(root/r['path'])==r['sha256'], 'own frozen source differs')
    plan = read(root/'configs/large_recall_stage.json')
    j.require(sha(root/'configs/large_recall_stage.json')==c['plan_sha256'], 'plan differs')
    j.require(plan['seed']==0 and plan['images']==548 and plan['main_budget']==600, 'wrong scope')
    publish(out/'ADMISSION.json',{'contract_sha256':sha(contract),'commit':c['commit'],
        'checkpoint':spec['identity']['checkpoint_sha256'],'resource':j.resources(out,j.read(j.PLAN)),
        'raw_refs':spec['full']['raw_view_refs'],'parent_sha256':PARENT_SHA})
    from slice_core import classwise_nms, box_iou64
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.data_protocol.evaluation import PrimaryImageV2
    proofs, expected = [], {}
    for arm in ('A','B','C'):
        ps=json.loads(j.exact(spec['evaluation']['predictions'][arm]['600'],contents=True))
        proofs.append(j.evaluation_input_proof(spec,rows,parsed,ps,arm,600))
        expected[arm]=j.group(ps,spec['ids'],600)
    publish(out/'INPUT_PROOFS.json',proofs)
    aggregates = defaultdict(Counter)
    group_rows = defaultdict(Counter)
    changes = defaultdict(Counter)
    paths = defaultdict(Counter)
    cases = {}
    counts=[]; resources=[]
    def add_values(key,sequence,cat,values):
        aggregates[key].update(values)
        group_rows[(key,'sequence',sequence)].update(values)
        group_rows[(key,'class',str(cat))].update(values)
    with (out/'GT_DETAILS.jsonl').open('x',encoding='utf-8') as f:
        for batch,ref in enumerate(spec['full']['raw_view_refs']):
            resources.append(j.resources(out,j.read(j.PLAN)))
            raw=torch.load(io.BytesIO(j.exact(ref,contents=True)),map_location='cpu',weights_only=True)
            part=rows[4*batch:4*batch+4]
            pools=j.decode_raw(raw,part,spec['identity'],0)
            geoms=raw_geometry(raw,part,j,torch)
            for k,(row,pool,raw_geom) in enumerate(zip(part,pools,geoms)):
                image=row['stable_image_id']; sequence=Path(row['annotation_relative_path']).stem.split('_')[0]
                ann=parsed[4*batch+k]
                geomap={(v,q):tuple(b) for v,q,b in raw_geom}
                j.require(len(geomap)==1500 and len(pool)==3000
                    and all(p.box_xyxy==geomap[(p.view_id,p.query_id)] for p in pool),'raw query mapping differs')
                data,_,_=ev.bridge_inputs([row],[ann],[])
                gt=data['annotations']
                im=PrimaryImageV2(image,row['width'],row['height'])
                integral=primary._build_integral(im,[g for g in ev.truth(row,ann) if g.category_id==0])
                def filtered(ps): return [p for p in ps if primary._ignored_fraction(integral,im,p.box_xyxy)<.5]
                whole=[p for p in pool if p.view_id==0]
                original={}; sets={}; nms_cache={}
                for arm,candidates in (('A',whole),('B',whole),('C',pool)):
                    ordered=sorted(candidates,key=lambda p:p.sort_key)
                    nms_cache[arm]=classwise_nms(ordered) if arm!='A' else ordered
                    top=nms_cache[arm][:600]
                    rebuilt=[j.prediction(image,p,whole=arm=='A') for p in top]
                    j.require(rebuilt==expected[arm][image], 'original '+arm+' reconstruction differs')
                    filt=filtered(top)
                    _,canonical,fc=ev.bridge_inputs([row],[ann],rebuilt)
                    as_bridge=[dict(image_id=1,category_id=p.category_id,score=p.score,
                        bbox=[p.box_xyxy[0],p.box_xyxy[1],p.box_xyxy[2]-p.box_xyxy[0],p.box_xyxy[3]-p.box_xyxy[1]]) for p in filt]
                    j.require(as_bridge==canonical, 'original raster filter differs')
                    stages=(ordered,nms_cache[arm],top,filt,cap(filt))
                    for stage,ps in zip(plan['stages'],stages): sets[arm+'/'+stage]=ps
                    original[arm]=arm+'/class100'
                    counts.append(dict(image=image,arm=arm,pool=len(ordered),nms=len(nms_cache[arm]),
                                       top=len(top),filtered=len(filt),class100=len(stages[-1]),**fc))
                for arm,candidates in (('B',whole),('C',pool)):
                    for n,b,cl in itertools.product((0,1),repeat=3):
                        ps=nms_cache[arm] if n else sorted(candidates,key=lambda p:p.sort_key)
                        if b: ps=ps[:600]
                        sets[f'{arm}/cf_n{n}b{b}c{cl}']=cap(filtered(ps),bool(cl))
                # Exact duplicated sets are matched once, not scientific reruns.
                cache={}
                boxes=[[g['bbox'][0],g['bbox'][1],g['bbox'][0]+g['bbox'][2],g['bbox'][1]+g['bbox'][3]] for g in gt]
                geometries={}
                for name,ps in sets.items():
                    identity=tuple(p.identity for p in ps)
                    if identity not in geometries:
                        geometries[identity]=(iou_matrix([p.box_xyxy for p in ps],boxes),
                                             iou_matrix([p.box_xyxy for p in ps],boxes,[bool(g['iscrowd']) for g in gt]))
                final_ids={tuple(p.identity for p in sets[original[a]]) for a in ('A','B','C')}
                raw_best_by_arm={a:iou_matrix([b for v,q,b in raw_geom if a=='C' or v==0],boxes).max(axis=0,initial=0.)
                                 for a in ('A','B','C')}
                for area in ('small','medium','large'):
                    for t in (.5,.75):
                        results={}
                        for name,ps in sets.items():
                            identity=tuple(p.identity for p in ps)
                            key=(identity,area,t)
                            if key not in cache: cache[key]=analyze_set(ps,gt,area,t,geometries[identity],identity in final_ids)
                            a=cache[key]; results[name]=a
                            eligible=a['eligible']
                            for cat in range(1,11):
                                mask=eligible & np.array([g['category_id']==cat for g in gt])
                                values={'GT':int(mask.sum()),'geometry':int((mask&a['geometry']).sum()),
                                        'correct_geometry':int((mask&a['correct_geometry']).sum()),
                                        'matched':int((mask&a['match']).sum())}
                                add_values(f'{area}|{t}|{name}',sequence,cat,values)
                            if a['upper_match_count'] is not None:
                                aggregates[f'{area}|{t}|{name}']['upper_match_count']+=a['upper_match_count']
                        for arm in ('A','B','C'):
                            stage_names=[arm+'/'+s for s in plan['stages']]
                            for bef,aft in zip(stage_names,stage_names[1:]):
                                j.require(not np.any(results[aft]['geometry'] & ~results[bef]['geometry'])
                                    and not np.any(results[aft]['correct_geometry'] & ~results[bef]['correct_geometry']),
                                    'non-nested geometric coverage')
                                for field in ('geometry','correct_geometry','match'):
                                    e=results[bef]['eligible']
                                    tr=transition(results[bef][field][e],results[aft][field][e])
                                    changes[f'{area}|{t}|{bef}->{aft}|{field}'].update(tr)
                            raw_best=raw_best_by_arm[arm]
                            ae=results[original[arm]]['eligible']
                            for cat in range(1,11):
                                mask=ae & np.array([g['category_id']==cat for g in gt],dtype=bool)
                                add_values(f'{area}|{t}|{arm}/raw_geometry',sequence,cat,
                                    {'GT':int(mask.sum()),'geometry':int((mask&(raw_best>=t)).sum()),
                                     'correct_geometry':0,'matched':0})
                            for gidx,g in enumerate(gt):
                                if not results[original[arm]]['eligible'][gidx]:continue
                                track=[]
                                for name in stage_names:
                                    a=results[name]; ps=sets[name]
                                    def cand_at(index):
                                        return ps[int(index)].as_dict() if index>=0 else None
                                    track.append({'stage':name.split('/')[1],
                                        'geometry_best':cand_at(a['geometry_index'][gidx]),
                                        'geometry_iou':float(a['geometry_iou'][gidx]),
                                        'correct_best':cand_at(a['correct_index'][gidx]),
                                        'correct_iou':float(a['correct_iou'][gidx]),
                                        'match':cand_at(a['matched_index'][gidx])})
                                first=next((x['stage'] for x in track if x['correct_iou']<t),'available')
                                bucket='no_raw_geometry' if raw_best[gidx]<t else first
                                paths[f'{area}|{t}|{arm}|geometry_first_missing'][bucket]+=1
                                f.write(json.dumps(dict(image=image,sequence=sequence,raw_annotation_id=g['raw_annotation_id'],
                                    category=g['category_id'],area_group=area,IoU=t,arm=arm,raw_best_iou=float(raw_best[gidx]),
                                    first_correct_geometry_unavailable=first,track=track),allow_nan=False)+'\n')
                        for before,after in (('A','B'),('A','C'),('B','C')):
                            x,y=results[original[before]],results[original[after]]
                            e=x['eligible']; tr=transition(x['match'][e],y['match'][e])
                            changes[f'{area}|{t}|{before}->{after}|match'].update(tr)
                            for cat in range(1,11):
                                mask=e & np.array([g['category_id']==cat for g in gt])
                                values=transition(x['match'][mask],y['match'][mask])
                                add_values(f'{area}|{t}|{before}->{after}',sequence,cat,values)
                            for gidx in np.flatnonzero(e & (x['match'] != y['match'])):
                                lost=bool(x['match'][gidx]); g=gt[gidx]
                                names=[after+'/'+s for s in plan['stages']]
                                flags=[bool(results[n]['match'][gidx]) for n in names]
                                first=next((n.split('/')[1] for n,v in zip(names,flags) if not v),'none')
                                pkey=f'{area}|{t}|{before}->{after}|'+('lost' if lost else 'gained')
                                if lost: paths[pkey][first]+=1
                                if area=='large':
                                    # One lexicographically first witness per comparison/sign/first path.
                                    ckey=pkey+'|'+(first if lost else 'final')
                                    ident=(str(image),str(g['raw_annotation_id']))
                                    if ckey not in cases or ident<tuple(cases[ckey]['case_order']):
                                        witness={'case_order':ident,'image':image,'sequence':sequence,
                                            'GT':g,'before':before,'after':after,'match_trace_after':dict(zip(names,flags)),
                                            'recovered_after_first_loss':any(not a and b for a,b in zip(flags,flags[1:])),
                                            'before_match':sets[original[before]][int(x['matched_index'][gidx])].as_dict() if lost else None,
                                            'after_match':sets[original[after]][int(y['matched_index'][gidx])].as_dict() if not lost else None}
                                        pre=results[after+'/view_pool']; idx=int(pre['correct_index'][gidx])
                                        best=sets[after+'/view_pool'][idx] if idx>=0 else None
                                        witness['correct_best_before_NMS']=best.as_dict() if best else None
                                        if best and best not in nms_cache[after]:
                                            suppressors=[p for p in nms_cache[after] if p.category_id==best.category_id
                                                and p.sort_key<best.sort_key and box_iou64(p.box_xyxy,best.box_xyxy)>.5]
                                            j.require(bool(suppressors),'missing actual NMS suppressor')
                                            witness['suppressor']=suppressors[0].as_dict()
                                        cases[ckey]=witness
                        for arm in ('B','C'):
                            base=results[original[arm]]
                            for n,b,cl in itertools.product((0,1),repeat=3):
                                name=f'{arm}/cf_n{n}b{b}c{cl}'
                                e=base['eligible']
                                changes[f'{area}|{t}|{original[arm]}->{name}|match'].update(
                                    transition(base['match'][e],results[name]['match'][e]))
            print(json.dumps({'batch':batch+1,'batches':137,'time':time.time()}),flush=True)
    summaries={k:dict(v) for k,v in aggregates.items()}
    # Sequence means include every sequence with positive fixed GT denominator.
    seq_rates={}
    for key in summaries:
        groups=[v for (q,kind,_),v in group_rows.items() if q==key and kind=='sequence' and v.get('GT',0)>0]
        if groups:
            seq_rates[key]={'sequences_with_GT':len(groups),
                **{field:sum(v[field]/v['GT'] for v in groups)/len(groups) for field in ('geometry','correct_geometry','matched')}}
    publish(out/'SUMMARY.json',{'totals':summaries,'sequence_equal':seq_rates,
                              'transitions':{k:dict(v) for k,v in changes.items()},
                              'reference_loss_first_unmatched':{k:dict(v) for k,v in paths.items()}})
    publish(out/'BY_GROUP.json',[dict(setting=q,kind=kind,group=g,**dict(v)) for (q,kind,g),v in group_rows.items()])
    publish(out/'CASES.json',dict(sorted(cases.items())))
    publish(out/'COUNTS.json',counts)
    publish(out/'RESOURCE.json',resources)
    refs=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file()]
    j.require(not torch.cuda.is_initialized(),'CUDA initialized')
    return {'status':'COMPLETE','seed':0,'images':548,'raw_batches':137,
            'original_A_B_C_exact':True,'input_proofs':3,'files':refs,
            'training_updates':0,'forward_calls':0,'AP_calls':0,'GPU_calls':0,
            'scientific_elapsed_seconds':time.monotonic()-start,'automatic_next':False}
