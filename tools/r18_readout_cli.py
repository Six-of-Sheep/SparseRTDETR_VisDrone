"""PR25 frozen-feature learning CLI. No detector/image reader or training chain."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/research/r18_reference_topic'
MEMBER_SHA='df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d'
N_SHA='be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'
DEFINITION_SHA='86e5503a8829e813747dc3e1e5723dc0fd3077ffead2acb8b582654b12f03425'
SOURCE_FILES=['tools/r18_readout_cli.py','tools/r18_readout_math.py','tools/r18_extract.py','tools/r18_full_chain.py','tools/r18_deploy.py','docs/research/r18_reference_topic/CPU_DEFINITION.json','docs/research/r18_reference_topic/LEARNING_CONFIG.json','docs/research/r18_reference_topic/SELECTED_IDS.json']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def need(ok,s):
    if not ok: raise ValueError(s)
def sources(): return {p:sha(ROOT/p) for p in SOURCE_FILES}


def select(path,out):
    need(sha(path)==MEMBER_SHA,'uncertified member bytes')
    d=read(path);need(set(d)=={'records'} and len(d['records'])==4869,'member schema')
    ids=[r['stable_image_id'] for r in d['records']]
    need(len(set(ids))==4869 and all(isinstance(s,str) and len(s)==64 for s in ids),'member identity')
    ranked=sorted(ids,key=lambda s:(hashlib.sha256(('PR25-probe-v1\0'+s).encode()).digest(),s.encode()))
    result={'schema':'pr25-selection-v1','member_sha256':MEMBER_SHA,'fit':ranked[:128],'probe':ranked[128:192],
            'probe_is_N_training_data':True}
    result['ids_sha256']=digest({'fit':result['fit'],'probe':result['probe']})
    write(out,result)


def torch_setup(cpu):
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if cpu:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
        os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0);random.seed(0)
    return torch


def fixture(out,variant):
    t=torch_setup(True);out=Path(out);out.mkdir()
    ids=['SYNTHETIC-%03d'%i for i in range(24)]
    q=t.randn(24,3,2,256)*.01
    r=t.tensor([.5,.5,.2,.2]).repeat(24,2,1)
    gt=r.clone();gt[:,:,0]+=.002; mask=t.ones(24,2,dtype=t.bool)
    if variant=='empty':mask[:]=False
    if variant=='invalid':r[0,0,2]=-1
    if variant=='nonfinite':q[0,0,0,0]=float('nan')
    if variant=='probe_changed':q[16:]+=.3;gt[16:,:,0]+=.01
    if variant=='empty_batch':mask[:8]=False
    data={'ids':ids,'q':q,'r':r,'gt':gt,'mask':mask,'small':mask.clone(),
          'gt_ids':t.arange(2).repeat(24,1),'category':t.ones(24,2,dtype=t.int64)}
    t.save(data,out/'features.pt')
    c={'schema':'pr25-readout-v1','mode':'SYNTHETIC_NOT_SCIENTIFIC','definition_sha256':DEFINITION_SHA,
       'sources':sources(),'features_sha256':sha(out/'features.pt'),'ids':ids,'fit_count':16,'probe_count':8,
       'queries':2,'updates_per_arm':2,'detector_forward_used':0,'limits':{'detector_forward':48,'head_update':9,'head_eval':12},
       'wall_seconds':300,'gpu_seconds':0,'peak_allocated_bytes':18<<30,'storage_bytes':512<<20,'export_bytes':5<<20,
       'coordinate':'normalized_canvas_cxcywh','matching':'fixed_N_final_Hungarian_ordinary','N_checkpoint_sha256':N_SHA,
       'roundtrip_check':True}
    if variant=='identity':c['N_checkpoint_sha256']='0'*64
    if variant=='budget':c['limits']['head_update']=0
    write(out/'contract.json',c)


def bind(selection,features,receipt,out,synthetic=False):
    """Bind already-extracted material; does not authorize/extract any features."""
    s=read(selection);r=read(receipt)
    need(r['status']=='CLOSED' and r['features_sha256']==sha(features),'missing closure or feature SHA')
    need(r.get('semantics')=='pr25-native-N-EMA-layer012-initial-v1','capture semantics')
    if synthetic:
        need(r['mode']=='SYNTHETIC_NOT_SCIENTIFIC' and r['detector_forward']==6 and s['fit']==['SYNTHETIC-%03d'%i for i in range(16)] and s['probe']==['SYNTHETIC-%03d'%i for i in range(16,24)],'synthetic capture binding')
        c={'schema':'pr25-readout-v1','mode':'SYNTHETIC_NOT_SCIENTIFIC','definition_sha256':DEFINITION_SHA,
           'sources':sources(),'features_sha256':sha(features),'ids':s['fit']+s['probe'],'fit_count':16,'probe_count':8,
           'queries':300,'updates_per_arm':2,'detector_forward_used':6,'limits':{'detector_forward':48,'head_update':9,'head_eval':12},
           'wall_seconds':min(300,3600-r['wall_seconds']),'gpu_seconds':0,'peak_allocated_bytes':18<<30,'storage_bytes':512<<20,'export_bytes':5<<20,
           'coordinate':'normalized_canvas_cxcywh','matching':'fixed_N_final_Hungarian_ordinary','N_checkpoint_sha256':N_SHA,
           'roundtrip_check':True,'extraction_receipt_sha256':sha(receipt)}
        write(out,c);return
    need(r['mode']=='REAL_N_EMA','synthetic receipt cannot authorize real execution')
    need(s==read(DOC/'SELECTED_IDS.json'),'selection differs from frozen actual IDs')
    need(s['member_sha256']==MEMBER_SHA and s['ids_sha256']==digest({'fit':s['fit'],'probe':s['probe']}),'selection')
    need(len(s['fit'])==128 and len(s['probe'])==64 and len(set(s['fit']+s['probe']))==192,'selection sizes')
    need(r['status']=='CLOSED' and r['N_checkpoint_sha256']==N_SHA and r['ids_sha256']==s['ids_sha256'],'extraction identity')
    need(r['detector_forward']==48 and r['features_sha256']==sha(features),'extraction budget/SHA')
    need(r['frozen_parameters'] and r['fixed_final_Hungarian'] and r['ordinary_only'],'extraction semantics')
    need(0<=r['gpu_seconds']<1800 and 0<=r['wall_seconds']<3600,'extraction time')
    c={'schema':'pr25-readout-v1','mode':'REAL_REQUIRES_SEPARATE_AUTHORIZATION','definition_sha256':DEFINITION_SHA,
       'sources':sources(),'features_sha256':sha(features),'ids':s['fit']+s['probe'],'fit_count':128,'probe_count':64,
       'queries':300,'updates_per_arm':200,'detector_forward_used':48,
       'limits':{'detector_forward':48,'head_update':600,'head_eval':96},'wall_seconds':3600-r['wall_seconds'],
       'gpu_seconds':1800-r['gpu_seconds'],'peak_allocated_bytes':18<<30,'storage_bytes':512<<20,'export_bytes':5<<20,
       'coordinate':'normalized_canvas_cxcywh','matching':'fixed_N_final_Hungarian_ordinary','N_checkpoint_sha256':N_SHA,
       'roundtrip_check':False,'extraction_receipt_sha256':sha(receipt),'selection_sha256':sha(selection)}
    write(out,c)


def run(args):
    out=Path(args.out);out.mkdir()  # atomic consumption: existing attempt never resumes
    started=time.monotonic();ledger={'detector_forward':0,'head_update':0,'head_eval':0};t=None;completed_updates=0
    def emit_ledger():
        tmp=out/'ledger.tmp';tmp.write_text(json.dumps(ledger));os.replace(tmp,out/'ledger.json')
    try:
        c=read(args.contract);identity=sha(args.contract)
        write(out/'CONSUMED.json',{'contract_sha256':identity,'no_retry':True})
        need(c['schema']=='pr25-readout-v1' and c['sources']==sources(),'source identity')
        need(c['definition_sha256']==DEFINITION_SHA==sha(DOC/'CPU_DEFINITION.json'),'science identity')
        need(c['N_checkpoint_sha256']==N_SHA,'N identity')
        need(c['coordinate']=='normalized_canvas_cxcywh' and c['matching']=='fixed_N_final_Hungarian_ordinary','coordinate/matching')
        synthetic=c['mode']=='SYNTHETIC_NOT_SCIENTIFIC'
        need(synthetic or c['mode']=='REAL_REQUIRES_SEPARATE_AUTHORIZATION','mode')
        if synthetic:
            need((c['fit_count'],c['probe_count'],c['updates_per_arm'])==(16,8,2) and c['queries'] in (2,300),'fixed synthetic schedule')
            need(c['limits']['head_update']<=9 and c['limits']['head_eval']<=12 and c['detector_forward_used']==(6 if c['queries']==300 else 0),'synthetic budget')
        if not synthetic:
            need(args.authorization is not None,'separate GPU authorization required')
            auth=read(args.authorization)
            need(auth=={'contract_sha256':identity,'authorize_once':True,'scope':'PR25_FROZEN_READOUT_ONLY'},'authorization identity')
            need((c['fit_count'],c['probe_count'],c['queries'],c['updates_per_arm'])==(128,64,300,200),'fixed real schedule')
            selected=read(DOC/'SELECTED_IDS.json')
            need(c['ids']==selected['fit']+selected['probe'],'fixed real IDs')
            need(c['limits']=={'detector_forward':48,'head_update':600,'head_eval':96} and not c['roundtrip_check'],'fixed budgets')
        need(0<c['wall_seconds']<= (300 if synthetic else 3600),'wall limit')
        need(c['peak_allocated_bytes']<=18<<30 and c['storage_bytes']<=512<<20 and c['export_bytes']<=5<<20,'resource limit')
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')))
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(RuntimeError('termination requested')))
        signal.alarm(max(1,int(c['wall_seconds'] if synthetic else min(c['wall_seconds'],c['gpu_seconds']))))
        need(sha(args.features)==c['features_sha256'],'features SHA')
        t=torch_setup(synthetic);from r18_readout_math import Readout,loss,overlaps,xyxy,local_uniform
        device='cpu' if synthetic else 'cuda'
        if not synthetic:need(t.cuda.is_available(),'CUDA unavailable')
        data=t.load(args.features,map_location='cpu',weights_only=True)
        if c['queries']==300:
            need(data['layer_order']==[0,1,2] and t.equal(data['query_ids'],t.arange(300)) and data['coordinate']==c['coordinate'],'captured layer/query/coordinate identity')
        need(data['ids']==c['ids'] and len(set(data['ids']))==len(data['ids']),'feature IDs/order')
        n=c['fit_count']+c['probe_count'];q=c['queries']
        for key,shape in [('q',(n,3,q,256)),('r',(n,q,4)),('gt',(n,q,4)),('mask',(n,q)),('small',(n,q)),('gt_ids',(n,q)),('category',(n,q))]:
            v=data[key];need(tuple(v.shape)==shape and not v.requires_grad,'feature shape/frozen '+key)
            if key in ('q','r','gt'):need(v.dtype==t.float32 and bool(t.isfinite(v).all()),'nonfinite/FP32 '+key)
        need(data['mask'].dtype==t.bool and data['small'].dtype==t.bool,'mask dtype')
        need(bool((data['r'][...,2:]>0).all()) and bool((data['gt'][data['mask']][...,2:]>0).all()),'invalid input box')
        for j in range(n):
            a=data['gt_ids'][j][data['mask'][j]].tolist();need(len(a)==len(set(a)),'one-to-one GT identity')
        need(bool(data['mask'][:c['fit_count']].any()),'empty fitting matches')
        frozen=digest({k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in data.items() if isinstance(v,t.Tensor)})
        ledger['detector_forward']=c['detector_forward_used'];emit_ledger()
        def guard():
            need(time.monotonic()-started<c['wall_seconds'],'wall budget')
            if not synthetic:
                need(time.monotonic()-gpu_start<c['gpu_seconds'],'GPU stage budget')
                need(t.cuda.max_memory_allocated()<=c['peak_allocated_bytes'],'allocated budget')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+Path(args.features).stat().st_size
            need(size<=c['storage_bytes'],'storage budget')
        def charge(key):
            guard();need(ledger[key]<c['limits'][key],key+' budget');ledger[key]+=1;emit_ledger()
        need(ledger['detector_forward']<=c['limits']['detector_forward'],'detector budget')
        gpu_start=time.monotonic()
        models={a:Readout(a).to(device) for a in ('C0','C1','M')}
        need(all(sum(p.numel() for p in m.parameters())==101772 for m in models.values()),'parameter count')
        need(all(t.equal(x,y) for a in ('C1','M') for x,y in zip(models['C0'].parameters(),models[a].parameters())),'initial state')
        opts={a:t.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0) for a,m in models.items()}
        need(all({id(p) for g in opts[a].param_groups for p in g['params']}=={id(p) for p in models[a].parameters()} for a in models),'optimizer head only')
        def batch(ix):return {k:v[ix].to(device).detach() for k,v in data.items() if isinstance(v,t.Tensor)}
        def step(a,index):
            nonlocal completed_updates
            ix=[(index*8+j)%c['fit_count'] for j in range(8)]
            b=batch(ix);charge('head_update');opts[a].zero_grad(set_to_none=True)
            state=models[a](b['q'],b['r']);v=loss(state,b['gt'],b['mask']);need(bool(t.isfinite(v)),'nonfinite loss')
            v.backward();need(all(p.grad is not None and bool(t.isfinite(p.grad).all()) for p in models[a].parameters()),'nonfinite gradient')
            opts[a].step();need(all(bool(t.isfinite(p).all()) for p in models[a].parameters()),'nonfinite parameter')
            completed_updates+=1
            guard();return float(v.detach())
        def save(a,i,path):
            t.save({'model':models[a].state_dict(),'optimizer':opts[a].state_dict(),'torch_rng':t.get_rng_state(),
                    'cuda_rng':t.cuda.get_rng_state_all() if not synthetic else [],'python_rng':random.getstate(),
                    'arm':a,'next_update':i,'contract':identity,'features_sha256':c['features_sha256'],'ledger':dict(ledger)},path)
        def restore(a,path):
            s=t.load(path,map_location=device,weights_only=False)
            need(s['contract']==identity and s['features_sha256']==c['features_sha256'] and s['arm']==a,'checkpoint identity')
            models[a].load_state_dict(s['model']);opts[a].load_state_dict(s['optimizer'])
            t.set_rng_state(s['torch_rng'].cpu());random.setstate(s['python_rng'])
            if not synthetic:t.cuda.set_rng_state_all([state.cpu() for state in s['cuda_rng']])
            # Ledger intentionally never refunded by restored optimizer state.
            return s['next_update']
        evaluations={};roundtrips=[]
        def evaluate(a,phase):
            rows=[];local=[];models[a].eval()
            with t.no_grad():
                for start in range(c['fit_count'],n,4):
                    ix=list(range(start,min(start+4,n)));b=batch(ix);charge('head_eval')
                    states=models[a](b['q'],b['r']);pred=states[-1]['box'];iou,_=overlaps(pred,b['gt'])
                    err=(xyxy(pred)-xyxy(b['gt'])).abs()/b['gt'][...,[2,3,2,3]]
                    for k,j in enumerate(ix):
                        for z in b['mask'][k].nonzero().flatten().tolist():
                            rows.append({'image':data['ids'][j],'gt':int(b['gt_ids'][k,z]),'small':bool(b['small'][k,z]),
                                         'category':int(b['category'][k,z]),'iou':float(iou[k,z]),'edge_l1':float(err[k,z].mean()),
                                         'outside':[s['outside'][k,z].cpu().tolist() for s in states[1:]]})
                    if a=='M' and phase=='final':
                        for layer,u in enumerate(local_uniform(states),1):
                            ui,_=overlaps(u,b['gt']);pi,_=overlaps(states[layer]['box'],b['gt'])
                            local.extend((pi-ui)[b['mask']].cpu().tolist())
                    guard()
            models[a].train();evaluations[a+'-'+phase]={'rows':rows,'local_prior_iou_differences':local}
        for a in models:
            evaluate(a,'initial')
            for i in range(c['updates_per_arm']):
                if synthetic and c['roundtrip_check'] and i==1:
                    save(a,i,out/(a+'-roundtrip.pt'));v=step(a,i)
                    expected={k:x.detach().clone() for k,x in models[a].state_dict().items()}
                    expected_opt=opts[a].state_dict()
                    expected_rng=t.get_rng_state().clone();expected_py=random.getstate()
                    idx=restore(a,out/(a+'-roundtrip.pt'));need(idx==i,'next batch cursor')
                    v2=step(a,idx)
                    need(v==v2 and all(t.equal(expected[k],x) for k,x in models[a].state_dict().items()),'restored next update')
                    need(t.equal(expected_rng,t.get_rng_state()) and expected_py==random.getstate(),'RNG restore')
                    for k,state in expected_opt['state'].items():
                        for name,x in state.items():need(t.equal(x,opts[a].state_dict()['state'][k][name]),'optimizer restore')
                    roundtrips.append(a)
                else:step(a,i)
            save(a,c['updates_per_arm'],out/(a+'-final.pt'))
            restore(a,out/(a+'-final.pt'));evaluate(a,'final')
        need(frozen==digest({k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in data.items() if isinstance(v,t.Tensor)}),'features/matching mutated')
        need(sha(args.features)==c['features_sha256'],'feature file mutated')
        # No AP/AR, early stopping, best model, or training escalation.
        small={a:[r for r in evaluations[a+'-final']['rows'] if r['small']] for a in models}
        verdict={}
        for control in ('C0','C1'):
            base={(r['image'],r['gt']):r for r in small[control]};vals=[];errs=[]
            for r in small['M']:
                b=base[(r['image'],r['gt'])];vals.append(r['iou']-b['iou']);errs.append(r['edge_l1']-b['edge_l1'])
            median=float(t.tensor(vals).quantile(.5)) if vals else None
            mean=sum(errs)/len(errs) if errs else None
            verdict[control]={'count':len(vals),'median_iou_difference':median,'mean_edge_l1_difference':mean,
                              'gate':len(vals)>=100 and median>0 and mean<=0}
        export=out/'slim';export.mkdir()
        write(export/'RESULT.json',{'status':'WORK_COMPLETE_ROOT_CLOSURE_REQUIRED','scope':c['mode'],'ledger':ledger,'comparisons':verdict,
              'decision':'SYNTHETIC_NOT_SCIENTIFIC' if synthetic else ('REVIEW_ONLY' if all(x['gate'] for x in verdict.values()) else 'END_FIXED_EXPLORATION'),
              'roundtrip_arms':roundtrips,'feature_and_matching_unchanged':True,'optimizer_steps_completed':completed_updates,'seconds':time.monotonic()-started})
        payload=json.dumps(evaluations,allow_nan=False,separators=(',',':')).encode()
        with (export/'METRICS.json.gz').open('xb') as f: f.write(gzip.compress(payload,mtime=0))
        write(export/'CONTRACT.json',c)
        index={p.name:sha(p) for p in export.iterdir()};write(export/'SHA_INDEX.json',index)
        need(sum(p.stat().st_size for p in export.iterdir())<=c['export_bytes'],'slim budget');guard()
        write(out/'CLOSED.json',{'status':'PASS_ENGINEERING','contract_sha256':identity,'ledger':ledger,'exit':0})
        return 0
    except Exception as e:
        write(out/'STOP.json',{'error':repr(e),'ledger':ledger,'optimizer_steps_completed':completed_updates,'no_retry':True,'seconds':time.monotonic()-started})
        print(repr(e));return 1
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('select');a.add_argument('--members',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('fixture');a.add_argument('--out',required=True);a.add_argument('--variant',choices=['normal','empty','invalid','nonfinite','identity','budget','probe_changed','empty_batch'],default='normal')
    a=s.add_parser('bind');a.add_argument('--selection',required=True);a.add_argument('--features',required=True);a.add_argument('--receipt',required=True);a.add_argument('--out',required=True);a.add_argument('--synthetic',action='store_true')
    a=s.add_parser('run');a.add_argument('--contract',required=True);a.add_argument('--features',required=True);a.add_argument('--out',required=True);a.add_argument('--authorization')
    a=p.parse_args()
    if a.cmd=='select':select(a.members,a.out)
    elif a.cmd=='fixture':fixture(a.out,a.variant)
    elif a.cmd=='bind':bind(a.selection,a.features,a.receipt,a.out,a.synthetic)
    else:return run(a)
    return 0


if __name__=='__main__':raise SystemExit(main())
