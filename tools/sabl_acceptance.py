"""PR26 isolated [1,2,2] engineering acceptance. No formal/evaluation subcommand."""
import argparse,hashlib,json,os,sys,time,subprocess,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# The actual CLI owns its import/runtime environment; caller PYTHONPATH is unnecessary.
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1'}.items():os.environ[k]=v

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def plan(a):
    c={'schema':'PR26_ACCEPTANCE_ONLY_V1','mode':'CPU_SYNTHETIC_SABL' if a.synthetic else 'GPU_SABL_ACCEPTANCE_ONLY',
       'arms':['N','S','A'],'updates':[1,2,2],'forward_cap':18,'actual_update_cap':9,
       'gpu_seconds':0 if a.synthetic else 900,'wall_seconds':300 if a.synthetic else 1800,'allocated_bytes':18<<30,'storage_bytes':30<<30,'return_bytes':5<<20,
       'manifest_sha256':a.manifest_sha,'attempt_root':str(Path(a.attempt_root).resolve()),'metadata':None,'gpu_uuid':a.gpu_uuid,
       'formal_training':False,'science_scoring':False,'N_checkpoint_SHA':'be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'}
    if not a.synthetic:
        need(a.metadata is not None and a.gpu_uuid and a.gpu_uuid.startswith('GPU-'),'certified metadata and GPU UUID required')
        old=read(a.metadata);ep=old['historical_N']['endpoint'];need(sha(ep['path'])==ep['sha256'],'endpoint receipt SHA')
        n=read(ep['path']);need(n['checkpoint']['sha256']==c['N_checkpoint_SHA'],'N registered checkpoint')
        c['metadata']={'checkpoint':n['checkpoint']['path'],'members':old['data']['train_core']['manifest'],
                       'roots':old['data']['train_core']['roots'],'source_metadata_sha256':sha(a.metadata),'python':old['python']}
    write(a.out,c)


def run(a):
    out=Path(a.out);out.mkdir();start=time.monotonic();ledger={'detector_forward':0,'actual_updates':0};engine=None
    try:
        c=read(a.contract);synthetic=c['mode']=='CPU_SYNTHETIC_SABL'
        need(c['schema']=='PR26_ACCEPTANCE_ONLY_V1' and c['mode'] in ('CPU_SYNTHETIC_SABL','GPU_SABL_ACCEPTANCE_ONLY'),'unknown contract')
        need(c['arms']==['N','S','A'] and c['updates']==[1,2,2] and c['forward_cap']==18 and c['actual_update_cap']==9,'fixed acceptance computation')
        need(c['formal_training'] is False and c['science_scoring'] is False,'forbidden workflow')
        need(c['gpu_seconds']==(0 if synthetic else 900) and c['wall_seconds']==(300 if synthetic else 1800) and c['allocated_bytes']==18<<30 and c['storage_bytes']==30<<30 and c['return_bytes']==5<<20,'fixed resources')
        need(Path(c['attempt_root']).resolve()==out.resolve(),'attempt identity')
        from r18_deploy import verify_release
        release=verify_release(ROOT,c['manifest_sha256'])
        if not synthetic:
            need(not release['candidate_not_for_real_execution'],'candidate source cannot execute GPU')
            need(a.authorization and read(a.authorization)=={'contract':digest(c),'scope':'PR26_SHORT_GPU_ACCEPTANCE_ONLY','authorize_once':True},'GPU authorization absent/mismatched')
            os.environ['CUDA_VISIBLE_DEVICES']=c['gpu_uuid']
            query=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            gpu={line.split(',')[0].strip():int(line.split(',')[1]) for line in query.strip().splitlines()}
            need(gpu[c['gpu_uuid']]>=18432,'free GPU memory')
            need(shutil.disk_usage(out).free>=c['storage_bytes'],'fresh free disk')
            need(Path(sys.executable).resolve()==Path(c['metadata']['python']).resolve(),'certified interpreter path')
            need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU task exists')
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=''
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
        import torch,numpy as np,random,torchvision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        need(torch.__version__.startswith('2.4.1') and torchvision.__version__.startswith('0.19.1'),'frozen runtime')
        if not synthetic:need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory>=18<<30,'CUDA identity/runtime')
        torch.backends.cudnn.benchmark=False
        from sparse_rtdetr.research.sabl_control import make_engine,VERSION
        from sparse_rtdetr.research import sabl_state as state
        from sparse_rtdetr.protocol_r18.engine import output_digest,tensor_digest
        from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
        timeout=300 if synthetic else 900
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('acceptance deadline')));signal.alarm(timeout)
        write(out/'CONSUMED.json',{'contract':digest(c),'scope':c['mode'],'no_retry':True})
        def guard():
            need(time.monotonic()-start<timeout,'cumulative time cap')
            need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=c['storage_bytes'],'task storage')
            if not synthetic:need(torch.cuda.max_memory_allocated()<=c['allocated_bytes'],'allocated cap')
        def charge(module,args):
            guard();need(ledger['detector_forward']<18,'forward cap')
            ledger['detector_forward']+=1
            tmp=out/'ledger.tmp'
            with tmp.open('w') as f:json.dump(ledger,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,out/'ledger.json')
            from sparse_rtdetr.protocol_r18.runio import fsync_directory
            fsync_directory(out)
        def seed():
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            if not synthetic:torch.cuda.manual_seed_all(0)
        if synthetic:
            from PIL import Image
            from sparse_rtdetr.localization.data import RawAugmentation,parse_raw
            windows=[]
            for window in range(2):
                rows=[]
                for i in range(2):
                    id='SYNTHETIC-%d-%d'%(window,i)
                    raw=parse_raw(b'12,16,24,20,1,1,0,0\n50,42,36,30,1,2,0,0\n0,0,8,8,0,0,0,0\n',id)
                    pixels,target,_=RawAugmentation((128,128))(Image.new('RGB',(160,128),(30+window*40,50+i*10,80)),raw,epoch=1,seed=0,position=i,image_id=id,training=False)
                    rows.append((pixels,target))
                windows.append((torch.stack([x[0] for x in rows]),[x[1] for x in rows]))
        else:
            from sparse_rtdetr.localization.data import RawRoleDataset,WindowSampler
            m=c['metadata'];need(sha(m['checkpoint'])==c['N_checkpoint_SHA'],'fresh N SHA')
            need(sha(m['members']['path'])=='df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d','train_core manifest')
            roots=m['roots'];ds=RawRoleDataset(role='train_core',manifest_path=m['members']['path'],images_path=roots['images'],annotations_path=roots['annotations'],root_binding=roots)
            # First two native seed0/epoch1 logical windows, not a new GT-selected subset.
            keys=[key for window in WindowSampler(len(ds),seed=0,epoch=1,completed_windows=0,window_limit=2) for key in window]
            rows=[ds[key] for key in keys];need(len(rows)==32,'two native logical16 windows')
            windows=[(torch.stack([x[0] for x in rows[k:k+16]]),[x[1] for x in rows[k:k+16]]) for k in (0,16)]
        input_hash=[output_digest([images,targets]) for images,targets in windows]
        results={};common_initial=None;head_initial=None
        for arm in c['arms']:
            seed();engine=make_engine(arm,synthetic,None if synthetic else c['metadata']['checkpoint'],'cpu' if synthetic else 'cuda')
            shared=tensor_digest(engine.model.native.state_dict().items())
            if common_initial is None:common_initial=shared
            need(shared==common_initial,'shared N/BN initial bytes')
            if arm!='N':
                hd=tensor_digest(engine.model.boundary.state_dict().items())
                if head_initial is None:head_initial=hd
                need(hd==head_initial,'S/A initial head bytes')
            need(not engine.optimizer.state and engine.ema.updates==0,'fresh optimizer/EMA')
            binding={'scope':c['mode'],'arm':arm,'version':VERSION,'contract':digest(c),'release_manifest':c['manifest_sha256'],'N_SHA':c['N_checkpoint_SHA']}
            folder=out/arm;folder.mkdir();engine.model.register_forward_pre_hook(charge);seed()
            engine.train_window(*windows[0],capture=True);ledger['actual_updates']+=1
            cp=state.save(engine,binding,folder/'C1.pt')
            direct=engine.train_window(*windows[1],capture=True);ledger['actual_updates']+=1
            expected=state.capture(engine,binding);audit=state.restore(engine,binding,folder/'C1.pt',cp['sha256'])
            replay=engine.train_window(*windows[1],capture=True);ledger['actual_updates']+=1
            actual=state.capture(engine,binding)
            need(_equal(expected,actual),'post-replay complete state')
            need(all(direct[k]==replay[k] for k in ('microbatch_losses','gradient_sha256_before_clip','output_sha256','dn_noise_sha256')),'replay input/DN/output/loss/gradient')
            boundary_dtype=engine.model.last_boundary_dtype
            need(boundary_dtype==(None if arm=='N' else 'torch.float32'),'boundary FP32 dtype')
            dtype=engine.model.last_dtype;need(dtype==('torch.float32' if synthetic else 'torch.bfloat16'),'real native logits dtype')
            final=state.save(engine,binding,folder/'C2.pt');guard()
            results[arm]={'restore':audit,'replay_full_state_exact':True,'logits_dtype':dtype,'boundary_logits_dtype':boundary_dtype,'updates':3,'checkpoint':final,'shared_initial_SHA':shared,'head_initial_SHA':None if arm=='N' else hd,'input_SHA':input_hash,'new_losses':engine.criterion.recipe()['new_losses']}
            need(input_hash==[output_digest([images,targets]) for images,targets in windows],'input tensors/matching data mutated')
            del engine,expected,actual;engine=None
            import gc;gc.collect()
            if not synthetic:torch.cuda.empty_cache()
        need(ledger=={'detector_forward':18,'actual_updates':9},'fixed terminal ledger')
        # Last charge emitted before the final optimizer step; publish final cumulative ledger.
        (out/'ledger.json').write_text(json.dumps(ledger))
        slim=out/'slim';slim.mkdir();guard()
        result={'status':'PASS_ENGINEERING','scope':c['mode'],'arms':results,'ledger':ledger,'seconds':time.monotonic()-start,
                'peak_allocated_bytes':None if synthetic else torch.cuda.max_memory_allocated(),'formal_training_started':False,'scientific_quality':'NOT_SCIENTIFIC','used_acceptance_state_for_formal':False,'must_exit':True}
        write(slim/'RESULT.json',result);write(slim/'CONTRACT.json',c)
        write(slim/'SHA_INDEX.json',{p.name:sha(p) for p in slim.iterdir()})
        need(sum(p.stat().st_size for p in slim.iterdir())<=c['return_bytes'],'return cap')
        write(out/'CLOSED.json',{'status':'PASS_ENGINEERING','contract':digest(c),'ledger':ledger,'no_followup':True});write(out/'EXIT.json',{'returncode':0,'seconds':time.monotonic()-start})
        return 0
    except Exception as e:
        if engine:ledger['failed_engine_updates']=engine.completed_updates
        write(out/'STOP.json',{'error':repr(e),'ledger':ledger,'seconds':time.monotonic()-start,'no_retry':True})
        write(out/'EXIT.json',{'returncode':1});print(repr(e),file=sys.stderr);return 1
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('plan');a.add_argument('--synthetic',action='store_true');a.add_argument('--metadata');a.add_argument('--gpu-uuid');a.add_argument('--manifest-sha',required=True);a.add_argument('--attempt-root',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('run');a.add_argument('--contract',required=True);a.add_argument('--authorization');a.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='plan':plan(a)
    else:raise SystemExit(run(a))
