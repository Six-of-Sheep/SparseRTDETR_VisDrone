"""One authorization, one extraction/readout chain, terminal exit. No e120."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from r18_readout_cli import ROOT,DOC,read,write,sha,digest,need
from r18_deploy import verify_release


def plan(args):
    need(args.synthetic or args.attempt_root is not None,'real plan requires unique attempt root')
    selection=read(DOC/'SELECTED_IDS.json')
    c={'mode':'SYNTHETIC_NOT_SCIENTIFIC' if args.synthetic else 'REAL_N_EMA',
       'semantics':'pr25-native-N-EMA-layer012-initial-v1','coordinate':'normalized_canvas_cxcywh','layer_order':[0,1,2],
       'ids':['SYNTHETIC-%03d'%i for i in range(24)] if args.synthetic else selection['fit']+selection['probe'],
       'forward_cap':48,'wall_seconds':300 if args.synthetic else 3600,'gpu_seconds':0 if args.synthetic else 1800,
       'release_manifest_sha256':args.manifest_sha,'checkpoint':args.checkpoint,'members':args.members,
       'roots':{'role':'train_core','images':args.images,'annotations':args.annotations},'gpu_uuid':args.gpu_uuid,'attempt_root':args.attempt_root}
    write(args.out,c)


def execute(args):
    c=read(args.contract);out=Path(args.out);out.mkdir();start=time.monotonic();records=[]
    synthetic=c['mode']=='SYNTHETIC_NOT_SCIENTIFIC'
    try:
        release=verify_release(ROOT,c['release_manifest_sha256'])
        need(c['forward_cap']==48 and c['wall_seconds']==(300 if synthetic else 3600) and c['gpu_seconds']==(0 if synthetic else 1800),'fixed full-chain budget')
        if not synthetic:
            need(not release['candidate_not_for_real_execution'],'candidate package cannot execute real experiment')
            need(Path(c['attempt_root']).is_absolute() and Path(c['attempt_root']).resolve()==out.resolve(),'unique attempt identity')
            need(args.authorization is not None,'single full-chain authorization missing')
            need(read(args.authorization)=={'chain_contract':digest(c),'authorize_once':True,'scope':'PR25_FULL_FROZEN_EXPLORATION'},'authorization mismatch')
            need(c['gpu_uuid'].startswith('GPU-'),'physical GPU binding')
            os.environ['CUDA_VISIBLE_DEVICES']=c['gpu_uuid']
            query=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            rows={line.split(',')[0].strip():int(line.split(',')[1]) for line in query.strip().splitlines()}
            need(rows[c['gpu_uuid']]>=18*1024,'fresh free GPU memory')
            env=subprocess.check_output([sys.executable,'-c','import torch,torchvision;print(torch.__version__);print(torchvision.__version__)'],text=True)
            need(env.splitlines()[0].startswith('2.4.1') and env.splitlines()[1].startswith('0.19.1'),'certified runtime')
        need(shutil.disk_usage(out).free>=512<<20,'fresh disk capacity')
        write(out/'CONSUMED.json',{'chain_contract':digest(c),'release':release['commit'],'no_retry':True})
        deadline=start+(300 if synthetic else 1800)
        def child(tool,*argv):
            remaining=deadline-time.monotonic();need(remaining>0,'chain time exhausted')
            command=[sys.executable,'-B',str(ROOT/'tools'/tool),*map(str,argv)]
            with (out/(tool+'.stdout')).open('w') as stdout,(out/(tool+'.stderr')).open('w') as stderr:
                p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=remaining)
            records.append({'tool':tool,'returncode':p.returncode});need(p.returncode==0,'child STOP '+tool)
        extraction_args=['--contract',args.contract,'--out',out/'extract']
        if not synthetic:extraction_args+=['--authorization',args.authorization]
        child('r18_extract.py',*extraction_args)
        receipt=out/'extract/CLOSED.json';features=out/'extract/features.pt'
        need(receipt.is_file(),'missing extraction closure')
        if synthetic:
            sel=out/'synthetic-selection.json';write(sel,{'fit':c['ids'][:16],'probe':c['ids'][16:]})
        else:sel=DOC/'SELECTED_IDS.json'
        bind_args=['bind','--selection',sel,'--features',features,'--receipt',receipt,'--out',out/'readout-contract.json']
        if synthetic:bind_args+=['--synthetic']
        child('r18_readout_cli.py',*bind_args)
        rc=read(out/'readout-contract.json')
        # Include orchestration/setup time as well as extraction in the inherited cap.
        elapsed=time.monotonic()-start
        rc['wall_seconds']=min(rc['wall_seconds'],c['wall_seconds']-elapsed)
        if not synthetic:rc['gpu_seconds']=min(rc['gpu_seconds'],1800-elapsed)
        live=out/'readout-contract-bound.json';write(live,rc)
        run_args=['run','--contract',live,'--features',features,'--out',out/'readout']
        if not synthetic:
            auth=out/'derived-readout-authorization.json';write(auth,{'contract_sha256':sha(live),'authorize_once':True,'scope':'PR25_FROZEN_READOUT_ONLY'})
            run_args+=['--authorization',auth]
        child('r18_readout_cli.py',*run_args)
        done=read(out/'readout/CLOSED.json');need(done['status']=='PASS_ENGINEERING','readout closure')
        need(done['ledger']['detector_forward']==(6 if synthetic else 48),'inherited forward counter')
        need(time.monotonic()<deadline,'chain time exceeded')
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=512<<20,'chain storage exceeded')
        slim=out/'slim';shutil.copytree(out/'readout/slim',slim)
        shutil.copy2(receipt,slim/'EXTRACTION_CLOSED.json')
        closure={'status':'PASS_ENGINEERING','scope':c['mode'],'ledger':done['ledger'],'seconds':time.monotonic()-start,'children':records,'no_more_work':True}
        write(slim/'CHAIN_CLOSED.json',closure)
        # Re-index the final combined whitelist, without recursively including itself.
        (slim/'SHA_INDEX.json').unlink();write(slim/'SHA_INDEX.json',{p.name:sha(p) for p in slim.iterdir()})
        need(sum(p.stat().st_size for p in slim.iterdir())<=5<<20,'combined slim cap')
        write(out/'CLOSED.json',closure);return 0
    except Exception as e:
        ledgers={}
        for stage in ('extract','readout'):
            p=out/stage/'ledger.json'
            if p.is_file():ledgers[stage]=read(p)
        write(out/'STOP.json',{'error':repr(e),'stages':ledgers,'seconds':time.monotonic()-start,'no_retry':True});return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('plan');a.add_argument('--synthetic',action='store_true');a.add_argument('--manifest-sha',required=True);a.add_argument('--checkpoint');a.add_argument('--members');a.add_argument('--images');a.add_argument('--annotations');a.add_argument('--gpu-uuid');a.add_argument('--attempt-root');a.add_argument('--out',required=True)
    a=sub.add_parser('run');a.add_argument('--contract',required=True);a.add_argument('--authorization');a.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='plan':plan(a)
    else:raise SystemExit(execute(a))
