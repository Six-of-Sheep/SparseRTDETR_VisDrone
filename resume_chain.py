"""Single authorized remaining chain; never replays the historical launcher."""
import argparse,os,sys,time,traceback,subprocess
from pathlib import Path
import new_runtime as rt

SEED1=('accept-A','accept-B','accept-C','full-C','evaluation')
SEED2=('gpu-acceptance','formal','infer-A',*SEED1)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);a=p.parse_args(argv)
    from sparse_rtdetr.localization.runio import read,require,reference,publish,digest
    e=read(a.execution);root=Path(e['job_root']);start=time.time();seed=None;mode=None;done=[]
    require(e['execution_sha256']==digest({k:v for k,v in e.items() if k!='execution_sha256'}), 'resume execution digest differs')
    require(e['seeds']==[1,2] and e['no_retry'] and not e['holdout'] and e['kind']=='remaining_seed12_protocol_repair', 'resume workload differs')
    require(not any((root/n).exists() for n in ('CHAIN_STARTED.json','STOP.json','CHAIN_RESULT.json')), 'resume chain already used')
    require(read(root/'RESUME_CONSUMED.json')['execution_sha256']==e['execution_sha256'], 'resume credential missing')
    publish(root/'CHAIN_STARTED.json',{'started_epoch':start,'pid':os.getpid(),'seeds':[1,2],'source_sha256':e['source_sha256'],'execution_sha256':e['execution_sha256'],'original_launcher_replayed':False})
    try:
        from new_chain import phase
        for ref in e['contracts']:
            require(reference(ref['path'])==ref,'resume seed contract changed');c=rt.load(ref['path']);rt.install(c);seed=c['seed']
            require(bool(c.get('parent_reuse')) == (seed==1),'resume parent reuse scope differs')
            for mode in (SEED1 if seed==1 else SEED2):
                command=[c['python'],'-B',str(Path(__file__).with_name('new_entry.py')),'preflight','--contract',c['contract_path'],'--mode',mode]
                front=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True)
                publish(Path(c['job_root'])/(mode+'-PREFLIGHT.json'),{'status':'PASS' if front.returncode==0 else 'FAIL','returncode':front.returncode,'stdout':front.stdout.decode(),'stderr':front.stderr.decode(),'model_constructed':False,'CUDA_initialized':False})
                require(front.returncode==0,'actual resume/formal preflight failed: '+mode)
                phase(c,mode)
                publish(root/f'seed{seed}-{mode}-complete.json',{'seed':seed,'mode':mode,'result':reference(Path(c['job_root'])/mode/'RESULT.json'),'finished_epoch':time.time()})
            done.append({'seed':seed,'comparison':reference(Path(c['job_root'])/'evaluation/COMPARISON.json'),'result':reference(Path(c['job_root'])/'evaluation/RESULT.json')})
            publish(root/f'seed{seed}-COMPLETE.json',{'status':'FIXED_SEED_COMPLETE','seed':seed,'results':done[-1],'finished_epoch':time.time()})
        import aggregate_seeds
        summary=aggregate_seeds.run(e,root/'summary')
        publish(root/'CHAIN_RESULT.json',{'status':'FIXED_PAIRED_EXECUTION_COMPLETE','seeds':[1,2],'completed':done,'summary':summary,'started_epoch':start,'finished_epoch':time.time(),'elapsed_seconds':time.time()-start,'automatic_retry':False,'holdout_accessed':False,'seed1_retrained':False})
        import slim_delivery;slim_delivery.server_export(e)
        return 0
    except BaseException as error:
        publish(root/'STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','seed':seed,'mode':mode,'error':repr(error),'traceback':traceback.format_exc(),'started_epoch':start,'finished_epoch':time.time(),'completed':done,'automatic_retry':False})
        try:
            import slim_delivery;slim_delivery.server_export(e)
        except BaseException as failure:publish(root/'EXPORT_ERROR.json',{'error':repr(failure),'traceback':traceback.format_exc()})
        return 2

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    rt.install()
    raise SystemExit(main())
