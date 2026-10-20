"""Actual admission CLI from uncompleted raw spec, no GPU or model forward."""
import argparse,copy,json,os,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    import torch
    rt.require(torch.version.cuda is None and not torch.cuda.is_initialized(),'CPU-only test environment required')
    src=rt.source_receipt(out/'source.json') if a.source is None else rt.reference(a.source.resolve())
    sample=rt.fixture_sample();data={}
    for role in ('train_core','development'):
        m=out/(role+'.json');rt.publish(m,dict(role=role,records=[]))
        data[role]=dict(role=role,count=6,manifest=rt.reference(m),roots=dict(role=role,images=str(out/role),annotations=str(out/role)))
    rt.publish(out/'prior.json',dict(task_id=rt.TASK,profile='CPU_FIXTURE_ONLY',GPU_seconds=0,forward_calls=0))
    raw=dict(profile='CPU_FIXTURE_ONLY',seed=0,execution_scope='GPU_ACCEPTANCE_ONLY',
        attempt='attempt-cpu-admission',job_root=str(out/'attempt-cpu-admission'),nonce=uuid.uuid4().hex,
        boot_id=sample['boot_id'],gpu_uuid='CPU_FIXTURE_NO_GPU',
        storage_identity={k:sample['storage'][k] for k in ('canonical_mount','mount','st_dev')},data=data,
        pretrained=src,prior_cost=rt.reference(out/'prior.json'),baselines=src,fixed_ids=[])
    rt.require('resources' not in raw and 'source' not in raw and 'stages' not in raw,'test raw spec prematurely filled')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    entry=ROOT/'r18_sampling_admission.py';checks=[]
    def call(name,c,reason=None,observation=None):
        spec=out/(name+'.json');rt.publish(spec,c)
        command=[sys.executable,'-B',str(entry),'check','--spec',str(spec),'--source',src['path'],'--output',str(out/name)]
        if observation is not None:
            p=out/(name+'-observation.json');rt.publish(p,observation);command+=['--resource-observation',str(p)]
        r=recorded_subprocess(out/(name+'-process'),command,cwd=ROOT,env=env)
        rt.require((r.returncode==0)==(reason is None),'admission CLI unexpected exit: '+name)
        if reason:rt.require(reason in r.stderr.decode(),'admission rejection message: '+name)
        checks.append(name)
    call('normal',raw)
    contract=rt.read(out/'normal-contract.json')
    rt.require(contract['resources']==rt.execution()['resources'],'production constructor did not bind resources')
    checks.append('resources from frozen execution via actual constructor')
    for field in ('data','boot_id','prior_cost','storage_identity','pretrained','gpu_uuid'):
        bad=copy.deepcopy(raw);bad.pop(field);call('missing-'+field,bad,'raw acceptance spec missing fields:')
    for field,value in (('resources',{}),('task_id','foreign'),('source',dict(src,sha256='0'*64)),('protocol',rt.PROTOCOL),('stages',['admission'])):
        bad=copy.deepcopy(raw);bad[field]=value
        call('conflict-'+field,bad,'acceptance entry refuses' if field=='protocol' else 'raw acceptance spec conflicts')
    bad=copy.deepcopy(raw);bad['boot_id']='other';call('boot-identity',bad,'boot identity differs')
    bad=copy.deepcopy(raw);bad['storage_identity']['st_dev']=999;call('storage-identity',bad,'storage identity differs')
    bad=copy.deepcopy(raw);bad['data']['train_core']['role']='development';call('dataset-identity',bad,'dataset role/count mapping differs')
    bad=copy.deepcopy(raw);bad['seed']=1;call('seed-identity',bad,'only fixed seed0 admitted')
    low=rt.fixture_sample();low['snapshot']['resources']['ram_available']=0
    call('resource-shortage',raw,'CPU/RAM admission failed',low)
    r=recorded_subprocess(out/'wrong-order-process',[sys.executable,'-B',str(entry),'screen','--contract',str(out/'normal.json')],cwd=ROOT,env=env)
    rt.require(r.returncode!=0 and b'resource screen requires constructed contract' in r.stderr,'raw resource screen was not refused');checks.append('CLI wrong call order')
    r=recorded_subprocess(out/'right-order-process',[sys.executable,'-B',str(entry),'screen','--contract',str(out/'normal-contract.json')],cwd=ROOT,env=env)
    rt.require(r.returncode==0,'constructed screen refused');checks.append('CLI constructed screen')
    rt.require(not torch.cuda.is_initialized(),'CPU checks initialized CUDA')
    rt.publish(out/'CPU_GATE.json',dict(status='ADMISSION_CPU_ENTRY_PASS',source=src,checks=checks,checks_run=len(checks),skips=0,
        CUDA_initialized=False,model_forward_calls=0,actual_raw_resources_absent=True,metadata_only_not_real_material_GPU_acceptance=True))
    print(json.dumps(dict(status='ADMISSION_CPU_ENTRY_PASS',checks_run=len(checks),skips=0)))
if __name__=='__main__':main()
