"""Local metadata preparation only. Does not connect, load models or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def ref(path, remote=None):
    p=Path(path);raw=p.read_bytes()
    return dict(path=str(p) if remote is None else remote,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def write(p,value):
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--terminal',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--proposal',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];term=Path(a.terminal)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    old=read(term/'receipts/formal-execution-003/contract.json')
    files=read(term/'SLIM_RETURN_VERIFIED.json')['files']
    index={r['relative']:dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256']) for r in files}
    task=old['job_root'].rsplit('/',2)[0]+'/r18-endpoint-support-20261023-001'
    bundle=task+'/bundle-001'
    selected=sorted(read(a.manifest)['records'],key=lambda r:r['stable_image_id'])[:32]
    assert ref(a.manifest)['sha256']==old['data']['development']['manifest']['sha256']
    names=['tools/r18_support_math.py','tools/r18_support_diagnostic.py','tools/r18_support_launch.py',
           'tools/prepare_r18_support.py','tests/test_r18_support.py']
    sources=[ref(root/n,bundle+'/'+n) for n in names]
    proposal_raw=Path(a.proposal).read_bytes();(out/'FROZEN_RULES.md').write_bytes(proposal_raw)
    result=subprocess.run(['wsl.exe','-e','/home/liu/localization-f0-cpu-20261009/bin/python','-B',
                          '/mnt/d/Lzy_CV/development/worktrees/r18-target-support/tests/test_r18_support.py','-v'],capture_output=True,text=True)
    (out/'CPU.stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'CPU.stderr.txt').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0, result.stderr
    write(out/'CPU_GATE.json',dict(status='PASS',tests=11,skips=0,torch='2.4.1+cpu',torchvision='0.19.1+cpu',
        real_images=False,detector_loaded=False,backward=False,GPU_calls=0,sources=sources,
        log=ref(out/'CPU.stderr.txt'),rules=ref(out/'FROZEN_RULES.md',bundle+'/FROZEN_RULES.md')))
    eps={arm:index[f"attempt-formal-{'001' if arm=='B0' else '003'}/train-{arm}/endpoint.json"] for arm in ('P','B0')}
    cps={}
    for arm in ('P','B0'):
        e=read(term/'receipts'/f"attempt-formal-{'001' if arm=='B0' else '003'}/train-{arm}/endpoint.json")['checkpoint']
        cps[arm]=dict(path=e['path'],bytes=e['size_bytes'],sha256=e['sha256'])
    histories=['attempt-formal-003/CLOSED.json','attempt-formal-003/RESULT.json','attempt-formal-003/COST_LEDGER.json',
               'formal-execution-003/EXECUTION_EXIT.json','formal-execution-003/COMPLETED_B0_AUTHENTICATED.json']
    histories += [r['relative'] for r in files if r['relative'].startswith('attempt-formal-001/') and r['relative'].endswith('STOP.json')]
    c=dict(schema='r18-endpoint-support-diagnostic-v1',id='r18-endpoint-support-20261023-001',
        contract_path=bundle+'/contract.json',output=task+'/attempt-diagnostic-001',precheck_output=task+'/visible-precheck-001',
        arms=['P','B0'],limits=dict(images=32,batch=4,forward_calls=16,GPU_seconds=600,allocated_GiB=18,slim_bytes=20*1024**2),
        policy='stable_id_ascending_first32_final_match_v1',selected_ids=[r['stable_image_id'] for r in selected],
        forbidden=['train','backward','AP','tuning','1602','retry','expand_images','DN'],
        gpu_uuid=old['gpu_uuid'],frozen_commit=old['commit'],frozen_release=old['source_root'],frozen_source=old['source'],
        parent_contract=index['formal-execution-003/contract.json'],endpoints=eps,checkpoints=cps,
        historical_receipts=[index[r] for r in histories],diagnostic_files=sources+[ref(out/'FROZEN_RULES.md',bundle+'/FROZEN_RULES.md')],
        CPU_gate=ref(out/'CPU_GATE.json',bundle+'/CPU_GATE.json'),
        diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        matcher=dict(head='final',weights=[2,5,2],alpha=.25,gamma=2,backend='original native matcher on CUDA FP32'),
        small='0<raw_area<=1024; training-valid clipped GT; not AP raster-filter population',
        trigger='reference width<1/W or height<1/H; strict; all ordinary query/layers',
        support='I/A O/A V/A; M=I/(I+O); pool heads before ratio; undefined remains null')
    write(out/'contract.json',c)
    with tarfile.open(out/'source.tar','w') as tf:
        for n in names:
            tf.add(root/n,arcname=n,recursive=False)
        for n in ('contract.json','CPU_GATE.json','FROZEN_RULES.md'):
            tf.add(out/n,arcname=n,recursive=False)
    write(out/'PREPARATION.json',dict(status='PASS',contract=ref(out/'contract.json'),archive=ref(out/'source.tar'),
        selected=selected,commit=c['diagnostic_commit'],remote_root=task,remote_bundle=bundle))
    print(json.dumps(dict(status='PASS',images=len(selected),remote_bundle=bundle,source=ref(out/'source.tar'))))


if __name__=='__main__':
    main()
