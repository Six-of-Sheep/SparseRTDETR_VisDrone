"""Server-native R25 immutable export and single diagnostic launch. No formal training route."""
import argparse,hashlib,json,os,subprocess,sys,time,tarfile,shutil
from pathlib import Path
T=Path(__file__).resolve().parents[2]
W=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace')
R=W/'experiments/localization-distribution/r25-query-local-regression-20261104-001'
PY='/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python'
TASK_SHA='ec7ef314720179855ca8c6a74be4e37f65495891d5f0227f5736e78aabe71241'
GPU='GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=1)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','launch']);p.add_argument('--release',required=True);p.add_argument('--attempt');p.add_argument('--cpu-gate');a=p.parse_args()
    rel=R/a.release
    if a.action=='freeze':
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=T).strip(),'dirty worktree'
        assert sha(R/'TASK_R25_001.md')==TASK_SHA
        rel.mkdir(exist_ok=False);source=rel/'source';source.mkdir()
        archive=subprocess.check_output(['git','archive','HEAD'],cwd=T)
        import io
        with tarfile.open(fileobj=io.BytesIO(archive)) as f:f.extractall(source,filter='data')
        rows=[{'path':str(x.relative_to(source)),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(source.rglob('*')) if x.is_file()]
        v={'version':'R25-001-source-manifest-v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=T).decode().strip(),'task_sha256':TASK_SHA,
            'source_files':rows,'phase_scope':'CPU and 64train-image learning response only; no development AP or formal three-arm training'}
        write(rel/'MANIFEST.json',v);print(json.dumps({'release':str(rel),'manifest_sha256':sha(rel/'MANIFEST.json'),'commit':v['commit'],'files':len(rows)}))
    else:
        assert a.attempt and a.cpu_gate
        gate=Path(a.cpu_gate);assert json.loads((gate/'CLOSED.json').read_text())['status']=='PASS'
        assert json.loads((gate/'SOURCE_BINDING.json').read_text())['sha256']==sha(rel/'MANIFEST.json')
        assert sha(R/'TASK_R25_001.md')==TASK_SHA
        out=R/a.attempt;log=R/(a.attempt+'.log');receipt=R/(a.attempt+'-LAUNCH.json')
        assert not out.exists() and not log.exists() and not receipt.exists(),'duplicate attempt'
        # Only the intended GPU and no existing detector training process.
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader']).decode()
        assert GPU in devices and float(devices.split(',')[1].split()[0])>21000,'insufficient currently available VRAM'
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).decode().strip()
        assert not apps,'other GPU compute processes present'
        assert shutil.disk_usage(R).free>16*1024**3,'insufficient next-stage disk'
        active=[]
        for d in Path('/proc').iterdir():
            if not d.name.isdigit():continue
            try:
                args=(d/'cmdline').read_bytes().split(b'\0')
                if b'diagnostic' in args and any(x.endswith(b'r25_run.py') for x in args):active.append(d.name)
            except (OSError,PermissionError):pass
        assert not active,'existing R25 diagnostic process'
        tmp=Path('/tmp')/('r25-'+a.attempt);tmp.mkdir(mode=0o700,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',MKL_THREADING_LAYER='GNU',
                 PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        argv=[PY,'-B',str(rel/'source/tools/query_local/r25_run.py'),'diagnostic','--out',str(out),'--manifest',str(rel/'MANIFEST.json')]
        with log.open('x') as f:child=subprocess.Popen(argv,cwd=rel/'source',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        v={'status':'STARTED','pid':child.pid,'epoch':time.time(),'argv':argv,'cwd':str(rel/'source'),'manifest_sha256':sha(rel/'MANIFEST.json'),
          'task_sha256':TASK_SHA,'cpu_gate':str(gate),'temporary_directory':str(tmp),'resources_before':devices,'no_artificial_resource_caps':True,
          'workload':'A1/A2 each128 effective updates + one independently counted restore replay each + same64-image four-update timing per arm; no AP/formal route'}
        write(receipt,v);print(json.dumps(v))
if __name__=='__main__':main()
