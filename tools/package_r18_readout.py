"""Export full committed source and strict deployment tools; no execution."""
import argparse,hashlib,io,json,subprocess,tarfile,os,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def git(*a):
 prefix=['git']
 link=root/'.git'
 if os.name=='posix' and link.is_file():
  location=link.read_text().strip().removeprefix('gitdir: ')
  if re.match(r'^[A-Za-z]:/',location):
   location='/mnt/'+location[0].lower()+location[2:]
   prefix+=['--git-dir='+location,'--work-tree='+str(root)]
 return subprocess.check_output([*prefix,*a],cwd=root)
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--candidate',action='store_true');a=p.parse_args()
if not a.candidate:assert not git('status','--porcelain'), 'commit and clean worktree required'
head=git('rev-parse','HEAD').decode().strip()
names=git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0') if a.candidate else git('ls-tree','-r','--name-only','-z',head).decode().split('\0')
names=sorted(set(n for n in names if n));out=Path(a.out);out.mkdir();index=[]
with tarfile.open(out/'source.tar.gz','w:gz') as tar:
 for path in names:
  b=(root/path).read_bytes() if a.candidate else git('show',head+':'+path)
  info=tarfile.TarInfo(path);info.size=len(b);info.mode=0o644;tar.addfile(info,io.BytesIO(b))
  index.append({'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
m={'commit':head,'candidate_not_for_real_execution':a.candidate,'files':index,'bundle_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),'scope':'PR25 full extraction and frozen readout; no real execution authorization'}
(out/'MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
deploy=(root/'tools/r18_deploy.py').read_bytes() if a.candidate else git('show',head+':tools/r18_deploy.py')
(out/'deploy.py').write_bytes(deploy)
h=hashlib.sha256((out/'MANIFEST.json').read_bytes()).hexdigest()
commands=f'''# Review only. No GPU authorization is included.
python -B deploy.py --bundle source.tar.gz --manifest MANIFEST.json --sha256 {h} --out /ABS/UNIQUE_RELEASE
# In UNIQUE_RELEASE; only role-bound train_core roots are accepted:
python -B tools/r18_full_chain.py plan --manifest-sha {h} --checkpoint /ABS/CERTIFIED_B0_E120.pth --members /ABS/train_core_manifest.json --images /ABS/train_core/images --annotations /ABS/train_core/annotations --gpu-uuid GPU-EXACT-UUID --attempt-root /ABS/UNIQUE_ATTEMPT --out /ABS/chain-contract.json
# One separately authorized record: chain_contract=canonical SHA256 of chain-contract.json,
# authorize_once=true, scope=PR25_FULL_FROZEN_EXPLORATION. Then ONE launch:
python -B tools/r18_full_chain.py run --contract /ABS/chain-contract.json --authorization /ABS/authorization.json --out /ABS/UNIQUE_ATTEMPT
# Automatic extraction -> bind -> C0/C1/M -> closure. STOP has no automatic retry.
# Require root CLOSED.json. Return slim/* and root CLOSED/STOP only (5MiB).
'''
(out/'COMMANDS.txt').write_bytes(commands.encode())
print(json.dumps({'commit':head,'candidate':a.candidate,'manifest_sha256':h,'files':len(index),'source_bytes':sum(x['bytes'] for x in index)}))
