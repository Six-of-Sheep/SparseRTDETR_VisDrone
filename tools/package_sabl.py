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
m={'commit':head,'candidate_not_for_real_execution':a.candidate,'files':index,'bundle_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),'scope':'PR26 short acceptance only; formal training and scoring forbidden'}
(out/'MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
deploy=(root/'tools/r18_deploy.py').read_bytes() if a.candidate else git('show',head+':tools/r18_deploy.py')
(out/'deploy.py').write_bytes(deploy)
h=hashlib.sha256((out/'MANIFEST.json').read_bytes()).hexdigest()
commands=f"""# PR26 short GPU acceptance ONLY; no authorization token included.
python -B deploy.py --bundle source.tar.gz --manifest MANIFEST.json --sha256 {h} --out /ABS/UNIQUE_RELEASE
# Use the certified historical metadata record; no guessed material paths.
python -B tools/sabl_acceptance.py plan --manifest-sha {h} --metadata /ABS/CERTIFIED_H2_CONTRACT.json --gpu-uuid GPU-EXACT-UUID --attempt-root /ABS/UNIQUE_ATTEMPT --out /ABS/accept-contract.json
# Separate human authorization: scope=PR26_SHORT_GPU_ACCEPTANCE_ONLY, contract=canonical JSON SHA256, authorize_once=true.
python -B tools/sabl_acceptance.py run --contract /ABS/accept-contract.json --authorization /ABS/authorization.json --out /ABS/UNIQUE_ATTEMPT
# MUST exit after 9 actual updates/18 forwards and root CLOSED/EXIT. No formal training command.
"""
(out/'COMMANDS.txt').write_bytes(commands.encode())
print(json.dumps({'commit':head,'candidate':a.candidate,'manifest_sha256':h,'files':len(index),'source_bytes':sum(x['bytes'] for x in index)}))
