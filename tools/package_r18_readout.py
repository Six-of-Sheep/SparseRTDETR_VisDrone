"""Export committed readout-only source, manifest and commands, never execute."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=['tools/r18_readout_cli.py','tools/r18_readout_math.py',
       'docs/research/r18_reference_topic/CPU_DEFINITION.json',
       'docs/research/r18_reference_topic/LEARNING_CONFIG.json',
       'docs/research/r18_reference_topic/SELECTED_IDS.json']
def git(*a):return subprocess.check_output(['git',*a],cwd=root)
assert not git('status','--porcelain'), 'commit and clean worktree required'
head=git('rev-parse','HEAD').decode().strip()
out=Path(sys.argv[1]);out.mkdir()
index=[]
for path in files:
 b=git('show',head+':'+path);p=out/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 index.append({'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest={'commit':head,'files':index,'scope':'readout consumer only; no detector or feature extraction implementation',
          'GPU_authorized':False,'required_inputs':['selected192 features and fixed matches','CLOSED extraction receipt inheriting48 forwards/time','separate one-shot authorization bound to contract SHA']}
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'COMMANDS.txt').write_text('''# Review-only package. No command below has been run on real features.
# 1. After separately authorized, validated extraction, bind existing materials:
python -B tools/r18_readout_cli.py bind --selection docs/research/r18_reference_topic/SELECTED_IDS.json --features /ABS/features.pt --receipt /ABS/extraction-CLOSED.json --out /ABS/new-contract.json
# 2. Only after human authorization, create an authorization JSON bound to the
# exact contract SHA: {contract_sha256, authorize_once:true, scope:PR25_FROZEN_READOUT_ONLY}.
python -B tools/r18_readout_cli.py run --contract /ABS/new-contract.json --features /ABS/features.pt --authorization /ABS/authorization.json --out /ABS/unique-new-attempt
# Never reuse an attempt path. Require root CLOSED.json, not just slim RESULT.
# Return only slim/* (<=5MiB), root CLOSED/STOP and ledger. No checkpoints/features.
''')
print(json.dumps({'commit':head,'manifest_sha256':hashlib.sha256((out/'MANIFEST.json').read_bytes()).hexdigest(),'files':len(index)}))
