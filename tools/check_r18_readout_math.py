"""Compare vectorized FP32 readout to the prior frozen scalar FP64 oracle."""
import ast
import json
from pathlib import Path
import resource
import torch
import r18_readout_math as m

resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
torch.set_num_threads(2)
root=Path(__file__).resolve().parents[1]
src=ast.parse((root/'tools/r18_probability_transport_cpu.py').read_text())
cfg=json.loads((root/'docs/research/r18_reference_topic/CPU_DEFINITION.json').read_text())
scope={'torch':torch,'CFG':cfg,'EPS':cfg['epsilon']}
names={'tensor','weights','grids','matrix','transport','update','decode','target_distribution','chain'}
# Read the frozen function definitions only, without executing its CLI globals.
tree=ast.Module(body=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
exec(compile(tree,'frozen_scalar_oracle','exec'),scope)
torch.manual_seed(9)
r=torch.tensor([[.5,.4,.2,.3],[.6,.3,.1,.2]],dtype=torch.float32)
g=m.grid(r);ng=m.grid(r*torch.tensor([1.,1.,1.2,1.3]))
p=torch.randn(2,4,33).softmax(-1)
v,_,_=m.transport(p,g,ng)
errors=[]
for b in range(2):
 for e in range(4):
  # FP32 normalized p rounded to FP64 need normalization for oracle's 1e-10 check.
  pd=p[b,e].double();pd=pd/pd.sum()
  torch.set_default_dtype(torch.float64)
  ref,_=scope['transport'](pd,g[b,e].double(),ng[b,e].double())
  torch.set_default_dtype(torch.float32)
  errors.append(float((ref-v[b,e].double()).abs().max()))
assert max(errors)<2e-6
q=torch.randn(1,3,2,256)*.01
for arm in ('C0','C1','M'):
 model=m.Readout(arm)
 with torch.no_grad():
  for head in model.heads:head.bias.copy_(torch.linspace(-.02,.02,132))
 state=model(q,r.unsqueeze(0))
 z=torch.stack([head(q[:,i]).reshape(1,2,4,33) for i,head in enumerate(model.heads)],1)
 for j in range(2):
  torch.set_default_dtype(torch.float64)
  expected=scope['chain'](z[0,:,j].detach().double(),r[j].double(),arm)
  torch.set_default_dtype(torch.float32)
  for layer in range(3):
   assert torch.allclose(state[layer]['box'][0,j].double(),expected[layer][2],atol=2e-6,rtol=0)
 mask=torch.ones(1,2,dtype=torch.bool)
 loss=m.loss(state,r.unsqueeze(0),mask);loss.backward()
 assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
 # Uniform replacement must not invoke ANY head or update a later state.
 if arm=='M':
  handles=[h.register_forward_hook(lambda *_: (_ for _ in ()).throw(AssertionError('extra head'))) for h in model.heads]
  before=[s['box'].clone() for s in state]
  m.local_uniform(state)
  assert all(torch.equal(x,s['box']) for x,s in zip(before,state))
  for h in handles:h.remove()
print(json.dumps({'status':'PASS','scope':'synthetic FP32 oracle and backward, no optimizer steps','max_transport_error':max(errors),'head_forward_equivalent':6,'backward':3,'detector_forward':0}))
