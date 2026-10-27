"""Native full-model CPU construction, not real N weights."""
import json,sys
from pathlib import Path
from r18_readout_cli import torch_setup,write
t=torch_setup(True)
from r18_extract import construct,capture,state_digest
model,matcher,config=construct(True)
pixels=t.linspace(0,1,3*128*128).reshape(1,3,128,128)
before=state_digest(model)
with t.no_grad():off=model(pixels)
with t.no_grad(),capture(model) as observation:on=model(pixels)
assert t.equal(off['pred_boxes'],on['pred_boxes']) and t.equal(off['pred_logits'],on['pred_logits'])
assert state_digest(model)==before and len(observation['layers'])==3
assert all(p.grad is None and not p.requires_grad for p in model.parameters())
assert all(not m.training for m in model.modules())
assert observation['r'].shape==(1,300,4)
assert not t.equal(observation['r'],on['pred_boxes'])  # detects replacing r0 with final ref
write(sys.argv[1],{'status':'PASS','model':'native common R18 synthetic initialization','input':[1,3,128,128],
 'CPU_detector_forward':2,'on_off_outputs_bitwise_equal':True,'parameters_buffers_unchanged':True,
 'real_weights_loaded':False,'real_weight_on_off_check':False,'cuda_initialized':t.cuda.is_initialized()})
