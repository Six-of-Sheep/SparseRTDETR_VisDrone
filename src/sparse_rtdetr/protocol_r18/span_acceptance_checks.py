"""Small CPU regressions supplementing the actual CLI lifecycle check."""
import copy
from pathlib import Path
import torch
from . import span_runtime as rt, span_budget as budget, span_checkpoint as cp
from .span_model import build,validate_sampling
from .span_observation import bilinear,observe
from .model import Config
from ..baseline import training_v2b as native
from ..research.sampling_span import MSDeformableAttention,from_baseline


def check_interfaces(c,out):
    checks=[]
    def check(name,value):
        if not value:raise RuntimeError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('not rejected: '+name)
    # Equal stencil can still change interpolation weights and sampled output.
    a=torch.tensor([[[[[.56,.56]]]]]);b=torch.tensor([[[[[.57,.57]]]]])
    sa,wa,_=bilinear(a,(4,4));sb,wb,_=bilinear(b,(4,4))
    check('same stencil unequal bilinear weights',torch.equal(sa,sb) and not torch.equal(wa,wb))
    from ..baseline.training_v2b_deterministic_sampling import bilinear_grid_sample
    value=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
    check('same stencil unequal actual interpolation output',not torch.equal(
        bilinear_grid_sample(value,a.reshape(1,1,1,2)*2-1),bilinear_grid_sample(value,b.reshape(1,1,1,2)*2-1)))
    checks+=check_observer_purity()
    # Real append-only journal consumer: no full-file rescans per monitor tick.
    journal_path=out/'incremental-events.jsonl'
    original=(rt.stage_path(c,'admission')/'events.jsonl').read_bytes()
    lines=original.splitlines(keepends=True)
    writer=rt.verify_events(c,'admission',rt.stage_path(c,'admission')/'events.jsonl')[0]['pid']
    reader=rt.EventReader(c,'admission',journal_path)
    check('event reader initial absent',reader.poll(writer) is None)
    journal_path.write_bytes(lines[0]+lines[1][:20])
    check('event reader partial tail not progress',reader.poll(writer)['number']==1 and reader.offset==len(lines[0]))
    with journal_path.open('ab') as stream:stream.write(lines[1][20:]+b''.join(lines[2:]))
    check('event reader complete append',reader.poll(writer)['number']==len(lines) and reader.offset==len(original))
    check('event reader unchanged cursor',reader.poll(writer)['number']==len(lines) and reader.offset==len(original))
    with journal_path.open('ab') as stream:stream.write(lines[-1])
    reject('event reader duplicate refused',lambda:reader.poll(writer))
    journal_path.write_bytes(lines[0]);reject('event reader truncation refused',lambda:reader.poll(writer))
    journal_path.unlink();reject('event reader disappearance refused',lambda:reader.poll(writer))
    journal_path.write_bytes(original);reader=rt.EventReader(c,'admission',journal_path)
    reject('event reader foreign PID refused',lambda:reader.poll(writer+1))
    reader=rt.EventReader(c,'admission',journal_path);reader.poll(writer)
    replacement=out/'replacement-events.jsonl';replacement.write_bytes(original);replacement.replace(journal_path)
    reject('event reader replacement refused',lambda:reader.poll(writer))
    cfg=Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32')
    model,_,_=build(cfg,'P',synthetic=True)
    validate_sampling(model,'P')
    reject('legacy guard still rejects P',lambda:native.validate_model_sampling(model,sampling_backend='deterministic_gather'))
    model.decoder.decoder.layers[0].cross_attn.span_enabled=False
    reject('candidate disabled identity rejected',lambda:validate_sampling(model,'P'))
    del model
    from .span_worker import data,parsed
    rows,reader=data(c,'development')
    bad=dict(rows[0],image_sha256='0'*64)
    reject('reader wrong member metadata',lambda:reader.bound_bytes(bad))
    # Validate actual byte SHA guard using the ORIGINAL production primitive.
    from ..localization.data import read_bound
    row=rows[0];path=Path(c['data']['development']['roots']['images'])/Path(row['relative_path']).name
    reject('reader wrong content SHA',lambda:read_bound(path,'f'*64,row['image_size_bytes']))
    reject('reader missing file',lambda:read_bound(path.with_name('NOT_PRESENT.png'),row['image_sha256'],row['image_size_bytes']))
    # Identical real evaluator interfaces see the same physical rows/ignore.
    from ..r0.evaluation import primary_inputs,bridge_inputs
    truth=[parsed(reader,r) for r in rows]
    pred=rt.exact(rt.verify_stage(c,'infer-P')['predictions']['600'])
    pi=primary_inputs(rows,truth,pred);_,_,counts=bridge_inputs(rows,truth,pred)
    check('parsed GT to both actual evaluator inputs',len(pi.ground_truth)==sum(len(p['rows']) for p in truth)
          and counts==rt.exact(rt.verify_stage(c,'score')['results']['P']['600'])['spatial_filter_counts'])
    from .span_worker import live
    journal=out/'journal-interface';journal.mkdir();sample=rt.fixture_sample()
    rt.publish(journal/'resource-entry.json',sample)
    (journal/'resources.jsonl').write_bytes(rt.canonical(sample)+b'\n{"partial":')
    check('resource last complete append record',live(c,journal,gpu=False)['fixture_only'])
    stale=copy.deepcopy(sample);stale['sampled_epoch']=0
    (journal/'resources.jsonl').write_bytes(rt.canonical(stale)+b'\n')
    reject('stale last complete record refuses',lambda:live(c,journal,gpu=False))
    check('prepare refreshes resource sample',rt.read(out/'prepare-success/resource-live.json')['sampled_epoch']>
          rt.read(out/'prepare-success/resource-entry.json')['sampled_epoch'])
    for arm in ('B0','B1','P'):
        stage=rt.stage_path(c,'accept-'+arm)
        events=rt.verify_events(c,'accept-'+arm,stage/'events.jsonl')
        observations=events[2]['fields']['result']['observations']
        check(arm+' separate normal/DN observations',len(observations)==6 and
              {r['group'] for r in observations}=={'ordinary','DN'} and all(not r['uses_GT'] for r in observations))
        check(arm+' sampling source never remains installed',rt.verify_stage(c,'accept-'+arm)['full_state_RNG_exact'])
    # Fixed checkpoint and augmentation-stage policy is mechanical, not selected by loss/AP.
    check('120 epoch strong weak boundary fixed',rt.recipe()['training']['augmentation']=='unchanged_common_strong_e001_e117_weak_e118_e120')
    check('bounded checkpoint schedule',rt.execution()['checkpoint_epochs']==[0,1,40,80,117,118,119,120])
    from .span_worker import training_dataset
    ds=training_dataset(c)
    check('actual augmentation transition 117 to118',ds[(117,0,1)][2]['augmentation_strong'] is True
          and ds[(118,0,1)][2]['augmentation_strong'] is False)
    # Ledger pure-function boundary; no CUDA allocation/device APIs involved.
    ledger=dict(completed_GPU_seconds=100,active=dict(started=10))
    check('cumulative stage walltime includes prior',budget.seconds(ledger,now=25)==115)
    root=out/'budget-fixture';root.mkdir()
    bc=copy.deepcopy(c);bc.update(profile='LAB_GPU_ONCE',job_root=str(root))
    log=dict(completed_GPU_seconds=60*3600-170,forward_calls=0,active=None,stages=[])
    rt.atomic(root/'COST_LEDGER.json',log);saved=[]
    reject('60h reserve saves and stops',lambda:budget.guard(bc,save=lambda:saved.append('saved') or {'synthetic':True}))
    check('budget safe save happened before STOP',saved==['saved'] and (root/'BUDGET_STOP.json').exists())
    log.update(completed_GPU_seconds=0,forward_calls=219693,active=dict(started=__import__('time').monotonic()))
    rt.atomic(root/'COST_LEDGER.json',log)
    reject('forward overspend rejected before compute',lambda:budget.guard(bc,forward_calls=1))
    from unittest.mock import patch
    log.update(completed_GPU_seconds=0,forward_calls=0)
    rt.atomic(root/'COST_LEDGER.json',log)
    with patch.object(budget,'storage_bytes',return_value=98*rt.GIB):
        reject('100GiB checkpoint reserve stops',lambda:budget.guard(bc))
    for stage in rt.STAGES:
        directory=rt.stage_path(c,stage);record=rt.read(directory/'supervisor-exit.json')
        check(stage+' separate command stdout stderr rc',bool(record['command']) and record['returncode']==0
              and record['stdout']==rt.reference(directory/'worker.stdout') and record['stderr']==rt.reference(directory/'worker.stderr'))
    return checks


def check_observer_purity():
    checks=[]
    def check(name,value):
        if not value:raise RuntimeError(name)
        checks.append(name)
    from types import SimpleNamespace
    for dtype in ('float32','bfloat16'):
        leaf=from_baseline(MSDeformableAttention(256,8,3,[4,4,4]),'P');clone=copy.deepcopy(leaf)
        query=torch.randn(1,304,256);value=torch.randn(1,21,256)
        refs=torch.tensor([.5,.5,.01,.01]).repeat(1,304,1,1)
        gradients=[];outputs=[];obs=[]
        for i,m in enumerate((leaf,clone)):
            fake=SimpleNamespace(decoder=SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(cross_attn=m)])))
            q=query.clone().requires_grad_();v=value.clone().requires_grad_()
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=dtype=='bfloat16'):
                with observe(fake,'P',obs) if i else __import__('contextlib').nullcontext():
                    result=m(q,refs,v,[(4,4),(2,2),(1,1)])
            result.float().square().sum().backward();outputs.append(result.detach())
            gradients.append([q.grad,v.grad,*[p.grad for p in m.parameters()]])
        check(dtype+' observer exact output and every gradient',torch.equal(*outputs)
              and all(x is not None and y is not None and torch.equal(x,y) for x,y in zip(*gradients)))
        check(dtype+' offset projection gradient retained',float(gradients[1][2].abs().sum())>0)
    return checks
