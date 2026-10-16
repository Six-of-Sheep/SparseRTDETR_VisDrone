"""Only new lifecycle/readout interfaces; mathematical 42 checks reused."""
import copy,json,runpy
from pathlib import Path
import torch,pytest
from sparse_rtdetr.protocol_r18.endpoint import compare_metrics,verify_cache
from sparse_rtdetr.protocol_r18.model import ROOT,ARM
from sparse_rtdetr.protocol_r18.control import main
from sparse_rtdetr.protocol_r18.lifecycle import MODES
torch.set_num_threads(2)

def metrics():
    return {'COCO_bridge':{'units':'0..1 AP, -1 means no GT in an area/class aggregation','maxDets':[1,10,100],'iouThrs':[.5,.75],'areaRng':[[0,1e10]],
        'metrics':{'AP':.3,'APS':.2,'AP75_small':.1,'APL':-1.}},'official':{'AP':31.2,'units':'percent'}}

def test_comparison_real_unit_once_and_official_separate():
    r=metrics();n=copy.deepcopy(r);n['COCO_bridge']['metrics']['APS']+=.002
    value=compare_metrics(n,r)
    assert abs(value['COCO_bridge']['APS']['native_minus_R0_pp']-.2)<1e-12
    assert value['COCO_bridge']['APL']['native_minus_R0_pp'] is None
    assert value['official_native']==n['official'] and value['official_R0']==r['official'] and value['causal_attribution_to_single_structure_change'] is False

@pytest.mark.parametrize('field',['units','maxDets','iouThrs','areaRng'])
def test_comparison_wrong_metric_convention_refused(field):
    r=metrics();bad=copy.deepcopy(r);bad['COCO_bridge'][field]='wrong'
    with pytest.raises(RuntimeError):compare_metrics(bad,r)

def cache_fixture():
    rows=[{'stable_image_id':f'{i:064x}'} for i in range(548)]
    raw={'schema':'native_r18_common_raw_queries_v1','arm':ARM,'epoch':120,'EMA':True,'image_ids':[r['stable_image_id'] for r in rows],
        'pred_logits':torch.zeros(548,300,10),'pred_boxes':torch.tensor([.5,.5,.2,.2]).expand(548,300,4).clone(),'query_ids':torch.arange(300),
        'checkpoint_sha256':'a'*64,'source_sha256':'b'*64,'contract_sha256':'c'*64,'evaluation_protocol_sha256':'d'*64}
    endpoint={'checkpoint':{'sha256':'a'*64}};c={'source_sha256':'b'*64,'contract_sha256':'c'*64,'evaluation_protocol_sha256':'d'*64}
    return raw,endpoint,c,rows

def test_native_548_cache_schema_dtype_identity_pass():
    verify_cache(*cache_fixture())

@pytest.mark.parametrize('field,value',[('arm','R0'),('epoch',119),('checkpoint_sha256','0'*64),('contract_sha256','0'*64),('evaluation_protocol_sha256','0'*64)])
def test_wrong_endpoint_cache_identity_refused(field,value):
    raw,end,c,rows=cache_fixture();raw[field]=value
    with pytest.raises(RuntimeError):verify_cache(raw,end,c,rows)

def test_original_image_coordinates_and_rank_no_nms():
    from sparse_rtdetr.r0.evaluation import rank_candidates
    logits=torch.zeros(300,10);boxes=torch.tensor([.5,.5,.2,.4]).expand(300,4).clone();row={'stable_image_id':'a'*64,'width':1000,'height':500}
    p=rank_candidates(logits,boxes,row,600)
    assert len(p)==600 and [r['query_id'] for r in p[:11]]==[0]*10+[1] and [r['category_id'] for r in p[:10]]==list(range(1,11))
    assert p[0]['bbox_xyxy']==[400.,150.,600.,350.]

def test_new_entry_mp_main_does_not_dispatch_or_init_CUDA():
    r=runpy.run_path(str(ROOT/'tools/protocol_r18/deploy.py'),run_name='__mp_main__')
    assert r['_dispatch_started'] is False and not torch.cuda.is_initialized() and MODES==('gpu-acceptance','formal','inference','evaluation','comparison')

def test_old_preparation_identity_not_accepted_by_new_entry(tmp_path):
    from sparse_rtdetr.localization.runio import digest
    c={'task_id':'rtdetrv2-r18-common-protocol-prep-20261013-001','arm':ARM};c['contract_sha256']=digest(c);p=tmp_path/'old.json';p.write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='task/model'):main(['worker','--contract',str(p),'--mode','gpu-acceptance'])

def test_actual_native_CPU_forward_readout_two_evaluators_synthetic_only():
    from sparse_rtdetr.protocol_r18.model import Config,build
    from sparse_rtdetr.r0.evaluation import rank_candidates,score_budget
    model,_,_=build(Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32'),synthetic=True);model.eval()
    with torch.inference_mode():output=model(torch.rand(1,3,128,128))
    row={'stable_image_id':'a'*64,'width':128,'height':128}
    parsed={'rows':[{'id':'b'*64,'box':[40.,40.,60.,60.],'category':1,'score':1,'raw_area':400.}]}
    for budget in (600,300):
        predictions=rank_candidates(output['pred_logits'][0],output['pred_boxes'][0],row,budget)
        result=score_budget([row],[parsed],predictions,ROOT)
        assert result['COCO_bridge']['units'].startswith('0..1') and result['official']['evaluated_image_count']==1 and result['official_APS'] is None
        assert all(__import__('math').isfinite(v) for v in result['COCO_bridge']['metrics'].values())
    assert not torch.cuda.is_initialized()
