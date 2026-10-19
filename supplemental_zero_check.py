"""Affected CPU checks: real1602 input worker + reviewed rows + development.

No model predictions, no full-set AP. Small dual-evaluator fixtures are driven
by the actual launcher in supplemental_entry_check, not by fake event writers.
"""
import copy
from pathlib import Path
import subprocess
import sys

import supplemental_runtime as rt
from supplemental_runtime import require, exact, read, publish, reference, digest, ROOT


def check_real_inputs(template, context, base):
    from supplemental_data import dataset, extension
    from supplemental_job import environment, verify_phase
    from supplemental_entry_check import tree
    from confirmation_prepare import evaluator_inputs
    from sparse_rtdetr.data_protocol.supplemental_zero_gt import SupplementalZeroGT, specification, ROWS
    from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.localization.data import parse_raw, read_bound
    from tools.zero_area_gt_review import coco_counts
    # Same final launcher/supervisor/worker on all1602 original raw inputs.
    c = copy.deepcopy(template)
    c.update(synthetic=False, CPU_input_only=True, sealed_access_authorized=True,
             authorization=context['authorization'], evaluation_data=context['evaluation_data'],
             job_root=str(base/'real-inputs'), runtime_root=str(Path('/home/lyy/.cache')/rt.TASK/base.name/'real-inputs'),
             contract_path=str(base/'real-inputs-contract.json'))
    token=base/'real-inputs-token.json'
    publish(token, dict(task_id=rt.TASK,purpose=rt.PURPOSE,source_sha256=c['source_sha256'],job_root=c['job_root']))
    c['launch_token']=reference(token);c['contract_sha256']=digest(c);publish(c['contract_path'],c)
    env={**environment(c),'SUPPLEMENTAL_PYTHON':sys.executable}
    argv=['bash',str(ROOT/'supplemental_launch.sh'),c['contract_path']]
    with (base/'real-inputs.log').open('xb') as f:
        result=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT)
    require(result.returncode==0,'real1602 input launcher failure; inspect real-inputs.log')
    proof=verify_phase(c,0,'input-check')
    require(proof['members']==1602 and proof['physical_rows']==83478 and proof['admitted_zero_rows']==2 and not proof['AP_scored'],'real1602 coverage')
    root=Path(c['job_root']);before=tree(root)
    for action in ('start','run'):
        command=argv if action=='start' else [sys.executable,'-B',str(ROOT/'supplemental_job.py'),'run','--contract',c['contract_path']]
        with (base/('real-inputs-duplicate-'+action+'.log')).open('xb') as f:
            r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        require(r.returncode!=0 and tree(root)==before,'real1602 duplicate changed evidence')
    require(read(root/'CLOSED.json')['status']=='PASS' and read(root/'EXIT.json')['returncode']==0 and read(root/'SLIM_EXPORT.json')['status']=='READY_FOR_SLIM_RETURN','real input terminal/export/exit')
    # Authentication and scope refusals. No original file is changed.
    _,rows,parsed=dataset(context)
    policy=extension(context,rows,parsed)
    checks=[]
    def rejects(name,call):
        try:call()
        except (RuntimeError,ValueError,KeyError):checks.append(name)
        else:raise RuntimeError('invalid GT extension admitted: '+name)
    for name in ('negative','nonfinite','fractional','wrong_line','wrong_file_SHA','wrong_image','wrong_area','unknown_zero'):
        rr=copy.deepcopy(rows);pp=copy.deepcopy(parsed)
        ix=next(i for i,r in enumerate(rr) if r['stable_image_id']==ROWS[0][1]);o=pp[ix]['rows'][ROWS[0][3]-1]
        if name=='negative':o['box'][2]=o['box'][0]-1
        elif name=='nonfinite':o['box'][1]=float('nan')
        elif name=='fractional':o['box'][0]+=.5
        elif name=='wrong_line':o['raw_line_sha256']='0'*64
        elif name=='wrong_file_SHA':rr[ix]['annotation_sha256']='0'*64
        elif name=='wrong_image':rr[ix]['stable_image_id']='0'*64
        elif name=='wrong_area':o['raw_area']=1
        else:pp[0]['rows'][0]['box']=[1.,1.,2.,1.];pp[0]['rows'][0]['raw_area']=0
        rejects(name,lambda:SupplementalZeroGT.bind(rr,pp,specification()))
    for name,box in (('prediction_zero',[1.,1.,2.,1.]),('prediction_negative',[2.,1.,1.,2.]),('prediction_nonfinite',[1.,1.,2.,float('inf')])):
        predicts=[dict(image_id=rows[0]['stable_image_id'],category_id=4,score=.9,bbox_xyxy=box)]
        rejects(name,lambda:evaluator_inputs(rows,parsed,predicts,600,gt_extension=policy))
    rejects('default_primary_remains_positive',lambda:primary._validate_input(primary_inputs(rows,parsed,[])))
    bad=specification();bad['manifest_sha256']='0'*64
    rejects('wrong_protocol_manifest',lambda:SupplementalZeroGT.bind(rows,parsed,bad))
    # Two isolated authentic rows: exact denominator, 22 pixels, .5 boundary.
    local=[]
    for z in ROWS:
        ix=next(i for i,r in enumerate(rows) if r['stable_image_id']==z[1]);row=rows[ix]
        original=parsed[ix];obj=original['rows'][z[3]-1]
        one=dict(gt=[],spatial_ignore=[],class_ignore=[],rows=[copy.deepcopy(obj)],counts={'nonpositive_area':1})
        pol=SupplementalZeroGT.bind([row],[one],specification())
        ds,det,count=bridge_inputs([row],[one],[],gt_extension=pol)
        value=primary_inputs([row],[one],[]);primary._validate_input(value,gt_extension=pol)
        image=value.images[0];regions=[g for g in value.ground_truth if g.category_id==0]
        integral=primary._build_integral(image,regions,gt_extension=pol)
        x,y,x2,_=z[5]
        predicted=[dict(image_id=row['stable_image_id'],category_id=4,score=.9,bbox_xyxy=[x,y,x2,y+1])]
        ds,det,count=bridge_inputs([row],[one],predicted,gt_extension=pol)
        result=primary.evaluate_primary_v1(primary_inputs([row],[one],predicted),primary.primary_evaluator_contract_binding(ROOT),gt_extension=pol)
        if z[6]==4:
            require(count['GT_after']==1 and result.match_counts_by_class_iou_max_dets[3][0][3]==(1,0,1,0),'zeroGT denominator/matching')
        else:
            require(float(integral[-1,-1])==22 and count['detections_after']==0,'22pixel ignore')
            boundary=[dict(predicted[0],bbox_xyxy=[x,y-1,x2,y+3])]
            _,dd,cc=bridge_inputs([row],[one],boundary,gt_extension=pol)
            require(not dd and cc['detections_after']==0,'ignore .5 boundary')
        local.append(dict(raw_file=reference(Path(context['evaluation_data']['annotations'])/z[0]),line=z[3],
                          counts=count,primary_class4_counts=list(result.match_counts_by_class_iou_max_dets[3][0][3]),COCO_matching=coco_counts(ds,det)))
    # Positive-only development remains byte-equivalent at both actual inputs.
    spec=c['parents'][0]['construction']['data']['development'];dr=read(spec['manifest'])['records'];dp=[]
    for r in dr:
        path=Path(spec['roots']['annotations'])/Path(r['annotation_relative_path']).name
        dp.append(parse_raw(read_bound(path,r['annotation_sha256'],r['annotation_size_bytes']),r['stable_image_id'],expected_rows=r['raw_annotation_rows']))
    pol=SupplementalZeroGT.bind(dr,dp,specification())
    preds=[dict(image_id=r['stable_image_id'],category_id=4,score=.9,bbox_xyxy=[1.,1.,2.,2.]) for r in dr]
    dev={}
    for budget in (600,300):
        old=evaluator_inputs(dr,dp,preds,budget);new=evaluator_inputs(dr,dp,preds,budget,gt_extension=pol)
        require(old==new and new['canonical_input_sha256']=='6438c1ec0e3cea70e6300a9b00f0d0b4bf1c46132c9a4df49436fdbfb0799650','development changed')
        require(bridge_inputs(dr,dp,preds)==bridge_inputs(dr,dp,preds,gt_extension=pol),'development bridge payload changed')
        dev[str(budget)]=new
    outcome=dict(status='PASS',real1602_chain=reference(root/'RESULT.json'),real1602_proof=proof,
                 extension_refusals=checks,real_rows_cases=local,development_input_equivalence=dev,
                 no_full_set_AP=True,no_model_predictions=True,training_updates=0,GPU_forward_calls=0)
    publish(base/'GT_EXTENSION_CPU.json',outcome)
    return reference(base/'GT_EXTENSION_CPU.json')
