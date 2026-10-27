"""CPU admission regression: real size schema, synthetic local payloads only.

External historical receipts/role allowlists and free-disk readings are explicit
IO fixtures. The unmodified production 30GiB disk gate is tested at its boundary
and one byte below, including inherited storage. The
production material_admission, size reader, file stat/SHA, raw parser and image
reader are not replaced. No checkpoint model, CUDA, scoring, or remote access.
"""
import copy
from collections import namedtuple
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
from sparse_rtdetr.research import sabl_formal_contract as ct
import sabl_formal_preflight as admission
from sparse_rtdetr.localization import data,material_paths
from PIL import Image


def main(output):
    checks=[]
    disk_usage_result=namedtuple('DiskUsage',('total','used','free'))
    original_disk_usage=admission.shutil.disk_usage
    storage_cap=30<<30
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    def rejects(name,fn,expected):
        try:fn()
        except ValueError as error:check(name,expected in str(error))
        else:check(name,False)
    check('production_storage_cap_is_30GiB',ct.CAPS['storage_bytes']==storage_cap)
    snapshot=ct.read(ROOT/'tests/fixtures/pr26_historical_N_checkpoint.json')
    original=snapshot['checkpoint'];before=copy.deepcopy(original)
    check('real_historical_size_bytes_snapshot',admission.checkpoint_bytes(original)==323457665 and 'bytes' not in original)
    check('authenticated_snapshot_identity',original['sha256']==ct.N_SHA and original['completed_updates']==36480 and snapshot['parent_metadata_sha256']=='cbf9fab7c6a9de54293716091064715e27bff5b5905b624fade5cc1fe0f12d88')
    with tempfile.TemporaryDirectory(prefix='pr26-size-admission-') as temp:
        root=Path(temp);cp=root/'SYNTHETIC_NOT_A_MODEL.bin'
        cp.write_bytes(b'PR26 constructed CPU admission payload; never a model or real N')
        size=cp.stat().st_size;cp_sha=ct.sha(cp)
        history=ct.read(ROOT/'docs/research/r18_sabl_control/HISTORICAL_N.json')['historical_N']
        specs={};manifests={};members={};by_path={}
        # 2/64 synthetic images exercise the complete loops and fixed64 rule.
        # Counts/hashes and external metadata allowlists are fixture boundaries.
        image_buffer=io.BytesIO();Image.new('RGB',(16,12),(10,20,30)).save(image_buffer,format='PNG')
        image_bytes=image_buffer.getvalue();annotation=b'1,2,3,4,1,1,0,0\n'
        for role,count,split in (('train_core',2,'train'),('development',64,'val')):
            dirs={kind:root/role/kind for kind in ('images','annotations')}
            for folder in dirs.values():folder.mkdir(parents=True)
            records=[]
            for i in range(count):
                filename='%04d'%i
                (dirs['images']/(filename+'.png')).write_bytes(image_bytes)
                (dirs['annotations']/(filename+'.txt')).write_bytes(annotation)
                records.append({'stable_image_id':'SYNTHETIC/'+role+'/'+filename,'split':split,
                    'relative_path':filename+'.png','image_sha256':hashlib.sha256(image_bytes).hexdigest(),'image_size_bytes':len(image_bytes),
                    'annotation_relative_path':filename+'.txt','annotation_sha256':hashlib.sha256(annotation).hexdigest(),'annotation_size_bytes':len(annotation),
                    'raw_annotation_rows':1,'width':16,'height':12})
            path=root/(role+'_manifest.json');path.write_text(json.dumps({'records':records}))
            ref=ct.reference(path);members[role]=records;manifests[role]=(ref['sha256'],count,split)
            specs[role]={'role':role,'count':count,'manifest':ref,'roots':{'role':role,**{k:str(v) for k,v in dirs.items()}}}
        selected=sorted(members['development'],key=lambda r:(hashlib.sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest(),r['stable_image_id']))[:64]
        selection=root/'trace-selection.json'
        selection.write_text(json.dumps({'selected':[dict(r,selection_key=hashlib.sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest()) for r in selected]}))
        selection_ref=ct.reference(selection)
        m={'parent_record':{'path':str(root/'EXTERNAL_PARENT_METADATA_FIXTURE.json'),'bytes':1,'sha256':snapshot['parent_metadata_sha256']},
            'checkpoint':{},'historical_N':history,'prior_storage_bytes':0,'data':specs,'selection':selection_ref,
            'fixed_ids':[r['stable_image_id'] for r in selected],'python':sys.executable,'environment':{}}
        c={'metadata':m,'acceptance_cost':{'EXTERNAL_SHORT_RECEIPT_FIXTURE':True}}
        original_read=ct.read;original_exact=ct.exact
        def external_exact(ref):
            if ref==m['parent_record']:return {k:copy.deepcopy(m[k]) for k in ('data','selection','python','environment','historical_N')}
            if ref==history['endpoint']:return {'checkpoint':copy.deepcopy(m['checkpoint'])}
            if any(ref==history[k] for k in ('authentication','source','original_STOP','score_result','infer_result')):return {'EXTERNAL_RECEIPT_FIXTURE':True}
            if any(ref==v for v in history['scores'].values()):return {'official':{'evaluated_image_count':548},'COCO_bridge':{'maxDets':[1,10,100]}}
            return original_exact(ref)
        sentinel=root/'UNCHANGED_SOURCE_IO_FIXTURE.txt';sentinel.write_text('synthetic source IO fixture')
        def external_read(path):
            if Path(path)==admission.ROOT/'docs/research/r18_sabl_control/GPU_EVIDENCE_REUSE.json':
                return {'accepted_commit':ct.ACCEPT_COMMIT,'accepted_manifest':ct.ACCEPT_MANIFEST,'unchanged_files':[{'path':str(sentinel),'sha256':ct.sha(sentinel)}]}
            return original_read(path)
        def metadata_boundary(kind,path,expected):
            ref=selection_ref if kind=='selection' else specs[kind]['manifest']
            assert str(path)==ref['path'] and expected==ref['sha256']
            return Path(path)
        with ExitStack() as stack:
            # Replace only the OS reading in this CPU process, never the gate.
            disk_probe=stack.enter_context(patch.object(admission.shutil,'disk_usage',
                return_value=disk_usage_result(2*storage_cap,storage_cap,storage_cap)))
            stack.enter_context(patch.object(ct,'N_SHA',cp_sha))
            stack.enter_context(patch.object(ct,'exact',side_effect=external_exact))
            stack.enter_context(patch.object(ct,'read',side_effect=external_read))
            stack.enter_context(patch.object(ct,'accepted_storage',return_value=0))
            stack.enter_context(patch.dict(material_paths.MANIFESTS,manifests,clear=True))
            stack.enter_context(patch.object(material_paths,'SELECTION_SHA',selection_ref['sha256']))
            stack.enter_context(patch.object(material_paths,'metadata_path',side_effect=metadata_boundary))
            def ref(fields):
                value={k:v for k,v in original.items() if k not in ('bytes','size_bytes')}
                return dict(value,path=str(cp),sha256=cp_sha,**fields)
            def material(fields):
                m['checkpoint']=ref(fields)
                return admission.material_admission(c,root)
            for prior in (0,4096):
                required=storage_cap-prior
                m['prior_storage_bytes']=prior
                with patch.object(ct,'accepted_storage',return_value=prior):
                    for delta in (0,1,-1):
                        disk_probe.return_value=disk_usage_result(2*storage_cap,2*storage_cap-required-delta,required+delta)
                        disk_probe.reset_mock()
                        label='disk_prior_%d_free_delta_%d'%(prior,delta)
                        if delta<0:
                            rejects(label+'_refused',lambda:material({'size_bytes':size}),'fresh free disk')
                        else:
                            result=material({'size_bytes':size})
                            check(label+'_complete_admission',result['all_member_bytes_verified']=={'train_core':2,'development':64})
                        check(label+'_actual_probe_path',disk_probe.call_args_list==[((root,),{})])
            m['prior_storage_bytes']=0
            disk_probe.return_value=disk_usage_result(2*storage_cap,storage_cap,storage_cap)
            for label,fields in [('current_bytes',{'bytes':size}),('historical_size_bytes',{'size_bytes':size}),('equal_dual_fields',{'bytes':size,'size_bytes':size})]:
                result=material(fields)
                check(label+'_complete_material_admission',result['all_member_bytes_verified']=={'train_core':2,'development':64} and result['checkpoint_SHA']==cp_sha)
                check(label+'_record_not_mutated',m['checkpoint']==ref(fields))
            rejects('missing_size_actual_admission',lambda:material({}),'checkpoint size field missing')
            for key in ('bytes','size_bytes'):
                for label,value in [('zero',0),('negative',-1),('true',True),('false',False),('float',float(size)),('string',str(size)),('null',None),('nan',float('nan')),('inf',float('inf')),('list',[]),('mapping',{})]:
                    rejects(key+'_'+label+'_actual_admission',lambda key=key,value=value:material({key:value}),'checkpoint size must be a positive integer')
                rejects(key+'_wrong_physical_size',lambda key=key:material({key:size+1}),'physical certified N checkpoint')
            rejects('dual_size_conflict_actual_admission',lambda:material({'bytes':size,'size_bytes':size+1}),'checkpoint size aliases conflict')
            rejects('dual_invalid_alias_not_ignored',lambda:material({'bytes':size,'size_bytes':None}),'checkpoint size must be a positive integer')
            rejects('dual_boolean_alias_not_ignored',lambda:material({'bytes':True,'size_bytes':size}),'checkpoint size must be a positive integer')
            cp.write_bytes(b'X'+cp.read_bytes()[1:])
            rejects('same_size_SHA_tamper_actual_admission',lambda:material({'size_bytes':size}),'physical certified N checkpoint')
            cp.unlink()
            rejects('missing_checkpoint_actual_admission',lambda:material({'size_bytes':size}),'physical certified N checkpoint')
    check('disk_probe_restored',admission.shutil.disk_usage is original_disk_usage)
    check('production_storage_cap_unchanged',ct.CAPS['storage_bytes']==storage_cap)
    check('real_snapshot_remains_unchanged',original==before)
    import torch
    check('CUDA_never_initialized',not torch.cuda.is_initialized())
    report={'status':'PASS','checks':checks,'scope':'CPU_SYNTHETIC_PAYLOADS_REAL_HISTORICAL_SIZE_SCHEMA','external_receipt_and_role_IO_are_fixtures':True,'disk_probe_IO_is_fixture':True,'production_disk_gate_replaced':False,'storage_cap_bytes':storage_cap,'disk_cases':{'prior_storage_bytes':[0,4096],'free_minus_required_bytes':[0,1,-1]},'production_material_admission_called':True,'physical_checkpoint_model_verified':False,'model_forward':0,'optimizer_updates':0,'GPU_seconds':0}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main(sys.argv[1])
