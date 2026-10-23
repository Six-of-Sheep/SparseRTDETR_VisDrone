"""Real authenticated N metadata and strict format guards through actual CLI.

No detector, image, checkpoint, evaluator, CUDA call or historical() replacement.
"""
import argparse, copy, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sparse_rtdetr.localization.runio import read,publish,reference,sha

SNAP=ROOT/'docs/research/r18_local_detail_context/historical_cpu'
INDEX_SHA='65bad9dcc8274eaa2e89fa3a282bca83edd415aaf5d0c8ea143487f9eb97ea09'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);cases=[];count=0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    def cli(name,spec,ok=True,index=None,index_sha=None):
        nonlocal count
        count+=1;sp=out/(name+'-spec.json');publish(sp,spec)
        argv=[sys.executable,'-B',str(ROOT/'h2_formal.py'),'historical-check','--spec',str(sp),'--output',str(out/(name+'-result.json'))]
        if index:argv+=['--snapshot-index',str(index),'--snapshot-index-sha256',index_sha]
        r=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True)
        (out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
        if (r.returncode==0)!=ok:raise RuntimeError(name+': '+r.stderr.decode(errors='replace')[-3000:])
        cases.append(name)
    real=read(SNAP/'real-N-spec.json');idx=SNAP/'snapshot/INDEX.json'
    cli('authenticated_real_N_complete_production_historical_reader',real,index=idx,index_sha=INDEX_SHA)
    report=read(out/(cases[-1]+'-result.json'))
    assert report['snapshot_only'] and not report['physical_checkpoint_verified'] and report['endpoint_updates']==36480
    assert len({x['path'] for x in report['references_read']})==19
    for name,mut in (
        ('real_data_identity_conflict',lambda v:v['data']['train_core'].update(count=4868)),
        ('real_initial_identity_conflict',lambda v:v['pretrained'].update(sha256='0'*64)),
        ('real_score_reference_conflict',lambda v:v['historical_N']['scores'].update({'600':v['historical_N']['scores']['300']})),
        ('real_source_identity_conflict',lambda v:v['historical_N']['source'].update(sha256='0'*64)),
        ('real_checkpoint_endpoint_reference_conflict',lambda v:v['historical_N']['endpoint'].update(bytes=1)),
    ):
        v=copy.deepcopy(real);mut(v);cli(name,v,False,idx,INDEX_SHA)
    for name in ('real_snapshot_corrupt_file','real_snapshot_missing_file'):
        d=out/name;shutil.copytree(SNAP/'snapshot',d);q=d/'attempt-formal-001/train-B0/binding.json'
        if 'corrupt' in name:q.write_bytes(q.read_bytes()+b'x')
        else:q.unlink()
        cli(name,real,False,d/'INDEX.json',INDEX_SHA)
    # Clearly synthetic fixtures exercise both registered formats through the
    # SAME historical() interface; they never impersonate authenticated real N.
    n=out/'synthetic';n.mkdir();binding=n/'binding.json';proof=n/'proof.json';stop=n/'STOP.json';endpoint=n/'endpoint.json'
    for q in (binding,proof,stop):publish(q,dict(CPU_fixture=True,not_historical_evidence=True))
    publish(endpoint,dict(updates=36480,epoch=120,EMA=True,used_acceptance_state=False,binding=reference(binding)))
    scores={}
    for budget in ('600','300'):
        q=n/(budget+'.json');publish(q,dict(COCO_bridge={},official={},CPU_fixture=True));scores[budget]=reference(q)
    spec=dict(profile='CPU_FIXTURE_ONLY',historical_N=dict(endpoint=reference(endpoint),authentication=reference(proof),
        original_STOP=reference(stop),original_CLOSED_path=str(n/'CLOSED.json'),
        status='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE',scores=scores))
    row=dict(path='config.json',bytes=3,sha256='a'*64)
    legacy=dict(commit='1'*40,scope='actual release bytes; no execution authority',files=[row])
    modern=dict(schema_version=1,experiment_id='CPU_FORMAT_FIXTURE',commit='1'*40,branch='codex/CPU-format-fixture',
        config='config.json',config_sha256='a'*64,source_files=[dict(row,mode='100644')],
        execution_authorized=False,scope='CPU fixture; no execution authority')
    def format_case(name,value,ok):
        q=n/(name+'.json');publish(q,value);v=copy.deepcopy(spec);v['historical_N']['source']=reference(q);cli(name,v,ok)
    format_case('registered_legacy_files',legacy,True);format_case('registered_v1_source_files',modern,True)
    for name,mut in (
        ('missing_format',lambda v:v.pop('files')),
        ('empty_entries',lambda v:v.update(files=[])),
        ('duplicate_path',lambda v:v['files'].append(copy.deepcopy(v['files'][0]))),
        ('conflicting_path',lambda v:v['files'].append(dict(v['files'][0],bytes=4))),
        ('missing_entry_SHA',lambda v:v['files'][0].pop('sha256')),
        ('unknown_entry_field',lambda v:v['files'][0].update(size_bytes=3)),
        ('negative_size',lambda v:v['files'][0].update(bytes=-1)),
        ('boolean_size',lambda v:v['files'][0].update(bytes=True)),
        ('bad_SHA',lambda v:v['files'][0].update(sha256='x'*64)),
        ('path_escape',lambda v:v['files'][0].update(path='../config.json')),
        ('absolute_path',lambda v:v['files'][0].update(path='/tmp/config.json')),
        ('noncanonical_path',lambda v:v['files'][0].update(path='a//config.json')),
        ('dual_format',lambda v:v.update(source_files=[])),
        ('unknown_legacy_scope',lambda v:v.update(scope='unknown')),
    ):
        v=copy.deepcopy(legacy);mut(v);format_case(name,v,False)
    for name,mut in (
        ('unknown_version',lambda v:v.update(schema_version=2)),
        ('missing_mode',lambda v:v['source_files'][0].pop('mode')),
        ('invalid_mode',lambda v:v['source_files'][0].update(mode='120000')),
        ('config_SHA_conflict',lambda v:v.update(config_sha256='b'*64)),
        ('authority_conflict',lambda v:v.update(execution_authorized=True)),
    ):
        v=copy.deepcopy(modern);mut(v);format_case(name,v,False)
    publish(out/'HISTORY_CLI_GATE.json',dict(status='PASS',tests=len(cases),skips=0,cases=cases,
        actual_CLI_invocations=count,python=sys.version.split()[0],model_loaded=False,checkpoint_loaded=False,
        forward_calls=0,snapshot_index_sha256=INDEX_SHA,real_original_references=report['references_read'],
        physical_checkpoint_verified=False,current_server_absence_verified=False,
        source_files=[dict(path=x,bytes=(ROOT/x).stat().st_size,sha256=sha(ROOT/x)) for x in
            ('src/sparse_rtdetr/research/h2_history_metadata.py','src/sparse_rtdetr/research/h2_formal_contract.py',
             'src/sparse_rtdetr/research/h2_formal.py','tools/check_h2_history_cli.py')]))
    print(json.dumps(dict(status='PASS',tests=len(cases),CLI_invocations=count)))


if __name__=='__main__':main()
