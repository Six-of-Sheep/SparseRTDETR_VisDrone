"""Strict binding of real CPU protocol evidence to the executing implementation."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .runio import require, read, reference

KINDS=('real_chain','negative_cases','stage_manifest','chain_program','negative_program','unit_xml','unit_log')
IMPLEMENTATION=('channel.py','runio.py','guard.py','storage.py','cloud_probe.py')
CASES={'unknown_first','phase2_first','third_micro','wrong_phase','wrong_clock','wrong_parent','wrong_mode','missing_parent_authority',
       'duplicate_micro0_after_micro1','writer_rejects_existing_halfline','halfline_then_complete_duplicate',
       'latest_rejects_sha','archive_rejects_sha','latest_rejects_identity','archive_rejects_identity',
       'latest_rejects_time','archive_rejects_time','verify_archive_rejects_semantic_parent_despite_matching_SHA'}

def verify(c,source,base):
    proof=c['protocol_cpu_evidence'];source=Path(source);base=Path(base)
    require(set(proof)==set(KINDS)|{'implementation'},'CPU protocol evidence membership differs')
    refs=[proof[k] for k in KINDS]+list(proof['implementation'].values())
    require(set(proof['implementation'])==set(IMPLEMENTATION),'CPU tested implementation absent')
    require({r['path'] for r in refs}<={r['path'] for r in c['materials']},'CPU protocol evidence not in frozen materials')
    for r in refs:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(base) and not p.is_symlink(),'CPU evidence outside isolated root')
        require(reference(p)==r,'CPU protocol evidence bytes changed')
    manifest=read(proof['stage_manifest']['path'])['files']
    require(len(manifest)==len({r['path'] for r in manifest}),'CPU stage implementation duplicated')
    rows={r['path']:r for r in manifest}
    for name in IMPLEMENTATION:
        row=rows['candidate/source/src/sparse_rtdetr/localization/'+name];tested=proof['implementation'][name]
        current=reference(source/'src/sparse_rtdetr/localization'/name)
        require(all(row[k]==tested[k]==current[k] for k in ('bytes','sha256')),'executing protocol differs from real CPU implementation: '+name)
    require(all(rows['cpu_chain.py'][k]==proof['chain_program'][k] for k in ('bytes','sha256')),'real CPU chain program differs')
    happy=read(proof['real_chain']['path']);ex=happy['supervisor_exit'];phases=happy['observed_phases']
    require(happy['status']=='REAL_CPU_PROTOCOL_CHAIN_CLEAN_EXIT_AND_ARCHIVE_PASS' and happy['semantic_operations']==2 and phases==sorted(phases) and set(phases)=={1,2},'real CPU full chain incomplete')
    require(happy['gpu_calls']==happy['model_forward_calls']==0 and happy['torch_imported'] is False and happy['cuda_visible_devices']=='','real chain was not model/CUDA free')
    require(ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors','residual_at_leader_exit','residual_after_natural_drain','received_signals')) and ex['guard_error'] is None and not ex['process_table_cleanup_uncertain'],'real chain did not close cleanly')
    neg=read(proof['negative_cases']['path']);cases=[r['case'] for r in neg['cases']]
    require(neg['status']=='REAL_CPU_PRODUCER_OBSERVER_ARCHIVE_NEGATIVE_CASES_PASS' and len(cases)==18 and set(cases)==CASES,'real CPU negative coverage incomplete')
    require(neg['gpu_calls']==neg['model_forward_calls']==0 and neg['torch_imported'] is False and neg['old_scene_touched'] is False,'negative evidence scope differs')
    tests=list(ET.parse(proof['unit_xml']['path']).getroot().iter('testcase'))
    require(len(tests)==13 and all(not any(t.find(k) is not None for k in ('error','failure','skipped')) for t in tests),'CPU protocol unit tests incomplete')
    return {'status':'REAL_CPU_PROTOCOL_EVIDENCE_BOUND_PASS','real_chain':proof['real_chain'],'negative_cases':proof['negative_cases'],'implementation':proof['implementation']}
