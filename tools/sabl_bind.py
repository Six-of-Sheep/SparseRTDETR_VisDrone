"""Final contract binding via certified Python -B - < tools/sabl_bind.py.

The caller must cd to the release first. --check-only checks binding inputs and
imports without writing anything or performing material/resource admission.
--bind requires separate execution authorization; it never starts a job.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def need(ok, message):
    if not ok:
        raise ValueError(message)


def bootstrap(release, python):
    release = Path(release)
    need(release.is_absolute() and release.is_dir(), 'absolute release directory')
    need(not any(p.is_symlink() for p in (release, *release.parents)), 'release symlink')
    need(Path.cwd() == release, 'release working directory')
    need(Path(python).is_absolute() and Path(sys.executable).resolve() == Path(python).resolve(), 'binding interpreter')
    # stdin execution supplies no tools/ path; own it explicitly, before import.
    sys.path[:0] = [str(release / 'tools'), str(release / 'src')]
    formal = importlib.import_module('sabl_formal')
    for name, relative in (('sabl_formal', 'tools/sabl_formal.py'),
                           ('sabl_acceptance', 'tools/sabl_acceptance.py'),
                           ('r18_deploy', 'tools/r18_deploy.py'),
                           ('sparse_rtdetr.research.sabl_formal_contract', 'src/sparse_rtdetr/research/sabl_formal_contract.py')):
        module = importlib.import_module(name)
        need(Path(module.__file__).resolve() == (release / relative).resolve(), 'binding import origin: ' + name)
    return formal


def validate_binding(a, formal):
    task, preflight, release = Path(a.task), Path(a.preflight), Path(a.release)
    need(task.is_absolute() and task.is_dir() and release.parent == task / 'releases', 'binding task/release paths')
    need(preflight.is_absolute() and preflight.parent == task and preflight.is_dir(), 'binding preflight path')
    need(not any(p.is_symlink() for p in (task, *task.parents, preflight)), 'binding path symlink')
    need(re.fullmatch('[0-9a-f]{40}', a.commit) and re.fullmatch('[0-9a-f]{64}', a.manifest_sha), 'binding version format')
    deployed = formal.verify_release(release, a.manifest_sha)
    need(deployed['commit'] == a.commit and not deployed['candidate_not_for_real_execution'], 'binding release version/candidate')
    c = formal.read(preflight / 'PROPOSED_CONTRACT.json')
    formal.ct.validate(c)
    need(c['manifest_sha256'] == a.manifest_sha and c['mode'] == formal.ct.GPU, 'binding contract version/mode')
    need(c['attempt_root'] == str(task / 'attempt-formal-001'), 'binding attempt path')
    need(c['metadata']['python'] == a.python, 'binding certified interpreter record')
    source = formal.read(preflight / 'SOURCE.json')
    need(source['status'] == 'PASS' and source['commit'] == a.commit and source['manifest_sha256'] == a.manifest_sha, 'binding source receipt')
    material = formal.read(preflight / 'MATERIAL_HIDDEN.json')
    need(material['status'] == 'PASS' and material['real_forward'] == 0, 'binding material receipt')
    need(material['material']['all_member_bytes_verified'] == {'train_core':4869, 'development':548}
         and material['material']['accepted_computation_files'] == 300 and material['fixed64_count'] == 64, 'binding data receipt')
    need(material['N_e120_EMA']['sha256'] == formal.ct.N_SHA and material['N_e120_EMA']['updates'] == 36480
         and material['N_e120_EMA']['epoch'] == 120, 'binding parent EMA receipt')
    digest = formal.digest(c)
    need(material['proposal_digest'] == digest, 'binding proposal digest')
    resources = formal.read(preflight / 'RESOURCE_VISIBLE.json')
    need(resources['status'] == 'PASS' and resources['probe_forward'] == 0, 'binding resource receipt')
    need(resources['inherited_storage_bytes'] == c['metadata']['prior_storage_bytes']
         and resources['storage_cap_bytes'] == formal.ct.CAPS['storage_bytes'], 'binding storage inheritance')
    need(formal.sha(task / 'preflight/STOP.json') == a.old_stop_sha, 'binding old STOP changed')
    for p in (task / 'contract.json', task / 'authorization.json', task / 'FORMAL_ONCE.json',
              task / 'attempt-formal-001', task / 'stdout.txt', task / 'stderr.txt', preflight / 'FINAL_BIND.json'):
        need(not p.exists() and not p.is_symlink(), 'binding existing output: ' + str(p))
    need(not (preflight / 'STOP.json').exists(), 'binding current preflight STOP')
    return c, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'python', 'task', 'preflight', 'commit', 'manifest-sha', 'old-stop-sha'):
        p.add_argument('--' + name, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--bind', action='store_true')
    a = p.parse_args()
    formal = bootstrap(a.release, a.python)
    need(re.fullmatch('[0-9a-f]{64}', a.old_stop_sha), 'old STOP digest format')
    # This identity is fixed, not a caller-selected historical scene.
    need(a.old_stop_sha == 'd042caa937755513ddca494cacac26d5da5c56562f6caf721261f687b4744178', 'registered old STOP identity')
    c, digest = validate_binding(a, formal)
    result = {'status':'PASS_BINDING_CHECK_ONLY', 'contract_canonical_sha256':digest,
              'commit':a.commit, 'manifest_sha256':a.manifest_sha, 'cwd':str(Path.cwd()),
              'interpreter':sys.executable, 'formal_import':formal.__file__,
              'stdin_script':globals().get('__file__') == '<stdin>', 'production_files_written':False,
              'material_resource_admission_performed':False, 'job_started':False}
    if a.bind:
        # Preserve the original production binding gates; no test fixture path.
        formal.ct.verify_acceptance(c['acceptance_cost'])
        need(formal.ct.accepted_storage(c['acceptance_cost']) == c['metadata']['prior_storage_bytes'], 'accepted storage changed')
        need(subprocess.run(['tmux','has-session','-t','r18-sabl-formal24-001'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0, 'existing formal tmux')
        rows = subprocess.check_output(['ps','-eo','pid,ppid,args'], text=True).splitlines()[1:]
        need(not [s for s in rows if any(x in s for x in ('tools/sabl_formal.py','tools/sabl_acceptance.py','r18-sabl-formal24-001'))], 'existing formal process')
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'existing GPU process')
        # Recheck absence and binding immediately before exclusive writes.
        c, digest = validate_binding(a, formal)
        task, preflight = Path(a.task), Path(a.preflight)
        formal.write(task / 'contract.json', c)
        formal.write(task / 'authorization.json', {'contract':digest, 'scope':formal.ct.SCOPE, 'authorize_once':True})
        result.update(status='PASS_AUTHORIZATION_BOUND', production_files_written=True,
                      contract_file_sha256=formal.sha(task / 'contract.json'))
        formal.write(preflight / 'FINAL_BIND.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
