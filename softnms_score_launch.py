"""Freeze and launch one CPU scientific scoring attempt after closed real-data gates."""
def main():
    import argparse,copy,hashlib,json,sys,time,subprocess
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();root=Path(a.revision)
    assert root.is_absolute() and Path(a.attempt).name==a.attempt
    src=Path(__file__).resolve().parent;sys.path.insert(0,str(src));import softnms_job as j
    from softnms_input_check import gate_identity,admitted_preparation
    prep=root/'preparation-001';closed=j.read(prep/'PREPARATION_CLOSED.json')
    j.require(closed['status']=='PASS','all preparation phases must close before freezing a scoring attempt')
    old=json.loads(j.exact(j.read(root/'PREPARATION_CHAIN_REQUEST.json')['historical_contract'],contents=True))
    manifest=j.read(root/'FINAL_SOURCE_MANIFEST.json')
    for row in manifest['files']:
     p=src/row['path'];j.require(p.stat().st_size==row['bytes'] and j.sha(p)==row['sha256'],'release identity changed')
    gate=json.loads(j.exact(closed['real_data_gate'],contents=True))
    contract=copy.deepcopy(old)
    contract.update(schema='softnms-fixed-scoring-contract-v2-real-data-gated',attempt_id=a.attempt,
     output_root=str(root.parent/a.attempt),source_root=str(src),source_files=manifest['files'],
     source_commit=manifest['commit'],source_manifest=j.reference(root/'FINAL_SOURCE_MANIFEST.json'),
     parent_schema_cpu_gate=closed['parent_schema_cpu_gate'],
     real_data_gate={'preparation':closed['real_data_gate'],'input_proof':gate['input_proof'],'cache_gate':gate['cache_gate']},
     preparation_closure=j.reference(prep/'PREPARATION_CLOSED.json'),
     inherited_synthetic_CPU=j.read(root/'PREPARATION_CHAIN_REQUEST.json')['prior_synthetic_CPU'],
     scope='one frozen E/D x 600/300 x seeds0/1/2 scoring; original ABC points reused, no GPU/new inference/training',
     attempt_history_preserved=True)
    j.require(contract['python']==sys.executable,'interpreter differs')
    j.require(j.sha(sys.executable)==contract['python_sha256'],'interpreter SHA differs')
    j.require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==contract['boot_id'],'boot differs')
    j.require(not Path(contract['output_root']).exists(),'scoring output already consumed')
    j.require(j.parent_ready(contract),'parent no longer closed')
    specs,rows,_=j.context(contract);admitted_preparation(contract,specs,rows)
    j.publish(root/'SCORING_RESOURCE_PREFLIGHT.json',j.resources(root,j.read(j.PLAN)))
    contract_path=root/'execution-contract.json';j.publish(contract_path,contract)
    j.load_contract(contract_path)
    log=(root/'scoring-supervisor.log').open('xb')
    child=subprocess.Popen([sys.executable,'-B',str(src/'softnms_job.py'),'run','--contract',str(contract_path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'start':j.proc(child.pid)['start'],'time':time.time(),'contract':j.reference(contract_path),
     'source_commit':manifest['commit'],'source_manifest':j.reference(root/'FINAL_SOURCE_MANIFEST.json'),
     'new_scientific_workload':12,'new_GPU_calls':0,'automatic_retry':False}
    j.publish(root/'SCORING_LAUNCH_RECEIPT.json',receipt)
    print(json.dumps(receipt))

if __name__ == "__main__":
    main()
