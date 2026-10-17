"""The single synthetic-acceptance and scientific scoring launch entry."""
def main():
    import argparse,json,sys,time,subprocess
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--revision',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--synthetic-request')
    p.add_argument('--wait',action='store_true')
    a=p.parse_args();root=Path(a.revision);src=Path(__file__).resolve().parent
    import softnms_job as j
    j.require(root.is_absolute() and Path(a.attempt).name==a.attempt,'launch path invalid')
    manifest=j.read(root/'FINAL_SOURCE_MANIFEST.json')
    for row in manifest['files']:
        path=src/row['path']
        j.require(path.stat().st_size==row['bytes'] and j.sha(path)==row['sha256'],'release identity changed')
    if a.synthetic_request:
        from softnms_entry_fixture import make_contract as make_synthetic_contract
        contract=make_synthetic_contract(Path(a.synthetic_request),root,src,manifest,a.attempt)
    else:
        from softnms_reuse import make_contract as make_scientific_contract
        contract=make_scientific_contract(root,src,manifest,a.attempt)
    j.require(not Path(contract['output_root']).exists(),'scoring output already consumed')
    j.require(j.parent_ready(contract),'parent not closed')
    j.publish(root/'SCORING_RESOURCE_PREFLIGHT.json',j.resources(root,j.read(j.PLAN)))
    contract_path=root/'execution-contract.json';j.publish(contract_path,contract)
    j.load_contract(contract_path)
    log=(root/'scoring-supervisor.log').open('xb')
    child=subprocess.Popen([sys.executable,'-B',str(src/'softnms_job.py'),'run','--contract',str(contract_path)],
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'start':j.proc(child.pid)['start'],'time':time.time(),
             'contract':j.reference(contract_path),'source_commit':manifest['commit'],
             'source_manifest':j.reference(root/'FINAL_SOURCE_MANIFEST.json'),
             'synthetic_validation':contract.get('synthetic_validation') is True,'automatic_retry':False}
    j.publish(root/'SCORING_LAUNCH_RECEIPT.json',receipt)
    print(json.dumps(receipt))
    if a.wait:
        started=time.monotonic();code=child.wait()
        j.publish(root/'SUPERVISOR_EXIT.json',{'pid':child.pid,'returncode':code,
                  'waited_seconds':time.monotonic()-started,'automatic_retry':False})
        return code
    return 0

if __name__=='__main__':
    raise SystemExit(main())
