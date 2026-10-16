"""Verify actual local gate receipts, rather than a caller supplied PASS flag."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .runio import TASK_ID,require,read,sha,scoped

def verify(path,source_sha256):
    path=Path(path);value=read(path)
    require(value.get('schema')=='f0-local-ready-v1' and value.get('task_id')==TASK_ID and value.get('status')=='LOCAL_READY','local readiness schema/status')
    require(value.get('source_sha256')==source_sha256,'local readiness source mismatch')
    files=value['evidence_files'];names=[row['path'] for row in files]
    require(len(names)==len(set(names)) and len(names)>0,'local readiness evidence inventory')
    for row in files:
        file=scoped(path.parent,row['path'])
        require(file.is_file() and file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'local check receipt changed: '+row['path'])
    gates=value['gates'];require({g['id'] for g in gates}=={'operators','core_reused','affected','lifecycle'} and len(gates)==4,'local gate coverage missing')
    for gate in gates:
        if gate['id']=='operators':
            require(gate['path'] in names,'local gate is not hash bound')
            file=scoped(path.parent,gate['path'])
            report=read(file);require(report['status']=='PASS' and len(report['fixtures'])==16,'operator validation not passed')
        else:
            reports=gate.get('reports',[gate]);require(len(reports)>0,'local gate has no reports')
            for entry in reports:
                require(entry['path'] in names,'local gate is not hash bound')
                file=scoped(path.parent,entry['path']);report=ET.parse(file).getroot();cases={}
                for case in report.iter('testcase'):
                    require(case.attrib['name'] not in cases,'duplicate testcase name')
                    cases[case.attrib['name']]=case
                selected=entry['testcases']
                require(len(selected)>0 and len(selected)==len(set(selected)),'empty/duplicate selected gate tests')
                for name in selected:
                    require(name in cases and not any(cases[name].find(tag) is not None for tag in ('failure','error','skipped')),'local check did not pass: '+name)
    return value
