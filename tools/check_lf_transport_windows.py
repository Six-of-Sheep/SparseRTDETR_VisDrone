"""Windows client -> binary stdin -> real WSL bash -s. No remote host."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import lf_transport as transport


def main(a):
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix='pr26-windows-binary-') as temp:
        root=Path(temp)
        args=['含 空格',"quote'\"",'$HOME; $(false)']
        code='import json,os,sys; print(json.dumps({"cwd":os.getcwd(),"argv":sys.argv[1:],"import":json.__file__}))'
        payload=transport.shell('/tmp',[a.python,'-B','-c',code,*args])
        path=root/'actual-client.sh';digest=transport.write_payload(path,payload)
        check('Windows_generator_POSIX_cwd_and_LF',payload.decode('utf-8').startswith('set -euo pipefail\n') and b'\r' not in payload)
        r=transport.send(path,['wsl','--','bash','-s'],root/'windows-receipt')
        check('actual_Windows_binary_sender_bash_success',r['returncode']==0)
        value=json.loads(r['stdout'])
        check('cwd_arguments_import',value['cwd']=='/tmp' and value['argv']==args and value['import'].endswith('/json/__init__.py'))
        check('Windows_sent_SHA',r['sent_payload_sha256']==digest==hashlib.sha256(path.read_bytes()).hexdigest())
        path=root/'exit.sh';transport.write_payload(path,transport.shell('/tmp',['bash','-c','exit 7']))
        r=transport.send(path,['wsl','--','bash','-s'],root/'exit-receipt')
        check('actual_Windows_nonzero_return',r['returncode']==7)
        legacy=Path(a.legacy_payload);before=legacy.read_bytes();receipt=root/'legacy-receipt'
        try:transport.send(legacy,['wsl','--','bash','-s'],receipt)
        except ValueError:check('original_CRLF_refused_before_sending',not Path(str(receipt)+'.begin.json').exists())
        else:check('original_CRLF_refused_before_sending',False)
        check('original_failure_payload_unchanged',legacy.read_bytes()==before)
    result={'status':'PASS','host_platform':sys.platform,'client_interpreter':sys.executable,'checks':checks,'real_WSL_bash_s_invocations':2,'legacy_SHA':hashlib.sha256(before).hexdigest(),'server_connections':0,'production_writes':0,'model_forward':0}
    Path(a.output).write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':'PASS','host_platform':sys.platform,'checks':len(checks),'real_WSL_bash_s_invocations':2}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--python',required=True);p.add_argument('--legacy-payload',required=True);main(p.parse_args())
