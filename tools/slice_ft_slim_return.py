"""One-shot process-exit notification and slim SHA return; no periodic polling.

The remote kernel pidfd becomes readable on chain exit. Scientific execution is
never restarted and the running release is never modified. An interrupted SSH
connection leaves delivery STOP locally; the server experiment continues.
"""
import argparse,base64,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path


REMOTE = r'''
import os,sys,json,select,hashlib,base64,time
from pathlib import Path
spec=json.loads(sys.argv[1]);cp=Path(spec['contract'])
assert hashlib.sha256(cp.read_bytes()).hexdigest()==spec['sha256']
c=json.loads(cp.read_text());root=Path(c['job_root'])
assert c['scope']=='PRODUCTION' or spec['CPU_fixture']
assert root.is_absolute() and not root.is_symlink()
consumed=json.loads((root/'CONSUMED.json').read_text());assert consumed['contract_sha256']==c['contract_sha256'] and consumed['task_id']==c['task_id']
pid=consumed['pid'];attachment={'pid':pid,'wait':'kernel_pidfd_exit_event','periodic_polling':False}
try:fd=os.pidfd_open(pid)
except ProcessLookupError:
 assert (root/'CHAIN_EXIT.json').is_file() and (root/'SLIM_EXPORT.json').is_file()
 attachment['already_exited']=True
else:
 try:
  proc=Path('/proc')/str(pid)
  try:
   argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
   assert str(Path(c['source_root'])/'slice_ft_run.py') in argv and 'launch' in argv and argv[argv.index('--contract')+1]==str(cp)
   assert proc.stat().st_uid==os.getuid()
   attachment['start']=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
  except FileNotFoundError:assert (root/'CHAIN_EXIT.json').exists()
  print(json.dumps({'event':'WAIT_ATTACHED','attachment':attachment}),flush=True)
  select.select([fd],[],[])  # one blocking event wait; no timer or file probes
 finally:os.close(fd)
exit_record=json.loads((root/'CHAIN_EXIT.json').read_text())
export_path=root/'SLIM_EXPORT.json';export=json.loads(export_path.read_text())
records=export['records'][:]
for p in (export_path,cp,Path(c['source_manifest']),Path(c['plan']),Path(c['CPU_gate']['path']) if c.get('CPU_gate') else None):
 if p is not None:records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert sum(r['bytes'] for r in records)<16*1024**2
payload=[];seen=set()
for r in records:
 p=Path(r['path']);assert p.is_absolute() and not p.is_symlink()
 if str(p) in seen:continue
 seen.add(str(p));b=p.read_bytes();assert len(b)==r['bytes']<2*1024**2 and hashlib.sha256(b).hexdigest()==r['sha256']
 if p.is_relative_to(root):name=str(Path('job')/p.relative_to(root))
 else:name=str(Path('bindings')/p.name)
 payload.append(dict(r,relative=name,data=base64.b64encode(b).decode()))
print(json.dumps({'event':'SLIM_READY','at':time.time(),'task_id':c['task_id'],'source_commit':c['source_commit'],
 'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'scope':c['scope'],
 'attachment':attachment,'chain_exit':exit_record,'records':payload}),flush=True)
'''


def sha(b):return hashlib.sha256(b).hexdigest()


def collect(args):
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=False)
    spec={'contract':args.contract,'sha256':args.sha256,'CPU_fixture':args.cpu_fixture}
    # One remote process, low priority, no reconnect/retry loop.
    import shlex
    cmd=['ssh.exe','-T','-o','BatchMode=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
         'visdrone-gpu','nice -n 19 ionice -c 3 python3 - '+shlex.quote(json.dumps(spec))]
    started=datetime.now().astimezone().isoformat()
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    p.stdin.write(REMOTE.encode());p.stdin.close()
    lines=[]
    # Each line is an event notification, not periodic status sampling.
    for line in p.stdout:
        value=json.loads(line)
        if value['event']=='WAIT_ATTACHED':
            (dest/'WAIT_ATTACHED.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
            print(json.dumps(value),flush=True)
        else:lines.append(value)
    error=p.stderr.read();code=p.wait()
    if code or len(lines)!=1:
        (dest/'WIRE_PARTIAL.json').write_text(json.dumps(lines),encoding='utf-8')
        raise RuntimeError('one-shot SSH delivery exited '+str(code)+': '+error.decode('utf-8','replace')[-1000:])
    bundle=lines[0];require_production=bundle['scope']=='PRODUCTION'
    known={}
    for text in args.reuse:
        f=Path(text)
        if not f.is_file():continue
        raw=f.read_bytes();known[(len(raw),sha(raw))]=f
        try:v=json.loads(raw)
        except (ValueError,UnicodeError):continue
        for r in v.get('records',[]) if isinstance(v,dict) else []:
            q=r.get('local',r.get('local_path'))
            if q:known[(r['bytes'],r['sha256'])]=Path(q)
    records=[];total=sum(r['bytes'] for r in bundle['records'])
    if shutil.disk_usage(dest).free<2*total+256*1024**2:raise RuntimeError('D slim return headroom insufficient')
    for r in bundle['records']:
        b=base64.b64decode(r.pop('data'));assert len(b)==r['bytes'] and sha(b)==r['sha256']
        rel=Path(r['relative']);assert not rel.is_absolute() and '..' not in rel.parts
        same=known.get((r['bytes'],r['sha256']))
        if same is not None and same.is_file() and same.stat().st_size==r['bytes'] and sha(same.read_bytes())==r['sha256']:
            q=same;reused=True
        else:
            q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);partial=q.with_suffix(q.suffix+'.partial')
            with partial.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
            assert sha(partial.read_bytes())==r['sha256'];os.rename(partial,q);reused=False
        records.append(dict(r,local=str(q),reused=reused))
    receipt={'status':'SHA_PASS','started':started,'finished':datetime.now().astimezone().isoformat(),
          **{k:bundle[k] for k in ('task_id','source_commit','source_sha256','contract_sha256','scope','chain_exit','attachment')},
          'records':records,'files':len(records),'new_bytes':sum(r['bytes'] for r in records if not r['reused']),
          'reused_files':sum(r['reused'] for r in records),'large_state':'remote only; index is not backup','automatic_retry':False}
    (dest/'SHA_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    paths={r['relative']:Path(r['local']) for r in records}
    def get(name):return json.loads(paths['job/'+name].read_text(encoding='utf-8'))
    success='job/RESULT.json' in paths and get('RESULT.json')['status']=='COMPLETE'
    if success:
        result=get('RESULT.json');comparison=get('evaluation/COMPARISON.json')
        status='COMPLETE_QUALITY_'+comparison['quality']
        rows=['| 臂 | 推理 | 预算 | AP | APS | AP75-small | APM | APL |','|---|---|---|---:|---:|---:|---:|---:|']
        for arm in ('T-control','T-slice'):
            for view in ('A','C'):
                for b in ('600','300'):
                    v=comparison['scores'][arm][view][b]
                    rows.append('| '+' | '.join([arm,view,b]+[f'{100*v[m]:.6f}' for m in ('AP','APS','AP75_small','APM','APL')])+' |')
        text='# 双臂固定微调终点交付\n\n'+status+'；成本门槛'+('PASS' if comparison['cost_passed'] else 'FAIL')+'。按未舍入值裁决，不改门槛。\n\n'+'\n'.join(rows)+'\n\n完整28条件、成对差值、T-control相对父模型变化和实测成本见COMPARISON.json及TABLE.csv；双评价器原记录均已回传核SHA。\n'
        delta=['\n| 比较 | 预算 | AP pp | APS pp | AP75-small pp | APM pp | APL pp |','|---|---|---:|---:|---:|---:|---:|']
        for name,values in comparison['differences_pp'].items():
            for b in ('600','300'):
                delta.append('| '+' | '.join([name,b]+[f'{values[b][m]:+.6f}' for m in ('AP','APS','AP75_small','APM','APL')])+' |')
        for view,values in comparison['T-control_minus_parent_pp'].items():
            for b in ('600','300'):
                delta.append('| '+' | '.join(['T-control '+view+' minus parent',b]+[f'{values[b][m]:+.6f}' for m in ('AP','APS','AP75_small','APM','APL')])+' |')
        text+='\n'.join(delta)+'\n\n成本原值：\n\n```json\n'+json.dumps(comparison['cost'],indent=2,ensure_ascii=False)+'\n```\n'
        text+='\n正式训练及验收计数/耗时原回执：\n\n```json\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\n```\n'
        text+='\n仅seed0从已训练父EMA微调；不代表独立多seed可重复收益。科学失败不会追加轮数、调参或自动扩seed。\n'
    else:
        stop=get('STOP.json');status='STOP_NO_RETRY'
        text='# 双臂固定微调执行STOP\n\n'+stop['error']+'\n\n新的GPU/正式异常，未自动修复、重试或接续；未完成终点部分不评价。完整STOP和各阶段结果见SHA_RECEIPT.json的路径索引。\n'
    if not require_production:text='CPU_RETURN_INTERFACE_FIXTURE_ONLY\n\n'+text
    text+='\n运行commit '+bundle['source_commit']+'；完整checkpoint/预测/审计仅实验室，D已核验材料见SHA_RECEIPT.json；索引不是备份。\n'
    (dest/'REPORT.md').write_text(text,encoding='utf-8')
    if args.shared and require_production:
        shared=Path(args.shared);now=datetime.now().astimezone().isoformat()
        current=shared/'CURRENT.md';before=current.read_bytes();entry=f'## {now} · 单次退出事件回传完成：{status}\n\n服务器执行链已退出，精简白名单{len(records)}文件核SHA通过；没有本机定时轮询/重试。来源{bundle["source_commit"]}，状态{status}。报告{dest}/REPORT.md，详细结果和28门槛/成本见回传COMPARISON与TABLE；完整状态仅远端。科学结果不等于Git合并，PR5不自动合并。此为自动生成的终态交接，保留此前所有记录。\n\n'
        updated=before.decode('utf-8').replace('# 最新交接摘要\n','# 最新交接摘要\n\n'+entry,1)
        if current.read_bytes()!=before:raise RuntimeError('CURRENT changed concurrently; report preserved, shared update deferred')
        current.write_text(updated,encoding='utf-8')
        with (shared/'HISTORY.md').open('a',encoding='utf-8') as f:f.write(f'\n- {now} · 固定双臂单次退出事件回传：{status}，{len(records)}必要文件SHA通过，无持续轮询或重试。报告{dest}/REPORT.md；大文件仅远端。\n')
    print(json.dumps({'status':status,'files':len(records),'new_bytes':receipt['new_bytes'],'report':str(dest/'REPORT.md')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reuse',action='append',default=[]);p.add_argument('--shared');p.add_argument('--cpu-fixture',action='store_true')
    args=p.parse_args()
    if Path(args.output).exists():p.error('refuse existing return output without modifying old evidence')
    try:collect(args)
    except BaseException as e:
        q=Path(args.output);q.mkdir(parents=True,exist_ok=True)
        (q/'TRANSFER_STOP.json').write_text(json.dumps({'error':repr(e),'automatic_retry':False,'server_execution_not_modified':True},indent=2),encoding='utf-8')
        raise


if __name__=='__main__':main()
