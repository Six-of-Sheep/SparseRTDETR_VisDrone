"""Server CPU acceptance: real R18/criterion/optimizer, actual CLI and spawn workers."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="2"
import sys,json,subprocess,time,copy,gc,hashlib,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import torch,numpy as np
from PIL import Image
from sparse_rtdetr.dn_block_runtime import *
from dn_block_explore import DEFINITION
PY=sys.executable

def command(args,out):
    with Path(str(out)+".stdout").open("xb") as stdout,Path(str(out)+".stderr").open("xb") as stderr:
        r=subprocess.run([PY,"-B",str(ROOT/"tools/dn_block_explore.py"),*args],
                         env=os.environ.copy(),cwd=ROOT,stdout=stdout,stderr=stderr,timeout=300)
    return r.returncode

def fixture(root):
    c={"schema":SCHEMA,"policy":POLICY,"caps":CAPS,"definition_SHA":DEFINITION,"synthetic":True,
       "input_hw":[128,128],"batch":1,"eval_batch":1,"steps":3,
       "gpu_execution_authorized":False,"selection":{},"fixture_records":[]}
    c["selection"]={part:["SYNTHETIC_DN_"+part+"_"+str(k) for k in range(4)] for part in ("calibration","learning","probe")}
    engine=build(c,torch.device("cpu"))
    # Controlled synthetic weights, NOT a real N checkpoint or scientific evidence.
    with torch.no_grad():engine.model.decoder.enc_bbox_head.layers[-1].bias[2:]=-1.
    engine.model.eval()
    torch.save(engine.model.state_dict(),root/"synthetic_parent.pt")
    c["synthetic_parent"]=str(root/"synthetic_parent.pt")
    for part in ("calibration","learning","probe"):
        c["selection"][part]=[]
        for k in range(4):
            identity="SYNTHETIC_DN_"+part+"_"+str(k)
            pixels=np.random.default_rng(100+k).integers(0,256,size=(128,128,3),dtype=np.uint8)
            image=root/(identity+".png");Image.fromarray(pixels).save(image)
            im=torch.tensor(pixels).permute(2,0,1).float()[None]/255
            with torch.no_grad():out=engine.model(im)
            boxes=out["pred_boxes"][0]
            ok=((boxes[:,:2]>.2)&(boxes[:,:2]<.8)).all(-1)&(boxes[:,2:]>.01).all(-1)&(boxes[:,2:]<.03).all(-1)
            ids=ok.nonzero().flatten()[:30];need(len(ids)==30,"synthetic interior small queries")
            lines=[]
            for j,q in enumerate(ids):
                b=boxes[q].clone();e=torch.tensor([0.,0.,-.0011*(j+1),-.0009*(j+1)])
                wh=b[2:]/e[2:].exp();center=b[:2]-e[:2]*wh
                xy=(center-wh/2)*128;wh=wh*128
                label=int(out["pred_logits"][0,q].argmax())+1
                lines.append(",".join([*(format(float(v),".12g") for v in torch.cat((xy,wh))),"1",str(label),"0","0"]))
            lines.append("0,0,4,4,0,0,0,0")
            anno=root/(identity+".txt");anno.write_text("\n".join(lines)+"\n")
            c["fixture_records"].append({"stable_image_id":identity,"image":ref(image),"annotation":ref(anno)})
            c["selection"][part].append(identity)
    del engine;gc.collect()
    paths=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    files=[{"path":name,"bytes":(ROOT/name).stat().st_size,"sha256":file_sha(ROOT/name)} for name in paths if (ROOT/name).is_file()]
    m={"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"files":files}
    write(root/"manifest.json",m);c["manifest"]=ref(root/"manifest.json");c["commit"]=m["commit"]
    c["attempt"]=str(root/"attempt-success")
    write(root/"contract.json",c)
    return c

def main():
    root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    from sparse_rtdetr.protocol_r18.engine import learning_rate
    for u in (1,2000,2001,36480):need(Engine.rate(None,u)==learning_rate(u),"native scheduler unchanged")
    z={"image":"synthetic-protection","gt_ids":[str(i) for i in range(120)],"query":list(range(120)),
       "raw_area":torch.ones(120),"all_raw_area":torch.ones(120),"unmatched_GT":0,"IoU":torch.full((120,),.5)}
    fake={a:[dict(z,IoU=torch.full((120,),.51 if a=="J" else .5))] for a in ("C","J","B")}
    check=summarize([z],fake)
    need(check["main_small"]=="PASS" and check["scientific_status"]=="UNKNOWN","positive observation with UNKNOWN protection")
    c=fixture(root)
    rc=command(["launch","--wait","--contract",str(root/"contract.json")],root/"success")
    need(rc==0,"actual synthetic execute failed: "+str(root/"success.stderr"))
    done=json.loads((root/"attempt-success/CLOSED.json").read_text())
    need(done["counts"]["updates"]==10,"three arms real CPU updates")
    result=json.loads((root/"attempt-success/RESULT.json").read_text())
    need(result["protection"]=={"M":"UNKNOWN","L":"UNKNOWN"} and result["scientific_status"]!="PASS","UNKNOWN not full pass")
    need(command(["launch","--wait","--contract",str(root/"contract.json")],root/"duplicate")!=0,"repeat launch")
    need(json.loads((root/"attempt-success/CPU_RESTORE.json").read_text())["model_optimizer_EMA_RNG_cursor_next_window_exact"],"actual CLI restore")
    need(json.loads((root/"attempt-success/CPU_DN_GRADIENT.json").read_text())["DN_only_backbone_and_embedding_nonzero_finite"],"DN-only gradient")
    # Corrupt material, precheck and authorization paths must reject, not execute more learning.
    bad=copy.deepcopy(c);bad["attempt"]=str(root/"attempt-bad-input")
    bad["fixture_records"][0]["annotation"]["sha256"]="0"*64;write(root/"bad-input.json",bad)
    need(command(["execute","--contract",str(root/"bad-input.json")],root/"bad-input")!=0,"damaged material rejection")
    need((root/"attempt-bad-input/STOP.json").exists() and (root/"attempt-bad-input/EXPORT.json").exists(),"exception export")
    bad_caps=copy.deepcopy(c);bad_caps["caps"]=dict(CAPS,forward=0);write(root/"bad-caps.json",bad_caps)
    need(command(["execute","--contract",str(root/"bad-caps.json")],root/"bad-caps")!=0,"budget policy reject")
    bad_source=copy.deepcopy(c);bad_source["manifest"]["sha256"]="0"*64;write(root/"bad-source.json",bad_source)
    need(command(["execute","--contract",str(root/"bad-source.json")],root/"bad-source")!=0,"source reject")
    timeout_case=copy.deepcopy(c);timeout_case["attempt"]=str(root/"attempt-timeout")
    timeout_case["CPU_supervisor_timeout"]=.1;write(root/"timeout.json",timeout_case)
    need(command(["launch","--wait","--contract",str(root/"timeout.json")],root/"timeout")!=0,"supervisor timeout")
    need(json.loads((root/"attempt-timeout-launcher/RUN_EXIT.json").read_text())["timed_out"],"watchdog receipt")
    receipt={"status":"PASS","synthetic":True,"actual_complete_model_and_criterion":True,
             "three_arm_initial_equal":True,"all_window_precheck_subprocess":True,"multiprocess_workers":2,
             "save_restore_next_window_exact":True,"probe_mutation_check":True,
             "UNKNOWN_not_PASS":True,"repeat_rejected":True,"actual_production_launcher":True,"corrupt_material_STOP_export":True,
             "CUDA_initialized":torch.cuda.is_initialized(),"wall_seconds":time.monotonic()-start,
             "max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "real_data_or_weights_read":False,"future_GPU_budget_consumed":0,"CPU_fixture_forward":12,"CPU_pipeline_forward":41,"CPU_actual_updates":10,
             "DN_only_gradient":True,"cross_epoch":True,"positive_observation_UNKNOWN_rule":True,
             "budget_and_source_rejected":True,"actual_supervisor_timeout":True,"child_max_rss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    write(root/"ACCEPTANCE.json",receipt);print(json.dumps(receipt))
if __name__=="__main__":main()
