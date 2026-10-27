# 仅供另行授权的短GPU验收

范围固定：N/S/A各一次[1,2,2]，9实际更新、18检测器forward；每次更新原物理8×2、BF16原检测器/FP32边界头与loss。原seed0/epoch1两个逻辑窗口的原图/GT/ignore/DN输入绑定；只载入原N e120 EMA作为model初始化，新optimizer/EMA/RNG。验收只核保存恢复与有限性，不算AP/AR，不测速，不进入N24/S24/A24正式训练。

限额：累计GPU阶段900秒、wall1800秒、allocated18GiB、任务新增30GiB、精简5MiB。阶段切换/重放不退款。正常结束需root CLOSED=PASS_ENGINEERING、EXIT returncode0、slim SHA；任一异常STOP并保留stderr/账本/状态，不自动重试。缺CLOSED不推断成功。完成即退出，入口没有formal子命令。

包内COMMANDS.txt为精确受验argv模板，manifest值在冻结时填入：

```text
python -B deploy.py --bundle source.tar.gz --manifest MANIFEST.json --sha256 <冻结manifest> --out <全新release>
python -B tools/sabl_acceptance.py plan --manifest-sha <同一manifest> --metadata <认证历史N元数据合同> --gpu-uuid <新鲜核实物理UUID> --attempt-root <唯一attempt绝对路径> --out <全新contract.json>
python -B tools/sabl_acceptance.py run --contract <contract.json> --authorization <authorization.json> --out <同一唯一attempt>
```

authorization必须在另行GPU授权后生成，严格为`{"contract":"合同canonical SHA256","scope":"PR26_SHORT_GPU_ACCEPTANCE_ONLY","authorize_once":true}`。canonical算法是JSON sort_keys=True、separators=(",",":")后UTF-8 SHA256；不是合同文件字节SHA。包中没有token。部署、plan和run都不得覆盖原目录/合同/材料。

配置服务器别名visdrone-gpu；既有材料记录来源为`.../r18-h2-formal-20261024-001/metadata-003/contract.json`（完整路径由已有交付contract给出），Python为`/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python`。这只是认证记录，非本轮在线观测；真实执行前必须新鲜核实包字节/精确head CI、原N实物EMA与SHA、train_core角色及32成员字节、原环境/CUDA/GPU UUID/资源/原硬件约束、无重复任务和全新路径，不能猜路径或把本次CPU构造plan当实物准入。

大C1/C2/state/日志留服务器，只回RESULT/CONTRACT/SHA_INDEX/CLOSED/EXIT或STOP与必要stderr/成本；D逐SHA。验收state格式专属，不得交给正式训练；未来完整实验需新合同、新初态、新授权及独立CPU/正式链验收，本包不包含该执行能力。
