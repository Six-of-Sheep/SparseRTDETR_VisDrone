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


## 2026-10-27限定观测修订

每臂先CUDA synchronize，再将重置前的max_memory_allocated并入独立整链high-water标量，随后reset_peak_memory_stats，才开始monotonic阶段计时。阶段包含建模/初始化、3次实际更新、C1保存/恢复/完整状态核验、C2保存；C2完成后同步、记录阶段耗时与allocated峰值。阶段之外的数据准备、输入SHA核对、释放/empty_cache及最终导出仍纳入原整链总耗时/900秒保护，因此各臂耗时之和不应冒充整链总耗时。

更新耗时直接保存原Engine.train_window返回值：其原有CUDA同步及计时边界完全不改，包含两microbatch前反向、梯度检查/clip、optimizer和EMA，不包括窗口输入准备和阶段checkpoint I/O。记录顺序明确为update1、update2-direct、update2-replay。三次原finite/gradient norm返回值照录；不增加forward、反向、探针或测速。

每次原资源guard读取当前CUDA peak并与跨阶段累计最大值取max；最终RESULT写该整链最大值，不因逐臂重置漏掉准备/前臂峰值。异常STOP补充已观测阶段和未完成阶段状态，不修复重试；原数据、模型、RNG/恢复次序、9/18预算及科学条件均不改。CPU无CUDA时allocated保持null，不用0冒充实测。旧10a5f5c完整包与原提议补丁保留，本修订需新commit/manifest/合同身份。
