# 固定P12跨seed接续：准备阶段STOP

本轮尚未完成seed1/seed2缓存准入、真实评分或测速。状态为工程STOP，不能写作科学质量失败或通过。沿用PR #12和codex/sahi-two-of-four；旧seed0六组合筛选及唯一P12选择原样保留。

## 已冻结范围

唯一方案P12=原整图view0＋原左上view1＋右上view2；原裁剪、画布、身份、稳定排序、类内NMS严格IoU>.5、top600→raw-ignore及不补位不变。原`merge_pair`和`screen`保持字节一致。seed1/2各用自身原e120 EMA缓存，同548 development、top600，原双评价器。两个seed各自8项条件：AP/APS保留至少80% C−A收益；AP75-small相对C/A均≥−.20pp，APM/APL相对C/A均≥−.50pp，未舍入判断。AR-large只完整报告，不添加门槛。见[计划](../P12_CROSS_SEED_PLAN_20261019.md)和[配置](../../../configs/p12_cross_seed.json)。

两个seed质量均通过才可进入三seed固定64真实测速，新增每seed P12/C≤.75目标；测速入口、暖机、重复、平衡顺序及forward预算仍须执行前冻结和CPU验收。本轮未进入该阶段，未把视图比例当作速度结果。seed0是development配置选择，seed1/2仅为同development跨训练seed复验，不能称独立数据验证。

## 真实执行情况

执行commit `b0fc276782428dded4f538a67a16d757d244794c`已提交推送；438文件运行manifest SHA `09c989cf1cd1d0837a4ec64396faf652d5a2751a1593dfcaf6c0f1592b7839ae`。新服务器任务`sahi-p12-cross-seed-20261019-001/release-001`，从固定提交导出，未修改历史运行目录。

5项新增CPU检查、0跳过，使用真实CLI夹具执行seed1/2双评价器、RESULT/CLOSED、导出及退出，另验worker异常、重复启动拒绝、每seed独立门槛、质量失败仍完成第二seed、跨seed身份拒绝。旧13项算法证据经旧门禁/新旧manifest及四个文件逐SHA绑定复用。其通过只覆盖相应夹具路径，**不代表真实生产准备已验收**。

随后第一次生产准备入口在`tools/p12_job.py::prepare`最先执行`from slice_core import classwise_nms`时失败：

```
ModuleNotFoundError: No module named 'slice_core'
```

该分支在设置仓库导入路径之前导入根目录模块。合成fixture分支预先设置了该路径，导致5项检查没有发现生产准备路径缺口。错误发生于父依赖/缓存准入之前；未读取本轮真实缓存或启动AP计算。未据此认定材料损坏或身份一致。保留`preparation-001/STOP.json`及全日志，未修复、未重跑、未生成生产合同/令牌/scientific attempt。用户规定新工程异常立即STOP，本轮在此结束。

| 项目 | 状态 |
|---|---|
| 新增小型CPU检查 | 5通过、0跳过，生产准备路径覆盖不足 |
| 旧算法CPU证据字节绑定 | PASS，未重跑旧六组合 |
| seed1/2真实输入及缓存重建 | 未执行完成，未准入 |
| seed1/2真实评分 | 0/2组；0/4评价器调用 |
| AP/APS/AP75-small/APM/APL/AR-large与8项判定 | 均未产生新结果，不能以seed0替代 |
| 固定64真实测速、显存、重复波动 | 未执行 |
| 新模型forward/训练更新 | 0/0 |
| 自有生产进程 | 终态核对无残留；无后台接续 |

## 成本、保全与交付

新增CPU门禁2.950532382秒；失败准备0.013540877秒，记录阶段小计2.964073259秒。包含合成小夹具评价，非真实548科学评价成本；未计部署、开发、GitHub CI或传输，非项目全生命周期总成本。旧正式657.849814606秒及原累计记录不变，不用零forward抹除历史成本。明细见[COST.json](COST.json)。

7份必要回执共5518字节已回D盘并逐SHA核验，服务器release/STOP/原材料保持；清单与SHA不是数据备份。D盘delivery为`workspace_management/delivery/experiments/localization-distribution/sahi-p12-cross-seed-20261019-001/`。无GPU、训练、1602访问、自适应、持续本机轮询或main合并。PR/CI属于工程记录，与尚未完成的跨seed科学验收分开。

接续需要另行授权修复入口导入路径并补真实prepare分支的定向CPU检查，使用新release/新准备输出；不得重放本次失败目录。当前不自动实施该修复或启动后续。
