# Soft-NMS真实数据准入PASS／新评分入口STOP

**真实验收已完成，固定E/D科学评分未完成。** 新评分attempt在启动worker前因入口名称遮蔽异常退出。未自动修复或重试，无新的AP/APS和质量裁决；原SAHI三seed结果、门槛和全部旧STOP保持。

| 独立阶段 | 本轮结果 | 证据 |
|---|---|---|
| 原合成CPU工程检查 | 认证复用PASS；原362文件逐字节不变 | UNCHANGED_CPU_REUSE.json及旧CPU_RESULT/REAL_FLOW_CPU_GATE引用 |
| 输入路径/引用前置核验 | PASS | preparation-001/PREFLIGHT.json |
| 真实三seed父元数据 | PASS，完整context正例及126项拒绝 | parent-check/PARENT_CPU_GATE.json、real-data/METADATA_GATE.json |
| 原A/B/C×600/300×3seed输入 | 18/18 PASS；完整输入SHA和过滤计数一致 | real-data/EVALUATION_INPUT_PROOF.json、ORIGINAL_INPUT_18.csv |
| 真实抑制前缓存和原输出重建 | PASS，411batch；548图/seed，原A/B/C两预算逐项精确一致 | real-data/CACHE_REBUILD_GATE.json及三个PREPARED.json |
| 新门禁的实际消费接口 | PASS，正例及8项拒绝 | real-data/GATE_CONSUMER_CPU_GATE.json |
| 准备闭合 | PASS，未调用AP评分 | PREPARATION_CLOSED.json及REAL_DATA_GATE.json |
| 唯一新科学评分attempt | STOP；worker未启动，科学评分0/12 | production-20261018-003/STOP.json、SCORING_EXIT_READBACK.json |

上述证据分别认证，不能用元数据或CI PASS替代科学评分。

## 本次路径修复与真实验收

新增Git入口softnms_prepare_chain.py。它从已认证的旧execution-contract引用取得PARENT_CPU_GATE，再读取其中real_snapshot的准确路径、字节数和SHA，不再拼接快照子目录。新检查输出位于revision-006/preparation-001，旧revision-005及其他现场不改。

认证快照位于revision-004/SNAPSHOT_AUTHENTICATED.json，10812081B，SHA6fa6d6b079eef4058e8c12150dc5ceb0182279173edf32b41b347b28352ba50c。启动前验证全部小型快照文件SHA、后续输入引用的路径/大小及输出未占用；实际阶段再逐文件验证大型预测及raw缓存SHA。成员、seed、阶段、checkpoint、源码、契约和旧闭合关系仍严格检查。

计数沿用原顺序：原始query/class分数→每视图候选池→原后处理→top600/300截断→raw-ignore→评价。全部18份canonical输入SHA与原记录相同；原bridge_inputs直接生成before/after，after必须等于原official evaluated计数，不补位、不重写过滤器，也未重算旧AP。

例如seed0/A600：原query/class分数1644000，选取候选328800，预算后328800，raw-ignore后293711，原official evaluated同为293711。seed0/C600：五视图池1644000，原NMS后633230，预算后328532，raw-ignore后290276，与原评价相同。三个seed均GT_before38759、GT_after38599。18组完整阶段计数见STAGE_COUNTS_SUMMARY.json；原候选及详细逐图计数仅服务器保存。

准备阶段严禁score_budget、evaluate_primary_v1和bridge_metrics；GT只用于原输入证明，prepare_seed不接收GT。准备阶段生成的E/D预测未评分，仍仅服务器保存。最终门禁核验其SHA与身份，未用已删除的NMS输出反推候选。

## 新科学入口异常

全部准备通过后才创建execution-contract和唯一production-20261018-003。最终科学源码commit0c135a446c35255c20b345e3eb1a873fcc1b7add，365文件，final-source.tar SHA0012d32852061583d38da1582aa97ec042b8ef189843ad48fdec75f87ab304ec；manifest SHA565c0a9a2c27b3d7668dc5c6eedb0a2732e30ee63ac519ac77b27cc2931ed2c5。执行契约文件SHA a2cd01eb52b31dda9074c249c44240c4612662b66f86b76469fca819a5aac065，canonical digest ee98ba7d642430be480d2c0862d79bcfe001ef48e4452e215aeb20bb719001f3。

`softnms_job.py main()`在input-check分支含`from softnms_input_check import run, supervise`。Python因此把supervise视为整个main函数的局部名称；run分支未经过该导入，调用supervise时抛出`UnboundLocalError("local variable 'supervise' referenced before assignment")`。这是新增工具入口缺陷，不是材料、SHA、数值或科学配置异常。此前合成检查覆盖cpu-check和认证准备消费路径，却未覆盖run→supervisor的真实分派；`--help`通过也不能认证这条分支。

STOP发生于2026-10-18T07:05:13+08:00；精确时间以STOP和退出回读epoch字段为准。主进程PID1133977已退出。新目录只有CONSUMED、STOP、SLIM_EXPORT；没有worker.log、WORKER_EXIT、events.jsonl、seed目录、ADMISSION、RESULT或CLOSED。控制流和现场共同证明worker未启动，E/D科学score_budget为0。没有把已消费attempt追认PASS，也没有创建第二次评分attempt。

按本次授权评分阶段新异常STOP，不修复或重试。σ、候选池、FP32/FP64规则、排序、零分/下溢、预算、raw-ignore、双评价器和收益门槛均保持；无GPU测速、模型构造、新推理、训练、bootstrap、holdout或本机持续轮询。

## 版本、成本和精简交付

准备入口commit426d0791a3c4178d30d6231b7de4d391bd5afb7b，独立364文件release；最终评分commit仅再增加受Git管理的评分启动入口，其余364文件字节不变。旧原362文件字节也与revision005一致，未重复原合成检查。

原任务分支和PR #4继续使用，不合并main。失败后只追加本报告，提交/推送和CI状态见PULL_REQUEST.json、CI_READBACK.json；CI不代替实际run入口或科学验收。

D盘必要门禁、18组输入证明、契约、STOP及退出回执22文件570085B已逐文件SHA核验，见SLIM_RETURN_RECEIPT.json。完整raw缓存、已准备E/D预测、checkpoint及审计仅远端；本地索引不等于独立大材料备份。共享CURRENT/HISTORY保留旧记录并更新。STAGE_COUNTS_SUMMARY仅汇总已认证计数，没有评分；其首个收尾脚本因系统Python缺NumPy失败，随后改用纯stdlib JSON/SHA读取，未改冻结源码或重试科学评分，记录见CLOSEOUT_ENGINEERING.json。

新增元数据外层4.053519779s，真实输入/缓存准备外层512.437439544s；旧累计及本轮成本详见COST.json。内外层不重复累计，启动/STOP事件间隔单列；开发、网络和Git未全程计时，不宣称这些数字是全部新增墙钟。端到端推理成本未新增测量。

后续最小修复仅记录建议：为input-check导入使用独立别名，补实际run→supervisor→worker的隔离CPU分派验收；对协议接线与科学代码不变作明确源差异认证，按新绑定条件复用本轮18输入和缓存证据。不得直接放宽旧source校验或重放已消费attempt。本轮未实施该修复或接续，无质量PASS/FAIL裁决。
