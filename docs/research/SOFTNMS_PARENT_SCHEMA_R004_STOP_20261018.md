# Soft-NMS 完整父格式审查、真实快照验收及生产STOP

本轮未完成固定Soft-NMS评分。已完成三seed小型父材料/生成源码盘点与schema适配，最终真实快照CPU检查通过；新production-20261018-002在评价输入证明阶段STOP。未读取抑制前raw-view批、未重建ABC、未评分。原SAHI三seed PASS、全部旧结果/门槛保持。

## 字段来源、含义与SHA关系

| 字段/关系 | seed0 | seed1 | seed2 |
|---|---|---|---|
| task/source/contract | CLOSED引用RESULT和runtime-index；index SHA认证archive identity，旧evaluation RESULT缺task/checkpoint，显式取archive并交叉检查 | RESULT/COMPARISON/metrics均含seed/scope/checkpoint；与archive及执行契约相符 | 同seed1 |
| 旧归档固定64 | evaluation archive.image_ids精确等于contract.fixed64；原deploy.py仅full-C选择full_image_ids，其他阶段均填fixed64 | archive.fixed64为独立字段 | 同seed1 |
| full-C 548 | archive.image_ids精确等于contract.full_image_ids，需顺序一致 | archive.evaluation_authorization.image_ids为548；starting_cursor=36480 | 同seed1 |
| 实际评价548 | evaluate_cached.run读取认证parent.data.development，Development548按固定SHA manifest生成有序records，使用RESULT预测引用与raw GT；saved metric.official.canonical_input_sha256绑定完整实际输入 | infer_seed.evaluate读取相同固定manifest、自己的A/C预测，生成B；每份metric.sahi_binding绑定prediction和seed/checkpoint | 同seed1 |
| e120来源 | 原恢复终点26752+9728=36480；contract.checkpoint与endpoint引用一致 | parent_reuse认证旧formal CLOSED和endpoint；不把旧源码绑定改成新训练身份 | 自己formal CLOSED及endpoint |
| 生成源码 | contract.source_manifest SHA绑定7文件；定向检查deploy/evaluate_cached/chain。parent源manifest绑定数据/evaluator/canonicalization代码 | source_manifest SHA绑定infer_seed/new_entry/new_runtime及共享输入/评价代码 | 同seed1 |
| 评价前后计数 | canonical_input_sha256是过滤前完整输入；evaluated_detection_count和evaluated_ground_truth_count是raw-ignore等过滤后计数，不能与原输入长度直接比较 | 同一实现语义 | 同一实现语义 |

完整第二次盘点118份小型元数据/源码，SNAPSHOT.json 7,717,305字节留服务器。生成认证原始bytes的快照供真实CPU回归；本机仅回传47,471字节INVENTORY摘要。各文件准确路径/字节/SHA在其中；生成源码原字节未改。第一次诊断把旧source_manifest引用dict当路径而出错，按本轮诊断修正并保留inventory-001。诊断写回引用接线错误也已保留；未作为生产尝试。

## 已完成的修复与CPU结果

运行commit `66b4071fc4ee8c26e98b58ef5294945b6a2f440a`，359文件独立release，原任务分支。修复64/548字段区分、旧task/checkpoint读取、seed1复用与seed2正式终点、生成源码身份、18份原metric/prediction引用及实际评价输入SHA认证入口。

真实入口parent-check使用三个seed认证原始小元数据快照，完整parent_metadata/context正例通过；126项拒绝通过，覆盖身份缺失、成员缺失/重复/错序、跨seed、阶段、路径、源码/契约/plan/checkpoint冲突及真实SHA损坏。最终冻结源码门禁内层4.101542069s、外层4.320832767s。先前开发版3.854078514s另计。语义负例为认证后的内存读注入；真实SHA故障用隔离文件，不改旧父材料。

CPU_GATE只覆盖元数据快照，不代表大型原评价输入SHA、完整缓存重建或科学评分通过。原23项算法/JSON和合成双评价器闭合门禁保留原范围，不重跑。AST及字节核对确认decode_raw/prediction/group/prepare_seed/parsed_gt/evaluate_seed/aggregate/process_inputs/Journal/supervise和softnms_core、冻结plan保持原科学/生命周期规则；pipeline只新增前置认证，export增加小证明白名单。

## 新生产异常与裁决

新独立契约已消费，parent_schema_cpu_gate与源文件SHA绑定。生产CPU资源、context和小型父身份检查通过；第一组seed0 A top600进入实际评价输入证明。代码先验证canonical_input_sha256，然后在附加计数比较失败，因此执行路径证明首组canonical SHA已匹配，但没有完整18组PASS回执。

失败：`ValueError('saved evaluation input counts differ')`。原因是本次新增计数检查把过滤前payload detections/ground_truth长度与过滤后official evaluated计数直接比较。原seed0 A600预测328,800条，原metric spatial_filter_counts detections_before=328,800、detections_after=293,711，official evaluated_detection_count=293,711；GT_after与official evaluated_ground_truth_count同为38,599。原primary_evaluator.py显式在ignored-region过滤后累计evaluated计数。该错误是本次适配实现缺陷，不是证实旧材料损坏。

按本轮“新执行异常STOP、不自动重试”保存现场；没有应用计数修复或再运行生产。worker退出2，监督退出证据保留，未产生ADMISSION/RESULT/CLOSED；原ABC重建0、E/D评分0、GPU/训练/新推理0，尚无Soft-NMS质量裁决。完整18组输入证明未完成，其余17组未知，不从首组推广。

最小后续修复应逐项采用正确计数语义：过滤前长度对应保存的spatial_filter_counts detections_before；若保留过滤后计数检查，必须调用原确定性过滤规则并比较after，不能把原输入数等同after，也不能为通过而跳过canonical SHA认证。应在诊断/CPU准备中对全部18组真实输入完成证明与计数语义正反例后才冻结下一attempt；不得只修当前一行直接重新评分。本轮未实施该方案。

## Git、交付与成本

Git提交/推送、PR/CI、真实父CPU、生产缓存准入、科学收益分别记录。PR #4保持开放未合并。原源码/STOP/已消费令牌/历史成本保留，未恢复Windows轮询。

本机白名单8文件68,652字节已核SHA：STOP、WORKER_EXIT、SLIM_EXPORT、RESOURCE_START、CONSUMED、LAUNCH、最终PARENT_CPU_GATE、INVENTORY。完整快照、原预测/GT/checkpoint/审计留服务器；不是全量回传。报告/配置/源码manifest在revision-004。

新生产supervisor测得10.055287482s（包含其worker，不重复累计）。新增其他测量详见COST.json；开发、网络、Git和零散只读检查未全程计时，不将短CPU门禁时间宣称为完整新增成本。
