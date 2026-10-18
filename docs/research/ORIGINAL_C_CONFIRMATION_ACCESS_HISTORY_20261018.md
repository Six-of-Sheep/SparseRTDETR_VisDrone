# 1602图确认候选：限定访问历史追溯

结论：**仍无法确认是否未用于模型开发；保持 `BLOCKED_PENDING_ACCESS_HISTORY`，无合格确认集。** 本轮限定范围已完成，到此停止；不自动解封、重划、修改门槛或扩展取证。

审查时间：2026-10-18T21:51:27.901691+08:00。对象manifest SHA：`1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4`（972181 bytes，1602图/57组），只沿既有inventory引用，不打开封存manifest payload、lineage、图像或标注。GPU、模型加载、forward、训练和科学评分均0。

## 身份、时间及别名

找到原转换进程的归档回执。其记录生成窗口为 **2026-08-09 12:14:36.344828—12:14:52.962341 +08:00**。process_inventory验证timing/invocation SHA；process_completion绑定entry completion与artifact_inventory，后者绑定目标manifest SHA。这是进程记录时间，非外部签名时间或持续访问审计。转换确实解析了原train标注并按分布要求选择封存组；属于划分设计信息接触，不是已证明的模型评分使用。

精确路径仍为`/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone/artifacts/data/visdrone_protocol_v2_conversion_r3/confirmatory_sealed_manifest.json`。旧进程证据路径经2026-10-15精确迁移表映射至`/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/archive/reorganized-20261015-001/history/from-main-artifacts/process_evidence/visdrone_protocol_v2_conversion_r3`。2026-09-30映射也已核对；迁移不是访问历史证明。原数据、日志和契约未改写。

已区分V2/conversion_r3、P3-confirmatory-v1、confirmatory/sealed/holdout、旧647目标及实际1602。647是目标数，不能当实际封存数量；V1/conversion_r1失败先例不是同一集合。registry的1610 test-dev RESERVED_CONFIRMATORY不能凭名称并入本集合，也不能证明完全无关。机器可读别名、任务、Git和路径索引见IDENTITY_ALIASES_TASK_INDEX.json。

## 证据强度与覆盖

| 证据 | 覆盖 | 可以说明 | 不能说明 |
|---|---|---|---|
| E01–02 转换真实回执、完成链 | 2026-08-09单次转换 | train,val转换实际完成；本进程声明未访问确认指标 | 后续所有任务从未用过；既有false不推广为终身证明 |
| E03 迁移映射 | 09-30、10-15快照 | 旧路径与当前归档回执的连续性 | 路径下全部材料的使用历史 |
| E04 登记及协议 | 历史快照至本轮观测 | 已登记评价以development548为口径；旧7项登记并非全生命周期索引 | RESERVED标签就是未用数据 |
| E05 C2/旧smoke绑定 | 已列元数据任务 | 相关任务引用conversion_r3中core/dev材料 | 引用父目录就等于使用sealed子集 |
| E06 R0及三seed契约 | 对应合同覆盖阶段 | **授权禁止**holdout/confirmatory；当前训练成员绑定core4869 | 授权条款等于实际拒绝，或等于从未绕过 |
| E07 T6H-R8/旧审查 | 09-11、09-19文档 | 报告未访问声明；647/1602及C1/C2讨论 | 报告断言是完备机器记录；讨论即已执行 |
| E08 限定检索 | 10-18观测 | **没有发现记录**将精确集合关联至已完成模型评价/预测/选模 | **从未使用**，或缺失记录就是否定证据 |
| E09 入口/源码 | 历史Git及现准备入口 | 静态test路径拒绝；confirmatory转换解析允许；准备合同范围限制 | 未找到本集合历史尝试访问被拒回执，不能称**实际入口已拒绝** |
| E10 Git | 旧服务器库全部可达refs、指定代码路径 | 协议/转换器/V2规划/运行绑定演变 | 未跟踪、删除、手工/异机执行都入Git |
| E11 产物索引 | 登记和index快照 | 有历史评价/预测路径，仅记录身份、存在性；不读取候选确认指标 | 产物存在即使用本集合；未解析payload则不能证明其成员 |

尚未建立“确实发生本精确集合模型评价/选模/调参使用”的正证据。已确认的实际使用仅是封存生成时标注解析/划分规划，必须与模型开发指标使用分开报告。不能将两类使用混为一谈。

## 限定范围与遗漏

服务器1343份/21371909 bytes、本地621份/11298632 bytes的小型登记、合同、CLOSED/completion、身份、输入/产物索引和报告元数据已检索；其中远端122份CLOSED命名记录/135份contract命名记录、本地5/57。计数是扫描文件数，包含副本及CPU记录，不是独立实验次数或完整闭合链。另定向核对迁移表、转换回执、5份registry/index及Git历史。

在该扫描白名单中，精确SHA命中远端0、本地2（均本次准备引用）；原数据inventory及定向转换证据另列，不能把远端0误读为集合不存在。搜索也包含旧名和协议路径，不只confirmatory。`1602/647`偶然出现在其他SHA中的命中剔除为身份依据。

完整根、层数、文件名/大小限制、跳过路径、各文件SHA与观测时间见AUDIT_SUMMARY及两份METADATA索引。单文件上限512KiB、深度有限、排除大型数据/预测/逐窗日志/深层worktree和payload；部分DATA_VERIFICATION/DATA_CLOSURE、大型GT元数据、压缩旧盘点因此未读。JSON投影有80字段/300列表项上限；注册信息另作定向路径存在性核查，也不是全产物内容审计。混合报告通过字节搜索提取token/行号；登记status文本可能夹带development数字，未用于资格判断；没有打开候选确认集评价payload或读取其指标。本轮不声称对所有混合文件的字节零接触。

明确缺口：封存后至今并无完整可核验的、绑定该SHA的访问/预测/评价/选模/调参生命周期记录；不覆盖已删/未登记/非标准入口、旧异机或人工操作。Git提交日期不能替代封存日期。服务器旧库HEAD30b176d8与当前GitHub main不同，审查分别记账；本轮不pull运行目录或改变任何实验。

## 如需补人工说明，仅以下具体问题

人工说明本轮尚未收到，不能列为机器审计。未来若继续核资格，可请当事人说明并附原记录：

1. T6H-R8及2026-09-11 P3_BASELINE_V2_DECISION的C1保留1602/C2重划讨论后，是否手工评价/导出过该封存集合，或实际实施过V3？对应命令、时间、任务及输出索引在哪里？
2. 2026-08-09封存之前的P1/P2或早期P3开发，是否用原train6471中的未来1602成员做过指标、可视化错误分析或选模/调参？既有报告只证明当前三checkpoint的训练core隔离，不能覆盖更早开发决策。
3. 08-09至本轮之间，是否有绕过已认证launcher、以647/holdout/sealed/旧路径运行的手工脚本或迁出副本？有则给出具体任务/机器/日期及元数据，不要求此时开封。

没有这些记录时，结论停留“仍无法确认”，不补写一个lifetime_used=false。

## 另行未解决：近重复及飞行/地点关系

复用原资格证据：与三个checkpoint实际core4869和development的已记录图像SHA、sequence_key无交集。**这仅证明精确成员/已有序列键隔离，不证明完全独立。** 感知近重复须取得经过授权的封存图像访问，或已有可审计的全源感知指纹及算法阈值/配对结果；本轮不计算。飞行/地点关系需要原采集flight/event/site标识、时间/GPS、跨序列映射或官方来源说明；即便序列键不同也可能相关。只有资料不足时明确未知，不以SHA或键无交集替代。

## 处置与版本

原DATA_QUALIFICATION、PLAN、CPU_GATE/CPU_REPORT、UNSEAL_SCOPE及BLOCKED全部保留原字节；PRESERVED_PREPARATION.json保存SHA。本附录和DATA_QUALIFICATION_REVISION_002.json是新审查版本，不追认资格PASS。原CPU准备证据不受影响，未重复运行；当前只做文档/元数据绑定和Git管理检查。条件协议/门槛不变，PR #9保持开放且不合并。本任务无后台工作或新GPU/评分链，无解封请求。
