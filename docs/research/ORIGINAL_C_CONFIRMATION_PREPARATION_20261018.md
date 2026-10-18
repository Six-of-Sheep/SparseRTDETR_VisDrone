# 原C独立确认准备

本轮仅历史元数据、development真实CPU接口和协议准备。没有合格确认集的无条件提名，最终执行被阻断；不能把CPU闭合写成独立确认通过。共同协议适配RT-DETRv2-R18不是完全未经修改的官方复现。原A/B/C三seed、关闭候选及原裁决保持。

## 数据资格

唯一继续核资格的既有对象是V2 conversion_r3从原train封存的1602张/57序列组，不重新抽样。split_plan SHA `6e7c029cfc707f3eb28cd71496a39e3581aac266b85e827011d056d2cce9966f`，封存manifest仅引用inventory已记SHA `1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4`，没有打开该manifest或任何封存图像/标注/lineage。目标647张是旧算法目标，实际冻结选择1602张，不更改历史选择。

读取既有split_plan、sequence_groups、source_identity、completion、artifact_inventory及训练/开发manifest，核对inventory的字节数/SHA、completion链。57组与train_core以及development均无精确图像SHA/序列键交集；core的路径集合与实际训练manifest4869条精确一致。三seed训练契约使用同一train_core manifest SHA `df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d`。这证明已记录的成员隔离，不证明感知近重复、相同地点/飞行任务的独立性；仅按已有sequence_key语义。

conversion completion的confirmatory_metrics_accessed=false只覆盖转换过程。三seed的holdout_authorized=false以及既有development报告提供限定的未授权/未使用证据，不是封存后全部访问的完备日志。当前缺少从封存生成至本轮冻结的连续访问/评价/选模/调参记录或可核验的专项审计，能以该manifest SHA精确关联所有决策接触。故资格为BLOCKED_PENDING_ACCESS_HISTORY，不能声称未见确认。另需澄清registry中1610张test-dev RESERVED标签与历史P1/P2 observed-test的准确成员对应；旧文字标签彼此不证明独立，不能用它替代1602候选。development已反复评价和选择，不能重新抽样当独立确认。

原封存生成曾解析train标注并按预注册分布容差选可行前缀，这是既有划分设计信息接触，不等同模型指标泄漏；仍须报告。转换后的所有模型评价历史尚不完整，不以“没有找到”证明从未使用。

## 条件冻结的一次性协议

机器方案见configs/sahi_independent_confirmation.json。只有资格证明闭合且用户另行授权精确开封范围，才能生成新执行合同/令牌；本准备入口没有解封、推理或科学评分命令，不消费未来授权。

三个原e120 EMA各自固定A/B/C，共18个两评价器组（36次评价器调用）。A原整图；B整图加原类内hard NMS；C原整图+TL/TR/BL/BR四切片+相同NMS。geometry、FP32候选、view/query/class/rank同分顺序及预算均复用未改的slice_core和原解码路径。h/w=ceil(5H/9)、ceil(5W/9)，原半开窗口；每视图top600，最终top600/top300前缀，预算后原raw-ignore，不补位、不clip、不调分。不是center/flip/Soft-NMS/投票/微调对照。

主判C−A：每seed top600 AP≥+.30pp、APS≥+.50pp、AP75-small≥−.20pp、APM/APL各≥−.50pp；top300 AP/APS各≥−.20pp。原始0..1点值相减后乘100一次，官方风格已有pp单列。空面积指标−1判不可评价，不以零代替。各seed七项与成本C/A≤6均须通过，均值不能抵消失败。科学失败仍完整执行其余固定seed，工程/完整性/数值/资源异常STOP，不自动重试。B−A和C−B完整报告，不能事后提升B为主候选。

统计只做三个seed内配对差、均值、样本SD(ddof=1)及最小增量；不bootstrap、不生成p值/CI、不搜索子组。不把图像当独立重复，不以3个seed估计总体可靠置信水平；这是预冻结实际收益/保护复核，不是显著性认证。官方风格和COCO bridge分列。

同硬件、各seed旧固定development64只做工程/成本，不计算64图AP或选配置。A/B/C独立进程，各2个原batch暖机，FP32/batch4，各18/18/90 batch forwards。C含全部五视图、预处理、H2D、解码回映射/NMS；报告分段和完整端到端、峰值allocated/reserved、硬件和温度。成本C/A≤6；缓存复用不能省掉计时分支。顺序A→B→C固定，不挑最快重放，页缓存口径与旧报告一致。正式1602图可从C的未抑制整图分支产生A/B以避免重复forward，但先由64一致性认证；末批2图，不drop/pad/repeat。完整模型/BN/RNG保护和资源保护仍须最终生产版本验收。

若1602资格通过：每seed full-C为401原batch×5视图=2005，加2暖机原batch×5=2015；含64成本126，共2141/seed，三seed6423 batch forward。每视图样本数与batch次数分开记；不是本轮执行数量。原548或其他集合的缓存不可替代新集合结果。

## CPU接口及局限

最终confirmation_prepare start→run→原guard.supervise→独立worker，使用真实三seed父元数据及允许的development。明确context第三返回值为data_spec；认证parsed_gt(rows,data_spec)逐标注验SHA并解析后，才进入原primary_inputs/bridge_inputs。18组原A/B/C×预算输入canonical SHA及原before/after计数核对，不重算AP。接口允许零预测成员，以数据rows定义覆盖，不能因缺候选删图。

不同数量用development有序前缀1/3/4/5/547/548检查真实标注与两输入；这只是工程输入，不是新科学子集。1602仅用整数索引检查401批/末批2，不接触封存成员内容。三seed身份、协议缺失/乱序/重复终态、worker异常、重复启动、闭合/归档、精简导出/退出分别检查。合成拒绝检查不替代真实18组接线。CPU准备不会认证未来GPU解码、末批forward或解封路径；正式授权前仍需冻结并验收受影响生产入口。

## 预算与结论边界

历史64实测A约2.34–2.41s、C约10.91–11.35s，548 full-C seed0约93.06s。线性按图数估计1602三seed五视图主体约13.6min，加固定64/暖机后约15–20min端到端GPU阶段主体；不同图像尺寸与I/O可能改变，不能将其当新实测或硬截止。历史4组双评价器监督约269.6s，按18组×1602/548粗估CPU约59min，规划30–90min；含状态审计/资源/导出与排队，整链约1–2h或更长。训练更新为零。预算依据仅用于安排，不调整旧成本或设置总墙钟截断。

若未来独立资格和全部固定门槛均通过，只支持该共同协议适配R18、这三个seed、该既有集合上的固定SAHI式实用收益；不证明新颖性、唯一尺度/细节根因、跨数据域泛化或全部随机seed鲁棒性。未通过不调尺寸/NMS/预算、不重抽确认集。大型状态/预测/日志仅实验室保存，D盘只必要资格、协议、CPU证据与SHA索引。
