# 既有B0能否作为N：专项结论

**可以复用原B0固定e120 EMA及两档评分作为明确标注的历史N终点，不再安排N的120轮重训。** 它不是本轮同时重训的N。D−U是检验直接中心通路约束的主要受控对照；D−N是协议匹配的历史基线比较，不能声称排除了跨运行的平台/时间变化。旧B0缺少原阶段CLOSED这一事实保留，不将其改写为原阶段正常闭合。

## 本次依据

本次仅重新读取本地精简回执并核SHA，未连接服务器、未读取checkpoint或数据实物。审查入口为[local_detail_reuse_audit.py](../../../tools/local_detail_reuse_audit.py)，可复核回执见[审查索引](full_cpu/B0_N_REUSE_AUDIT.json)。

|核对项|本次结论|证据边界|
|---|---|---|
|模型与参数|N为共同协议原生R18，20,094,584参数；构造完整模型与原模型初state、训练输出、DN和梯度相等|CPU构造backbone，不是重新载入真实ImageNet|
|训练/初始化配置|原B0 binding.config与当前完整Config.binding逐项相等；seed0、ImageNet SHA、120×304、8×2、BF16、损失/匹配/EMA/增强时钟一致|真实reader和环境再次使用前仍需新鲜准入|
|受依赖科学源码|183份vendor、原模型/criterion/engine/checkpoint、baseline和r0源码与B0原运行manifest逐大小/SHA相等|不宣称整个新release与旧release相同；cuda_identity准入辅助文件不在科学算术同字节结论内|
|B0输出|原e120、36,480更新、EMA；used_acceptance_state=false；后续正式链认证了完整输出并完成原评分|认证状态始终是AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE|
|原异常|原supervisor被SIGKILL、原STOP保留；未补原CLOSED、未重训B0|不把怀疑OOM写成已证实根因|
|评分与取件|13份必要回执/索引本地SHA读回通过，含B0 top600/300、推理/评分闭合及端点|没有重新核15.5GB events或323MB checkpoint实物，也没有重算AP|

原初model摘要`40f9d83dff1acf1918855c17b1d7a000c03a6153394487a3e4b168a4c2bee995`；固定checkpoint SHA `be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83`。原train/development成员manifest、ImageNet身份和完整回执来源均在轻量索引中，索引不是备份。

## 复用的准确范围

1. N两档绝对指标直接引用原评分JSON，保留评价器/预算/raw-ignore/548成员身份，不为本轮重新跑N的548图。新U/D使用相同冻结口径；若身份不一致，STOP而非拼接比较。
2. N/U/D仍须**同一新会话**测固定64图FP32成本。原N成本不替代新测速；因此成本仍为384次forward。没有第三层粗/细中心配对记录的N项保持UNKNOWN，不为补表增加N端点推理。
3. 未来使用原checkpoint前须在服务器新鲜核SHA、旧binding与固定e120/EMA/缓存。显式[只读N导入接口](../../../src/sparse_rtdetr/research/local_detail_reference.py)只将旧EMA模型权重导入eval模式N供成本测试，拒绝U/D、错误scope/状态/缓存；不恢复旧optimizer、RNG或engine。新H2训练状态入口仍拒绝旧格式。
4. 新U/D从共同ImageNet独立初始化，绝不能从B0终点或短验收state开始。短GPU验收仍覆盖新N/U/D三种接线，各3个实际更新，共18次forward；这不是N正式重训。

据此新预算为**18验收＋145,920 U/D训练＋274 U/D终点＋384三臂成本＝146,596次batch forward**，比原三臂新训计划少73,097次（N训练72,960＋N端点137）。CPU验收不抵扣GPU预算。旧研究219,693次及51.00024阶段小时、支持诊断16次/14.396206秒均保留原账，不重复计为本轮新增或退款。

复用没有消除原单seed、development反复使用、数据关联和公开复现材料缺证。若未来必须回答“同一轮重新训练是否可重复”，须另立问题和预算；当前方案不冒充已经回答。
