# 一次原N终点缓存诊断：执行方案

2026-10-26更新：用户授权的CPU接线、验收和单次真实执行均已完成，CLOSED PASS；不再等待启动或重放。本文件保留受验统计/资源上限，语义澄清见第7节，实际闭合见第8节。三个收尾答案在DECISION.md，不自动恢复候选。

对应[研究决定](DECISION.md)。**补证已完成，仍不保留可直接训练的候选。** 初次仅完成索引/源码/文献审查，随后依据单独授权执行了以下CPU诊断；没有AP/AR计算。

## 1. 本次已授权的准确范围

用户授权完善并检验CPU-only缓存读取适配器，通过后在实验室服务器单次读取原N/U/D的548 development终点缓存及对应原标注，运行下述固定诊断，回传≤5 MiB摘要/引用/SHA并退出。实际已完成。只读原件，输出到新独立目录，禁止覆盖。**GPU0、模型forward0、训练/反向0、AP/AR计算0、图像读取0、checkpoint加载0、1602访问0。** 不启动或重演H2正式入口。

实际入口为`tools/r18_cache_diag.py check|execute --contract <path> --sha256 <SHA>`，依赖`r18_cache_worker.py`与`r18_cache_core.py`。真实执行前须完成`check_r18_cache_cli.py`实际CLI验收，并提交冻结运行字节；合成数学内核不替代这个入口。

## 2. 输入、身份与缓存能力

|臂|冻结来源|已交付索引范围|可用字段及不足|
|---|---|---|---|
|N|PR16 a97e1eaf750c69c4ee72af232d07632b1aced931，attempt-formal-003/infer-B0|137 raw引用，9437656B；top600/300引用|ids、FP32 logits[batch,300,10]、boxes[batch,300,4]；无粗框或训练梯度|
|U/D|PR21 e50bedbe8e41bdfd3c47d09f7076d9644e438e26，H2 attempt-formal-001/infer-U或D|各137 raw引用、12093812B；两档预测引用|同上，加第三层输入coarse；本诊断不借其代替N粗框|

所有缓存都在服务器原task目录；完整路径、字节数、SHA已在本地原infer RESULT引用中，本目录EVIDENCE.json只存最小入口/哈希，不复制411条原索引。逐原始字节核SHA后才可CPU `torch.load(map_location='cpu', weights_only=True)`；严格只接受这些索引列出的预测缓存，禁止接收checkpoint或任意pickle。按137批顺序流式处理，每臂必须恰好548个原稳定ID，各批4图；原manifest SHA `5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd`，不得从目录自动发现样本。

成员metadata/标注按该development清单的逐文件SHA核验；只访问其annotations，不打开images。既有N权重认证、原STOP及缺训练CLOSED仍沿原报告，本次对物理checkpoint只分块核SHA，不加载tensor。原源文件/配置的固定Git字节与EVIDENCE.json核对。原方案阶段远端状态未知，本次材料check和独立worker准入已通过；失败STOP规则保留，不能重推理补缓存。

U/D的同一GT按`stable_image_id + raw_annotation_id`对应；**不同模型query ID绝不能直接配对。** 训练Hungarian和下文诊断最大匹配不同，分别命名输出。N/U/D历史/同轮身份原样记录；不把终点关联解释成训练因果。

## 3. 坐标、总体、排序和固定计算

**坐标。** 缓存框是归一化原整图resize-only画布的cxcywh；复用冻结`r0/evaluation.py::rank_candidates`变为原图连续xyxy，不clip，不round，类别为1–10。主诊断IoU按连续正面积框；小目标以原标注raw area `0<area<=1024`，同时报告`1024<area<=9216`和`area>9216`两个互斥保护组，明确这不是COCO边界重叠口径。框/尺寸/score非有限或正面积不符均STOP。

**两套明确分母。** A为原训练有效GT（score=1、类别1–10、原正面积、clip后正面积），用于重建原最终Hungarian匹配；B为原raw-ignore raster预滤后仍保留的score1有效GT，用于输出覆盖诊断。分别保留raw ID、排除计数及两者交集/差集，不能把A中26554类匹配分母叫作AP小GT总数。score0类ignore、category0和其他类别不作为诊断正GT；也不将落在这些区域的预测自动称背景错误。B按原`bridge_inputs`/官方raster辅助函数准备，只到输入转换，不调用任何evaluate/accumulate/summarize或score_budget。原标注面积与clip框几何的区别显式输出。

**排序。** 每图所有300×10个FP32 sigmoid项，原规则为score降序、query ID、class ID。先选top600或top300，再原category0 raster ignore去除，不补位；原类别score0的匹配语义不在这里另造。用已有预测JSON核对转换结果，包括分数和坐标；不保存替换预测。FP32 sigmoid饱和可能产生原logit不同但分数相同，生产适配不得改为logit排序。合成脚本直接接收人为score，仅检查排序/计数逻辑，不能替代该生产一致性检查。

**固定阈值和图。** 仅IoU=.50/.75两档（区分粗覆盖与高IoU），不是调参扫描。对每个尺度组、每图建立GT到物理query的二分图：

\[
E_t^{all}=\{(g,q):\operatorname{IoU}(b_q,g)\ge t\};\quad
E_{t,K}=\{(g,q)\in E_t^{all}:(q,c_g)\in\mathrm{TopK}\};\quad
E_{t,K}^{keep}=\{(g,q)\in E_{t,K}:q\text{通过原raster ignore}\}.
\]

输出逐GT独立可覆盖数和最大基数一对一匹配数 \(M(E)\)，以及固定同分母下的差值。最大匹配仅为“不同GT不能共享一个物理query”的诊断容量约定；跨类也使用相同query唯一性，故**不是两评价器的匹配或召回上界**。忽略分类、对每GT找最优框的独立覆盖可重复使用同一框，应与M并列，不能只报更大的那个。

必须成立：\(0\le M(E^{keep})\le M(E_K)\le M(E^{all})\le n_{GT}\)。保留零覆盖GT。空GT组的率为null/UNKNOWN，不加epsilon。各尺度独立计算的匹配容量不可相加当全尺度容量；另做一次全GT图保留跨尺度竞争，其配对具体身份不是唯一真值。跨模型只比较共同GT独立覆盖状态的转移矩阵，不比较任意最大匹配解选择的GT身份。

为每个GT记录（服务器保留）几何支持有无、正确类是否留存、ignore前后状态、最佳IoU、最佳IoUquery的原名次，以及最高分正确类query的IoU；报告分布，不仅均值。几何支持丢失、预算丢失、ignore丢失依次定义为不相交的逐GT事件，另外输出同query竞争造成的容量差，**不能把这些诊断计数相加解释成AP损失分解**。

**B假设的特定事件。** CPU复用冻结原matcher及resize-only GT构造，对缓存最终预测重新得到终点Hungarian配对；不加载模型，不做损失更新，不重建DN。只读计算DECISION中的 \(\mathcal P_{600}\) 及其涉及的高IoU小GT数、图数/类别数，并分列落在A但不在B的事件。分母同时给出全部A小GT、已匹配A小GT、其中IoU≥.75数、这些高IoU正确类项被预算截掉数；没有分母就不报比例。对数不能当独立样本量。top300只展示同一缓存的兼容现象，不用来挑K。

## 4. 输出和解释

一份汇总表：N/U/D×尺度×IoU×预算的全部分母、几何覆盖、预算前后与ignore后覆盖、唯一query容量及共同GT状态转移；一份B限定事件表；若缺字段，记录UNKNOWN并停止依赖它的结论。保留所有负向差值、逐类和逐图数量，原始逐GT记录留服务器，不落Git或D。

这些都是终点描述，不能证明特征信息被丢失、模型容量不足、训练匹配不稳定或某结构造成训练退化。预算内全部保留仍可能有跨图排序和重复误报导致AP较低，本诊断对该部分保持UNKNOWN。它不产生oracle AP、不把GT辅助几何可用性称作可部署收益。

N本来可覆盖而U/D不可覆盖的GT，是H2新增几何损失的终点线索；N本身也未覆盖的GT才是原基线问题线索。二者均不定位到单个共享改动；只用于拒绝错误动机。结果写入一个新的诊断目录，引用旧报告，不再改写H2门槛或结果。

## 5. 阶段、上限和停止

|阶段|工作与终点|上限/准入|
|---|---|---|
|初次方案阶段已做|索引逐本地SHA、冻结源码字段和先例审查；合成数学/反例|当时0真实缓存/标注读取|
|本次CPU接线已完成|实际CLI输入/输出保护、原FP32排序/ignore/matcher适配及合成对照；受测版本提交冻结|不构造detector；CPU检查不冒充真实分布|
|本次单次真实诊断已完成|N/U/D同548图缓存+清单标注；完整覆盖检查、固定统计、SHA导出、退出|最多2 CPU线程、RSS≤4 GiB、累计wall≤30分钟、GPU0、forward0；D回传≤5 MiB，服务器新输出≤20 MiB|
|诊断后一次研究决定已完成|事件存在，但不足以保留B；当前无待执行候选|不自动GPU、不扩样本/阈值/候选；缺证不自动补推理|

预算依据：缓存411文件约32.07 MiB（33625280B），并已有两档预测索引用于核对；预测JSON总量约0.4 GiB，仅服务器顺序读取并及时释放。GT矩阵逐图计算，远小于训练状态。30分钟/4GiB是保守工程硬上限，**不是已测速度承诺**；哈希/读取/匹配均计入wall，触界STOP，输出partial说明，不删旧材料、不自动重试。资源/路径/身份/计数/数值异常同样STOP。样本不足如实报告，不现场扩图。

**本轮没有保留训练候选，GPU方案不激活。** 若下一次真正保留方法，先做纯CPU完整N/C/候选接线与保存恢复，再另申请一次短GPU工程验收（可参考三臂9实际更新/18forward、15分钟GPU/30分钟wall/18GiB；必须以新实现实测更新）。短窗仅检验工程、梯度/配对存在性，不能作科学质量筛选。

只有方法差异与常规控制明确、输入与计算配方冻结、CPU/短GPU工程通过、完整训练预算及科学标准另获授权后，才能进行e120完整试验。成本量级参考：三臂各120×304×2=218880训练forward，三臂548图batch4=411终点forward；三模型固定64、每模型2暖机+6测量为384forward；加18短验收，共219693forward。这只是未来同轮N/C/候选的计算账，不是当前授权或新冻结门槛。H2实测每新训臂约23.84/24.06阶段小时，按此规划三臂训练约72小时且另加后链；新候选实际速度未知，不能承诺沿旧60小时上限完成或静默超限。若资源不够，不缩短日程冒充e120质量结论，也不未经核验复用历史N省去同轮控制。

## 6. 本轮复核入口

在本任务worktree运行：

```text
python -B tools/r18_evidence_cpu.py --synthetic-check
python -B tools/repo_manage.py check
python -B -m unittest discover -s tests -p test_repo_management.py
git diff --check
```

脚本只内置构造框、分数和匹配标签，不接受生产文件参数。覆盖几何/预算/类别/ignore/密集竞争/稳定tie/空分母/排序充分性反例及B纸面hinge有限差分。输出`NOT_SCIENTIFIC`；通过不认定N存在该问题或B机制成立。生产CLI已经单独验收并执行，不用本合成CLI替代。

## 7. 按用户授权补全的语义、绑定和执行保护

- **截断前证据：** `infer/RESULT.json`、CLOSED、原合同/source/endpoint/binding、物理checkpoint分块SHA，以及411份raw缓存的逐SHA/size/键/FP32 shape/548稳定ID均核验。仅raw缓存用CPU weights_only加载，完整300×10 logits和300×4 boxes必须存在；不是以topK预测推断几何缺失。原H2和N源码清单逐项核SHA，导入共同辅助函数前先验证冻结字节。
- **类别支持：** 不添加置信阈值。每query的10个有限分数不等于10个语义正确候选。以argmax正确类作并列描述；原生多类topK可保留非argmax正确类项，故argmax不是后续预算统计的必经滤器。原几何→正确类项进预算→ignore后留存的嵌套图定义不变，argmax单列，不强行唯一归因。
- **两个匹配口径：** 原方案的一query一GT最大容量和固定score/q/class顺序的诊断贪心匹配分列；后者按IoU最大再GT顺序选取，只是物理query竞争描述，不能冒充评价TP。另按用户要求核“最终匹配”：仅调用冻结主评价器`_match`，沿原raw-ignore、先全图maxDets500再分类、score0 ignore语义；只取.50/.75两档GT匹配标记和计数，并与六份原评分JSON中已保存的对应计数严格一致性检查。**不调用evaluate/accumulate/summarize、_voc_ap或AP/AR运算。** COCO逐GT匹配身份没有读取记录，本次不另跑COCO评价，保持UNKNOWN。两匹配口径不可混为一条严格嵌套链，因为主评价器可在不同类别使用同一物理query。
- **分母与配对：** A为训练有效GT；B为raw-ignore后score1有效GT。B可包含clip后不属于A的原始GT，交/差集显式计数。覆盖使用原图raw框连续IoU；CPU原生Hungarian重建使用原resize-only FP32目标。U/D与既有CENTER_RECORDS比对匹配身份并披露CPU/GPU舍入差异；N原匹配未记录，只能记CPU重建，不能声称还原训练过程。跨模型仅用图像/GT原始ID配对。
- **原冻结统计和资源：** 仍仅top600/300、IoU.50/.75、预定义互斥raw面积小中大；类别/序列必要汇总限定高IoU的小/大目标。执行进程CPU affinity限定两核、Torch/BLAS两线程；监督器监控自身+唯一worker RSS、累计wall及新任务目录字节，触界仅终止本次自有worker并保存STOP。真实execute以独占目录消费一次，旧输出拒绝，异常不重试。workers内禁止detector模块调用、反向、CUDA初始化和AP/AR入口，审计拒绝图像及禁用数据目录读取。正常返回后独立SHA核验小回执，不执行第二次诊断。
- **部署：** 逐Git字节复用PR21的`tools/deploy_h2_source.py`及其10项实际CLI检查，安装独立新release；不使用临时安装器、不改旧release，不更改实验环境。源码小包、运行合同和诊断输出合计受20MiB保护；GT逐项CSV压缩原件只留服务器，精简摘要/合同/闭合/SHA回传≤5MiB。

## 8. 本次闭合与禁止重放

执行commit09e5217edfd7e3cb8aed79c09a51d8f8a7c4a751；独立task `r18-baseline-cache-20261026-001/attempt-cpu-001`，14:07:49+08只启动一次。CLOSED/EXIT PASS，99.902062秒、峰总RSS1127411712B、affinity[0,1]，411原生CPU matcher调用、548×3完整覆盖；全部原生排序/坐标与预测一致，六评分的120个原计数向量一致，U/D保存的小GT匹配身份各26554项完全一致。没有训练、新detector forward、GPU、AP/AR或1602访问。约7.82MiB GT逐项压缩CSV仅服务器，15文件66206B必要回传逐SHA通过。

材料索引、统计表、原始SHA及三个结论见[诊断结果索引](DIAGNOSTIC_RESULT.json)和[研究决定第5–6节](DECISION.md)。原H2/P FAIL、旧STOP及冻结门槛保留。任何后续诊断、训练或方法执行均不在当前已消费授权内，不能通过更换attempt目录重复运行。
