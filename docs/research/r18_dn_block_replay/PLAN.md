# DN中心—尺寸联合误差回放：限定定义与短探索方案 v1
状态：CPU原型；未执行GPU、真实图像/特征读取、学习或AP/AR。只保留一个问题，不认定原创性或收益。
本文件替代2026-10-30一页选题中“逐维置乱”的旧提法；现在只有原生C、联合J、块置乱B三臂。仍替换固定约一半正DN，不增加DN数量。

## 1. 假设与证据边界
原N存在中等IoU几何支持而高IoU不足，中心/尺寸敏感性混合。这不证明DN失配是根因。
唯一假设：在相同中心二维块、尺寸二维块多重集和抽样次数下，把终点普通query误差的中心—尺寸依赖送入DN重建任务，可能比打断该依赖更有利于短窗普通query定位。
J−B是核心否定控制；J−C只说明经验扰动整套选择相对原生DN的结果，不能单独证明相关性有价值。没有逐维置乱第四臂。

## 2. 固定误差来源和坐标
未来从认证原N e120 EMA（不是N24、S/A或历史验收state）eval模式、固定768×1344预处理下，取decoder最后/第三层全部300个普通query的pred_boxes/pred_logits，排除DN、encoder输出及topK预算后结果。
实际位置：vendor/.../rtdetrv2_decoder.py的decoder输出out_bboxes[-1]及out_logits[-1]；初始reference在_get_decoder_input中另行生成并detach，本方案不取它。
采用原共同协议HungarianMatcher：focal分类cost=2，cxcywh L1=5，GIoU=2，alpha=.25、gamma=2；按稳定图ID、GT原ID、query序号输入，沿已绑定SciPy linear_sum_assignment返回的确定性并列处理，不另加epsilon改变cost。执行环境须绑定SciPy版本；同一输入重复CPU匹配不一致则STOP。
GT沿认证train_core的类别映射、过滤和raw-ignore规则；ignore不作为正GT，未匹配普通query不进入误差库，逐图记query/GT未匹配数量；不以分数、正确类别或IoU挑框。零IoU匹配必须保留。非法、非有限、非正宽高使材料整体拒绝，不能删除该条后继续。
对匹配预测b及GT g（归一化cxcywh）：
e=( (bx−gx)/gw, (by−gy)/gh, ln(bw/gw), ln(bh/gh) )=(c,s)。
库记录图ID/GT-ID/query-ID/阶段/类别/原图GT面积/当前画布GT面积/原始b,g和e，全部detach、只读、一次生成后固定。
**终点残差是拟注入的经验难例先验，不是普通query初始reference误差分布。** 已收敛残差可能过小、偏向仍困难的匹配对象，且DN有GT标签、普通query没有；这两种任务的迁移效用未知。没有把该选择当作分布失配诊断。

## 3. 注入和跨块控制
分组用当前输入画布GT面积a=gw×1344×gh×768：a<32²、32²≤a<96²、a≥96²。校准和训练都按当前视图面积分组；这是扰动分层，不冒充COCO原图small定义。probe报告另用原图面积。
先完整调用未修改的原生get_contrastive_denoising_training_group，消耗完全相同的原生RNG，保留生成结果。有效正DN按(batch行,DN槽)字典序，取第0,2,4,…槽替换，即ceil(P/2)；另外正DN、全部负DN和padding不变。选择与目标难度/误差无关。
每microbatch用独立CPU Generator，seed=20261030+microbatch_index。每尺度组有放回抽取对应数量donor；J使用完整向量；B保持同一中心块逐槽不动，仅将该组已抽尺寸二维块循环移动一个均匀抽取的非零步长。维度内部绝不拆散；两臂共享donor列表、位置和次数，步长也在J中生成但不用。组内仅1槽时不动并计数；重复donor造成实际不变照实报告。
重建：b'c=g_c+g_wh⊙c，b'wh=g_wh⊙exp(s)。转xyxy后沿原规则逐角clip到[0,1]，再转cxcywh，再调用原inverse_sigmoid：log(max(x,1e−5)/max(1−x,1e−5))。进入decoder的有效reference是sigmoid(unact)，不等同于裁边前框。
不增加幅度系数、半径、IoU筛选、排序、回退粗框或重抽样。溢出/非有限/裁边后零宽高均保留审计并STOP。
库与重建无梯度；原标签embedding、decoder/共享特征按原DN损失反传；reference迭代detach、普通matcher、aux及DN损失权重/分母保持原实现。推理不使用库或GT。
当前原型dn_entry显式调用真实vendor DN函数后替换正坐标；vendor、模型、criterion均未改。**完整decoder/训练接线尚未实施，不能把本CPU入口验收称为完整模型验收。**

## 4. 注入后的可比性及正负语义关口
记录每尺度组的总正DN/替换数/未改正DN/负DN/padding、donor与块置换、实际移动比例，注入前和裁边/逆sigmoid后各4维均值、分位数和中心×尺寸2×2协方差，裁边/无效/零IoU/正负相对难度数量及分母。
J/B误差库边缘相同不意味着绝对DN框集合相同。即使不裁边，尺寸块被分给不同recipient也改变绝对框分布；结论限定于分组后的**GT归一化误差**。
更强的可检验关口：重编码最终有效reference，比较J/B同槽中心块，以及按已知置换对应的尺寸块。每分量最大差≤2e−5才允许“仅跨块配对改变”的解释；这强于只比较均值。裁边前、裁边后、最终注入后分开报告。任何有效性失败或超差，整窗STOP，不删难例/调幅/补图/配平IoU。真实材料可能无法通过，此时结束该固定注入定义，而不是继续学习后仍宣称控制成立。
正/负角色由原dn_positive_idx、dn_num_group、attention mask和criterion决定；同GT同组负槽=正槽+max_gt。负DN原角点扰动幅度[1,2)、正DN[0,1)，裁边后原生也不保证IoU严格排序。
为避免把明显改变正负任务作为隐含变量，预先规定：替换槽中“正IoU<配对负IoU”的比例相对该槽原生正DN增量≤5pp，正零IoU比例增量≤2pp；分母为替换有效正槽。分别检查J和B，不要求J/B IoU配平，不改变任何标签/负DN，也不把失败槽跳过。阈值为本轮探索的任务保持规则，非原研究质量门槛。
每窗原始计数保留，跨窗及各尺度另聚合。实际尺寸块不同的槽比例不足80%、小尺度有效donor<100或probe配对小GT<100：证据不足，停止，不现场扩样。单槽与重复抽样保留。

## 5. 直接先例核对（限定范围）
|直接来源|已核实机制|本定义的可检验差异与未知|
|---|---|---|
|[DINO](https://arxiv.org/html/2203.03605)，§3.3|有正/负DN及重建/背景监督；GT扰动、去噪本身已有|保留原规则，只研究部分正DN的经验二维块跨依赖；不是发明DN|
|[RHINO v3](https://arxiv.org/html/2305.07598v3)，§3.4式3–6|将refined正DN与普通预测合并参与Hungarian，决定正DN保留原GT监督还是转背景；已用模型相关信息调整DN|本方案固定数量/角色/损失，不依据当前预测改标签或筛query；固定终点残差库及跨块控制。不能统称“首个自适应DN”|
|[Ray Denoising](https://arxiv.org/html/2402.03634v2)，§3.2–3.4|沿相机射线生成3D深度扰动，采用分布族模拟深度歧义造成的假阳性|结构化/误差针对性的扰动已有；这里检验2D中心—尺寸经验联合残差，不使用3D几何，也不据此声称全新整体机制|
限定搜索还命中D2OM的Matching-guided Adaptive Denoising及高斯衰减DN摘要，但未完成可访问正文核实，不能据摘要判定逐式差异或排除重复。**未找到并核实完全相同的固定经验库跨块控制，不等于证明不存在先例。** 新意与效用均未建立；不继续泛搜。上述三项正文已核对，未声称复现其代码或性能。

## 6. 冻结短探索规则（待授权；尚非执行合同）
按sha256("dn-block-v1:"+稳定图ID)排序认证train_core；前64图校准、随后128图学习、随后32图probe，集合互斥。数量不足拒绝。只按ID选，不读取目标数重选。均属于原N见过的训练集；不是未见验证。不得访问548/1602。
三臂从同一原N e120 EMA模型参数和缓冲独立开始，optimizer/EMA/RNG重新建立，不接旧optimizer。探针用固定u100 raw模型，既不早停也不挑EMA/最优步。建议AdamW lr=1e−5恒定、betas(.9,.999)、weight_decay1e−4，norm/BN零衰减，clip_norm=.1；100更新、batch8×accum2，原BF16/确定性/增强配置，三臂相同200个microbatch ID/增强/原生DN随机序列。该短窗日程是待审设计，不是e120收益代理；实施前需把现有训练器的日程差异单独受验，不得悄悄套原2000步warmup。
probe初态用原N最后层普通query经同一matcher得到GT—query一对一关系并冻结；末态三臂沿同图/GT/query索引比较，不重新匹配，不按类别正确或原IoU过滤；未匹配GT另报全GT分母，末态非法框IoU计0并保留。该固定身份指标反映短窗query对应的定位，query角色重排可能使其恶化，不能解读为AP。
主要量：原图small配对IoU的J−B平均差；最小有意义差异预定+0.005（绝对IoU），并要求J−C≥0、J相对初态退化比例比B不多于2pp（退化阈值−1e−6），中/大目标J−C平均差均≥−0.005。逐尺度报数量、分母、均值/中位数/分位数及改善/持平/退化；少于30个的中/大组保护项UNKNOWN，不自动记通过。保留全部负向项。
达到上述条件只说明值得进一步检验的固定表征与全模型共同更新短窗信号；并非检测收益确认。J/B无差或J更差：终止本次固定短窗的支持，不据此否定所有DN或完整训练。J/B均超过C但无J优势：经验噪声整体可能有用，跨块依赖额外价值未获支持，不自动换方法。
工程异常/资源超限、输入或身份不一致、控制/语义关口失败立即STOP，无重试/调参/扩图。首个真实残差库不足或注入控制失败即止于准备阶段。probe从不参与参数更新或选模。后续完整训练另行决定。

## 7. 预算与本轮交付
|未来阶段|检测器forward|optimizer更新|用途|
|---|---:|---:|---|
|64图校准，batch4|16|0|真实误差库；非冻结特征探针学习|
|32图初态probe，batch4|8|0|一次共同初始关联|
|C/J/B，各100更新、2 microbatch|600|300|端到端短窗|
|三臂各32图终态probe，batch4|24|0|固定关联定位统计|
|总上限|648|300|无额外预热/验收forward|
预算是重新逐项核算后仍为648，不是沿旧数字自动授权。建议独立上限GPU阶段1800s、wall3600s、allocated21GiB、新增2GiB、精简回传1MiB；时间是硬上限提案，不是实测完成时长。所有失败成本计入，不借用耗尽旧预算；存储/家族余额需新鲜核验。若实现真实入口需要额外验收forward，先修订预算。
本轮只有合成tensor/实际DN生成器CPU操作，无检测器、真实权重、图像、特征或学习。CPU命令：
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python -B tools/check_dn_block_replay.py
```
真实误差采集、训练注入接线、资源合同和完整CLI尚未实现；本轮交付不能启动GPU。下一授权应先补这些入口及CPU完整闭环，再单独批准真实短探索。

## 8. 本轮CPU结果与仍未知项
2026-10-30服务器认证解释器，CUDA隐藏、2线程；最终实际CLI 15/15 PASS。进程wall 1.15s，最大RSS 522276 KiB；测试内部计时0.015s不替代进程wall。检测器forward=0，optimizer更新=0，CUDA未初始化。先前13/14项开发运行及一次源签名定位断言失败均保留于服务器research_cpu/dn-block-replay-20261030，不覆盖日志。
|核验|结果/限制|
|---|---|
|独立GT坐标闭环、非方形/零IoU|PASS；FP64独立预期，与实现非同公式抄写|
|二维块完整性、同尺度置换、多重集、私有RNG|PASS；单槽、重复donor不冒充有效去相关|
|真实vendor DN旁路、标签/mask/meta、负DN、padding、未替换正DN|PASS；密集101GT触发单组及空行覆盖|
|裁边与inverse_sigmoid|保留反例：裁边前多重集相同，注入后最大块差0.431798736；关口拒绝。eps改变近边界坐标，不能称恒等|
|异常/语义变化|非法/非有限/缺尺度库/明显恶化正DN关系拒绝，不重新抽样|
|源码作用范围|只新增研究adapter/测试/文档/CI，未改vendor、model、criterion、原release。DN/matcher/decoder与原科学79ba593对应三文件逐字节一致|
|未覆盖|真实残差分布、真实材料控制可行性、完整decoder接线、CUDA/BF16整模型、学习收益均UNKNOWN|
CPU通过不建立经验误差库质量、监督迁移效果或论文贡献。保留该问题进入**入口准备**，没有真实探索启动授权。


## 9. 完整执行入口准备（2026-10-30追加，尚无GPU授权）
以上1—8节保留为6228e20原定义与原型阶段记录，不覆盖历史状态。当前追加真实采集、完整decoder注入、三臂短学习及闭合入口；科学统计和门槛不变。

执行顺序固定为：认证N e120 EMA的64图校准（16 forward）→库闭合及全部100训练窗口/200 microbatch材料与注入检查→32图共同初态探针（8 forward）→C/J/B各100更新（600 forward）→三臂终态raw探针（24 forward）→比较与导出。任一库/控制检查失败在学习前退出，不重采donor、不删难例、不扩大样本。128图按已定义稳定ID选择，logical16共8窗口/虚拟epoch，循环12整轮及4窗口；学习是固定100更新，非e120。

全窗口预检是独立进程，不构建检测器。读取相同增强窗口，复用真实native DN及独立Replay RNG；真实执行时在相同CUDA设备计算DN随机操作，以核对设备相关原生随机序列，因此预检耗时/资源属于整链，但检测器forward为0。主进程重新建立模型、数据迭代及RNG，不接续预检游标；运行中逐窗核对图像/GT/增强哈希与DN输入、标签、mask、meta和RNG摘要。J/B donor列表及中心/尺寸二维块多重集、裁边后误差和正负语义按原关口检查，不事后配平IoU。若真实模型出现预检未包含的随机调用，立即STOP，不放宽比较。

完整状态含model/optimizer/EMA/RNG、数据位置和DN位置，实际CPU CLI验证u1保存→u2→恢复u1重放u2逐项一致；入口无自动恢复或扩大训练路径。探针eval/no-grad，核参数/buffer/RNG不变，固定初态query关联；无效终点框按IoU0保留。中/大样本不足30时保护为UNKNOWN，即便观察到正向主指标也不得判全部科学PASS。

实现复用原Engine、增强、matcher、criterion及DN。原Engine仅将原学习率调用抽为默认等价rate方法，短入口覆盖为原定义常数1e-5；decoder只增加可选DN调用器，默认仍调用原函数。原误差adapter公式及监督定义不改。原生C与J/B共有参数、BN、输入、监督/分母及optimizer/EMA初态一致；CPU DN-only反向确认预定三层DN损失及分母、共享backbone与DN embedding的有限非零梯度。

服务器实际CLI CPU验收使用明确合成图像和构造父状态，运行launch→supervise→execute→独立precheck→三臂→导出。覆盖双worker实际tensor传输、跨epoch、完整重放、错误材料STOP、源码/预算拒绝、重复启动拒绝和监督进程超时。最终验收记录另以对应Git/冻结交付索引绑定；单次完整构造检查53次CPU检测器forward（12构造+41管线）、10次CPU实际更新，全部不计未来GPU预算。早期导入/夹具/摘要错误与复测日志保留；扩张fixture裁边后B出现4项裁边、J为0及最大块差约0.00041955，按2e-5原关口拒绝，未继续学习。

真实准备只读取认证成员清单和核checkpoint文件SHA，不读取真实图像或把真实权重加载进模型。实际真实CUDA/BF16、真实误差库及全部窗口控制可行性、物理显存/耗时、学习收益仍UNKNOWN；CPU通过不替代CUDA验收或科学结果。没有额外隐含GPU验收forward。

待批独立上限保持648 forward/300更新、1800秒GPU阶段/3600秒wall、21GiB allocated、2GiB新增（含任务临时目录/编译cache）、1MiB精简回传。GPU时间保守按整个执行子进程wall计，硬监督超时不重试；原研究累计62111 forward/78363.83911894578 GPU秒只读继承，原正式forward余额0。本提案不是余额转移或执行授权。实际执行前重新核固定GPU身份、无其他计算任务、21GiB之外至少1GiB空闲余量、native allocator、环境与磁盘余量。模板approved=false，未创建真实授权/attempt。唯一命令与完整commit/manifest/合同由tools/package_dn_block.py从已提交Git字节冻结后生成；须另行批准该固定身份方可启动。


## 10. 统计设备修复与只读校准库复用预检（2026-10-30）
原aa3213ab唯一执行完成64图校准16forward/0更新，库small donor2395；随后microbatch0因quantile的q张量在CPU、输入CUDA工程STOP，成本14.0458804811351秒保留。未执行任何学习或probe。原始partial audit为335槽、J/B裁边5/2、无效1/0，但该次未完成控制判定。

限定修复只为distribution中q补device=x.device，保留原dtype=x.dtype及x.double()；公式、抽样、边界、容差及语义关口均不改。统计路径其它常量中donors/perm属于明确CPU抽样，后续已有to(device)；zeros_like及mask已有对应设备，无其它遗漏被发现。新dn_block_recheck.py不暴露模型构建、探针、学习、optimizer或恢复入口，只能校验、最小统计检查、冻结和一次库复用预检。
旧calibration/library分别绑定SHA 3addf6f9f80d4ea852f995ccff262c3d4aa5e95a9907911269ed9c508764ca76 / 3cabea9aad68c64aec6cabcab4f0e362228fb68c72922d391e44e50dd703c246。逐项核旧423文件、父N bytes、64图ID及query/GT/误差重建、闭合回执和启动合同。无权重模型加载，无重新采集。

CPU实际DN入口与独立GT反例检查；FP32/FP64/BF16输入统一原FP64统计，预先固定CPU/CUDA统计比较rtol=atol=1e-12；空输入、semantic统计及无效框拒绝覆盖。零宽框经epsilon变为正的最终reference不消除原invalid标记。最小CUDA只做构造统计/算子，不调用检测器，时间与唯一真实预检合计扣本次300秒，非额外免费验证。

旧错误在observation写入前发生，旧文件无GT/donor具体坐标及完整input/RNG摘要。唯一重放用原loader、seed0、原生CUDA DN、独立Replay seed20261030+microbatch索引重建；记录新input/RNG摘要及与旧partial audit的严格一致性。不能声称与未保存的旧完整摘要逐位比对。发生无效框时保留recipient GT/ID、donor索引/来源/error、裁边前后框、prospective final reference和原无效判据；其reference仅计算取证，实际注入被拒绝。首次关口失败即退出，不通过修正、重抽或删除样本继续。

本次零检测器forward/零更新，300秒GPU阶段/600秒wall，21GiB，原探索累计2GiB存储及1MiB回传（旧回传按100000B保守预留）。旧16forward和14.045880秒不退回。专属临时目录/新包/新attempt纳入累计存储。统计修复工程PASS、固定控制PASS/FAIL和学习收益NOT_TESTED分列；全部预检PASS也退出，不自动进入学习。原release、STOP、库及账本保持只读，新费用独立追加。
