# 多视图教师向单视图R18学生迁移：唯一可行性准备

状态：**唯一配方及CPU工程准备已形成；不授予GPU/训练执行，先例审查仍有一个明确缺件。** 本轮没有科学结果，不预设有效性或原创性。旧稿、PR #10（研究准备已结束，未合并）、原A/B/C和全部失败候选不变。此任务独立于已关闭切片微调：学生没有固定切片输入，教师信息才是变量。

## 1. 可否定的问题与控制

在共同协议R18 seed0 e120 EMA相同初始化、相同整图学生与同一短期优化预算下，**离线C的预测监督，是否比监督续训及整图教师监督都更能提高单视图A的精度**？若T2不能同时超过T0、T1与原A，或突破保护/成本条件，关闭这一个固定配方。失败只约束本配方，不否定所有蒸馏。通过也只是反复开发集上的单seed可行性，不证明跨seed、独立数据或新方法贡献。

|臂|共同学生/起点|教师|唯一差异/局限|
|---|---|---|---|
|T0|原共同协议R18；严格提取seed0 e120 `EMA.module`|无|监督续训控制|
|T1|相同父EMA，独立模型/optimizer/新EMA/RNG|冻结父EMA的原整图view0|整图教师也用与T2相同hard NMS/置信门/预算，是B式教师，不伪称未经NMS的A|
|T2|同上|相同父EMA，原C view0+四角切片|模型相同，只扩教师视图；产生的框数、分数、类别分布和匹配会随之变化，不能单独归因于分辨率或新增细节|

T1复用T2的认证view0，避免重复forward。548 development预测及1602补充数据都不是训练教师输入/监督。训练只有原`train_core`4869，manifest SHA `df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d`；原训练数据历史边界不另作“完全独立”声明。

父checkpoint固定SHA `13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757`，323460353 bytes；父绑定15150 bytes/SHA `93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938`。未来逐字节验证后才提取EMA；不是继续父optimizer/RNG/EMA计数。三个学生分别独立构造、严格load，同权重和BN buffer起步，AdamW与EMA计数归零，之后各自训练。

## 2. 唯一公式：置信加权的正预测集合蒸馏

只用教师**最后正常decoder输出**，不蒸馏feature、DN、encoder/aux或query先验，不改变原GT匹配。教师eval/冻结/FP32；每图五视图各Q300×10 sigmoid，按原FP32解码到各视图像素，再用原C整数crop原点映射回整图。切片尺寸与位置完全复用`slice_core.tile_windows`，无GT选位置。

T1/T2均执行：每视图top600 → 原坐标类内hard NMS（仅IoU>0.5抑制）→ global top600 → score≥0.5 → top300教师目标 → 实际学生增强对齐/训练ignore → 不补位。排序始终分数降序、view/query/category升序；同query多类别不新增跨类别抑制，不把query ID当师生对应。阈值0.5是预先固定的保守正预测门，**未经精度校准**；无温度、无权重/阈值扫描。候选门只读预测；后续训练ignore只读对应训练图的原标注，不使用GT IoU筛选“正确”教师框。

令增强后的教师集合为 `(b_j,c_j,q_j)`，至多300项；学生最终正常head为 `(s_i,z_i)`，均在增强后画布归一化cxcywh坐标。直接复用本库原生HungarianMatcher，对独立于GT的教师集合求一次一对一对应π：

`π = Hungarian(2*focal_cost(c_j,s_i; alpha=.25,gamma=2) + 5*||b_j-z_i||_1 - 2*GIoU(b_j,z_i))`。

无query对应假设，不额外引入新的匹配代价或GT约束。原GT↔学生的监督匹配仍由原criterion自己计算；两个匹配可能对同一query产生不同监督，这属于本方法风险，不能声称已解决冲突。

单physical batch的唯一KD损失：

`L_KD = sum_j q_j * [ KL(Ber(q_j) || Ber(sigmoid(s_{π(j),c_j}))) + 5*||b_j-z_{π(j)}||_1 + 2*(1-GIoU(b_j,z_{π(j)})) ] / max(1,sum_j q_j)`。

`L_T0=L_supervised`；`L_T1=L_supervised+L_KD(T1)`；`L_T2=L_supervised+L_KD(T2)`，λ=1，温度1。只选中类别logit参与KL，其他类别、未匹配学生及教师负框没有额外KD；这不是完整类别分布蒸馏。教师框、分数stop-gradient，所有损失FP32；KL用BCE减Bernoulli熵，q=1的0log0由xlogy处理。空教师/空GT不丢图、不重采样；KD给可微0，监督仍按原GT/空GT规则。

`L_supervised`原正常/aux/encoder/DN VFL、L1、GIoU的所有原权重、归一化和训练raw-ignore保留。两个等大物理microbatch各乘0.5，累积后clip0.1、一步optimizer和一步新EMA。KD按置信和归一，而监督按原GT数归一；T2教师数和质量分布不同仍可能改变梯度方向/噪声，归一化不能消除这一差异。

### 增强与ignore的明确冻结选择

**三臂均用父e120末期已有的弱增强：sanitize→原随机水平翻转→resize768×1344→sanitize，强增强整个12轮关闭。** 不是旧微调“强9弱3”配方，也不把旧T-control结果当T0。原source order沿用seed0 sampler epochs109..120；增强随机时钟使用118..129，使原RawAugmentation始终进入既有末期弱路径。三臂相同source/position/image ID/seed，不让教师参与随机抽样；cache不读取开发/补充集合。

选此方案的科学依据：从已收敛、强增强已关闭的父EMA做小LR短续训，维持其末期图像分布，并将本次问题限制到预测监督。工程依据：cpu-001发现本地TorchVision0.23下旧CaptureZoom/Crop的`_get_params`回调未记录真实强几何，不能用不完整trace对齐离线教师。本轮**未修改旧增强/旧release**；在冻结前只选择一套可证实对齐的共同弱增强，不凭结果改参。此处发现不反向重裁旧运行；旧服务器环境回调行为未重新执行核实。生产适配会拒绝strong trace，绝不静默套用缺失几何。

教师原框初始clip到原图，按实际flip/resize及原min_size1进行训练GT式几何处理，原坐标审计保留。该clip是KD目标对齐，**不是修改C评价预测**。训练ignore复用原`negative_keep_mask`连续union/类别IoA≥0.5，对pseudo selected class使用空matched exemptions；不借用评价栅格规则，不将教师目标一律豁免。目标处理后不重新NMS、补位或resample。学生GT仍走原parser/训练ignore。该规则属于新训练协议，不称其与原C输出集合逐项相同。

## 3. 三臂固定训练与终点

- 各12轮×304=3648更新，三臂10944更新；每更新16学生视图，物理8×累积2，BF16 forward、FP32 loss，无scaler。
- 每臂58368张增强输入，共175104。每轮4869按原drop-last规则使用4864，绝不以教师分数改变采样。
- 新AdamW：原norm/bn分组WD0，其余1e-4，betas .9/.999；新LR峰1e-5，前200更新线性warmup，后cosine至3648的1e-6。
- 新EMA从该臂刚加载的父EMA复制，包括buffer；updates0/decay.9999/warmups2000。每臂构造完成后复位seed0 Python/NumPy/CPU及未来CUDA RNG；不接收父RNG。
- 默认T0→T1→T2顺序。每轮保存完整状态，固定本地FT012 EMA为唯一评价终点，不看中途AP、不选最佳epoch。
- 日程/优化预算仅为12轮可行性筛选，未证明最优；未授权扩到e120、其他seed或参数搜索。

旧双臂微调7296更新及其GPU验收/评价完整链实测12593.1099s≈3.50h；它与本次弱增强/KD工作量不同。可用它估计训练数量级，不能直接当本次实测。原切片微调T-control整图精度曾下降，故本轮必须同时对原A与T0判断，不能仅用相对较弱T0的正差宣称进步。

## 4. 预先数值条件及理由

主判断为**原A式单视图**的COCO bridge未舍入0..1值相减×100，原primary完整单列；不混用指标，不新增bootstrap。每个比较均独立通过，不能用均值抵消。

|T2相对对象|600 AP|600 APS|600 AP75-small|600 AP75|600 APM/APL/AR-large各|300 AP/APS各|
|---|---:|---:|---:|---:|---:|---:|
|T0监督续训|≥+.20pp|≥+.30pp|≥−.10pp|≥−.10pp|≥−.30pp|≥−.10pp|
|T1整图教师|≥+.20pp|≥+.30pp|≥−.10pp|≥−.10pp|≥−.30pp|≥−.10pp|
|原A父模型|≥+.40pp|≥+.60pp|≥0|≥−.10pp|≥−.30pp|≥−.10pp|

共27条件。**这是本任务新的实用最小效应标准，不是旧C门槛，不是统计显著性或期刊门槛。** 原seed0 C−A约+1.98AP/+2.79APS，本轮相对原A要求约1/5该改善、同时保持一个单视图，才值得额外教师/训练投入；.20/.30的控制增益是区分续训/普通自蒸馏的最低增量选择，非已估计检测能力。严格AP75-small相对原A不降及AR-large保护，针对旧C和旧微调的高IoU/大目标代价；.10/.30是明确的容许退化预算，未被证明是噪声界限。单seed开发集通过仍需另行决定复验。

同步报告T0−A、T1−T0、T1−A、T2−C及AP50/AP75、尺度AP/AR、两预算；C只复用原认证指标，不运行C学生推理或重评分旧AP。T2−T1隔离的是这套教师信息包，不能进一步把框数、视角、定位等因素分离归因。

成本保护：同轮A/T2端到端耗时比≤1.05、峰值allocated显存比≤1.05，并核学生结构和推理路径完全相同。5%是未改结构单视图的工程允许偏差，不是“免费”断言；包含预处理、H2D、forward、解码/排序及postprocess，CUDA同步。报告reserved显存但不将其作为复用allocator时的主要比值。固定development64不计算AP。另报告T0/T1成本，不用旧A耗时作同轮分母。

## 5. 预算与未来执行顺序（未执行）

|阶段|固定数量|依据/范围|
|---|---:|---|
|训练教师一次五视图|4869×5=24345 view-images；batch4共6090 calls|ceil(4869/4)×5；每view最后batch1；T1共享view0的1218calls|
|三臂GPU恢复验收|每臂3个actual updates，共9更新/18 forward-backward calls|1保存→2→恢复1重放2；正式重新父EMA更新0，禁止接验收state|
|三臂正式训练|10944更新，21888 batch8 forward-backward calls|每臂3648×2，不含前行验收|
|三臂学生终点推理|1644图，batch4共411calls|548×3，同一raw用于600/300|
|固定64同轮成本|A/T0/T1/T2各2整套warmup+4计时重复，共384calls|4模型×6遍×16batch；timed顺序四轮拉丁方如下|
|新评价|6组×双评价器=12调用|三学生×两预算；只有548；旧A/C直接复用|
|总新GPU forward预算|**28791calls**|6090+18+21888+411+384，含训练，不是全部纯推理；0已执行|

计时每轮顺序：A,T0,T1,T2；T0,T1,T2,A；T1,T2,A,T0；T2,A,T0,T1。warmup先按共同顺序各两遍，正式计时每模型四遍64全图，报告单轮/均值/样本SD、峰显存及比值；不从快的重复中挑选。验收和正式不额外做未计入的试forward。若需要改变预算或测速工况，先改执行计划获授权，不偷增。

预计GPU **5.5–8.0h**（纯数量级估计，不是截止）：三臂约4.5–6.5h训练，教师/缓存/恢复/评价等待约1–1.5h；CPU双评价器与I/O可能延长总墙钟。依据仅旧7296更新完整链3.50h及固定C64约4.75–4.81×A，非本次实测。原raw浮点下限约409MB，紧凑JSON候选可能数GB；全checkpoint/审计按≥80GiB空闲盘准备，正式前按实际文件峰值重新准入。大型缓存、checkpoint和全审计留实验室；D仅计划/指标/回执/SHA。

未来执行在**另外明确GPU授权**后：最终运行入口和source/manifest/contract绑定 → 先例缺件审阅与GPU入口闭环 → 新鲜资源/训练4869及父checkpoint逐SHA准入 → 一次训练集教师缓存生成/闭合/原始输出重建（先最初8图检查，包含预算内）→ 三臂3更新GPU保存恢复验收 → 三臂各从父EMA重新正式12轮 → 固定端点/6组双评价 → 成本/比较/精简导出/进程退出。**本准备入口目前只实现prepare/cpu-check，不冒称GPU总launcher、训练缓存生产合同或监督器已经验收。** 下一授权须明确包括这些生产封装及实际运行，不能只改布尔值解锁。

新生产工程/数值/身份/资源/恢复异常立即STOP，保留现场，不修复重试、不自动续训。科学条件失败仍完成本次三臂所有固定终点评价，再关闭本固定配方；无数据驱动中停/扩轮/换seed/改教师阈值。新鲜资源沿原实验室保护框架，GPU/CPU温度、显存、RAM、磁盘、功耗/频率、无进展与异源进程准入在将来合同显式绑定；本轮未连服务器，不能声称当前资源足够。

## 6. 文献核对与缺件

用户提供DETRDistill.pdf逐页解析共10页，SHA `fd33fd5257b2c571cac79b957708b4f1e5b0df320b704ac1bcc403e417cb9a91`，7292695 bytes。页内题名 *DETRDistill: A Universal Knowledge Distillation Framework for DETR-families*，Jiahao Chang、Shuo Wang、Hai-Ming Xu、Zehui Chen、Chenhongyi Yang、Feng Zhao；arXiv:2211.10156v4 /17 Mar2023。按该预印本字节登记，未用其推定正式会议版/DOI或出版社字节认证；10页无独立补充正文。正文§5.2提到补充中的轻骨干/自蒸馏，当前文件没有这些材料，不能写已核。

|先例/核查程度|已经回答|本次采用/修改与限制|
|---|---|---|
|DETRDistill v4，§3–6/式1–9、表1–7，PDF4/8页关键公式原表目视|预测无query天然对应；Hungarian预测对齐；正/负定位、逐层KD、特征和query prior；COCO多DETR验证|只采用集合对齐/正预测KD思想；复用本库matcher，不是作者代码逐项复现。改为同R18不同视图、末head、置信加权selected-class KL、离线缓存/弱增强。没有负框/逐层/feature/query module|
|DETRDistill表6/7负向证据|表6正cls+正reg 42.3→42.5AP但APS25.3→24.9；全正负logit43.7、APS25.3；表7末阶段43.1/25.4、逐层43.7/25.3|不能用论文总体+2AP承诺本次APS；本方案有明确失败风险。此处“末阶段”也不冒称等同作者用末层监督各层，本轮仅学生末层|
|原SAHI/既读直接航拍先例|切片改尺度/上下文并融合可提高检测，代价与大目标取舍已报道|只作教师来源依据，不证明迁移成功或蒸馏创新；旧稿关闭保持|
|最接近多视图/跨尺度蒸馏：**尚未完成原文核查**|不能写“无先例”或最近性已定|2026-10-20 Bing请求被保存权限拒绝，未换通道；以下唯一优先缺件|

最少下一原文：Ilija Radosavovic等，*Data Distillation: Towards Omni-Supervised Learning*，CVPR2018，官方入口 `https://openaccess.thecvf.com/content_cvpr_2018/html/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.html`；PDF `https://openaccess.thecvf.com/content_cvpr_2018/papers/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf`。题名与链接为待获取线索，本轮未访问/未读，不据此写具体结果、作者代码版本或“最近”。需核方法中多变换teacher预测合并、box回映/伪标签与训练增强，以及实验中监督训练控制/多尺度收益；它是否完整覆盖本问题需正文判断。取得该文后先判断是否还需一篇DETR跨尺度直接先例，不先开展宽泛扩搜。

DETRDistill作者实现/补充和多视图近作仍未固定核验，因此**原创性/最近先例结论保留未知；本准备不是文献审查全部完成或GPU_READY**。可审查公式和CPU代码已具体化，不以这一缺口安排试错训练。

## 7. CPU验证范围与明确未做

生产数据路径：`r18_distill_prepare.py prepare`读取原4869 manifest（真实冻结SHA、全成员、split/稳定ID/顺序）与真实seed0父绑定（原15150bytes/SHA）。**只读小型本地元数据**；未连服务器读原checkpoint/训练图像，未生成真实教师。

真实准备进程链：`tools/check_distill_cpu.py`从repo cwd、去除PYTHONPATH→最终`r18_distill_prepare.py cpu-check`→独立worker→原生R18 128² FP32物理1→合成五视图教师forward/映射/候选/写读重建→三臂真实native forward/backward与完整保存恢复重放→原primary/bridge输入适配→RESULT/CLOSED→退出。单元与集成均不skip；两图fixture含空GT/空教师。真实Dataset.__getitem__/bound_bytes以单成员合成磁盘夹具跑通，SHA损坏拒绝；4869真实构造身份另由prepare核对，不能把它写成全真实图像数据流通过。

独立进程覆盖正常、错误plan、重复启动不改闭合文件、worker异常STOP且无CLOSED。检查同分、类隔离、NMS边界、预算/置信门、坐标/flip、teacher detach、KL q=1/空池、query置换、原监督逐项一致、三臂同像素GT、错成员/父终点/缓存绑定/恢复身份/SHA。计时与模型训练科学评分未执行；双评价器**输入**适配已执行，未调用AP。真实768×1344/BF16/physical8、资源、GPU恢复、完整生产监督链尚待下一授权，不能靠CPU PASS追认。

准备过程中保留cpu-001（强增强对齐拒绝）、cpu-002（合成空图原rows身份重复）、entry-004（synthetic OrderedDict未按真实safe序列化构造）及依赖下载超时/控制台编码错误。它们是准备错误，不是科学执行异常或材料冲突；没有重试生产。精简原始回执与最终CPU证据在D的本任务delivery；每个失败目录原样保留。已成功的临时合成checkpoint只留下SHA/大小/恢复证明，生产checkpoint从未生成。

代码均新增在origin/main隔离分支，无未合并PR源码依赖；`distill_engine.train_window`按原共同Engine复制，只改3648终点及新局部LR import，原engine字节未变。CI加CPU原生入口，仍不是科学通过。最终commit、源码清单、CPU证据及CI实际状态见交付索引；不合并main，不改旧release。
