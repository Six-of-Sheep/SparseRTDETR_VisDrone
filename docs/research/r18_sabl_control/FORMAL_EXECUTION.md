# PR26正式24轮链：CPU准备完成后仍须单次执行授权

科学定义沿用[PLAN](PLAN.md)，不改N/S/A模型、初始化、监督、优化器、日程、匹配、评价或门槛。本次仅本地合成CPU、仓库检查、CI及冻结；没有连接实验室服务器、读取真实图像/权重或执行真实评分。CPU构造指标全部NOT_SCIENTIFIC。

## 受验计算复用范围

短GPU受验commit `0aecdf798565bab5e4628f3b2b2da94e55808afd`，manifest `3915bbf03f5c335b2a499979e55876cf5856b39bbee92fec45b7d68324f4b7c1`。模型、参数/BN初态、RoI/四边/监督、原matcher/aux/DN、两微批BF16/FP32更新、AdamW、EMA、完整恢复及原评价器等300份既有src/configs/vendor及关键工具的**已提交字节**保持不变，逐项SHA见[GPU_EVIDENCE_REUSE](GPU_EVIDENCE_REUSE.json)。Windows工作区可能CRLF；另列LF标准化SHA仅供工作区检查，生产包必须匹配原始Git字节SHA，不以换行归一化绕过生产校验。

可复用原9次实际更新/18forward中实测BF16、有限梯度、全状态精确重放和短更新资源证据。新增的是正式driver、独立正式state scope、局部epoch增强适配、材料准入、终点及后链接线；这些由CPU检查覆盖，**不冒充已经实测24轮、终点评价或正式成本**。没有重跑短GPU验收。若以后这些300份计算依赖变更，生产准入拒绝，必须先报告影响，不能自行重验GPU。

## 正式链与固定终点

1. 严格部署manifest/逐文件SHA；读取已登记H2历史元数据及原N认证、STOP、源清单、评分/推理引用；物理核验父checkpoint的SHA、e120 EMA身份，重新核验4869训练及548 development成员字节/尺寸/原始标注、既有64成员选择、环境和唯一GPU资源。路径来自认证元数据，不猜测。原N仍为AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE，不修补旧CLOSED。
2. N/S/A分别通过原`make_engine`严格载入同一父N EMA。每臂独立seed0、fresh optimizer、复制本臂EMA/updates0；共有native与S/A头初始化摘要一致。验收state的format/scope与正式状态隔离，不能载入；正式状态也不能交给短验收恢复。
3. 各臂先完成局部e1–2/608更新，保存完整状态，核实局部头/F8/backbone梯度与输入/GT身份、精度、有限性、资源和更新时钟；不计算中途AP。三臂均通过后，从**各自正式e2 state**经同一严格恢复原语继续e3–24。每臂总7296更新，不把前两轮再计算一次。
4. 局部e1–21原强增强，e22–24弱增强。适配器只让photo/zoom/crop在e22退出，保留本地epoch参与样本seed与原sanitize/flip/resize顺序；不会把e22映射为e118而偷偷改变随机样本。CPU与原增强源码仅替换cutoff的独立构造oracle逐字节核对。
5. 固定e24完整state保存/读回，通过CLOSED、endpoint、contract、manifest、arm、7296更新和EMA时钟绑定加载**EMA模块**。548图FP32/batch4，每臂137forward，最后4图完整覆盖；同一raw query输出生成top600/top300，经既有预算后raw-ignore、无补位与原双评价器。原N两份评分只读引用，不重新推理或评分。
6. N/S/A同会话原固定64图、FP32/batch4，两次暖机+六次平衡实测顺序严格沿PLAN，共384forward。时间包含decode/resize/H2D/全forward/top600后处理/raw-ignore/sync，排除权重加载及GT文件解析；成本阶段不额外生成top300或raw-query缓存。保存每次秒数、均值/样本SD、allocated/reserved峰值。训练阶段峰值/更新/epoch耗时独立记录，不能代替推理成本。
7. 报告所有绝对指标、S−N24、S−A24、S−原N、N24−原N、A24−N24，保留负项、UNKNOWN及无效精修框计数。复用原结果工具和全部门槛；只超过同期对照但仍低于原N时，使用既有“相对控制改善，尚未形成现有基线净收益”标签。质量FAIL也完成固定后链后退出；无调参、扩轮、扩seed或e120路径。

## 累计资源与停止

|项目|本次正式上限|
|---|---:|
|训练（含前三臂各前2轮）|43776 detector forward / 21888实际更新|
|548图固定终点评价|411 detector forward / 6评分组、12评价器调用|
|固定64成本|384 detector forward|
|正式新增合计|44571 detector forward|
|既有短验收，不退款|18 detector forward / 9实际更新 / 53.45389260002412秒|
|研究累计总上限|44589 detector forward / 86400 GPU阶段秒|
|正式剩余GPU阶段秒|86346.54610739998（展示86346.546107400）|

wall上限48小时、allocated18GiB、累计服务器新增30GiB、精简回传5MiB。既有短验收任务审计2065342933B计入storage，不删除旧state来重置预算。正式入口从开始（包括CPU/准入/I/O）保守计入GPU阶段分配，不是GPU利用率积分；最终EXIT账本是闭合时成本权威，slim RESULT是导出前快照，两者时间戳边界不混同。forward在调用前持久化预留，实际optimizer更新单列，失败不退款。

前两轮共同比较边界按每臂两轮中较慢的epoch秒/更新估计剩余22轮，乘1.25余量，再留1800秒给终点/成本。若明显超剩余GPU预算则工程STOP/未完成；这只是原PLAN“保守预算明显不足即停止”的可执行化，**不是新增质量门槛或收益预测**，不能借机延长预算。

每次forward、更新、阶段和I/O完成边界检查累计时间、storage和allocated；同步吸收旧峰后才reset，保留整链高水位。SIGALRM、SIGTERM/INT及普通异常写stderr/STOP/EXIT与累计账本，不清除现场、不修复重试。重复attempt、父任务唯一启动标记、错误授权scope/contract、候选未提交包、错误材料/闭合/权重/输入均拒绝。SIGKILL/断电可能只有最后持久化账本而无CLOSED，不能追认为正常闭合。

## CPU证据与启动命令

- `python -B tools/check_sabl_formal_cli.py /ABS/UNIQUE_CPU_DIR`：实际完整包→部署→CLI plan/run→三臂前两工程epoch→严格恢复/第三fixture epoch→EMA→6构造图（含不满batch）→两预算/双评价器→平衡成本→比较→slim/CLOSED/EXIT；检查重复、资源/策略/父身份、数值异常、坏endpoint/输入及缺CLOSED。固定CPU轮/窗口缩小只为接线，不模拟7296次更新已发生。
- `python -B tools/check_sabl_formal_state.py /ABS/UNIQUE_BOUNDARY_DIR`：三臂真实CPU更新、显式构造303时钟，实际304边界保存、305直接/恢复重放，model/optimizer/EMA/RNG/输入/DN/输出/梯度精确一致；303是夹具时钟，不能算作执行303更新。另测e1/21/22/24增强与双向state scope拒绝。
- `python -B tools/check_sabl_formal_policy.py /ABS/report.json`：stdlib预算/时间/内存/存储、整链峰值、CLOSED/SHA、唯一标记及资源IO单元检查；模拟nvidia-smi响应不是实物准入。

CPU每子进程2线程/4GiB/5分钟，合成消耗单列，不计入未来真实forward预算。精确head CI还运行既有模型/状态/短CLI及Python3.10/3.12管理回归。最终CPU/CI和全commit/manifest以D盘精简交付回执为准。

冻结：`python -B tools/package_sabl.py /ABS/UNIQUE_PACKAGE --formal`，只能从clean、已提交head导出；`COMMANDS.txt`包含完整部署、认证元数据路径、plan/run命令、唯一任务/attempt及GPU UUID。包**不含授权文件**。未来需单独授权该完整commit/manifest下的唯一正式合同，重新核当前材料/环境/无重复作业后仅启动一次。正常PASS可连续到收尾；本次CPU准备不消费正式启动授权，不连接服务器、不建立本机持续轮询，PR26保持草稿未合并。

## 2026-10-27限定父N大小字段适配

46aa版本的新鲜准入在读取父N引用`bytes`时STOP；认证历史引用实际使用`size_bytes=323457665`。旧commit、冻结包、合同提案、STOP与stderr全部保留。本修订只让父N物理准入显式读取`bytes`或`size_bytes`：必须为严格正整数（不接受bool、浮点、字符串、空值、零或负数），两字段同时出现时分别验证且必须相等。缺字段、非法值或冲突均拒绝；不修改认证引用、不改SHA、路径、文件大小或身份的严格校验。

`python -B tools/check_sabl_formal_metadata.py /ABS/metadata-cpu.json`读取仓库内已回传认证历史checkpoint元数据快照，核实原字段值；随后以明确标记的本地构造payload和外部元数据/role IO夹具调用**原生产material_admission**完整路径，覆盖两格式、相同双字段及全部拒绝项。该检查不替换大小适配器、物理stat/SHA、原标注解析或图像reader，不加载检测器/真实权重，不连接实验室；外部认证回执与2/64构造图不能冒充4869/548真实成员或服务器准入。完整CPU CLI回归仍单独执行，合成消耗不进入真实预算。

新的包命令先核对既有`preflight/STOP.json`的精确SHA，拒绝任何已有正式attempt、最终合同、authorization、一次性标记、同名tmux或目标release。未来如另获授权，仅部署新的commit命名release并绑定新合同；旧上传目录和preflight现场不覆盖。沿用原任务根目录可让旧失败上传与新release一同纳入原30GiB累计计数，不重置存储。此处仅生成启动命令，本轮没有执行部署、合同生成或GPU动作。

## 最终绑定调用器的stdin环境修复

b77bdb9执行在辅助绑定调用的首条导入前STOP：调用没有进入release工作目录，也没有设置tools/src导入路径。旧STOP、上传、proposal和原release保留。`tools/sabl_bind.py`现纳入Git：调用必须使用认证Python、先`cd "$RELEASE"`，再`"$PY" -B - ... < tools/sabl_bind.py`；脚本在导入前验证解释器与cwd，显式设置release/tools和release/src，并核对实际模块来源。无需外部PYTHONPATH。生产模型、准入与资源门槛、短GPU证据和预算不变。

绑定首先核release/manifest/commit、原合同策略/累计预算、proposal canonical SHA及源码/材料/EMA/数据/资源回执。`--check-only`只读，不加载模型、探测GPU、执行生产准入或写文件；它不是执行授权或当前实物准入。`--bind`仅在另行正式授权和全新鲜准入后使用，保留短GPU证据/存储继承、无重复作业及GPU进程检查，独占创建正式合同与一次性授权；不创建attempt/启动标记、不启动tmux，不提供重试。原STOP原件不修改。

`python -B tools/check_sabl_bind.py /ABS/bind-cpu.json`在临时release中复制真实Git模块，用当前实际CPU解释器`sys.executable -B -`、清除PYTHONPATH/PYTHONHOME后通过标准输入执行同一调用器。核模块真实来源、cwd/解释器、合同摘要，以及错版本/SHA/策略/回执/已有输出/STOP拒绝；每次子进程前后逐文件SHA确认只读。已认证STOP只有小型原字节快照，其他合同/准入回执明示构造；不冒充真实认证、服务器解释器实测或当前资源准入。CPU本地使用Python3.10.16，CI在3.10/3.12执行；实际实验室解释器路径仅作为未来命令绑定值，本轮不连接服务器核验该二进制。测试仅运行check-only，同一绑定校验在生产bind写出前复用，无生产合同/授权/marker/attempt写入，无forward/更新。

## 2026-10-28：原v1 N/S e24限定评价（仅准备，待独立执行授权）

原链因A数值异常STOP；N与S各自的训练RESULT/CLOSED及e24 checkpoint完整，均为24轮、7296更新。原N24权重SHA为`fbf798c33649253a410c2feeac01bb62757089221709726a02493117469de921`，S24为`deb67cf4bfc7e1a25998fb8490309a0ee812471859bacfbe8a42d8be4673f21f`。这里的N24是同期继续训练控制；“原N”专指共同父模型的认证B0 e120 EMA，SHA `be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83`。CPU准入严格核对各臂完整layout/cache、配置/监督、EMA时钟、训练输入顺序摘要及endpoint/complete/RESULT/CLOSED；只复制`ema.module`，不恢复optimizer、engine、RNG，不使用raw model或PR27验证state。

科学计算固定为`79ba59371c21ecfd4fa76885c413941403797214`、manifest `1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab`的`r18-sabl-control-v1`；新分支只增加任务限定入口/CPU检查和本文。全部src/configs/vendor字节须与原release相同。**不得替换为logsigmoid-v2，即使旧S评价出现数值异常也STOP。** 原STOP、N/S终点、A938、历史成本及两个PR均保留；此限定评价不修改或恢复原formal attempt。

原N两档评分可直接复用：`r18-sampling-span-20261020-001/attempt-formal-003/score/B0-{600,300}.json`，SHA分别为`86be4fcc1478442bd4bb6f3b6e5b0c421e29f854e6977c54ba906814e79dfc7e`、`5a829d1ed3c68022053605ab5f143d3dd988eb277960918b898ffcaacd7e3129`。原N inference/score回执、源清单与评价合同绑定同一548成员manifest，且五个评价/数据/ignore实现的SHA相同。保留`AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE`及原train-B0缺CLOSED的限制；指标复用不是补造历史工程闭合。该核对不重新评分原N。

评价继续使用固定548 development成员（manifest SHA `5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd`），原图坐标/类别、raw-ignore、768×1344、FP32 batch4、Q300、无NMS/阈值/TTA；先top600/top300截断再执行原ignore，不补位。两档共享一次检测器前向。复用`sabl_formal.predictions`和原`r0.evaluation.score_budget`：官方风格Python主评价器及COCO bridge（maxDets[1,10,100]），未运行MATLAB；COCO small使用原图面积<1024，主评价器APS不提供时仍为UNKNOWN。不得访问1602。

仅输出两种差值：S24−N24和S24−原N；两档绝对指标及全部负向项均保留，负哨兵/缺字段为UNKNOWN。原收益/保护阈值从`sabl_results.render`原字面定义读取，不新设净收益门槛：top600 AP75-small≥+0.30pp、APS≥+0.20pp、AP≥0，APM/APL/AR-large/主AP≥−0.20pp；top300 AP75-small≥0，AP/APS/主AP≥−0.20pp。同期N条件可以单列，但S−A24、完整三臂质量裁决及空间排列机制为UNKNOWN；不能宣称三臂完成。原成本门槛E2E比≤1.30、allocated比≤1.20保留，**本次不测速，因此其状态UNKNOWN**。若只超过同期N而AP/APS/AP75-small仍低于原N，明确写“相对控制改善，尚未形成现有基线净收益”，不把子条件通过写作原配方成功。

|新增评价预算|固定范围/依据|
|---|---|
|检测器forward|274=2×ceil(548/4)，N/S各137；原N0；预热0、训练/反向/optimizer/EMA更新0|
|评分|4份新评分JSON、8次评价器调用（两臂×两预算×双评价器）；原N两JSON只读引用|
|阶段/总wall上限|保守1800秒阶段（包括评价CPU/I/O），3600秒wall；不是新增总额度|
|显存/存储/回传|allocated18GiB；该任务新增512MiB（含预测、源码包、日志、独立编译cache）；旧SABL任务合计仍≤30GiB，磁盘free仍≥30GiB；全回传≤5MiB（stdout/stderr预留256KiB，白名单不含大预测/权重）|
|余额继承|已用31743 forward / 36543.70207419119秒；当前余12846 / 49856.29792580881秒。若用满本次上限则余12572 / 48056.29792580881秒；失败不退款|

时间依据是历史B0 FP32137次forward约63.14秒、原六组双评分合计351.08秒；S EMA实际速度未知。估计实际约10–20分钟，30分钟阶段上限留足加载/I/O/评分空间，但不保证完成，触及上限STOP。N/S两档预测依历史B0文件大小约290MB，不额外保存raw-query缓存；精简回传只含四评分、比较表、终态/成本/来源回执。

入口`tools/sabl_ns_eval.py`仅为以上任务作适配：`prepare`在CPU核真实材料；`launch --check-only`用最终解释器/cwd/包检查且不创建授权或attempt；未来唯一`launch`在独立新准入目录核新鲜材料/版本/环境/硬件/无重复作业，再独占绑定、运行、白名单导出并退出。命令和准确commit/manifest来自冻结包COMMANDS，不使用旧启动标记。编译cache放该任务根内并计量，首次异常STOP，无自动修复/重试、评分后续训练或自动测速。CPU构造入口验收不调用真实评价器，输出均标记SYNTHETIC，不能算新增科学结果；真实CUDA编译/资源/数值与实际评分仍待独立授权验证。

### 后续修复版S/A公平比较：仅预算说明，不实现恢复/训练

若未来另行决定公平比较，S/A应使用同一个修复版本，从共同父N e120 EMA独立建立optimizer/EMA/RNG各训练24轮；不能把旧v1 S与修复A拼接，也不能将A938续训视作同初始化双臂。当前两次raw前向通过只证明指定输入下数值/输出有限，未证明训练反向和完整训练稳定；是否需新训练验收应在未来实现差异审查后决定。

两臂24轮共14592更新/29184训练forward；双终点274 forward；若沿原三臂同会话成本比较再加384，共29842。若另行授权9更新/18forward短验收，总29860；本轮两项都不执行。有限评价完成后预期余12572，分别缺17270或17288 forward，**现有余额不足，不可自动启动**。按原S24耗时20214.014秒和原A前两轮较慢865.142秒推算，乘1.25余量并加1800秒收尾，约53021.8秒；大于有限评价上限扣除后余48056.3秒，且修复版真实训练耗时未知，必须另审总时间预算。存储仍须纳入旧30GiB，不能删除旧state或重置账本腾出“新额度”。此说明不是重训提案获批、启动命令或新增验收授权。

## 2026-10-28：原S监督／解码源码和CPU审查（未修复）

本轮只读原79ba/1af3科学release和68d990fc/077b629b评价release、既有S日志与输出索引；无GPU、检测器构造/前向、checkpoint/真实图像加载、训练反向/更新、AP/AR重算。审查工具在独立`codex/r18-sabl-side-audit`管理，PR28已消费包及原结果不改变。误差判据在CPU执行前固定：原函数FP32，归一化cxcywh绝对误差≤1e−6、rtol=0且valid=true；RoI线性场误差≤2e−5，不随结果放宽。

### 已证实的实现不一致

`src/sparse_rtdetr/research/sabl_control.py`（原文件SHA `c5945edd78ba7159ce83f433a1cc9a266543675ba4f3edac478d059d7875f05b`）：

|位置|实际行为|判断|
|---|---|---|
|26–27 `corners`|cxcywh→`[L,T,R,B]`（标准xyxy）|适用于RoI及普通框角点转换|
|30–41 `buckets`|堆叠四边为`[L,R,T,B]`；scale=`[Δx,Δx,Δy,Δy]`，域1.7倍|与头的四边输出及解码一致|
|44–47 `encode`|直接令g=corners(GT)，未将xyxy改成桶的四边顺序|**GT的T进入R监督，R进入T监督**；桶logit正负标签、邻桶mask和offset目标均受影响|
|57–68 `decode`|用四边`[L,R,T,B]`计算w=e1−e0、h=e3−e2并转cxcywh|与错误编码目标不一致；无效时两宽高置0且valid=false|
|78–81、176 `boundary_loss`|实际Criterion普通final匹配后调用同一个encode|错误进入生产训练监督，不是仅展示时字段名问题|

实际编码器输出的t作为理想offset，正桶logits=+20、其余−20，再直接调用实际decode。11个合法GT构造样例（非方形、1像素/细框、图像边缘、桶中心/边界、偶然等值控制）中**10个未通过**，1个因GT的T=R而偶然通过。审查程序完成51项一致性/反例验证，不等于编码闭环通过。

最小反例：reference=GT=`[.5,.5,.2,.2]`，GT角点`[.4,.4,.6,.6]`。实际encode的理想预测使offset loss=0、bucket loss≈2.06e−9；decode却返回`[.4,.6,0,0]`、valid=false，最大误差≈.2（远超1e−6）。另有合法GT`[.4,.6,.2,.1]`，理想decode为`[.425,.575,.25,.15]`、valid=true，误差≈.05：**并非只要丢弃逆序框就能消除该问题**。

这些都是人为CPU构造，不是模型终点张量或训练根因诊断。缺陷足以证明监督语义不一致，但不能证明它解释全部实测退化或125242的全部组成。原固定S质量FAIL、所有原指标和STOP保持；这次阴性不能作为“正确接线的常规SABL式局部精修也无效”的证据，更不能否定局部空间信息。

### 其余实际路径、监督和设计限制

RawAugmentation从图像xyxy转**归一化cxcywh**，x/w除画布W、y/h除H。Model取第三层普通Q300粗框detach作r；RoI将r转像素xyxy，用7×7 RoIAlign、stride8、aligned=True、sampling_ratio=2。实际RoI线性非方形场最大误差7.63e−6，满足预定2e−5。RoI不扩大、1.7仅用于桶域，与本方案原定义一致；特征窗口与扩展搜索域不同是已登记适配，不能误报为本次发现的隐藏实现错误。

水平/纵向池化及14位置拆分为L/R/T/B（右、下反向）；每边7桶、同一个offset符号`t=(x−g)/Δ`与`e=x−Δd`代数一致。训练和评价共用r及头输出形状；训练返回coarse native输出和sabl logits/offsets/reference，评价才argmax选桶并解码替换pred_boxes，保留原分类score。没有发现额外的像素/归一化倍数错误或这条路径的符号倒置；本结论限定于已核源码及CPU例。

没有构造检测器：仅按原resolved config创建真实native criterion/matcher。实际CPU构造中query3配GT idA（目标数组索引1）、query17配idB（索引0）；wrapper使用普通final诊断和GT ID映射，不靠query编号等于GT编号。原final、2 aux、3 DN、1 encoder共21损失逐项不变，新头仅增ordinary两损失，分母4M/M。未匹配query和实际mask=0项的扰动不改变对应损失；loss图连到提供的logits/offsets，GT/reference detach；空匹配为可微零。没有执行反向。`Engine`仍把原21项与新增两项求和再乘microbatch系数1/2；损失能接到张量，不代表它接到正确语义目标。

7296个既有S更新、14592份microbatch损失/normalizer全部检查；两新增loss存在、finite=true，ordinary匹配数及4M/M分母全部一致。首步offset loss约285.89/281.90，末步约82.58/91.86；首步bucket约4.56/4.52，末步约1.93/2.01。有限、末步较低以及已有梯度有限/恢复证据，均不能替代四边语义正确性。本轮未重新验证完整训练梯度或新增训练。

各边独立argmax/offset没有强制L<R、T<B的耦合约束，允许逆序/零面积，且不交换边、不abs、不裁成正宽高、不回退粗框。这是**符合当前定义的设计限制**，与已证实的错误编码分开记录。修正目标顺序也不保证学习后没有逆序，不能预写收益。

### 125242的含义与UNKNOWN

原`tools/sabl_formal.py::predictions`在每个batch的model.eval返回后，把`(~refined_valid).sum()`累计；是548×300普通query位置中的失效数，**发生在类别展开、top600/300截断和raw-ignore之前**，不按GT匹配/类别或两预算重复计数，不含DN。`rank_refined`先保持原query/class score排序并截预算，再按valid删除无效项，**不补位**；私有占位框不输出为检测。随后只将有效归一化cxcywh转原图像素`bbox_xyxy`，进入未改的原双评价器输入转换。

新评价infer-S只保存两档**已过滤**预测与汇总PROVENANCE；raw列表仅在内存且删除，未保存全部普通query的pre-valid四边、logits/offsets或refined_valid数组。训练日志亦无这些字段。decode在valid=false时同时清零w/h，即使只有这个结果，也丢失原负/零符号组合。因此125242中“仅左右翻转、仅上下翻转、双轴翻转、精确退化、混合”的数量均**UNKNOWN**；PR27的raw单batch不同状态/输入不代表本548图EMA集合。没有自动追加提取或前向。

### 影响范围与未执行修复

同一encode被S和A共用，二者既有boundary监督均受影响，可能连同新增监督到共享F8的训练作用受污染；N分支不调用encode，N/原N结果不因该bug直接失效。不能只改A与旧S拼接公平比较，不能把旧S预测重新交换边视作得到正确训练后的S。数值归一化修复与GT边语义是两个不同问题。

最小修复方向应是**明确统一GT目标与桶/头/解码的四边约定**并添加外部合法GT的实际闭环回归，不是交换已预测左右边或放宽无效判据。本轮未实现任何修复、未改原科学源码/预测/阈值、未重算AP或重开训练；后续修复及公平验证另行授权。

审查工具/原误差判据/CPU结果/流式日志SHA与字段缺失证明，登记于服务器r18-sabl-s-path-audit-20261028-001及D盘同名精简交付。为遵守本轮“零检测器前向”，该审查分支CI仅执行新CPU审查与纯管理检查；既有含模型前向的通用jobs显式跳过，不能记作PASS。审查结果为**已证实的实现不一致 + 已登记设计限制 + 失效类别细分证据不足**，不自动进入新实验。

## 2026-10-28：S/A共同修正版v3（仅模型修正和CPU冻结）

本节保留上节原v1错误与实测FAIL，不覆写原结果。新科学身份为`r18-sabl-control-lrtb-logsigmoid-v3`，与v1及PR27的`r18-sabl-control-logsigmoid-v2`区分。本轮只包含两项计算修正：

1. `encode`明确将标准xyxy角点`[L,T,R,B]`重排为`[L,R,T,B]`，再创建所有桶分类、邻桶排除mask、offset目标及mask。桶坐标/Δ、最近桶、阈值<1、BCE与SmoothL1(beta=.1)、4M/M分母、四边头/解码顺序不变。
2. 原样纳入PR27 `normalized_sigmoid_axis`：`softmax(logsigmoid(logits), dim=原轴)`，水平沿2、纵向沿3；导入函数SHA `d550eb4e9d3f34ee786d9822b2bd9fa07be613b3b81abaf1107025291c5ff41e`，Head类与已验证PR27实现AST一致。无新参数，S/A同用。实数等价不表示FP32逐位或训练轨迹等价。

N/粗框/分类、F8/RoI、采样轴与精度、匹配、aux/DN、日程、监督权重、clip、EMA和无效框处理不改。没有加入排序、abs/裁成正宽高、粗框回退、门控或新loss。

### CPU证据及边界

事前独立GT判据保持FP32归一化误差≤1e−6、rtol0且valid=true。新测试直接从cx/cy/w/h计算`cx−w/2,cx+w/2,cy−h/2,cy+h/2`，不调用生产corners或复用其排列生成期望。原11例及新增7例（不同reference、非对称、越出桶域、小/细框及图像/桶边界）**18/18还原GT**；正确理想预测使实际offset损失0。118项检查还覆盖actual桶分类/offset/mask、四边非理想预测梯度的索引与符号、GT/reference detach；保留左右/上下/双轴逆序、零宽/零高五个反例。独立选边限制仍存在，不预写收益。

PR27既定容差、失败张量fixture及FP64参考保持。归一化108项检查/288比较通过；原“encode AST不变”1项因明确修正GT映射不再适用，改由上述独立GT回归覆盖，未放宽其余数值判据。常值特征可给零logit梯度、有限卷积输入仍可溢出、极端FP64直接sigmoid也非通用oracle等反例仍保留。

完整模型CPU最终检查8次合成检测器forward、2次实际合成optimizer更新（S/A各1逻辑窗口），37项通过。两microbatch均经actual Model→Criterion→boundary_loss→encode，真实收到正确四边目标。头参数梯度由额外损失产生且非零；F8观测的是原总目标梯度，不冒充“仅额外loss的F8梯度”。原coarse reference实际detach。新旧N在各自完整源码根/独立进程下，完整输出和参数/BN/cache字节摘要一致；S/A共有参数与新头初态一致。新完整状态在model、optimizer矩、EMA、RNG被有意扰动后严格恢复一致。

两个实际restore接口均拒绝v1/v2 S/A版本（8项构造元数据拒绝）；测试使用明确标记的header-only夹具证明在任何模型加载前拒绝，不能称为旧真实state续训回放。正常新状态的完整恢复使用实际CPU全状态文件。schema结构名仍为v1，但科学VERSION与criterion/config绑定为v3，禁止改旧receipt/header来“兼容”S/A；正式初态只能来自共同认证父N EMA，不使用验收state或旧S/A。

CPU开发失败均保留：混载旧包被源码归属检查拒绝后改为完整旧源码独立进程；GPU-build共享库虚拟映射触发4GiB AS夹具保护（当时RSS约1.05GiB、CUDA未初始化），随后将实际内存保护修正为父/子进程RSS合计4GiB与单进程DATA4GiB。一个被导入的CLI曾重新施加AS限制，已去掉该CLI导入副作用。模型、输入、误差标准及GPU预算未随这些工具修复改变。最终实测RSS树峰值2220818432B，仍低于4GiB；最终CPU无CUDA初始化。合成CPU消耗单列，不算未来真实forward或训练进度。

### N24复用：核验通过，免重复训练N

结论：**可复用已完成N24的e24 EMA与两档认证评分**。使用原68d/v1生产endpoint+EMA读取接口，在独立CPU进程重新核物理checkpoint（SHA `fbf798c33649253a410c2feeac01bb62757089221709726a02493117469de921`）、e24/7296/EMA、训练输入/父N、4869/548成员manifest及两档评分绑定。N的Model/Criterion/Engine计算AST、24轮/seed/输入/优化器/日程/精度等配置除版本标识外相同；错误encode和稳定归一化均不在N旁路。上述全输出CPU比较补充了计算一致性证据。

同时用新版本**真实生产fresh初始化路径**，在CPU从认证父N（SHA `be5e9e1f…`）分别初始化S/A，未做真实权重前向：native/BN/cache与原N初始记录逐字节一致，S/A新头同初态，optimizer空、EMA/更新0、RNG各自重置seed0一致。N24只作为只读终点/评分控制，**不是将旧N完整训练state恢复成v3**。未来同会话N测速仍需N24原生EMA前向，不能复用旧资源观测；真实执行前需在新受验入口显式绑定该N引用，不绕过旧S/A拒绝规则。

### 修订实验：提案，GPU与扩额未授权

S/A从共同父N e120 EMA独立重新建立native/附加头、optimizer、EMA及RNG，固定seed0；不得续A938、旧S e24、PR27 raw验证state或新短验收state。沿原24轮×304更新，batch8×累积2、4869 train_core、原增强e1–21/e22–24、lr/warmup/cosine、23项监督权重/clip/EMA不变。前两轮工程检查仍保留，正常后固定e24 EMA；548 development、top600/top300共推理、原双评价器/ignore/不补位和原收益/保护/成本门槛全部保留，1602不访问。N24和原N两档评分只读复用，比较S−N24、S−A24、S−原N并保留全部负项；不增加中途AP、选模、调参或seed。

|阶段|更新|新增forward|GPU阶段提案上限|wall提案上限|存储/回传|
|---|---:|---:|---:|---:|---|
|一次S/A短训练工程验收|两臂各3实际更新，合计6|12|900秒|1800秒|新增≤2GiB，回传≤1MiB|
|S/A正式各24轮|各7296，合计14592|29184|与下两行合计57600秒（16h）|与下两行合计86400秒（24h）|正式全链新增≤12GiB，回传≤4MiB|
|两臂548固定终点|0|274|计入正式上限|计入正式上限|4评分组/8评价器调用；N24/原N0重算|
|同会话N24/S/A固定64图成本|0|384|计入正式上限|计入正式上限|3臂×(2暖机+6实测)×16；沿原顺序|
|新增合计|14598实际更新|29854|58500秒|88200秒|新增≤14GiB，回传合计≤5MiB|

显存allocated18GiB、原SABL家族累计存储30GiB仍保留；上表新存储要加到全部旧现场，不能删除旧证据/重置余额。短验收每臂为u1、u2-direct、恢复u1后u2-replay，6次实际更新不是“4次不同更新”；使用固定两个logical16窗口，不额外验证forward。短验收核真实BF16/局部FP32、梯度有限和完整恢复重放，**不判检测收益、不评分/完整测速**，之后退出。旧验收和两次raw前向不能代替修正监督后的新短训练验收。正式科学FAIL仍在预算内完成既定后链，生产新异常STOP不自动修复重试。

历史已用32017forward / 36824.02296776826 GPU阶段秒，当前余12572 / 49575.97703223174秒。若未来批准上述全部范围，则累计需61871forward / 95324.02296776826秒；相对原44589forward/86400秒总上限，**还需另批17282forward及8924.02296776826秒GPU阶段额度**。本轮不改原上限或账本。16h是正式链的拟申请硬上限，不是实测承诺；按旧S24和A前两轮较慢耗时×1.25再留1800秒，估算正式约53021.8秒，修正版真实速度仍UNKNOWN。短窗只排工程/预算问题，不预测e24收益。

历史累计active wall未有统一已核账本；原48h累计wall约束不重置，GPU秒不能冒充wall。未来任何正式合同必须先用既有EXIT/PROCESS回执闭合wall余额及新鲜存储/资源准入，不能将表中的24h当成已批准新余额。

本轮冻结的是**两项模型修正、状态版本保护、CPU测试与修订方案**。旧三臂/旧v1正式命令不得用于本包；本包只交付CPU命令，不生成GPU合同、授权、attempt或启动标记。新的两臂短验收/正式执行入口及其生产CLI闭合、合同与预算批准另行完成；当前不声称GPU执行链已受验或可以直接启动。工程CPU PASS不等于CUDA/完整训练稳定，更不等于质量改善。

## 2026-10-28：v3两臂短训练入口与单次验收

科学基准固定f25b74488f0082cf835b844b89fc8a1f3205b365、manifest 54529a9faccecde599753b03b42095f6d7b8d83f7259cf5e3aadae0109cf7d9b；新增tools/sabl_corrected_short.py只连接现有make_engine、sabl_state和Telemetry，不调用旧三臂命令。302份src/configs/vendor依赖逐字节固定，科学定义/初始化/数据/损失/精度/确定性/门槛不变。原proposal配置继续保留其历史“未授权”状态；当前执行授权绑定新入口commit/manifest及独立合同，不重写历史配置。

限定范围为S/A各u1、u2-direct、恢复u1后u2-replay：6实际更新、12 detector forward；无预热、额外验证forward、评分、测速或完整训练。两臂各自从认证原父N e120 EMA fresh初始化native/附加头、optimizer、EMA与RNG，不能读原S/A状态或用本验收state接正式训练。fixed seed0/e1首两个logical16窗口，32图逐成员原始字节SHA与两窗张量/GT哈希和原短验收相同，不按目标筛图。ordinary final匹配和原aux/DN保持。

CPU验收经最终CLI的prepare→launch→run→slim/CLOSED和实际解释器/release cwd，无预设PYTHONPATH；临时夹具manifest明确synthetic，GPU授权不能用于夹具。覆盖完整FP32训练与全状态重放、原真实历史bytes/size_bytes接口、错误身份/双字段冲突、无效资源合同、forward/update耗尽、非有限输入、错输入、重复启动及缺导出不闭合。夹具状态只在owned临时目录产生，结束自动移除，保留回执/失败stderr；CPU工作不计真实GPU预算。CPU通过不能代替CUDA编译、BF16反向或真实恢复验证。

GPU PASS须同时满足：
1. 实际loss调用的encode目标与独立GT直接给出的L/R/T/B一致，reference/GT detach；两个microbatch均有匹配与两项新增监督。
2. 真实native BF16、boundary FP32；所有原损失/新增损失、梯度和健康检查有限。
3. native及pre、att_x/y、post_x/y、up_x/y、cls、offset预定组确有非零有限梯度和参数更新。钩子只观察原目标的既有反向，不新增autograd调用/拦截器。
4. u1完整model/optimizer/EMA/RNG/engine/下一窗口恢复一致，u2 direct/replay全状态、输出、DN、损失、梯度哈希严格相同。
5. 逐臂与整链peak allocated/阶段/更新耗时、编译cache与存储、12forward/6更新硬计数及slim/CLOSED/EXIT通过。异常尽力采集最后峰值，采集失败UNKNOWN；新GPU异常STOP不自动重试。

上限900秒GPU阶段、1800秒wall、allocated18GiB、全任务新存储2GiB、全部回传1MiB。每个forward先记账，失败不退款；实际optimizer更新与预留次数分列。编译器、Triton、CUDA driver、extensions及TMP cache绑定新task，计入2GiB；所有旧SABL现场加入原30GiB累计门槛。全子进程wall保守计GPU秒，原成本账本不覆盖。

历史wall采取有证据的保守上界而非虚构精确active时长：六次已知GPU执行的forward数18+31061+661+1+2+274=32017，与最新账本一致；认证首个短验收合同的UTC当日零点早于这六次串行执行，向当前准入时刻的整个日历包络（含停机/开发间隔）均算占用wall。合同保留exact active wall UNKNOWN，但用该更大的上界计算可保证的48h余额；若上界余额不足1800秒则拒绝。启动时重新核算，不以GPU秒代替wall。即使新GPU验收PASS，也只证明这个短窗的工程条件，不证明24轮稳定或检测收益。

本轮只实现并授权该短入口；正式S/A各24轮/548评价/三臂同会话测速仍仅提案29842forward、57600 GPU秒（无扩额授权）。短验收消耗实际追加后重新计算正式缺口，不将预留未用部分当成已花成本。正式入口未接线，不自动续接。

开发失败留痕：初始工具import把research模块误当tools顶层模块，真实CLI立即拒绝；已统一为完整包导入。夹具结束前保留大量临时state可能使缺导出测试先触发存储拒绝，现逐项清理自有synthetic state并断言确切拒绝原因，不将别的错误冒充目标故障通过。未修改科学源码或放宽数值/资源判据。

## 2026-10-28：修正版S/A正式24轮链（CPU准备，未授权启动）

计算基准为已短GPU PASS的2c85c22d5738f8dc84c7527c3408602561baae09、manifest 5f9e5a2b977ad3e5b2881324eadf670f4e2b2ef82bb1e0ef79aa2e74047fd69d。302份src/configs/vendor依赖及5个复用执行文件逐SHA固定。新增tools/sabl_corrected_formal.py只做执行接线：复用现有sabl_formal.py的train、endpoint、predictions、infer、cost、compare，以及原make_engine和sabl_formal_state；不改变模型、初始化、训练目标、优化器/日程、采样、精度或原门槛。短GPU证据严格核其commit/manifest/CLOSED/12forward6更新和恢复结果，只读复用，不重跑验收；它不冒充全正式链GPU认证。

训练仅S/A，各从原共同父N e120 EMA fresh创建新头、optimizer、EMA、RNGseed0。保留两臂前两轮工程检查，随后从各自本次e2完整状态在同一链严格恢复到e24；训练输入/GT/增强顺序还须与认证N24登记一致。旧错误S/A、A938以及短验收state不能进入本次正式训练。N24终点与已有两档评分只读复用，不重训或重算N质量；N仅进入新同会话测速。

新增N-only只读EMA桥接不跳过版本检查：明确要求原N24 v1完整checkpoint SHA fbf798c33649253a410c2feeac01bb62757089221709726a02493117469de921、原binding/manifest/父N、e24/7296、runtime/config/criterion、EMA clock、完整raw/EMA layout及不可变cache；将EMA tensor严格加载到计算路径未变的N wrapper，核原认证EMA module SHA 46de3cb580a54421e8f77c699487d7a12bfa0ea188986369b4b31040894a0a93。仅为eval/测速，不恢复optimizer、engine或RNG，不把旧state伪称v3训练恢复。S/A终点继续使用本次新v3完整state严格读取EMA。

完整次序为新鲜源码/材料/环境/资源/防重复准入→S/A各e1–2工程检查→各自恢复本链e2并训练到e24→两臂548图e24 EMA FP32整图推理（top600/300共享）→每档原双评价器→N24/S/A同会话固定64图FP32 batch4成本（各2暖机+6实测，原平衡顺序）→原sabl_results.render全部绝对指标/差值/保护与成本门槛→slim/CLOSED/EXIT退出。科学FAIL仍完成固定后链；工程/数值/资源/完整性新异常STOP，不自动重试。没有中途AP/选模/调参/扩seed/1602访问。

|阶段|新增forward|实际更新|
|---|---:|---:|
|S/A各24轮训练|29184|14592|
|两臂548图e24 EMA|274|0|
|N24/S/A固定64成本|384|0|
|合计|29842|14592|

新GPU阶段拟上限57600秒、整链wall86400秒、allocated18GiB、正式task新增12GiB、回传4MiB；家族累计存储仍30GiB，原新鲜空闲磁盘30GiB下限保持。编译/driver/Triton/TMP cache绑定本task且计入存储，全链显存峰值不因阶段重置漏计。累计原forward44589/GPU86400秒限制没有被本次修改。

权威短验收后成本32029forward/36872.36945352436秒，余额12560/49527.63054647564秒。正式链若未来批准，应另批17282forward、8072.36945352436 GPU秒；届时请求累计61871forward/94472.36945352436 GPU秒。本次合同只列请求，不创建真实authorization、FORMAL_ONCE或attempt；真实launch必须同时收到显式版本和上述精确扩额参数，否则在启动前拒绝。合同/命令文件不是执行授权。

wall已改为逐运行的可核保守上界：除原正式链外六次已完成调用各计其完整允许wall（1800+1800+600+600+3600+1800秒）；原正式链用其真实authorization文件至EXIT文件的同服务器时间包络35582.6703秒，覆盖原entry monotonic35530.649秒，再向上取整并加60秒保护。合计上界45843秒，原48h可保证余额126957秒，能覆盖本次拟86400秒而无需wall扩额。精确active总和仍UNKNOWN；这是证据更细的上界收紧，不是重置额度或用GPU秒替代wall。后续准入重新核原回执SHA/时间与最新成本不变，任何新运行或材料变化要求重新核账。

CPU实际CLI使用明确合成图/GT/随机模型：三压缩epoch×每epoch一window，S/A合计6更新/12训练forward，2臂6合成评价图共4forward，N/S/A合成4图成本24forward，总40。真实scorer产生4评分组/8双评价器调用；用于测试表语义的历史N/N24行明确为S合成评分别名，绝非N质量证据。用实际完整state接口跨工程阶段保存恢复、真实EMA读出、原门槛比较和slim闭合；CPU只验证压缩流程与FP32接口，不能冒充真实304-window跨epoch CUDA验证。已知无收益的合成门槛也必须闭合，不因阴性中止后链。

CPU负例覆盖错误合同/预算、重复启动、worker异常、非有限输入、forward耗尽、语义错误终点、缺闭合和未知预测ID。语义错误fixture会重算其自有摘要/引用SHA，使拒绝确实达到epoch/成员语义层，而不是只因文件SHA不同；完整旧科学材料不修改。所有合成state为测试临时文件，结束移除但保留索引/日志；开发错误可留痕修复复测，不计GPU实验预算。冻结后唯一COMMANDS仅调用Git内新入口，真实新鲜准入/绑定/授权均在受验CLI内，无外部临时绑定脚本。本轮未执行该命令。

N24桥接补充：真实CPU准入发现唯一配置差异为engine.config.arm的版本身份标签v1→v3；桥接显式要求旧标签v1、当前标签v3，且其余全部字段与完整key集合严格相同。不允许删除未知字段、忽略数值差异或将其称为完整训练恢复；CPU补数值漂移/未知key拒绝检查。初次拒绝日志及旧candidate包保留。正式成本出口沿用GPU_stage_seconds标准键并显式记录已批准累计额度，旧成本不覆盖/不退款。

最终调用器另要求显式已批准CONTRACT.json SHA：launch在授权前核对原始字节，子run再次核对传入的同一SHA；CPU增加错误合同SHA拒绝且不创建attempt/authorization检查。最终命令同时绑定commit、manifest和合同字节，不借外部脚本临时生成绑定。


## 2026-10-29 IPC修订：原正式链STOP后重新准备（不授权GPU）

旧3d69171正式入口将TMPDIR设在任务compiler-cache/tmp，并在解释器已导入后修改tempfile缓存。
旧日志只保存`AF_UNIX path too long`及300秒DataLoader超时，没有保存实际旧地址，故旧地址仍UNKNOWN。
隔离进程采用与旧配置同为156字节的目录前缀，标准`multiprocessing.connection.arbitrary_address`生成188字节地址并在Listener.bind复现同错误。
新`tools/sabl_ipc.py exec`用任务绝对路径的SHA前16位建立0700独占`/tmp/sabl-<16hex>`，以OWNER绑定任务/uid。
它在execve启动新Python前统一TMPDIR/TEMP/TMP；正式launch、run和CPU取数复用此入口。
不再在run内迟设TMPDIR或重写tempfile缓存；进程核对环境、tempfile缓存及已存在的multiprocessing缓存，冲突拒绝。
编译缓存和实验产物继续留原任务目录；不改变worker=2、spawn、file_descriptor共享、prefetch=2、300秒超时、确定性或任何采样/增强。

CPU `data-check`仅读前两个逻辑16训练窗口，有限采样器停在两窗，使用真实Dataset/LocalAugmentation/collate/seed_worker及DataLoader。
观测放在spawn_observer内，生产input_record排除它；核原N24前两条科学input摘要及已认证短GPU的两窗input+GT SHA。
记录真实socket地址/字节长度、主/worker临时环境、worker正常退出；CUDA隐藏，不加载检测器、权重或做模型前向。
CPU状态/游标/RNG不供正式训练；下一正式进程重新seed并从父N独立fresh初始化S/A。
CI增加同一exec入口的真实spawn tensor跨进程传输、长路径拒绝、所有者冲突和旧缓存拒绝；CI构造模型检查不作真实CUDA证明。

原STOP成本306.5618230102118秒继续继承：累计32029 forward/37178.931276534575 GPU秒。
已批准累计上限保持61871 forward/94472.36945352436 GPU秒，不再次扩额。
本次拟新29842 forward，GPU精确余额57293.438176989785秒（普通浮点显示57293.43817698979）；合同以十进制字符串保持精度。
新的requested_addition为0，不再沿用旧17282/8072扩额申请；GPU时限、总进程超时和两轮投影均读取新的余额。
wall历史上界追加ceil306.561823=307秒，剩余下界126650秒，仍覆盖拟86400秒；精确activewall仍UNKNOWN。
所有独占短临时目录按OWNER归入家族存储，运行guard同时计任务目录与自身IPC目录；资源准入保留30GiB家族/30GiB磁盘floor/新12GiB。
源码src/configs/vendor和受验train/endpoint/infer/cost/compare字节不变，复用2c85c22短GPU证据，不复跑模型验收。
旧release、原attempt、STOP、e000、N/S终点和成本不修改；新合同和唯一命令仅准备，真实GPU执行需另授权。
