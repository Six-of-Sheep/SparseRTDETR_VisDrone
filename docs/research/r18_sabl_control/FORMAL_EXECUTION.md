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
