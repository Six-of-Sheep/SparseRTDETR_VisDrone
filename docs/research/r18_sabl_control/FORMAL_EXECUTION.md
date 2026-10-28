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
