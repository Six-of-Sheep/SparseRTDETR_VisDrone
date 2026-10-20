# 唯一生产执行准备版（PR15，等待单独GPU授权）

## execution-001 STOP后的最小CUDA兼容修订

用户另行授权设备身份兼容及完整诊断留痕，科学配方、执行预算、训练/评价日程、资源门槛和固定环境不变。旧release/STOP及原子进程stdout/stderr缺失保持；当前新诊断确认torch2.4.1成功初始化CUDA并查询设备属性后，在`uuid`属性处抛出AttributeError。这不补回旧stderr，也不证明旧exit1没有其他原因。

新增`tools/distill_cuda_preflight.py --gpu-uuid <完整GPU-UUID> --output <新目录>`，实际调用同release的`r18_distill.py cuda-preflight`。父子进程及nvidia-smi命令、cwd、CUDA可见性、退出码、stdout/stderr原字节与SHA均独立留存，失败不重试。模型forward为0，不创建生产合同或令牌。生产worker使用相同身份接口，每阶段单独保存身份审计。

身份依据为torch已加载的唯一CUDA runtime库，调用runtime API得到**CUDA逻辑设备0实际对应的PCI地址**，再从nvidia-smi物理清单中唯一匹配该PCI的真实UUID。该方法实际测量CUDA_VISIBLE_DEVICES重排后的映射，不将逻辑序号当作nvidia-smi序号，不用名称代替UUID；若torch自身有uuid，还必须与物理/契约一致。生产仍要求完整UUID显式可见且仅一个CUDA设备。缺失、重复、歧义、查询失败、UUID/PCI冲突及环境不一致均拒绝，不引入第二CUDA runtime，不升级环境。开发现场首版误调用未导出的cudaDeviceGetUuid，已保留完整诊断并在兼容范围内修正为PCI映射；该失败未进入生产链。

CPU兼容检查仅模拟设备接口与错误分支；现场PyTorch2.4.1无forward预检另行留证。因修改最终CLI和worker初始化，重新走完整合成CPU生命周期以保留严格源码绑定，不放宽原CPU gate的全文件一致性规则；未变的算法/真实成员认证仅按明确SHA引用复用。新版本及执行状态以新delivery的VERSION/REPORT和共享CURRENT为准，下文保留原生产准备记录。

科学依据仍为PLAN.md与`configs/research/r18_multiview_distill_v1.json`，本轮未改变任何科学参数、门槛或历史裁决。执行工程合同为`configs/research/r18_multiview_distill_execution_v1.json`；独立协议`r18-distill-production-lifecycle-v1`、任务`r18-multiview-distill-20261020-001`。7578876及旧STOP/准备成本保留。Data Distillation缺件已由正文补齐，见同目录审查；PLAN原“尚未读/入口未实现”是旧准备版本的状态，现由本记录补充，不能追认旧版本通过。

本轮只实现/CPU验收，不连接实验室、不发放真实合同/令牌、不生成真实teacher cache、不训练或科学评分。CPU合成AP只检查两个原评价器调用与数值序列化，不是development科学评分。

## 固定入口及阶段

最终CLI为`python -B /absolute/release/r18_distill.py run --contract /absolute/contract.json`，cwd为该release。所有CLI分支统一初始化自身ROOT/src，移除传给子进程的PYTHONPATH；逐文件核验release源码，明确解释器及source_root，不通过旧channel借身份。仅支持Linux POSIX生产/CPU生命周期；本机用WSL CPU、CI用Ubuntu，保留Windows试接线失败证据，不改原评价器POSIX安全读取器。

`run → supervise → worker → RESULT/complete event/worker exit → supervisor exit/CLOSED`，每阶段独立进程及单次消费凭据，必须按序闭合才进入下一阶段。启动握手绑定直接child PID；Linux子进程PID/start、subreaper、残留清理、资源及进展保护沿用认证接口，不替换旧全局模块。缺失/乱序/重复终态、错误源码/角色/缓存/恢复绑定均拒绝；活跃JSONL只读取完整行前缀，最终不接受截断尾行。异常留STOP/可用法证state，不自动重试。

顺序：

1. admission：真实4869与548 manifest/成员图像/原始标注逐字节准入；训练、development身份隔离；父e120 EMA/checkpoint/binding、fixed64、原A/C汇总精确绑定。只允许这两个数据角色，1602没有入口。
2. teacher：同父EMA FP32、batch4五视图。前8图在前两个原计划batch内核对raw/compact、T1/T2选择与实际弱增强对齐。T1为**B式整图teacher**，同hard NMS；T2复用该teacher五视图及同一view0。4869全量raw→compact重建、成员/源码/合同/SHA全部合格后才闭合cache。
3. accept-T0/T1/T2：每臂独立父EMA及新optimizer/EMA/RNG；更新1保存→更新2→重新构造、恢复1、重放2。每臂实际3更新/6batch forward；输入/输出/DN/损失/梯度/fullstate/RNG精确。三臂全部通过后才可训练。
4. train-T0/T1/T2：每臂再次从同一父EMA local0独立初始化，**不读取acceptance state**；固定12×304/3648更新，弱增强与GT监督相同；跨epoch保存，唯一FT012 EMA，不选最优epoch。三个学生的像素/GT/增强摘要必须完全相同。
5. infer-T0/T1/T2：548图原A单视图FP32，原top600/300、原raw-ignore顺序；每臂137calls，无teacher依赖或新增head。缓存完整raw及两预算预测。
6. score：一次6组/12双评价器调用，实际parsed GT进入primary/bridge，记录输入覆盖与过滤计数；原A/C数值直接复用。
7. cost：A/T0/T1/T2相同硬件/FP32/batch4/fixed64，2遍warmup+4遍平衡顺序计时；包含文件读入、预处理、H2D、forward、排名和raw-ignore；排除模型加载和GT解析，各臂相同。报告每轮、均值/样本SD与峰allocated/reserved，T2/A耗时和allocated各≤1.05。使用一次明确计时边界，不选择快的重复。
8. compare：未舍入27项质量条件，T2−T0/T1/A分别判断；另外T1−T0、T0−A、T1−A、T2−C完整差值，检查累计forward/更新数及公平输入。
9. export：只白名单配置、合同、源码清单、指标/比较、终态索引；不复制raw/prediction/checkpoint/逐更新日志。run最终全链CLOSED/退出，质量FAIL仍正常完成固定全部任务后关闭配方，不能自动扩展。

## 预算与资源

|阶段|计划batch forward|说明|
|---|---:|---|
|teacher|6090|4869×5=24345图像视图；每view batch4，末批1；前8图检查已包含，不额外试forward|
|三臂GPU保存恢复|18|每臂1、2、重放2，9实际更新；状态不进入formal|
|正式三臂|21888|10944更新×两物理batch8；175104学生图像|
|固定端点|411|3×ceil(548/4)|
|成本|384|4模型×(2+4)×16固定batch|
|合计|**28791**|包括训练forward，非纯推理预算|

5.5–8 GPU小时仅依据历史链的数量级估计，CPU评价、缓存写读/重建和I/O可使总墙钟更长；本轮未实测。RAM≥16GiB、可用CPU≥16、数据盘空闲≥80GiB、SSD≥4GiB、shm准入≥8/live≥4GiB；GPU准入空闲≥20000MiB/live≥3072MiB、GPU<80°C/CPU<90°C、CPU功率上限125/150W、GPU≤1500MHz，均只核对不改设置。资源样本年龄<25s，监控间隔2s，OOM/boot/挂载改变立即STOP；无进展300s/语义操作1800s。不是旧MAL训练显存门槛。大型材料留服务器。

## CPU证据与将来批准边界

复用entry-007的17项算法/数据对齐/原生恢复检查与metadata-002真实4869小manifest/父binding准入；原科学配置未变。新增最终入口合成六train/六dev成员，真实R18 128²/FP32，末批2、两合成epoch，完整15阶段、原primary+COCO实际调用、导出/退出；资源fixture经**同一真实screen合同接口**，不替换GPU资源实现。GPU/BF16/真实生产成员不由此认证。新增fixture cache明确synthetic_train/非e120身份，生产仍硬绑定真实训练manifest和e120 EMA；不放宽旧缓存认证。

CPU具体命令、通过/拒绝列表、旧失败及累计合成成本以delivery `production-preparation-002`的CPU_GATE/REPORT/版本索引为准；CI必须按精确head实际状态记录，排队不算通过。测试后若仅Git commit标识改变，可在所有source files字节完全一致时生成显式parent gate→最终source复用证明；任意文件改变不得按此规则复用。CPU receipt的材料路径迁移只能另列相同bytes/SHA引用，不能改旧证据。

未来获得一次GPU授权后，先从提交冻结独立release，提供新鲜spec（当前实验室解释器/数据路径、完整path/bytes/SHA、boot/UUID/挂载身份）和独立approval。`bind --spec ... --source ... --cpu-gate ... --authorization ... --output ...`验证独立授权精确source/commit/science/execution/attempt/nonce，然后新鲜资源准入，才能生成合同；`run`一次消费。**本轮不生成这些生产令牌或approval，也不预认当前材料位置可用。** approval范围为`GPU_TRAIN_EVAL_ONCE`，不因准备完成自动获得。

仍须GPU现场验证：真实4869图像/标注与父checkpoint可用性、Linux冻结环境、显存/热/磁盘峰值、前8图teacher输出、BF16有限性、三臂完整GPU保存恢复、实际耗时。任何新生产异常STOP、保留现场、不自动修复重试。
