# H2正式U/D链与历史N

用户于2026-10-24授权从7527c464追加正式入口。原短验收CLI继续只允许9次实际更新/18次forward并闭合退出；新入口为仓库根目录`h2_formal.py`，不扩大旧入口。本文描述执行规则，实时状态以冻结attempt回执为准。

## 科学范围与外部验收

四份完整模型、更新器、checkpoint和历史N接口以及`r18_h2_nud_v1.json`保持已验收字节。原生数据、匹配、损失、增强、学习率和EMA计算不改。正式入口只调度这些科学函数；终点评价和成本边界参照PR16的a97e1eaf实现，仍调用原`r0.evaluation.score_budget`两评价器，不另实现AP。

唯一可引用的生产短验收是7527c464的`r18-h2-acceptance-20261023-001/attempt-acceptance-001`：完整合同、三臂RESULT/CLOSED/退出、根闭合与累计成本绑定后只读复用。每臂6次BF16 forward、3次实际更新和全状态/RNG精确重放必须全部通过。验收state不进入正式训练，不再次运行验收。

N为已认证历史B0 e120 EMA及两档评分。原SIGKILL/STOP和缺CLOSED保留，认证输出不等于追认原阶段闭合。正式准入重新核N的物理checkpoint SHA、固定e120/初始化/Config、旧认证/STOP和原评分引用；共享科学依赖逐项比较原source manifest及已完成的B0/N审查。N只允许旧EMA在新同会话测速时载入，不载入旧optimizer/RNG，不新增548图N推理。D−U是本轮直接通路对照；D−N带历史比较限制。

## 固定阶段与成本

`train-U → train-D → infer-U → infer-D → score → cost → compare → slim-export/CLOSED/EXIT`。

U/D各从共同ImageNet、seed0独立构造，optimizer为空、EMA从各自初态拷贝且updates=0，独立重置RNG。各120轮、304更新/轮、36,480更新、物理8×累积2，原768×1344/BF16 forward/FP32 loss。所有训练forward均在实际模型前向钩子预记账；实际BF16输出和完成计数分列。保存e000、e001、e030、e060、e090、e117、e120全状态，固定e120完整读回后锁定，仅使用其EMA。无恢复入口、最佳epoch选择或中途评价。

|项目|forward|
|---|---:|
|已完成GPU短验收，继承|18|
|U/D正式训练|145,920|
|U/D各548图、batch4终点|274|
|N/U/D固定64图，2 warmup+6实测、batch4|384|
|本次新增|146,578|
|累计上限|146,596|

继承77.11586332297884 GPU阶段秒；累计60小时、allocated18GiB、任务存储100GiB不重置。GPU阶段时间含构造、读取、保存、监督与阶段核验，不是利用率积分；CPU评分/比较时间分列，不计为GPU阶段。另设66小时总wall安全退出界限，不扩大60小时GPU上限。预记账失败不退款，实际成功完成数另列。入口和各阶段均独占消费，无自动重试/继续令牌。

训练及短验收共用科学实现；CPU夹具仅缩小画布/日程/成员，明确标记CPU_FIXTURE_ONLY，不能进入生产合同。构造指标、CPU秒数和零显存不作科学/硬件性能证据。

## 终点与成本口径

U/D仅固定e120 EMA、单整图FP32、batch4；同一次预测按原规则生成top600和top300，无NMS/新阈值，保留预算后原始ignore、不补位。仅访问原548 development，不访问1602。两预算分别给出13项COCO和7项原主评价指标，并报告D−N、D−U、U−N全部正负差值。两组收益/保护门槛和D/N耗时、峰allocated≤1.05均取原配置，未定义指标保留并不能判通过。质量失败仍完成固定后链。

成本使用原SHA绑定的固定64 development ID，FP32/batch4，每模型2遍warmup，六次实测顺序为N-U-D、N-D-U、U-N-D、U-D-N、D-N-U、D-U-N，每模型在每位置两次。18次实测及均值/样本SD、allocated/reserved峰值全部保留。边界包含解码/resize/H2D/forward/排序/原ignore与CUDA同步，排除模型载入和GT解析；不是batch1实时性结论。

## 预定中心记录

读取既有第三层输入的detached reference作为第二层粗框，不改变forward；与同次最终300个ordinary query的框一起保留。用本模型最终预测、原Hungarian及训练有效GT独立配对；按原图raw area `0<area<=1024`筛小GT，无置信筛选。归一化中心转回各图像素，误差为像素L2除以`sqrt(raw_area)`；报告粗/最终均值、中位数、最终减粗正/零/负计数和实际配对分母。两模型的配对不强制相同，不能把差值当配对因果效应。N缺粗中心记录保持UNKNOWN，不补推理。原始逐query中心留服务器，只精简回传摘要。这些是终点局部支持证据，不证明训练根因。

## CLI、监督和精简导出

实际命令依次为`prepare --spec --source --output`、`bind --spec --source --cpu-gate --prepare-gate --authorization --output`、`contract-check --contract`、`cuda-check --contract --output`、`launch --contract --session`。所有路径由冻结spec和引用回执绑定，不能手工更改合同来跳过准入。父进程隐藏CUDA，工作进程逻辑0绑定物理PCI和完整UUID；原环境、资源年龄25秒、温度/功率/时钟及存储保护不改。双层硬期限和PID/starttime绑定的子树清理来自已验收短入口；新生产异常STOP，保留现场，不自动修复或重试。

每阶段完整事件按序号/身份/前项SHA链流式验证，内存不随完整日志长度增长；检查点完整SHA/状态与事件流分别验证。根闭合要求全部阶段、实际forward/更新、累计成本和导出索引一致。精简白名单≤20MiB，含六份评分JSON、完整比较、成本、中心摘要及必要合同/源/完成/退出回执。完整checkpoint、预测、逐更新事件和逐query中心留服务器；ROOT_CLOSED由收尾collector单独取回，避免索引循环引用。启动后仅一次即时快照，不建立本机持续轮询。

## 部署兼容性修订

85e9196外部安装器在系统Python3.10调用`hashlib.file_digest`，解包前STOP；原提交、metadata-001上传材料和STOP保留。用户限定授权后，实际入口改为Git中的`tools/deploy_h2_source.py`，只用Python3.10可用的stdlib及1MiB分块SHA-256，不升级训练环境，不加载模型或启动作业。

安装器要求显式固定源码包、manifest及安装器自身SHA和commit。bootstrap脚本必须等于包内`tools/deploy_h2_source.py`的manifest条目；拒绝已有目标/回执、不规则成员、越界路径、缺项/额外项、大小或逐文件SHA不符。独占创建新release，不覆盖旧上传和STOP；失败回执及已解包部分保留，不自动清理/重试。新部署使用metadata-002及独立release。

实际CLI测试`tools/check_h2_deploy_cli.py`在Python3.10/3.12分别覆盖正常包校验/解包/完整manifest、损坏包、缺包、缺成员、包内同大小损坏文件、重复目标、额外成员、越界、链接和错误commit。多块构造数据只验证SHA与拒绝行为；不代表真实材料准入。生产中先用该入口取得SOURCE_VERIFIED，再从正式prepare/绑定/隐藏CUDA检查/物理身份检查/唯一launch推进；短GPU验收和原成本仍仅只读复用。
