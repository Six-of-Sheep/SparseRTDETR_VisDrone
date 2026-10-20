# P：三臂生产执行准备（不是GPU执行授权）

本修订只把已接受的B0/B1/P方案接入显式生产接口。原设计版本、CPU原型及所有旧失败保持；原A/C、DQ暂停和蒸馏关闭裁决保持。B0不是历史后处理B。三臂科学字段与前版逐项相同，仅设计JSON的状态/已实现标记更新。没有新候选、门槛修改或早期选模。

## 机制与观测

P限制的是最细层**reference sampling extent**：`w'=max(w,1/W0), h'=max(h,1/H0)`。实际位置仍是 `c+0.5*(offset/n)*extent`，实际跨度为这些位置的最大值减最小值；offset可以为零或收缩，所以不是“实际采样跨度至少一个单元”。框回归reference及位置编码不变，普通query和DN都适用。reference宽高不能由GT大小代替；DN统计包含原生padding。

固定观测为e001/e040/e080/e117/e118/e120的第一个更新、第一物理microbatch、三层decoder。GPU验收的两个窗口也观测以检查重放一致性。每层分别汇总普通300 query和原生DN：

- 全部reference的x/y/任一维低于一个单元的计数、分母和比例，以及P实际触发数；B0/B1只记录“若使用下限会触发”的比例。
- 每图每组按原query次序取前8个，汇总真实采样位置x/y、距中心偏移（feature cell单位）、每head实际跨度x/y、图外比例，记录count/min/mean/p50/p90/max；不记录逐点全量张量。
- 在同一query、学习到的offset、value和attention权重下，计算去掉下限的局部反事实：最细层同stencil比例、有效双线性权重绝对变化、attention聚合输出绝对变化。相同stencil仍可能有不同权重和输出，不能解释成采样等价。

这些是固定时点、有限样本的机制描述，不是质量裁决。观测不读GT、无随机采样、不反向、不改变训练输出/RNG/state；增加少量core计算，训练占用时间照计。它不是额外model forward。保存前恢复原core callable；测速关闭观测。

## 最终入口与绑定

`r18_sampling_span.py`统一所有CLI分支初始化，主动移除子进程PYTHONPATH；模块必须来自本release。新`span_model`验证三层精确class、点分配、scale buffer、采样core及P开关。新checkpoint格式绑定arm、scope、源/合同/输入/初始化SHA；**旧公共入口和checkpoint保护未放宽**，旧validator仍拒绝P。

生命周期：`prepare → bind → run → admission → accept-B0/B1/P → train-B0/B1/P → infer-B0/B1/P → score → cost → compare → export → RESULT/CLOSED → 进程退出`。正式run14个监督阶段，按顺序闭合，缺事件、错序、篡改结果或非零退出拒绝。普通质量不通过仍完成固定三臂/评价/测速；工程异常STOP，不自动重试/恢复。输出目录和一次性消费回执以排他创建拒绝重复启动。

准备使用现有严格train_core4869和development548 reader、manifest SHA、role路径、逐成员图像/物理标注SHA。两个role精确隔离，不允许1602或其他集合。原A checkpoint仅用于后续成本；历史A/C标量只作上下文，不替代同轮B0因果控制。准备不做model forward、不初始化CUDA，不生成GPU合同或令牌。

每个子进程（包括失败）持久保存argv、cwd、解释器、stdout、stderr、退出码/异常；监督器按PID/start身份处理自己的子树，不操作其他作业。准备版使用PR15固定commit的生命周期/资源/CUDA身份基础设施，源文件SHA和改编范围见`manifests/sampling_span_infrastructure_origin.json`；没有复用蒸馏科学配方、父EMA或12轮日程。

## 固定GPU短验收（本轮不执行）

三臂分别使用原训练集epoch1的前两个逻辑16窗口；batch8×accum2。实际更新序列`1,2,restore1,2`，每臂3次真实更新、2个唯一窗口、6次forward。固定BF16 forward/FP32 loss；记录真实CUDA UUID与逻辑/物理映射、BF16支持、有限loss/梯度/参数/optimizer/EMA、裁剪前范数、输入/输出/DN/梯度摘要。保存C1，再独立重建并恢复，要求重放第二更新结果及完整model/optimizer/EMA/RNG/游标精确一致；任何不一致STOP，不换精度/容差。

记录每次完整更新耗时、阶段耗时、峰值allocated/reserved。短验收含初始化、强增强、早期optimizer/EMA，不能完整代表晚期训练、weak阶段或最终速度，也不根据早期loss/触发率决定收益。三臂全部PASS才进入正式训练；正式各自重新构建共同ImageNet初态、新optimizer/EMA和seed0 RNG，从update0开始，禁止接验收state。

## 预算与触界

| 阶段 | 实际更新 | batch forward | GPU阶段占用时间预算依据 |
|---|---:|---:|---|
| 准备与CPU验收 | 科学0 | 生产0；合成另记 | 不计GPU小时 |
| 三臂保存恢复短验收 | 9（含3次重放） | 18 | 未实测；纳入60小时 |
| 三臂正式120轮 | 109440 | 218880 | 原A32轮14599.8449秒线性外推约45.6小时，仅估计 |
| e120 EMA development548 | 0 | 411 | 每臂137批，两个预算复用一次预测 |
| A/B0/B1/P同轮固定64成本 | 0 | 384 | batch4；每模型2暖机+4测量完整pass |
| 双评价与比较/导出 | 0 | 0 | 6组、12评价器调用，CPU |
| 合计 | 109449 | **219693** | 原计划45–55 GPU小时或更长；60为硬上限，不是完成保证 |

成本逐attempt保留，生产必须显式提供同任务`prior_cost`回执（GPU秒、forward整数、原attempt索引，零也需说明）；旧成本不归零。GPU账本在每个GPU阶段启动前建立、退出时结算，按独占分配的阶段墙钟计时，包含加载、保存、I/O/等待，不按GPU利用率折算。失败forward预算事前预留不退回，未知成本不能默认填0。

达到60小时前180秒、forward余额不足或100 GiB前2 GiB时，worker在完整更新边界保存新的`budget-stop.pth`并STOP；若已在推理，固定训练endpoint已保存。supervisor另守60小时硬界和实时资源保护；若无法到达安全边界，保留最近完整checkpoint/故障现场，不能承诺强杀时的新checkpoint完整。无自动恢复。峰值allocated训练上限18 GiB，任务产物100 GiB，现有温度、RAM、SSD/shm及UUID保护保持。

固定checkpoint e000/e001/e040/e080/e117/e118/e119/e120，保留全部8份/臂，不进行滚动删除；其他epoch写完成审计事件并注明没有新checkpoint。另保留验收C1和异常现场；状态/缓存留服务器。60小时不足时结束，不扩预算或科学范围。

测速FP32、batch4，完整decode/resize/H2D/forward/排序/raw-ignore/synchronize同一边界；4轮Latin square平衡顺序，报告每轮与均值/样本SD、allocated/reserved峰值。不得按点数推算提速，P/B0端到端及allocated各≤1.05，历史A额外作参照。I/O及累计预算检查开销包含在本次保守端到端边界内并一致施加。

## 可执行命令与剩余准入

所有命令从冻结release根执行，用原隔离R18解释器，不安装/升级原训练环境。以下是待授权执行顺序，不是本轮已运行记录。`lab-spec.json`必须提供新attempt/root/nonce、真实解释器、boot/GPU UUID/storage身份、固定资源/environment、两role manifest三字段引用和roots、ImageNet引用、原A checkpoint/binding、selection/fixed64、冻结标量和prior_cost；路径不得凭目录名或默认值替代认证。

```bash
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_span.py prepare --spec "$SPAN_META/lab-spec.json" --source "$SPAN_META/source.json" --output "$SPAN_META/preparation-001"
# 上项PREPARE_PASS绑定入lab-spec的prepare_gate；获得另行GPU授权后：
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_span.py bind --spec "$SPAN_META/lab-spec.json" --source "$SPAN_META/source.json" --cpu-gate "$SPAN_META/CPU_GATE.json" --authorization "$SPAN_META/GPU_AUTHORIZATION.json" --output "$SPAN_META/contract.json"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_span.py run --contract "$SPAN_META/contract.json"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_span.py verify --contract "$SPAN_META/contract.json"
```

源manifest只描述字节，不授予启动权限；CPU证据绑定最终运行字节，必要迁移通过明确父证据比对，不重写旧source或contract。生产合同、GPU授权和nonce须待用户授权/新鲜材料与资源准入后建立，本轮没有它们。

CPU检查命令：`CUDA_VISIBLE_DEVICES='' python -B tools/check_sampling_span_production_cpu.py --output <新的输出目录>`，必须从仓库/部署cwd启动，无额外PYTHONPATH。CPU profile用6图两role、128画布FP32和缩短两epoch，只证明真实CLI/公共数据解析原语/三臂/恢复/双评价器/阶段闭合/导出及拒绝路径。不能冒充真实4869/548材料、GPU、BF16、768画布forward或120轮执行已验收。最终实际CPU回执、最终commit/source manifest及CI状态见D盘本次交付索引；旧13项叶层/构建证据保留。

尚须未来服务器新鲜准入与真实GPU验收：当前材料路径及字节、ImageNet/原A、真实训练loader/spawn、PyTorch2.4.1 BF16/deterministic backward完整重放、768×1344显存/速度、60小时能否完成。CPU通过不预判它们。正式质量只用e120 EMA固定终点及原门槛，不修改旧科学结果。

观测实现通过实际Linear输出hook取得offset，不在autocast内额外以no_grad重算投影；CPU FP32/BF16叶层分别检查开关观测的输出及全部梯度一致，不能替代真实GPU BF16验收。准备阶段每32成员刷新真实资源观测；监督阶段使用追加日志中最后完整资源记录，仍拒绝过期/缺失/身份冲突，不把重复读取当作科学重试。

监督器的语义进展来自经身份/SHA链验证的追加事件日志，原progress.json保留为人类查看指针；不是在读失败后重试或忽略错误。原训练reader对物理行的分类和过滤不变，development继续使用既有严格几何保护。

实时监督只增量读取完整追加事件并保存游标、链SHA、PID及单调时钟，拒绝日志消失、替换、截断和错误追加，避免120轮逐次重扫完整日志。闭合时仍全量核验最终事件字节和完整SHA链，不放宽终态保护。

## preparation-002：用途字段适配（仅CPU授权）

旧preparation-001在读取真实成员后STOP，原因是外部spec构造器将历史`fixed_ids`（train_core两组16张）误作development固定64张。旧release、合同和STOP不改。本修订在实际`prepare --spec`及`bind`入口调用显式`span_inputs.adapt`，已封印合同在`load`中再次验证；缺失或冲突不接受默认值。

原始spec必须显式提供`input_lists.training_acceptance`与`input_lists.timing`，各含`purpose/role/source/ids`四字段。前者purpose为`historical_training_acceptance_2x16_audit_only`、role=train_core、source为SHA `6dbe86d56d5dd4053bcbc373276ad0c137cc4e892e48035bb41ee2173d1ddba7`的原合同，ids保留原2×16结构。后者purpose为`fixed64_development_cost`、role=development、source为原认证trace-selection三字段引用，ids逐项等于原`selected[].stable_image_id`。两者分别检查认证来源字节、数量、顺序、成员role及manifest SHA，并保存用途证明。仅timing派生下游`fixed_ids`；已有字段若冲突立即拒绝。

历史2×16清单仅作审计，**不是新GPU验收的输入选择**。新验收仍由原训练采样器取得epoch1首两个logical16窗口，未改变公式、增强、采样、更新、门槛或219693 forward/60 GPU小时上限。

真实CPU检查使用`tools/check_sampling_span_lab_inputs.py --spec <未预映射的真实spec> --source <本release manifest> --output <新目录>`。九类拒绝直接调用最终CLI（清单互换、两类各自同数量错成员、缺字段、缺清单、两类各自错SHA、错大小、旧字段混接），然后正常spec原样交CLI完成全部真实成员reader、原A状态绑定与三臂CPU初始化。子进程不继承PYTHONPATH，留存命令、stdout/stderr、退出码；拒绝须发生在封印合同/reader前。正常路径任何异常立即STOP，不重试。

108项production-cpu-006旧门禁保持原source/contract身份。适配修订通过明确源码差异和受影响路径检查绑定其未受影响的恢复、跨epoch、双评价器、闭合与异常/重复拒绝证据；不声称整个旧source字节未变，不重写旧门禁或放宽其byte-identical校验。GPU/BF16/真实更新与正式训练继续等待授权。
