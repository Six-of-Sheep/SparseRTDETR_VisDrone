# NMS抑制归因：seed0限定CPU诊断（冻结配置）

任务ID：`r18-nms-suppressor-attribution-20261030-001`。日期：2026-10-30。

## 目的与边界

对原C（五视图、每视图top600、classwise hard NMS IoU>.5）候选池，统计在NMS阶段失去IoU≥τ正确类支持的小GT：每个被删的高质量候选由哪个保留框实际抑制，抑制者与本GT及同类其他GT的关系，以及抑制者在预算/raw-ignore/class100后的去向。

输出只是描述性计数，用于判断下一步是否存在一个具体、可检验的方法依据。不宣称创新、不宣称收益、不换算为可追回TP或AP上界，不比较Soft-NMS/投票等旧结果，不扫阈值。有规律不等于有创新；不自动提出新模块、不进入评分或训练。

授权范围：必要代码、针对性CPU检查、CI、冻结，以及关口全部通过后的**单次seed0执行**。仅137批、548图。GPU、模型构造、检测器前向、训练、AP/AR评分均为0。seed1/2不可从本入口到达，不自动执行。

## 复用工具

- `tools/c_stage_audit.py`、`tools/c_stage_core.py`：从`codex/sahi-c-stage-audit`@3a36cf6逐字节导入，与2026-10-18实际执行的58e0838相同（SHA 0592a54f…/30cbf592…）。
- `tools/nms_attribution.py`：镜像NMS循环与归因记录，纯numpy binary64。
- `tools/nms_suppressor_audit.py`：seed0专用CLI，沿用原admit/run/fixture模式、父契约b2ac33e7…、370份外部源码逐SHA、`cpu_only`、资源门禁与评分禁用补丁。
- 冻结外部源：`sahi-softnms-fixed-20261017-001/revision-007/release/source`；解释器为契约绑定的`/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python`。

## 定义

- 候选池：五视图`select_view_candidates`跨类top600并集，身份(view, query, category)唯一。
- 顺序：`(−score, view, query, category)`；分类别处理；每轮首项为pivot并保留，删除IoU严格>0.5者（numpy binary64）。
- 实际抑制者s(c)：按该顺序第一个与c的IoU>0.5的pivot；唯一、确定、必为保留框；无链式抑制。并列由(view, query, category)升序决定，与原C一致。IoU落在0.5±1e−9的比较单独计数，不改判。
- GT：原bridge过滤后iscrowd=0、0<area≤1024的小GT；主τ=0.75、副τ=0.5；归因对象为该τ下首次丢失阶段为hard_nms的GT。NMS在预算前，归因与预算无关；抑制者去向按600/300分别记录。
- 被删候选：候选池中全部同类且IoU(c,GT)≥τ者，逐个记录，不只取最佳。

## 多GT歧义（不作排他指派）

- 同一候选对多个GT均≥τ：各GT下各计一次并记`shared_with_gts`。
- 抑制者对同图同类全部非crowd GT（不限尺寸）求IoU，argmax取binary64精确并列列表，不破并列；无重叠时列表为空。
- 同一保留框删除多个GT的支持：按框聚合直方图。
- crowd GT与类别0 ignore区域不进入argmax；抑制者落入ignore区域由去向字段体现。NMS分类别，跨类抑制在结构上不存在。

## 类别（描述性，按(GT, 候选)对计）

K1_mid：抑制者argmax为本GT且IoU≥.5；K1_low：argmax为本GT且IoU<.5；K2：argmax为另一GT且≥.5；K3：与任何GT均<.5或无重叠。每类再按同视图/跨视图交叉。GT级记录类别组合，混合者单列。

## 复现关口（任一失败即STOP，不重试，无容差）

1. 父契约SHA、370份外部源码SHA、解释器字节、boot_id按`load_contract`原规则。
2. 137个raw batch逐SHA解码；每图1500几何框；raw映射一致。
3. 每图、每预算，`kept[:budget]`序列化与原C canonical预测逐候选相等；raw-ignore后与原bridge过滤结果相等。
4. 镜像NMS保留身份序列与冻结`classwise_nms`逐图逐类相等（全部548图）。
5. 预算前三阶段最大IoU在600/300间相等。
6. 全部完成后，seed0六桶计数在600/300×0.5/0.75四组上与2026-10-18 `seed0-SUMMARY.json`完全相等。

## 删除的规则与禁止推断

不设"某类占比≥60%即支持某方法"；不设"同视图占多则多视图无用"；不设"分布接近均衡即关闭方向"。阴性只限定本固定统计。

## 资源与回传上限

| 项目 | 上限 |
|---|---|
| 计算 | 单进程；OMP/MKL/OPENBLAS/NUMEXPR线程1；torch线程1；CUDA、模型构造、前向、AP调用为0并由断言检查 |
| 时间 | 进程墙钟≤900秒，逐批检查，超限STOP |
| 内存 | VmHWM≤6 GiB，逐批检查；另沿原RAM≥4GiB/磁盘≥4GiB/温度<85°C门禁 |
| 服务器新增 | ≤1 GiB（含逐GT jsonl），逐批检查 |
| 回传 | ≤1 MiB：ADMISSION、RESULT、CLOSED/STOP、汇总、CPU_GATE、SOURCE与SHA回执；逐GT jsonl留服务器只记索引 |

## 执行命令（冻结release内，关口通过后各一次）

```
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
PARENT=.../sahi-softnms-fixed-20261017-001/revision-007/execution-contract.json
$PY -B release-001/source/tools/nms_suppressor_audit.py admit --parent $PARENT --output <task>/admission-001
$PY -B release-001/source/tools/nms_suppressor_audit.py run   --parent $PARENT --output <task>/attempt-001
```

冻结后的真实诊断若出现异常、身份不一致或重建失败，保留证据并STOP，不自动重跑。
