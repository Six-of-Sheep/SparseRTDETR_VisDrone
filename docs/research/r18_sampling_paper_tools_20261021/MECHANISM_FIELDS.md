# 已有机制摘要的展示契约

本工具只读取已有JSON，不导入torch、不加载模型、不运行推理或评价。源接口只读核对固定提交 `ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb` 的 `span_observation.py` 和 `span_worker.py`。普通query与DN分开，三个decoder层及各观测窗口保留独立记录。

## 输入与命令

支持原生 `sampling-eNNN-w000.json` 的row列表，既有 `MECHANISM_SUMMARY.json`，以及既有 `MECHANISM_EFFECT_SUMMARY.json`。后者可以单独作为 `--input`，也可以通过 `--effects` 与第一种summary配合。每个源文件都须提供SHA；错SHA、arm/group不符、错误结构拒绝。缺失字段显示 `UNKNOWN`。

```powershell
python -B tools/paper_sampling_mechanism.py `
  --input <已有MECHANISM_SUMMARY.json的绝对路径> --sha256 <该文件SHA256> `
  --effects <已有MECHANISM_EFFECT_SUMMARY.json的绝对路径> --effects-sha256 <该文件SHA256> `
  --scope gpu-acceptance-only --plots --output <新的独立输出目录>
```

`--scope`必须显式选择 `gpu-acceptance-only`、`selected-training-windows` 或 `test-fixture`。验收summary的 `accept-*` 身份不能标为正式训练。测试夹具用 `data_kind=SYNTHETIC_TEST_ONLY` 包裹 `observations`，不能进入实验scope。原始row没有内嵌闭合或来源权限，scope仍由调用者明确绑定；SHA只证明字节，不替代合同、闭合或来源认证。输出目录不得已存在。

输出 `reference-counts.csv`、`measurements.csv`、`MECHANISM_DISPLAY.md`、`DISPLAY_PROVENANCE.json`；`--plots`另生成ordinary与DN分开的SVG/PNG。每图包含对应scope水印。真实工程验收示例全部标注 **GPU engineering acceptance only, not formal training**；不得并入正式e120结果表。

## 字段和分母

|源字段|范围/单位|展示原则|
|---|---|---|
|reference_instances / below_cell_x/y/any / trigger_fraction_*|被观测microbatch的该组全部reference实例；不是GT实例|原计数与原比例直接显示，计数可核对时检查比例一致；不以采样query数为分母|
|floor_applied_instances|P实际施加下限的reference数，B0/B1为0|与“如果按一cell阈值判断，reference会偏小”的below计数区分|
|reference_extent_cells|原reference宽高乘最细层W/H，宽高轴展平后的统计|count包含轴；不能冒称目标数或实际跨度|
|sampled_queries_per_image|每图每组最多8个query，选择组内最前序query|不是所有query的随机样本；不能用其位置/跨度代表总体|
|actual_location_normalized_x/y|上述样本在最细层的真实位置，所有head/points|位置可越界；不clip、不生成新分布|
|actual_offset_cells|真实位置相对reference中心，换算最细层cell，两个轴展平|不是预测offset参数本身|
|actual_span_cells_x/y|同一个样本query/head在最细层各点的max-min|reference extent有下限不保证这个span有下限；offset可缩到0|
|outside_image_fraction|样本点的归一化位置任一轴在[0,1]之外|不等同四个双线性邻点全部无效；不重算|
|same_stencil_fraction|本次和去下限反事实的双线性base索引相同的比例|同stencil仍可有不同权重、插值值和输出|
|bilinear_effective_weight_abs_change|真实位置与去下限反事实的四邻点有效插值权重绝对差|包含越界邻点的零填充；B1每层点数不同，count可不同|
|attention_aggregate_output_abs_change|保持相同已学offset/value/attention weights，去下限的attention输出绝对差|反事实不是另一个已训练模型，不支持AP或因果质量结论|
|DN_includes_native_padding|原生DN的padding包括在DN组中|原始row明确记录；summary未含该flag时字段保持UNKNOWN，正文说明接口本身包含padding|

所有count/min/mean/p50/p90/max按原record保留，**不合并p50/p90为总体分位数**。`MECHANISM_SUMMARY`只返回每record的span均值，未返回对应采样count及层/窗口身份；展示逐条点及 `identity_unavailable`，不做无权均值。`MECHANISM_EFFECT_SUMMARY`已经给出count加权均值，工具按原值显示并标为 `source_count_weighted_mean`，不重新加权/声称总体分布。

图中优先展示源已有的count加权均值；若无，逐条显示record均值，jitter只用于避免遮挡。reference比例是原reference分母，span和插值变化是至多8-query样本分母。原始row的分位数仅保留表内，不绘制伪总体箱线图或置信区间。

## 已知解释限制

既有验收summary只有update1/2三层累计reference计数、逐record span均值，以及四类count加权effect均值；实际位置、offset、分位数、stencil和按GT尺度关联没有在这两个小型summary中保留，显示UNKNOWN。正式训练只在冻结窗口记录，工具不要求运行任务补采，也不通过模型forward补齐。现有row使用 `uses_GT=false`，故不从这些字段推断“对小GT触发率”。

工具不产生任何科学门槛或收益PASS，输出显式 `scientific_quality_evaluated=false`。测试数据及工程验收早期数据都不能用于填补三臂正式结果。
