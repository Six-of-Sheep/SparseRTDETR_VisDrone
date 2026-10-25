# H2终态材料转论文表图

本工具完成“收到精简JSON后整理成表图”这一环节。它不连接服务器、不导入模型或评价器、不重算AP、不生成新的科学门槛，也不修改稿件的摘要与结论。运行实验仍固定在PR21的`e50bedbe8e41bdfd3c47d09f7076d9644e438e26`，本工具属于PR22论文分支。

入口：[tools/h2_paper_results.py](../../../tools/h2_paper_results.py)。解释规则见[RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md)。

## 输入：原始精简导出＋独立根CLOSED

使用原`slim-export`平面目录中的`SHA_INDEX.json`和其列出的文件。原件保持原字节，不预先改名、转换单位或手写摘要。索引至少需要包含：

|材料|文件名或模式|用途|
|---|---|---|
|六份评分|`N-600.json`、`N-300.json`、`U-600.json`、`U-300.json`、`D-600.json`、`D-300.json`|两评价器、两预算的完整绝对指标|
|评分引用和原比较|`score-RESULT.json`、`compare-RESULT.json`|角色/预算绑定，原差值、22项门槛与质量/成本裁决|
|成本|`cost-RESULT.json`、`CUMULATIVE_COST.json`|每臂6次测量、原均值/样本SD/峰值/比值与累计成本|
|中心|`infer-U-CENTER_SUMMARY.json`、`infer-D-CENTER_SUMMARY.json`|各臂自己的匹配小GT，粗/最终中心统计|
|身份和闭合|`contract.json`、`source-manifest.json`、根`RESULT.json`和`EXECUTION_EXIT.json`、七阶段RESULT/CLOSED/退出、U/D终点和binding|核对回执、e120 EMA和原运行身份|

**根`CLOSED.json`由生产程序在export之后写入，不在`SHA_INDEX.json`内。** 收尾时需单独取回并核SHA，再通过`--closed`和`--closed-sha256`传入。工具不自动寻找它，也不按JSON中的服务器路径打开文件。

SHA参数应取自已核验的精简回传收据，而不是用一个刚被修改的文件自算SHA来替代准入。根CLOSED缺失可以生成缺证预览，所有可报告总体裁决保留UNKNOWN。输入明确STOP、错身份、坏SHA、重复角色、冲突单位或相互矛盾的原裁决会拒绝，且不创建结果目录。

## 正式材料的离线命令

在单独的论文工具环境安装绘图依赖；这不是训练环境安装命令：

```powershell
python -m pip install -r environment/requirements-paper-results.txt
python -B tools/h2_paper_results.py `
  --index '<已核验精简目录>/SHA_INDEX.json' `
  --index-sha256 '<回传收据中的索引SHA256>' `
  --closed '<另行回传的根CLOSED.json>' `
  --closed-sha256 '<根CLOSED的已核验SHA256>' `
  --output '<已存在父目录下的新结果目录>'
```

不传`--synthetic`。需要纯stdlib表格时传`--no-figures`；默认用matplotlib输出SVG及PNG。已有输出目录一律拒绝，不覆盖旧表图；输出不能放在输入回执目录内。

固定身份为task `r18-h2-formal-20261024-001`、attempt `attempt-formal-001`、seed0、profile `LAB_GPU_ONCE`、原commit及canonical contract SHA `8450bbaff7d15da0db0343da49aa0d59d9455ff186458a664c2707db0397e8df`。这不是可放宽身份的通用实验导入器；未来另一个实验需明确适配，不能仅改参数套入。

## 输出与单位

|输出|内容|
|---|---|
|`TABLES.md`|两预算、两评价器的绝对值与D−N/D−U/U−N全部正负差值；22门槛、成本、中心及缺证清单|
|`absolute.csv`、`deltas.csv`|120项绝对值、120项成对指标；每行有正式/缺证/合成身份|
|`gates.csv`|22项原门槛；`recorded_status`与当前证据可解释的`status`分开|
|`cost_summary.csv`、`cost_trials.csv`、`ratios.csv`|三臂均值/测速样本SD、18条实测、三组原比值；allocated/reserved分列|
|`centers.csv`|N缺失保持UNKNOWN，U/D的均值、中位数、正零负计数和匹配分母|
|`MAIN_TABLE.tex`|可编辑主表；合成或缺证标识直接出现在可渲染表内|
|`paired_deltas.svg/png`|两预算六个核心指标的全部三组差值，零线和负值保留|
|`cost.svg/png`|固定64图耗时、测速重复样本SD及峰allocated|
|`centers.svg/png`|U/D各自粗/最终中心均值和中位数，标各自分母|
|`NORMALIZED.json`、`OUTPUT_INDEX.json`|机器可读标量、输入来源/缺证/限制、输出逐文件SHA及工具SHA|

COCO原始13项为0..1，展示为百分数；`-1`表示无定义，显示UNKNOWN，不变成−100%。原主评价器7项已经为0..100，不再乘100。`compare.deltas_pp`已经是百分点，直接引用。

原裁决基于未舍入值。工具检查副本、差值和固定门槛的一致性，**不根据表格的六位小数重新决定PASS/FAIL**。差值略低于0.30但显示为0.300000时仍保留原FAIL。内部互相矛盾时拒绝，不修写回执。

`engineering=RECEIPTS_COMPLETE`仅表示已返回的小文件身份/引用/闭合检查齐备；不是重新核验服务器大checkpoint、逐更新事件或实时进程。`reported_quality/reported_cost`照录原布尔值；必要材料缺失时报告层`quality/cost=UNKNOWN`。合成模式始终`NOT_SCIENTIFIC`，即使测试材料写了假设性的PASS布尔值。

## 中心和成本解释边界

- 中心误差为原图像素距离除以√原GT面积，无量纲；`mean_change=final_mean-coarse_mean`，负值表示该臂自己的已选匹配子集均值下降。
- 每臂使用自己的最终Hungarian配对；不能把U/D摘要之差当作同一GT的配对干预。中位数方向与均值相反时同时保留；不写成梯度或训练因果证据。
- 原摘要只有均值、中位数、计数等汇总，不提供完整误差分布。工具不推断其他分位数、置信区间或共同GT覆盖，不读取`raw_records`、预测或图像来补足。
- 六次成本重复的SD是测速波动，不是跨seed误差条。峰allocated/reserved为各臂六次最大值；D/N成本条件维持原1.05，U/N及D/U只报告。
- N沿用历史B0终点，旧STOP与缺原训练CLOSED保留；其评分不属于本次新评分，中心资料不补造。D−U是主要受控比较。

## 合成示例与检查

以下命令只生成虚构标量回执，不运行实际实验。输出路径必须新建，父目录已存在：

```powershell
python -B tests/h2_paper_fixture.py --output '<新建的合成输入目录>'
# 上一步打印index/closed路径及SHA。
python -B tools/h2_paper_results.py --synthetic `
  --index '<合成目录>/SHA_INDEX.json' --index-sha256 '<打印的SHA>' `
  --closed '<打印的ROOT_CLOSED路径>' --closed-sha256 '<打印的SHA>' `
  --output '<新建的合成表图目录>'
python -B -m unittest discover -s tests -p test_h2_paper_results.py -v
```

合成模式要求显式CLI开关和contract中的`paper_tool_synthetic`标记，不能作为生产输入。测试刻意给部分比较设置假设性科学布尔值以检查FAIL/PASS原样保留；实际生产CPU夹具的这两个字段为null。所有CSV行、Markdown、可见LaTeX标题及SVG/PNG均标`SYNTHETIC`，不提交这些构造分数到论文正式结果。

验收覆盖完整CLI与实际绘图、单位、舍入边界、−1/缺指标、缺根闭合/评分文件、坏SHA、调换角色、错误commit/seed/单位、重复/改变门槛、定义性冲突、科学FAIL/成本PASS分离、空中心/分母、STOP、NaN/溢出和已有目录拒绝。CI在Python3.10/3.12运行同一套检查，不跳过绘图测试。

## 生产schema依据

只读核对固定e50bedb：

- `src/sparse_rtdetr/research/h2_formal_worker.py`：训练终点、评分202–212行、成本214–245行、比较247–266行、中心179–200行。
- `src/sparse_rtdetr/research/h2_formal.py`：阶段闭合163–164行、精简导出222–239行、根闭合298行。
- `src/sparse_rtdetr/r0/evaluation.py`：62–87行，双评价器输出及COCO单位。

代码审查与合成CLI通过不等于真实终态材料已准入。本轮仅交付工具；真实六评分、根闭合及成本/中心材料仍待原链终态按既有授权回传后处理。
