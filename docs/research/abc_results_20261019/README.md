# 原A/B/C成果冻结包

本轮实验扩展已结束：A共同协议基线、B常规后处理控制、C固定SAHI式实用方案冻结保留，不自动调参或启动后续训练。失败候选及原裁决原样保留。

- [实验主表与全部成对统计](MASTER.md)：两个集合独立报告，含两预算与所有AP/AR取舍。
- [论文四部分草稿](MANUSCRIPT.md)：设置、主结果、消融及失败对照、局限性。
- [论文缺口](GAPS.md)：已有证据／需新证据或实验／当前限制披露。
- [PR #9工程审查与集成建议](PR9_REVIEW.md)：无硬前置PR依赖；CI新增路径覆盖缺口须在集成前处理，未自动合并。
- `ABSOLUTE_ALL.csv`、`ABSOLUTE_MEAN_SD.csv`：36个各自集合/seed/方案/预算终点及绝对值均值/样本SD。
- `PAIRED_ALL_PP.csv`、`PAIRED_MEAN_SD_PP.csv`：C−A、B−A、C−B逐seed及均值/SD/最差，13项COCO指标完整保留。
- `OFFICIAL_MEAN_SD_PP.csv`、`OFFICIAL_PAIRED_MEAN_SD_PP.csv`：认证摘要中可用的主评价器字段独立汇总，原pp不再乘100。
- `FROZEN_QUALITY_GATES.csv`：两个集合共42项条件，未舍入裁决。
- `COST_FIXED64.csv`、`COST_MEAN_SD.csv`：历史与补充执行复测均为development固定64，非数据集全量吞吐；allocated/reserved分列。
- `CLOSED_CANDIDATES.csv`：36行旧失败候选/控制的两预算结果，不虚构未执行seed或未测成本。
- `TABLE_MAIN.tex`及`FIG1_ABSOLUTE`、`FIG2_TRADEOFFS`、`FIG3_COST64`的PDF/SVG/PNG：论文矢量图、预览和LaTeX表。
- `SOURCE_SCALARS.json`、`VALIDATION.json`：仅必要认证标量与SHA引用、纯统计验证；大材料只在服务器，索引不是备份。

同一repo的`tools/consolidate_abc_results.py`只读取此标量包、生成表图，不导入模型、预测、评价器、torch或CUDA。CSV保留未舍入精度；图中误差棒是3 seed样本SD，非CI；两数据集不合并AP。生产运行源码保持97c40e6；本包是后续整理成果，不能冒充运行时版本。
