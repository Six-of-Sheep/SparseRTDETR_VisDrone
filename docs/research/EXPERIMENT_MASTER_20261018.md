# 当前实验主表（原记录汇总，不重评分）

原C保留，所有已关闭配方不重开。原共同协议适配R18三seed；同缓存的Soft-NMS/投票不是新训练seed。微调仅seed0双臂。完整两预算分数、成本/源码/报告与小文件SHA见MASTER.json/CSV。

| 系列 | seed | 主600方案 | AP% | APS% | 质量状态 | 成本 |
|---|---:|---|---:|---:|---|---|
| ABC | 0 | A | 29.789280 | 22.176186 | REFERENCE | 1.000000×A（实测） |
| ABC | 0 | B | 30.121781 | 22.444626 | CONTROL_NO_INDEPENDENT_PASS_CLAIM | 1.113487×A（实测） |
| ABC | 0 | C | 31.764409 | 24.969143 | PASS_C_MINUS_A_SEVEN | 4.672504×A（实测） |
| Vote | 0 | V-whole | 29.871795 | 22.227830 | CONTROL_ONLY | 未测/未裁决 |
| Vote | 0 | V-five | 31.346144 | 24.840764 | FAIL_CLOSED | 未测/未裁决 |
| ABC | 1 | A | 29.941229 | 22.416615 | REFERENCE | 1.000000×A（实测） |
| ABC | 1 | B | 30.166152 | 22.595735 | CONTROL_NO_INDEPENDENT_PASS_CLAIM | 1.133253×A（实测） |
| ABC | 1 | C | 31.894856 | 24.815421 | PASS_C_MINUS_A_SEVEN | 4.710083×A（实测） |
| Vote | 1 | V-whole | 29.984170 | 22.451512 | CONTROL_ONLY | 未测/未裁决 |
| Vote | 1 | V-five | 31.620409 | 24.807534 | FAIL_CLOSED | 未测/未裁决 |
| ABC | 2 | A | 30.161416 | 22.313012 | REFERENCE | 1.000000×A（实测） |
| ABC | 2 | B | 30.396373 | 22.457419 | CONTROL_NO_INDEPENDENT_PASS_CLAIM | 1.122644×A（实测） |
| ABC | 2 | C | 32.094800 | 24.912004 | PASS_C_MINUS_A_SEVEN | 4.665624×A（实测） |
| Vote | 2 | V-whole | 30.243257 | 22.315977 | CONTROL_ONLY | 未测/未裁决 |
| Vote | 2 | V-five | 31.625461 | 24.871626 | FAIL_CLOSED | 未测/未裁决 |
| Soft-NMS | 0 | E | 30.267629 | 22.551132 | CONTROL_ONLY | 未测/未裁决 |
| Soft-NMS | 0 | D | 31.842120 | 24.972549 | FAIL_CLOSED | 未测/未裁决 |
| Soft-NMS | 1 | E | 30.314755 | 22.718971 | CONTROL_ONLY | 未测/未裁决 |
| Soft-NMS | 1 | D | 31.946101 | 24.841415 | FAIL_CLOSED | 未测/未裁决 |
| Soft-NMS | 2 | E | 30.575795 | 22.600047 | CONTROL_ONLY | 未测/未裁决 |
| Soft-NMS | 2 | D | 32.195877 | 24.928955 | FAIL_CLOSED | 未测/未裁决 |
| FT | 0 | T-control/A | 29.337512 | 21.705717 | CONTROL_ONLY | 协议成本PASS（实测） |
| FT | 0 | T-control/C | 31.163462 | 24.258866 | CONTROL_ONLY | 协议成本PASS（实测） |
| FT | 0 | T-slice/A | 29.571287 | 21.886792 | FAIL_CLOSED | 协议成本PASS（实测） |
| FT | 0 | T-slice/C | 31.622006 | 24.593615 | FAIL_CLOSED | 协议成本PASS（实测） |

A是参考而非新增方法PASS；B未按C的七项政策独立通过，不追认。原C逐seed质量及≤6成本通过。Soft-NMS42项7FAIL；框投票42项21FAIL；固定微调全局FAIL、成本PASS。未测速不写成本FAIL或PASS。本轮新C-center/C-flip均未GPU/评分，质量和成本均未评价。

源码：原seed0仅原manifest身份（2c9c37...），不伪造原启动commit；seed1/2实际release bcbda80b...及eee776...，与后续观察器/文档head区别。Soft-NMS c4a8f124...、FT57ae4d19...、投票f563dfb9...；完整SHA见JSON。

报告索引：
- ABC: r18-sahi-seed12-20261016-001/protocol-resume-20261017-001/terminal-review-20261018-001/summary/REPORT.md
- Soft-NMS: sahi-softnms-fixed-20261017-001/revision-007/REPORT.md
- FT: r18-slice-finetune-20261018-001/execution-001/terminal/REPORT.md
- Vote: sahi-box-vote-fixed-20261018-001/REPORT.md

来源口径：ABC的source字段指本次引用的认证推理缓存/64图测速release，不冒充旧seed0 A最初点值评分的源码commit；旧A点值由原比较记录与原报告绑定复用。具体原评价器/输入身份须沿原报告追溯，不能把新汇总器commit当成实验运行commit。
