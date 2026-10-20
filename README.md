# SparseRTDETR · VisDrone

这是 VisDrone 小目标检测项目的代码入口。当前研究使用**共同协议适配 RT-DETRv2-R18**，验证固定 SAHI 式切片推理；不是原封不动的官方配方，也尚未证明新算法贡献。

## 从这里开始

- [开发与 Git 工作流](docs/DEVELOPMENT.md)：修改、提交、合并和冻结。
- [实验导航](docs/EXPERIMENTS.md)与[机器可读登记](manifests/experiments.json)：科学状态和代码身份分别记录。
- [源码导入说明](docs/SOURCE_IMPORT.md)：317文件的出处及复现限制。
- [历史分支索引](docs/HISTORY_BRANCHES.md)：保留历史，不用旧分支充当默认入口。

## 已有结果

2026-10-19成果冻结：A为共同协议整图基线，B为整图＋常规NMS控制，C为固定整图＋四切片＋相同NMS的SAHI式实用适配。三个训练seed各用自己的原e120 EMA；两集合分别报告，不合并AP。

| 集合 | C−A AP均值±样本SD | C−A APS均值±样本SD | 状态 |
|---|---:|---:|---|
|548图development|+1.9540 ±0.0209 pp|+2.5969 ±0.1971 pp|各seed质量/固定64成本通过|
|1602图补充集合（历史使用未完全核实）|+3.3207 ±0.0270 pp|+4.4864 ±0.2271 pp|各seed质量/固定64成本通过；独立资格仍BLOCKED|

主top600。原主评价器、次top300、三seed绝对指标、所有增减、固定64成本与失败对照见[成果入口](docs/research/abc_results_20261019/README.md)、[冻结登记](manifests/frozen_abc_results_20261019.json)。C−B存在APL下降，C−A主预算AR-large三个seed均下降；不宣称所有指标改善。成本约4.3–4.8×只对应匹配development固定64，不是全量吞吐。补充集历史/数据关联未知、限定零GT输入扩展和未MATLAB限制保持。

实验扩展到此结束。Soft-NMS、固定微调、框投票和第六中心视图均按原门槛关闭；flip仅控制不升级，不自动调参或新训练。[历史集成审查](docs/research/abc_results_20261019/PR9_REVIEW.md)提出的CI缺口已追加覆盖，见[CPU工程收尾](docs/research/PR9_CI_CLOSEOUT_20261019.md)；合并前核对最终head的CI，PR #9未自动合并。

## 代码地图

| 位置 | 责任 |
|---|---|
| src/sparse_rtdetr/protocol_r18/ | 当前共同协议R18模型、数据、状态与训练适配 |
| src/sparse_rtdetr/localization/ | 当前仍依赖的运行、监控和完整性基础设施 |
| src/sparse_rtdetr/baseline/ | 历史基线与当前依赖，不按名字删除 |
| slice_core.py / infer_seed.py | 固定切片规则与推理接线 |
| new_entry.py / new_chain.py | 导入的复验入口，仍绑定外部契约 |
| configs/ / manifests/ | 配置、依赖身份和实验版本索引 |
| vendor/ | 第三方源码与许可证 |
| tools/repo_manage.py | 无模型、无GPU的Git管理检查 |

保持认证源码的原目录及字节；未来目录重构应单独提交并测试。

## 本地管理检查

管理工具只需要Git和Python 3.10+；切片测试另需NumPy。CI使用Python 3.10/3.12及environment/requirements-ci.txt，不要求PyTorch。模型环境另按原依赖绑定，不能将测试依赖安装命令用于覆盖训练环境。

```bash
python -B tools/repo_manage.py status
python -B tools/repo_manage.py check
python -B tools/repo_manage.py verify-import
python -B -m unittest discover -s tests -p test_repo_management.py
# 在独立的Python 3.10/3.12 CPU测试环境中安装：
python -m pip install -r environment/requirements-ci.txt
python -B test_slice_core.py
python -B test_seed_readout.py
```

训练还需要原数据、权重、匹配环境、新任务配置和既有资源/完整性验收。不要直接重放历史启动脚本。服务器现有作业始终使用原冻结release。

Git保存代码、配置、测试、方法文档和小型索引；checkpoint、预测、数据、日志、审计、环境留在独立工作区。索引不是大文件备份。
历史test划分曾在P1被观察，不能重新称为未见确认集。


## 当前受控探索：采样reference extent下限

[三臂设计](docs/research/r18_sampling_span_20261020/PLAN.md)及[生产准备](docs/research/r18_sampling_span_20261020/PRODUCTION_PREPARATION.md)：B0 / 原生8-2-2控制B1 / 候选P，共同ImageNet初始化、完整120轮、显式入口和有限采样观测。本轮只做CPU工程验收，GPU、正式训练及科学评分未执行；不宣称实际采样跨度有固定下限或质量改善。旧A/C与失败裁决不变。

[独立GPU工程验收入口](docs/research/r18_sampling_span_20261020/ACCEPTANCE_ONLY.md)使用单独合同，仅允许三臂合计9次更新／18次batch forward，闭合导出后退出；拒绝正式训练、科学评分和完整测速。CPU准备通过不授予GPU执行权限，旧全链入口不接受验收合同。
