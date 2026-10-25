# H2复现材料与版本入口

2026-10-25。**方法代码、精确成员与初始化身份可追溯；完整跨机器生产复现尚未执行。** 本次补齐轻量可逆成员元数据和使用说明，不将LAB合同复制为公共启动命令。

## 1 版本分层

|用途|固定入口|边界|
|---|---|---|
|实际H2科学实现与正式入口|[e50bedbe8e41bdfd3c47d09f7076d9644e438e26](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/e50bedbe8e41bdfd3c47d09f7076d9644e438e26)，PR21|独立冻结release运行；此文稿分支不包含该分支的全部代码，不能在文稿检出里直接寻找H2启动入口|
|模型/训练/状态|该commit下`src/sparse_rtdetr/research/local_detail_model.py`、`local_detail_engine.py`、`local_detail_checkpoint.py`|公式、采样复用、监督与保存恢复对应这里；不是仅凭论文伪代码重建|
|有效科学配置|同commit的`configs/research/r18_h2_nud_v1.json`与共同协议`Config.binding`|配置中的早期授权字段不是本轮正式授权，生命周期查正式回执|
|官方来源|共同模型声明官方commit `1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`，vendor已固定|共同协议适配，不称原封不动作者配方|
|论文与成员工具|当前`codex/r18-h2-paper-preparation`分支/PR|独立文稿交付；正式结果待原链，不改运行源码|

原H2源码manifest SHA为`c2200ead1a3a45620124b6c1fece499d64515c4cedca538e0231458da9728d07`，来源是既有`history-adapter-003`安装/启动回执，本轮不重新认证远端release。科学源码字节索引见[SOURCE_INDEX.json](SOURCE_INDEX.json)。

## 2 精确数据成员：本次已补齐

原train_core JSON为2,955,809字节，超过仓库单文件2MiB管理上限。这里以保持原顺序、全部16字段及类型的TSV保存，使用标准库可恢复原JSON字节，不重新选择成员。

|材料|成员|TSV字节数|恢复后的原JSON SHA-256|
|---|---:|---:|---|
|[train_core.tsv](splits/train_core.tsv)|4,869|1,617,048|`df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d`|
|[development.tsv](splits/development.tsv)|548|178,713|`5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd`|

TSV含图像/标注相对路径、大小、SHA、原ID、尺寸、序列键及已有行计数；没有图片、框坐标、类别标注正文、私人绝对路径或1602成员。`sequence_key`必须作为字符串保留前导零。字段类型和TSV自身SHA见[SPLIT_INDEX.json](splits/SPLIT_INDEX.json)。文件名与成员规则不构成数据授权，实际数据仍须从[VisDrone官方入口](https://github.com/VisDrone/VisDrone-Dataset)按原条件取得。

从本分支根目录，只读校验：

```bash
python -B tools/restore_h2_split_manifests.py
```

需要恢复原JSON时，指定一个**不存在、父目录已存在**的输出目录：

```bash
python -B tools/restore_h2_split_manifests.py --output-dir ../h2-member-manifests
```

工具先在内存验证两份材料，再独占创建目录，写出`train_core_manifest.json`、`development_manifest.json`。它不加载模型、不打开原图/标注、不联系网络；已有输出目录拒绝覆盖。恢复成功只证明成员metadata字节一致，不证明下载的数据实物匹配。生产reader之前仍需按清单核实际图像/标注大小与SHA。

原train/val来源、548反复开发、24个共享文件名前缀序列键、场景关联未知及1602独立资格BLOCKED全部保留。公开成员提高可复现性，不使原划分变为独立确认集。

## 3 初始化与环境

ImageNet初始化由固定vendor的`src/nn/backbone/presnet.py`登记[作者发布入口](https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth)，受验文件为44,878,642字节、SHA `911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720`。此处仅核原源码与本地绑定，未重新下载或核当前URL实物。

生产回执绑定torch2.4.1、torchvision0.19.1、CUDA build12.4、cuDNN90100。这四项不等于完整环境锁。系统部署安装器Python3.10.12也不能替代训练venv的完整Python/pip/驱动记录。

仓库`requirements-confirmation-ci.txt`是torch2.8.0+cpu/torchvision0.23.0+cpu的CPU工程环境，`requirements-local-detail-cpu.txt`补充CPU模型检查依赖。它们不能安装到当前生产环境替代冻结依赖。旧`environment/README.md`描述P3，不是H2安装指南；本次不追溯改写该旧说明。

仍需在未来复现授权中补齐完整可携环境记录/镜像、操作系统与驱动/编译依赖、实际GPU能力及相应资源合同；不能宣称本次已经实现一键跨机器复现。

## 4 历史N与新机器入口

历史N权重SHA为`be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83`。这提供身份，不能证明存在公共下载或本轮已重新读取实物。外部完整复现需要另行决定发布认证权重/侧车，或授权从头训练N；两者证据角色不同。原STOP和缺CLOSED不改写。

当前正式CLI绑定LAB数据根、旧N、短验收、存储、设备及唯一attempt。禁止直接重放旧launch或改旧合同中的路径/UUID绕过准入。新机器需要独立源码导出、输入物理核验、环境和资源绑定、必要CPU/GPU验收及新的执行授权。

|复现层级|当前可交付内容|尚缺内容|
|---|---|---|
|公式/源码与成员核对|8式、固定科学commit、成员可逆恢复、原初始化SHA|无新增GPU需求；不能据此声称效果重现|
|重新训练U/D|完整科学代码与原日程入口|数据实物、便携生产环境及新合同验收/授权|
|严格复现历史D−N比较|原N绑定和复用边界|公共N权重/侧车或明确另训N的替代协议|
|检测结论与论文|原门槛、终态填表规则|H2本轮终态、随后条件验证及贡献审查|

## 5 本次检查与轻量交付

成员恢复工具纳入Python3.10/3.12仓库CI，实际读取这两份TSV并核原JSON SHA，无跳过路径、无原图/标注正文或模型访问。此次也检查真实写出字节、拒绝既有输出目录、损坏TSV及矛盾类型声明；具体执行记录在同目录[准备索引](PREPARATION_INDEX.json)登记，CI状态以PR精确head为准。

只发布源码、文稿、元数据及必要索引；完整checkpoint、预测、图像、标注及全量审计仍不入Git。原Git TLS失败作为历史留存；本次正常同步不修改TLS/凭据、不改写历史、不合并main。
