# 当前源码追溯导入

来自已回传认证的r18-sahi-seed12-20261016-001/release-002/source，共317文件、4,591,139bytes。
原manifest SHA256：
`39f9e06730a52d1c354d550fe3169e4a9fb1913ebe742f94d4c4f216d748391e`。

[原清单](../manifests/imports/r18-seed12-release-002.json)保留原字节，完整导入commit见[实验登记](../manifests/experiments.json)的source_import。
317项以Git blob大小/SHA核对；31个CRLF源文件采用精确路径-text属性。第三方许可证保留vendor/rtdetrv2_pytorch/LICENSE。

- 这是事后源码追溯登记，原运行启动时绑定release/manifest，不能倒写成启动于此commit。
- 仅核验源码身份，没有重跑模型验收；原模型数学、阈值、日程和切片规则未变。
- 当前seed1/2源码与seed0共同基线/切片版本不同，不能代替后者的精确运行身份。
- 外部数据、权重、环境和契约仍不可缺少；导入不等于可移植运行认证。
- 当前在服务器运行的release没有被修改。

verify-import核验历史commit；可加--working-tree验证当前源码仍等同导入。未来修改科学代码后，历史commit核验仍有效，当前比较应如实显示差异。

## 准备修复记录

旧.gitignore未锚定data/，首次暂存误排除vendor的数据加载源码，并产生不完整中间提交。
后续改为根目录/data/，补齐后重新核对317项暂存Git blob。中间提交保留，登记使用补齐后的完整commit。
