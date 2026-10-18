# 原C补充评价：限定材料定位，图像SHA通过／标注位置未闭合

状态：**STOP_ANNOTATION_LOCATION_UNRESOLVED**。本轮只处理当前材料位置，未重启访问历史追溯。原独立确认资格继续BLOCKED，三项人工说明均“用户无法确认”，历史使用、近重复及飞行/地点关联仍未知。旧release、STOP、门槛与累计成本全部保留。

## 映射与身份结论

对象manifest原SHA `1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4`（972181B）先核验后解析，仍1602图。原inventory SHA `aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7`也核验。未重建manifest或改变成员。

| 材料 | 历史路径 | 当前路径 | 证据与裁决 |
|---|---|---|---|
| 图像 | `/media/lyy/Data/JupyterLab/LiuZhiyang/yolo/datasets/VisDrone/VisDrone2019-DET-train/images` | `/media/lyy/Data/JupyterLab/LiuZhiyang/p3_data_roots/train_core/VisDrone2019-DET-train/images` | 1602/1602逐文件大小和SHA一致，共418043432B；CONTENT_IDENTICAL_SHA_PASS |
| 原始标注文件 | 同旧源根的`annotations` | 未定位 | 0/1602原文件完成内容核验，全部1602引用未闭合，不提供猜测路径或默认通过值 |
| 必要lineage | 原conversion_r3的`confirmatory_lineage.jsonl` | 原路径保留 | 50212648B，SHA `998f7063cbb1fba5cd6380e5f98ebdf34c8bdad6b8236e18faf6a360cfce7cf5`匹配原inventory；本轮仅核字节SHA，未用它重建标注或替代原raw-ignore输入 |

图像内容一致已证实；图像从旧根到当前根的完整迁移记录尚未建立，两者不混同。当前物理目录虽叫train_core，其内实际6471图，不能由目录名认定都参加训练。另以认证train_core manifest SHA `df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d`核对4869实际成员：该1602的精确图像SHA/ID无交集；不将此升级为开发历史或完全独立证明。

逐图“旧路径→当前路径→大小/SHA”完整表845726B留服务器`material-location-001/IMAGE_CONTENT_MAP.json`，SHA `858816774e26946d250346ecfa17c859b2908f31ee8907b69b05bba2af9ea3c4`。逐标注缺口表也仅服务器保留；D/Git只保留本报告、摘要及引用。

## 限定定位覆盖及局限

已检查09-30 locations/plan、10-15 PATH_MAP及重整/保留清单、registry数据相关登记、p3_data_roots与认证SSD/annotations根。训练SSD图像缓存及当前raw标注目录各4869份，仅核心成员；对应1602在这些扁平目录中未命中。最初检查较浅候选路径未命中；最后补查p3_data_roots至深度5才发现嵌套完整图像目录，已明确修正先前过早的STOP判断；随后按原SHA完成图像核验，没有按同名直接准入。

退役迁移总表FINAL_PATH_MAP.jsonl共40652条，未命中该1602图像/标注内容SHA或原训练源目录记录；归档content-addressed对象路径逐项存在性检查亦未命中。表内有两条同一materials包的D_PRIOR_VERIFIED引用，不是已确认的实验室标注原件；本轮未新增跨盘复制、解包或云端访问。

Workspace/archive只读目录搜索限深度12、上限18000目录，实际8566目录、未触上限；排除.git、环境、日志/runtime/逐窗raw/备份传输等目录，不跟随软链接。未发现可直接绑定的raw图像/标注目录或VisDrone/train命名压缩包。此范围不是全服务器、所有深层目录或所有压缩包内容的完备搜索，不能宣称原件全局不存在或丢失。

## 停止条件及工程状态

按用户“必要文件缺失或缺身份依据则STOP”的条件，本轮停在标注当前位置未闭合；尚不能建立完整显式当前路径配置。因此没有修复或放宽reader，没有生成新release、生产合同、令牌或attempt；不以空路径、目录猜测、重建清单或新旧路径软链绕过准入。没有移动、删除任何原件，也未重建1602原始标注。

原369文件运行commit `1fa3b3d8172cfc3c1611f625f08b64ca57f7f87e`、source manifest SHA `cc87a46ccd56eabbdb6b394e89ed107105576c46941411a8b5956d5e3343329d`及原CPU_GATE/STOP引用核对保持；未受影响CPU验收不重跑。本轮没有新路径生产验收PASS，不把manifest/图像身份PASS替代完整数据准入。

GPU forward0/6423、1602科学评分0/18组、训练更新0。固定64工程/成本未开始，质量NOT_EVALUATED、实际推理成本NOT_MEASURED。仅读取授权manifest、lineage字节以核SHA和1602图像字节以核SHA；未读取1602原始标注，未模型、推理、评分、近重复或flight/site检查，未恢复本机轮询。最后NVML compute观察为空。

图像/引用核验阶段测得9.581436858s，包含该次相关SHA检查，不重复计入内层；其他定位/开发/网络未全计，不写零。前次CPU111.680037004s、旧失败及所有历史成本继续保留，未因本次路径定位改写。

后续缺口是1602份原始标注文件的可核验当前位置/备份引用，或另行明确授权的恢复方案。本轮没有开展恢复，也没有扩大访问历史审查。即使未来材料齐备，也须新合同/身份和受影响路径验收，不重放旧入口；结果仍只能称披露限制的补充评价。
