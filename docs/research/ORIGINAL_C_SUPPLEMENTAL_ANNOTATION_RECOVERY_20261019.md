# 原C补充评价：原始标注恢复通过，真实reader几何准入STOP

状态：**STOP_CPU_READER_UNSUPPORTED_NONPOSITIVE_GT**。原独立确认资格仍为BLOCKED_PENDING_ACCESS_HISTORY；历史使用、近重复及飞行/地点关联仍未知。三项人工说明仍为“用户无法确认”。旧报告、release、STOP、原始清单、期望SHA及累计成本保持。

## 已恢复的材料与证据链

两条D_PRIOR_VERIFIED引用来自两台历史实例，但指向同一个D盘materials.tar.gz，129497758B，重新核SHA `7daad853e1ec150e5293c9c6c98d88613433055f2ee72b889301b47da9fe6f83`通过。只读归档目录996项，无txt标注成员；未整包解压、未提取其他科研内容、未将大型材料复制到C盘。

原manifest `1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4`和lineage `998f7063cbb1fba5cd6380e5f98ebdf34c8bdad6b8236e18faf6a360cfce7cf5`先核SHA再读取。lineage保留八个原始字段字符串及每条物理行SHA；用原字段、原逗号和由SHA唯一确定的CRLF还原83478行。每行均匹配原行SHA，1602个完整文件再逐份匹配原manifest的大小和SHA，写入后读回核验再次通过，共1941456B。不是YOLO标签、浮点重新格式化或仅解析等价恢复，没有更改任何期望SHA。

历史annotations根 `/media/lyy/Data/JupyterLab/LiuZhiyang/yolo/datasets/VisDrone/VisDrone2019-DET-train/annotations` → 已认证原lineage → 服务器新目录 `SparseRTDETR_Workspace/experiments/localization-distribution/sahi-original-c-supplemental-20261018-001/annotation-recovery-001/restored-original-annotations`。逐文件来源/物理行/恢复路径/大小/SHA链1450520B，SHA `01b5e599c6cb22f8f710cf9466f4fdacbdcb042595a5678e88bf2e3543ceb873`仅服务器保存。没有覆盖、移动或删除旧材料，没有旧路径软链接。

图像复用此前1602/1602逐SHA认证及完整映射SHA `858816774e26946d250346ecfa17c859b2908f31ee8907b69b05bba2af9ea3c4`；本轮只重新核对这份认证映射，不重复目录搜索或全图像哈希。当前图像根仍为 `p3_data_roots/train_core/VisDrone2019-DET-train/images`。图像内容一致已证明，历史迁移过程记录仍不完整，不作独立性推断。

## 新路径版本与实际CPU结果

在原codex/sahi-independent-confirmation分支追加b45d94a和93ea756。显式CURRENT_PATH_BINDING绑定manifest、原图像映射、恢复标注映射及当前两个根；生产reader验证完整有序成员及这些引用，原合同/科学配置/模型/推理/双评价器/门槛/监督器字节未修改。旧369文件入口CPU证据保留，计划复用其不受影响部分；没有因恢复成功而宣称新整体CPU_GATE通过。

新冻结运行commit `93ea7561c80930c7d1e96cd56450b43419d5332e`，373文件，source manifest SHA `724f578f9bc4f5381a1fde2c231f9ba6d4e05ce5feadb0f6fc402717c6349386`，release-003。release-001/002及旧STOP保留。真实CPU命令见CPU-PATH-003-invocation.json；当前路径映射校验通过后，dataset在原有 `unsupported nonpositive GT` 检查失败，CPU_GATE未生成，后续拒绝检查及双评价器输入构造未执行。不能将旧合成闭环或Git CI替代本次真实reader PASS。

限定诊断仅解析已授权、逐SHA恢复的1602文件，不评分：83478原始行中valid_gt81318、category11_not_spatial184、spatial_ignore1974、nonpositive_area2。两行分别是：

- `9999985_00000_d_0000020.txt`第12物理行，category4/score1，框[611,158,615,158]，面积0；原manifest filtered_rows=1。
- `9999999_00590_d_0000267.txt`第89物理行，category0/score0，框[545,414,555,414]，面积0；原manifest filtered_rows=0。

两文件的SHA均等于冻结值；不是损坏或恢复差异。现有supplemental_data.dataset拒绝非正面积，confirmation_prepare.validate_parsed也要求每行正几何。原解析器虽分类nonpositive_area，但如何与本集合的双评价输入一致处理尚未认证。不能仅删掉一条检查、丢弃原始行、伪造有效框或改清单继续；本轮恢复/路径授权内未改这些语义。

## 执行与收尾边界

GPU forward **0/6423**，科学评价 **0/18组**，训练更新 **0**，固定64图工程/成本未开始；质量NOT_EVALUATED，成本NOT_MEASURED。只生成材料恢复/当前路径绑定及CPU上下文用途授权，不生成生产合同、启动令牌或生产attempt，不调用launcher、不后台接续、不恢复Windows轮询。最后只读GPU compute快照为空。

准备工具第一次用系统python读取运行模块缺NumPy，保留失败输出，改为认证解释器；本地一次PowerShell引号错误未执行远端；CPU证据复用比对补区分文档/索引变化。均非生产重试，未改变科学规则。所有现场保留。恢复0.726197701s、限定几何诊断0.875012476s、D包目录/SHA0.376691500s为各自计时，CPU失败外层/开发/网络未全计，不写零。原CPU111.680037004s、图像认证9.581436858s及全部旧失败成本不重计或抹去。

D盘必要回传7文件8820B逐SHA通过；完整标注、lineage、图像、逐成员恢复映射和checkpoint仅服务器保存，索引不是备份。新报告独立保存，不覆盖旧STOP报告。后续接续缺口已由“标注未定位”变为“两个原始零面积标注的reader/评价协议兼容性未认证”；本轮到此停止，不自动改协议或新启动。
