# 第六视图 execution-002：资源适配通过，评价入口工程STOP

最小资源字段适配已完成并通过验收。新attempt已经执行seed0全部602次batch4 forward，检测器更新0；在第一组评价构建GT输入时发生新的工程异常，按授权保持STOP，没有修复或重试。质量NOT_EVALUATED，seed0固定64图成本PASS，seed1/2未启动。

## 修复与验收

- 显式把新合同storage_identity适配为旧prepare_cuda/live_admission需要的storage.identity；缺失/冲突拒绝，输入不变，不修改旧保护源码。
- 实际新生产合同通过28项真实pre-CUDA路径检查，覆盖三seed正常映射、同值双字段、顶层/嵌套缺字段、字段冲突、存储身份不符、boot/时效、存储/RAM/GPU资源不足。正常案例在被拦截的torch.cuda.init边界结束；CUDA未初始化。
- 4项新增单元检查、9项原第六视图、21项原切片、11项管理检查通过。真实最终CPU进程闭环（七阶段、四组合成双评价器、异常/重复/闭合/导出）重新通过；三seed CPU模型info一致。
- c81d6d0485b0fbbc7f2f634dfaef2f1766fad2a5 / 370文件；manifest SHA c5b4a3065a28221b278e7ab4a6a14befc1f992aecd58ff69c083e12e3d50c5e3。合同文件SHA 4dadff4d6b1c2d0e35a5b68324257631a5517fc904692489fb36d5279e25bc25；内部canonical digest f0fdecade46d2908e2156a9f9e800c51250048e74e25c71245642eafed3e0da8。不同用途不可混用。

## 本次实际完成

| 阶段 | 状态 | batch4 forward |
|---|---|---:|
| seed0 accept-A | PASS，严格完整恢复、状态/RNG精确不变 | 18 |
| seed0 accept-C | PASS，原C64输出精确一致 | 90 |
| seed0 accept-center | PASS，原五视图输出/像素精确一致 | 108 |
| seed0 accept-flip | PASS，原五视图输出/像素精确一致 | 108 |
| full-center / full-flip | 各548图完成，状态/RNG不变 | 139 + 139 |
| 原C重建与新raw准入 | 137批原C两档精确，新增raw身份/SHA通过 | 0 |
| 四组科学评分 | 第一组GT输入构造异常，0/4完成 | 0 |
| seed1/2 | 未执行 | 0 |

计数含冻结暖机。原A固定64端到端2.350641763秒，C11.491431075秒，center13.434924678秒，flip13.531004960秒。center/A=5.715428395，flip/A=5.756302460，均低于6；不是科学质量PASS。六个GPU阶段峰值allocated均1070716928字节、reserved均1616904192字节。

## 新STOP的已确认原因

`parent_context`调用原`old.context`，其返回`specs, rows, spec`的第三项是数据配置字典。新入口将该值命名为parsed，直接交给score_budget。primary_inputs遍历该字典得到字符串，truth访问parsed["rows"]时抛出`TypeError: string indices must be integers`。

失败发生在官方评价器实际评分前；无metrics/COMPARISON输出。不是GT内容已证明损坏，也不是资源修复失败。此前合成评分路径直接生成正确结构的parsed列表，未覆盖真实parent_context到GT解析路径；本次资源专项验收也没有覆盖该独立接口。运行源码不改，科学评分不重放。

## 现场、退出与成本

2026-10-18T19:57:30.226045+08:00逐PID/start核对15个自有链/监督器/worker均退出，NVML无compute进程。七阶段71份runtime/归档文件SHA一致；评价worker失败后监督器执行既定清理，remaining_live/unknown_pids/cleanup_errors均空。

274份新增全量raw（18,998,064字节）已逐份核SHA；四份已冻结预测文件建立SHA索引，均留服务器。本机只回传75份必要材料、478,632字节，核SHA通过。索引不是独立大文件备份。

生产链墙钟243.518625347秒；新记录不重叠墙钟下界286.270036400秒，与旧成本合计下界369.230860247秒。并行CPU检查单列，不重复累计；开发/传输/审计等未全计，不称GPU忙时。旧STOP和成本完整保留。

本地兼容性检查首次漏列既有旧STOP文档/索引差异，断言失败后已逐路径核对（PREPARATION_NOTE.json）；发生于生产启动前，不是科学执行重试。

## 后续边界

保持STOP。本次未修复GT接线或另开attempt。后续若获授权，应只修复数据配置到认证parsed_gt的明确转换、补真实548GT输入/顺序/计数/双评价器准入检查，并认证复用本次274份raw及四份预测；原则上无需重复seed0的602次forward。复用链和评分新attempt仍须独立验收，不因缓存存在自动启动。

原14项质量、同视图控制与成本门槛不改；seed0尚无质量裁决，seed1/2不得启动。PR #8保持未合并；代码/CI状态与科学结果分别记录。大型材料保留服务器，无Windows持续轮询。
