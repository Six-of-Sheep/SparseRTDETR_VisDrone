# 原C固定第六视图：准备交付

**准备CPU PASS；GPU/训练/科学评分均未执行。** 原C保留，旧关闭配方不重开。完整计划见PLAN.md/PLAN.json，现有实验主表见master/MASTER.md、MASTER.csv、MASTER.json（50条原端点，无旧AP重算）。

唯一方案：C-center主候选、C-flip同六视图控制，原C五视图+view5。中心ceil(5H/9)×ceil(5W/9)、居中floor、半开；flip原图RGB先翻再认证resize，连续坐标x1=W−x2'、x2=W−x1'，不clip。保原FP32每视图600、score/view/query/category排序、原hard NMS IoU>.5、先预算后原raw-ignore且不补位。新增位置/方向/上下文，不能称新增原生细节。

## 质量和阶段规则

先seed0固定64图A/C/center/flip真实工程和端到端成本（不算64 AP）；各两暖机、batch4/FP32，任一候选/A>6不放行。然后seed0固定548两方案两预算原双评价器。center−C主AP≥+.20pp、APS≥+.30、AP75-small≥0、APM/APL≥−.20；次AP/APS≥−.20；center−A原七项原门槛保持。

center−flip主AP及APS必须均严格>0（未舍入）；否则只写同视图数控制未超过，不能宣称中心位置额外价值、不能将flip升级。14质量+控制规则+成本同时通过才扩seed1/2。seed0过后seed1和seed2固定完成（科学失败不省略另一个seed），异常STOP不自动重试。所有比较/指标/预算和实际时间都报告，不要求先证明唯一根因。

## 已完成CPU证据

- 9项新增核心/几何/决策测试、21项原切片/NMS、11项管理测试PASS；317文件历史源导入核对PASS（仅管理，不追认科学）。
- 实际部署准备CLI→CPU子进程：奇偶图像及中心半开裁剪、flip像素、原5视图像素逐张精确；原FP32解码/越界、view5身份/同分规则/原C关闭精确。原NMS生成1200幸存者后，600/300预算先截断再原ignore，结果0，不用低分非ignore候选补位。正常CLOSED、预期worker异常、重复父入口及worker拒绝且无文件改动。2.852330673s PASS，CUDA未初始化，detector构造/forward/AP=0。
- 每seed首个已有raw batch（共3批12图）通过原严格SHA/来源/成员检查，新增关闭时两预算原C序列化输出精确一致；1.932622999s PASS。**不是重新跑411批**，原完整认证由旧严格源证书及一致NMS字节绑定复用；正式新执行仍须新鲜准入。
- 第一次CPU代码72d65eab6592e5368df05eaccfafa8b167a75f1c；最终准备代码9778b3506f43a3e7dba24ba8f9f317fbd1b4ead7只增加独立样本检查器，七个已测入口/模块/config hash未变，SOURCE_COMPATIBILITY记录复用范围。source359文件，原slice_core/infer_seed不改。合成与真实缓存检查两者范围分列。

## forward与成本预算（本轮0）

每seed64工程/成本：A18+C90+center108+flip108=324次batch4 forward。全548复用原5视图raw，只新增view5：每方案137批+2暖机=139，两方案278。总602/seed，最多三seed1806（2408/seed图像视图，包括暖机）。固定64与548批成员/次序不同，不未经认证混用64输出；每个计分batch新增视图一次。禁止从已NMS输出反推候选。

实测旧C/A约4.6725/4.7101/4.6656；旧seed0 A2.335102s、C10.910771s。6/5比例只作工作量估算，不是实测成本，六视图NMS/预处理可能超6。初步每seedGPU阶段5–15min+CPU合并/4评分5–12min，全部三seed30–81min另计准备/网络；不是截止。无当前新速度或显存值。GPU计时必须包含所有6视图真实forward，不能把缓存复用用于少算成本。

## 交付边界

当前只有准备、预处理/解码/合并和CPU检查接口；没有GPU/科学生产run分支。后续仍需独立生产资源/supervisor/真实恢复/两预算评分/导出入口、所受影响CPU验收和固定64GPU验收，不能把准备PASS当完整GPU链已验收。旧infer_seed.gpu/反序列化对seed0/view5的限制已在计划中标出，不能直接套旧入口。

必要CPU文件共8份155784B已D盘核SHA，全部4个本任务CPU PID退出。原始图像/大cache/checkpoint和CPU全现场仅服务器。本轮远端CPU阶段共4.784953672s；本地测试及未全计的开发/网络单列COST.json。无后台接续、无持续本机轮询、无holdout、无main合并。PR和CI状态见PULL_REQUEST/CI_READBACK，工程与科学未执行分别记录。
