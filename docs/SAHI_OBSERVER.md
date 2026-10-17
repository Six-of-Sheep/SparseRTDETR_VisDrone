# SAHI 接续链的只读 Windows 观察器

`tools/sahi_observer.py` 不启动、恢复或停止服务器实验。服务器独立执行训练、推理、评价及导出；Windows 观察器只读取状态，并在终态按已冻结 `SLIM_EXPORT.json` 的引用回传必要小文件。网络重连只重做只读请求，不重试 GPU 阶段。

## 旧观察器证据与限制

2026-10-17 07:47 +08:00 的本机定向检查确认：

- 旧任务 `r18-sahi-seed12-20261016-001` 的 `OBSERVER_LAUNCH_RECEIPT.json` 与 `OBSERVER_STARTED.json` 记录 PID 104376，于 2026-10-16 12:46:02 启动。
- `OBSERVER_STATE.json` 只有 5 次成功读取，最后成功时间为 12:54:11；同时间 `OBSERVER_LATEST.json` 记录 seed1/update805。
- `OBSERVER.stdout.log` 只有首次 `REMOTE_CHAIN_RUNNING`，stderr 为 0 bytes；没有该观察器的 STOP/COMPLETE 回执。当前已不存在该 PID 对应的观察器进程。
- Windows `LastBootUpTime` 为 2026-10-10 09:09:28，不能把观察器消失归因为本次 Windows 重启。缺少原进程 creation/start 及终止事件证据，退出的历史根因仍未知；不推断为系统睡眠、终端关闭或进程被杀。

旧实现的确定缺口是：网络错误只刷新 STATE，留下旧 LATEST 的 `RUNNING`，且没有查询时的新鲜度判定。进程被外部强制终止时无法保证产生退出回执，因此不能仅靠退出文件判断观察器仍活着。旧源码、快照及失败证据保持原样。

## 调用与新鲜度

先把已核 SHA 的新 `EXECUTION.json` 和 `LAUNCH_RECEIPT.json` 放入本地新 attempt。launch receipt 也可放在其 `returned-small/` 中。两者与远端字节绑定，execution 内部规范化摘要也须一致。

```powershell
python -B tools/sahi_observer.py status --execution D:/Lzy_CV/<attempt>/EXECUTION.json --local-root D:/Lzy_CV/<attempt> --project D:/Lzy_CV
python -B tools/sahi_observer.py watch --execution D:/Lzy_CV/<attempt>/EXECUTION.json --local-root D:/Lzy_CV/<attempt> --project D:/Lzy_CV --observer-id resume-001 --interval 120
```

`status` 每次都发起新的服务器读取，不读取旧 LATEST 来回答实时状态。成功结果包括 `checked_at`、`received_epoch`、`freshness=fresh`；网络不可达返回 `freshness=unknown`、空 snapshot 和非零退出码，不推断训练失败。磁盘快照只代表其接收时刻；调用 `cached_state` 时超过 180 秒标为 `stale`，即使年龄尚短也明确 `cached_only`，不能替代新查询。快照中的服务器采样时间不作为本机年龄的时钟来源。

watch 每 90–120 秒一次，只做有界末尾读取。每次请求包括网络失败都刷新自己的 LATEST/STATE；错误身份、SHA、游标倒退或缺少终态等证据异常停止本地观察器，不采取远端修复。每个新观察器创建独立 `observers/<observer-id>/`，拒绝重复身份和第二个跨进程 singleton；记录 PID 与操作系统创建标识，保留全部旧证据。

Windows 断网后，活着的观察器可继续只读重连。Windows 退出、断电或外部强制终止观察器后不能承诺自动回传；服务器链仍独立运行。后续状态请求必须重新读取服务器，不能把最后一次 LATEST 当作当前事实。主执行者可通过独立后台进程启动 watch，但它不能重新启动实验。

Windows SSH子进程使用`CREATE_NO_WINDOW`，避免隐藏观察器每次轮询时短暂打开终端；标准输出/错误捕获和超时保护保持。主机重启后先核对没有同任务观察器存活，再用新的observer-id恢复监控，保留旧身份和记录。监控工具可有单独的新commit；实验仍绑定其原commit和运行manifest，不为本机监控修复修改服务器冻结release。

## 精简回传与交接

先按当前终态记录的 SHA 锁定 SLIM_EXPORT，再只读取其精确引用与显式路径白名单交集。总预算保留 200 个 manifest 数据成员／32 MiB，单文件不超过 4 MiB；完整 checkpoint、预测、audits、全量日志字节禁止回传。已有同 SHA 的小文件直接复用，已有异字节材料拒绝覆盖。所有请求字节核验完成后才写新文件；终态回执独占创建。

科学 FAIL 仍可完成固定执行和回传；STOP 回传只报告已完成阶段，不宣称三 seed 结论。观察器的完成回执保存在 `observers/<observer-id>/`，数据在 attempt 的 `returned-small/`。CURRENT 在新读后前置带唯一 marker 的终态说明；替换前复核字节，发现并发变化则重新读并合并，8 次仍冲突则停止本地写入留证。HISTORY 只追加。不会使用观察器启动时的旧副本覆盖前台执行者的新记录。

## CPU 检查范围

```powershell
python -B -m unittest discover -s tests -p test_sahi_observer.py -v
```

19 项定向检查通过。真实入口在独立 CPU 进程执行，SSH 边界替换为执行原读取代码的另一个 CPU 子进程；OS/Git/GPU 保护没有被冒充为真实实验验收。覆盖新鲜查询、网络错误与恢复、PID 重用、独立身份和 singleton、阶段消失、科学 FAIL/STOP 交付、冻结 SHA、同 SHA 复用、禁止大文件、游标与尾记录、两个真实进程的并发记录保全，以及新接续 task 身份和全部新增导出路径。没有 SSH、模型、CUDA、训练或 GPU 调用。
