# 日常开发与版本管理

## 主线与短期分支

main是经工程检查的集成入口。一个开发任务使用一个短期分支；修复继续追加提交，不为每次修改、每个seed或切换助手重复开分支。同源码多seed共用commit，以配置区分。

1. 读仓库入口和可用的共享PROJECT/CURRENT；核对Git身份、状态、差异，保留他人工作。
2. git fetch origin后，新开发通常从包含所需认证代码的origin/main开始，分支使用codex/<简短任务名>；历史复现明确从指定commit开始。
3. 用同库git worktree隔离任务，统一放development/worktrees，不复制独立仓库。
4. 修改后执行相关检查，git add -- <明确文件>，运行管理脚本check --staged，再commit/push。
5. 通过PR整合回main，按当前授权合并。未获合并授权时交付可审查PR，不能把任务分支长期当成没人维护的主线。
6. 已合并的普通分支确认无独有材料后可清理；历史科研分支先核对commit、工作区和实验依赖，删除另行确认。

不强推、不改写科研历史，不对他人工作执行reset --hard或clean -fd。失败科学对照可作为可选配置保留，代码合并不代表效果PASS。

## 开发与运行分离

服务器日常主检出和Workspace/development/worktrees负责开发；Workspace/experiments下冻结release负责运行。禁止在运行目录pull、切分支或改源码。

新实验的代码与轻量配置先提交，再生成代码身份回执：

```bash
python -B tools/repo_manage.py freeze --experiment example-seed0 --config configs/my-experiment.json --output ../run-metadata/example-seed0/source.json
git archive --format=tar --output=../run-metadata/example-seed0/source.tar <回执的完整commit>
```

以上为占位示例，使用实际已审核配置替换。清单只描述代码身份，不授予启动权限；环境、数据、运行契约和验收仍按实验规则。发布解包字节须与清单核对，记录推送状态；关键系列可加annotated tag，不靠可移动分支名复现。

## 目录与检查

源码留src，共享工具留tools，配置留configs，轻量实验索引留manifests。当前认证根入口先保留原路径；目录重构单独开展。数据与结果不进Git。

check --staged读取真实暂存字节，拒绝产物、凭据形态、机器目录、超2MiB文件及未审查链接；它不是完整秘密审计。确需大源码文件时先明确审查例外。

CI运行管理、切片、汇总CPU测试与历史源码身份核验，并在Python 3.10/3.12中执行确认集入口、补充用途协议及零面积GT生产接口回归。后者通过`python -B tools/check_confirmation_ci.py`运行，任何skip均判失败；CPU-only依赖固定于`environment/requirements-confirmation-ci.txt`。测试只用仓库内的小型夹具，不读取实验室路径、封存材料或checkpoint，不评分完整数据集，不训练、不连接服务器，也不能替代GPU验收。

多视图蒸馏准备另用`environment/requirements-distill-cpu.txt`及`tools/check_distill_cpu.py --output <全新目录>`。它实际执行小画布原生R18的合成CPU forward/backward、保存恢复及准备入口拒绝检查，不运行科学训练或AP；它不提供GPU生产启动器。依赖和证据范围见`docs/research/r18_multiview_distill_20261020/CPU_REPORT.md`。

PR15生产准备002另有`r18_distill.py`最终生命周期及`tools/check_distill_production_cpu.py --output <全新目录>`。后者必须在Linux/WSL的CPU环境运行原POSIX评价器合同读取器，实际走通合成15阶段与两个原评价器、闭合/精简导出/退出；CI双Python版本无skip。CPU合成AP不代表科学评分。源码清单可用显式`--source`从实际工作树字节冻结，支持无Git元数据的独立release/WSL，不注入额外PYTHONPATH。真实GPU需另外精确source授权与新鲜材料/资源准入；详见PRODUCTION_EXECUTION.md。
