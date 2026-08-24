# torch_npu 2.7.1 Dynamic Shape FlexAttention 测试包

这个目录可以整体复制到其他 NPU 服务器。它针对已安装的 `torch_npu 2.7.1` wheel：先把 patch 应用到当前 Python 环境的 `site-packages/torch_npu`，再逐用例运行动态 shape FlexAttention，并保留编译产物和报告。

## 快速开始

在目标服务器激活已经安装 `torch==2.7.1`、`torch_npu==2.7.1` 的环境后执行：

```bash
cd flex_attention_dynamic_shape_v271

# 可选：只检查 patch 是否能应用，不修改 site-packages
python apply_torch_npu_patch.py --dry-run

# 应用 patch；需要当前环境对 site-packages 有写权限
python apply_torch_npu_patch.py

# 默认运行主验收 + patch 契约 + 动态 shape 社区用例
python run_flex_attention_tests.py --suite all

# 重新生成某一批次报告
python dump_flex_attention_report.py --run-dir flex_attention_test_runs/<批次目录>
```

如果 `torch_npu` 不能被当前 Python 自动发现，可以显式指定其所在目录：

```bash
python apply_torch_npu_patch.py --package-root /path/to/site-packages
```

`--package-root` 可以是 `site-packages`，也可以直接是 `site-packages/torch_npu`。应用器只提取原始 git patch 中的 `torch_npu/` 部分；patch 里附带的源码测试文件不会写入 site-packages，因为本目录已经提供了可直接运行的测试文件。

## 用例

`S00`–`C07` 是动态 shape exact-capacity 主验收：覆盖 `T=0/1/C`、Q/KV 动态变化、非 128 对齐尾块、非连续 metadata、forward/backward、非默认 stream 以及生成 code 顺序。

`P01`–`P16` 是 patch 随附的源码/runtime 契约用例。它们从已安装的 `torch_npu` 定位源码，检查 DynamicScalar、offset/mapping、旧 metadata 清理、backward 任务数和动态 forward/backward 复用。

`M01`–`M23` 来自 TritonAutomation 的 NPU 适配社区文件 `community_tests/test_flex_attention.py`，只选择动态 shape 相关节点。完整筛选依据见 [COMMUNITY_DYNAMIC_ANALYSIS.md](COMMUNITY_DYNAMIC_ANALYSIS.md)。单独运行这组用例：

```bash
python run_flex_attention_tests.py --suite community
```

该套件通过 `python -m pytest` 启动，因此目标环境需要安装 `pytest` 以及 PyTorch 内部测试依赖（通常 TritonAutomation 测试环境已经具备）。

先看用例列表：

```bash
python run_flex_attention_tests.py --list
```

只跑一个用例：

```bash
python run_flex_attention_tests.py --case C01_capacity_reuse
python run_flex_attention_tests.py \
  --case M01_builtin_score_mods_dynamic_float16_score_mask_mod0
```

不同服务器的 NPU 编译时间可能差异较大，运行器对每个用例单独设置超时；`C05` 和社区 backward 用例默认 600 秒。失败或超时只终止当前子进程，不删除该用例的 `run.log`、`torch_compile_debug/`、`output_code.py` 和 cache 目录。

## 产物与报告

每次运行产生：

```text
flex_attention_test_runs/<run_id>/
├── run_metadata.json
├── results.json
├── REPORT.md
├── REPORT.json
└── <case>/
    ├── run.log
    ├── output_code.py              # 若该用例触发编译
    ├── output_code_1.py            # backward/多图时可能存在
    └── torch_compile_debug/        # 完整编译 debug 树
```

报告脚本支持从复制回来的 run 目录重新生成 Markdown 和 JSON：

```bash
python dump_flex_attention_report.py \
  --run-dir flex_attention_test_runs/20260819_120000 \
  --output /tmp/flex_attention_report.md
```

返回码为 `0` 表示没有 FAIL/TIMEOUT；有 FAIL/TIMEOUT 时返回 `1`，便于 CI 或批量服务器脚本判断。无 NPU 时 patch unittest 可能显示为 `SKIP`，主验收和 NPU 社区用例会失败或跳过，避免把未实际执行的验收误报为全通过。

## Patch 注意事项

- patch 基线是 `torch_npu 2.7.1`，应用器默认拒绝其他主版本；确认源码兼容后可显式使用 `--allow-version-mismatch`。
- 应用前会先做完整 dry-run，预检失败时不修改已安装文件。
- 已应用时会输出 `status=ALREADY_APPLIED`，不会重复修改。
- 如需撤销，可在确认目标是本工具包 patch 后执行 `python apply_torch_npu_patch.py --reverse`。
- patch 会修改已安装 wheel 的 Python 源文件；重新安装/升级 torch_npu 后需要重新应用。
