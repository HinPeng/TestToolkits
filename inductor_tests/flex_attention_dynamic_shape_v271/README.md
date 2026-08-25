# FlexAttention dynamic-shape test helpers

根目录的 `a` 到 `s` 用例可以按字母或范围运行。当前没有对应文件的字母会显示在 `--list` 的 `missing` 行中；使用范围时会自动跳过这些字母。

```bash
# 查看当前可运行的用例
python run_flex_attention_tests.py --list

# 运行一个用例
python run_flex_attention_tests.py a

# 运行多个用例；也可以写成 --cases a,c-f
python run_flex_attention_tests.py --cases a,c,f

# 运行 a-s 范围内所有实际存在的用例
python run_flex_attention_tests.py a-s
```

每次运行会在 `flex_attention_test_runs/YYYYmmdd_HHMMSS/` 下生成批次目录：

```text
run_metadata.json       # 环境、选择和时间信息
runner.log              # 运行器日志
results.json            # 每个字母的机器可读结果
REPORT.md               # Markdown 报告
REPORT.json             # JSON 报告
a/run.log               # a 用例完整输出
a/torch_compile_debug/  # 若测试生成
```

默认每个字母最多运行 3600 秒。可以使用 `--timeout`、`--stop-on-failure` 或 `--output-root` 调整行为。

报告可以在运行结束后单独重新生成：

```bash
python dump_flex_attention_report.py --run-dir flex_attention_test_runs/20260825_120000
```

运行器和报告脚本只依赖 Python 标准库；真正执行测试时仍需要当前环境中的 `pytest`、`torch` 和 `torch_npu`。
