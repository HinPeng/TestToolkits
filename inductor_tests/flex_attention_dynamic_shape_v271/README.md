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

# 运行 score_mod 动态 Shape 用例（A/B/C/D）
python run_flex_attention_tests.py score_mod_a
python run_flex_attention_tests.py --cases score_mod_a,score_mod_b,score_mod_c,score_mod_d
```

`score_mod_a` 到 `score_mod_d` 是显式注册的 special case，分别对应
`test_score_mod_a_batch_dynamic.py` 到 `test_score_mod_d_b_q_kv_joint_dynamic.py`；
它们也会出现在 `--list` 输出中，并参与不带选择参数时的默认运行。

每次运行会在 `flex_attention_test_runs/YYYYmmdd_HHMMSS/` 下生成批次目录：

```text
run_metadata.json       # 环境、选择和时间信息
runner.log              # 运行器日志
results.json            # 每个字母的机器可读结果
REPORT.md               # Markdown 报告
REPORT.json             # JSON 报告
a/run.log               # a 用例完整输出
a/torch_compile_debug/  # 最终调试树：fx_graph、output_code.py 等
```

Inductor/Triton 的中间编译 cache 会放在系统临时目录，单个用例结束后自动清理，不会写入 `a/` 等用例目录。

默认每个字母最多运行 3600 秒。可以使用 `--timeout`、`--stop-on-failure` 或 `--output-root` 调整行为。

报告可以在运行结束后单独重新生成：

```bash
python dump_flex_attention_report.py --run-dir flex_attention_test_runs/20260825_120000
```

运行器和报告脚本只依赖 Python 标准库；真正执行测试时仍需要当前环境中的 `pytest`、`torch` 和 `torch_npu`。
