# Q2/Q3 DSL 快照迁移验证

将全部 32 个 profile 使用的 22 个测试定义（17 个 kernel、5 个 helper），
连同 6 个额外依赖 JIT helper，统一保存到 `../kernels.py`。
原始路径、revision、定义哈希和移除的 host 装饰器见 `../sources.json`。

- `snapshot.py --upstream /path/to/TritonAscendTest`：28 个定义逐字对比通过，
  比较内容包含函数正文及原始 JIT 装饰器。
- 搬移前后 `run_analysis.py` 的 32 个 profile / 132 项断言完全一致：
  104 项符合预期、28 项差异（其中 22 项为已有 G1），JIT 哈希、分类理由及行号、
  动态参数计划和 recipe 均相同。
- pytest 搬移前：114 passed、22 xfailed、36 failed。
- pytest 搬移后：115 passed、22 xfailed、36 failed。
  新增快照校验通过；36 个失败用例 ID 与基线完全一致，没有新增失败。
  其中 28 项为已有 G2 标记的 strict XPASS，6 项为既有分类预期差异，
  2 项为既有 vLLM target 策略对照失败。

默认读取快照；显式 `--kernel-root` 仍可读取原仓库用于对照。
本次为 CPU 源码分析验证，没有执行 NPU kernel 或修改分类预期。
