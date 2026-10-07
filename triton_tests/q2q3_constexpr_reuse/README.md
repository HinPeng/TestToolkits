# Q2/Q3 constexpr 参数分类测试

默认从本目录的 [`kernels.py`](kernels.py) 读取 Q2/Q3 算子 DSL 快照，
覆盖 `cases.py` 的全部 32 个 profile：22 个测试定义（17 个 kernel、5 个 helper），
以及 6 个额外依赖 JIT helper，共 28 个定义。
函数正文和 `@triton.jit` 装饰器保持原样；只移除 host wrapper、外部导入及
非 JIT 装饰器（autotune、heuristics、libentry 等）。快照用于源码分类分析，
没有复刻上游 host 调度配置。

[`sources.json`](sources.json) 记录原仓库 revision、原始文件路径/行号、文件和定义 SHA-256，
以及移除的装饰器；源码中的版权和迁移说明随快照保留。
默认运行不再依赖 `TritonAscendTest` 或 Q2/Q3 源码仓库。

测试调用当前 `triton-ascend/third_party/ascend/backend/reuse/analysis.py`，
检查每个原始 constexpr 的唯一类别、分类理由和动态参数计划。
预期依据 `docs/triton_constexpr_reuse_design.md` v0.3 §4 的用途规则及首版支持边界手工编写；
不会把分析器当前输出自动作为答案。

仅需 **Python 3.9+ 和 pytest**；独立报告命令只依赖 Python 标准库。
不需要安装 torch、torch_npu、Triton、CANN，也不需要 NPU。

## 运行

从包含 `TestToolkits/`、`triton-ascend/` 的工作区运行：

```bash
python3 -m pytest -q TestToolkits/triton_tests/q2q3_constexpr_reuse

# 严格显示当前实现与预期的所有差异，取消已知问题的 xfail 标记。
python3 -m pytest -q TestToolkits/triton_tests/q2q3_constexpr_reuse --runxfail

# 生成完整分类和理由报告；只要有差异，返回码就是 1。
python3 TestToolkits/triton_tests/q2q3_constexpr_reuse/run_analysis.py \
  --output /tmp/q2q3-constexpr-analysis
```

其他目录布局可用 `--triton-ascend-root` 指定分析器仓库。
如果需要直接分析上游源码，可显式传入 `--kernel-root`；以下选项同时适用于 pytest 和报告命令：

```bash
python3 -m pytest -q /path/to/TestToolkits/triton_tests/q2q3_constexpr_reuse \
  --triton-ascend-root /path/to/triton-ascend \
  --kernel-root /path/to/TritonAscendTest
```

也支持 `TRITON_ASCEND_ROOT`、`TRITON_KERNEL_ROOT`；后者设置时会选择上游源码。
分析器默认在当前目录和测试目录的祖先中查找同名仓库及 `Ascend/` 子目录。
缺少源码、入口或 constexpr 清单发生变化会报错，不会静默跳过。
默认分析前还会校验快照清单和定义哈希。报告的 `path`/`definition_line` 指向实际分析的
定义，`source_path` 保留上游路径，`kernel_source` 区分 `frozen` 与 `upstream`。

```bash
# 只校验本地快照，不需要 torch、Triton 或上游仓库。
python3 TestToolkits/triton_tests/q2q3_constexpr_reuse/snapshot.py

# 与 Q2/Q3 原仓库逐个比较 28 个 JIT 定义（包括 helper）。
python3 TestToolkits/triton_tests/q2q3_constexpr_reuse/snapshot.py \
  --upstream /path/to/TritonAscendTest
```

T 正向对照使用相邻 `vllm_constexpr_reuse/kernels.py` 的原始 `muls_add_kernel`；
特殊布局可设置 `TESTTOOLKITS_ROOT`。

## 覆盖范围

当前为 **22 个 JIT 定义（17 个入口、5 个 helper）、32 组输入配置、132 项逐参数断言**。
每个选定定义的全部原始 constexpr 都有明确预期；没有把局部 constexpr 或普通运行时参数混入清单。

| 来源 | 定义/算子族 | 检查重点 |
| --- | --- | --- |
| Q2 | store_lowrank | 真实 head 维度、token mask、七个地址 stride、批次 tile；FP32/int32 指针对照 |
| Q2 | store_paged_kv_cache | IS_DECODE/HAS_KV_LENS 两种取值、block_size 整除/取余、CHUNK_SIZE 与真实 head_dim |
| Q2 | DLLM attention bwd_d | stride 的 static_assert、真实归约 H、独立行 tile |
| Q2 | activation SiLU/SwiGLU | N 的 T01、M 的 grid-stride 行轴证明及完整候选的 M/N 组合复用；同文件激活 helper |
| Q2 | RMSNorm | 行 tile 与改变浮点累加分组的列 tile 区分 |
| Q2 | position_ids、conv compute_dh0 | 元组解包缺少摘要时整条使用链不能动态化；后者包含尚未遍历到的 static_range |
| Q2 | load/store_with_pred_1d | 控制是否有 mask 的静态布尔条件 |
| Q2 | RoPE 两个 helper | inverse 模式；CANN slice 无摘要时返回 Unknown |
| Q2 | SDPA bwd_d | `.to(HIGH_TYPE)` 的静态 dtype；未使用 LOW_TYPE；地址参数与真实归约维度 |
| Q3 | FLA _flat_offset | D 仅用于整数地址运算；IS_LINEAR 两种取值均不隐藏另一分支 |
| Q3 | add_position_embeddings | SCALE_JAGGED 两种取值、两个 tile、数据相关边界 |
| Q3 | jagged_to_dense | JAGGED_DIM、字符串/None 融合模式；add/mul 两个真实 helper 都被分析 |
| Q3 | SiLU | 一维 arange 不足以证明已有版本可用于不同 grid |
| Q3 | LayerNorm/L2Norm | 归约 tile、模式分支、缺少操作摘要时尚未分析到的用途 |
| Q3 | MiniMax top-k partial | topk/BLOCK_SIZE_K 的 static_assert、输出槽形状、元组赋值中断 |
| Q3 | pack/unpack、mean_pooling | 未支持元组赋值时保守返回 Unknown |

另有 32 项完整参数及局部变量重命名检查、helper 双分支检查、分析预算耗尽检查、
无执行/无导入副作用检查，以及原有 vLLM kernel 的 ScheduleReusable 正向和目标不支持对照。
SiLU/SwiGLU 的 M/N 均为 ScheduleReusable 正例：N 单独证明完整列遍历，M 单独证明完整行遍历；组合配方逐步检查同一旧二进制的完整 M/N 配置。原有 vLLM 调度对照继续保留。

## 当前结果与已知问题

本地 Python 3.9.6，分析器 `3105ad280023074be867d748b20b61f3d3eef950` **包含工作区未提交改动**，
规则版本 3。精确实现和源码哈希、绑定值、理由行号见
[JSON 报告](validation/host_analysis.json)，阅读版见 [Markdown 报告](validation/host_analysis.md)。

- 分类矩阵：**82 项符合预期，50 项不符**，后者不能计为识别正确。
- 默认 pytest：**122 passed、50 xfailed**。122 中包含 82 项符合预期的分类断言及 40 项补充检查。
- `--runxfail`：50 项差异应作为失败呈现；报告命令同样返回 1。
- 原有 vLLM `test_host.py`：44 passed。

| 问题 | 参数断言数（含多配置） | 预期 → 实际 | 依据 |
| --- | --- | --- | --- |
| G1_SHAPE_IS_NOT_STATIC_PROOF | 22 | Unknown → StaticRequired | `analysis.py` 把 `STATIC_ARANGE`/shape 静态用途直接当成最终类别；§4 要求 R3 继续检查调度证明，证明不足为 Unknown |
| G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC | 28 | RuntimeEligible → Unknown | `dsl/uses.py` 在二元运算中用 `"float" in kind` 匹配到 `ptr:float32`，错误给整数地址链添加 FLOAT_USE_NOT_SUPPORTED；相同地址链换成 int32 指针即获准 |

两类差异都使当前实现更保守；这些结果没有证明存在错误执行或数值错误。
测试保留设计预期，按**具体参数、实际类别和理由**登记已知问题。
只有该具体差异抛出的 `KnownClassificationGap` 会 xfail，其他异常正常失败；
修复后会 strict XPASS，提示维护者移除对应记录。
没有修改分析器来使结果通过。

## 纯分析入口的边界

`analysis_adapter.py` 只适配 `SourceJIT` 的源码、签名和 capture scope，以及语言对象身份。
语言导出及标准库 JIT helper 从同一 Triton checkout 读取；算子装饰器、host wrapper、
torch 导入、自动调优和设备查询均不执行。分类、依赖传播、操作约束、helper 遍历、
tile 匹配使用原始实现，没有在适配层重写分类规则，也没有给未知操作补造语义摘要。
指针只有 dtype 元数据，读取地址或执行 kernel/语言运算会立即报错。
注入的模块在分析结束和异常时恢复，可与现有测试一起收集。

这是当前 Python AST 分析器的分类回归，**不验证真实 JIT 前端、改写后的 ABI、
编译、设备加载、运行时 guard、数值结果或性能**。helper 直接输入只验证声明的标量类型上下文。
不会因为分类 RuntimeEligible 就宣称该输入已通过运行时 i32 范围检查或可在 NPU 上执行。

Unknown 用例明确区分“没有 tile 重用证明”和“使用链尚未分析完”。
例如 pack/pooling/conv 的元组赋值会中断当前分析，所以不把未读到的静态模式和
static_range 误记为已成功识别。规则扩展后应按源码重新审查这些预期。
理由行号保持分析器的 JIT 相对行号；跨 helper 理由没有函数身份时不伪造绝对源码位置。

源码直接来自两个算子仓库，不复制或修改上游算子。若源码变化，报告中的文件/JIT SHA-256
也会变化；更新分类预期需要人工检查对应源码用途。
