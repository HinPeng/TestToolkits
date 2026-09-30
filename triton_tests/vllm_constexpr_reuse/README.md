# vLLM Ascend constexpr 复用算子测试集

用于验证 `docs/triton_constexpr_reuse_design.md` v0.2 中的普通 JIT/launch 算子行为。
基线为 vllm-ascend `18d0537d919c303952fbe6f3442349d0de42970d`，设计对应 TA `38945468f2` / 3.6.0。
测试集不实现分析器、重写器或复用分发器。

按用户指定范围覆盖 **19 个算子**。`kernels.py` 冻结了 **22 个原始 JIT 定义**：
这 19 个入口，加上保留的 ngram 对照，以及 `_compute_slot_mapping_request`、`bonus_renew` 两个 helper。
函数文本、装饰器、
constexpr 标注、参数顺序和计算体均与上游逐字一致；仅替换模块导入和移除 host wrapper，
因此不需要安装整个 vLLM 服务栈。Ascend 扩展按上游相同的 extension → tl 顺序解析。
来源、原始行号和每个函数的 SHA-256 见 `sources.json`，许可证见 `LICENSE`。
`probes.py` 中两个派生对照用例单独标注，不冒充生产 kernel。

## 运行

需要 Python 3.10+、pytest、NumPy，以及相互匹配的 torch、torch_npu、Triton-Ascend、CANN 和 NPU。
按所在设备已有方式配置 CANN 环境。本目录不安装或切换这些组件，不连接远程服务器。

```bash
cd TestToolkits/triton_tests/vllm_constexpr_reuse

# 无需 torch / Triton / NPU：检查冻结源码、参考结果、报告完整性。
python3 -m pytest -q test_host.py

# 可选：与本地指定的 vllm-ascend checkout 逐字比较。
python3 snapshot.py --upstream /path/to/vllm-ascend

# 设备数值回归：每个场景的关闭/开启各起一个新进程、使用各自空缓存。
python3 run_suite.py --mode both --device npu:0

# M1/M2：同时要求正例实际复用已经执行过的二进制。
python3 run_suite.py --strict-reuse \
  --select 'muls_add or slot_mapping or renamed_pointwise'

# 完整验收包含三种 split QKV kernel 的 eps、SwigluStep LIMIT 动态化；未实现时应失败。
python3 run_suite.py --strict-reuse

# 只运行拒绝/保守处理案例的数值与静态身份检查。
python3 run_suite.py \
  --select 'rms_dim or metadata or bincount or ngram or expand or rejection or external_bound'

# 新增算子可按测试名选择；不带 --select 默认执行全部 19 个算子的场景。
python3 run_suite.py --select 'split_qkv or triton_rope or swiglu or penalties or greedy or recovered or prepare_inputs'
```

默认汇总到 `results/<时间戳>/off.json`、`on.json`，每个场景的报告与 Triton 缓存
位于 `off/<序号>/`、`on/<序号>/`。每个子进程只执行一个场景，场景内的调用序列共享 JIT 与缓存。
可以传 `--output /path/to/new-directory`；目录必须不存在，防止混用历史结果。
`--mode off` / `--mode on` 可以单独运行，但只有 `both` 会逐项比较开关前后的输出字节。
全局开关在子进程启动前设置，不在同一个进程内切换。

也可直接执行 pytest：

```bash
TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=0 python3 -m pytest -q \
  test_operators.py test_extended_operators.py --require-npu
TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1 python3 -m pytest -q test_operators.py \
  -k test_renamed_pointwise_reuse --require-npu --verify-reuse --reuse-report=/tmp/reuse-on.json
```

不带 `--require-npu` 的普通 pytest 在无设备/依赖时跳过设备测试，方便本地收集。
正式运行器始终加 `--require-npu`，缺少设备会失败；零用例、跳过、缺少输出、
测试 teardown 失败均不能形成通过报告。
完整运行还根据实际 launch 记录校验 `catalog.py` 中的 19 个算子均已执行；
只有冻结定义、没有实际执行的算子不能算覆盖。指定 `--select` 时允许只验收所选子集。

## 用户清单与源码名称

`catalog.py` 将用户名称、上游真实符号和测试函数绑定；host 检查同时验证源码存在、测试引用及
直接 launch 与上游参数签名一致。函数原名不为对齐清单而修改。

| 分组 | 用户清单 / 实际 JIT 符号 | 主要测试函数 |
| --- | --- | --- |
| Vllm | `split_qkv_rmsnorm_rope_kernel` | `test_rope_eps_and_complete_versions` |
| Vllm | `split_qkv_rmsnorm_mrope_kernel` | `test_split_qkv_mrope` |
| Vllm | `split_qkv_rmsnorm_rope_simt_kernel` | `test_split_qkv_simt` |
| Vllm | `_triton_rope` | `test_triton_rope_layouts` |
| Vllm | `swiglu_quant_kernel` → `_swiglu_quant_kernel` | `test_swiglu_quant_modes` |
| Vllm | `swiglustep_kernel` → `_swiglustep_kernel` | `test_swiglustep_limit` |
| Vllm | `muls_add_kernel` | `test_muls_add_tile_reuse` |
| Vllm | `apply_all_penalties_kernel` | `test_apply_all_penalties` |
| Vllm | `token_bin_counts_and_mask_kernel` | `test_bincount_external_bound` |
| Vllm/MY | `build_local_metadata_triton` | `test_metadata_capacity` |
| Vllm/MY | `rejection_random_sample_kernel` | `test_rejection_modes_and_entropy` |
| Vllm/MY | `prepare_inputs_padded_kernel` | `test_prepare_inputs_padded` |
| Vllm/MY | `rejection_greedy_sample_spec_len_1_triton` | `test_greedy_spec_len_one` |
| Vllm/MY | `expand_kernel` | `test_expand_signature_and_integer_width` |
| Vllm/MY | `triton_rms_kernel` | `test_rms_dim_static` |
| Vllm/MY | `rejection_greedy_sample_triton` | `test_greedy_variable_lengths` |
| Vllm/MY | `sample_recovered_tokens_kernel` | `test_sample_recovered_tokens` |
| Vllm/MY | `_compute_slot_mapping_fused_groups_adaptive_kernel` | `test_slot_mapping_dynamic` 的 adaptive 参数组 |
| Vllm/MY | `_compute_slot_mapping_fused_groups_kernel` | `test_slot_mapping_dynamic` 的 fused 参数组 |

## 判定方式

- 正式运行器为每个场景使用独立进程和空磁盘缓存，场景内部连续执行多个请求，
  避免其他测试提前产生候选。仅重建 JIT 对象不足以隔离按源码索引的进程级版本表，
  因此直接 `pytest --verify-reuse` 也强制每个进程只选中一个场景。
  场景结束检查原 JIT 源码、参数名和 constexpr 列表未被修改。
- 调用只有普通 `kernel[grid](*args, **kwargs)`；每一步传的是**本次请求配置**。
  测试不会直接启动旧 CompiledKernel，也不会替 TA 修改旧 tile 对应的 n_blocks/grid。
- 独立 CPU 参考计算检查数值；整数、token、计数和可精确表示的逐元素结果零容差；
  RMS/RoPE、SwigluStep 和量化 scale 的 CPU 参考采用文件中显式容差。另检查输出尾部哨兵、部分覆盖区域、
  原位更新及应保持的输入。采样比较最终 token、bonus 和未写入槽位，不只比较 entropy。
- `run_suite.py --mode both` 对两进程的全部输出做 SHA-256 字节比较，浮点同样零容差，
  不把 CPU 参考容差作为复用后放宽精度的许可。输入在 CPU 确定性生成。
- 观测普通 JIT 返回的**实际 CompiledKernel** 的 `hash`、`src.constants`、`src.signature`。
  `--verify-reuse` 要求 D 参数离开 constants 并进入运行时 signature，同类型多值使用同一产物；
  T 正例首次请求另一 tile 时应使用已预热的产物。没有可观测返回值时失败，不猜测命中。
- 首次 T 回退之后，后台请求版本可能已经 READY，后续调用允许切回精确版本，
  因此不错误地要求所有后续 tile 调用永远具有同一 hash。
- `distinct_selected_binaries` 是执行中观察到的产物身份数，**不是编译次数**。
  `synchronized_wall_ms` 包含 JIT、初始化和同步，只用于排查；不是稳定性能/P99 基准。

数值模式在尚未实现复用的 TA 上也可能通过，只代表算子基线与开关透明性。
严格模式的正例会在仍逐值编译时失败，不能把“环境变量设成 1”视为功能已经生效。

## 用例矩阵

| 用例 | 请求序列与关键边界 | 对应设计与断言 |
| --- | --- | --- |
| muls_add tile | N=5000，G=1/3/7，1024↔2048，再次请求新 tile | §7：首次跨 tile 返回同一 hash；完整覆盖；grid callable 每次只求值一次 |
| muls_add 边界 | N=1/31/1023/1024/1025/4097 | 小于块长、整块与非整块；覆盖 n_blocks=1 特化场景，不强求不合法候选复用 |
| muls_add 部分覆盖 | 先完整覆盖，再请求 B=2048/n_blocks=1/N=5000 | 只写前 2048 元素；不得擅自扩大为完整计算；保留本次 B |
| muls_add 原位 | output 与 x 是同一 tensor | 精确别名数值回归；允许保守回退，不要求一定跨 tile |
| fused slot mapping | G=2，P=1/2，R=3→4→1→0→3；PAD=-1/-7/1/-13/i32 两端点 | §6/§8：NUM_REQS/PAD_ID 和局部 constexpr 派生链；R=0 仍初始化 padding；固定 tile/window 的多值同产物 |
| adaptive slot mapping | 同上，request_tokens=1/33/130，small tile=32 | U1 正向 helper 分析，覆盖 small/large 两处调用与 PAD_ID store；circular 负位置 |
| RMS | DIM=256→512→256，实际 stride=DIM+16 | S1：DIM 保留静态，真实列数、分母和行间 padding 正确 |
| ngram | min_n=2→3→2，max_n=4，max_seq_len=128/2048 | S2：static_range 边界不动态化；构造只有长度 2 匹配的序列；同时测 discard、无匹配、非法 sampled ID、跨块分支 |
| rejection random | 普通→synthetic→entropy→不同 SUB_BLOCK→普通 | S3/S4：模式常量保留；entropy 分支 SUB_BLOCK=32/64 不跨值；固定 uniform，覆盖接受、拒绝、bonus、greedy 跳过、draft=-1 |
| rejection 可选指针 | NO_DRAFT_PROBS/NO_ORI_TARGET_PROBS 四组合；非 synthetic rates=None | 静态分支与 None 指针签名保护，不引入本来不可达的非法 load |
| local metadata | (R,B)=(63,64)→(63,128)→(65,128)→(63,64)→(64,64)→(1,128) | 按用户要求，选中 BLOCK_NUM_REQS 必须等于本次请求值；即使旧容量足够也不得跨值复用；检查前缀和、长度、COMPUTE_START_POS 两模式 |
| expand | int32→int64→int32，int64 数据与替换值>2^40；含零长度请求 | S6：真实指针签名宽度匹配；不得沿用 int32 二进制或截断整数 |
| bincount | batch=2/seq=300，SEQ_BLOCK=128→256→128；rank=0/1；连续/跨步输入 | U2：保留本次 tile，total_blocks 按本次合法请求计算；结果计数可发现漏覆盖或重复执行 |
| RoPE eps | 固定 B/H=(2,32)，eps=1e-5→0.03125→1e-3 | M5：FP32 统计量加法的 constexpr 动态化；不同 eps 实际生效且同产物 |
| RoPE 完整版本 | (B,H)=(2,32)→(4,64)→(2,32)，真实 head 数固定 | §5.1：只能返回已经请求过的完整配置，禁止 (2,64)/(4,32) 拼接；不强求尚无证明的 RoPE T 优化 |
| MRoPE | eps=1e-5→0.03125；连续/交错 t/h/w；gate 开关；bias 与 partial/full RoPE | eps 的 ABI/同产物正例；模式、gate_size 和真实维度保持静态；检查 q/k/v/gate；5 tokens 分配到 3 个 program |
| split QKV SIMT | force_simt_only=True，5 tokens/2 program，行 tile=2→4→2；eps 变化 | 原生产 SIMT 编译选项；不整除尾部与完整 B/H 配置；eps 同产物；bias、partial/full 模式 |
| `_triton_rope` | Q/K heads=5/3，head tile=2/4，FP32/FP16；NEOX/交错；合并 cache/独立 cos-sin | 原位 q/k，真实 head 尾部 mask、非零位置映射、行间 padding、未旋转维度保持不变；模式值保留 |
| SwigluStep | FP16/BF16，7 rows/4 program；LIMIT=7→2→0.5→7 | silu 后 gate 单侧 clamp、up 双侧 clamp；LIMIT 的 ABI/同产物正例；真实列数保留 |
| Swiglu quant | group_list int32/int64；cumsum/count，含空 expert；SCALE 开/关；COL_BLOCK_SIZE=32/64，DTYPE_MAX=127/63 | 输出、scale、SCALE=False 时未写入 scale；分组模式与量化模式保持静态；FP32 固定输入验证 int8 截断；不强求未证明的 COL_BLOCK_SIZE 重用 |
| all penalties | 3 sequences/513 vocab，BLOCK_SIZE=128→256→128；连续/跨步 logits | 正/负/零 logit，prompt/output mask、频率计数、presence；原位结果与 padding，能发现重复惩罚 |
| prepare padded | (R,B)=(65,32)→(65,64)→(1,64)→(0,32)→(129,32) | 接受/拒绝计数、无 draft、空请求集、首元素前缀读取 mask、grid-stride 尾部；检查两个输出 |
| greedy spec_len=1 | 普通/synthetic 模式，BLOCK_SIZE=2/4，batch=5 | 匹配/失配、draft=-1、固定 uniform、bonus、int32/int64 输入、None 可选指针与尾部 |
| greedy 多长度 | draft lengths=3/0/1/2/3，greedy mask 有/无；普通/synthetic，BLOCK_SIZE=2/4 | 零 draft 追加 bonus、提前拒绝、非 greedy 不写入、helper `bonus_renew`、越界请求 mask |
| recovered tokens | 全词表70/缩减35；有/无 draft_probs；SUB_BLOCK/VOCAB_BLOCK_SIZE=32/64 | 固定 q，非连续全局 token ID、并列最大值取最早候选、尾部、零 draft 请求；缩减分支 q=0/inf/nan；精确比较最终 token |
| 改名逐元素 probe | kernel、所有参数及局部名全部改名 | 无名称依赖；CPU AST 检查除命名外结构不变；严格模式仍要求首次 T 复用 |
| 普通复制 probe | 与 U2 同类索引和外部循环边界，移除 atomic_add | 未证明覆盖/启动适配时仍保留 CHUNK；不能仅以“没有 atomic”准入 |

直接 kernel 的配置探针不声称生产 wrapper 已经会产生所有这些配置。
例如 bincount wrapper 固定 SEQ_BLOCK=256；128 是设计中的合法直接 kernel 对照。
原始 vector split QKV RoPE 用 batch=4、单 program 和完整行 tile，避免上游间接 position 读取的无 mask 尾部路径；
SIMT 入口直接接收独立构造的 precomputed cos/sin，测试其已有 mask 的尾部路径，不额外调用 precompute wrapper。
ngram sampled 数据遵循有效前缀加 -1 padding 的正常输入约定。

`BLOCK_NUM_REQS` 的断言采用本次用户指定的更严格契约：**禁止跨值复用**。
设计文档 §4.3 原本只举了容量不足的拒绝例子，本套测试也拒绝容量充足但取值不同的旧产物，
没有将这一新增要求写回设计文档。除文中明确标出的同产物正例外，新增 tile 序列允许保守原 JIT
回退，不以“数值相同”宣称已经证明所有 tile 可复用。

## 验证范围与后续接入

这里验证算子级可观测结果、实际产物身份和 ABI/static 常量保护。
设计中的 AutoReuseProfile、reason code、VariantRegistry、worker/loader 观测接口尚无稳定实现，
测试不伪造这些接口，也不把 hash 差异当作 `Unknown` 或 `StaticRequired` 的分类证据。

U1 的 `JIT_CALLEE_SUMMARY_MISSING` 只在人为限制分析能力时适用；本测试的完整 helper 可见，
严格模式检查其正向动态化。未来接入真实分析诊断后，应补独立的缺摘要/预算耗尽注入测试。
U2 的 `LAUNCH_ADAPTATION_UNPROVEN`、metadata 禁止跨值和 S6 的明确拒绝原因目前由静态身份/签名及结果
间接约束，尚不能断言具体 reason code。

后台初始化并发安全、READY 发布、编译/加载去重、确定性失败熔断、异常后不重放、
多设备/多 stream、autotuner 精确配置上下文、服务关闭状态、偏移重叠别名以及 P50/P95/P99
属于 TA 运行时专项测试，不由这套算子测试替代。

## 验证记录

44 项 host 检查和 22 个 JIT 定义的源码一致性检查通过，Python 语法检查通过。
2026-09-30 已在 A5-37 / torch-213 / Ascend950PR 上完成关闭动态复用的基线验证：
用户指定的 **19 个算子、57 个场景全部通过**；完整测试集 **59/61 通过**。
两项失败均为额外 ngram 对照：当前安装的 Triton-Ascend 3.6.0+dev20260929220210
不接受原码 `tl.load(..., care_padding=False)`，在编译前端报错。

完整结果、版本、逐算子覆盖和失败日志见
[A5-37 基线验证报告](validation/A5-37_torch-213_20260930.md)。
本次尚未开启动态复用或进行严格复用验收。
