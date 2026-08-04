# AscendC 与移植原有 Triton 算子迁移到 rtp-llm 指导文档

> 本文档指导把 `causal_conv1d`、`causal_conv1d_update`、`chunk_gated_delta_rule`、`chunk_local_cumsum` 等算子迁移到 NPU 版 rtp-llm 仓，服务于 Qwen3.5 等线性注意力模型。
>
> **AscendC 类**：有原生 AscendC 实现（来自 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)代码仓），以 Python 包装 `fla_npu.ops.ascendc` 调用。
> **纯 Triton 类**：flash-linear-attention-npu 仓无等价实现时，移植原有的 Triton kernel 到 `triton_kernels/`，经 triton-ascend JIT 在 NPU 上运行。

---

## 第 1 章 概述：两类迁移对比

| 维度 | AscendC 类算子 | 纯 Triton 类算子 |
|---|---|---|
| 代表算子 | `causal_conv1d`、`causal_conv1d_update`、`chunk_gated_delta_rule` | `chunk_local_cumsum`、`layernorm_gated`、`moe_gating`、`solve_tril` |
| 实现来源 | flash-linear-attention-npu wheel（`fla_npu.ops.ascendc`） | 原仓库的 Triton 实现 |
| 运行机制 | Python ctypes 直调 aclnn（原生 AscendC） | triton-ascend JIT 编译运行 |
| rtp-llm 侧改动 | 新建 Python 包装层 + device 分发 | 在 `triton_kernels/` 新增 `.py` + NPU 适配 |
| 性能 | 高（原生 kernel） | 取决于 triton-ascend 编译质量 |
| 选型判据 | flash-linear-attention-npu 有等价实现 → AscendC | flash-linear-attention-npu 无等价实现 → 移植原有 Triton |

**判据**：先查 `from fla_npu.ops import ascendc; dir(ascendc)` 是否有对应算子；若无，再查询flash-linear-attention-npu代码仓是否有等价实现。
有 → 走 AscendC 类；无 → 移植原有 Triton kernel 到 `triton_kernels/`。

---

## 第 2 章 参考资源

| 资源 | 用途 |
|---|---|
| [flashserve/flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) | AscendC 算子源 + wheel + Python wrapper |
| [ningweikang/rtp-llm PR#24](https://github.com/ningweikang/rtp-llm/pull/24/files) | `gpu_npu_comparison_guide.md`：精度比对方法 |
| [vllm-ascend `csrc/moe/causal_conv1d/`](https://github.com/vllm-project/vllm-ascend/tree/main/csrc/moe/causal_conv1d/) | `*_tiling_validation.h`：NPU 输入约束 |
| [rtp-llm-npu `triton_kernels/`](rtp_llm/models_py/triton_kernels/) | 现有 Triton 实现（两类迁移的接入点/回落路径） |
| [rtp-llm-npu `triton_kernels/fla/gdn_gating.py`](rtp_llm/models_py/triton_kernels/fla/gdn_gating.py) | 移植原有 Triton 范例 |

---

## 第 3 章 迁移单算子精度比对（精简自 PR#24）

两类迁移都用同一套方法验证 NPU 输出正确性。

**流程**：`GPU 黄金数据(.pt) → 恢复非连续 stride → 格式转换 → 调 NPU 算子 → 精度比对`

### 3.1 恢复非连续 stride（可选）

对于非连续的输入数据，`torch.save()` 的数据如果经过`contiguous()`处理，丢失原始 stride，则需要恢复 stride 后比对。
GPU dump 脚本保存 `input_meta`（含 shape/stride/dtype/contiguous信息），用 `torch.empty_strided` 恢复：

```python
def _restore_strided_tensor(saved: torch.Tensor, meta: dict) -> torch.Tensor:
    if meta.get("contiguous", True):
        return saved
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    t = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    t.copy_(saved)
    return t
```
如果数据未经过`contiguous()`处理，直接使用`torch.load()`加载。

### 3.2 精度与断言

> 具体参考《[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》
- GPU↔CPU 参考容许差 ≈ `0.0156`（fp16 单 ULP）。
- NPU↔GPU 建议用 `rtol=5e-2, atol=5e-2`。
- 恢复 stride 后**必须断言非连续性被保留**：

```python
self.assertFalse(x_gpu.is_contiguous())
self.assertEqual(conv_states_npu.stride(-1), 1)   # dim 轴 stride 必须为 1
```

### 3.3 测试模板骨架

```python
import unittest, torch
from fla_npu.ops import ascendc as ascendc_ops   # AscendC 类
# from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating  # 纯 Triton 类

class TestOpGpuGolden(unittest.TestCase):
    rtol = atol = 5e-2
    def assertTensorClose(self, a, e, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertTrue(torch.allclose(a.cpu().float(), e.cpu().float(), rtol=rtol, atol=atol))
```

---

## 第 4 章 AscendC 类算子迁移

### 4.1 fla-npu 安装

```bash
# 1) CANN 环境（每次新 shell 都要 source）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) 编译 wheel（A3 机器用 ascend910_93，A5 用 ascend950）
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
FLA_NPU_SOC=ascend950 python -m pip wheel --no-build-isolation --no-deps . -w dist

# 3) 安装 + 验证
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python -c "from fla_npu.ops import ascendc; print('ok')"
bash test.sh --device 0 --op causal_conv1d   # 单算子测试
```

关键环境变量：

| 变量 | 作用 |
|---|---|
| `FLA_NPU_SOC` | 目标芯片：`ascend910b`/`ascend910_93`/`ascend950` |
| `FLA_NPU_INCREMENTAL_BUILD=1` | 增量构建（本地调试） |
| `FLA_NPU_OPS=op1,op2` | 仅构建指定算子（定位用，勿用于 release） |

> wheel 内嵌 OPP，通过绝对路径加载 `libcust_opapi.so`；不自动装 torch/torch_npu/triton-ascend，需自行匹配版本。

### 4.2 算子 API 示例

导入：`from fla_npu.ops import ascendc`（首次调用自动加载 opapi 库）。

**causal_conv1d**（prefill 与 update 同一算子，靠 `run_mode` 区分）：

```python
ascendc.causal_conv1d(
    x, weight, bias=None, conv_states=None, *,
    query_start_loc=None, cache_indices=None, initial_state_mode=None,
    num_accepted_tokens=None, activation_mode=0, pad_slot_id=-1,
    run_mode=0,   # 0=prefill, 1=decode/update
    head_num=0,
)
```

### 4.3 迁移步骤

1. **装 wheel**：按 4.1，确认 `import fla_npu.ops.ascendc` 可用。
2. **建包装层（可选）**：在 [rtp_llm/models_py/](../../rtp-llm-npu/rtp_llm/models_py/) 下新建 `ascendc_kernels/`（与 `triton_kernels/` 并列），提供与 Triton **同名接口** + device 分发（NPU→ascendc）。
3. **参数适配**：按 4.4 映射表做转换，将入参形式与 Ascendc 签名一致，转换方式与算子本身相关，此处只做示例。
4. **模型层接入**：把 `from ...triton_kernels.causal_conv1d import ...` 改为从 `ascendc_kernels` 导入（签名不变）。

包装层骨架（device 分发 + 布局适配）示例：

```python
# rtp_llm/models_py/ascendc_kernels/causal_conv1d.py
import torch
from fla_npu.ops import ascendc

def _is_npu(x): return x.device.type in ("npu", "privateuseone")

def causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc, block_map,
                     prefix_lengths, seq_size_per_block, activation="silu",
                     pad_slot_id=-1, metadata=None, validate_data=False):
    if not _is_npu(x):
        from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_fn as _t
        return _t(x, weight, bias, conv_states, query_start_loc, block_map,
                  prefix_lengths, seq_size_per_block, activation, pad_slot_id,
                  metadata=metadata, validate_data=validate_data)
    y = ascendc.causal_conv1d(
        x.T,                                   # (dim,cu_seqlen) -> (cu_seqlen,dim)
        weight.T,                              # (dim,width)     -> (width,dim)
        bias, conv_states,
        query_start_loc=query_start_loc.to(torch.int64),
        cache_indices=_block_map_to_cache_indices(block_map, prefix_lengths),
        activation_mode=1 if activation in ("silu", "swish") else 0,
        pad_slot_id=pad_slot_id, run_mode=0,
    )
    return y.T.to(x.dtype)
```

### 4.4 参数映射表

**causal_conv1d_fn（prefill）**：源 [triton_kernels/causal_conv1d/causal_conv1d.py](../../rtp-llm-npu/rtp_llm/models_py/triton_kernels/causal_conv1d/causal_conv1d.py) → `ascendc.causal_conv1d(run_mode=0)`

> **注意**：AscendC 算子支持非连续输入，无需强制 `contiguous()`

| Triton 参数 | Triton 形态 | fla-npu | 转换 |
|---|---|---|---|
| `x` | `(dim,cu_seqlen)` channel-last | `x` `(cu_seqlen,dim)` | `x.T`（保留非连续） |
| `weight` | `(dim,width)` | `weight` `(width,dim)` | `.T` |
| `bias` | `(dim,)`/None | `bias` | 透传 |
| `conv_states` | `(...,dim,width-1)` dim 轴 stride=1 | `(num_cache_lines,state_len,dim)` | transpose 使 dim 在末维 |
| `query_start_loc` | int32 | int64 | `.to(torch.int64)` |
| `block_map`+`prefix_lengths` | int32 | `cache_indices` int64 | 按 prefix_lengths 取 page 索引 |
| `activation` | `"silu"`/`None` | `activation_mode` | `1`/`0` |
| `pad_slot_id` | int | `pad_slot_id` | 透传 |
| `metadata`/`seq_size_per_block` | — | — | NPU 无此概念，忽略 |
| 返回 `(dim,cu_seqlen)` | — | `y` `(cu_seqlen,dim)` | `y.T` |

**causal_conv1d_update（decode/投机）**：Triton `causal_conv1d_update` → `ascendc.causal_conv1d(run_mode=1)`

| Triton 参数 | Triton 形态 | fla-npu | 转换 |
|---|---|---|---|
| `x` | `(batch,dim,seqlen)`/`(batch,dim)` | `x` `(batch,seqlen,dim)`/`(batch,dim)` | 单步 squeeze；多步 `transpose(1,2)` |
| `conv_state` | `(...,dim,state_len)` | `conv_states` `(num_cache_lines,state_len,dim)` | transpose |
| `weight` | `(dim,width)` | `(width,dim)` | `.T` |
| `block_map`+`sequence_lengths` | int32 | `cache_indices` int64 | cal_block_idx 取 page |
| `activation` | `"silu"`/`None` | `activation_mode` | `1`/`0` |
| `cache_seqlens`/`max_query_len` | int | — | NPU 不支持/由 x 推断，忽略 |
| — | — | `num_accepted_tokens` | 投机解码传 `(batch,)` int64（仅 width=4） |

---

## 第 5 章 原有 Triton 类算子迁移

### 5.1 迁移步骤

1. **定位原有 kernel**：找到原模型（如 Qwen3.5）的 `@triton.jit` kernel 及其 Python 侧调用函数。
2. **移植到 `triton_kernels/`**：在对应子目录（`fla/`、`common/`、`moe/` 等）新增 `.py`，保留 `@triton.jit` kernel + wrapper 函数。
3. **NPU/triton-ascend 适配**：按 5.2 处理 gather/TMA/autotune/contiguous/device 等 NPU 差异点，建议使用AI完成，可使用[cannbot skill 集合](https://gitcode.com/cann/cannbot-skills)。
4. **包装与导出**：在子目录 `__init__.py` 导出；wrapper 内做 stride 断言与输出张量分配。
5. **模型层接入**：在 [model_desc/qwen3_next.py](../../rtp-llm-npu/rtp_llm/models_py/model_desc/qwen3_next.py) 等调用点 import 新 kernel。

### 5.2 NPU / triton-ascend 适配

> **官方迁移指南**：《[GPU Triton算子迁移](https://triton-ascend.readthedocs.io/zh-cn/latest/migration_guide/migrate_from_gpu.html)》

移植原有 Triton kernel 时，需要先参考官方迁移指南将 GPU 专用语法移除，针对 NPU 做适配。

### 5.3 完整范例：chunk_local_cumsum

**背景**：计算 chunk 内的局部累加和，用于线性注意力模型的状态更新。来自 fla-org 的 Triton 实现，flash-linear-attention-npu 仓无等价实现 → 移植原有 Triton kernel。

**kernel + wrapper**（[triton_kernels/fla/cumsum.py](../../rtp-llm-npu/rtp_llm/models_py/triton_kernels/fla/cumsum.py)，已含 NPU 适配）：

```python
import torch, triton, triton.language as tl
from rtp_llm.models_py.triton_kernels.common.decorators import cuda_autotune
from rtp_llm.models_py.triton_kernels.fla.utils import check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

@triton.heuristics({
    "HAS_SCALE": lambda args: args["scale"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@cuda_autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
    REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # ... kernel 实现（详见源文件）
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@input_guard
def chunk_local_cumsum(g: torch.Tensor, chunk_size: int,
                       reverse: bool = False, scale: float = None,
                       cu_seqlens: Optional[torch.Tensor] = None,
                       head_first: bool = False,
                       output_dtype: Optional[torch.dtype] = torch.float, **kwargs):
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype)
    else:
        raise ValueError(f"Unsupported input shape {g.shape}")
```

**NPU 适配要点（对应 5.2）**：
- **`cuda_autotune`**：在 NPU 上退化为 no-op（`torch.cuda.is_available()` 为假），规避 CUDA-only autotune 问题。
- **`check_shared_mem`**：探测 NPU 共享内存，失败时用 `BS_LIST = [16, 32]` 兜底，避免共享内存探测失败。
- **`input_guard`**：装饰器，处理输入张量的非连续 stride，支持非连续输入。
- **`@triton.heuristics`**：自动推断编译时常量（如 `HAS_SCALE`、`IS_VARLEN`），NPU 上同样支持。
- **`make_block_ptr`**：用于高效访存，NPU 上同样支持，避免 scalar 低效映射。
- **`tl.dot`**：用于矩阵乘，NPU 上映射为 Cube Core 指令。

**模型层接入**（[triton_kernels/fla/chunk.py](../../rtp-llm-npu/rtp_llm/models_py/triton_kernels/fla/chunk.py)）：

```python
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
# chunk 内状态更新
g = chunk_local_cumsum(g, chunk_size=self.chunk_size)
```

---

## 第 6 章 验证清单

### 6.1 环境
- [ ] 检查triton版本在当前环境可用
- [ ] `import fla_npu.ops.ascendc` 可用，查 `dir(ascendc)` 确认算子存在（AscendC 类）
- [ ] `bash test.sh --device 0 --op causal_conv1d` 通过

### 6.2 AscendC 类
- [ ] `causal_conv1d` prefill / update：GPU 黄金数据比对通过（`5e-2`）+ 非连续断言
- [ ] 与原算子端到端相比 Triton 调用路径一致

### 6.3 纯 Triton 类
- [ ] `chunk_local_cumsum`：与 PyTorch 参考比对通过（`5e-2`）
- [ ] 模型层调用点（prefill/decode）输出与参考一致
- [ ] 无 `tl.gather`/TMA/autotune 等兼容性报错


---

## 附录 A：常见问题与排错

### A.1 环境
- **`Unable to initialize fla_npu Ascend C op_api libraries`**：未 `source CANN set_env.sh` 或 wheel 与 CANN/芯片不符（`FLA_NPU_SOC` 须匹配运行机器）。
- **triton 报错 / 性能差**：triton<3.2.0 或 Python<3.11，按 `check_environments` 警告升级。

### A.2 AscendC 调用
- **`aclnnStatus=561002`**：weight 维序 `(W,D)`、`dim%16==0`、`state_len>=W-1`、int64、dtype 一致。
- **conv_states 原地更新与 autograd 冲突**：`conv_states` 不可 `requires_grad`；推理路径确保无梯度。
- **读不到更新后的 conv_states**：保存 NPU tensor 引用再 `.cpu()`（见 3.3）。

### A.3 移植原有 Triton
- **非连续输入结果错乱**：wrapper 内补 stride 断言，或用 `input_guard` 强制 contiguous。
- **autotune 在 NPU 报错**：换 `cuda_autotune`（NPU 上 no-op），手写一组 NPU 友好的 num_warps/num_stages。
- **共享内存探测失败**：用 `Backend.DEFAULT` 兜底，不要硬编码 A100/H100 的 shared_mem。

### A.4 精度
- 先确认 CPU 参考能匹配 GPU，再查 NPU；NPU↔GPU 用 `rtol/atol=5e-2`。
- 非连续 stride 必须从 `input_meta` 恢复并断言保留。
