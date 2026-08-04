# rtp-llm Qwen3.5 算子迁移指导

> **AscendC 类**：有原生 AscendC 实现（来自 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)代码仓），以 Python 包装 `fla_npu.ops.ascendc` 调用。
> **纯 Triton 类**：flash-linear-attention-npu 仓无等价实现时，移植原有的 Triton kernel 到 `triton_kernels/`，经 triton-ascend JIT 在 NPU 上运行。

---

## 第 1 章 NPU 算子来源分类

| NPU 算子来源分类（按使用优先级排序） | 典型代表 |
|---|---|
| torch_npu | torch.searchsorted |
| FLA（AscendC） | causal_conv1d |
| FLA（Triton） | l2norm |
| 自定义 Triton 算子 | store_ssm_state_to_block_map |
| 自定义 AscendC 算子 | 不在上述类别需要AscendC开发的算子 |

---

## 第 2 章 算子迁移速查

| GPU 算子 | NPU 对应算子 | NPU 算子来源 | 算子说明文档 | 测试用例 |
|---|---|---|---|---|
| causal_conv1d_fn | causal_conv1d | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d | https://github.com/ningweikang/rtp-llm/pull/24 |
| causal_conv1d_update | causal_conv1d | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d | https://github.com/ningweikang/rtp-llm/pull/24 |
| chunk_local_cumsum | chunk_local_cumsum | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum | https://github.com/ningweikang/rtp-llm/pull/26 |
| chunk_scaled_dot_kkt_fwd | chunk_scaled_dot_kkt | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt | https://github.com/ningweikang/rtp-llm/pull/28 |
| recompute_w_u_fwd | recompute_w_u_fwd | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd | https://github.com/ningweikang/rtp-llm/pull/27 |
| chunk_fwd_o | chunk_fwd_o | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o | https://github.com/ningweikang/rtp-llm/pull/28 |
| fused_recurrent_gated_delta_rule | recurrent_gated_delta_rule | FLA（AscendC） | https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule | https://github.com/ningweikang/rtp-llm/pull/28 |

---

## 第 3 章 fla 算子迁移

### 3.1 fla-npu 安装

```bash
# 1) CANN 环境（每次新 shell 都要 source）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) 编译 wheel
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
FLA_NPU_SOC=ascend950 python -m pip wheel --no-build-isolation --no-deps . -w dist

# 3) 安装 + 验证
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python -c "from fla_npu.ops import ascendc; print('ok')"
```

| 关键环境变量 | 作用 |
|---|---|
| `FLA_NPU_SOC` | 目标芯片：`ascend910b`/`ascend910_93`/`ascend950` |
| `FLA_NPU_INCREMENTAL_BUILD=1` | 增量构建（本地调试） |
| `FLA_NPU_OPS=op1,op2` | 仅构建指定算子（勿用于 release） |

> wheel 内嵌 OPP，通过绝对路径加载 `libcust_opapi.so`；不自动装 torch/torch_npu/triton-ascend，需自行匹配版本。

### 3.2 迁移步骤

1. **装 wheel**：按 3.1 完成安装，确认 `import fla_npu.ops.ascendc` 可用。
2. **查算子签名**：在第 2 章速查表中找到对应 GPU 算子 → 点开"算子说明文档"链接，查看 AscendC API 参数签名。
3. **建包装层**：在 [rtp_llm/models_py/](../../rtp-llm-npu/rtp_llm/models_py/) 下提供与 Triton **同名接口** + device 分发（NPU→ascendc），按第 2 章算子 API 签名做参数适配。
4. **模型层接入**：把 `from ...triton_kernels.xxx import yyy` 改为从包装层导入（签名不变）。

**包装层骨架**（device 分发 + 布局适配，以 `causal_conv1d` 为例）：

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
        x.T,                                   # Triton (dim,cu_seqlen) → AscendC (cu_seqlen,dim)
        weight.T,                              # Triton (dim,width) → AscendC (width,dim)
        bias, conv_states,
        query_start_loc=query_start_loc.to(torch.int64),
        cache_indices=_block_map_to_cache_indices(block_map, prefix_lengths),
        activation_mode=1 if activation in ("silu", "swish") else 0,
        pad_slot_id=pad_slot_id, run_mode=0,
    )
    return y.T.to(x.dtype)
```
