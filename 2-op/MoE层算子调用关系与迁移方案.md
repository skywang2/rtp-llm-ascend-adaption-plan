# MoE 层算子调用关系与 NPU 迁移方案

> 基于 `rtp-llm/rtp_llm/models_py/model_desc/generic_moe.py`、`triton_kernels/moe/fused_moe_kernel.py`、`triton_kernels/common/moe_gating.py` 与 GPU 侧 nsys 采样结果整理。

## 一、MoE 层内部算子调用关系

`GenericMoeLayer.forward` 完成一次 MoE 前向时，按以下顺序触发各算子：

```
GenericMoeLayer.forward(hidden_states)
│
├─ 1. self.gate(hidden_states)                       # Router Linear，走 cuBLAS
│      └─ cublasLtMatmul / cublasGemvStridedBatchedEx
│
├─ 2. 路由选择 (二选一)
│   ├─ if correction_bias is not None:
│   │    self.group_topk(...)                         # GroupTopKOp（C++），含 sigmoid + correction_bias + topk
│   │
│   └─ else:
│        self.select_topk(router_logits_fp32, topk_ids, topk_weights)
│        └─ topkGatingSoftmaxKernelLauncher           # C++ CUDA Kernel
│             ├─ moeSoftmax
│             └─ moeTopK
│
├─ 3. (可选) self.fake_balance_expert(topk_ids, topk_weights)
│
├─ 4. self.fused_moe(hidden_states, topk_weights, topk_ids, activation="SiGLU")
│      └─ TritonFusedMoeExecutor.execute
│           ├─ moe_align_block_size_torch(topk_ids, block_size, E)
│           │    ├─ torch.zeros / scatter_add_         # 统计每 expert token 数
│           │    ├─ torch.cumsum                      # 计算 expert_offsets
│           │    ├─ flat_ids.argsort(stable=True)      # torch.sort (radixSortKVInPlace)
│           │    │     └─ cub::DeviceScan / DeviceScanInit   # sort 内部 prefix sum
│           │    └─ torch.searchsorted                  # 每个 block 对应的 expert_id
│           │
│           ├─ invoke_fused_moe_kernel (GEMM1: hidden_states @ w1.T)
│           │    └─ fused_moe_kernel (Triton)
│           │         ├─ Swizzle 调度
│           │         ├─ K 维分块循环累加
│           │         └─ (可选) 融合 routing weight 乘法
│           │
│           ├─ silu_and_mul(intermediate1)              # 激活融合
│           │    └─ _silu_and_mul_kernel (Triton)
│           │
│           ├─ moe_align_block_size_torch(...)          # 若 block_m1 != block_m2 再算一次
│           │
│           ├─ invoke_fused_moe_kernel (GEMM2: intermediate2 @ w2.T)
│           │    └─ fused_moe_kernel (Triton, mul_routed_weight=True)
│           │
│           └─ out.view(M, top_k, K).sum(dim=1)        # torch.sum reduce
│
└─ 5. (可选) shared expert 路径
       ├─ self.shared_expert(hidden_states)            # DenseMLP
       ├─ self.shared_expert_gate(hidden_states)        # Linear
       └─ self.sigmoid_gate_scale_add(gate, shared, experts)
            └─ sigmoid_gate_scale_add_triton
                 └─ _SigmoidGateScaleAdd_kernel (Triton)
```

**关键融合点**：
- `fused_moe_kernel`：融合跨 expert 的 GEMM + routing weight 乘法（GEMM2 中 `mul_routed_weight=True`）。
- `sigmoid_gate_scale_add_triton`：融合 sigmoid + mul + add 三步。
- `topkGatingSoftmaxKernelLauncher`：融合 softmax + topk（CUDA 专用 case 模板，default 走两次 kernel）。

## 二、nsys 采样中的 MoE 相关算子

来源：`GPU侧nsys采样/nsys_profiling/all_ops.csv` 中 `category = MoE` 的条目（已剔除 PyTorch 原生 `torch.*` 与 `cub::` 内部辅助算子，仅保留 MoE 专属算子）。

| 算子 (api_name) | kernel_name | 来源/触发位置 |
|---|---|---|
| `invoke_fused_moe_kernel` | `fused_moe_kernel` | Triton GEMM，GEMM1/GEMM2 各 8800 次 |
| `topkGatingSoftmaxKernelLauncher (MoE gating)` | `topkGatingSoftmax` | `moe_routing_kernels.cu:436`，C++ 路由选择 |
| `sigmoid_gate_scale_add_triton (MoE gating)` | `_SigmoidGateScaleAdd_kernel` | `moe_gating.py:110`，shared expert 融合 |

## 三、各算子 NPU 迁移方案（一句话陈述，按 MoE 前向调用顺序排列）

1. **`topkGatingSoftmaxKernelLauncher` (topkGatingSoftmax)**  — 路由选择阶段（`fused_moe` 之前）
   替换为 `torch_npu.npu_moe_gating_top_k_softmax`，并在 Python 层补充 `start_expert` 偏移、`startk/endk` 切片以及 RENORMALIZE 归一化三处补丁（详见 `topkGatingSoftmaxKernelLauncher_算子解析与NPU迁移建议.md`）。

2. **`invoke_fused_moe_kernel` (fused_moe_kernel)**  — GEMM1/GEMM2 阶段（`fused_moe` 内部）
   将 GEMM1/GEMM2 + silu_and_mul 整体替换为 `torch_npu.npu_grouped_matmul_swiglu_quant_v2`（含量化场景），非量化可用 `torch_npu.npu_grouped_matmul` + 单独 `silu_and_mul`，routing weight 乘法下沉到 Python 层逐 expert 应用或交给 CANN 内部融合。

3. **`sigmoid_gate_scale_add_triton` (_SigmoidGateScaleAdd_kernel)**  — Shared Expert 融合阶段（`fused_moe` 之后，可选）
   优先使用 `torch_npu.npu_sigmoid_mul_add`（若存在等价融合算子），否则退化为纯 PyTorch 实现 `experts.add_(torch.sigmoid(gate) * shared)`（当前 `rtp-llm-npu/.../ascend/moe_gating.py` 已采用此 fallback）。

## 四、备注

- `silu_and_mul`、cuBLAS GEMV/GEMM（Router Linear、Shared Expert Linear）虽在 MoE 前向中被调用，但在 nsys 分类中归为「基础通用/cuBLAS」「基础通用/激活函数」，不在本表 MoE 算子范围。
- `moe_align_block_size_torch` 是 `invoke_fused_moe_kernel` 的 Python 前置数据准备函数，本身不算独立算子，但触发了 `torch.sort`/`torch.searchsorted`/`scatter_add_`/`cumsum` 等多个原生算子；迁移时建议整体替换为 CANN 的 `npu_moe_compute_block_status` + `npu_moe_token_unpermute` 流水，可一次性消除 sort/searchsorted/cumsum 调用。
- 若采用 vllm-ascend 的 `unquant_apply_mlp` 路径（基于 `npu_grouped_matmul`），`moe_align_block_size_torch` 仍然需要，但其输出可直接喂给 `npu_grouped_matmul` 的 `group_list` 参数。
