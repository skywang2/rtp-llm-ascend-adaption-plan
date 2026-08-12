# Qwen3.5 MoE 层参考 vllm-ascend 的整体迁移方案

> 适配 rtp-llm 框架,核心 MoE 流水基于 vllm-ascend CANN 算子实现。

---

## 一、整体调用关系图

```
GenericMoeLayer.forward(hidden_states)                    ← 保留 rtp-llm 框架
│
├─ 1. self.gate(hidden_states)                           ← 保留 rtp-llm LinearFactory
│      → router_logits [M, E]
│
├─ 2. 路由选择                                            ← 替换为 CANN 算子
│      └─ torch_npu.npu_moe_gating_top_k_softmax(router_logits, top_k)
│         → topk_weights [M, top_k], topk_ids [M, top_k]
│
├─ 3. NpuFusedMoe.execute(hidden_states, topk_weights, topk_ids)  ← 核心替换
│      │
│      ├─ 3.1 router.prepare()                            ← 多卡分发
│      │      ├─ (TP) 不通信,纯量化/重排
│      │      └─ (EP) torch_npu.npu_moe_distribute_dispatch_v2  ← 多卡 all_to_all
│      │
│      ├─ 3.2 token_dispatch                              ← Token 排序
│      │      └─ torch_npu.npu_moe_init_routing_v2
│      │         → sorted_x, expanded_row_idx, group_list
│      │
│      ├─ 3.3 fused_experts.execute()                     ← GEMM 流水
│      │      ├─ GEMM1: torch_npu.npu_grouped_matmul(x, w1, group_list)
│      │      ├─ 激活:   torch_npu.npu_swiglu(gate_up_out)
│      │      ├─ 路由权重: gate_up_out *= topk_scales  (可选)
│      │      └─ GEMM2: torch_npu.npu_grouped_matmul(x, w2, group_list)
│      │
│      └─ 3.4 router.finalize()                            ← 多卡合并
│             ├─ (TP) all_reduce(output, Group.TP)        ← 多卡 all_reduce
│             └─ (EP) torch_npu.npu_moe_distribute_combine_v2  ← 多卡 all_to_all
│
└─ 4. Shared Expert 融合                                  ← 保留 rtp-llm
       ├─ self.shared_expert(hidden_states)               ← DenseMLP
       ├─ self.shared_expert_gate(hidden_states)           ← Linear
       └─ output = experts + sigmoid(gate) * shared       ← 纯 PyTorch
```

### 多卡场景对照

| 场景 | prepare(分发) | finalize(合并) | token_dispatch |
|---|---|---|---|
| **单卡** | 不通信 | 不通信 | `npu_moe_init_routing_v2` |
| **TP**(Qwen3.5 默认) | 不通信 | `all_reduce(Group.TP)` | `npu_moe_init_routing_v2` |
| **EP** | `npu_moe_distribute_dispatch_v2`(all_to_all) | `npu_moe_distribute_combine_v2`(all_to_all) | 融合在 dispatch 算子内 |

---

## 二、算子对应替换表

### 2.1 保留 rtp-llm 的部分

| 组件 | rtp-llm 实现 | 说明 |
|---|---|---|
| `GenericMoeLayer` 框架 | `generic_moe.py` | 层入口,保留 |
| `self.gate`(Router Linear) | `LinearFactory` | NPU 自动走 CANN GEMM |
| `self.shared_expert` | `DenseMLP` | Shared Expert,保留 |
| `self.shared_expert_gate` | `LinearFactory` | Shared Expert 门控,保留 |
| `fake_balance_expert` | rtp-llm 实现 | ⚠️ **NPU 不可用**:迁移时需关闭 `moe_config.fake_balance_expert`,或另行补齐实现 |

### 2.2 替换为 vllm-ascend CANN 算子的部分

| 阶段 | rtp-llm (CUDA/Triton) | vllm-ascend (NPU/CANN) | 触发位置 |
|---|---|---|---|
| 路由选择 | `topkGatingSoftmaxKernelLauncher` | `npu_moe_gating_top_k_softmax` | GenericMoeLayer.forward |
| Token 排序 | `moe_align_block_size_torch` | `npu_moe_init_routing_v2` | router.prepare |
| GEMM1 | `invoke_fused_moe_kernel` | `npu_grouped_matmul` | fused_experts.execute |
| 激活 | `silu_and_mul` (Triton) | `npu_swiglu` | fused_experts.execute |
| 路由权重 | 融合在 GEMM2 epilogue | `* topk_scales` (显式 mul) | fused_experts.execute |
| GEMM2 | `invoke_fused_moe_kernel` | `npu_grouped_matmul` | fused_experts.execute |
| Token 合并 | `out.view(M,topk,K).sum(1)` | `npu_moe_token_unpermute` | router.finalize |
| TP 合并 | `all_reduce(Group.TP)` | `all_reduce(Group.TP)` (保留) | router.finalize |
| EP 分发 | (rtp-llm 无 EP) | `npu_moe_distribute_dispatch_v2` | router.prepare |
| EP 合并 | (rtp-llm 无 EP) | `npu_moe_distribute_combine_v2` | router.finalize |
| Shared Expert 融合 | `sigmoid_gate_scale_add_triton` | `experts + sigmoid(gate) * shared` (PyTorch) | GenericMoeLayer.forward |

### 2.3 量化路径(可选)

| 阶段 | 非量化 | 量化融合 | 量化非融合 |
|---|---|---|---|
| GEMM1 | `npu_grouped_matmul` | `npu_grouped_matmul_swiglu_quant` | `npu_grouped_matmul` (带 scale) |
| 激活 | `npu_swiglu` | (融合在 GEMM1) | `npu_dequant_swiglu_quant` |
| GEMM2 | `npu_grouped_matmul` | `npu_grouped_matmul_gmm2` | `npu_grouped_matmul_gmm2` |
| 输入量化 | 无 | `npu_dynamic_quant` (前置) | `npu_dynamic_quant` (前置) |

---

## 三、适配层实现

### 3.1 NpuMoeLayer(替换 GenericMoeLayer)

```python
class NpuMoeLayer(nn.Module):
    """适配 rtp-llm 框架,核心流水分用 vllm-ascend CANN 算子"""

    def __init__(self, config, parallelism_config, weights, moe_config, ...):
        # 1. Router Linear (保留 rtp-llm)
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, config.quant_config
        )

        # 2. MoE 权重 (适配 CANN 布局,注意 transpose)
        self.w1 = weights[W.moe_w1].transpose(1, 2)  # [E, hidden, 2*inter]
        self.w2 = weights[W.moe_w2].transpose(1, 2)  # [E, inter, hidden]
        self.top_k = config.moe_k
        self.E = config.expert_num
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.ep_size = parallelism_config.ep_size

        # 3. Shared Expert (保留 rtp-llm, moe_style=2)
        if config.moe_style == 2:
            self.shared_expert = DenseMLP(
                config.activation_type, parallelism_config, weights, config.quant_config
            )
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )

    def forward(self, hidden_states):
        # 步骤 1: Router Linear
        router_logits = self.gate(hidden_states)

        # 步骤 2: 路由选择 (CANN)
        topk_weights, topk_ids = torch_npu.npu_moe_gating_top_k_softmax(
            router_logits, top_k=self.top_k
        )

        # 步骤 3: 核心 MoE 流水 (CANN)
        experts_output = self._npu_fused_moe(
            hidden_states, topk_weights, topk_ids
        )

        # 步骤 4: Shared Expert 融合 (保留 rtp-llm)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
                experts_output = experts_output + gate * shared_output
            else:
                experts_output = experts_output + shared_output

        return experts_output

    def _npu_fused_moe(self, hidden_states, topk_weights, topk_ids):
        """核心 MoE 流水,对应 rtp-llm 的 self.fused_moe"""
        # 3.1 Token 排序 (CANN)
        sorted_x, expanded_row_idx, group_list = torch_npu.npu_moe_init_routing_v2(
            hidden_states, topk_ids, expert_num=self.E
        )

        # 3.2 GEMM1 (CANN)
        gate_up = torch_npu.npu_grouped_matmul(
            x=sorted_x, weight=self.w1, group_list=group_list, group_list_type=0
        )

        # 3.3 激活 (CANN)
        activated = torch_npu.npu_swiglu(gate_up)

        # 3.4 路由权重 (显式 mul)
        activated = activated * topk_weights[expanded_row_idx].unsqueeze(-1)

        # 3.5 GEMM2 (CANN)
        expert_out = torch_npu.npu_grouped_matmul(
            x=activated, weight=self.w2, group_list=group_list, group_list_type=0
        )

        # 3.6 Token 合并 (CANN, scatter + 乘权重)
        output = torch_npu.npu_moe_token_unpermute(
            expert_out, expanded_row_idx, probs=topk_weights
        )

        # 3.7 多卡合并 (TP: all_reduce; EP: 已在 dispatch/combine 中处理)
        if self.tp_size > 1 and self.ep_size == 1:
            output = all_reduce(output, group=Group.TP)

        return output
```

### 3.2 EP 场景的 _npu_fused_moe(若未来需要 EP)

```python
def _npu_fused_moe_ep(self, hidden_states, topk_weights, topk_ids):
    # 3.1 EP 分发 (CANN,含 all_to_all + 排序)
    expand_x, dynamic_scale, expert_token_nums, ep_recv_counts = (
        torch_npu.npu_moe_distribute_dispatch_v2(
            hidden_states, topk_weights, topk_ids, ...
        )
    )

    # 3.2-3.5 GEMM 流水 (同 TP 路径,使用 expert_token_nums 作为 group_list)
    gate_up = torch_npu.npu_grouped_matmul(x=expand_x, weight=self.w1, ...)
    activated = torch_npu.npu_swiglu(gate_up)
    expert_out = torch_npu.npu_grouped_matmul(x=activated, weight=self.w2, ...)

    # 3.6 EP 合并 (CANN,含 all_to_all + scatter + 乘权重)
    output = torch_npu.npu_moe_distribute_combine_v2(
        expert_out, topk_weights, ep_recv_counts, ...
    )
    return output
```

---

## 四、迁移步骤

### 步骤 1:权重布局适配

```python
# rtp-llm 权重布局 → CANN 要求的布局
self.w1 = weights[W.moe_w1].transpose(1, 2)  # [E, 2*inter, hidden] → [E, hidden, 2*inter]
self.w2 = weights[W.moe_w2].transpose(1, 2)  # [E, hidden, inter]  → [E, inter, hidden]
```

### 步骤 2:实现 NpuMoeLayer

按 3.1 实现 `NpuMoeLayer`,替换 `GenericMoeLayer`。

### 步骤 3:在 GenericMoeDecoderLayer 中切换

```python
# generic_moe.py#L225
if layer_idx not in config.moe_layer_index:
    self.mlp = DenseMLP(...)
else:
    self.mlp = NpuMoeLayer(...)   # ← 替换 GenericMoeLayer
```

### 步骤 4:多卡配置

| 场景 | 配置 | 走的路径 |
|---|---|---|
| 单卡 | `tp_size=1, ep_size=1` | 纯本地,无通信 |
| TP | `tp_size>1, ep_size=1` | `npu_moe_init_routing_v2` + `all_reduce` |
| EP | `ep_size>1` | `npu_moe_distribute_dispatch_v2/combine_v2` |

### 步骤 5:验证

1. 单卡正确性:对比 GPU 输出
2. TP 多卡:`all_reduce` 结果与单卡一致
3. 性能测试:对比 Triton kernel

---

## 五、关键差异说明

| 维度 | rtp-llm 原始 | 迁移后 |
|---|---|---|
| 路由选择 | `topkGatingSoftmaxKernelLauncher` (C++) | `npu_moe_gating_top_k_softmax` (CANN) |
| Token 排序 | `moe_align_block_size_torch` (argsort+cumsum) | `npu_moe_init_routing_v2` (一步融合) |
| GEMM | `invoke_fused_moe_kernel` (Triton) | `npu_grouped_matmul` (CANN) |
| 路由权重 | 融合在 GEMM2 epilogue | 显式 `* topk_scales` 或 `unpermute(probs=)` |
| Token 合并 | `out.view(M,topk,K).sum(1)` | `npu_moe_token_unpermute` (融合 scatter+mul) |
| Shared Expert | `sigmoid_gate_scale_add_triton` | 纯 PyTorch `sigmoid + mul + add` |
| EP 通信 | 无(PureTpRouter) | `npu_moe_distribute_dispatch/combine_v2` (MC2) |

---

## 六、参考文档

> 文档地址：[链接](https://gitcode.com/skywang2/rtp-llm-porting/tree/main/docs/Qwen3.5/%E7%AE%97%E5%AD%90%E8%BF%81%E7%A7%BB%E6%A2%B3%E7%90%86%E6%96%87%E6%A1%A3)

- [MoE层算子调用关系与迁移方案.md](https://gitcode.com/skywang2/rtp-llm-porting/blob/main/docs/Qwen3.5/算子迁移梳理文档/MoE层算子调用关系与迁移方案.md) — rtp-llm MoE 算子调用关系
- [vllm-ascend MoE 层算子调用关系.md](https://gitcode.com/skywang2/rtp-llm-porting/blob/main/docs/Qwen3.5/算子迁移梳理文档/vllm-ascend%20MoE%20层算子调用关系.md) — vllm-ascend MoE 算子调用关系
- [策略A_最小替换.md](https://gitcode.com/skywang2/rtp-llm-porting/blob/main/docs/Qwen3.5/算子迁移梳理文档/策略A_最小替换.md) — 仅替换 GEMM 流水
- [策略B_中等替换.md](https://gitcode.com/skywang2/rtp-llm-porting/blob/main/docs/Qwen3.5/算子迁移梳理文档/策略B_中等替换.md) — 替换路由+GEMM 流水
- [invoke_fused_moe_kernel_拆分与NPU等价替换方法.md](https://gitcode.com/skywang2/rtp-llm-porting/blob/main/docs/Qwen3.5/算子迁移梳理文档/invoke_fused_moe_kernel_拆分与NPU等价替换方法.md) — 拆分细节
