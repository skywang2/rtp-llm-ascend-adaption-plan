# rtp-llm Qwen3.5 MoE 层整体迁移方案

> 适配 rtp-llm 框架,核心 MoE 流水参考 vllm-ascend CANN 算子实现。

---

## 一、整体调用关系图

```
GenericMoeLayer.forward(hidden_states)                    ← 保留 rtp-llm 框架
│
├─ 1. self.gate(hidden_states)                           ← 保留 rtp-llm LinearFactory
│      → router_logits [M, E]
│
├─ 2. 路由选择                                            ← 替换为 CANN 算子（GroupTopK用于deepseek、kimi，暂不移植）
│      └─ torch_npu.npu_moe_gating_top_k(router_logits, k=top_k, norm_type=0)
│         → topk_weights [M, top_k], topk_ids [M, top_k]
│
├─ 3. FusedMoe.forward(hidden_states, topk_weights, topk_ids)  ← 复用,内部 router/executor 替换为 NPU 实现
│      │
│      ├─ 3.1 NpuRouter.prepare()                         ← 多卡分发
│      │      ├─ (TP) 不通信,直接透传
│      │      └─ (EP) torch_npu.npu_moe_distribute_dispatch_v2  ← 多卡 all_to_all
│      │
│      ├─ 3.2 token_dispatch                              ← Token 排序(NpuFusedExpertsExecutor 内部)
│      │      └─ torch_npu.npu_moe_init_routing_v2
│      │         → sorted_x, expanded_row_idx, group_list
│      │
│      ├─ 3.3 NpuFusedExpertsExecutor.execute()           ← GEMM 流水
│      │      ├─ GEMM1: torch_npu.npu_grouped_matmul(x, w1, group_list)
│      │      ├─ 激活:   torch_npu.npu_swiglu(gate_up_out)
│      │      ├─ GEMM2: torch_npu.npu_grouped_matmul(x, w2, group_list)
│      │      └─ 路由权重 + Token 合并: torch_npu.npu_moe_token_unpermute(expert_out, row_idx, probs=topk_weights)
│      │
│      └─ 3.4 NpuRouter.finalize()                        ← 多卡合并
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
| 路由选择 | `topkGatingSoftmaxKernelLauncher` | `npu_moe_gating_top_k` (norm_type 选 softmax/sigmoid) | GenericMoeLayer.forward |
| Token 排序 | `moe_align_block_size_torch` | `npu_moe_init_routing_v2` | fused_experts.execute |
| GEMM1 | `invoke_fused_moe_kernel` | `npu_grouped_matmul` | fused_experts.execute |
| 激活 | `silu_and_mul` (Triton) | `npu_swiglu` | fused_experts.execute |
| 路由权重 | 融合在 GEMM2 | 融合在后续`npu_moe_token_unpermute` | fused_experts.execute |
| GEMM2 | `invoke_fused_moe_kernel` | `npu_grouped_matmul` | fused_experts.execute |
| Token 合并 | `out.view(M,topk,K).sum(1)` | `npu_moe_token_unpermute` | fused_experts.execute |
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

### 3.1 复用 FusedMoe 容器(替换 self.fused_moe 成员)

> 不替换 `GenericMoeLayer` 外层框架,`self.gate`/`self.shared_expert`/`self.shared_expert_gate` 均保留。仅通过 `FusedMoeFactory` 注册 Ascend Strategy,让现成的 `FusedMoe` 容器内部改用 `NpuRouter` + `NpuFusedExpertsExecutor`。

#### 注册入口(fused_moe/__init__.py)

```python
elif device_type == DeviceType.Ascend:
    from .impl.ascend.strategy import (
        AscendBf16FallbackStrategy,
        AscendCannStrategy,        # ← 新增 CANN 策略
    )
    registry = StrategyRegistry()
    registry.register(AscendCannStrategy())           # 优先匹配
    registry.register(AscendBf16FallbackStrategy())    # 保留 fallback
    FusedMoeFactory.set_registry(registry)
```

#### Strategy(选择 Router + Executor 类)

```python
class AscendCannStrategy(MoeStrategy):
    """Ascend CANN MoE 策略:用 vllm-ascend CANN 算子替换 Triton kernel"""

    def get_attributes(self) -> StrategyAttributes:
        return StrategyAttributes(
            router_class=NpuRouter,                  # 替换 router
            executor_class=NpuFusedExpertsExecutor,   # 替换 fused_experts
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(not MoeConfigResolver().has_quantization(config))
```

#### NpuRouter(对应关系图 3.1 prepare + 3.4 finalize)

```python
class NpuRouter(FusedMoeDataRouter):
    """TP 场景 Router:prepare 不通信直接透传,finalize 走 all_reduce"""

    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker, config):
        resolver = MoeConfigResolver()
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))
        checker.check(not resolver.has_quantization(config))

    def __init__(self, config, quant_config):
        super().__init__(config, quant_config)
        self.tp_size = config.tp_size

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        # TP 场景:不通信,直接透传(token 排序在 executor 内部)
        return ExpertForwardPayload(
            expert_x=a1,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(self, payload, topk_weights, topk_ids, apply_router_weight_on_input, extra):
        output = payload.fused_expert_output
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
```

#### NpuFusedExpertsExecutor(对应关系图 3.2 token_dispatch + 3.3 GEMM 流水)

```python
class NpuFusedExpertsExecutor(FusedMoeExpertExecutor):
    """核心 GEMM 流水,对应 rtp-llm 的 invoke_fused_moe_kernel"""

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker, config):
        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        # 权重布局适配:[E, 2*inter, hidden] → [E, hidden, 2*inter]
        self.w1 = weights[W.moe_w1].transpose(1, 2)
        self.w2 = weights[W.moe_w2].transpose(1, 2)
        self.E = config.expert_num

    def execute(self, payload, activation, expert_map, a2_scale,
                apply_router_weight_on_input, extra_expert_args):
        x = payload.expert_x
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        # Token 排序 (CANN,一步融合)
        sorted_x, expanded_row_idx, group_list = torch_npu.npu_moe_init_routing_v2(
            x, topk_ids, expert_num=self.E
        )

        # GEMM1 (CANN)
        gate_up = torch_npu.npu_grouped_matmul(
            x=sorted_x, weight=self.w1, group_list=group_list, group_list_type=0
        )

        # 激活 (CANN)
        activated = torch_npu.npu_swiglu(gate_up)

        # GEMM2 (CANN)
        expert_out = torch_npu.npu_grouped_matmul(
            x=activated, weight=self.w2, group_list=group_list, group_list_type=0
        )

        # Token 合并 (CANN,融合 scatter + 乘路由权重)
        output = torch_npu.npu_moe_token_unpermute(
            expert_out, expanded_row_idx, probs=topk_weights
        )

        return CombineForwardPayload(fused_expert_output=output)
```

#### GenericMoeLayer 保持不变

`GenericMoeLayer.forward` 中调用 `self.fused_moe(...)` 的代码无需改动,`FusedMoeFactory` 在 NPU 上自动选择 `AscendCannStrategy`,创建的 `FusedMoe(NpuRouter, NpuFusedExpertsExecutor)` 即作为 `self.fused_moe` 注入。gate / shared_expert / shared_expert_gate 均沿用 rtp-llm 原实现。

### 3.2 EP 场景拓展点(当前非必需)

Qwen3.5 默认 TP 场景无需 EP。若未来启用 EP,在 `AscendCannStrategy` 旁新增 `AscendCannEpStrategy`,复用 `NpuFusedExpertsExecutor`,仅替换 Router:

- `NpuEpRouter.prepare`: `torch_npu.npu_moe_distribute_dispatch_v2`(all_to_all + 排序)
- `NpuEpRouter.finalize`: `torch_npu.npu_moe_distribute_combine_v2`(all_to_all + scatter + 乘权重)
- 差异点:`group_list` 改用 dispatch 返回的 `expert_token_nums`;`check_conditions` 判定 `ep_size > 1`

接口沿用 `FusedMoeDataRouter.prepare/finalize` 签名,实现细节待 EP 需求明确后补齐。

---

## 四、迁移步骤

### 步骤 1:权重布局适配

在 `NpuFusedExpertsExecutor.__init__` 中完成权重转置:

```python
# rtp-llm 权重布局 → CANN 要求的布局
self.w1 = weights[W.moe_w1].transpose(1, 2)  # [E, 2*inter, hidden] → [E, hidden, 2*inter]
self.w2 = weights[W.moe_w2].transpose(1, 2)  # [E, hidden, inter]  → [E, inter, hidden]
```

### 步骤 2:实现并注册 Ascend Strategy

按 3.1 实现 `AscendCannStrategy` + `NpuRouter` + `NpuFusedExpertsExecutor`,放入 `impl/ascend/` 目录,并在 `fused_moe/__init__.py` 的 Ascend 分支注册:

```python
elif device_type == DeviceType.Ascend:
    from .impl.ascend.strategy import AscendBf16FallbackStrategy, AscendCannStrategy
    registry = StrategyRegistry()
    registry.register(AscendCannStrategy())           # ← 新增
    registry.register(AscendBf16FallbackStrategy())    # 保留 fallback
    FusedMoeFactory.set_registry(registry)
```

注册后,`GenericMoeLayer.__init__` 中 `FusedMoeFactory().create_fused_moe(...)` 会自动创建 `FusedMoe(NpuRouter, NpuFusedExpertsExecutor)` 并作为 `self.fused_moe` 注入。**`GenericMoeLayer` 与 `GenericMoeDecoderLayer` 代码无需改动**。

### 步骤 3:多卡配置与验证

| 场景 | 配置 | 走的路径 |
|---|---|---|
| 单卡 | `tp_size=1, ep_size=1` | 纯本地,无通信 |
| TP | `tp_size>1, ep_size=1` | `npu_moe_init_routing_v2` + `all_reduce` |
| EP | `ep_size>1` | `npu_moe_distribute_dispatch_v2/combine_v2` |

验证:单卡对比 GPU 输出;TP 多卡 `all_reduce` 结果与单卡一致;性能对比 Triton kernel。

---

## 五、关键差异说明

| 维度 | rtp-llm 原始 | 迁移后 |
|---|---|---|
| 路由选择 | `topkGatingSoftmaxKernelLauncher` (C++) | `npu_moe_gating_top_k` (norm_type 选 softmax/sigmoid) |
| Token 排序 | `moe_align_block_size_torch` (argsort+cumsum) | `npu_moe_init_routing_v2` (一步融合) |
| GEMM | `invoke_fused_moe_kernel` (Triton) | `npu_grouped_matmul` (CANN) |
| 路由权重 | 融合在 GEMM2 epilogue | `npu_moe_token_unpermute(probs=)` 统一乘 |
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
