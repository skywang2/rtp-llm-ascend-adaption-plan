# RTP-LLM 移植 NPU MXFP8 量化适配概要 (v0.1)

> 本文档聚焦于将 RTP-LLM 的 **FP8 BlockWise 量化**移植到 NPU 的 **MXFP8** 方案，是 [RTP-LLM移植NPU量化适配概要](./RTP-LLM移植NPU量化适配概要-v0.1.md) 的精简版。

---

## 一、目标与范围

### 1.1 量化目标

| 维度 | RTP-LLM 原方案 | NPU 目标方案 |
|------|--------------|-------------|
| 量化方法 | FP8 BlockWise (128x128) | MXFP8 (128x128) |
| 权重格式 | FP8 E4M3 + 手动 E8M0 scale | FP8 E4M3 + 原生 E8M0 scale |
| 推理算子 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（MXFP8） |
| 量化工具 | Load Quant（框架内） | Load Quant + ModelSlim（预量化） |

### 1.2 为什么选择 MXFP8

1. **RTP-LLM 已有 FP8 BlockWise 实现**，MXFP8 是其 NPU 原生等价方案
2. **算法等价**：两者都是 128x128 block-wise FP8 量化，精度一致
3. **NPU 原生支持**：`torch_npu.npu_dynamic_mx_quant` 原生支持 E8M0 scale
4. **适配工作量最小**：相比 INT4/INT8 方案，MXFP8 改动最少

### 1.3 不在范围内

- INT8 / INT4 / FP4 等其他量化方法（见完整版概要）
- 激活动态量化（NPU 特有，后续引入）
- MLA KV Cache 量化（独立工作项）

---

## 二、适配工作总览

MXFP8 移植涉及 **4 个层面**：

| 层面 | 当前实现 | NPU 目标 | 工作量 |
|------|---------|---------|--------|
| 配置层 | `Fp8BlockWiseQuantConfig` | 保留，调整 dtype | 低 |
| 权重层 | `per_block_cast_to_fp8` + `requant_weight_ue8m0` | `npu_dynamic_mx_quant`，移除手动 E8M0 | 中 |
| 推理层 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（MXFP8） | 中 |
| 桥接层 | `scaled_fp8_quant.cu` | 移除或用 `torch_npu` 替代 | 低 |

---

## 三、配置层适配

### 3.1 保留的配置类

```python
# rtp_llm/config/quant_config.py
class Fp8BlockWiseQuantConfig(QuantizationConfig):
    # 保留，仅需调整以下接口
```

### 3.2 需调整的接口

| 接口 | 当前实现 | NPU 适配 |
|------|---------|---------|
| `get_method()` | `FP8_PER_BLOCK` | 保留 |
| `is_quanted()` | `False`（Load Quant） | 保留 |
| `get_supported_compute_dtypes()` | `[torch.float16, torch.bfloat16]` + FP8 | 调整为 NPU 支持的 dtype |

### 3.3 可保留的部分

- `QuantizationConfig` 基类与注册表机制
- `is_quanted` 二分逻辑
- `from_config` / `load_from_ckpt` 入口

---

## 四、权重层适配

### 4.1 `_postprocess` 算子替换

| CUDA 算子 | 功能 | NPU 替代方案 |
|-----------|------|-------------|
| `per_block_cast_to_fp8` | FP8 BlockWise 在线量化 | `torch_npu.npu_dynamic_mx_quant`（FP8 + E8M0） |
| `requant_weight_ue8m0` | E8M0 scale 重量化 | **移除**（NPU 原生支持 E8M0） |
| `convert_fp8_weight_params` | FP8 权重格式转换 | **移除或简化** |

### 4.2 关键改动：移除手动 E8M0 转换

**RTP-LLM 当前实现**（CUDA）：
```python
# 需要手动将 FP32 scale 转为 E8M0 格式
weight_fp8, weight_scale = per_block_cast_to_fp8(weight)
weight_scale = requant_weight_ue8m0(weight, weight_scale)  # ← 手动转换
output = deepgemm_fp8_gemm(input, weight_fp8, weight_scale)
```

**NPU 适配后**：
```python
# NPU 原生支持 E8M0，无需手动转换
weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, dtype=torch.float8_e4m3fn)
# weight_scale 直接是 E8M0 格式，无需 requant
output = torch_npu.npu_quant_matmul(input, weight_fp8, weight_scale)
```

### 4.3 可保留的部分

- `WeightModule` 四步加载流程
- `CompositeWeight` / `QuantWeight` 类层次
- TP 切分策略框架

---

## 五、推理层适配

### 5.1 量化 GEMM 替换

| 推理模块 | CUDA 实现 | NPU 替代 |
|---------|----------|---------|
| FP8 BlockWise Linear | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（MXFP8） |
| FP8 BlockWise MoE | DeepGEMM masked executor | `torch_npu.npu_grouped_matmul`（MXFP8） |

### 5.2 代码改动示例

**RTP-LLM 当前实现**（CUDA）：
```python
# fp8_deepgemm_linear.py
class Fp8DeepGemmLinear(Linear):
    def forward(self, x):
        # DeepGEMM FP8 BlockWise GEMM
        output = deepgemm.fp8_gemm_nt(
            x_fp8, self.weight_fp8,
            x_scale, self.weight_scale  # E8M0 scale
        )
        return output
```

**NPU 适配后**：
```python
# ascend/fp8_mxfp8_linear.py
class Fp8MxFp8Linear(Linear):
    def forward(self, x):
        # NPU MXFP8 GEMM
        output = torch_npu.npu_quant_matmul(
            x_fp8, self.weight_fp8,
            x_scale, self.weight_scale,  # 原生 E8M0 scale
            dtype=torch.float8_e4m3fn
        )
        return output
```

### 5.3 目录结构

```
models_py/modules/factory/
├── linear/impl/
│   ├── cuda/
│   │   └── fp8_deepgemm_linear.py      # 现有 CUDA 实现
│   └── ascend/                          # 新增 NPU 实现
│       └── fp8_mxfp8_linear.py
├── fused_moe/impl/
│   ├── cuda/
│   └── ascend/
```

---

## 六、桥接层适配

### 6.1 需处理的 CUDA Kernel

| CUDA Kernel | 处理方式 |
|-------------|---------|
| `scaled_fp8_quant.cu` | **移除**，用 `torch_npu.npu_dynamic_mx_quant` 替代 |
| `librtp_compute_ops`（`per_block_cast_to_fp8` 等） | **移除**，NPU 原生算子替代 |

### 6.2 第三方库依赖

| 第三方库 | 处理方式 |
|---------|---------|
| DeepGEMM | **移除**，用 `torch_npu.npu_quant_matmul` 替代 |
| `torch._scaled_mm` | **移除**（MXFP8 不需要） |

---

## 七、ModelSlim 预量化支持（可选）

除了 Load Quant（框架内运行时量化），还可支持 ModelSlim 预生成的 MXFP8 模型：

### 7.1 新增配置类

```python
class ModelSlimConfig(QuantizationConfig):
    @classmethod
    def get_method(cls):
        return "modelslim"

    def is_quanted(self):
        return True  # 预量化模型
```

### 7.2 新增权重加载器

```python
class ModelSlimWeight(QuantWeight):
    def _postprocess(self, tensor, param_info, quant_config, ...):
        # 1. 加载 ModelSlim MXFP8 权重
        # 2. Scale 参数处理（E8M0 格式）
        # 3. 无需 NZ 格式转换（FP8 不需要）
```

---

## 八、适配优先级

### P0（必须，阻塞推理）

1. **推理层 GEMM 替换**：DeepGEMM → `torch_npu.npu_quant_matmul`（MXFP8）
2. **权重层算子替换**：`per_block_cast_to_fp8` → `npu_dynamic_mx_quant`

### P1（简化代码）

3. **移除手动 E8M0 转换**：删除 `requant_weight_ue8m0` 调用
4. **移除 DeepGEMM 依赖**：删除 `fp8_deepgemm_linear.py` 中的 CUDA 调用
5. **移除桥接层 CUDA kernel**：`scaled_fp8_quant.cu`

### P2（扩展功能）

6. **ModelSlim 预量化支持**：新增 `ModelSlimConfig` + `ModelSlimWeight`
7. **配置层 dtype 调整**：`get_supported_compute_dtypes`

---

## 九、总结

MXFP8 移植的核心工作是 **3 个替换 + 1 个移除**：

| 序号 | 工作项 | 具体内容 |
|------|--------|---------|
| 1 | **替换推理 GEMM** | DeepGEMM `fp8_gemm_nt` → `torch_npu.npu_quant_matmul` |
| 2 | **替换权重量化** | `per_block_cast_to_fp8` → `npu_dynamic_mx_quant` |
| 3 | **移除手动 E8M0** | 删除 `requant_weight_ue8m0`（NPU 原生支持） |
| 4 | **移除第三方依赖** | DeepGEMM / `scaled_fp8_quant.cu` |

**优势**：MXFP8 与 RTP-LLM 的 FP8 BlockWise 算法等价，精度一致，适配工作量最小，是最优先落地的 NPU 量化方案。
