# RTP-LLM 移植 NPU MXFP8 量化适配详细设计 (v0.1)

> 本文档是 NPU MXFP8 量化适配的**可实施详细设计**，基于 [调用流程图](./RTP-LLM-MXFP8量化代码调用流程图-v0.1.md) 和 [跟读指南](./RTP-LLM移植NPU-MXFP8量化适配跟读指南-v0.1.md)。

---

## 一、概述

### 1.1 目标

将 RTP-LLM 的 FP8 BlockWise 量化移植到 NPU MXFP8，支持**静态量化**和**动态量化**两条路径。

### 1.2 核心替换

| 序号 | 工作项 | CUDA 实现 | NPU 替代 |
|------|--------|----------|---------|
| 1 | 权重量化 | `per_block_cast_to_fp8` | `torch_npu.npu_dynamic_mx_quant` |
| 2 | Linear GEMM | `deep_gemm.fp8_gemm_nt` | `torch_npu.npu_quant_matmul` |
| 3 | MoE GEMM | `deep_gemm.masked_fp8_gemm` | `torch_npu.npu_grouped_matmul_swiglu_quant_v2` |
| 4 | E8M0 转换 | `requant_weight_ue8m0` | **移除**（NPU 原生） |

### 1.3 范围

| 路径 | 预处理 | 配置层 | 权重层 | 推理层 |
|------|--------|--------|--------|--------|
| 静态量化 | ModelSlim | `ModelSlimConfig` ➕ | `ModelSlimWeight` ➕ | `NpuFp8MXFP8Linear` ➕ |
| 动态量化 | 无 | `Fp8BlockWiseQuantConfig` ✅ | `LoadQuantPerBlockFp8Weight` 🔄 | 同静态 |

---

## 二、Device 层适配

### 2.1 新增 DeviceType

**文件**：`rtp_llm/device/device_type.py` L7-13

```python
class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    ...
    Npu = 6  # ➕ 新增
```

### 2.2 新增 NpuImpl

**文件**：`rtp_llm/device/device_impl.py`

**设计**：继承 `GpuImpl`，重写 MXFP8 相关方法，不重写的方法自动继承。

```python
class NpuImpl(GpuImpl):
    """NPU 设备实现，参考 RocmImpl (L696) 的继承模式"""

    @property
    def arch(self) -> int:
        # NPU 架构版本，用于条件判断
        return 910  # Ascend 910

    def per_block_cast_to_fp8(self, weight, group_size=128):
        """🔄 替换：使用 NPU 原生 MXFP8 量化算子

        CUDA 原实现: per_block_cast_to_fp8(weight, group_size) → FP8 + FP32 scale
        NPU 替代: torch_npu.npu_dynamic_mx_quant → FP8 + E8M0 scale (原生)

        Args:
            weight: BF16/FP16 权重 tensor [M, N]
            group_size: block 大小 (128, NPU 固定)

        Returns:
            (weight_fp8, scale_e8m0)
            weight_fp8: torch.float8_e4m3fn [M, N]
            scale_e8m0: torch.float8_e8m0fnu [M//128, N//128]
        """
        weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(
            weight,
            torch.float8_e4m3fn  # 输出 dtype
        )
        return weight_fp8, scale

    def convert_fp8_weight_params(self, weight, weight_scale):
        """🔄 替换：NPU 无需 FP8 格式转换（e4m3fnuz 等）

        CUDA 原实现: ROCm 需要转 e4m3fnuz, scale × 2
        NPU: 直接返回，无需转换
        """
        return [weight, weight_scale]

    # 🗑️ 不实现以下方法（NPU 原生支持 E8M0 + MX 格式）
    # def requant_weight_ue8m0(self, weight, scale):  # 不需要
    # def swizzle_blockscale(self, scale):            # 不需要

    # ✅ 保留以下方法（其他量化方法使用，不影响 MXFP8）
    # apply_int8 (L215)
    # moe_apply_int8 (L224)
    # preprocess_groupwise_weight_params (L242)
    # convert_fp4_gemm_weight_params (L510)
```

### 2.3 注册 NpuImpl

**文件**：`rtp_llm/device/__init__.py` L11-23

```python
def get_device_cls(type: DeviceType):
    if type == DeviceType.Cuda:
        return CudaImpl
    elif type == DeviceType.ROCm:
        return RocmImpl
    elif type == DeviceType.Npu:        # ➕ 新增
        return NpuImpl
    ...
```

### 2.4 DeepGEMM Wrapper 适配

**文件**：`rtp_llm/models_py/kernels/cuda/deepgemm_wrapper.py`

```python
# 🗑️ NPU 场景下这些函数返回 False
@functools.cache
def has_deep_gemm() -> bool:
    if get_current_device().arch == 910:  # NPU
        return False
    return has_module("deep_gemm")

@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    if get_current_device().arch == 910:  # NPU
        return False  # NPU 原生 E8M0，不需要 DeepGEMM 的 E8M0 转换
    return torch.cuda.get_device_capability()[0] in [10, 12]
```

---

## 三、配置层适配

### 3.1 保留 Fp8BlockWiseQuantConfig

**文件**：`rtp_llm/config/quant_config.py` L376-404

```python
class Fp8BlockWiseQuantConfig(QuantizationConfig):
    # ✅ 保留，不修改核心逻辑
    DEFAULT_FP8_QUANT_BLOCK_SIZE = 128

    @classmethod
    def get_method(cls) -> str:
        return "FP8_PER_BLOCK"  # ✅ 保留

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"  # ✅ 保留

    def get_supported_compute_dtypes(self) -> List[torch.dtype]:
        # 🔄 验证 NPU 是否支持 torch.float8_e4m3fn
        return [torch.bfloat16]  # ✅ NPU 支持 BF16

    def get_supported_kv_cache_dtypes(self) -> List[torch.dtype]:
        # 🔄 验证 NPU KV Cache FP8 支持
        return [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
```

### 3.2 修改 load_from_ckpt — 新增 ModelSlim 检测

**文件**：`rtp_llm/config/quant_config.py` L99-273

在 `load_from_ckpt` 方法中，**在 smoothquant.ini 检测之后**新增 ModelSlim 检测分支：

```python
@classmethod
def load_from_ckpt(cls, ckpt_path: str) -> Optional["QuantizationConfig"]:
    # ... 现有 smoothquant.ini / pertensorquant.ini 检测 ...

    # ➕ 新增: ModelSlim 检测
    modelslim_config_path = os.path.join(ckpt_path, "quant_model_description.json")
    if os.path.exists(modelslim_config_path):
        with open(modelslim_config_path, "r") as f:
            quant_desc = json.load(f)
        return ModelSlimConfig.from_config(quant_desc)

    # ... 现有 config.json 检测 ...
```

### 3.3 新增 ModelSlimConfig

**文件**：`rtp_llm/config/quant_config.py` ➕ 新增

```python
class ModelSlimConfig(QuantizationConfig):
    """ModelSlim 预量化模型配置

    ModelSlim 输出 quant_model_description.json，包含每层量化类型:
    - W8A8: 权重和激活均为 INT8/FP8
    - W8A8_DYNAMIC: 动态激活量化
    - FLOAT: 未量化
    """

    def __init__(self, bits=8, group_size=128, is_quanted=True, **kwargs):
        super().__init__(bits=bits, group_size=group_size, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "modelslim"

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"  # MXFP8 算法标识

    def is_quanted(self) -> bool:
        return True  # 预量化模型

    @classmethod
    def from_config(cls, quant_desc: Dict[str, Any]) -> "ModelSlimConfig":
        """从 quant_model_description.json 解析配置

        Args:
            quant_desc: quant_model_description.json 的内容
                包含每层的量化类型 (W8A8, W4A8, FLOAT 等)

        Returns:
            ModelSlimConfig 实例
        """
        # 检查是否包含 MXFP8 层
        has_fp8 = any(
            "W8A8" in v or "MXFP8" in v
            for k, v in quant_desc.items()
            if isinstance(v, str) and k != "fa_quant_type"
        )
        if not has_fp8:
            return None  # 不包含 FP8 量化层

        return cls(bits=8, group_size=128, is_quanted=True)
```

### 3.4 注册 ModelSlimConfig

ModelSlimConfig 继承 `QuantizationConfig`，通过 `__init_subclass__` 自动注册到 `_registry`，`from_config` 和 `load_from_ckpt` 会自动匹配。

---

## 四、权重层适配

### 4.1 静态量化：修改 PerBlockFp8Weight._postprocess

**文件**：`rtp_llm/model_loader/per_block_fp8_quant_weight.py` L761-808

**改动**：移除 `requant_weight_ue8m0` 调用

```python
def _postprocess(self, tensor, device, load_config):
    processed_res = super()._postprocess(tensor, device, load_config)
    kernel_weight = processed_res[self.kernel.name]

    # ✅ NPU 场景: is_deep_gemm_e8m0_used() 返回 False, 走非 E8M0 路径
    if not is_deep_gemm_e8m0_used():
        kernel_weight = kernel_weight.reshape(
            kernel_weight.shape[-1], -1
        ) if kernel_weight.dim() == 2 else kernel_weight
    processed_res[self.kernel.name] = kernel_weight

    if self.scale is not None:
        scale_weight = processed_res[self.scale.name]

        if not is_deep_gemm_e8m0_used():
            scale_weight = scale_weight.reshape(
                scale_weight.shape[-1], -1
            ) if scale_weight.dim() == 2 else scale_weight

        # ✅ 保留: device 特定的权重重写
        kernel_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "weight", kernel_weight
        )
        scale_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "scale", scale_weight
        )

        # 🗑️ 移除这段: NPU 不需要 requant_weight_ue8m0
        # if is_deep_gemm_e8m0_used():
        #     kernel_weight, scale_weight = requant_weight_ue8m0(
        #         kernel_weight, scale_weight
        #     )

        processed_res[self.scale.name] = scale_weight
        processed_res[self.kernel.name] = kernel_weight

    return processed_res
```

**关键点**：
- NPU 场景下 `has_deep_gemm()` 和 `is_deep_gemm_e8m0_used()` 均返回 False
- 所以 NPU 走非 E8M0 路径，`requant_weight_ue8m0` 分支自然不执行
- 但建议**直接删除**这段死代码，保持代码清晰

### 4.2 动态量化：修改 LoadQuantPerBlockFp8Weight._load_raw_tensor

**文件**：`rtp_llm/model_loader/per_block_fp8_quant_weight.py` L854-886

**改动**：替换 `per_block_cast_to_fp8` 为 NPU 算子

```python
def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
    # 加载原始 BF16 权重
    kernel = self.kernel._load_raw_tensor(tensor_source, layer_id, device, load_config)

    res = {}
    scale = None
    if self.scale:
        # 🔄 NPU 替换: per_block_cast_to_fp8 → npu_dynamic_mx_quant
        if isinstance(load_config.exported_device, NpuImpl):
            # NPU 路径: 原生 MXFP8 量化, 直接输出 E8M0 scale
            quant_kernel, scale = torch_npu.npu_dynamic_mx_quant(
                kernel.get(self.kernel.name),
                torch.float8_e4m3fn
            )
        else:
            # CUDA 路径: 保留原实现
            quant_kernel, scale = per_block_cast_to_fp8(
                kernel.get(self.kernel.name), self.group_size
            )

        if quant_kernel.dim() == 2:
            scale = scale.reshape([scale.shape[0], -1])
    else:
        quant_kernel = cast_to_fp8(kernel.get(self.kernel.name))

    # 转置 (非 MoE)
    if self.kernel.name == W.moe_w1 or self.kernel.name == W.moe_w2:
        pass
    elif quant_kernel.dim() == 2:
        quant_kernel = quant_kernel.T

    res = {self.kernel.name: quant_kernel.contiguous().to(device)}
    if self.scale:
        scale = scale.T if scale.dim() == 2 else scale
        res.update({self.scale.name: scale.contiguous().to(device)})

    return res
```

**关键点**：
- NPU 的 `npu_dynamic_mx_quant` 直接输出 `float8_e8m0fnu` 格式的 scale
- 不需要后续的 `requant_weight_ue8m0` 转换
- CUDA 路径保持不变，通过 `isinstance(NpuImpl)` 分流

### 4.3 新增 ModelSlimWeight

**文件**：`rtp_llm/model_loader/modelslim_weight.py` ➕ 新建

```python
from rtp_llm.model_loader.weight_module import CompositeWeight, QuantWeight
from rtp_llm.model_loader.atomic_weight import AtomicWeight
import torch_npu


class ModelSlimWeight(CompositeWeight, QuantWeight):
    """ModelSlim 预量化权重加载器

    与 PerBlockFp8Weight 的区别:
    - ModelSlim 权重已是 FP8 + E8M0 scale 格式
    - 无需 requant_weight_ue8m0 转换
    - 无需 NZ 格式转换 (FP8 不需要)
    """

    # 权重名称映射 (参考 PerBlockFp8Weight.w8a8_weight_list)
    modelslim_weight_list: Dict[str, str] = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w1: W.ffn_s1,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w3: W.ffn_s3,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    @classmethod
    def support(cls, quant_config, src_weight_info) -> bool:
        if not isinstance(quant_config, ModelSlimConfig):
            return False
        if not quant_config.is_quanted():
            return False
        name = src_weight_info.name
        return name in cls.modelslim_weight_list

    def __init__(self, src_weight_info, quant_config, *args, **kwargs):
        self.group_size = quant_config.group_size()
        # 创建 kernel 和 scale 子权重
        params = src_weight_info.extract_params(...)
        kernel = AtomicWeight(name=src_weight_info.name, data_type=torch.float8_e4m3fn, ...)
        sub_weights = {kernel.name: kernel}

        scale_name = self.modelslim_weight_list.get(src_weight_info.name)
        if scale_name:
            scale = AtomicWeight(name=scale_name, data_type=torch.float8_e8m0fnu, ...)
            sub_weights[scale.name] = scale

        CompositeWeight.__init__(self, sub_weights, quant_config=quant_config, ...)
        self.kernel = kernel
        self.scale = scale

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        """直接加载 ModelSlim 预量化的 FP8 权重"""
        kernel = self.kernel._load_raw_tensor(tensor_source, layer_id, device, load_config)
        res = {self.kernel.name: kernel.get(self.kernel.name)}

        if self.scale:
            scale = self.scale._load_raw_tensor(tensor_source, layer_id, device, load_config)
            res[self.scale.name] = scale.get(self.scale.name)

        return res

    def _postprocess(self, tensor, device, load_config):
        """后处理: 仅 reshape, 无需 requant"""
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]

        # reshape 为 (N, K) 布局
        if kernel_weight.dim() == 2:
            kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1)
        processed_res[self.kernel.name] = kernel_weight

        if self.scale is not None:
            scale_weight = processed_res[self.scale.name]
            if scale_weight.dim() == 2:
                scale_weight = scale_weight.reshape(scale_weight.shape[-1], -1)
            processed_res[self.scale.name] = scale_weight

        # 🗑️ 无需 requant_weight_ue8m0 — ModelSlim 已输出 E8M0 格式
        # 🗑️ 无需 NZ 格式转换 — FP8 不需要

        return processed_res
```

---

## 五、推理层适配

### 5.1 Factory 选择机制

**文件**：`rtp_llm/models_py/modules/factory/linear/__init__.py` L19-27

```python
# ➕ 新增 NPU 分支
if device_type == DeviceType.Cuda:
    from .impl.cuda import *
elif device_type == DeviceType.Npu:
    from .impl.ascend import *  # ➕
```

**目录结构**：

```
models_py/modules/factory/linear/impl/
├── cuda/
│   ├── __init__.py
│   ├── fp8_deepgemm_linear.py      # CUDA DeepGEMM (NPU 不使用)
│   ├── fp8_gemm_linear.py          # CUDA 入口策略
│   └── f16_linear.py               # 基线
└── ascend/                          # ➕ 新建
    ├── __init__.py
    ├── fp8_mxfp8_linear.py          # NPU MXFP8 Linear
    └── f16_linear.py                # NPU F16 基线
```

### 5.2 新增 NpuFp8MXFP8Linear

**文件**：`rtp_llm/models_py/modules/factory/linear/impl/ascend/fp8_mxfp8_linear.py` ➕ 新建

**CUDA 原实现**：`fp8_deepgemm_linear.py` L171-236

```python
import torch
import torch_npu
from rtp_llm.models_py.modules.factory.linear.linear import Linear


class NpuFp8MXFP8Linear(Linear):
    """NPU MXFP8 Linear 层

    替换 CUDA CudaFp8DeepGEMMLinear (fp8_deepgemm_linear.py L171)

    CUDA 原流程:
        1. sgl_per_token_group_quant_fp8(input) → 在线量化输入
        2. deep_gemm.fp8_gemm_nt(input_fp8, weight_fp8, scales) → GEMM

    NPU 新流程:
        1. torch_npu.npu_dynamic_mx_quant(input) → 在线量化输入
        2. torch_npu.npu_quant_matmul(input_fp8, weight_fp8, scale) → GEMM
    """

    @classmethod
    def can_handle(cls, quant_config, weight, weight_scales, **kwargs) -> bool:
        """匹配条件: FP8 BlockWise 量化 + FP8 权重"""
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False
        # 匹配 Fp8BlockWiseQuantConfig 和 ModelSlimConfig
        method = quant_config.get_method()
        return method in ("FP8_PER_BLOCK", "modelslim")

    def __init__(self, weight, weight_scales, bias=None, **kwargs):
        super().__init__(weight=weight, weight_scales=weight_scales, bias=bias)
        self.weight = weight          # FP8 E4M3 [N, K]
        self.weight_scale = weight_scales  # E8M0 [N//128, K//128]
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """MXFP8 GEMM 前向

        Args:
            input: BF16 [M, K] 或 FP8 [M, K]

        Returns:
            BF16 output [M, N]
        """
        M, K = input.shape

        # 1. 在线量化输入 (如果输入是 BF16)
        if input.dtype == torch.bfloat16:
            input_fp8, input_scale = torch_npu.npu_dynamic_mx_quant(
                input,
                torch.float8_e4m3fn
            )
        elif input.dtype == torch.float8_e4m3fn:
            input_fp8 = input
            # 构造全 1.0 的 scale (E8M0 格式)
            input_scale = torch.ones(
                M // 128, K // 128,
                dtype=torch.float8_e8m0fnu,
                device=input.device
            )
        else:
            raise ValueError(f"Unsupported input dtype: {input.dtype}")

        # 2. NPU MXFP8 GEMM
        output = torch_npu.npu_quant_matmul(
            input_fp8,           # [M, K] FP8
            self.weight,         # [N, K] FP8
            self.weight_scale,   # [N//128, K//128] E8M0
            dtype=torch.float8_e4m3fn
        )

        # 3. 加 bias
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output  # [M, N] BF16
```

### 5.3 新增 NpuMoEMXFP8Executor

**文件**：`rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors/npu_mxfp8_executor.py` ➕ 新建

**CUDA 原实现**：`deepgemm_masked_executor.py` L147-351

```python
import torch
import torch_npu


class NpuMoEMXFP8Executor:
    """NPU MXFP8 MoE Executor

    替换 CUDA DeepGemmMaskedExecutor (deepgemm_masked_executor.py)

    CUDA 原流程:
        1. m_grouped_fp8_gemm_nt_masked(expert_x, w1) → upgate_output BF16
        2. silu_mul_masked_fp8_post_quant_fwd(upgate) → down_input FP8
        3. m_grouped_fp8_gemm_nt_masked(down_input, w2) → down_output BF16

    NPU 新流程:
        1. npu_grouped_matmul_swiglu_quant_v2 融合 GEMM + SwiGLU + 量化
        (或分步执行，取决于算子支持)
    """

    @classmethod
    def check_conditions(cls, checker, config):
        """条件检查"""
        # 检查 NPU 可用
        checker.check(torch.npu.is_available())
        # 检查 BF16 模式
        checker.check(resolver.is_bf16(config))
        # 量化方法匹配
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK", "modelslim"])

    def __init__(self, config, quant_config, weights):
        self._w1 = weights[W.moe_w1]       # [E, N, K] FP8
        self._w2 = weights[W.moe_w2]       # [E, K, N//2] FP8
        self._E, self._N, self._K = self._w1.size()
        self._w1_scale = weights.get(W.moe_s1, None)  # E8M0
        self._w2_scale = weights.get(W.moe_s2, None)  # E8M0
        self._use_fp8 = quant_config.is_quanted() or not quant_config.is_quanted()
        # (静态和动态都使用 FP8 推理)

    def execute(self, payload, activation, expert_map, **kwargs):
        if self._use_fp8:
            return self._execute_fp8(payload, **kwargs)
        else:
            return self._execute_bf16(payload, **kwargs)

    def _execute_fp8(self, payload, expert_x_scale=None, **kwargs):
        expert_x = payload.expert_x           # [E, M, K] FP8
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        expected_m = min(M, payload.expert_tokens_meta.expected_m)

        # ======== 方案 A: 融合算子 (优先) ========
        try:
            down_output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x=expert_x,                    # [E, M, K] FP8
                weight=self._w1,               # [E, N, K] FP8
                weight_scale=self._w1_scale,   # [E, N//128, K//128] E8M0
                x_scale=expert_x_scale,       # [E, M//128, K//128] E8M0
                dtype=torch.float8_e4m3fn,
                # 融合 GEMM + SwiGLU + 量化
            )
            # 还需第二次 GEMM (down)
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x=down_output,
                weight=self._w2,
                weight_scale=self._w2_scale,
                dtype=torch.float8_e4m3fn,
            )
        # ======== 方案 B: 分步执行 (降级) ========
        except Exception:
            # Step 1: Gate-Up GEMM
            upgate_output = torch_npu.npu_grouped_matmul_quant(
                expert_x, self._w1, self._w1_scale,
                dtype=torch.float8_e4m3fn
            )
            # Step 2: SiLU + Mul + 量化
            down_input, down_input_scale = self._silu_mul_and_quant(upgate_output)
            # Step 3: Down GEMM
            output = torch_npu.npu_grouped_matmul_quant(
                down_input, self._w2, self._w2_scale,
                dtype=torch.float8_e4m3fn
            )

        return CombineForwardPayload(fused_expert_output=output)

    def _silu_mul_and_quant(self, upgate_output):
        """SiLU + Mul + FP8 量化"""
        # 分割 gate 和 up
        gate, up = upgate_output.chunk(2, dim=-1)
        # SiLU
        act = torch.nn.functional.silu(gate) * up
        # 量化
        act_fp8, act_scale = torch_npu.npu_dynamic_mx_quant(
            act, torch.float8_e4m3fn
        )
        return act_fp8, act_scale
```

### 5.4 MoE Strategy 注册

**文件**：`rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/__init__.py` ➕ 新建

```python
from .executors.npu_mxfp8_executor import NpuMoEMXFP8Executor
from .strategy.npu_mxfp8 import NpuMXFP8Strategy

# 注册策略
FusedMoEFactory.register(NpuMXFP8Strategy)
```

---

## 六、桥接层适配

### 6.1 可移除的 CUDA Kernel

| CUDA Kernel | 文件 | 处理方式 |
|-------------|------|---------|
| `scaled_fp8_quant.cu` | `bindings/cuda/kernels/` | 🗑️ 移除，NPU 用 `npu_dynamic_mx_quant` |
| `librtp_compute_ops` | C++ 扩展 | 🗑️ 移除，NPU 原生算子替代 |
| `mla_quant_kernel.cu` | `bindings/cuda/kernels/` | 🔄 ACLNN 重写 (独立工作项) |

### 6.2 可移除的第三方库

| 库 | 用途 | 处理方式 |
|----|------|---------|
| DeepGEMM | FP8 BlockWise GEMM | 🗑️ 移除，`npu_quant_matmul` 替代 |
| `torch._scaled_mm` | FP8 PerTensor GEMM | 🗑️ MXFP8 场景移除 |

### 6.3 可保留的第三方库

| 库 | 用途 | 说明 |
|----|------|------|
| FlashInfer | FP8 PerChannel / FP4 | 非 MXFP8 场景保留 |
| CUTLASS | INT8 / FP8 GEMM | 非 MXFP8 场景保留 |

### 6.4 fp8_kernel.py 函数处理

**文件**：`rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py`

| 函数 | 行号 | NPU 处理 |
|------|------|---------|
| `per_block_cast_to_fp8` | L329 | 🔄 NPU 不调用 (用 `npu_dynamic_mx_quant`) |
| `requant_weight_ue8m0` | L374 | 🗑️ 移除 (NPU 原生 E8M0) |
| `quant_weight_ue8m0` | L348 | 🗑️ 移除 |
| `ceil_to_ue8m0` | L51 | 🗑️ 移除 (NPU 原生) |
| `_transform_scale_ue8m0` | L56 | 🗑️ 移除 (NPU 原生) |
| `sgl_per_token_group_quant_fp8` | L110 | 🔄 NPU 用 `npu_dynamic_mx_quant` |

---

## 七、通信层适配

### 7.1 分布式通信后端

| 文件 | 行号 | 修改 |
|------|------|------|
| `collective_torch.py` | L586 | `backend="nccl"` → `backend="hccl"` |
| `backend_manager.py` | L50 | `backend="nccl"` → `backend="hccl"` |

### 7.2 验证

```python
import torch.distributed as dist
dist.init_process_group(backend="hccl")
# 验证 HCCL 可用
```

---

## 八、验证方案

### 8.1 单层验证

#### Device 层

```python
from rtp_llm.device import get_current_device, DeviceType

# 验证 NpuImpl 注册
device = get_current_device()
assert isinstance(device, NpuImpl)
assert device.arch == 910

# 验证 per_block_cast_to_fp8
weight = torch.randn(128, 128, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = device.per_block_cast_to_fp8(weight)
assert weight_fp8.dtype == torch.float8_e4m3fn
assert scale.dtype == torch.float8_e8m0fnu  # E8M0
```

#### 配置层

```python
# 验证 load_from_ckpt 检测 ModelSlim
quant_config = QuantizationConfig.load_from_ckpt("/path/to/modelslim_model")
assert isinstance(quant_config, ModelSlimConfig)
assert quant_config.is_quanted() == True
```

#### 权重层

```python
# 验证动态量化
weight = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(weight, torch.float8_e4m3fn)
assert weight_fp8.dtype == torch.float8_e4m3fn
assert scale.dtype == torch.float8_e8m0fnu
assert weight_fp8.shape == weight.shape
```

#### 推理层

```python
# 验证 Linear
x = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu")
weight_fp8 = torch.randn(4096, 4096, dtype=torch.float8_e4m3fn, device="npu")
scale = torch.ones(32, 32, dtype=torch.float8_e8m0fnu, device="npu")

output = torch_npu.npu_quant_matmul(x, weight_fp8, scale, dtype=torch.float8_e4m3fn)
assert output.dtype == torch.bfloat16
assert output.shape == (128, 4096)
```

### 8.2 端到端验证

```python
# 1. 动态量化启动
# python rtp_llm_cli.py --model_name <model> --quantization FP8_PER_BLOCK --device npu

# 2. 静态量化启动 (ModelSlim 预量化模型)
# python rtp_llm_cli.py --model_name <modelslim_model> --device npu

# 3. 验证推理结果
# 对比 BF16 基线输出与 MXFP8 输出的差异
# 允许误差: cos_sim > 0.99, max_diff < 0.1
```

### 8.3 性能验证

| 指标 | BF16 基线 | MXFP8 期望 | 验证方法 |
|------|----------|-----------|---------|
| 显存占用 | 1× | ~0.5× | `torch.npu.memory_allocated()` |
| 推理速度 | 1× | ≥1.5× | 时间测量 |
| 精度损失 | 0 | <1% | cos_sim 对比 |

---

## 九、适配优先级

### P0（必须，阻塞推理）

1. **推理层 GEMM 替换**：`NpuFp8MXFP8Linear` + `npu_quant_matmul`
2. **权重层算子替换**：`npu_dynamic_mx_quant` (动态量化路径)
3. **Device 层**：`NpuImpl` 注册

### P1（简化代码）

4. **移除 `requant_weight_ue8m0`**：NPU 原生 E8M0
5. **移除 DeepGEMM 依赖**：`has_deep_gemm()` 返回 False
6. **移除 CUDA kernel**：`scaled_fp8_quant.cu`

### P2（扩展功能）

7. **ModelSlim 预量化支持**：`ModelSlimConfig` + `ModelSlimWeight`
8. **MoE MXFP8**：`NpuMoEMXFP8Executor` + `npu_grouped_matmul_swiglu_quant_v2`
9. **通信层**：NCCL → HCCL

### P3（独立工作项）

10. **MLA KV Cache 量化**：ACLNN 重写
