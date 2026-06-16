# AscendC 自定义算子接入 rtp-llm 框架指南

## 概述

本文档说明如何将一个**全新的** AscendC 自定义算子（含 `op_host`、`op_kernel`、`op_api` 三层代码）从零接入 rtp-llm 推理框架的生产代码中。

**适用场景**：算子未在任何其他框架中复用，需要从编写源码开始，逐步完成 CMake 构建集成、C++ 封装，最终在 `CudaSampleOp.cc` 中调用。

**参考实现**：`applyTopKTopP` 算子（位于 `3rdparty/aclnn_custom_ops/apply_top_k_top_p_custom/`）。

**CMake 构建方式**：框架采用**注册式 CMake**——算子无需编写 `add_library` / `add_executable`，只需通过 `target_sources()` 将源文件注入顶层预定义好的库目标（`opapi`、`optiling`、`op_host_aclnnExc`）。顶层 `CMakeLists.txt` 通过 `op_add_subdirectory()` 自动发现 `**/op_host/CMakeLists.txt` 并注册算子。

---

## 第一步：编写算子源码

### 1.1 目录结构

在 `3rdparty/aclnn_custom_ops/` 下创建算子子目录，遵循以下结构：

```
3rdparty/aclnn_custom_ops/
├── CMakeLists.txt                      # 源项目顶层 CMake（复用现有，无需修改）
├── build.sh                            # 源项目构建脚本（复用现有）
├── build_for_bazel.sh                  # Bazel genrule 入口（无需修改）
├── cmake/                              # CMake 模块（复用现有）
│   ├── config.cmake                    #   工具链检测
│   ├── func.cmake                      #   算子注册宏定义
│   └── intf.cmake / intf_pub.cmake     #   接口库定义
├── utils/                              # CANN 工具库（tiling/proto headers，复用）
├── <your_op_name>/                     # ★ 新建
│   ├── op_host/                        # ★ 新建：Host 侧 ACLNN 注册代码
│   │   ├── CMakeLists.txt              # ★ 新建：算子注册 CMake
│   │   ├── <op_name>.cpp               #   Host 主逻辑（op_api）
│   │   ├── <op_name>.h                 #   Host 主逻辑头文件
│   │   ├── aclnn_<op_name>.cpp         #   ACLNN 封装层
│   │   ├── aclnn_<op_name>.h           #   ACLNN C 接口声明
│   │   ├── <op_name>_def.cpp           #   算子定义/注册
│   │   ├── <op_name>_tiling.cpp        #   Tiling 参数计算
│   │   └── <op_name>_tiling.h          #   Tiling 头文件
│   └── op_kernel/                      # ★ 新建：AscendC Device 侧 kernel
│       ├── <op_name>.cpp               #   AscendC kernel 实现
│       └── <op_name>.h                 #   AscendC kernel 头文件
└── BUILD                               # Bazel genrule（无需修改外层 BUILD）
```

### 1.2 Device 侧 Kernel（op_kernel）

AscendC kernel 是运行在 NPU 上的计算核心，使用 AscendC API 编写。

**文件**：`op_kernel/<op_name>.h`、`op_kernel/<op_name>.cpp`

核心结构：
```cpp
// <op_name>.h
#pragma once
#include "kernel_operator.h"

class Kernel<OpName> {
public:
    __aicore__ inline Kernel<OpName>() {}
    __aicore__ inline void Init(/* AscendC tensor 参数 */) { /* 初始化 */ }
    __aicore__ inline void Process() { /* 计算逻辑 */ }
private:
    // AscendC 中间张量、tiling 参数等
};
```

参考：[apply_top_k_top_p_custom.h](../3rdparty/aclnn_custom_ops/apply_top_k_top_p_custom/op_kernel/apply_top_k_top_p_custom.h)

### 1.3 Host 侧代码（op_host）

Host 侧代码运行在 CPU 上，负责：参数校验、与 CANN 交互、调度 kernel 执行。

**需要新建的文件**（参考 [apply_top_k_top_p_custom/op_host/](../3rdparty/aclnn_custom_ops/apply_top_k_top_p_custom/op_host/) 目录）：

| 文件 | 职责 | 注册到 |
|------|------|--------|
| `<op_name>.cpp` + `.h` | Host 主逻辑：参数校验、tiling 计算、调用 kernel | `opapi` |
| `aclnn_<op_name>.cpp` + `.h` | ACLNN C 接口封装（`GetWorkspaceSize` + 执行函数） | `opapi` |
| `<op_name>_def.cpp` | 算子元信息定义（输入/输出/Tiling 描述） | `op_host_aclnnExc` |
| `<op_name>_tiling.cpp` + `.h` | Tiling 参数计算逻辑 | `optiling` |

### 1.4 算子注册 CMakeLists.txt

**这是最关键的一步。** 框架使用注册式 CMake，算子**不需要**写 `add_library` / `add_executable`。

文件位置：`op_host/CMakeLists.txt`

以 `applyTopKTopP` 为例（参考文件：[CMakeLists.txt](../3rdparty/aclnn_custom_ops/apply_top_k_top_p_custom/op_host/CMakeLists.txt)）：

```cmake
# 1. 编译选项（可选）
add_ops_compile_options(
    OP_NAME <OpName>              # 必须首字母大写，与算子名对应
    OPTIONS --cce-auto-sync=on    # AscendC 编译选项
    -Wno-deprecated-declarations
    -Werror
)

# 2. 注册算子定义 → op_host_aclnnExc 目标
#    <op_name>_def.cpp 包含算子输入/输出/Tiling 的元信息描述
target_sources(op_host_aclnnExc PRIVATE
    <op_name>_def.cpp
)

# 3. 注册 op_api 源码 → opapi 目标
#    <op_name>.cpp       — Host 主逻辑（参数校验、调用 kernel）
#    aclnn_<op_name>.cpp — ACLNN C 接口封装
target_sources(opapi PRIVATE
    <op_name>.cpp
    aclnn_<op_name>.cpp
)

# 4. 注册 Tiling 源码 → optiling 目标
target_sources(optiling PRIVATE
    <op_name>_tiling.cpp
)

# 5. Tiling 头文件 include 路径
target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 6. 安装 ACLNN C 头文件（CPack 打包用）
file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_<op_name>.h")
install(FILES ${_GMM_Aclnn_header}
    DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
```

**关键说明**：
- 四个目标 `op_host_aclnnExc`、`opapi`、`optiling`、`op_host_aclnn` 由顶层 `CMakeLists.txt` 定义，算子只需 `target_sources` 注册
- 顶层 `CMakeLists.txt` 第 46 行通过 `file(GLOB OP_HOST_CMAKE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/**/op_host/CMakeLists.txt")` 自动发现所有算子——只需在 `**/op_host/` 下放好 `CMakeLists.txt` 即可
- **不需要修改顶层** `CMakeLists.txt`，**不需要添加** `add_subdirectory()`
- `OP_NAME` 必须与 `*_def.cpp` 中注册的算子名严格一致（首字母大写驼峰）

---

## 第二步：修改 Bazel 构建脚本

### 2.1 `build_for_bazel.sh`

该脚本是 Bazel `genrule` 的入口，负责：
1. 定位 CANN 工具链并加载编译环境
2. 调用源项目 `build.sh` 完成 CMake 编译
3. 从 CPack 产物中收集 `.so` 和 `opp/` 到 Bazel 输出目录

**需要修改的行**（文件位置：[build_for_bazel.sh](../3rdparty/aclnn_custom_ops/build_for_bazel.sh)）：

无需修改 —— 脚本接受算子名称作为参数（`$2`），多算子用分号分隔：
```bash
OP_NAMES="${2:-ALL}"
bash "${SCRIPT_DIR}/build.sh" -n "${OP_NAMES}" -c "${SOC_VERSION}"
```

### 2.2 `BUILD` 文件

修改 `3rdparty/aclnn_custom_ops/BUILD`（[文件位置](../3rdparty/aclnn_custom_ops/BUILD)），在 genrule 的 `cmd` 中追加新算子名：

```python
# 修改前（单算子）：
cmd = "bash $(location build_for_bazel.sh) $(RULEDIR) 'apply_top_k_top_p_custom' ascend910b",

# 修改后（多算子，分号分隔）：
cmd = "bash $(location build_for_bazel.sh) $(RULEDIR) 'apply_top_k_top_p_custom;<your_op_name>' ascend910b",
```

### 2.3 产物清单

genrule 输出以下产物（无需修改 `outs` 列表，新算子的 `.so` 和 `opp/` 会自动合并到同名文件中）：

| 产物 | 用途 |
|------|------|
| `lib/libcust_opapi.so` | ACLNN 算子接口库（dlopen 加载） |
| `lib/libcust_opsproto_rt2.0.so` | 算子原型库 |
| `lib/libcust_opmaster_rt2.0.so` | Tiling 参数计算库 |
| `opp/op_impl/` | Kernel 二进制 + 配置文件（CANN Runtime 加载） |

---

## 第三步：编写 C++ 封装层

封装层代码位于 `rtp_llm/models_py/bindings/ascend/ops/`。

### 3.1 复制 ACLNN C 接口头文件

文件名：`aclnn_<op_name>.h`

内容为算子自动生成的 C 接口声明，包含两个函数：

```cpp
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnn<OpName>GetWorkspaceSize(
    const aclTensor* input, ..., aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnn<OpName>(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
```

**注意**：此头文件由算子编译后自动生成在 `build/` 目录下，需手动复制到此处。函数名严格遵循 `aclnn<OpName>` 命名。

参考：[aclnn_apply_top_k_top_p_custom.h](../rtp_llm/models_py/bindings/ascend/ops/aclnn_apply_top_k_top_p_custom.h)

### 3.2 编写 C++ 封装函数

文件名：`ascend_<op_name>.h`

```cpp
#pragma once
#if USING_ASCEND

#include <torch/all.h>
#include "op_api_common.h"          // EXEC_NPU_CMD 宏
#include "aclnn_<op_name>.h"        // ACLNN C 接口
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace ascend {

/**
 * @brief <算子功能简述>
 *
 * 函数名遵循小驼峰 + 功能语义命名规范。
 * 应与 aclnn_<op_name>.h 中的 C 函数名（去掉 aclnn 前缀后首字母小写）对应。
 *
 * @param input  输入张量，形状 [batch_size, ...]，在 NPU 上
 * @param output 输出张量，形状同 input，在 NPU 上
 */
inline void yourFunctionName(
    const at::Tensor& input,
    // ... 其他参数 ...
    at::Tensor& output)
{
    // ---- 参数校验 ----
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");

    // ---- 调用 ACLNN 自定义算子 ----
    // EXEC_NPU_CMD 自动完成：
    //   1. dlopen("libcust_opapi.so") 查找函数地址
    //   2. 调用 GetWorkspaceSize 计算所需 workspace
    //   3. 分配 NPU workspace
    //   4. 提交到当前 NPU stream 执行
    EXEC_NPU_CMD(aclnn<OpName>, input, ..., output);
}

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND
```

**命名规范**：
- 封装函数名：小驼峰，体现功能语义（如 `applyTopKTopP`、`moeGatingTopK`）
- ACLNN C 函数名：`aclnn` + 首字母大写的 OpName（如 `aclnnApplyTopKTopPCustom`）
- 文件名：`ascend_` + 下划线分隔的功能名（如 `ascend_apply_top_k_top_p_custom.h`）

参考：[ascend_apply_top_k_top_p_custom.h](../rtp_llm/models_py/bindings/ascend/ops/ascend_apply_top_k_top_p_custom.h)

### 3.3 注册到 BUILD

在 [ascend/ops/BUILD](../rtp_llm/models_py/bindings/ascend/ops/BUILD) 的 `npu_bridge` 中添加新文件：

```python
cc_library(
    name = "npu_bridge",
    hdrs = [
        # ... 已有文件 ...
        "aclnn_<op_name>.h",        # ← 添加 ACLNN C 头文件
        "ascend_<op_name>.h",       # ← 添加 C++ 封装头文件
    ],
    # ... 其余不变 ...
)
```

---

## 第四步：在生产代码中调用

### 4.1 调用位置

在 [CudaSampleOp.cc](../rtp_llm/models_py/bindings/core/CudaSampleOp.cc) 的 `#elif USING_ASCEND` 分支的 `sampleGreedy()` 函数中调用。

```cpp
#if USING_ASCEND
#include "rtp_llm/models_py/bindings/ascend/ops/ascend_<op_name>.h"

GreedyOutput sampleGreedy(const GreedyParams& params) {
    // ...
    rtp_llm::ascend::yourFunctionName(input_tensor, ..., output_tensor);
    // ...
}
#endif
```

### 4.2 调用规范

- **必须放在 `#if USING_ASCEND` / `#elif USING_ASCEND` 条件编译块内**
- **禁止在任何 CUDA 或 ROCm 代码路径中调用**
- 调用前完成必要的设备转换（如 CPU→NPU 通过 `tensor.to(device_type)`）
- `EXEC_NPU_CMD` 要求所有 tensor 参数已在 NPU 设备上

---

## 第五步：环境变量配置

### 5.1 编译时配置

`.bazelrc` 已配置（无需修改）：

```
test:ascend --test_env ASCEND_CUSTOM_OPP_PATH="bazel-bin/3rdparty/aclnn_custom_ops/opp"
```

### 5.2 运行时配置

生产服务启动前必须设置：

```bash
export ASCEND_CUSTOM_OPP_PATH="/path/to/rtp-llm/bazel-bin/3rdparty/aclnn_custom_ops/opp"
```

`ASCEND_CUSTOM_OPP_PATH` 必须指向包含 `op_impl/` 子目录的路径，CANN Runtime 通过此变量加载 kernel 二进制文件。

> **注意**：`ASCEND_CUSTOM_OPP_PATH` 和 `ASCEND_CANN_PACKAGE_PATH`（指向 CANN SDK 安装路径）是两个独立的变量，不可混用。

参考：`.bazelrc` 中的 `--test_env` 配置。

---

## 第六步：验证

### 6.1 编译验证

```bash
cd /path/to/rtp-llm
bazel build //rtp_llm:rtp_llm --config=ascend --verbose_failures
```

### 6.2 单元测试

在 `rtp_llm/models_py/bindings/ascend/ops/tests/` 下编写 C++ 或 Python 测试验证算子正确性。

### 6.3 端到端验证

启动 rtp-llm 服务，发送推理请求，验证采样结果正确且跨 run 确定性满足预期。

---

## 附录：关键概念

### A. 编译工具链

| 组件 | 说明 |
|------|------|
| bisheng | 华为 Ascend 编译器（基于 LLVM），用于编译 AscendC kernel |
| CMake | 算子源项目的构建系统（vllm-ascend 标准） |
| Bazel | rtp-llm 的构建系统，通过 genrule 调用 CMake |
| CPack | CMake 打包工具，将算子产物安装到 `_CPack_Packages/` 目录 |

### B. EXEC_NPU_CMD 宏

定义在 `op_api_common.h` 中，封装了完整的 ACLNN 调用链：

1. `dlopen("libcust_opapi.so")` — 动态加载算子接口库
2. `aclnn<OpName>GetWorkspaceSize()` — 计算 NPU workspace 大小
3. `aclrtMalloc()` — 分配 NPU workspace
4. `aclnn<OpName>()` — 提交到当前 NPU stream 执行
5. `aclrtSynchronizeStream()` — 等待执行完成

### C. 平台条件编译

| 宏 | 含义 |
|----|------|
| `USING_ASCEND` | C++ 编译时判断 Ascend 平台 |
| `USING_CUDA` | C++ 编译时判断 CUDA 平台 |
| `@//:using_ascend` | Bazel select 判断 Ascend 平台 |
| `target_compatible_with` | Bazel 目标平台兼容性约束 |

### D. 常见问题

**Q: 编译报错 "aclnn<OpName> not found"**

确保 `aclnn_<op_name>.h` 中声明的函数名与 CMake 编译生成的符号名完全一致。检查 `build.sh` 的 `-n` 参数是否正确。

**Q: 运行时 crash "libcust_opapi.so not found"**

检查 `ASCEND_CUSTOM_OPP_PATH` 是否正确设置，`opp/op_impl/` 目录是否包含 kernel `.o` 文件。

**Q: 多算子接入时 genrule 产物冲突**

多算子会被编译到相同的 `libcust_opapi.so` 中，不存在冲突。只需在 `cmd` 中用分号分隔所有算子名即可。

---

## 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-06 | v1.0 | 初始版本，基于 `applyTopKTopP` 算子接入经验编写 |
