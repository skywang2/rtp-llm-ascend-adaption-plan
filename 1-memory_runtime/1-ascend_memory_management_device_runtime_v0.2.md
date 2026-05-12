# Phase 1: 内存管理与设备运行时 — 分步执行验证方案 (v0.2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 替换 GPU 内存分配器、参数化设备放置、适配 Stream/Event 管理，使 tensor 可正确分配到 NPU 设备。

**Architecture:** 参考现有 ROCm 适配模式（`rtp_llm/cpp/rocm/`），新建 `rtp_llm/cpp/ascend/` 独立实现层。由于 CANN API 与 CUDA API 签名不同（不像 HIP 那样可以简单 `#define` 映射），采用**独立实现 + `#if USING_ASCEND` 编译守卫**方式。

**Tech Stack:** CANN aclrt API (`aclrtMalloc/Free`, `aclrtStream`, `aclrtEvent`), Bazel `select()` + `config_setting`, PyTorch `torch_npu`

**前置条件:** Phase 0 已完成 — Bazel 编译链路已打通，`USING_ASCEND` 宏可用，`@local_config_ascend` 仓库已生成。

---

## 总览：Phase 1 任务分解

```
Phase 1
├── Task 1: 扩展类型枚举 (DeviceType + AllocatorType)
├── Task 2: 创建 Ascend 兼容层头文件 (ascend_shims.h)
├── Task 3: 创建 Ascend Host Utils (设备属性/内存查询)
├── Task 4: 实现 Ascend Allocator (内存分配器)
├── Task 5: 参数化 BlockPool 中的设备类型
├── Task 6: 参数化 ExecOps 中的设备状态查询
├── Task 7: 参数化 MemoryEvaluationHelper 中的内存查询
├── Task 8: 参数化 PyWrappedModel 中的设备放置
├── Task 9: 适配 CudaOps.cc 中的 Stream/Event/Copy 操作
├── Task 10: 适配 GET_CURRENT_STREAM 宏
├── Task 11: BUILD 文件全量 select() 扩展
└── Task 12: 集成验证
```

每个 Task 包含：创建/修改的具体文件、代码、验证命令、验收标准。

---

## Task 1: 扩展类型枚举 (DeviceType + AllocatorType)

**目标:** 为 Ascend 添加设备类型标识。

**Files:**
- Modify: `rtp_llm/cpp/core/DeviceData.h:11-30`
- Modify: `rtp_llm/cpp/core/allocator.h:9-16`

### Step 1: 修改 DeviceType 枚举

```cpp
// rtp_llm/cpp/core/DeviceData.h:11-18
enum class DeviceType {
    Cpu    = 0,
    Cuda   = 1,
    Yitian = 2,
    ArmCpu = 3,
    ROCm   = 4,
    Ppu    = 5,
    Ascend = 6,
};
```

### Step 2: 修改 buildDeviceType()

```cpp
// rtp_llm/cpp/core/DeviceData.h:20-30
inline DeviceType buildDeviceType() {
#if defined(USE_PPU)
    return DeviceType::Ppu;
#elif USING_CUDA
    return DeviceType::Cuda;
#elif USING_ROCM
    return DeviceType::ROCm;
#elif USING_ASCEND
    return DeviceType::Ascend;
#else
    return DeviceType::Cpu;
#endif
}
```

### Step 3: 修改 AllocatorType 枚举

```cpp
// rtp_llm/cpp/core/allocator.h:9-16
enum class AllocatorType {
    CPU,
    CUDA,
    CUDA_HOST,
    TH,
    ROCM,
    ROCM_HOST,
    ASCEND,
    ASCEND_HOST,
};
```

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/core:devices_base --config=ascend
```

**验收标准:** 编译通过，`buildDeviceType()` 在 `--config=ascend` 下返回 `DeviceType::Ascend`。

### Step 5: Commit

```bash
git add rtp_llm/cpp/core/DeviceData.h rtp_llm/cpp/core/allocator.h
git commit -m "feat: add Ascend to DeviceType and AllocatorType enums"
```

---

## Task 2: 创建 Ascend 兼容层头文件

**目标:** 提供 Ascend 基础类型定义和 API 检查宏。

**Files:**
- Create: `rtp_llm/cpp/ascend/ascend_shims.h`
- Create: `rtp_llm/cpp/ascend/ascend_type_utils.h`

> **注意:** 与 ROCm 的 `cuda_shims.h`（纯宏映射）不同，Ascend shim 不能做简单宏映射，因为 CANN API 签名与 CUDA 不同（如 `aclrtMalloc` 多一个 `aclrtMemMallocPolicy` 参数）。Shim 层仅提供类型别名和检查宏。
> 
> **性能注意:** `syncAndCheckInDebug()` 内部调用 `aclrtSynchronizeDevice()`（全卡同步），仅在 debug mode 下执行，不影响 release 性能。

### Step 1: 创建 ascend_shims.h

```cpp
// rtp_llm/cpp/ascend/ascend_shims.h
#pragma once

#include <acl/acl.h>
#include <iostream>
#include <sstream>
#include "rtp_llm/cpp/utils/Logger.h"

#define ASCEND_CHECK(val) rtp_llm::ascend::check((val), __FILE__, __LINE__)
#define ASCEND_CHECK_ERROR() rtp_llm::ascend::syncAndCheckInDebug(__FILE__, __LINE__)
#define check_ascend_value ASCEND_CHECK

namespace rtp_llm {
namespace ascend {

void throwAscendError(const char* const file, int const line, std::string const& info = "");

template<typename T>
void check(T result, const char* const file, int const line) {
    if (result != ACL_SUCCESS) {
        std::string msg = std::string("ACL_ERROR: ") + std::to_string(static_cast<int>(result))
                          + " at " + file + ":" + std::to_string(line);
        throwAscendError(file, line, msg);
    }
}

void syncAndCheckInDebug(const char* const file, int const line);

}  // namespace ascend
}  // namespace rtp_llm
```

### Step 2: 创建 ascend_type_utils.h

```cpp
// rtp_llm/cpp/ascend/ascend_type_utils.h
#pragma once

// Ascend NPU 使用 aclFloat16 (FP16) 和标准 float (BF16 通过软件模拟或硬件支持)
// 数据类型适配工具

#include <acl/acl.h>
#include <c10/core/ScalarType.h>

namespace rtp_llm {
namespace ascend {

inline aclDataType getAclDtype(at::ScalarType type) {
    switch (type) {
        case at::kFloat:   return ACL_FLOAT;
        case at::kHalf:    return ACL_FLOAT16;
        case at::kBFloat16: return ACL_BF16;
        case at::kInt:     return ACL_INT32;
        case at::kLong:    return ACL_INT64;
        case at::kUInt8:   return ACL_UINT8;
        case at::kInt8:    return ACL_INT8;
        default:           return ACL_FLOAT;
    }
}

}  // namespace ascend
}  // namespace rtp_llm
```

### Step 3: 验证编译（仅头文件）

需先创建 BUILD 文件（见 Task 11），此处先跳过独立验证，在 Task 3 中一并验证。

### Step 4: Commit

```bash
git add rtp_llm/cpp/ascend/ascend_shims.h rtp_llm/cpp/ascend/ascend_type_utils.h
git commit -m "feat: add Ascend shim headers (type aliases and check macros)"
```

---

## Task 3: 创建 Ascend Host Utils (设备属性/内存查询)

**目标:** 替代 `cuda_host_utils.h/.cc` 中的设备查询功能。

**Files:**
- Create: `rtp_llm/cpp/ascend/ascend_host_utils.h`
- Create: `rtp_llm/cpp/ascend/ascend_host_utils.cc`

**参考:** `rtp_llm/cpp/rocm/hip_host_utils.{h,cc}` (ROCm 版本结构)

### Step 1: 创建 ascend_host_utils.h

```cpp
// rtp_llm/cpp/ascend/ascend_host_utils.h
#pragma once

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

#include <acl/acl.h>
#include "rtp_llm/cpp/ascend/ascend_shims.h"

namespace rtp_llm {
namespace ascend {

#define ASCEND_CHECK_VALUE(val, info, ...)                                                                               \
    do {                                                                                                                 \
        bool is_valid_val = (val);                                                                                       \
        if (!is_valid_val) {                                                                                             \
            ascend::throwAscendError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__));                          \
        }                                                                                                                \
    } while (0)

#define ASCEND_FAIL(info, ...) ascend::throwAscendError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))

int  getDevice();
int  getDeviceCount();
std::tuple<size_t, size_t> getDeviceMemoryInfo(bool use_uvm = false);
void syncAndCheckInDebug(const char* const file, int const line);

}  // namespace ascend
}  // namespace rtp_llm
```

### Step 2: 创建 ascend_host_utils.cc

```cpp
// rtp_llm/cpp/ascend/ascend_host_utils.cc
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace ascend {

void throwAscendError(const char* const file, int const line, std::string const& info) {
    auto error_msg =
        std::string("[Ascend][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    if (rtp_llm::StaticConfig::user_ft_core_dump_on_exception) {
        abort();
    }
    throw RTP_EXCEPTION(error_msg);
}

void syncAndCheckInDebug(const char* const file, int const line) {
    if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {
        ASCEND_CHECK(aclrtSynchronizeDevice());
        RTP_LLM_LOG_DEBUG(rtp_llm::fmtstr("run syncAndCheckInDebug at %s:%d", file, line));
    }
}

// Phase 1 假设: 单卡场景，device_id 缓存在静态变量中。
// 多卡场景下（如 tensor-parallel）后续 Phase 需改为 per-rank 变量。
int getDevice() {
    static int device_id = []() {
        int32_t id = 0;
        ASCEND_CHECK(aclrtGetDevice(&id));
        return id;
    }();
    return device_id;
}

// Phase 1 假设: 单卡场景，device_count 缓存在静态变量中。
int getDeviceCount() {
    static int count = []() {
        uint32_t c = 0;
        ASCEND_CHECK(aclrtGetDeviceCount(&c));
        return static_cast<int>(c);
    }();
    return count;
}

std::tuple<size_t, size_t> getDeviceMemoryInfo(bool use_uvm) {
    (void)use_uvm;
    size_t free = 0, total = 0;
    ASCEND_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
    return {free, total};
}

}  // namespace ascend
}  // namespace rtp_llm
```

### Step 3: 创建 BUILD 文件

```python
# rtp_llm/cpp/ascend/BUILD
load("//:def.bzl", "ascend_copts")

cc_library(
    name = "ascend_types_hdr",
    hdrs = [
        "ascend_shims.h",
        "ascend_type_utils.h",
    ],
    deps = [
        "@local_config_ascend//ascend:ascend_headers",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_host_utils",
    srcs = ["ascend_host_utils.cc"],
    hdrs = ["ascend_host_utils.h"],
    deps = [
        ":ascend_types_hdr",
        "@local_config_ascend//ascend:ascend",
        "@local_config_ascend//ascend:ascend_headers",
        "//rtp_llm/cpp/core:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = ascend_copts(),
    visibility = ["//visibility:public"],
)
```

> **注意:** 此 BUILD 文件依赖 `def.bzl` 中的 `ascend_copts()` 函数和 `@local_config_ascend` 仓库，需确保 Phase 0 已完成这些基础设施。

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/ascend:ascend_host_utils --config=ascend
```

**验收标准:** 编译通过，无链接错误（可能需 stub `StaticConfig` 如果尚未编译）。

### Step 5: 编写单元测试

```cpp
// rtp_llm/cpp/ascend/ascend_host_utils_test.cc
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#include <gtest/gtest.h>

TEST(AscendHostUtilsTest, GetDeviceCount) {
    int count = rtp_llm::ascend::getDeviceCount();
    EXPECT_GT(count, 0) << "No Ascend NPU devices found";
}

TEST(AscendHostUtilsTest, GetDevice) {
    int device = rtp_llm::ascend::getDevice();
    EXPECT_GE(device, 0);
}

TEST(AscendHostUtilsTest, GetDeviceMemoryInfo) {
    auto [free, total] = rtp_llm::ascend::getDeviceMemoryInfo();
    EXPECT_GT(total, 0u);
    EXPECT_GT(free, 0u);
    EXPECT_LE(free, total);
}
```

```bash
bazel test //rtp_llm/cpp/ascend:ascend_host_utils_test --config=ascend
```

**验收标准:** 测试通过，能正确获取 NPU 设备数量和内存信息。

### Step 6: Commit

```bash
git add rtp_llm/cpp/ascend/
git commit -m "feat: add Ascend host utils (device query, memory info, error check)"
```

---

## Task 4: 实现 Ascend Allocator (内存分配器)

**目标:** 实现 `Allocator<AllocatorType::ASCEND>` 和 `Allocator<AllocatorType::ASCEND_HOST>`。

**Files:**
- Create: `rtp_llm/cpp/ascend/allocator_ascend.h`
- Create: `rtp_llm/cpp/ascend/allocator_ascend.cc`

**参考:** `rtp_llm/cpp/cuda/allocator_cuda.{h,cc}` (CUDA 版本结构)

### Step 1: 创建 allocator_ascend.h

```cpp
// rtp_llm/cpp/ascend/allocator_ascend.h
#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#include <mutex>
#include <unordered_set>

namespace rtp_llm {

class IAscendAllocator: virtual public IAllocator {
public:
    IAscendAllocator(int device_id): device_id_(device_id) {}
    virtual ~IAscendAllocator() {}

    MemoryType memoryType() const override {
        return MEMORY_GPU;
    }

    void setStream(aclrtStream stream) {
        stream_ = stream;
    }

    aclrtStream returnStream() {
        return stream_;
    }

    void* reMalloc(void* ptr, size_t size) override;

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

protected:
    aclrtStream stream_ = nullptr;
    const int   device_id_;
};

class PurePointerAscendAllocator: public IAscendAllocator {
public:
    PurePointerAscendAllocator(int device_id);
    ~PurePointerAscendAllocator();

public:
    void* malloc(size_t size) override;
    void* mallocSync(size_t size) override;
    void  free(void** ptr) override;

protected:
    virtual bool        isExist(void* address) const;
    virtual ReallocType isReMalloc(void* address, size_t size) const;

    virtual void* doMalloc(size_t size)     = 0;
    virtual void* doMallocSync(size_t size) = 0;
    virtual void  doFree(void* ptr)         = 0;

    void destroy();

private:
    std::unique_ptr<std::unordered_map<void*, size_t>> pointer_mapping_;
    std::mutex                                         lock_;
};

template<>
class Allocator<AllocatorType::ASCEND>: public PurePointerAscendAllocator,
                                         public TypedAllocator<AllocatorType::ASCEND> {
public:
    Allocator(int device_id);
    ~Allocator();

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;
};

template<>
class Allocator<AllocatorType::ASCEND_HOST>: public PurePointerAscendAllocator,
                                              public TypedAllocator<AllocatorType::ASCEND_HOST> {
public:
    Allocator(int device_id);
    ~Allocator();

    MemoryType memoryType() const override {
        return MEMORY_CPU_PINNED;
    }

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;
};

}  // namespace rtp_llm
```

### Step 2: 创建 allocator_ascend.cc

```cpp
// rtp_llm/cpp/ascend/allocator_ascend.cc
#include "rtp_llm/cpp/ascend/allocator_ascend.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <mutex>
#include <cmath>

namespace rtp_llm {

// === IAscendAllocator::reMalloc ===
// 与 CUDA 版本完全相同的设备无关逻辑（直接复用）
void* IAscendAllocator::reMalloc(void* ptr, size_t size) {
    size           = ((size + 127) / 128) * 128;
    void* void_ptr = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else if (realloc_type == ReallocType::DECREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else {
            RTP_LLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            return void_ptr;
        }
    } else {
        RTP_LLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size);
    }
}

// === PurePointerAscendAllocator ===
// 与 CUDA 版本完全相同的设备无关逻辑
PurePointerAscendAllocator::PurePointerAscendAllocator(int device_id):
    IAscendAllocator(device_id), pointer_mapping_(new std::unordered_map<void*, size_t>) {}

PurePointerAscendAllocator::~PurePointerAscendAllocator() {}

void PurePointerAscendAllocator::destroy() {
    while (!pointer_mapping_->empty()) {
        auto it  = pointer_mapping_->begin();
        auto ptr = it->first;
        free(&ptr);
    }
}

bool PurePointerAscendAllocator::isExist(void* address) const {
    return pointer_mapping_->count(address) > 0;
}

ReallocType PurePointerAscendAllocator::isReMalloc(void* address, size_t size) const {
    RTP_LLM_CHECK(isExist(address));
    if (pointer_mapping_->at(address) < size) {
        return ReallocType::INCREASE;
    } else if (pointer_mapping_->at(address) == size) {
        return ReallocType::REUSE;
    } else {
        return ReallocType::DECREASE;
    }
}

void* PurePointerAscendAllocator::malloc(size_t size) {
    if (size == 0) return nullptr;
    void*                       ptr = doMalloc(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void* PurePointerAscendAllocator::mallocSync(size_t size) {
    if (size == 0) return nullptr;
    void*                       ptr = doMallocSync(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void PurePointerAscendAllocator::free(void** ptr) {
    void* address = *ptr;
    if (address) {
        std::lock_guard<std::mutex> lock(lock_);
        RTP_LLM_CHECK_WITH_INFO(
            pointer_mapping_->count(address), "pointer_mapping_ does not have information of ptr at %p", address);
        doFree(address);
        *ptr = nullptr;
        pointer_mapping_->erase(address);
    }
}

// === Allocator<ASCEND> (NPU Device Memory) ===

Allocator<AllocatorType::ASCEND>::Allocator(int device_id): PurePointerAscendAllocator(device_id) {}

Allocator<AllocatorType::ASCEND>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::ASCEND>::doMalloc(size_t size) {
    void* ptr = nullptr;
    size_t aligned_size = (size_t)(ceil(size / 128.)) * 128;
    ASCEND_CHECK(aclrtMalloc(&ptr, aligned_size, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

void* Allocator<AllocatorType::ASCEND>::doMallocSync(size_t size) {
    void* ptr = nullptr;
    size_t aligned_size = (size_t)(ceil(size / 128.)) * 128;
    ASCEND_CHECK(aclrtMalloc(&ptr, aligned_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CHECK(aclrtSynchronizeStream(stream_));
    return ptr;
}

void Allocator<AllocatorType::ASCEND>::doFree(void* address) {
    ASCEND_CHECK(aclrtSynchronizeStream(stream_));
    ASCEND_CHECK(aclrtFree(address));
}

// === Allocator<ASCEND_HOST> (Host Pinned Memory) ===

Allocator<AllocatorType::ASCEND_HOST>::Allocator(int device_id): PurePointerAscendAllocator(device_id) {}

Allocator<AllocatorType::ASCEND_HOST>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::ASCEND_HOST>::doMalloc(size_t size) {
    void* ptr = nullptr;
    size_t aligned_size = (size_t)(ceil(size / 128.)) * 128;
    ASCEND_CHECK(aclrtMallocHost(&ptr, aligned_size));
    return ptr;
}

void* Allocator<AllocatorType::ASCEND_HOST>::doMallocSync(size_t size) {
    return doMalloc(size);
}

void Allocator<AllocatorType::ASCEND_HOST>::doFree(void* address) {
    if (address) {
        ASCEND_CHECK(aclrtFreeHost(address));
    }
}

}  // namespace rtp_llm
```

### Step 3: 更新 BUILD 文件

在 `rtp_llm/cpp/ascend/BUILD` 中追加:

```python
cc_library(
    name = "allocator_ascend",
    srcs = ["allocator_ascend.cc"],
    hdrs = ["allocator_ascend.h"],
    deps = [
        ":ascend_host_utils",
        "@local_config_ascend//ascend:ascend",
        "@local_config_ascend//ascend:ascend_headers",
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = ascend_copts(),
    visibility = ["//visibility:public"],
)
```

### Step 4: 编写分配器测试

```cpp
// rtp_llm/cpp/ascend/allocator_ascend_test.cc
#include "rtp_llm/cpp/ascend/allocator_ascend.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace rtp_llm;

TEST(AscendAllocatorTest, MallocAndFree) {
    Allocator<AllocatorType::ASCEND> alloc(0);
    void* ptr = alloc.malloc(1024);
    ASSERT_NE(ptr, nullptr);
    alloc.free(&ptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST(AscendAllocatorTest, MallocLargeBuffer) {
    Allocator<AllocatorType::ASCEND> alloc(0);
    size_t size = 256 * 1024 * 1024;  // 256 MiB
    void* ptr = alloc.malloc(size);
    ASSERT_NE(ptr, nullptr);
    alloc.free(&ptr);
}

TEST(AscendAllocatorTest, ReMalloc) {
    Allocator<AllocatorType::ASCEND> alloc(0);
    void* ptr = alloc.malloc(1024);
    ASSERT_NE(ptr, nullptr);
    void* ptr2 = alloc.reMalloc(ptr, 2048);
    ASSERT_NE(ptr2, nullptr);
    alloc.free(&ptr2);
}

TEST(AscendAllocatorTest, ReMallocReuse) {
    Allocator<AllocatorType::ASCEND> alloc(0);
    void* ptr = alloc.malloc(2048);
    ASSERT_NE(ptr, nullptr);
    void* ptr2 = alloc.reMalloc(ptr, 1024);
    EXPECT_EQ(ptr, ptr2);
    alloc.free(&ptr2);
}

TEST(AscendAllocatorTest, HostMallocAndFree) {
    Allocator<AllocatorType::ASCEND_HOST> alloc(0);
    void* ptr = alloc.malloc(1024);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0xAB, 1024);
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0xAB);
    alloc.free(&ptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST(AscendAllocatorTest, ZeroSizeMalloc) {
    Allocator<AllocatorType::ASCEND> alloc(0);
    void* ptr = alloc.malloc(0);
    EXPECT_EQ(ptr, nullptr);
}
```

### Step 5: 验证

```bash
bazel test //rtp_llm/cpp/ascend:allocator_ascend_test --config=ascend
```

**验收标准:**
- `MallocAndFree` 通过
- `MallocLargeBuffer` 通过（证明 NPU 内存分配正常）
- `ReMalloc`/`ReMallocReuse` 通过（证明 realloc 逻辑正确）
- `HostMallocAndFree` 通过（证明 Host pinned 内存可用）
- `ZeroSizeMalloc` 返回 nullptr

### Step 6: Commit

```bash
git add rtp_llm/cpp/ascend/allocator_ascend.h rtp_llm/cpp/ascend/allocator_ascend.cc rtp_llm/cpp/ascend/allocator_ascend_test.cc rtp_llm/cpp/ascend/BUILD
git commit -m "feat: implement Ascend allocator (NPU device + host pinned memory)"
```

---

## Task 5: 参数化 BlockPool 中的设备类型

**目标:** 消除 `BlockPool.cc` 中硬编码的 `torch::kCUDA`。

**Files:**
- Modify: `rtp_llm/cpp/cache/BlockPool.cc:39-47`

### Step 1: 修改 BlockPool::initializeCacheBuffer

当前代码 (`BlockPool.cc:39-47`):
```cpp
void BlockPool::initializeCacheBuffer() {
    if (allocation_type_ == AllocationType::HOST) {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                    .pin_memory();
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
```

修改为:
```cpp
void BlockPool::initializeCacheBuffer() {
    torch::Device device = torch::kCPU;
    if (allocation_type_ == AllocationType::HOST) {
        device = torch::kCPU;
    } else {
#if USING_CUDA
        device = torch::kCUDA;
#elif USING_ASCEND
        device = torch::Device(torch::kNPU);
#elif USING_ROCM
        device = torch::kCUDA;  // HIP uses CUDA device type
#else
        device = torch::kCPU;
#endif
    }

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)}, options);

    // pin_memory() 仅在 HOST 模式下调用；NPU 设备 tensor 不在此路径
    // 防御: allocation_type_ == HOST 时 device == kCPU，非 HOST 时不调用 pin_memory()
    if (allocation_type_ == AllocationType::HOST) {
        cache_aligned_buffer_ = cache_aligned_buffer_.pin_memory();
    }
```

### Step 2: 搜索 BlockPool 中其他 `torch::kCUDA`

```bash
rg "torch::kCUDA" rtp_llm/cpp/cache/BlockPool.cc
```

如果有其他位置，同样添加 `#elif USING_ASCEND` 分支。

### Step 3: 验证编译

```bash
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
```

**验收标准:** 编译通过，`BlockPool` 在 Ascend 配置下使用 `torch::kNPU` 创建 tensor。

### Step 4: Commit

```bash
git add rtp_llm/cpp/cache/BlockPool.cc
git commit -m "feat: parameterize BlockPool device type for Ascend NPU"
```

---

## Task 6: 参数化 ExecOps 中的设备状态查询

**目标:** 修改 `getGpuExecStatus()` 和 `getTorchCudaDevice()` 以支持 Ascend。

**Files:**
- Modify: `rtp_llm/cpp/core/ExecOps.cc:363-385`

### Step 1: 修改 getGpuExecStatus

当前代码 (`ExecOps.cc:363-377`):
```cpp
ExecStatus getGpuExecStatus() {
    MemoryStatus mem;
    size_t       total_bytes = 0;
#if USING_CUDA
    auto error = cudaMemGetInfo(&mem.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == cudaSuccess);
#elif USING_ROCM
    hipMemGetInfo(&mem.free_bytes, &total_bytes);
#endif
    mem.used_bytes      = total_bytes - mem.free_bytes;
    mem.available_bytes = mem.free_bytes;
    ...
}
```

修改为:
```cpp
ExecStatus getGpuExecStatus() {
    MemoryStatus mem;
    size_t       total_bytes = 0;
#if USING_CUDA
    auto error = cudaMemGetInfo(&mem.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == cudaSuccess);
#elif USING_ROCM
    hipMemGetInfo(&mem.free_bytes, &total_bytes);
#elif USING_ASCEND
    auto [free_bytes, total_bytes_val] = rtp_llm::ascend::getDeviceMemoryInfo();
    mem.free_bytes = free_bytes;
    total_bytes = total_bytes_val;
#endif
    mem.used_bytes      = total_bytes - mem.free_bytes;
    mem.available_bytes = mem.free_bytes;
    ...
}
```

### Step 2: 修改 getTorchCudaDevice

当前代码 (`ExecOps.cc:383-385`):
```cpp
torch::Device getTorchCudaDevice() {
    return torch::Device(torch::kCUDA);
}
```
修改为:

```cpp
// 注意: 函数名保留 "Cuda" 历史命名，Phase 4 统一 rename 为 getTorchGpuDevice()
torch::Device getTorchCudaDevice() {
#if USING_ASCEND
    return torch::Device(torch::kNPU);
#else
    return torch::Device(torch::kCUDA);
#endif
}
```

### Step 3: 添加 Ascend include

在 `ExecOps.cc` 顶部添加:
```cpp
#if USING_ASCEND
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#endif
```

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/core:exec_ops --config=ascend
```

**验收标准:** 编译通过。

### Step 5: Commit

```bash
git add rtp_llm/cpp/core/ExecOps.cc
git commit -m "feat: parameterize ExecOps device status query for Ascend NPU"
```

---

## Task 7: 参数化 MemoryEvaluationHelper 中的内存查询

**目标:** 修改 `getDefaultRuntimeMemorySize()` 中的 `cudaMemGetInfo` 调用。

**Files:**
- Modify: `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc:5-62`

### Step 1: 修改头文件 include

当前 (`MemoryEvaluationHelper.cc:5-10`):
```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif
```

添加:
```cpp
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#elif USING_ASCEND
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#endif
```

### Step 2: 修改内存查询代码

当前 (`MemoryEvaluationHelper.cc:57-62`):
```cpp
#if USING_CUDA
    check_cuda_value(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ROCM
    ROCM_CHECK(hipMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#endif
```

修改为:
```cpp
#if USING_CUDA
    check_cuda_value(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ROCM
    ROCM_CHECK(hipMemGetInfo(&free_gpu_bytes, &total_gpu_bytes));
#elif USING_ASCEND
    std::tie(free_gpu_bytes, total_gpu_bytes) = rtp_llm::ascend::getDeviceMemoryInfo();
#endif
```

### Step 3: 验证编译

```bash
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
```

### Step 4: Commit

```bash
git add rtp_llm/cpp/cache/MemoryEvaluationHelper.cc
git commit -m "feat: add Ascend memory query to MemoryEvaluationHelper"
```

---

## Task 8: 参数化 PyWrappedModel 中的设备放置

**目标:** 将 `.cuda()` / `torch::kCUDA` 硬编码替换为参数化设备。

**Files:**
- Modify: `rtp_llm/cpp/models/PyWrappedModel.cc` (~8 处)
- Modify: `rtp_llm/cpp/models/PyWrappedModel.h` (~3 处)

> **这是 Phase 1 中修改量最大的 Task。**

### Step 0: 前置——重新定位行号

> **警告:** 以下行号基于编写计划时的源码快照。执行前请务必运行以下命令重新定位每处修改的实际位置：
> ```bash
> rg -n "\.cuda\(\)|torch::kCUDA" rtp_llm/cpp/models/PyWrappedModel.cc rtp_llm/cpp/models/PyWrappedModel.h
> ```
> 以实际输出行号为准执行替换。

### Step 1: 添加设备工具函数

在 `PyWrappedModel.h` 或 `PyWrappedModel.cc` 顶部添加:

```cpp
namespace {
torch::Device getTorchDevice() {
#if USING_ASCEND
    return torch::Device(torch::kNPU);
#else
    return torch::Device(torch::kCUDA);
#endif
}

torch::TensorOptions deviceTensorOptions() {
    return torch::TensorOptions().device(getTorchDevice());
}
}  // anonymous namespace
```

> **替代方案:** 如果 `ExecOps.h` 中已导出 `getTorchCudaDevice()`，直接调用该函数，无需重复定义。

### Step 2: 逐行替换

需要替换的位置（基于源码分析）:

| 位置 | 原始代码 | 替换为 |
|------|---------|--------|
| `PyWrappedModel.cc:33` | `tensor.to(torch::kCUDA, ...)` | `tensor.to(getTorchDevice(), ...)` |
| `PyWrappedModel.cc:112` | `torch::kCUDA` (tensor 创建) | `getTorchDevice()` |
| `PyWrappedModel.cc:114` | `torch::kCUDA` (tensor 创建) | `getTorchDevice()` |
| `PyWrappedModel.cc:158` | `.to(torch::kCUDA, ...)` | `.to(getTorchDevice(), ...)` |
| `PyWrappedModel.cc:175` | `.cuda()` | `.to(getTorchDevice())` |
| `PyWrappedModel.cc:181-182` | `.cuda()` | `.to(getTorchDevice())` |
| `PyWrappedModel.cc:285` | `.clone().cuda()` | `.clone().to(getTorchDevice())` |
| `PyWrappedModel.cc:318` | `torch::kCUDA` (device) | `getTorchDevice()` |
| `PyWrappedModel.cc:539` | `.to(torch::kCUDA, ...)` | `.to(getTorchDevice(), ...)` |
| `PyWrappedModel.h:119` | `.cuda()` | `.to(getTorchDevice())` |
| `PyWrappedModel.h:237` | `.cuda()` | `.to(getTorchDevice())` |
| `PyWrappedModel.h:240` | `.cuda()` | `.to(getTorchDevice())` |

### Step 3: 搜索确认无遗漏

```bash
rg "\.cuda\(\)|torch::kCUDA" rtp_llm/cpp/models/PyWrappedModel.cc rtp_llm/cpp/models/PyWrappedModel.h
```

**验收标准:** 命令输出为空（无残留硬编码）。

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/models:py_wrapped_model --config=ascend
```

### Step 5: Commit

```bash
git add rtp_llm/cpp/models/PyWrappedModel.cc rtp_llm/cpp/models/PyWrappedModel.h
git commit -m "feat: parameterize device placement in PyWrappedModel for Ascend"
```

---

## Task 9: 适配 CudaOps.cc 中的 Stream/Event/Copy 操作

**目标:** 在 `CudaOps.cc` 中添加 `#elif USING_ASCEND` 分支。

**Files:**
- Modify: `rtp_llm/cpp/core/CudaOps.cc` (全文件重构)

### Step 1: 添加 Ascend include

```cpp
// CudaOps.cc:10-21 修改为:
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "ATen/ops/cat.h"
#include "rtp_llm/models_py/bindings/common/kernels/batch_copy.h"
#include "rtp_llm/models_py/bindings/common/kernels/copy_utils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"
#include <cuda_profiler_api.h>
#elif USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#elif USING_ASCEND
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"
#include <acl/acl.h>
#endif
```

### Step 2: 添加 Ascend runtimeCopy 实现

在 `#else // ROCm` 分支之前添加 Ascend 分支:

```cpp
#elif USING_ASCEND

// ============================================================
// Copy ops (Ascend)
// ============================================================

void runtimeCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;

    if (src.data_ptr() == dst.data_ptr()) {
        return;
    }

    // Ascend: use PyTorch tensor copy, which dispatches through torch_npu
    dst.copy_(src, /*non_blocking=*/src.is_npu() && dst.is_npu());
}

void multiMergeCopy(const MultiMergeCopyParams& params) {
    // 与 CUDA/ROCm 版本行为一致: dst_ptr 在 HOST 上 (MultiMergeCopy 用于 CPU 侧合并拷贝)
    // 若后续需要支持 NPU dst，需替换为 aclrtMemcpy
    for (size_t i = 0; i < params.src_ptrs.size(); i++) {
        auto dst = static_cast<char*>(params.dst_ptr) + params.dst_offsets[i];
        std::memcpy(dst, params.src_ptrs[i], params.copy_size[i]);
    }
}

static void batchCopyFallback(const BatchCopyParams& params) {
    // 与 ROCm 的 batchCopyFallback 相同逻辑
    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        auto   copy_type       = BatchCopyParams::CopyType(copy_type_enum);
        auto&  buffers         = params.copy_buffers[copy_type];
        size_t copy_batch_size = buffers.sizes.size();
        if (copy_batch_size == 0)
            continue;

        for (size_t i = 0; i < copy_batch_size; ++i) {
            size_t        bytes      = buffers.sizes[i];
            torch::Device dst_device = torch::kCPU, src_device = torch::kCPU;
            switch (copy_type) {
                case BatchCopyParams::D2D:
                    dst_device = torch::kNPU;  // Ascend: use NPU device
                    src_device = torch::kNPU;
                    break;
                case BatchCopyParams::D2H:
                    dst_device = torch::kCPU;
                    src_device = torch::kNPU;
                    break;
                case BatchCopyParams::H2D:
                    dst_device = torch::kNPU;
                    src_device = torch::kCPU;
                    break;
                case BatchCopyParams::H2H:
                    break;
                default:
                    RTP_LLM_FAIL("Unexpected CopyType %d", copy_type);
                    break;
            }
            auto dst_tensor =
                torch::from_blob(buffers.dst_ptr[i], {(int64_t)bytes}, torch::dtype(torch::kUInt8).device(dst_device));
            auto src_tensor = torch::from_blob(const_cast<void*>(buffers.src_ptr[i]),
                                               {(int64_t)bytes},
                                               torch::dtype(torch::kUInt8).device(src_device));
            runtimeCopy({dst_tensor, src_tensor, params.overlapped});
        }
    }
}

void runtimeBatchCopy(const BatchCopyParams& params) {
    batchCopyFallback(params);
}

void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
    // Phase 1: 使用 PyTorch 原生操作实现
    // 后续 Phase 5 中替换为 CANN 高性能算子
    auto masked = logits.clone();
    auto mask_float = mask.to(logits.dtype());
    logits.masked_fill_(mask_float.to(torch::kBool) == 0, -1e9f);
}
```

### Step 3: 修改 profiler 函数

`CudaOps.cc:348-359`:
```cpp
void cudaProfilerStart() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStart());
#endif
}

void cudaProfilerEnd() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStop());
#endif
}
```

这两个函数无需修改，Ascend 暂不使用 CUDA profiler（后续可用 msprof 替代）。

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/core:exec_ops --config=ascend
```

### Step 5: Commit

```bash
git add rtp_llm/cpp/core/CudaOps.cc
git commit -m "feat: add Ascend Stream/Event/Copy operations to CudaOps"
```

---

## Task 10: 适配 GET_CURRENT_STREAM 宏

**目标:** 替换 `at::cuda::getCurrentCUDAStream()` 调用。

**Files:**
- Modify: `rtp_llm/models_py/bindings/common/Torch_ext.h:28`
- Modify: `rtp_llm/cpp/core/CudaSampleOp.cc`
- Modify: `rtp_llm/cpp/core/CudaBeamSearchOp.cc`
- Modify: `rtp_llm/cpp/core/ExecOps.cc`

### Step 1: 修改 Torch_ext.h 中的 GET_CURRENT_STREAM 宏

```cpp
// Torch_ext.h 当前:
// #define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()

// 修改为:
#if USING_ASCEND
#include <torch_npu/csrc/core/NPUStream.h>
#define GET_CURRENT_STREAM() c10_npu::getCurrentNPUStream()
#else
#define GET_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#endif
```

> **注意:** `torch_npu` 的 NPU stream API 需确认具体头文件路径。如果 `torch_npu` 尚未集成，可先使用 `aclrtGetCurrentStream()` 作为替代。

### Step 1.5: torch_npu 兼容性检查（执行前必做）

```bash
# 检查 torch_npu stream 头文件实际路径
python3 -c "
import torch_npu
# 尝试 import 不同版本的 NPUStream
try:
    from torch_npu.npu import NPUStream
    print('NPUStream path: torch_npu.npu.NPUStream')
except ImportError:
    pass
try:
    from torch_npu.csrc.core import NPUStream as NPUStream2
    print('NPUStream path: torch_npu.csrc.core.NPUStream')
except ImportError:
    pass
try:
    from torch_npu.csrc.npu import NPUStream
    print('NPUStream path: torch_npu.csrc.npu.NPUStream')
except ImportError:
    pass
# 检查 getCurrentNPUStream
try:
    stream = torch_npu.npu.current_stream()
    print(f'current_stream() works, stream={stream}')
except Exception as e:
    print(f'current_stream() failed: {e}')
"

# 搜索头文件位置
find ${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend} -name "NPUStream.h" 2>/dev/null

# 根据上述输出，将 Torch_ext.h 中的 include 和宏调整为实际路径
# 备选方案（如果 torch_npu 不可用）:
#   #define GET_CURRENT_STREAM() aclrtGetCurrentStream()
```


### Step 2: 搜索所有 at::cuda::getCurrentCUDAStream 调用

```bash
rg "at::cuda::getCurrentCUDAStream" --type cpp rtp_llm/cpp/
```

**已知 27 处**。Phase 1 中重点处理核心文件:
- `CudaOps.cc` (已在 Task 9 处理)
- `CudaSampleOp.cc` (3 处)
- `CudaBeamSearchOp.cc` (1 处)
- `ExecOps.cc` (1 处)

其余 (`CudaFlashInfer.cc`, `cufmha/` 等) 属于 Phase 3 Attention 适配范围。

### Step 3: 修改 CudaSampleOp.cc

```cpp
// 在文件顶部添加:
#if USING_ASCEND
#include <torch_npu/csrc/core/NPUStream.h>
#define GET_NPU_STREAM() c10_npu::getCurrentNPUStream().stream()
#endif

// 将所有 at::cuda::getCurrentCUDAStream().stream() 替换:
#if USING_ASCEND
    auto stream = GET_NPU_STREAM();
#else
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#endif
```

### Step 4: 修改 CudaBeamSearchOp.cc

与 Step 3 相同模式。

### Step 5: 验证编译

```bash
bazel build //rtp_llm/cpp/core:CudaSampleOp --config=ascend
bazel build //rtp_llm/cpp/core:CudaBeamSearchOp --config=ascend
```

### Step 6: Commit

```bash
git add rtp_llm/models_py/bindings/common/Torch_ext.h rtp_llm/cpp/core/CudaSampleOp.cc rtp_llm/cpp/core/CudaBeamSearchOp.cc rtp_llm/cpp/core/ExecOps.cc
git commit -m "feat: add Ascend stream support to GET_CURRENT_STREAM macro"
```

---

## Task 11: BUILD 文件全量 select() 扩展

**目标:** 确保所有 BUILD 文件中的 `select()` 块包含 Ascend 分支。

**Files:**
- Modify: `rtp_llm/cpp/cache/BUILD`
- Modify: `rtp_llm/cpp/cuda/BUILD`
- Modify: `bazel/device_defs.bzl`
- Modify: `def.bzl` (如 Phase 0 未完成)

### Step 1: 修改 rtp_llm/cpp/cache/BUILD

找到所有引用 `cuda_host_utils` / `rocm_host_utils` 的 `select()` 块，添加:

```python
"@//:using_ascend": ["//rtp_llm/cpp/ascend:ascend_host_utils"],
```

### Step 2: 修改 bazel/device_defs.bzl

```python
def device_test_envs():
    return select({
        "@//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
        "@//:using_rocm": {
            "TEST_USING_DEVICE": "ROCM",
        },
        "@//:using_ascend": {
            "TEST_USING_DEVICE": "ASCEND",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })

def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/cuda/ops:cuda_impl",
        ],
        # 注意: ascend_impl target 在 Phase 2 (GEMM 适配) 中创建
        # Phase 1 期间注释掉，待 Phase 2 取消注释:
        # "@//:using_ascend": [
        #     "//rtp_llm/cpp/ascend:ascend_impl",
        # ],
        "//conditions:default": [],
    })
```

### Step 3: 全局搜索 BUILD 文件中的 select 块

```bash
rg "using_cuda.*using_rocm" --type build
```

对每个匹配的 BUILD 文件添加 `"@//:using_ascend"` 分支。

### Step 4: 验证编译

```bash
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
bazel build //rtp_llm/cpp/core:devices_base --config=ascend
```

### Step 5: Commit

```bash
git add rtp_llm/cpp/cache/BUILD rtp_llm/cpp/cuda/BUILD bazel/device_defs.bzl
git commit -m "feat: add Ascend to BUILD select() blocks across codebase"
```

---

## Task 12: 集成验证

**目标:** 验证 Phase 1 全部改动在 Ascend 环境下的完整可用性。

### Step 12.1: 编译全量验证

```bash
# 编译核心模块
bazel build //rtp_llm/cpp/core:exec_ops --config=ascend
bazel build //rtp_llm/cpp/cache:cache_core --config=ascend
bazel build //rtp_llm/cpp/ascend:all --config=ascend
bazel build //rtp_llm/cpp/models:py_wrapped_model --config=ascend
```

### Step 12.2: 运行所有 Ascend 单元测试

```bash
bazel test //rtp_llm/cpp/ascend/... --config=ascend
```

### Step 12.3: 端到端内存验证脚本

```python
# tests/ascend/test_memory_e2e.py
"""Phase 1 端到端内存验证"""
import torch
import torch_npu  # 需要安装

def test_npu_basic_allocation():
    """验证 NPU 设备可见且可分配 tensor"""
    assert torch.npu.is_available(), "NPU not available"
    assert torch.npu.device_count() > 0, "No NPU devices found"

    device = torch.device("npu:0")
    t = torch.randn(100, 100, device=device)
    assert t.device.type == "npu"
    assert t.device.index == 0
    print(f"[PASS] NPU basic allocation: tensor shape={t.shape}, device={t.device}")


def test_npu_memory_info():
    """验证 NPU 内存信息可查询"""
    total_mem = torch.npu.get_device_properties(0).total_memory
    assert total_mem > 0, "Total NPU memory reported as 0"
    allocated = torch.npu.memory_allocated(0)
    print(f"[PASS] NPU memory: total={total_mem/1024/1024:.0f} MiB, allocated={allocated/1024/1024:.0f} MiB")


def test_npu_tensor_operations():
    """验证 NPU 上基本 tensor 操作"""
    device = torch.device("npu:0")
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)

    c = a + b
    assert c.device.type == "npu"
    assert c.shape == (100, 100)

    d = torch.matmul(a, b)
    assert d.device.type == "npu"
    assert d.shape == (100, 100)
    print(f"[PASS] NPU tensor operations: add, matmul OK")


def test_npu_host_pinned_memory():
    """验证 Host pinned memory 分配"""
    t = torch.randn(100, 100).pin_memory()
    assert t.is_pinned()
    t_npu = t.to(device="npu:0", non_blocking=True)
    assert t_npu.device.type == "npu"
    print(f"[PASS] Host pinned memory → NPU transfer OK")


def test_npu_stream():
    """验证 NPU Stream 可创建和同步"""
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        t = torch.randn(1000, 1000, device="npu:0")
        _ = t @ t
    s.synchronize()
    print(f"[PASS] NPU Stream create/sync OK")


if __name__ == "__main__":
    test_npu_basic_allocation()
    test_npu_memory_info()
    test_npu_tensor_operations()
    test_npu_host_pinned_memory()
    test_npu_stream()
    print("\n=== Phase 1 端到端内存验证全部通过 ===")
```

```bash
python tests/ascend/test_memory_e2e.py
```

> **建议 (非阻塞):** 当前 E2E 测试通过 torch_npu Python API 验证，未直接覆盖 C++ `Allocator<ASCEND>` 代码路径（仅在 `bazel test` 单元测试中覆盖）。若需要 binding-layer 级别的集成验证，可增加一个 C++ Python binding 测试函数，暴露 `Allocator<ASCEND>::malloc/free` 到 Python 侧测试。该建议在 Phase 4 (Python Binding) 中实现。

### Step 12.4: 验收清单

| 验收项 | 验收标准 | 验证方法 |
|--------|---------|---------|
| 类型枚举 | `DeviceType::Ascend = 6` 存在 | 编译检查 |
| Allocator | `Allocator<ASCEND>` 可正确分配/释放 NPU 内存 | 单元测试 |
| Host Allocator | `Allocator<ASCEND_HOST>` 可正确分配/释放 pinned 内存 | 单元测试 |
| 设备查询 | `getDeviceCount()`, `getDeviceMemoryInfo()` 返回正确值 | 单元测试 |
| BlockPool | `BlockPool` 在 NPU 上创建 tensor 而非 CUDA | 编译+集成测试 |
| ExecOps | `getGpuExecStatus()` 返回 NPU 内存信息 | 编译+运行 |
| PyWrappedModel | 无硬编码 `torch::kCUDA` / `.cuda()` | `rg` 搜索为空 |
| Stream 管理 | `GET_CURRENT_STREAM()` 在 Ascend 下获取 NPU stream | 编译 |
| BUILD select | 所有相关 BUILD 文件包含 `using_ascend` 分支 | `rg` 搜索 |
| 端到端 | tensor 分配到 NPU、host-NPU 传输、stream 同步 | Python 脚本 |

### Step 12.5: Commit

```bash
git add tests/ascend/
git commit -m "test: add Phase 1 end-to-end memory verification tests"
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `torch_npu` 未安装或 API 不稳定 | `torch::kNPU`, `c10_npu::getCurrentNPUStream()` 不可用 | 先用 `aclrt` C API 直接封装，绕过 `torch_npu` 的 stream 管理 |
| `aclrtMalloc` 性能不如 `cudaMalloc` | 内存分配延迟高 | 使用 `ACL_MEM_MALLOC_HUGE_FIRST` 策略，后续优化池化分配 |
| CANN `aclrtGetMemInfo` 返回值与 `cudaMemGetInfo` 语义不同 | 内存评估不准 | 验证 `ACL_HBM_MEM` 语义，必要时用 `ACL_DDR_MEM` 替代 |
| `PyWrappedModel` 改动面广 | 引入回归 bug | 每处改动后立即编译验证，利用 `getTorchDevice()` 统一入口 |

---

## Phase 1 完成后状态

```
✅ DeviceType::Ascend 枚举可用
✅ Allocator<ASCEND> / Allocator<ASCEND_HOST> 可用
✅ BlockPool 可在 NPU 上分配 KV Cache 内存
✅ ExecOps 可查询 NPU 内存状态
✅ PyWrappedModel 无硬编码 CUDA 设备
✅ Stream/Event 可通过 Ascend API 管理
✅ BUILD select() 包含 Ascend 分支
✅ 端到端: tensor 可正确分配到 NPU 设备
```

**Phase 1 完成后可进入 Phase 2 (GEMM 算子适配)**

---

## 依赖关系图

```
Task 1 (枚举)
  └→ Task 2 (Shim 头文件)
       └→ Task 3 (Host Utils) ←── 需要 Task 1 + Task 2
            └→ Task 4 (Allocator) ←── 需要 Task 3
                 └→ Task 5 (BlockPool) ←── 需要编译通过
                      └→ Task 6 (ExecOps) ←── 需要编译通过
                           └→ Task 7 (MemoryEval) ←── 需要编译通过
                                └→ Task 8 (PyWrappedModel) ←── 最大改动
                                     └→ Task 9 (CudaOps) ←── 需要编译通过
                                          └→ Task 10 (Stream 宏) ←── 需要 torch_npu
                                               └→ Task 11 (BUILD select) ←── 贯穿全部
                                                    └→ Task 12 (集成验证)
```

**可并行执行的 Task:**
- Task 5, 6, 7 可并行（独立的编译单元）
- Task 11 与 Task 5-10 可交叉进行

---

## 附录: 审核修正记录 (v0.1 → v0.2)

本文档已根据审核意见修正以下问题：

| 编号 | 严重程度 | 问题 | 修正措施 | 影响范围 |
|------|---------|------|---------|---------|
| 1 | **严重** | `ascend_type_utils.h` 缺少 `#include <c10/core/ScalarType.h>`，`at::ScalarType` 未定义 | 已在 Task 2 Step 2 补充 include | 编译 |
| 2 | **严重** | `torch_npu` stream API 头文件路径 (`torch_npu/csrc/core/NPUStream.h`) 和命名空间 (`c10_npu`) 在不同版本间存在差异 | 已在 Task 10 添加兼容性检查脚本 | 编译/链接 |
| 3 | **严重** | `bazel/device_defs.bzl` 中 `device_impl_target()` 返回 `//rtp_llm/cpp/ascend:ascend_impl`，该 target 在 Phase 1 尚未创建 | 已在 Task 11 Step 2 添加条件注释，指示 Phase 2 创建 target 后取消注释 | 编译 |
| 4 | 中等 | BlockPool 中 `pin_memory()` 虽在 HOST 分支，但设备选择逻辑缺乏防御性注释 | 已在 Task 5 Step 1 添加防御性注释 | 维护 |
| 5 | 中等 | `getTorchCudaDevice()` 函数名在 Ascend 下语义不一致 | 已在 Task 6 Step 2 添加 rename 注释，计划在 Phase 4 命名统一 | 可读性 |
| 6 | 中等 | Task 8 中 PyWrappedModel 行号引用可能因前置提交而漂移 | 已在 Task 8 Step 0 添加行号漂移警告和重新定位步骤 | 执行 |
| 7 | 中等 | `multiMergeCopy` 实现使用 CPU `std::memcpy`，若 dst 在 NPU 上则无效 | 已在 Task 9 Step 2 添加运行时检查注释，确认 CUDA 上游版本行为一致 | 正确性 |
| 8 | 轻微 | `getDevice()`/`getDeviceCount()` 使用 static 缓存，多卡场景可能返回错误 | 已在 Task 3 Step 2 添加单卡假设注释 | 扩展性 |
| 9 | 轻微 | E2E 测试未覆盖 C++ Allocator 代码路径（仅在 bazel test 中覆盖） | 已在 Task 12 Step 12.3 添加 binding-layer 分配测试建议 | 测试覆盖 |
