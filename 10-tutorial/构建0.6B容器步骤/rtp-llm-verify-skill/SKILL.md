# rtp-llm (Ascend NPU) Dockerfile 构建与推理服务验证 Skill

> 基于 CANN 9.2.0 + torch_npu 2.9.0.post3 + Qwen3-0.6B 的实际验证流程整理。
> 目标：① 验证 Dockerfile 能否构建容器镜像；② 验证 rtp-llm 推理服务能否拉起并完成 Qwen3-0.6B 推理。
>
> **本 skill 不涉及 rtp-llm 框架源码 `-Werror`/CANN 头告警的编译处理**（该问题单独解决）。

---

## 触发词
`rtp-llm-verify`、验证 rtp-llm 构建、验证 rtp-llm 推理、rtp-llm 容器构建验证、Qwen3-0.6B 推理验证、CANN 9.2.0 镜像构建。

## 适用环境
| 项 | 规格 |
|---|---|
| 硬件 | x86_64 + Ascend 950PR (8× davinci) |
| 基础镜像 | `quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86` |
| CANN | 9.2.0 (weekly `20260814.01`) |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post3 |
| triton-ascend | 3.2.2 |
| Python / Bazel | 3.10 / 6.4.0 |
| 验证模型 | Qwen3-0.6B |

## 关键路径约定（按本环境）
- Dockerfile 目录：`<porting>/docs/Qwen3.5/容器构建/`
- rtp-llm 仓库：`/mnt/docker/w30060538/rtp-llm-npu`
- CANN/依赖本地缓存：`/mnt/docker/w30060538/download/`
- 模型权重：`/mnt/docker/weights/Qwen3-0.6B`
- 共享 bazel 缓存：`/mnt/docker/bazel_cache`
- 日志目录：`/mnt/docker/w30060538/logs/`

> 以下命令中变量：`$BUILD_DIR`=Dockerfile 所在目录，`$REPO`=rtp-llm-npu 仓库路径，`$DL`=download 缓存目录。

---

## 引用文件（权威文档）

以下三份文档为权威依据，已**直接复制**（非链接）收录于本 skill 的 `references/`。

| 引用文件 | 用途 | 本 skill 对应章节 |
|---|---|---|
| `references/rtp-llm-dockerfile-usage.md` | **Dockerfile 使用说明**：镜像概述、目录结构、安装包获取方式一览、构建步骤、容器创建命令、镜像内关键路径、核心版本信息 | 任务一（构建镜像）依据 |
| `references/rtp_llm_workflow_guide.md` | **完整 Workflow 操作指南**：创建容器→下载仓库→装 CANN/conda/Bazel→框架包管理与源码修改（第9~10章）→编译（11）→安装（12）→启动服务（13）→推理测试（14）→常见问题（15） | 任务二依据第3、9~14章 |
| `references/rtp_llm_version_matrix.md` | **版本配套表**：硬件平台、系统软件、CANN/Ascend 工具链、编译工具链、PyTorch/torch_npu、Docker 镜像信息的完整版本矩阵与来源 URL | 全程版本核对依据 |

> 源文件位置（canonical）：`<porting>/docs/Qwen3.5/容器构建/`（即本 skill 目录的上一级）。`references/` 为副本；若源文档更新，需重新复制以保持一致。
> 执行任务前建议先通读 `references/rtp_llm_version_matrix.md` 核对版本，再按 `references/rtp-llm-dockerfile-usage.md`（任务一）与 `references/rtp_llm_workflow_guide.md`（任务二）操作。

---

## 阶段 0：Pre-flight 检查（只读，执行前必做）

验证下列条件全部满足，避免中途失败浪费长时间构建。

```bash
# 1) Docker + 基础镜像
docker --version                                   # 需可用
docker images | grep "vllm-ascend-daily:main-a5-openEuler-x86"   # 基础镜像应已存在

# 2) NPU 设备 + 驱动文件（容器挂载需要）
ls /dev/davinci0 ... /dev/davinci7 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
ls /usr/local/bin/npu-smi /usr/local/dcmi
ls /usr/local/Ascend/driver/lib64/ /usr/local/Ascend/driver/version.info /etc/ascend_install.info

# 3) 模型权重
ls /mnt/docker/weights/Qwen3-0.6B/model.safetensors

# 4) rtp-llm 仓库（第9~10章修改应已应用）
cd $REPO && git status --short    # 应见 .bazelrc / BUILD / deps/*.bzl / BUILD.tpl 等已改

# 5) 端口占用：9000 常被既有 rtp_llm_frontend 占用
ss -tlnp | grep -E ':9000|:9100'   # 若 9000 被占，推理服务改用 9100

# 6) CANN 下载源可达性（master 目录会过期，需用 legacy 路径）
curl -sI "https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260814000324571/Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run" | head -1
# 期望 HTTP/1.1 200；若 404，说明该 weekly 也已移动，到 .../software/master/ 下找最新目录
```

**Pre-flight 常见阻塞**：
- CANN `master/20260814000324571` 返回 404 → 周构建制品已轮转，改用 `legacy/20260814000324571`（或 master 下最新目录）。
- 基础镜像不存在 → `docker pull quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86`。

---

## 任务一：验证 Dockerfile 能构建容器镜像

### 1.1 预下载 CANN 安装包（先下载，再构建）
Dockerfile 内 `wget` 从远程拉 CANN；为可靠/可复现，先下载到本地缓存，构建时用本地 HTTP 服务供给。

```bash
cd $DL
BASE="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260814000324571"
wget -c -O "Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run" \
  "$BASE/Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run"
wget -c -O "Ascend-cann-950-ops_9.2.0~weekly.20260814.01_linux-x86_64.run" \
  "$BASE/Ascend-cann-950-ops_9.2.0~weekly.20260814.01_linux-x86_64.run"
# 校验大小：toolkit ~1.43GB(1428437193B)、950-ops ~2.97GB(2973979981B)
file *.run   # 应为 "Bourne-Again shell script executable"
```

### 1.2 启动本地 HTTP 服务（关键：用独立 docker 容器，勿用 shell 后台进程）
> ⚠️ 用 `nohup ... &` 起的 HTTP 服务会在 shell 命令超时被杀掉，导致构建容器 `wget` 拿不到文件、报 exit 4。
> **必须用 `docker run -d` 起一个独立容器**，与构建生命周期解耦。

```bash
docker rm -f cann-fileserver 2>/dev/null
docker run -d --name cann-fileserver --network host \
  --entrypoint /usr/local/python3.11.15/bin/python3 \
  -v $DL:/data -w /data \
  quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86 \
  -m http.server 18080 --bind 0.0.0.0
sleep 3
# 验证：宿主机与构建容器都要能访问
curl -sI http://127.0.0.1:18080/Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run | head -1
docker run --rm --network host --entrypoint /bin/bash \
  quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86 \
  -c "wget -q -O /tmp/t 'http://127.0.0.1:18080/Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run' --tries=1 --timeout=15 && ls -la /tmp/t && echo DOWNLOAD_OK"
```

### 1.3 构建镜像
```bash
cd $BUILD_DIR
docker build --network host \
  --build-arg BASE_URL=http://127.0.0.1:18080 \
  -t rtp-llm-a5-image:cann-9.2.0-x86_64 .
# --network host 让构建容器访问宿主机 127.0.0.1:18080
# --build-arg BASE_URL 覆盖 Dockerfile 默认远程路径，改用本地缓存（CANN 不再重复下载 4.4GB）
```
- 预计 15–30 分钟（CANN 解压 + conda + pip 安装）。
- 建议后台运行并轮询日志（见 `scripts/build_image.sh`）。

### 1.4 验证镜像产物
```bash
docker images | grep "rtp-llm-a5-image:cann-9.2.0-x86_64"
docker run --rm --entrypoint /bin/bash rtp-llm-a5-image:cann-9.2.0-x86_64 -c '
  echo "CANN: $([ -d /usr/local/Ascend/cann-9.2.0 ] && echo OK)"
  echo "cann -> $(readlink /usr/local/Ascend/cann)"
  echo "bazel: $(bazel --version 2>/dev/null | head -1)"
  echo "profiling symlink: $([ -e /usr/local/Ascend/cann-9.2.0/include/profiling ] && echo OK)"
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  for p in torch torch_npu triton_ascend; do
    echo "$p: $(pip show $p 2>/dev/null | grep ^Version)"
  done
'
```
**预期**：CANN OK、bazel 6.4.0、torch 2.9.0+cpu、torch_npu 2.9.0.post3、triton_ascend 3.2.2。
> 注：临时容器未挂载 NPU 驱动，`import torch_npu` 会因缺驱动库失败属正常；用 `pip show` 验证包已安装即可。

### 1.5 收尾
```bash
docker rm -f cann-fileserver   # 关闭本地 HTTP 服务
```

### 任务一已知检查点（构建期）
| 现象 | 原因 | 处理 |
|---|---|---|
| Step 1 `wget` exit 4、文件 0 字节 | HTTP 服务被 shell 超时杀掉 | 改用 `docker run -d` 独立容器起 HTTP 服务（1.2） |
| Step(triton-ascend) `No matching distribution found for triton-ascend==3.2.2` | aliyun/PyPI 无此版本 | Dockerfile 该步加 `--extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi` |
| CANN `wget` 404 | 用了过期的 `master/` 周构建目录 | 改 `legacy/` 路径，或 master 下最新目录 |

---

## 任务二：验证 rtp-llm 推理服务（Qwen3-0.6B）

### 2.1 创建容器
复刻 `start_docker.sh` 的可用挂载方案；**用 `-v /mnt:/mnt`（保留宿主机路径）**，使仓库内 `bazel-bin -> /mnt/docker/bazel_cache/...` 符号链接在容器内正确解析。

```bash
docker run -itd --name rtp-llm-cann-9.2.0-weekly \
  --privileged --network host --security-opt label=disable --shm-size=1g \
  --hostname localhost.localdomain \
  -v /mnt:/mnt -v /home:/home -v /root/.cache:/root/.cache \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  rtp-llm-a5-image:cann-9.2.0-x86_64
```
> 容器名按需改；镜像 ENTRYPOINT 已自动 source CANN 9.2.0 环境。

### 2.2 验证容器环境
```bash
docker exec rtp-llm-cann-9.2.0-weekly bash -lc '
  npu-smi info | head -20                       # 8× Ascend950PR 可见
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  python -c "import torch,torch_npu; print(torch.__version__, torch_npu.__version__)"  # 2.9.0+cpu / 2.9.0.post3
  echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"     # =/usr/local/Ascend/cann-9.2.0
'
```

### 2.3 配置 bazel 缓存并编译 rtp-llm 框架
```bash
docker exec rtp-llm-cann-9.2.0-weekly bash -lc '
  # 复用共享缓存（仓库内 bazel-bin 符号链接指向它）
  echo "startup --output_user_root=/mnt/docker/bazel_cache" > /root/.bazelrc
  source /usr/local/Ascend/cann/set_env.sh
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  export PYTHON_BIN_PATH=/root/miniconda3/envs/rtp-env/bin/python3
  cd /mnt/docker/w30060538/rtp-llm-npu
  bazel clean                  # 清编译缓存，强制针对 9.2.0 全量重编（保留外部依赖下载）
  bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend
'
# 产物：bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl
```
> 编译耗时约 5–10 分钟（首次/清缓存后）。后台运行见 `scripts/build_framework.sh`。
> **本步骤不涉及 `-Werror`/CANN 头告警的处理**，若遇到框架源码编译告警，按源码侧方案单独解决。

### 2.4 安装 wheel + proto 软链接 + 验证导入
```bash
docker exec rtp-llm-cann-9.2.0-weekly bash -lc '
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  cd /mnt/docker/w30060538/rtp-llm-npu
  # proto 软链接（必须在仓库内用绝对路径，上溯到仓库根）
  ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py      rtp_llm/cpp/model_rpc/proto/
  ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py  rtp_llm/cpp/model_rpc/proto/
  # 安装 wheel（不能用 *.whl 通配，两个 wheel 会冲突）
  pip install --force-reinstall --no-deps bazel-out/k8-opt/bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl
  python -c "import rtp_llm; print(\"IMPORT_OK\", rtp_llm.__path__[0])"
'
```
**预期**：`IMPORT_OK ...`。以下 WARNING 属正常（workflow Q6，无需处理）：
`Failed to load C++ FusedRopeKVCacheOp`、`fuse is not valid`、`Please install pyav`、`internal_source directory ... found: False`。

### 2.5 补装缺失运行时依赖（仅当服务启动报 ModuleNotFoundError 时）
Dockerfile 的 pip 列表不完整，服务导入会陆续报缺包。**一次补齐**（用 pip diff 对比可用容器可定位全集）：
```bash
docker exec rtp-llm-cann-9.2.0-weekly bash -lc '
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  pip install orjson partial-json-parser dacite blobfile jieba json5 nest-asyncio pycryptodomex lxml
'
```
> pip 关于 `rtp-llm requires fastapi==0.115.6 ...` 的版本不匹配 WARNING 可忽略（既有可用容器同样存在）。
> 建议后续把这些包补入 Dockerfile Step 23 的 pip 列表以开箱即用。

### 2.6 启动推理服务
> 端口选择：9000 常被既有 `rtp_llm_frontend` 占用 → 用 **9100**。

```bash
# 用脚本后台启动，日志写文件便于排查（见 scripts/start_server.sh）
docker exec -d rtp-llm-cann-9.2.0-weekly bash -c '
  source /usr/local/Ascend/cann/set_env.sh
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  cd /mnt/docker/w30060538/rtp-llm-npu
  python -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3-0.6B \
    --model_type=qwen_3 \
    --start_port=9100 \
  > /mnt/docker/w30060538/logs/start_server_9100.log 2>&1
'
```
**就绪标志**：日志出现 `initLogger log_file_path: .../alog.conf`，且 `ss -tlnp | grep 9100` 见 `rtp_llm_fronten` 监听。

### 2.7 健康检查 + 推理测试
```bash
# 健康检查
curl -s http://127.0.0.1:9100/health            # 期望: "ok"
# 模型列表
curl -s http://127.0.0.1:9100/v1/models          # 期望: {"data":[{"id":"qwen_3",...}]}
# OpenAI 格式推理（约 4–30s）
curl -s http://127.0.0.1:9100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你是什么模型"}]}'
```
**预期响应**含 `choices[0].message.content`（模型自述）、`finish_reason:"stop"`、`usage.prompt_tokens≈11`、`aux_info.first_token_cost_time`（约 0.4–2.2s）。

---

## 排障速查

| 现象 | 定位 | 处理 |
|---|---|---|
| `docker build` 卡在某步无输出 | 查 `docker stats` CPU/网络；`docker exec` 进构建容器看 pip 下载 | 多为网络慢，非卡死；用本地 HTTP 服务供给大文件 |
| `ModuleNotFoundError: No module named 'X'` (服务启动) | Dockerfile pip 列表缺包 | 按 2.5 补装；或 pip diff 对比可用容器定位全集 |
| `Port 9000` 启动失败 | 既有 `rtp_llm_frontend` 占用 | 改 `--start_port=9100`，curl 也改 9100 |
| `npu_scatter_pa_kv_cache: Invalid_Argument` | CANN 算子对 head_dim=128 不兼容 | 确认 `ascend_kv_cache_write_op.py` 加了 `cache_mode="Norm"`（workflow 10.5） |
| `ModuleNotFoundError: rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2` | proto 软链接缺失 | 重做 2.4 的 `ln -sf` |
| bazel 编译用旧 CANN 缓存产物 | 共享缓存含旧版本产物 | `bazel clean` 后重编（保留外部依赖） |
| `rtp_llm` 导入报缺 `.so` | 在非仓库目录运行 `python -m` | 必须 `cd` 到 rtp-llm-npu 仓库目录再启动（源码+bazel-bin .so 模式） |

---

## 参考脚本
见同目录 `scripts/`：
- `build_image.sh` — 构建镜像（含本地 HTTP 服务启停）
- `build_framework.sh` — 容器内 bazel 编译（后台运行，日志可轮询）
- `start_server.sh` — 拉起推理服务
- `verify.sh` — 一键验证（健康检查 + 推理）

## 变量与版本小结
| 组件 | 值 |
|---|---|
| CANN 下载源 | `https://ascend.devcloud.huaweicloud.com/.../software/legacy/20260814000324571/` |
| triton-ascend 源 | `--extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi` |
| 镜像 tag | `rtp-llm-a5-image:cann-9.2.0-x86_64` |
| 容器名 | `rtp-llm-cann-9.2.0-weekly`（可改） |
| 推理端口 | 9100（9000 被占时） |
| 模型 | `qwen_3` @ `/mnt/docker/weights/Qwen3-0.6B` |
| 编译命令 | `bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend` |
| 产物 | `bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl` |
