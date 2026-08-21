#!/bin/bash
# 构建镜像：预下载 CANN → 本地 HTTP 服务 → docker build
# 用法: bash build_image.sh [BUILD_DIR]
set -u
BUILD_DIR="${1:-/mnt/docker/w30060538/rtp-llm-porting/docs/Qwen3.5/容器构建}"
DL="/mnt/docker/w30060538/download"
IMAGE="rtp-llm-a5-image:cann-9.2.0-x86_64"
BASE="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260814000324571"
BASE_IMG="quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86"

echo "### 1) 预下载 CANN (legacy 路径，-c 续传)"
mkdir -p "$DL" && cd "$DL"
wget -c -q --show-progress -O "Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run" \
  "$BASE/Ascend-cann-toolkit_9.2.0~weekly.20260814.01_linux-x86_64.run"
wget -c -q --show-progress -O "Ascend-cann-950-ops_9.2.0~weekly.20260814.01_linux-x86_64.run" \
  "$BASE/Ascend-cann-950-ops_9.2.0~weekly.20260814.01_linux-x86_64.run"

echo "### 2) 本地 HTTP 服务（独立 docker 容器，避免 shell 超时被杀）"
docker rm -f cann-fileserver 2>/dev/null
docker run -d --name cann-fileserver --network host \
  --entrypoint /usr/local/python3.11.15/bin/python3 \
  -v "$DL:/data" -w /data "$BASE_IMG" \
  -m http.server 18080 --bind 0.0.0.0
sleep 3
curl -s -o /dev/null -w "fileserver: %{http_code}\n" --max-time 5 http://127.0.0.1:18080/

echo "### 3) docker build"
cd "$BUILD_DIR"
docker build --network host --build-arg BASE_URL=http://127.0.0.1:18080 \
  -t "$IMAGE" . 2>&1 | tee /tmp/docker_build_cann920.log
BUILD_EXIT=${PIPESTATUS[0]}

echo "### 4) 收尾"
docker rm -f cann-fileserver 2>/dev/null
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" = "0" ] && docker images | grep "$IMAGE"
