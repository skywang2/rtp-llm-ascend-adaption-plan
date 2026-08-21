#!/bin/bash
# 容器内 bazel 编译 rtp-llm 框架（后台运行，日志轮询）
# 用法: bash build_framework.sh [CONTAINER_NAME]
set -u
CNAME="${1:-rtp-llm-cann-9.2.0-weekly}"
REPO="/mnt/docker/w30060538/rtp-llm-npu"
LOG="/mnt/docker/w30060538/logs/bazel_build.log"
mkdir -p "$(dirname "$LOG")"

cat > /mnt/docker/w30060538/logs/bazel_build_inner.sh <<EOF
#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
export PYTHON_BIN_PATH=/root/miniconda3/envs/rtp-env/bin/python3
echo "startup --output_user_root=/mnt/docker/bazel_cache" > /root/.bazelrc
cd $REPO
echo "BUILD_START: \$(date)"
bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend
echo "BUILD_EXIT=\$? : \$(date)"
EOF
chmod +x /mnt/docker/w30060538/logs/bazel_build_inner.sh

echo "### 在容器 $CNAME 内后台编译，日志 -> $LOG"
docker exec -d "$CNAME" bash -c "bash /mnt/docker/w30060538/logs/bazel_build_inner.sh > $LOG 2>&1"
echo "started. 轮询: tail -f $LOG"
echo "产物(成功后): $REPO/bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl"
