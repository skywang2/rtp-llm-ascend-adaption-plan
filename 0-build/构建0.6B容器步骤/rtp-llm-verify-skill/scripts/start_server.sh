#!/bin/bash
# 拉起 rtp-llm 推理服务（后台运行）
# 用法: bash start_server.sh [CONTAINER_NAME] [PORT]
set -u
CNAME="${1:-rtp-llm-cann-9.2.0-weekly}"
PORT="${2:-9100}"
REPO="/mnt/docker/w30060538/rtp-llm-npu"
WEIGHTS="/mnt/docker/weights/Qwen3-0.6B"
LOG="/mnt/docker/w30060538/logs/start_server_${PORT}.log"
mkdir -p "$(dirname "$LOG")"

# 端口占用检查
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
  echo "WARN: 端口 $PORT 已被占用（可能既有服务），仍继续启动"
fi

docker exec -d "$CNAME" bash -c "
  source /usr/local/Ascend/cann/set_env.sh
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtp-env
  cd $REPO
  python -m rtp_llm.start_server \
    --checkpoint_path=$WEIGHTS \
    --model_type=qwen_3 \
    --start_port=$PORT \
  > $LOG 2>&1
"
echo "started on port $PORT. 日志: $LOG"
echo "就绪标志: 日志出现 'initLogger log_file_path' 且 ss -tlnp | grep $PORT 见 rtp_llm_fronten 监听"
