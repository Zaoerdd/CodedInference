#!/bin/bash

# 设置 Master IP 和 Worker 数量 (根据 master.py 中的 k 和 r)
MASTER_IP="127.0.0.1"
MODEL_NAME="vgg16"
NUM_WORKERS=2  # 因为 k=1, r=1

echo "--- 正在启动 $NUM_WORKERS 个 Workers... ---"

# 启动 Workers 进程，并将日志重定向到文件
WORKER_PIDS=()
for i in $(seq 1 $NUM_WORKERS)
do
    echo "启动 worker_$i (PID 将被捕获)"
    python worker.py --master $MASTER_IP --model $MODEL_NAME > worker_$i.log 2>&1 &
    WORKER_PIDS+=($!)
done

echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo "等待 5 秒让 Workers 初始化..."
sleep 5

# 启动 Master 进程
echo "--- 正在启动 Master... ---"
python master.py > master.log 2>&1
echo "--- Master 已完成 ---"


# 清理
echo "正在终止所有 Worker 进程..."
for PID in "${WORKER_PIDS[@]}"; do
    kill $PID
done

echo "测试完成。请检查 'master.log' 和 'worker_*.log'。 "