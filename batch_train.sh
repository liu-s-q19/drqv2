#!/bin/bash

TASK_DIR="cfgs/task"
EXP_DIR="exp_local"
LOG_DIR="logs"
TRAIN_SCRIPT="train.py"
AGENT_NAME="drqv2"   # 或 vsac
GPU_LIST=(0 1)        # GPU编号
NUM_WORKERS=2         # 最大并发数

mkdir -p "$LOG_DIR"

# 获取所有任务文件
TASK_FILES=($TASK_DIR/*.yaml)
NUM_TASKS=${#TASK_FILES[@]}
CUR_IDX=0

running_pids=()

for TASK_FILE in "${TASK_FILES[@]}"; do
    TASK_NAME=$(basename "$TASK_FILE" .yaml)
    EXP_PATH="$EXP_DIR/$TASK_NAME"
    LOG_PATH="$LOG_DIR/$TASK_NAME.log"
    GPU_ID=${GPU_LIST[$((CUR_IDX % ${#GPU_LIST[@]}))]}

    # 跳过已完成
    if [ -d "$EXP_PATH" ]; then
        echo "已存在 $EXP_PATH，跳过 $TASK_FILE"
        ((CUR_IDX++))
        continue
    fi

    echo "开始训练: $TASK_FILE，分配到GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$TRAIN_SCRIPT" +task="$TASK_NAME" agent="$AGENT_NAME" > "$LOG_PATH" 2>&1 &

    running_pids+=($!)
    ((CUR_IDX++))

    # 控制并发数
    while [ "${#running_pids[@]}" -ge "$NUM_WORKERS" ]; do
        for i in "${!running_pids[@]}"; do
            if ! kill -0 "${running_pids[$i]}" 2>/dev/null; then
                unset 'running_pids[i]'
            fi
        done
        running_pids=("${running_pids[@]}") # 清理已结束的
        sleep 5
    done

done

# 等待所有任务结束
wait
echo "所有任务已完成。" 