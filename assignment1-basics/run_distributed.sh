#!/bin/bash

# 分布式训练启动脚本
# 用法: ./run_distributed.sh [num_gpus]

# 设置默认GPU数量
NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# 检查GPU是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令未找到，请检查NVIDIA驱动是否安装"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "错误: 未检测到可用的GPU"
    exit 1
fi

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "警告: 请求的GPU数量($NUM_GPUS)超过可用数量($AVAILABLE_GPUS)，使用最大可用数量"
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "可用GPU数量: $AVAILABLE_GPUS"
echo "使用GPU数量: $NUM_GPUS"

# 设置PyTorch分布式训练环境变量
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=1

# 启动分布式训练
echo "启动分布式训练..."
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "检测到 $NUM_GPUS 个GPU，启动分布式训练..."
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_distributed.py
else
    echo "警告：检测到少于2个GPU，建议使用单GPU训练脚本"
    python train_distributed.py
fi

echo "训练完成!"