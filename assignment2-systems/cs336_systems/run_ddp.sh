#!/bin/bash

# 获取 GPU 数量，如果未提供则使用所有可用 GPU
NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

# 检查 GPU 可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found."
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available. Using all available."
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "=================================================="
echo "Starting DDP Training with $NUM_GPUS GPUs"
echo "=================================================="

# 设置环境变量
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
# export NCCL_DEBUG=INFO

# ==================== 默认全部开启的优化选项 ====================
# 通过注释掉某些行来关闭特定功能

PERF_ARGS="--enable_tf32 --enable_amp --amp_dtype bfloat16 --enable_compile"

# 分布式策略参数 (默认使用最高级的分桶策略 + 优化器分片)
# 如果想用 naive DDP，可以改 --ddp_strategy naive
# 如果想用单参数 DDP，可以改 --ddp_strategy individual
DIST_ARGS="--ddp_strategy bucketed --bucket_size_mb 25.0 --sharded_optim"

ALL_ARGS="$PERF_ARGS $DIST_ARGS"

echo "Running with arguments: $ALL_ARGS"
echo ""

if [ "$NUM_GPUS" -ge 2 ]; then
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py $ALL_ARGS
else
    # 单卡模式下 ddp_strategy 仍然有效（除了通信部分）
    python train.py $ALL_ARGS
fi
