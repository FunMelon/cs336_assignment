#!/bin/bash
# 设置默认GPU数量
NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

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

echo "================================"
echo "DDP 分布式训练启动"
echo "================================"
echo "可用GPU数量: $AVAILABLE_GPUS"
echo "使用GPU数量: $NUM_GPUS"

# 显示GPU信息
echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n $NUM_GPUS
echo "================================"
echo ""

# 设置PyTorch分布式训练环境变量
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO  # 调试时取消注释
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=1

# 默认开启全部性能优化
PERF_ARGS="--enable_tf32 --enable_amp --amp_dtype bfloat16 --enable_compile"

# 获取额外的参数（从第二个参数开始）
shift
EXTRA_ARGS="$@"

echo "性能优化: $PERF_ARGS"
if [ -n "$EXTRA_ARGS" ]; then
    echo "额外参数: $EXTRA_ARGS"
fi
echo ""

# 启动分布式训练
echo "启动分布式训练..."
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "检测到 $NUM_GPUS 个GPU，启动多GPU分布式训练..."
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        ddp.py $PERF_ARGS $EXTRA_ARGS
else
    echo "检测到 $NUM_GPUS 个GPU，启动单GPU训练..."
    python ddp.py $PERF_ARGS $EXTRA_ARGS
fi

EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成!"
else
    echo "✗ 训练失败 (退出码: $EXIT_CODE)"
fi
echo "================================"

exit $EXIT_CODE
