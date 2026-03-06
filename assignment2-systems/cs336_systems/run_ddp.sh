#!/bin/bash
# DDP分布式训练启动脚本（含通信优化）
# 
# 使用方法:
#   ./run_ddp.sh [GPU数量]
# 
# 示例:
#   ./run_ddp.sh 2    # 使用2个GPU
#   ./run_ddp.sh 4    # 使用4个GPU
#   ./run_ddp.sh      # 自动检测所有可用GPU

# ==================== 全局配置参数 ====================
# GPU配置
NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}

# DDP优化模式 (0-2):
#   0 = 无优化（朴素DDP，每个参数单独通信）
#   1 = 只扁平化（避免分桶overhead）
#   2 = 标准分桶（启用计算通信重叠）
DDP_MODE=1

# 性能优化开关（布尔值：true/false）
ENABLE_TF32=true              # 启用TF32加速（Ampere架构GPU）
ENABLE_AMP=true               # 启用自动混合精度
AMP_DTYPE="bfloat16"          # AMP数据类型：bfloat16 或 float16
ENABLE_COMPILE=true           # 启用torch.compile编译优化
COMPILE_MODE="default"        # 编译模式：default, reduce-overhead, max-autotune

# ====================================================

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

# ==================== NCCL通信优化参数 ====================
# export NCCL_DEBUG=INFO  # 调试时取消注释，生产环境使用WARN
export NCCL_IB_DISABLE=1           # 单机训练禁用InfiniBand（多机时设为0）
export NCCL_P2P_DISABLE=0          # 启用GPU间P2P通信（单机多卡必须）
export NCCL_SHM_DISABLE=0          # 启用共享内存传输（单机高速）

# 通信性能调优
export NCCL_BUFFSIZE=8388608       # 通信缓冲区大小=8MB（默认4MB，增大可提升带宽利用率）
export NCCL_NTHREADS=4             # NCCL通信线程数（2-8之间，根据GPU数量调整）
export NCCL_NSOCKS_PERTHREAD=4     # 每个线程的socket数（提升多流并发）
export NCCL_SOCKET_NTHREADS=4      # Socket通信线程数
export NCCL_MIN_NCHANNELS=4        # 最小通信通道数（增加并行度）

# 针对all-reduce的优化（DDP核心操作）
# 根据GPU数量自动选择最优算法
if [ "$NUM_GPUS" -le 8 ]; then
    export NCCL_ALGO=Tree          # ≤8卡：Tree算法（低延迟）
else
    export NCCL_ALGO=Ring          # >8卡：Ring算法（高带宽）
fi
export NCCL_PROTO=Simple           # 协议选择：Simple（低延迟）或LL（大带宽）

# PyTorch/CUDA优化
export OMP_NUM_THREADS=1           # 避免CPU线程竞争
export CUDA_LAUNCH_BLOCKING=0      # 确保异步执行（不要改成1！）

# 构建训练参数
TRAIN_ARGS=""

# TF32优化
if [ "$ENABLE_TF32" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --enable_tf32"
fi

# AMP混合精度
if [ "$ENABLE_AMP" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --enable_amp --amp_dtype $AMP_DTYPE"
fi

# torch.compile编译优化
if [ "$ENABLE_COMPILE" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --enable_compile --compile_mode $COMPILE_MODE"
fi

# DDP优化模式
TRAIN_ARGS="$TRAIN_ARGS --ddp_mode $DDP_MODE"

echo "性能优化配置:"
echo "  - TF32: $([ "$ENABLE_TF32" = true ] && echo "启用" || echo "禁用")"
echo "  - AMP: $([ "$ENABLE_AMP" = true ] && echo "启用 ($AMP_DTYPE)" || echo "禁用")"
echo "  - torch.compile: $([ "$ENABLE_COMPILE" = true ] && echo "启用" || echo "禁用")"
echo "  - compile_mode: $COMPILE_MODE"

echo ""
echo "DDP通信优化模式: $DDP_MODE"
case $DDP_MODE in
    0)
        echo "  - 模式0: 无优化（朴素DDP）"
        echo "  - 每个参数单独通信，无扁平化"
        ;;
    1)
        echo "  - 模式1: 只扁平化（默认，推荐2-3卡）"
        echo "  - 避免分桶和重叠的overhead"
        ;;
    2)
        echo "  - 模式2: 标准分桶（推荐4-7卡）"
        echo "  - 分桶 + 计算通信重叠 + 零拷贝"
        ;;
esac

echo "  - NCCL Buffer: 8MB"
echo "  - NCCL Channels: 4+"
echo "  - NCCL Algorithm: ${NCCL_ALGO} (${NUM_GPUS}卡)"
echo "  - P2P + Shared Memory: 启用"
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
        ddp.py $TRAIN_ARGS
else
    echo "检测到 $NUM_GPUS 个GPU，启动单GPU训练..."
    python ddp.py $TRAIN_ARGS
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
