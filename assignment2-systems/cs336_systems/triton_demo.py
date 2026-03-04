"""
Triton 加权求和示例代码

本文件演示了如何使用 Triton 编写 GPU 内核，包括：
1. 前向传播内核 (weighted_sum_fwd)
2. 反向传播内核 (weighted_sum_backward)
3. 与 PyTorch autograd 集成

适合 Triton 编程初学者学习。
"""

import torch
import triton
import triton.language as tl
from math import ceil

# 辅助函数：向上取整除法
def cdiv(a, b):
    """计算 ceil(a / b)，即向上取整的除法"""
    return ceil(a / b)

# 辅助函数：重排张量维度（简化版 einops.rearrange）
def rearrange(x, pattern):
    """
    将张量重排为指定形状
    这里简化处理 "... d -> (...) d" 模式，将所有维度展平为2D
    """
    if pattern == "... d -> (...) d":
        return x.view(-1, x.shape[-1])
    raise NotImplementedError(f"不支持的模式: {pattern}")


# ============================================================================
# 前向传播内核
# ============================================================================
@triton.jit
def weighted_sum_fwd(
    x_ptr,              # 输入张量 x 的指针
    weight_ptr,         # 权重向量的指针
    output_ptr,         # 输出张量的指针
    x_stride_row,       # x 在行方向的步幅（移动到下一行需要跳过多少元素）
    x_stride_dim,       # x 在列方向的步幅（移动到下一列需要跳过多少元素）
    weight_stride_dim,  # 权重向量的步幅（通常为1，因为是连续的）
    output_stride_row,  # 输出向量的步幅（通常为1）
    ROWS,               # 总行数
    D,                  # 每行的维度大小（embedding dimension）
    ROWS_TILE_SIZE: tl.constexpr,  # 每个线程块处理的行数（编译时常量）
    D_TILE_SIZE: tl.constexpr,     # 每次迭代处理的列数（编译时常量）
):
    """
    加权求和的前向传播内核
    
    功能：计算 output[i] = sum(x[i, :] * weight[:])
    即每一行与权重向量做元素乘法后求和
    
    关键概念：
    - program_id: 获取当前线程块的索引，用于确定处理哪部分数据
    - block_ptr: 块指针，用于方便地访问 N 维张量的一个区域
    - boundary_check: 边界检查，处理不能整除的情况
    """
    
    # ========== 步骤1: 确定当前线程块负责处理哪些行 ==========
    # tl.program_id(0) 返回当前线程块在第0个网格维度的索引
    # 比如有1000行数据，ROWS_TILE_SIZE=16，那么会启动 ceil(1000/16)=63 个线程块
    # 每个线程块通过 program_id 知道自己负责哪16行
    row_tile_idx = tl.program_id(0)  # 类似于cuda编程中的 blockIdx.x
    
    # ========== 步骤2: 创建块指针 (Block Pointers) ==========
    # 块指针是 Triton 提供的高级抽象，让我们可以方便地：
    # 1. 从多维张量中选择一个矩形区域
    # 2. 自动处理边界情况
    # 3. 方便地移动选择区域
    
    # 创建输入张量 x 的块指针
    # 选择从 (row_tile_idx * ROWS_TILE_SIZE, 0) 开始的
    # (ROWS_TILE_SIZE, D_TILE_SIZE) 大小的块
    x_block_ptr = tl.make_block_ptr(
        x_ptr,                                      # 张量起始指针
        shape=(ROWS, D),                            # 张量的完整形状（用于边界检查）
        strides=(x_stride_row, x_stride_dim),       # 每个维度的步幅
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # 块的起始坐标
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),  # 要加载/存储的块大小
        order=(1, 0),                               # 内存布局顺序（从主到次）
                                                    # (1,0) 表示列方向变化更快（行优先存储）
    )
    
    # 创建权重向量的块指针（1维）
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),                    # 权重是1维向量
        strides=(weight_stride_dim,),  # 步幅
        offsets=(0,),                  # 从头开始
        block_shape=(D_TILE_SIZE,),    # 每次加载 D_TILE_SIZE 个元素
        order=(0,),                    # 1维只有一种顺序
    )
    
    # 创建输出向量的块指针
    # 每行输出一个标量，所以形状是 (ROWS_TILE_SIZE,)
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),                              # 输出是1维向量
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),   # 对应当前处理的行
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    
    # ========== 步骤3: 初始化输出缓冲区 ==========
    # 使用 float32 累加以保持数值精度
    # 每行一个累加器，共 ROWS_TILE_SIZE 个
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    # ========== 步骤4: 循环处理整个 embedding 维度 ==========
    # 由于 D 可能很大，我们分块处理，每次处理 D_TILE_SIZE 列
    # tl.cdiv(D, D_TILE_SIZE) 计算需要多少次迭代
    for i in range(tl.cdiv(D, D_TILE_SIZE)):  # triton代码内部必须使用tl.xxx系列函数
        # 加载当前块的数据
        # boundary_check: 边界检查，处理最后一个块可能不完整的情况
        #   - (0, 1) 表示检查第0维（行）和第1维（列）
        # padding_option: 越界的元素用0填充
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        
        # 计算加权和
        # weight[None, :] 将 (D_TILE_SIZE,) 广播为 (1, D_TILE_SIZE)
        # row * weight[None, :] 得到 (ROWS_TILE_SIZE, D_TILE_SIZE)
        # tl.sum(..., axis=1) 沿列方向求和，得到 (ROWS_TILE_SIZE,)
        output += tl.sum(row * weight[None, :], axis=1)
        
        # 移动块指针到下一个 tile
        # advance() 方法按指定的坐标增量移动块指针
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))      # 列方向移动
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    
    # ========== 步骤5: 将结果写入输出张量 ==========
    # 同样需要边界检查，因为最后一个块可能不完整
    tl.store(output_block_ptr, output, boundary_check=(0,))


# ============================================================================
# 反向传播内核
# ============================================================================
@triton.jit
def weighted_sum_backward(
    # 输入指针
    x_ptr,                     # 前向传播保存的输入 x
    weight_ptr,                # 前向传播保存的权重
    grad_output_ptr,           # 上游传来的梯度 (∂L/∂output)
    # 输出指针
    grad_x_ptr,                # 输出：x 的梯度 (∂L/∂x)
    partial_grad_weight_ptr,   # 输出：权重梯度的部分结果（需要后续归约）
    # 步幅参数
    stride_xr, stride_xd,      # x 的行/列步幅
    stride_wd,                 # weight 的步幅
    stride_gr,                 # grad_output 的步幅
    stride_gxr, stride_gxd,    # grad_x 的行/列步幅
    stride_gwb, stride_gwd,    # partial_grad_weight 的行/列步幅
    # 形状参数
    NUM_ROWS, D,
    # 编译时常量
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    加权求和的反向传播内核
    
    数学推导：
    前向: output[i] = sum_j(x[i,j] * weight[j])
    
    反向（链式法则）：
    - ∂L/∂x[i,j] = ∂L/∂output[i] * weight[j]  （外积）
    - ∂L/∂weight[j] = sum_i(∂L/∂output[i] * x[i,j])  （需要跨行归约）
    
    实现策略：
    - grad_x 可以直接计算并写入
    - grad_weight 需要跨所有行归约，但不同线程块处理不同行
      所以每个线程块先写入部分结果，最后在 PyTorch 中归约
    """
    
    # 当前线程块索引和总线程块数
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)  # 总共有多少个线程块
    
    # ========== 创建所有需要的块指针 ==========
    
    # 上游梯度（每行一个标量）
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    
    # 输入 x
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # 权重
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    
    # x 的梯度（与 x 形状相同）
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # 权重梯度的部分结果
    # 形状是 (n_row_tiles, D)，每个线程块写入一行
    # 之后需要在 PyTorch 中对第0维求和得到最终的 grad_weight
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),          # 当前线程块写入第 row_tile_idx 行
        block_shape=(1, D_TILE_SIZE),       # 每次写入 1 行 D_TILE_SIZE 列
        order=(1, 0),
    )
    
    # ========== 循环处理 embedding 维度 ==========
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载上游梯度
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)
        
        # ---------- 计算 grad_x ----------
        # grad_x[i,j] = grad_output[i] * weight[j]
        # 这是一个外积运算
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        # grad_output[:, None] 形状: (ROWS_TILE_SIZE, 1)
        # weight[None, :] 形状: (1, D_TILE_SIZE)
        # 广播相乘得到 (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        
        # ---------- 计算 partial_grad_weight ----------
        # grad_weight[j] = sum_i(grad_output[i] * x[i,j])
        # 由于不同线程块处理不同的行，这里先计算当前块负责的行的贡献
        # 最后在外部将所有块的结果相加
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        # row * grad_output[:, None] 形状: (ROWS_TILE_SIZE, D_TILE_SIZE)
        # tl.sum(..., axis=0) 沿行方向求和，得到 (D_TILE_SIZE,)
        # keep_dims=True 保持维度为 (1, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # 第0维不会越界
        
        # 移动所有指针到下一个 tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


# ============================================================================
# PyTorch autograd 集成
# ============================================================================
class WeightedSumFunc(torch.autograd.Function):
    """
    将 Triton 内核封装为 PyTorch 可微分函数
    
    继承 torch.autograd.Function 需要实现：
    - forward(): 前向传播，保存反向传播需要的张量
    - backward(): 反向传播，根据上游梯度计算输入的梯度
    
    使用方法：
        y = WeightedSumFunc.apply(x, weight)
    """
    
    @staticmethod
    def forward(ctx, x, weight):
        """
        前向传播
        
        参数:
            ctx: 上下文对象，用于保存反向传播需要的信息
            x: 输入张量，形状 (..., D)，最后一维是 embedding
            weight: 权重向量，形状 (D,)
        
        返回:
            输出张量，形状 (...)，即去掉最后一维
        """
        # 保存原始形状，用于返回时恢复
        D = x.shape[-1]
        output_dims = x.shape[:-1]
        input_shape = x.shape
        
        # 将输入展平为2D: (..., D) -> (N, D)
        # 这样内核只需要处理2D情况
        x = rearrange(x, "... d -> (...) d")
        
        # 保存张量供反向传播使用
        # ctx.save_for_backward 只能保存张量
        ctx.save_for_backward(x, weight)
        
        # 输入验证
        assert len(weight.shape) == 1 and weight.shape[0] == D, "维度不匹配"
        assert x.is_cuda and weight.is_cuda, "需要 CUDA 张量"
        assert x.is_contiguous(), "指针运算假设 x 是连续的"
        
        # 设置 tile 大小
        # D_TILE_SIZE: 大约循环16次处理完 embedding 维度
        # ROWS_TILE_SIZE: 每个线程块处理16行
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape
        
        # 创建输出张量（未初始化，需要内核填充所有值）
        y = torch.empty(output_dims, device=x.device)   # 数据在GPU上
        
        # 启动内核
        # grid = (cdiv(n_rows, ROWS_TILE_SIZE),) 表示启动多少个线程块
        # 这是一个1D网格
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,              # 输入
            y,                      # 输出
            x.stride(0), x.stride(1),   # x 的步幅
            weight.stride(0),           # weight 的步幅
            y.stride(0),                # y 的步幅
            ROWS=n_rows, D=D,           # 形状参数
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        
        # 返回与输入 batch 形状相同的输出
        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播
        
        参数:
            ctx: 上下文对象，包含 forward 保存的信息
            grad_out: 上游传来的梯度，形状与 forward 输出相同
        
        返回:
            (grad_x, grad_weight): 各输入的梯度
        """
        # 恢复保存的张量
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape
        
        # 创建输出张量
        # partial_grad_weight: 每个线程块贡献一行，形状 (n_blocks, D)
        # 之后需要 sum(axis=0) 得到最终的 grad_weight
        partial_grad_weight = torch.empty(
            (cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device,
            dtype=x.dtype
        )
        grad_x = torch.empty_like(x)
        
        # 确保 grad_out 是连续的（内核需要）
        grad_out = grad_out.contiguous()
        
        # 启动反向内核
        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,                                      # 前向保存的输入
            grad_out,                                       # 上游梯度
            grad_x, partial_grad_weight,                    # 输出
            x.stride(0), x.stride(1),                       # x 步幅
            weight.stride(0),                               # weight 步幅
            grad_out.stride(0),                             # grad_out 步幅
            grad_x.stride(0), grad_x.stride(1),             # grad_x 步幅
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),  # partial_grad_weight 步幅
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        
        # 归约得到最终的权重梯度
        # 将所有线程块的部分结果相加
        grad_weight = partial_grad_weight.sum(axis=0)
        
        # 恢复 grad_x 的原始形状
        grad_x = grad_x.view(ctx.input_shape)
        
        return grad_x, grad_weight


# ============================================================================
# 便捷接口
# ============================================================================
def weighted_sum(x, weight):
    """
    计算加权求和
    
    参数:
        x: 输入张量，形状 (..., D)
        weight: 权重向量，形状 (D,)
    
    返回:
        output: 形状 (...)，每个位置是对应 embedding 与 weight 的点积
    
    示例:
        >>> x = torch.randn(32, 128, 256, device='cuda')  # (batch, seq, embed)
        >>> w = torch.randn(256, device='cuda')           # (embed,)
        >>> y = weighted_sum(x, w)                        # (batch, seq)
    """
    return WeightedSumFunc.apply(x, weight)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size, seq_len, embed_dim = 4, 8, 64
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', requires_grad=True)
    weight = torch.randn(embed_dim, device='cuda', requires_grad=True)
    
    # Triton 实现
    y_triton = weighted_sum(x, weight)
    
    # PyTorch 参考实现
    y_torch = (x * weight).sum(dim=-1)
    
    # 验证前向结果
    print(f"前向误差: {(y_triton - y_torch).abs().max().item():.2e}")
    
    # 验证反向结果
    loss_triton = y_triton.sum()
    loss_triton.backward()
    grad_x_triton = x.grad.clone()
    grad_w_triton = weight.grad.clone()
    
    x.grad = None
    weight.grad = None
    
    loss_torch = y_torch.sum()
    loss_torch.backward()
    
    print(f"grad_x 误差: {(grad_x_triton - x.grad).abs().max().item():.2e}")
    print(f"grad_weight 误差: {(grad_w_triton - weight.grad).abs().max().item():.2e}")
