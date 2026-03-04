import torch
import math
import triton
import triton.language as tl


# =============================================================================
# Triton 内核：FlashAttention-2 前向传播
# =============================================================================
# 这个内核是 flash_attention_pytorch.py 中 PyTorch 实现的 GPU 并行版本。
# 核心思想完全相同（在线 softmax + 分块计算），但有以下关键区别：
#
# 1. PyTorch 版本：外层循环遍历 Q 块，内层循环遍历 K/V 块（两层 for 循环）
# 2. Triton 版本：外层循环被"并行化"了——每个 Triton Program（GPU 线程块）
#    负责一个 Q 块 + 一个 batch，内核里只剩一个 for 循环遍历 K/V 块
#
# 这就是 GPU 并行的威力：PyTorch 的外层循环是串行的，
# 而 Triton 把它变成了 (T_q × batch_size) 个并行任务同时执行。
# =============================================================================

@triton.jit
def flash_fwd_kernel(
    # ---- 输入/输出张量的基地址指针 ----
    Q_ptr, K_ptr, V_ptr,       # Q, K, V 的全局内存指针
    O_ptr, L_ptr,              # 输出 O 和 logsumexp L 的全局内存指针

    # ---- 各张量的内存步幅（stride）----
    # 步幅描述了在内存中"跳一个维度"需要跳多少个元素
    # 例如 stride_qb 是 Q 在 batch 维度的步幅：从 batch[0] 跳到 batch[1] 需要跳过的元素数
    stride_qb, stride_qq, stride_qd,   # Q 的步幅：batch, seq, d
    stride_kb, stride_kk, stride_kd,   # K 的步幅：batch, seq, d
    stride_vb, stride_vk, stride_vd,   # V 的步幅：batch, seq, d
    stride_ob, stride_oq, stride_od,   # O 的步幅：batch, seq, d
    stride_lb, stride_lq,              # L 的步幅：batch, seq

    # ---- 维度参数 ----
    N_QUERIES,                 # query 序列长度
    N_KEYS,                    # key 序列长度（和 N_QUERIES 相同，因为是自注意力）
    scale,                     # 缩放因子 1/sqrt(d)

    # ---- 编译时常量（Triton 编译器需要在编译时知道这些值以优化代码）----
    D: tl.constexpr,                   # 头嵌入维度 d
    Q_TILE_SIZE: tl.constexpr,         # Q 块大小 Br（对应 PyTorch 版本的 Br）
    K_TILE_SIZE: tl.constexpr,         # K/V 块大小 Bc（对应 PyTorch 版本的 Bc）
    is_causal: tl.constexpr,           # 是否启用因果掩码（编译时常量，True/False 会生成不同的内核）
):
    # =====================================================================
    # 第一步：确定当前线程块负责哪个 Q 块和哪个 batch
    # =====================================================================
    # tl.program_id(0) → 当前线程块在网格第 0 维的索引，对应 Q 块编号
    # tl.program_id(1) → 当前线程块在网格第 1 维的索引，对应 batch 编号
    # 这两行等价于 PyTorch 版本外层 for 循环的 "for i in range(num_q_tiles)"
    # 只不过这里每个 (i, batch) 组合都由一个独立的 GPU 线程块并行执行
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # =====================================================================
    # 第二步：创建块指针（Block Pointer）
    # =====================================================================
    # 块指针是 Triton 提供的高级抽象，用于方便地从全局内存中读取/写入一个二维块
    # 它会自动根据 shape, strides, offsets, block_shape 计算出正确的内存地址
    #
    # order=(1, 0) 表示"行优先"（row-major），即内存中最后一个维度（d）是连续的
    # 这通常与 PyTorch 默认的内存布局一致

    # Q 的块指针：读取当前线程块负责的 Q 块
    # offsets=(query_tile_index * Q_TILE_SIZE, 0) 表示从第 i 个 Q 块的起始位置开始
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,        # 基地址：跳到对应 batch
        shape=(N_QUERIES, D),                    # 整个 Q 矩阵的形状（当前 batch）
        strides=(stride_qq, stride_qd),          # 内存步幅
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # 块的起始偏移
        block_shape=(Q_TILE_SIZE, D),            # 要读取的块大小
        order=(1, 0),
    )

    # K 的块指针：初始指向第 0 个 K 块，循环中会通过 advance 逐块移动
    # 注意 K 用于计算 Q @ K^T，所以 block_shape 是 (K_TILE_SIZE, D)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),                          # 从第 0 个 K 块开始
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V 的块指针：同 K，初始指向第 0 个 V 块
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),                          # 从第 0 个 V 块开始
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # O 的块指针：写入当前线程块负责的输出块
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L 的块指针：写入当前线程块负责的 logsumexp 值
    # L 是一维的（每个 query 位置一个标量），所以 block_shape 是 (Q_TILE_SIZE,)
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # =====================================================================
    # 第三步：从全局内存加载 Q 块到片上缓存（SRAM）
    # =====================================================================
    # Q_i 只需加载一次，因为当前线程块在整个循环中始终使用同一个 Q 块
    # 形状: (Q_TILE_SIZE, D)
    Q_i = tl.load(Q_block_ptr)

    # =====================================================================
    # 第四步：初始化片上累加器（关键：必须用 float32！）
    # =====================================================================
    # 精度警告：GPU 上的 exp 运算容易溢出，所以 m, l, O_acc 必须用 float32
    # 即使输入是 bfloat16 或 float16，累加器也必须是 float32

    # O_acc: 未归一化的输出累加器，初始化为全零
    # 形状: (Q_TILE_SIZE, D)
    O_acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # m_i: 当前行最大值，初始化为负无穷
    # 形状: (Q_TILE_SIZE,)
    m_i = tl.full([Q_TILE_SIZE], value=float('-inf'), dtype=tl.float32)

    # l_i: 当前行 exp 累加和，初始化为 0
    # 形状: (Q_TILE_SIZE,)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)

    # =====================================================================
    # 第五步：唯一的 for 循环 —— 遍历所有 K/V 块
    # =====================================================================
    # 这对应 PyTorch 版本的内层循环 "for j in range(num_kv_tiles)"
    # 计算 K/V 块的数量
    num_kv_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)  # cdiv = 向上取整除法

    for _j in range(num_kv_tiles):

        # ----- 5a. 从全局内存加载当前 K 和 V 块到片上缓存 -----
        # K_j 形状: (K_TILE_SIZE, D)
        # V_j 形状: (K_TILE_SIZE, D)
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        # ----- 5b. 计算注意力分数 S_ij = Q_i @ K_j^T * scale -----
        # tl.dot 执行矩阵乘法：(Q_TILE_SIZE, D) @ (D, K_TILE_SIZE) = (Q_TILE_SIZE, K_TILE_SIZE)
        # trans_b=True 表示对第二个操作数转置，即 K_j^T
        # 结果 S_ij 形状: (Q_TILE_SIZE, K_TILE_SIZE)
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        # ----- 5b-causal. 应用因果掩码 -----
        # is_causal 是编译时常量（tl.constexpr），所以这个 if 在编译时就会被决定
        # True 和 False 会各自编译出一个版本的内核，不会有运行时分支开销
        if is_causal:
            # 构造 query 和 key 的绝对位置索引向量
            # q_indices: 当前 Q 块中每行 query 的全局位置
            #   = [query_tile_index * Q_TILE_SIZE + 0, +1, ..., +Q_TILE_SIZE-1]
            #   形状: (Q_TILE_SIZE,)
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            # k_indices: 当前 K 块中每列 key 的全局位置
            #   = [_j * K_TILE_SIZE + 0, +1, ..., +K_TILE_SIZE-1]
            #   形状: (K_TILE_SIZE,)
            k_indices = _j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # 因果掩码比较: query 位置 i 只能看到 key 位置 j <= i
            # q_indices[:, None] 广播为 (Q_TILE_SIZE, 1)
            # k_indices[None, :] 广播为 (1, K_TILE_SIZE)
            # 结果 causal_mask 形状: (Q_TILE_SIZE, K_TILE_SIZE)
            # 值为 True 的位置表示 key 在 query 的"未来"，需要被屏蔽
            causal_mask = k_indices[None, :] > q_indices[:, None]
            # 对被屏蔽的位置加上 -1e6（一个足够大的负数）
            # softmax 后 exp(-1e6) ≈ 0，相当于完全忽略这些位置
            S_ij = tl.where(causal_mask, S_ij + (-1e6), S_ij)

        # ----- 5c. 计算当前块的行最大值 -----
        # tl.max(S_ij, axis=1) 沿列维度取最大值（每行一个最大值）
        # 形状: (Q_TILE_SIZE,)
        m_ij = tl.max(S_ij, axis=1)

        # ----- 5d. 更新全局行最大值 -----
        # m_new = max(m_old, m_current_block)
        m_new = tl.maximum(m_i, m_ij)

        # ----- 5e. 计算修正因子 alpha -----
        # alpha = exp(m_old - m_new)，用于将之前的累加值修正到新的尺度
        # 当 m_new > m_i 时，alpha < 1，把之前的值缩小
        # 当 m_new == m_i 时，alpha = 1，不需要修正
        alpha = tl.exp(m_i - m_new)

        # ----- 5f. 计算未归一化的 softmax 权重 P_ij -----
        # P_ij = exp(S_ij - m_new)
        # 注意: m_new 是 (Q_TILE_SIZE,) 的向量，S_ij 是 (Q_TILE_SIZE, K_TILE_SIZE) 的矩阵
        # m_new[:, None] 广播到每一列
        P_ij = tl.exp(S_ij - m_new[:, None])

        # ----- 5g. 更新分母 l -----
        # l_new = alpha * l_old + rowsum(P_ij)
        l_i = alpha * l_i + tl.sum(P_ij, axis=1)

        # ----- 5h. 更新未归一化的输出 O_acc -----
        # O_new = alpha * O_old + P_ij @ V_j
        #
        # 先把旧的 O_acc 乘以修正因子 alpha
        O_acc = O_acc * alpha[:, None]
        #
        # 精度关键：在做 tl.dot 之前，必须将 P_ij 转换为与 V_j 相同的低精度类型
        # 因为 tl.dot 要求两个操作数类型一致，而 P_ij 目前是 float32
        # 使用 acc=O_acc 参数让 Triton 直接累加到 O_acc 上，避免额外的内存分配
        O_acc = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_acc)

        # ----- 5i. 更新行最大值，为下一轮循环做准备 -----
        m_i = m_new

        # ----- 5j. 移动 K 和 V 的块指针到下一个块 -----
        # advance((K_TILE_SIZE, 0)) 表示在第 0 维（序列维度）移动一个块的距离
        # 第 1 维（d 维度）不移动
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # =====================================================================
    # 第六步：循环结束，做最终归一化并写回全局内存
    # =====================================================================

    # 最终归一化：O = O_acc / l_i
    # l_i[:, None] 将 (Q_TILE_SIZE,) 扩展为 (Q_TILE_SIZE, 1) 以便广播
    O_acc = O_acc / l_i[:, None]

    # 计算 logsumexp: L = m + log(l)
    # 这个值在反向传播时会用到
    L_i = m_i + tl.log(l_i)

    # 写回全局内存前，将 O_acc 从 float32 转换回输入的原始精度（如 bfloat16）
    tl.store(O_block_ptr, O_acc.to(O_block_ptr.dtype.element_ty))
    tl.store(L_block_ptr, L_i)


# =============================================================================
# torch.autograd.Function 封装：负责分配输出张量、设置网格、发射内核
# =============================================================================

class FlashAttentionTriton(torch.autograd.Function):
    """
    使用 Triton 内核实现的 FlashAttention-2。

    与 PyTorch 版本的对比：
    - PyTorch 版本：双层 for 循环，在 CPU 上串行调度，适合调试
    - Triton 版本：外层循环并行化为 GPU 线程块，只保留一个内层循环，速度快得多
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        前向传播：分配输出张量，设置启动网格，发射 Triton 内核。

        参数:
            ctx:        autograd 上下文
            Q:          (batch_size, seq_len, d) 查询张量
            K:          (batch_size, seq_len, d) 键张量
            V:          (batch_size, seq_len, d) 值张量
            is_causal:  因果遮罩标志（True 时只允许 query 关注其之前的 key）

        返回:
            O:          (batch_size, seq_len, d) 注意力输出
        """
        # ----- 获取维度信息 -----
        batch_size, seq_len, d = Q.shape

        # ----- 设置分块大小 -----
        # Q_TILE_SIZE 和 K_TILE_SIZE 分别对应 PyTorch 版本的 Br 和 Bc
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # ----- 计算缩放因子 -----
        scale = 1.0 / math.sqrt(d)

        # ----- 分配输出张量 -----
        # O: 注意力输出，形状和 Q 相同
        O = torch.empty_like(Q)
        # L: logsumexp 值，每个 query 位置一个标量，形状 (batch_size, seq_len)
        L = torch.empty(batch_size, seq_len, device=Q.device, dtype=torch.float32)

        # ----- 设置启动网格 (Launch Grid) -----
        # 网格形状: (T_q, batch_size)
        # T_q = 序列长度 / Q 块大小 = Q 块的数量
        # 每个网格点对应一个 Triton Program（GPU 线程块），
        # 负责处理一个 batch 中的一个 Q 块
        num_q_tiles = math.ceil(seq_len / Q_TILE_SIZE)
        grid = (num_q_tiles, batch_size)

        # ----- 发射 Triton 内核 -----
        # 将所有需要的参数传递给内核
        # .stride() 返回 PyTorch 张量各维度的步幅（以元素为单位）
        # num_stages 控制 Triton 的软件流水线级数（software pipelining stages）
        # 流水线级数越多，Triton 会预取更多的数据块到共享内存中以隐藏延迟，
        # 但也会占用更多共享内存。当 d 较大（如 128）时，每个块本身就很大，
        # 默认的 num_stages 可能导致共享内存超出硬件限制（通常 ~164KB/SM）。
        # 设为 1 表示不做流水线预取，是最保守但最安全的选择。
        num_stages = 1

        flash_fwd_kernel[grid](
            # 张量指针
            Q, K, V, O, L,
            # Q 的步幅: (batch, seq, d)
            Q.stride(0), Q.stride(1), Q.stride(2),
            # K 的步幅
            K.stride(0), K.stride(1), K.stride(2),
            # V 的步幅
            V.stride(0), V.stride(1), V.stride(2),
            # O 的步幅
            O.stride(0), O.stride(1), O.stride(2),
            # L 的步幅
            L.stride(0), L.stride(1),
            # 维度参数
            seq_len, seq_len,
            scale,
            # 编译时常量
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            # 流水线级数：限制共享内存占用
            num_stages=num_stages,
        )

        # ----- 保存反向传播所需的张量 -----
        ctx.save_for_backward(L, Q, K, V, O)
        # 保存因果掩码标志，反向传播时需要用同样的掩码
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        """反向传播（尚未实现）"""
        raise NotImplementedError("FlashAttention Triton 反向传播尚未实现。")


def flash_attention_triton(Q, K, V, is_causal=False):
    """
    FlashAttention-2 Triton 版本的函数式封装。
    """
    return FlashAttentionTriton.apply(Q, K, V, is_causal)
