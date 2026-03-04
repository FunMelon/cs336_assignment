import torch
import math


class FlashAttention(torch.autograd.Function):
    """
    使用纯 PyTorch 实现 FlashAttention-2 的前向传播。

    ========== 背景知识 ==========
    标准注意力的计算公式为:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

    其中 softmax 作用在最后一个维度（即 K 的序列长度维度）。

    问题在于：Q @ K^T 会生成一个 (seq_len x seq_len) 的巨大矩阵，
    当序列很长时，这个矩阵的显存占用是 O(N^2)，非常昂贵。

    ========== FlashAttention 的核心思想 ==========
    FlashAttention 的关键创新是：**不需要一次性把整个 N×N 的注意力矩阵算出来**，
    而是把 Q、K、V 切成小块（tile），一块一块地计算，同时用一种叫
    "在线 softmax"（online softmax）的技巧来逐步维护正确的 softmax 结果。

    这样做的好处：
    1. 显存占用从 O(N^2) 降低到 O(N)（因为一次只处理一小块）
    2. 减少了对 GPU 高带宽内存（HBM）的读写次数，提升了实际运行速度

    ========== 在线 Softmax 原理 ==========
    标准 softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    为了数值稳定性，通常减去最大值: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))

    但如果数据是分块到达的（先看到一部分 x，再看到另一部分），
    我们需要在看到新数据后"修正"之前的计算结果。

    在线 softmax 维护三个量：
    - m: 到目前为止看到的所有值的行最大值
    - l: 到目前为止 exp(x - m) 的行求和
    - O: 到目前为止的未归一化输出累加

    当看到新的一块数据时，更新规则为：
    - m_new = max(m_old, 当前块的行最大值)
    - alpha = exp(m_old - m_new)          # 用于修正之前的累加值
    - P = exp(S - m_new)                  # 当前块的 softmax 分子（未归一化）
    - l_new = alpha * l_old + rowsum(P)   # 更新分母
    - O_new = alpha * O_old + P @ V       # 更新未归一化输出
    最后所有块处理完后: O_final = O / l   # 归一化得到最终输出
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 前向传播。

        参数:
            ctx:        PyTorch autograd 上下文对象，用于保存反向传播所需的张量
            Q:          查询张量 (Query),  形状 (batch_size, seq_len, d)
            K:          键张量   (Key),    形状 (batch_size, seq_len, d)
            V:          值张量   (Value),  形状 (batch_size, seq_len, d)
            is_causal:  是否使用因果遮罩（True 时只允许 query 关注其之前的 key）

        返回:
            O:          注意力输出张量, 形状 (batch_size, seq_len, d)
        """

        # =====================================================================
        # 第一步：定义分块大小（tile size）
        # =====================================================================
        # Br: Q 的分块大小（沿序列长度维度，每次取 Br 行 query）
        # Bc: K/V 的分块大小（沿序列长度维度，每次取 Bc 行 key/value）
        # 要求至少为 16×16。这里选择 64×64 作为一个合理的平衡点。
        Br = 64
        Bc = 64

        # =====================================================================
        # 第二步：获取输入形状，计算缩放因子
        # =====================================================================
        batch_size, seq_len, d = Q.shape
        # 标准注意力公式中的缩放因子 1/sqrt(d)，用于防止点积值过大
        # 当 d 很大时，Q@K^T 的值会很大，导致 softmax 梯度消失，所以要除以 sqrt(d)
        scale = 1.0 / math.sqrt(d)

        # =====================================================================
        # 第三步：初始化累加器
        # =====================================================================
        # O: 输出累加器，初始化为全零，形状与 Q 相同 (B, N, d)
        #    在内层循环中，O 会逐步累加各个 K/V 块的贡献
        O = torch.zeros_like(Q)

        # m: 每行的"到目前为止看到的最大注意力分数"，用于数值稳定性
        #    初始化为负无穷，因为还没看到任何数据，任何实际值都会比它大
        #    形状 (B, N)，每个 query 位置一个标量
        m = torch.full(
            (batch_size, seq_len), float('-inf'),
            device=Q.device, dtype=Q.dtype
        )

        # l: 每行的"到目前为止 exp(score - m) 的累加和"，即 softmax 的分母
        #    初始化为 0，因为还没有累加任何 exp 值
        #    形状 (B, N)，每个 query 位置一个标量
        l = torch.zeros(
            (batch_size, seq_len),
            device=Q.device, dtype=Q.dtype
        )

        # =====================================================================
        # 第四步：计算分块数量
        # =====================================================================
        # 把序列长度 seq_len 按 Br/Bc 切分，计算需要多少个块
        # 例如 seq_len=256, Br=64 → num_q_tiles=4（分成4个Q块）
        num_q_tiles = math.ceil(seq_len / Br)
        num_kv_tiles = math.ceil(seq_len / Bc)

        # =====================================================================
        # 第五步：外层循环 —— 遍历 Q 的每一个分块
        # =====================================================================
        # 对于每一块 Q_i，我们需要和所有的 K、V 块交互来计算该块的输出
        for i in range(num_q_tiles):
            # 计算当前 Q 块的起止索引
            q_start = i * Br
            q_end = min(q_start + Br, seq_len)

            # 取出当前 Q 块及其对应的累加器切片
            Q_i = Q[:, q_start:q_end, :]   # 当前 Q 块, 形状 (B, Br, d)
            O_i = O[:, q_start:q_end, :]    # 当前输出块, 形状 (B, Br, d)
            m_i = m[:, q_start:q_end]        # 当前行最大值, 形状 (B, Br)
            l_i = l[:, q_start:q_end]        # 当前行求和, 形状 (B, Br)

            # =================================================================
            # 第六步：内层循环 —— 遍历 K/V 的每一个分块
            # =================================================================
            # 对于当前 Q 块，逐个和每个 K/V 块做注意力计算
            # 这就是"在线 softmax"发挥作用的地方
            for j in range(num_kv_tiles):
                # 计算当前 K/V 块的起止索引
                kv_start = j * Bc
                kv_end = min(kv_start + Bc, seq_len)

                # 取出当前 K 和 V 块
                K_j = K[:, kv_start:kv_end, :]  # 当前 K 块, 形状 (B, Bc, d)
                V_j = V[:, kv_start:kv_end, :]  # 当前 V 块, 形状 (B, Bc, d)

                # ----- 6a. 计算注意力分数矩阵 S_ij -----
                # S_ij = Q_i @ K_j^T / sqrt(d)
                # 形状: (B, Br, d) @ (B, d, Bc) = (B, Br, Bc)
                # 这是一个小矩阵（64×64），而不是完整的 N×N，这就是省内存的关键！
                S_ij = torch.bmm(Q_i, K_j.transpose(1, 2)) * scale

                # ----- 6a-causal. 应用因果掩码 -----
                # 因果掩码的含义：位置 i 的 query 只能"看到"位置 j <= i 的 key
                # 即"未来"的信息被屏蔽掉，这是自回归语言模型的核心约束
                #
                # 实现方式：构造 query 和 key 的绝对位置索引，比较它们
                # 对于 key_pos > query_pos 的位置（即"未来"位置），
                # 在注意力分数上加一个很大的负数 -1e6
                # 经过 softmax 后 exp(-1e6) ≈ 0，相当于完全屏蔽了这些位置
                if is_causal:
                    # q_indices: 当前 Q 块中每个 query 的绝对位置 [q_start, q_start+1, ..., q_end-1]
                    # k_indices: 当前 K 块中每个 key 的绝对位置 [kv_start, kv_start+1, ..., kv_end-1]
                    q_indices = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  # (Br, 1)
                    k_indices = torch.arange(kv_start, kv_end, device=Q.device).unsqueeze(0)  # (1, Bc)
                    # causal_mask[i,j] = True 当 key 位置 > query 位置（即"未来"，需要屏蔽）
                    causal_mask = k_indices > q_indices  # (Br, Bc)，广播到 (B, Br, Bc)
                    S_ij = S_ij.masked_fill(causal_mask.unsqueeze(0), -1e6)

                # ----- 6b. 计算当前块的行最大值 m_ij -----
                # 对 S_ij 的最后一个维度（Bc维度）取最大值
                # 形状: (B, Br)，即每行 query 对当前 K 块的最大注意力分数
                m_ij = S_ij.max(dim=-1).values

                # ----- 6c. 更新全局行最大值 -----
                # 在之前所有 K 块的最大值 m_i 和当前块最大值 m_ij 之间取较大值
                # 这保证了 m_new 始终是"到目前为止所有块中的最大值"
                m_new = torch.maximum(m_i, m_ij)

                # ----- 6d. 计算修正因子 alpha -----
                # 因为最大值从 m_i 变成了 m_new（可能更大了），
                # 之前根据 m_i 计算的 exp 值都"偏大"了，需要乘以 alpha 来修正
                # alpha = exp(m_old - m_new)，当 m_new > m_old 时 alpha < 1，
                # 相当于把之前的值"缩小"到正确的尺度
                # 形状: (B, Br)
                alpha = torch.exp(m_i - m_new)

                # ----- 6e. 计算当前块的（未归一化的）softmax 权重 P_ij -----
                # P_ij = exp(S_ij - m_new)
                # 减去 m_new 是为了数值稳定性（防止 exp 溢出）
                # m_new.unsqueeze(-1) 将 (B, Br) 扩展为 (B, Br, 1) 以便广播
                # 形状: (B, Br, Bc)
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

                # ----- 6f. 更新分母（行求和 l） -----
                # l_new = alpha * l_old + sum(P_ij, dim=-1)
                # - alpha * l_old: 之前的分母经过修正（因为最大值可能变了）
                # - P_ij.sum(dim=-1): 当前块的 exp 值之和
                # 形状: (B, Br)
                l_new = alpha * l_i + P_ij.sum(dim=-1)

                # ----- 6g. 更新未归一化的输出 O_i -----
                # O_new = alpha * O_old + P_ij @ V_j
                # - alpha * O_old: 之前的输出经过修正（原因同上）
                # - P_ij @ V_j: 当前块注意力权重乘以 V，得到当前块的贡献
                #   形状: (B, Br, Bc) @ (B, Bc, d) = (B, Br, d)
                # alpha.unsqueeze(-1) 将 (B, Br) 扩展为 (B, Br, 1) 以便与 O_i 广播
                O_i = alpha.unsqueeze(-1) * O_i + torch.bmm(P_ij, V_j)

                # ----- 6h. 保存更新后的统计量，进入下一个 K/V 块 -----
                m_i = m_new
                l_i = l_new

            # =================================================================
            # 第七步：内层循环结束后，对当前 Q 块做最终归一化
            # =================================================================
            # 此时 O_i 是未归一化的输出（分子），l_i 是 softmax 分母
            # 最终输出 = O_i / l_i，即 softmax 分子 / softmax 分母
            # l_i.unsqueeze(-1) 将 (B, Br) 扩展为 (B, Br, 1) 以便与 O_i 广播
            O_i = O_i / l_i.unsqueeze(-1)

            # 计算 logsumexp 值: L = m + log(l)
            # logsumexp 是一个常见的数值稳定技巧:
            #   log(sum(exp(x))) = m + log(sum(exp(x - m)))
            # 其中 m 是最大值。这里 l 已经是 sum(exp(x - m)) 了，
            # 所以 L = m + log(l) 就是完整的 logsumexp
            # L 在反向传播时会用到（用于高效地重新计算 softmax）
            L_i = m_i + torch.log(l_i)

            # 将当前 Q 块的结果写回到全局输出张量中
            O[:, q_start:q_end, :] = O_i
            m[:, q_start:q_end] = L_i   # 复用 m 的存储空间来保存 logsumexp

        # =====================================================================
        # 第八步：保存反向传播所需的张量
        # =====================================================================
        # L: logsumexp 值，形状 (B, N)
        L = m

        # ctx.save_for_backward 是 PyTorch autograd 的机制，
        # 用于保存前向传播中的张量供反向传播使用
        # FlashAttention 的一大优势：只需保存 L, Q, K, V, O 这五个张量，
        # 而不需要保存巨大的 N×N 注意力矩阵，大幅节省显存
        ctx.save_for_backward(L, Q, K, V, O)
        # 保存因果掩码标志，反向传播时需要用同样的掩码
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        """反向传播（尚未实现）"""
        raise NotImplementedError("FlashAttention 反向传播尚未实现。")


def flash_attention_pytorch(Q, K, V, is_causal=False):
    """
    FlashAttention-2 的函数式封装。
    直接调用 FlashAttention2.apply() 来触发自定义的前向传播。
    """
    return FlashAttention.apply(Q, K, V, is_causal)
