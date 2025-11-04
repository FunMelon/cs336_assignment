# 自实现的torch.nn.Module模块
import torch
from math import sqrt
from .util import (
    SiLU,
    scaled_dot_product_attention,
)

class Linear(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的线性层模块
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ):
        """
        初始化线性层，创建权重和偏置参数。
        args:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            device (torch.device | None): 参数所在的设备。
            dtype (torch.dtype | None): 参数的数据类型。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 根据输入输出特征维度创建权重参数
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        sigma = sqrt(2 / (in_features + out_features))  # He初始化标准差
        torch.nn.init.trunc_normal_(
            tensor, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )
        self.weight = torch.nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，计算线性变换。
        args:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。
        returns:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """

        return x @ self.weight.T

    def __repr__(self) -> str:
        """
        返回线性层的字符串表示。
        returns:
            str: 线性层的字符串表示。
        """
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias=False)"


class Embedding(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的嵌入层模块
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化嵌入层，创建嵌入矩阵参数。
        args:
            num_embeddings (int): 嵌入词典的大小。
            embedding_dim (int): 每个嵌入向量的维度。
            device (torch.device | None): 参数所在的设备。
            dtype (torch.dtype | None): 参数的数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 根据词典大小和嵌入维度创建嵌入矩阵参数
        tensor = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1, a=-3.0, b=3.0)
        self.weight = torch.nn.Parameter(tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，查找嵌入向量。
        args:
            token_ids (torch.Tensor): 输入的token ID张量，形状为 (..., )。
        returns:
            torch.Tensor: 输出的嵌入向量张量，形状为 (..., embedding_dim)。
        """
        return self.weight[token_ids]

    def __repr__(self) -> str:
        """
        返回嵌入层的字符串表示。
        returns:
            str: 嵌入层的字符串表示。
        """
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


class RMSNorm(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的RMS归一化模块
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化RMS归一化层，创建缩放参数。
        args:
            d_model (int): 输入特征的维度。
            eps (float): 防止除零的小常数。
            device (torch.device | None): 参数所在的设备。
            dtype (torch.dtype | None): 参数的数据类型。
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 创建缩放参数
        tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.scale = torch.nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，计算RMS归一化。
        args:
            x (torch.Tensor): 输入张量，形状为 (..., d_model)。
        returns:
            torch.Tensor: 输出张量，形状为 (..., d_model)。
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        )  # 只在最后一个维度上计算RMS，形状为 (..., 1)
        x_norm = x / rms
        result = x_norm * self.scale

        return result.to(in_dtype)

    def __repr__(self) -> str:
        """
        返回RMS归一化层的字符串表示。
        returns:
            str: RMS归一化层的字符串表示。
        """
        return f"RMSNorm(d_model={self.d_model}, eps={self.eps})"


class PositionwiseFeedForward(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的逐位置前馈网络模块，使用SwiGLU激活函数
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        初始化逐位置前馈网络，创建两个线性层和激活函数。
        args:
            d_model (int): 输入和输出特征的维度。
            d_ff (int): 前馈网络中间层的维度。
            device (torch.device | None): 参数所在的设备。
            dtype (torch.dtype | None): 参数的数据类型。
        """
        super().__init__()
        assert (
            d_ff % 64 == 0
        ), f"d_ff ({d_ff}) must be a multiple of 64 for efficient computation."  # 确保d_ff是64的倍数以提高计算效率（CUDA的warp大小）
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，计算逐位置前馈网络的输出。
        args:
            x (torch.Tensor): 输入张量，形状为 (..., d_model)。
        returns:
            torch.Tensor: 输出张量，形状为 (..., d_model)。
        """
        x1 = self.linear1(x)
        x1 = SiLU(x1)
        x2 = self.linear3(x)
        x3 = x1 * x2
        output = self.linear2(x3)
        return output

    def __repr__(self) -> str:
        """
        返回逐位置前馈网络的字符串表示。
        returns:
            str: 逐位置前馈网络的字符串表示。
        """
        return f"PositionwiseFeedForward(d_model={self.linear1.in_features}, d_ff={self.linear1.out_features})"


class RoPE(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的旋转位置编码模块
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化旋转位置编码模块，将正弦和余弦位置编码注册为buffer。
        args:
            theta (float): 旋转角度的缩放因子。
            d_k (int): 位置编码的维度。
            max_seq_len (int): 最大序列长度。
            device (torch.device | None): 参数所在的设备。
        """
        super().__init__()
        position = torch.arange(max_seq_len, device=device).unsqueeze(
            1
        )  # 位置索引，形状为 (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_k, 2, device=device)
            * (-torch.log(torch.tensor(theta)) / d_k)
        )  # exp((i) * (-log θ / d_k)) = θ^(-i / d_k) = 1 / θ^(i / d_k)，形状为 (d_k/2, )

        sin = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)

        self.register_buffer(
            "sin", sin, persistent=False
        )  # 注册为非持久buffer（不保存在state_dict中）
        self.register_buffer("cos", cos, persistent=False)

    def rotation(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，应用旋转位置编码。
        args:
            x (torch.Tensor): 输入张量，形状为 (..., d_k)。
            token_positions (torch.Tensor): 位置索引张量，形状为 (..., )。
        returns:
            torch.Tensor: 输出张量，形状为 (..., d_k)。
        """
        sin = self.sin[token_positions]  # type: ignore
        cos = self.cos[token_positions]  # type: ignore

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)  # 先拼接后展平


class MultiheadSelfAttention(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的多头自注意力模块
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        """
        初始化多头自注意力模块，创建查询、键、值和输出线性层。
        args:
            d_model (int): 输入和输出特征的维度。
            num_heads (int): 注意力头的数量。
            device (torch.device | None): 参数所在的设备。
            dtype (torch.dtype | None): 参数的数据类型。
            rope (RoPE | None): 可选的RoPE位置编码模块。
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # d_k = d_v = d_model / num_heads

        self.qkv_linear = Linear(d_model, d_model * 3, device, dtype)   # 一次性生成Q、K、V
        self.out_linear = Linear(d_model, d_model, device, dtype)

        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播方法，计算多头自注意力的输出。
        args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            token_positions (torch.Tensor | None): 位置索引张量，形状为 (..., seq_len)。
        returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        batch_size, seq_len, _ = x.size()

        
        qkv = self.qkv_linear(x)  # (batch_size, seq_len, 3 * d_model)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)  # 各自 (batch_size, seq_len, d_model)
        # 线性变换并分割为多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        if token_positions is not None:
            assert self.rope is not None, "RoPE module must be provided when token_positions is used."
            # 应用RoPE位置编码
            Q = self.rope.rotation(Q, token_positions)  # (batch_size, num_heads, seq_len, d_k)
            K = self.rope.rotation(K, token_positions)  # (batch_size, num_heads, seq_len, d_k)

        # 创建下三角掩码以防止未来信息泄露
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()  # (seq_len, seq_len)，上三角为False，下三角为True

        # 计算缩放点积注意力
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)

        # 拼接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # 最终线性变换
        output = self.out_linear(attn_output)  # (batch_size, seq_len, d_model)

        return output