# 自实现的torch.nn.Module模块
import torch
from math import sqrt


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


class SiLU(torch.nn.Module):
    """
    继承自 torch.nn.Module 的自实现的SiLU激活函数模块
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，计算SiLU激活函数。
        args:
            x (torch.Tensor): 输入张量。
        returns:
            torch.Tensor: 输出张量，应用SiLU激活函数后的结果。
        """
        return x * torch.sigmoid(x)