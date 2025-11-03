import torch
from math import sqrt

class Linear(torch.nn.Module):
    """
        继承自 torch.nn.Module 的自实现的线性层模块
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None, dtype: torch.dtype | None):
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
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=sigma, a=-3.0 * sigma, b= 3.0 * sigma)
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
