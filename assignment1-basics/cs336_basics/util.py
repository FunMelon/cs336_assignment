# 工具函数
import torch


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU激活函数的实现。
    args:
        x (torch.Tensor): 输入张量。
    returns:
        torch.Tensor: 应用SiLU激活函数后的张量。
    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    继承自 torch.nn.Module 的自实现的Softmax模块
    args:
        x (torch.Tensor): 输入张量。
        dim (int): 归一化的维度。
    returns:
        torch.Tensor: 输出张量，应用Softmax后的结果。
    """
    max_val = x.max(dim=dim, keepdim=True).values  # 保留维度以便广播
    exp_x = torch.exp(x - max_val)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
