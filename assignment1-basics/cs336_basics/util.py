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


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算缩放点积注意力。
    args:
        query (torch.Tensor): 查询张量，形状为 (..., seq_len_q, d_k)。
        key (torch.Tensor): 键张量，形状为 (..., seq_len_k, d_k)。
        value (torch.Tensor): 值张量，形状为 (..., seq_len_v, d_v)。
        mask (torch.Tensor | None): 可选的掩码张量，形状为 (..., seq_len_q, seq_len_k)。
    returns:
        torch.Tensor: 注意力输出张量，形状为 (..., seq_len_q, d_v)。
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=query.dtype, device=query.device)
    )

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))  # 使用掩码

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算交叉熵损失。
    args:
        predictions (torch.Tensor): 预测概率分布，形状为 (batch_size, num_classes)。
        targets (torch.Tensor): 真实标签，形状为 (batch_size,)。
    returns:
        torch.Tensor: 交叉熵损失值。
    """
    x_max = predictions.max(dim=1, keepdim=True).values
    log_sum_exp = (
        torch.log(torch.sum(torch.exp(predictions - x_max), dim=1, keepdim=True))
        + x_max
    )  # 稳定的log-sum-exp计算
    log_probs = predictions - log_sum_exp  # 取对数除法变为减法
    loss = -log_probs[torch.arange(predictions.size(0)), targets].mean()

    return loss


def get_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    计算困惑度。
    args:
        loss (torch.Tensor): 交叉熵损失值。
    returns:
        torch.Tensor: 困惑度值。
    """
    return torch.exp(loss).mean()
