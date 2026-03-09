import torch
from typing import Callable

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算每组 rollout 回答的优势（组归一化奖励）。
    
    参数:
        reward_fn: Callable[[str, str], dict[str, float]] 
            评估函数。接收模型生成的回答和标准答案，返回包含 "reward", "format_reward", "answer_reward" 的字典。
        rollout_responses: list[str] 
            模型生成的回答列表。总长度为 rollout_batch_size = 提示词数量 * group_size。
        repeated_ground_truths: list[str] 
            重复的标准答案列表。长度与 rollout_responses 一致。
        group_size: int 
            每个问题生成的回答数量（即组的大小 G）。
        advantage_eps: float 
            一个极小的常数，用于防止除以零的情况发生。
        normalize_by_std: bool 
            如果为 True，则除以组内标准差（DeepSeek R1 默认做法）；
            如果为 False，则只减去组内均值（Dr. GRPO 极简做法）。
            
    返回:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages: 形状为 (rollout_batch_size,) 的张量，包含每个回答的组归一化优势值。
            raw_rewards: 形状为 (rollout_batch_size,) 的张量，包含未归一化的原始奖励分数。
            metadata: 用于日志记录的元数据字典（例如奖励的均值、最大值等）。
    """
    
    # 1. 遍历收集所有回答的原始奖励
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        # 调用评分函数获取奖励字典
        rewards_dict = reward_fn(resp, gt)
        raw_rewards_list.append(rewards_dict["reward"])
        
        # 收集细分奖励（如果有的话，方便打日志分析）
        format_rewards_list.append(rewards_dict.get("format_reward", 0.0))
        answer_rewards_list.append(rewards_dict.get("answer_reward", 0.0))
        
    # 将一维 Python 列表转换为 PyTorch 张量
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    
    # 2. 将一维张量重塑为二维张量，形状为 (N, group_size)
    # 这里的 N 代表不同的 Prompt（问题）的数量
    num_prompts = len(rollout_responses) // group_size
    grouped_rewards = raw_rewards.view(num_prompts, group_size)
    
    # 3. 计算每一组（沿着 dim=1）的均值
    # keepdim=True 是为了保持张量形状为 (N, 1)，从而可以触发 PyTorch 的广播机制 (Broadcasting)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    
    # 计算组中心化的奖励（也就是只比均值高/低多少）
    centered_rewards = grouped_rewards - group_means
    
    # 4. 根据策略计算最终的 Advantage (优势)
    if normalize_by_std:
        # TODO：为什么不使用无偏估计 (unbiased=False) 计算标准差？
        group_stds = grouped_rewards.std(dim=1, unbiased=True, keepdim=True)
        # 核心公式： (r - mean) / (std + eps)
        grouped_advantages = centered_rewards / (group_stds + advantage_eps)
    else:
        # 这里就是你之前提到的“不除以标准差”的消融实验（Dr. GRPO 论文做法）
        grouped_advantages = centered_rewards
        
    # 5. 将处理好的优势张量重新展平为一维张量 (rollout_batch_size,)
    advantages = grouped_advantages.view(-1)
    
    # 6. 计算元数据用于记录日志 (Logging)
    metadata = {
        "reward/mean": raw_rewards.mean().item(),
        "reward/std": raw_rewards.std().item() if len(raw_rewards) > 1 else 0.0,
        "reward/format_mean": sum(format_rewards_list) / len(format_rewards_list) if format_rewards_list else 0.0,
        "reward/answer_mean": sum(answer_rewards_list) / len(answer_rewards_list) if answer_rewards_list else 0.0,
    }
    
    return advantages, raw_rewards, metadata