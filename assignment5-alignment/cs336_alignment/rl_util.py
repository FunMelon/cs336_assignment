import torch
from typing import Callable, Literal

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

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个token的策略梯度损失，其中raw_rewards_or_advantages可以是原始奖励或已经归一化的优势值。
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), 每个rollout响应的标量奖励/优势值。
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), 每个token的对数概率。
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), 每个token的策略梯度损失（将在训练循环中跨batch和sequence维度进行聚合）。
    """

    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO 剪辑损失函数 (Clipped Loss)。

    GRPO 使用与 PPO 相同的裁剪目标函数，旨在限制策略更新的幅度，防止因单次更新变化过大而导致训练不稳定。

    核心公式:
        L(θ) = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

    其中:
        r(θ) = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob - old_log_prob)
        A 是优势值 (advantage)
        ε 是 cliprange (裁剪范围，如 0.2)

    当 A > 0 (正向优势) 时: 限制 r(θ) ≤ 1+ε，防止过度增加概率
    当 A < 0 (负向优势) 时: 限制 r(θ) ≥ 1-ε，防止过度减少概率

    Args:
        advantages: torch.Tensor Shape (batch_size, 1) 或 (batch_size,)，每个样本的优势值 A。
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length)，当前策略的每个 token 的对数概率。
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length)，旧策略的每个 token 的对数概率。
        cliprange: float 裁剪参数 ε (例如 0.2)。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: Shape (batch_size, sequence_length)，每个 token 的裁剪后的损失值。
            metadata: 包含日志信息的字典，建议记录每个 token 是否被裁剪。
    """
    # 1. 计算概率比率 r(θ) = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob - old_log_prob)
    # 使用 exp 计算比率，比直接相除更数值稳定
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 2. 确保 advantages 的形状与 ratio 匹配，以便进行逐元素运算
    # 如果 advantages 是一维的 (batch_size,)，则扩展为 (batch_size, 1)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    # 3. 计算裁剪后的比率: clip(r(θ), 1-ε, 1+ε)
    # 当 A > 0 时: 裁剪上限为 1+ε，防止过度增加概率
    # 当 A < 0 时: 裁剪下限为 1-ε，防止过度减少概率
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    # 4. 计算 PPO/GRPO 核心损失公式: L = -min(r*A, clip(r)*A)
    # 注意: 必须先计算 min(r*A, clip(r)*A)，再取负号
    # 如果先取负号再取 min，当 A < 0 时结果会错误！
    # 原因: min(-x, -y) = -max(x, y) ≠ -min(x, y)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # 计算裁剪统计信息用于日志记录
    # 当 ratio 被裁剪且裁剪后的损失更小时，认为发生了裁剪
    # 检查每个 token 是否被裁剪 (裁剪后的损失是否被选中)
    was_clipped = (ratio * advantages != clipped_ratio * advantages).float()

    if response_mask is not None:
        mask_float = response_mask.to(torch.float32)
        clip_fraction = masked_mean(was_clipped, mask_float).item()
        
        # 计算原始损失和裁剪后损失用于日志记录
        original_loss = -ratio * advantages
        clipped_loss = -clipped_ratio * advantages
        
        loss_mean = masked_mean(loss, mask_float).item()
        original_loss_mean = masked_mean(original_loss, mask_float).item()
        clipped_loss_mean = masked_mean(clipped_loss, mask_float).item()
        
        ratio_mean = masked_mean(ratio, mask_float).item()
        
        # min 和 max 只取 mask 内的值
        masked_ratio = ratio * mask_float + (1 - mask_float) * 1.0 # 不影响 min/max 如果不全为 0
        
        metadata = {
            "loss/clip_fraction": clip_fraction,
            "loss/mean": loss_mean,
            "loss/mean_original": original_loss_mean,
            "loss/mean_clipped": clipped_loss_mean,
            "ratio/mean": ratio_mean,
        }
    else:
        # 兼容没有传入 mask 的情况
        clip_fraction = was_clipped.mean().item()
        original_loss = -ratio * advantages
        clipped_loss = -clipped_ratio * advantages
        
        metadata = {
            "loss/clip_fraction": clip_fraction,
            "loss/mean": loss.mean().item(),
            "loss/mean_original": original_loss.mean().item(),
            "loss/mean_clipped": clipped_loss.mean().item(),
            "ratio/mean": ratio.mean().item(),
            "ratio/std": ratio.std().item(),
            "ratio/min": ratio.min().item(),
            "ratio/max": ratio.max().item(),
        }

    return loss, metadata

def compute_gspo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GSPO (Group Sequence Policy Optimization) 剪辑损失函数。
    
    与 GRPO 的关键区别：
    1. 重要性比率在序列级别计算（使用几何平均）
    2. Clip 在序列级别进行
    3. 所有 token 共享相同的序列级权重，消除了 token 级别的方差累积
    
    核心公式:
        s(θ) = (π_θ(y|x) / π_θ_old(y|x))^(1/|y|) = exp(mean(log_ratio))
        L(θ) = -min(s(θ) * A, clip(s(θ), 1-ε, 1+ε) * A)
    
    Args:
        advantages: torch.Tensor 形状 (batch_size,) 或 (batch_size, 1)，每个样本的优势值。
        policy_log_probs: torch.Tensor 形状 (batch_size, sequence_length)，当前策略的 token log probs。
        old_log_probs: torch.Tensor 形状 (batch_size, sequence_length)，旧策略的 token log probs。
        cliprange: float 裁剪参数 ε。
        response_mask: torch.Tensor 形状 (batch_size, sequence_length)，响应掩码。
    
    Returns:
        tuple[torch.Tensor, dict]:
            loss: 形状 (batch_size,)，每个样本的序列级损失。
            metadata: 包含日志信息的字典。
    """
    mask_float = response_mask.to(policy_log_probs.dtype)
    
    # 1. 计算 token 级 log ratio
    log_ratio = policy_log_probs - old_log_probs  # (B, T)
    
    # 2. 计算序列级 log ratio 的均值（几何平均的对数）
    # s(θ) = exp(1/|y| * Σ_t [log π_θ - log π_old])
    seq_lengths = mask_float.sum(dim=1).clamp(min=1)  # (B,)
    mean_log_ratio = (log_ratio * mask_float).sum(dim=1) / seq_lengths  # (B,)
    
    # 3. 序列级重要性比率（几何平均）
    seq_ratio = torch.exp(mean_log_ratio)  # (B,)
    
    # 4. 序列级 Clipping（而非 token 级）
    clipped_ratio = torch.clamp(seq_ratio, 1 - cliprange, 1 + cliprange)
    
    # 5. 确保 advantages 形状正确 (B,)
    if advantages.dim() == 2:
        advantages = advantages.squeeze(-1)
    
    # 6. 计算目标函数并取较小值（保守更新）
    unclipped_obj = seq_ratio * advantages
    clipped_obj = clipped_ratio * advantages
    objective = torch.min(unclipped_obj, clipped_obj)
    
    # 7. 损失 = -目标函数（最小化负的目标函数）
    loss = -objective  # (B,)
    
    # 统计信息
    was_clipped = (seq_ratio * advantages != clipped_ratio * advantages).float()
    
    metadata = {
        "loss/clip_fraction": was_clipped.mean().item(),
        "loss/mean": loss.mean().item(),
        "ratio/seq_mean": seq_ratio.mean().item(),
        "ratio/seq_std": seq_ratio.std().item() if seq_ratio.numel() > 1 else 0.0,
        "ratio/seq_min": seq_ratio.min().item(),
        "ratio/seq_max": seq_ratio.max().item(),
    }
    
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "gspo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    选择并计算所需的策略梯度损失函数。

    参数:
        policy_log_probs: torch.Tensor 形状为 (batch_size, sequence_length)，来自被训练策略的每个 token 的对数概率。
        loss_type: str 取值为 "no_baseline"、"reinforce_with_baseline"、"grpo_clip" 或 "gspo_clip" 之一。
        raw_rewards: torch.Tensor | None 当 loss_type == "no_baseline" 时需要，形状为 (batch_size, 1)。
        advantages: torch.Tensor | None 当 loss_type 为 "reinforce_with_baseline"、"grpo_clip" 和 "gspo_clip" 时需要，形状为 (batch_size, 1)。
        old_log_probs: torch.Tensor | None 当 loss_type == "grpo_clip" 或 "gspo_clip" 时需要，形状为 (batch_size, sequence_length)。
        cliprange: float | None 当 loss_type == "grpo_clip" 或 "gspo_clip" 时需要，用于裁剪的 epsilon 值。
        response_mask: torch.Tensor | None 当 loss_type == "gspo_clip" 时必须提供，用于计算序列级比率。

    返回:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: 形状为 (batch_size, sequence_length) 或 (batch_size,)，每个 token 或每个序列的损失值。
            metadata dict: 来自底层例程的统计数据（例如裁剪比例）。
    """
    if loss_type == "no_baseline":
        # 朴素策略梯度：直接使用原始奖励
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"loss_type": "no_baseline"}
    elif loss_type == "reinforce_with_baseline":
        # REINFORCE with Baseline：使用优势（减去基线的奖励）
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_type": "reinforce_with_baseline"}
    elif loss_type == "grpo_clip":
        # GRPO Clip：使用 token 级裁剪的策略梯度损失
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange are required for loss_type='grpo_clip'")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages, 
            policy_log_probs=policy_log_probs, 
            old_log_probs=old_log_probs, 
            cliprange=cliprange,
            response_mask=response_mask
        )
        metadata["loss_type"] = "grpo_clip"
    elif loss_type == "gspo_clip":
        # GSPO Clip：使用序列级裁剪的策略梯度损失
        if advantages is None or old_log_probs is None or cliprange is None or response_mask is None:
            raise ValueError("advantages, old_log_probs, cliprange, and response_mask are required for loss_type='gspo_clip'")
        loss, metadata = compute_gspo_clip_loss(
            advantages=advantages, 
            policy_log_probs=policy_log_probs, 
            old_log_probs=old_log_probs, 
            cliprange=cliprange,
            response_mask=response_mask
        )
        metadata["loss_type"] = "gspo_clip"
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    计算张量在指定维度上的掩码均值，只考虑掩码值为 1 的元素。

    参数:
        tensor: torch.Tensor，需要计算均值的张量。
        mask: torch.Tensor，掩码张量。我们只对掩码值为 1 的元素计算均值。
        dim: int | None，计算均值的维度。如果为 None，则对所有非掩码元素求和并除以它们的总数。

    返回:
        torch.Tensor，指定维度上只考虑掩码值为 1 的元素的均值。
    """
    # 1. 将掩码转换为与张量相同的数据类型
    mask_float = mask.to(tensor.dtype)

    # 2. 应用掩码：将掩码为 0 的位置置为 0
    masked_tensor = tensor * mask_float

    # 3. 计算掩码元素的数量
    # 如果指定了维度，则沿该维度计算；否则计算总体
    if dim is None:
        # 展平后计算
        count = mask_float.sum()
    else:
        count = mask_float.sum(dim=dim)

    # 4. 计算掩码均值：求和除以计数
    # 当 count = 0 时，PyTorch 会自动产生 NaN
    mean = masked_tensor.sum(dim=dim) / count

    return mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "gspo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_constant: float | None = None,
    use_length_normalization: bool = True,
    token_entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在一个微批次上计算策略梯度损失并执行反向传播。
    支持 GRPO (token 级) 和 GSPO (序列级) 两种损失类型。

    参数:
        policy_log_probs: torch.Tensor 形状为 (batch_size, sequence_length)，策略的每个 token 的对数概率。
        response_mask: torch.Tensor 形状为 (batch_size, sequence_length)，响应的掩码。
        gradient_accumulation_steps: int，梯度累加的步数。
        loss_type: str，损失函数类型，取值为 "no_baseline"、"reinforce_with_baseline"、"grpo_clip" 或 "gspo_clip"。
        raw_rewards: torch.Tensor | None，原始奖励，当 loss_type="no_baseline" 时需要。
        advantages: torch.Tensor | None，优势值，当 loss_type 需要优势时。
        old_log_probs: torch.Tensor | None，旧策略的对数概率，当 loss_type 需要时。
        cliprange: float | None，裁剪范围。
        normalize_constant: float | None，归一化常数。
        use_length_normalization: bool，是否对序列长度进行归一化（仅对 token 级损失有效）。
        token_entropy: torch.Tensor | None，token 级熵，形状为 (batch_size, sequence_length)。
        entropy_coef: float，熵奖励系数。>0 时在损失中加入 -entropy_coef * entropy_bonus。

    返回:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            缩放后的损失和元数据。
    """
    # 1. 计算策略梯度损失
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        response_mask=response_mask,
    )

    # 2. 根据损失类型选择聚合方式
    if loss_type == "gspo_clip":
        # GSPO: 损失已经是序列级别的 (batch_size,)，直接在 batch 维度求均值
        per_example_loss = loss  # (batch_size,)
    else:
        # GRPO 和其他: 损失是 token 级别的 (batch_size, seq_len)，需要在序列维度聚合
        mask_float = response_mask.to(loss.dtype)
        
        if use_length_normalization:
            # 对序列长度求均值（原始做法）
            per_example_loss = masked_mean(loss, response_mask, dim=1)
        else:
            # 对序列长度求和（推荐做法）
            per_example_loss = (loss * mask_float).sum(dim=1)

    # 3. 如果提供了 normalize_constant，则使用 Dr. GRPO 的做法
    if normalize_constant is not None and loss_type != "gspo_clip":
        # GSPO 已经是序列级别，不需要额外归一化
        normalized_loss = (per_example_loss / normalize_constant).mean()
    else:
        # 标准做法：在批次上求均值
        normalized_loss = per_example_loss.mean()

    # 4. 可选：添加熵奖励，抑制策略过早坍缩
    if entropy_coef > 0.0:
        if token_entropy is None:
            raise ValueError("token_entropy is required when entropy_coef > 0")
        entropy_bonus = masked_mean(token_entropy, response_mask)
        normalized_loss = normalized_loss - entropy_coef * entropy_bonus
    else:
        entropy_bonus = normalized_loss.new_zeros(())

    # 5. 梯度累加缩放
    scaled_loss = normalized_loss / gradient_accumulation_steps

    # 6. 执行反向传播
    scaled_loss.backward()

    # 7. 组装日志元数据
    metadata.update({
        "unscaled_loss": normalized_loss.detach(),
        "scaled_loss": scaled_loss.detach(),
        "entropy/bonus": entropy_bonus.detach(),
        "entropy/coef": float(entropy_coef),
    })

    return scaled_loss, metadata