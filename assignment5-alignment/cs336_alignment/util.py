import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> dict[str, torch.Tensor]: 
    """
    对提示和输出字符串进行分词，并构建一个掩码：响应token位置为1，其他token（提示或填充）位置为0。
    
    参数:
        prompt_strs: list[str] 提示字符串列表。
        output_strs: list[str] 输出字符串列表。
        tokenizer: PreTrainedTokenizer 用于分词的分词器。
    返回:
        dict[str, torch.Tensor]。设prompt_and_output_lens为分词后的提示和输出字符串长度列表，则返回的字典应包含以下键:
            input_ids torch.Tensor 形状为 (batch_size, max(prompt_and_output_lens) - 1):
                分词后的提示和输出字符串，去掉最后一个token。
            labels torch.Tensor 形状为 (batch_size, max(prompt_and_output_lens) - 1):
                偏移后的input ids，即去掉第一个token的input ids。
            response_mask torch.Tensor 形状为 (batch_size, max(prompt_and_output_lens) - 1):
                labels中响应token的掩码。
    """
    batch_size = len(prompt_strs)
    
    # 分别对提示和输出字符串进行分词
    prompt_token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_token_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    
    # 拼接提示和输出的token（分别分词后拼接，而不是对组合字符串分词）
    combined_token_ids = [p + o for p, o in zip(prompt_token_ids, output_token_ids)]
    
    # 找到最大长度
    max_len = max(len(ids) for ids in combined_token_ids)
    
    # 获取填充token id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    input_ids_list = []
    labels_list = []
    response_mask_list = []
    
    for i in range(batch_size):
        combined = combined_token_ids[i]
        prompt_len = len(prompt_token_ids[i])
        combined_len = len(combined)
        output_len = combined_len - prompt_len
        
        # 先将序列填充到max_len
        pad_len = max_len - combined_len
        padded = combined + [pad_token_id] * pad_len
        
        # input_ids是填充后去掉最后一个token
        # labels是填充后去掉第一个token（偏移后的序列）
        input_ids = padded[:-1]
        labels = padded[1:]
        
        # 创建响应掩码：labels中输出token位置为1
        # 在labels中:
        # - 位置0到prompt_len-2是提示token（共prompt_len-1个）
        # - 位置prompt_len-1到prompt_len-1+output_len-1是输出token（共output_len个）
        # - 剩余位置是填充（掩码为0）
        response_mask = [0] * (prompt_len - 1) + [1] * output_len + [0] * pad_len
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        response_mask_list.append(response_mask)
    
    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long),
        'response_mask': torch.tensor(response_mask_list, dtype=torch.long),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算下一个预测 token 的信息熵（即在整个词表维度上的概率分布的熵）。
    
    参数:
        logits: torch.Tensor 
            形状为 (batch_size, sequence_length, vocab_size) 的张量，包含模型输出的未归一化得分 (logits)。
            
    返回:
        torch.Tensor 
            形状为 (batch_size, sequence_length)。返回每个位置上预测下一个 token 的信息熵。
    """
    # 1. 使用 F.log_softmax 计算对数概率 log(p)。
    # 强烈建议不要直接先算 softmax 再算 torch.log，因为当概率极小(接近0)时，
    # torch.log 会出现 -inf 或 NaN。F.log_softmax 底层使用了 logsumexp，具有极佳的数值稳定性。
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 2. 通过指数函数还原出概率分布 p
    probs = torch.exp(log_probs)
    
    # 3. 计算信息熵公式: H(p) = -sum(p * log(p))
    # 我们需要在词表维度 (dim=-1) 上将所有的 token 概率进行求和，从而将维度降维。
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    获取模型对给定序列中目标 token 的条件对数概率 log pθ(xt | x<t)。
    
    参数:
        model: PreTrainedModel 
            用于打分的 HuggingFace 模型。
        input_ids: torch.Tensor 
            形状为 (batch_size, sequence_length)，由分词方法生成的 prompt + response 拼接 token。
        labels: torch.Tensor 
            形状为 (batch_size, sequence_length)，偏移后的目标 token（即去掉第一个 token 的序列）。
        return_token_entropy: bool 
            如果为 True，则通过调用 compute_entropy 返回每个 token 的信息熵。

    返回:
        dict[str, torch.Tensor] 包含以下键值对的字典:
            "log_probs": 形状为 (batch_size, sequence_length)，实际目标词的条件对数概率。
            "token_entropy": 形状为 (batch_size, sequence_length)（仅在 return_token_entropy=True 时存在）。
    """
    
    # 1. 前向传播：将输入序列喂给模型，获取原始预测得分 (logits)
    # outputs.logits 的形状是: (batch_size, sequence_length, vocab_size)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    
    # 2. 计算整个词表维度上的对数概率分布
    # log_probs_all 形状同样是: (batch_size, sequence_length, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)
    
    # 3. 精准摘取 (Gather) 目标 Token 的对数概率
    # 因为 labels 中可能包含 PyTorch 默认的忽略索引 (如 -100)，直接作为索引会报错 out-of-bounds。
    # 我们先克隆一份 labels，并将小于 0 的值替换为 0 (或其他合法词表索引)。
    # 反正这些 padding 位置最终会被 response_mask 给过滤掉，所以取什么值都不影响最终的 loss。
    valid_labels = labels.clone()
    valid_labels[valid_labels < 0] = 0 
    
    # 使用 torch.gather 从 vocab_size 这个维度 (dim=-1) 中，提取出 valid_labels 对应位置的概率值。
    # valid_labels.unsqueeze(-1) 将形状变为 (B, S, 1)，提取后返回 (B, S, 1)，最后 squeeze 掉最后一维恢复为 (B, S)。
    
    log_probs = torch.gather(
        log_probs_all, 
        dim=-1, 
        index=valid_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # 4. 构建返回字典
    result = {
        "log_probs": log_probs
    }
    
    # 5. 如果需要，计算并附加信息熵
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
        
    return result

import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    对张量进行掩码求和，并除以指定的常数进行归一化。
    
    参数:
        tensor: torch.Tensor 
            需要被求和和归一化的张量（通常是 per-token 的交叉熵损失或奖励值）。
        mask: torch.Tensor 
            与 tensor 形状完全相同的掩码；值为 1 的位置参与求和，值为 0 的位置被丢弃。
        normalize_constant: float 
            归一化常数，求和后需除以该值（比如有效 token 的总数，或批次大小）。
        dim: int | None 
            在归一化之前沿哪个维度进行求和。如果为 None，则将张量展平对所有元素求和。
            
    返回:
        torch.Tensor 
            归一化后的结果。被掩码遮蔽的元素（mask == 0）对总和的贡献严格为 0。
    """
    # 1. 对齐数据类型：掩码通常是 int 或 bool 类型，而 tensor（如 loss）是 float 或 bfloat16。
    # 必须先将 mask 转换成与 tensor 一致的数据类型，否则相乘时 PyTorch 会报错。
    mask_float = mask.to(tensor.dtype)
    
    # 2. 应用掩码 (Masking)：通过逐元素相乘 (Element-wise multiplication)，
    # 将 mask 为 0 的位置强制置为 0，有效位置的值保持不变。
    masked_tensor = tensor * mask_float
    
    # 3. 沿指定维度求和 (Summation)
    if dim is None:
        sum_val = masked_tensor.sum()
    else:
        sum_val = masked_tensor.sum(dim=dim)
        
    # 4. 归一化 (Normalization)：除以指定的常数
    normalized_val = sum_val / normalize_constant
    
    return normalized_val

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在一个微批次 (microbatch) 上执行前向损失计算与反向传播。
    """
    
    # 1. 计算每个 Token 的负对数似然损失
    per_token_loss = -policy_log_probs
    
    # 2. 掩码与归一化
    # 不能直接把整个 batch 糊在一起求 sum。每个 sample 长度不一样，要分别求。
    # 我们使用 dim=1（序列维度）来求和，这样会返回一个形状为 (batch_size,) 的张量，
    # 代表每个样本各自的归一化损失。
    per_example_loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1
    )
    
    # 然后再对整个批次求平均值 (mean)，这会自动除以 batch_size
    normalized_loss = per_example_loss.mean()
    
    # 3. 梯度累加缩放
    scaled_loss = normalized_loss / gradient_accumulation_steps
    
    # 4. 执行反向传播
    scaled_loss.backward()
    
    # 5. 组装日志元数据
    metadata = {
        "unscaled_loss": normalized_loss.detach(),
    }
    
    return scaled_loss, metadata


@torch.no_grad() # 生成阶段不需要计算梯度，节省显存并加速
def log_generations(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: callable,
    step: int = 0,
    max_new_tokens: int = 512,
):
    """
    在训练后台生成回复，计算奖励与指标，并打印/记录日志。
    """
    # 1. 切换到评估模式
    model.eval()
    
    # 2. 准备输入
    # 左侧填充 (Left Padding) 对于仅解码的大模型生成是必须的
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    
    # 3. 呼叫模型生成文本
    # 开启 return_dict_in_generate 和 output_scores 以便后续计算信息熵
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,  # 明确要求返回原始 logits
        do_sample=True, # 使用采样以观察真实分布的熵
        temperature=1.0,
    )
    
    generated_sequences = outputs.sequences
    # output_logits=True 时使用 outputs.logits，否则使用 outputs.scores
    if hasattr(outputs, 'logits') and outputs.logits is not None:
        scores = outputs.logits  # list of (batch_size, vocab_size)
    else:
        scores = outputs.scores  # tuple of (batch_size, vocab_size)
    
    # 4. 提取生成的纯回复部分 (截掉前面的 prompt)
    response_ids = generated_sequences[:, prompt_length:]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    # 5. 计算指标：长度与奖励
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_length = 0
    
    for i, (response, gt) in enumerate(zip(responses, ground_truths)):
        # 调用你之前提供的 reward_fn
        reward_dict = reward_fn(response, gt)
        total_format_reward += reward_dict["format_reward"]
        total_answer_reward += reward_dict["answer_reward"]
        
        # 计算该回答的有效长度 (非 pad token 的数量)
        valid_length = (response_ids[i] != tokenizer.pad_token_id).sum().item()
        total_length += valid_length
        
        # 仅在终端打印批次中的第一条数据作为肉眼检查
        if i == 0:
            print(f"\n{'='*20} 训练步数: {step} {'='*20}")
            print(f"❓ [提示词]:\n{prompts[i]}")
            print(f"✅ [标答]: {gt}")
            print(f"🤖 [模型生成]:\n{response}")
            print(f"🏅 [评分]: 格式={reward_dict['format_reward']}, 答案={reward_dict['answer_reward']}")
            print(f"{'='*55}\n")

    batch_size = len(prompts)
    avg_format_reward = total_format_reward / batch_size
    avg_answer_reward = total_answer_reward / batch_size
    avg_length = total_length / batch_size
    
    # 6. 计算平均信息熵 (Entropy)
    # scores 可能是:
    # - tuple of (batch_size, vocab_size) 当使用 output_scores=True 时
    # - tensor of (batch_size, generated_len, vocab_size) 当使用 output_logits=True 时
    try:
        if isinstance(scores, torch.Tensor):
            # output_logits=True 返回的格式已经是 (batch_size, generated_len, vocab_size)
            stacked_logits = scores
        else:
            # output_scores=True 返回的格式是 tuple，需要 stack
            stacked_logits = torch.stack(scores, dim=1)
        
        # 调用 compute_entropy，返回形状 (batch_size, generated_len)
        entropies = compute_entropy(stacked_logits)
        # 只计算有效位置的熵（非 pad 位置）
        # 使用 attention_mask 或根据生成的 token 来确定有效位置
        valid_entropy = entropies[entropies > 0]
        avg_entropy = valid_entropy.mean().item() if len(valid_entropy) > 0 else 0.0
    except Exception as e:
        print(f"计算熵时出错: {e}")
        avg_entropy = 0.0
        
    # 7. 汇总日志字典返回
    metrics = {
        "eval/avg_format_reward": avg_format_reward,
        "eval/avg_answer_reward": avg_answer_reward,
        "eval/avg_response_length": avg_length,
        "eval/avg_entropy": avg_entropy,
    }
    
    print(f"📊 [批次统计] 步数: {step} | 格式分: {avg_format_reward:.2f} | 答案分: {avg_answer_reward:.2f} | 长度: {avg_length:.1f} | 熵: {avg_entropy:.4f}")
    
    # 切回训练模式，不要影响后续的 SFT 梯度计算
    tokenizer.padding_side = "right" # 恢复训练时常用的右侧填充
    model.train()
    
    return metrics