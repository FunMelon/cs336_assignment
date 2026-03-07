import torch
import torch.nn.functional as F

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