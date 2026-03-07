import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
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