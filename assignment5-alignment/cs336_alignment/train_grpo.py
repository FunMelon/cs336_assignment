"""
GRPO 训练脚本 - 双 GPU 版本
GPU 0: PyTorch 策略模型训练（计算 log probs、损失和梯度更新）
GPU 1: vLLM 推理实例（负责快速生成/rollout）

基于 DeepSeek R1 Zero 的 GRPO 算法实现
"""
import os
import json
import random
from typing import Literal
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# 导入辅助函数
from util import (
    tokenize_prompt_and_output,
    get_response_log_probs,
)
from rl_util import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 配置参数
# ==========================================
class Config:
    # 路径配置
    model_path = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/SFT"
    train_data_path = "dataset/gsm8k/processed_with_label/train.jsonl"
    eval_data_path = "dataset/gsm8k/processed_with_label/test.jsonl"
    output_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/GRPO"
    log_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/GRPO/logs"
    prompt_template_path = "prompts/r1_zero.prompt"
    
    # GRPO 核心训练参数
    n_grpo_steps: int = 30              # 训练步数
    learning_rate: float = 1e-5          # 学习率
    
    # 批次大小配置
    rollout_batch_size: int = 256        # rollout 微批大小
    group_size: int = 8                  # 每个 prompt 生成的回答数量
    train_batch_size: int = 256          # 宏批大小
    gradient_accumulation_steps: int = 32  # 梯度累积步数
    epochs_per_rollout_batch: int = 2    # 每个 rollout batch 训练的 epoch 数，1 = On-policy
    
    # 采样配置
    sampling_temperature: float = 0.7    # 采样温度
    sampling_min_tokens: int = 4         # 生成答案的最小 tokens 数（防止生成空字符）
    sampling_max_tokens: int = 512       # 生成答案的最大 tokens 数
    
    # 损失函数配置
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "gspo_clip"] = "gspo_clip"
    use_std_normalization: bool = False   # 是否使用标准差归一化
    use_length_normalization: bool = True  # 是否对序列长度归一化（False=求和，推荐；True=求均值，会引入长度偏置）
    cliprange: float = 0.2               # GRPO-Clip 裁剪范围（仅 off-policy 使用）
    
    # KL 散度惩罚配置
    kl_coef: float = 0.01                 # KL 惩罚系数 β，设为 0 禁用 KL 惩罚
    kl_type: Literal["kl", "abs", "mse", "low_var"] = "kl"  # KL 散度计算方式
    
    # 优化器配置
    weight_decay: float = 0.0            # 权重衰减
    betas: tuple = (0.9, 0.95)           # AdamW betas
    max_grad_norm: float = 1.0           # 梯度裁剪阈值
    
    # 其他配置
    advantage_eps: float = 1e-6          # 计算优势时防止除零
    
    # 验证与日志配置
    eval_interval: int = 2               # 每隔多少步进行验证
    log_interval: int = 2               # 每隔多少步打印日志
    save_interval: int = 5              # 每隔多少步保存检查点
    max_eval_samples: int = 1024         # 验证样本数量（至少 1024 个以减少噪音）
    
    # GPU 配置
    train_device = "cuda:0"              # 训练 GPU
    vllm_device = "cuda:1"               # vLLM GPU
    vllm_gpu_memory_utilization: float = 0.85

# ==========================================
# 健全性检查（Sanity Checks）
# ==========================================
def validate_config(config: Config):
    """验证配置参数的合法性"""
    assert config.train_batch_size % config.gradient_accumulation_steps == 0, (
        f"train_batch_size ({config.train_batch_size}) 必须能被 "
        f"gradient_accumulation_steps ({config.gradient_accumulation_steps}) 整除"
    )
    
    assert config.rollout_batch_size % config.group_size == 0, (
        f"rollout_batch_size ({config.rollout_batch_size}) 必须能被 "
        f"group_size ({config.group_size}) 整除"
    )
    
    assert config.train_batch_size >= config.group_size, (
        f"train_batch_size ({config.train_batch_size}) 必须大于或等于 "
        f"group_size ({config.group_size})"
    )
    
    # rollout_batch_size 必须能被 train_batch_size 整除
    assert config.rollout_batch_size % config.train_batch_size == 0, (
        f"rollout_batch_size ({config.rollout_batch_size}) 必须能被 "
        f"train_batch_size ({config.train_batch_size}) 整除"
    )
    
    # GRPO-Clip 和 GSPO-Clip 仅在 off-policy 设置下使用
    if config.loss_type in ["grpo_clip", "gspo_clip"]:
        assert config.epochs_per_rollout_batch > 1, (
            f"{config.loss_type} 仅在 off-policy（epochs_per_rollout_batch > 1）设置下使用，"
            "因为它需要旧的对数概率"
        )
    
    micro_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    n_optimizer_steps_per_epoch = config.rollout_batch_size // config.train_batch_size
    
    print(f"[配置验证通过]")
    print(f"  - 微批大小 (micro_batch_size): {micro_batch_size}")
    print(f"  - 每个 rollout batch 的 prompt 数量: {config.rollout_batch_size // config.group_size}")
    print(f"  - 每个 rollout batch 的微批数量: {config.rollout_batch_size // micro_batch_size}")
    print(f"  - 每个 epoch 的优化器更新次数: {n_optimizer_steps_per_epoch}")
    print(f"  - 每个 rollout batch 总优化器更新次数: {n_optimizer_steps_per_epoch * config.epochs_per_rollout_batch}")

# ==========================================
# 数据集类
# ==========================================
class GSM8KDataset(Dataset):
    """GSM8K 数据集"""
    def __init__(self, data_path, max_samples=None):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.data.append(item)
        print(f"加载了 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'label': item.get('label', '')
        }

def load_prompt_template(template_path: str) -> str:
    """加载 prompt 模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def format_prompt(template: str, question: str) -> str:
    """使用模板格式化 prompt"""
    return template.format(question=question)

# ==========================================
# vLLM 初始化与管理
# ==========================================
def init_vllm(model_id: str, device: str, seed: int = 42, gpu_memory_utilization: float = 0.85):
    """
    初始化 vLLM 推理引擎
    使用 monkeypatch 确保 vLLM 在指定设备上运行
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    
    vllm_set_random_seed(seed)
    
    # Monkeypatch 来自 TRL
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm):
    """
    将训练中的模型权重直接加载到 vLLM 实例中
    避免保存到磁盘再加载的开销
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ==========================================
# Rollout 函数（使用 vLLM 生成）
# ==========================================
def rollout_with_vllm(
    llm,
    prompts: list[str],
    group_size: int,
    sampling_params,
) -> tuple[list[str], list[str]]:
    """
    使用 vLLM 为每个 prompt 生成 group_size 个回答
    
    Args:
        llm: vLLM 实例
        prompts: 原始 prompt 列表（未重复）
        group_size: 每个 prompt 生成的回答数量
        sampling_params: vLLM 采样参数
    
    Returns:
        repeated_prompts: 重复后的 prompt 列表，长度 = len(prompts) * group_size
        responses: 生成的回答列表，长度 = len(prompts) * group_size
    """
    # 每个 prompt 重复 group_size 次
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * group_size)
    
    # 批量生成
    outputs = llm.generate(repeated_prompts, sampling_params)
    
    # 提取生成的回答
    responses = [output.outputs[0].text for output in outputs]
    
    return repeated_prompts, responses

def evaluate_with_vllm(
    llm,
    eval_data: list[dict],
    prompt_template: str,
    sampling_params,
) -> dict:
    """
    使用 vLLM 进行验证评估
    
    Args:
        llm: vLLM 实例
        eval_data: 验证数据列表
        prompt_template: prompt 模板
        sampling_params: 采样参数（验证时通常用 greedy）
    
    Returns:
        包含评估指标的字典
    """
    # 准备 prompts
    prompts = [format_prompt(prompt_template, item['question']) for item in eval_data]
    labels = [item.get('label', '') for item in eval_data]
    
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)
    
    # 评估
    # 根据文档要求：准确率必须通过对 answer_reward 求平均值来计算
    total_answer_reward = 0.0
    total_format_reward = 0.0
    total_reward = 0.0
    
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(response, labels[i])
        
        total_reward += reward_dict['reward']
        total_answer_reward += reward_dict['answer_reward']
        total_format_reward += reward_dict['format_reward']
    
    n_samples = len(outputs)
    return {
        # 准确率 = answer_reward 的平均值（文档明确要求）
        'accuracy': total_answer_reward / n_samples if n_samples > 0 else 0,
        # 格式率 = format_reward 的平均值
        'format_rate': total_format_reward / n_samples if n_samples > 0 else 0,
        'mean_reward': total_reward / n_samples if n_samples > 0 else 0,
        'total_samples': n_samples,
    }

# ==========================================
# 主训练函数
# ==========================================
def train():
    config = Config()
    
    # 设置随机种子，确保可复现性
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 验证配置
    validate_config(config)
    
    # 计算派生参数
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size
    n_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size
    n_optimizer_steps_per_epoch = config.rollout_batch_size // config.train_batch_size
    
    print(f"\n[派生参数]")
    print(f"  - micro_train_batch_size: {micro_train_batch_size}")
    print(f"  - n_prompts_per_rollout_batch: {n_prompts_per_rollout_batch}")
    print(f"  - n_microbatches_per_rollout_batch: {n_microbatches_per_rollout_batch}")
    print(f"  - n_optimizer_steps_per_epoch: {n_optimizer_steps_per_epoch}")
    print(f"  - total_optimizer_steps_per_rollout: {n_optimizer_steps_per_epoch * config.epochs_per_rollout_batch}")
    
    # ==========================================
    # 1. 初始化 TensorBoard
    # ==========================================
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"\nTensorBoard 日志目录: {config.log_dir}")
    
    # ==========================================
    # 2. 加载 Prompt 模板
    # ==========================================
    prompt_template = load_prompt_template(config.prompt_template_path)
    print(f"Prompt 模板:\n{prompt_template[:100]}...")
    
    # ==========================================
    # 3. 加载训练模型（GPU 0）
    # ==========================================
    print(f"\n在 {config.train_device} 上加载策略模型...")
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(config.train_device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"策略模型加载完成，参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
    # ==========================================
    # 4. 初始化 vLLM（GPU 1）
    # ==========================================
    print(f"\n在 {config.vllm_device} 上初始化 vLLM...")
    vllm_engine = init_vllm(
        model_id=config.model_path,
        device=config.vllm_device,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization
    )
    print("vLLM 初始化完成")
    
    # ==========================================
    # 5. 设置 vLLM 采样参数
    # ==========================================
    from vllm import SamplingParams
    
    # Rollout 采样参数（带温度）
    rollout_sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        top_p=1.0,
        min_tokens=config.sampling_min_tokens,
        max_tokens=config.sampling_max_tokens,
        stop=["</answer>"],          # 在 </answer> 标签处停止
        include_stop_str_in_output=True,  # 包含停止字符串
    )
    
    # 验证采样参数（与推理保持一致）
    eval_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=config.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # ==========================================
    # 6. 准备数据集
    # ==========================================
    train_dataset = GSM8KDataset(config.train_data_path)
    eval_dataset = GSM8KDataset(config.eval_data_path, max_samples=config.max_eval_samples)
    eval_data_raw = eval_dataset.data
    
    # 创建无限数据迭代器
    def infinite_data_iterator(dataset, batch_size):
        """创建无限循环的数据迭代器"""
        indices = list(range(len(dataset)))
        while True:
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) == batch_size:
                    yield [dataset[idx] for idx in batch_indices]
    
    data_iterator = infinite_data_iterator(train_dataset, n_prompts_per_rollout_batch)
    
    # ==========================================
    # 7. 初始化优化器
    # ==========================================
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ==========================================
    # 8. 训练前初始验证
    # ==========================================
    print("\n[训练前] 进行初始验证...")
    initial_eval_results = evaluate_with_vllm(
        llm=vllm_engine,
        eval_data=eval_data_raw,
        prompt_template=prompt_template,
        sampling_params=eval_sampling_params,
    )
    print(f"初始验证结果: 准确率={initial_eval_results['accuracy']*100:.2f}%, "
          f"格式率={initial_eval_results['format_rate']*100:.2f}%, "
          f"平均奖励={initial_eval_results['mean_reward']:.4f}")
    
    # TensorBoard 记录初始验证（step=0）
    writer.add_scalar("eval/accuracy", initial_eval_results['accuracy'], 0)
    writer.add_scalar("eval/format_rate", initial_eval_results['format_rate'], 0)
    writer.add_scalar("eval/mean_reward", initial_eval_results['mean_reward'], 0)
    
    # ==========================================
    # 9. GRPO 训练循环
    # ==========================================
    print("\n" + "=" * 60)
    print("开始 GRPO 训练 (双 GPU 模式)...")
    print(f"训练 GPU: {config.train_device}")
    print(f"vLLM GPU: {config.vllm_device}")
    print(f"损失类型: {config.loss_type}")
    print(f"标准差归一化: {config.use_std_normalization}")
    print("=" * 60 + "\n")
    
    for grpo_step in range(1, config.n_grpo_steps + 1):
        # ------------------------------------------
        # 8.1 同步权重到 vLLM
        # ------------------------------------------
        load_policy_into_vllm_instance(policy, vllm_engine)
        
        # ------------------------------------------
        # 8.2 采样一批 prompts
        # ------------------------------------------
        batch_data = next(data_iterator)
        questions = [item['question'] for item in batch_data]
        ground_truths = [item['label'] for item in batch_data]
        
        # 格式化 prompts
        prompts = [format_prompt(prompt_template, q) for q in questions]
        
        # ------------------------------------------
        # 8.3 使用 vLLM 进行 Rollout
        # ------------------------------------------
        repeated_prompts, rollout_responses = rollout_with_vllm(
            llm=vllm_engine,
            prompts=prompts,
            group_size=config.group_size,
            sampling_params=rollout_sampling_params,
        )
        
        # 重复 ground_truths 以匹配 rollout_responses
        repeated_ground_truths = []
        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * config.group_size)
        
        # ------------------------------------------
        # 8.4 计算组归一化奖励（优势）
        # ------------------------------------------
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )
        
        # 将优势移动到训练设备
        advantages = advantages.to(config.train_device)
        raw_rewards = raw_rewards.to(config.train_device)
        
        # ------------------------------------------
        # 8.5 对于 off-policy（GRPO-Clip/GSPO-Clip），预先计算旧的 log probs
        # ------------------------------------------
        old_log_probs_all = None
        if config.loss_type in ["grpo_clip", "gspo_clip"] and config.epochs_per_rollout_batch > 1:
            # 只计算一次，不对其求导
            policy.eval()
            with torch.inference_mode():
                # 分词
                tokenized = tokenize_prompt_and_output(
                    repeated_prompts, rollout_responses, tokenizer
                )
                input_ids = tokenized['input_ids'].to(config.train_device)
                labels = tokenized['labels'].to(config.train_device)
                
                # 分批获取旧的 log probs，防止 OOM
                all_old_log_probs = []
                for i in range(0, config.rollout_batch_size, micro_train_batch_size):
                    batch_input_ids = input_ids[i:i + micro_train_batch_size]
                    batch_labels = labels[i:i + micro_train_batch_size]
                    
                    batch_old_log_probs_dict = get_response_log_probs(
                        model=policy,
                        input_ids=batch_input_ids,
                        labels=batch_labels,
                        return_token_entropy=False,
                    )
                    all_old_log_probs.append(batch_old_log_probs_dict['log_probs'].detach())
                
                old_log_probs_all = torch.cat(all_old_log_probs, dim=0)
            policy.train()
        
        # ------------------------------------------
        # 8.6 训练 epochs（每个 rollout batch）
        # ------------------------------------------
        policy.train()
        
        # 计算每个 epoch 可以进行多少次优化器更新
        # 更新频率由 train_batch_size 控制
        n_optimizer_steps_per_epoch = config.rollout_batch_size // config.train_batch_size
        
        # 分词一次（整个 rollout batch）
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )
        input_ids_all = tokenized['input_ids'].to(config.train_device)
        labels_all = tokenized['labels'].to(config.train_device)
        response_mask_all = tokenized['response_mask'].to(config.train_device)
        
        total_loss = 0.0
        total_entropy = 0.0  # 累积熵值
        total_entropy_tokens = 0  # 累积有效 token 数
        total_optimizer_steps = 0
        grad_norm = 0.0
        
        for epoch in range(config.epochs_per_rollout_batch):
            # 每个 epoch 打乱索引
            indices = list(range(config.rollout_batch_size))
            random.shuffle(indices)
            
            # 全局微批次索引（在当前 epoch 内）
            global_micro_idx = 0
            
            # 每个 epoch 进行 n_optimizer_steps_per_epoch 次优化器更新
            for opt_step in range(n_optimizer_steps_per_epoch):
                optimizer.zero_grad()
                
                # 每次优化器更新累积 gradient_accumulation_steps 个微批次
                for accum_idx in range(config.gradient_accumulation_steps):
                    start_idx = global_micro_idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size
                    batch_indices = indices[start_idx:end_idx]
                    
                    # 提取微批数据
                    input_ids = input_ids_all[batch_indices]
                    labels = labels_all[batch_indices]
                    response_mask = response_mask_all[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_raw_rewards = raw_rewards[batch_indices]
                    
                    # 获取当前策略的 log probs 和 token 熵
                    log_probs_dict = get_response_log_probs(
                        model=policy,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=True,  # 开启熵计算
                    )
                    policy_log_probs = log_probs_dict['log_probs']
                    
                    # 累积 token 熵（只计算 response 部分的熵）
                    token_entropy = log_probs_dict['token_entropy']
                    # 使用 response_mask 计算掩码内的熵总和和 token 数
                    masked_entropy_sum = (token_entropy * response_mask).sum().item()
                    n_response_tokens = response_mask.sum().item()
                    total_entropy += masked_entropy_sum
                    total_entropy_tokens += n_response_tokens
                    
                    # 准备旧的 log probs（仅 off-policy）
                    batch_old_log_probs = None
                    if old_log_probs_all is not None:
                        batch_old_log_probs = old_log_probs_all[batch_indices]
                    
                    # 计算损失并反向传播
                    # 注意：KL 惩罚使用 old_log_probs 作为参考模型的 log probs
                    # 这意味着我们惩罚策略偏离 rollout 时的策略太远
                    scaled_loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=batch_raw_rewards.unsqueeze(-1) if config.loss_type == "no_baseline" else None,
                        advantages=batch_advantages.unsqueeze(-1) if config.loss_type != "no_baseline" else None,
                        old_log_probs=batch_old_log_probs,
                        cliprange=config.cliprange if config.loss_type in ["grpo_clip", "gspo_clip"] else None,
                        use_length_normalization=config.use_length_normalization,
                        # KL 散度惩罚参数
                        ref_log_probs=batch_old_log_probs,  # 使用 rollout 时的策略作为参考
                        kl_coef=config.kl_coef,
                        kl_type=config.kl_type,
                    )
                    
                    total_loss += loss_metadata['unscaled_loss'].item()
                    global_micro_idx += 1
                
                # 梯度裁剪（返回的是裁剪前的梯度范数）
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 
                    max_norm=config.max_grad_norm
                )
                
                # 更新参数
                optimizer.step()
                total_optimizer_steps += 1
        
        # 计算平均损失（基于总的优化器更新次数 * 梯度累积步数）
        n_total_microbatches = total_optimizer_steps * config.gradient_accumulation_steps
        avg_loss = total_loss / n_total_microbatches if n_total_microbatches > 0 else 0.0
        
        # 计算平均 token 熵
        avg_entropy = total_entropy / total_entropy_tokens if total_entropy_tokens > 0 else 0.0
        
        # ------------------------------------------
        # 8.7 记录日志
        # ------------------------------------------
        if grpo_step % config.log_interval == 0:
            clipped_indicator = " [CLIPPED]" if grad_norm > config.max_grad_norm else ""
            # 获取 KL 相关日志（如果启用）
            kl_info = ""
            if config.kl_coef > 0 and 'kl/mean' in loss_metadata:
                kl_info = f" | KL: {loss_metadata.get('kl/mean', 0):.4f}"
            
            print(f"[Step {grpo_step}/{config.n_grpo_steps}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Reward: {reward_metadata['reward/mean']:.4f} (std: {reward_metadata['reward/std']:.4f}) | "
                  f"Format: {reward_metadata['reward/format_mean']:.2f} | "
                  f"Answer: {reward_metadata['reward/answer_mean']:.2f} | "
                  f"Entropy: {avg_entropy:.4f}{kl_info} | "
                  f"GradNorm: {grad_norm:.2f}{clipped_indicator} (max: {config.max_grad_norm})")
            
            # TensorBoard 记录
            writer.add_scalar("train/loss", avg_loss, grpo_step)
            writer.add_scalar("train/grad_norm", grad_norm, grpo_step)
            writer.add_scalar("train/grad_norm_clipped", min(grad_norm, config.max_grad_norm), grpo_step)
            writer.add_scalar("train/grad_clip_ratio", grad_norm / config.max_grad_norm, grpo_step)
            writer.add_scalar("train/reward_mean", reward_metadata['reward/mean'], grpo_step)
            writer.add_scalar("train/reward_std", reward_metadata['reward/std'], grpo_step)
            writer.add_scalar("train/format_reward", reward_metadata['reward/format_mean'], grpo_step)
            writer.add_scalar("train/answer_reward", reward_metadata['reward/answer_mean'], grpo_step)
            writer.add_scalar("train/advantage_mean", advantages.mean().item(), grpo_step)
            writer.add_scalar("train/advantage_std", advantages.std().item(), grpo_step)
            # Token 熵 - 监控模型是否过度自信或模式崩溃的关键指标
            writer.add_scalar("train/token_entropy", avg_entropy, grpo_step)
            
            # KL 散度相关日志（如果启用）
            if config.kl_coef > 0:
                writer.add_scalar("train/kl_mean", loss_metadata.get('kl/mean', 0), grpo_step)
                writer.add_scalar("train/kl_loss", loss_metadata.get('kl/loss', 0), grpo_step)
                writer.add_scalar("train/policy_loss", loss_metadata.get('loss/policy', avg_loss), grpo_step)
        
        # ------------------------------------------
        # 8.8 定期验证
        # ------------------------------------------
        if grpo_step % config.eval_interval == 0:
            print(f"\n[Step {grpo_step}] 进行验证（{len(eval_data_raw)} 个样本）...")
            
            # 同步最新权重到 vLLM
            load_policy_into_vllm_instance(policy, vllm_engine)
            
            # 执行验证
            eval_results = evaluate_with_vllm(
                llm=vllm_engine,
                eval_data=eval_data_raw,
                prompt_template=prompt_template,
                sampling_params=eval_sampling_params,
            )
            
            print(f"验证结果: 准确率={eval_results['accuracy']*100:.2f}%, "
                  f"格式率={eval_results['format_rate']*100:.2f}%, "
                  f"平均奖励={eval_results['mean_reward']:.4f}")
            
            # TensorBoard 记录
            writer.add_scalar("eval/accuracy", eval_results['accuracy'], grpo_step)
            writer.add_scalar("eval/format_rate", eval_results['format_rate'], grpo_step)
            writer.add_scalar("eval/mean_reward", eval_results['mean_reward'], grpo_step)
            
            print()
        
        # ------------------------------------------
        # 8.9 保存检查点
        # ------------------------------------------
        if grpo_step % config.save_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint-{grpo_step}")
            os.makedirs(ckpt_path, exist_ok=True)
            policy.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"[Step {grpo_step}] 模型已保存到 {ckpt_path}\n")
    
    # ==========================================
    # 10. 保存最终模型
    # ==========================================
    print(f"\n训练完成! 保存最终模型到 {config.output_dir}")
    policy.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 最终验证
    print("\n进行最终验证...")
    load_policy_into_vllm_instance(policy, vllm_engine)
    final_eval_results = evaluate_with_vllm(
        llm=vllm_engine,
        eval_data=eval_data_raw,
        prompt_template=prompt_template,
        sampling_params=eval_sampling_params,
    )
    print(f"最终验证结果: 准确率={final_eval_results['accuracy']*100:.2f}%, "
          f"格式率={final_eval_results['format_rate']*100:.2f}%")
    
    # 关闭 TensorBoard writer
    writer.close()
    print(f"\nTensorBoard 日志已保存到: {config.log_dir}")
    print("使用命令查看: tensorboard --logdir=" + config.log_dir)
    print("\nGRPO 训练全部完成!")

if __name__ == "__main__":
    train()
