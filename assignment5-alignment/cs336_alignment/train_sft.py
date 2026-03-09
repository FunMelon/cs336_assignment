"""
SFT 快速训练脚本 - 双 GPU 同进程版本
GPU 0: PyTorch 训练
GPU 1: vLLM 推理（使用 load_policy_into_vllm_instance 直接内存传输权重）

优点: 省去磁盘 I/O，验证速度更快
缺点: 两个模型同时在显存中，需要更多总显存
"""
import os
import json
import math
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from cs336_basics import cosine_anneal_schedule

# 导入辅助函数
from util import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
)
from drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 配置参数
# ==========================================
class Config:
    # 路径配置
    model_path = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/Qwen2.5-Math-1.5B"
    train_data_path = "dataset/gsm8k/processed_with_label/train.jsonl"
    eval_data_path = "dataset/gsm8k/processed_with_label/test.jsonl"
    output_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/SFT"
    log_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/SFT/logs"
    
    # 训练超参数
    epochs = 3
    batch_size = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    max_seq_length = 1024
    
    # 验证配置
    eval_steps = 500
    log_steps = 100
    save_steps = 1000
    eval_batch_size = 16
    max_new_tokens = 512
    max_eval_samples = 200
    
    # GPU 配置
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    vllm_gpu_memory_utilization = 0.80

# ==========================================
# 数据集类
# ==========================================
class GSM8KDataset(Dataset):
    """GSM8K 数据集"""
    def __init__(self, data_path, tokenizer, max_samples=None):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.data.append(item)
        self.tokenizer = tokenizer
        print(f"加载了 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item['question'],
            'response': item['answer'],
            'label': item.get('label', '')
        }

def collate_fn(batch, tokenizer):
    """数据整理函数"""
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    labels = [item['label'] for item in batch]
    
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    tokenized['ground_truths'] = labels
    tokenized['prompts'] = prompts
    
    return tokenized

# ==========================================
# vLLM 初始化与权重加载
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

def evaluate_with_vllm(llm, eval_data, max_new_tokens=512):
    """使用 vLLM 进行验证"""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    
    prompts = [item['question'] for item in eval_data]
    labels = [item.get('label', '') for item in eval_data]
    
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)
    
    # 评估
    total_correct = 0
    total_format_correct = 0
    
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        reward_dict = r1_zero_reward_fn(response, labels[i])
        
        if reward_dict['answer_reward'] > 0:
            total_correct += 1
        if reward_dict['format_reward'] > 0:
            total_format_correct += 1
    
    accuracy = total_correct / len(outputs) if outputs else 0
    format_rate = total_format_correct / len(outputs) if outputs else 0
    
    return {
        'accuracy': accuracy,
        'format_rate': format_rate,
        'total_correct': total_correct,
        'total_samples': len(outputs),
    }

# ==========================================
# 主训练函数
# ==========================================
def train():
    config = Config()

    # ==========================================
    # 1. 初始化 TensorBoard
    # ==========================================
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"TensorBoard 日志目录: {config.log_dir}")

    # ==========================================
    # 2. 加载训练模型（GPU 0）
    # ==========================================
    print(f"在 {config.train_device} 上加载训练模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"  # 使用 PyTorch 原生 SDPA，兼容性更好
    ).to(config.train_device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"训练模型加载完成，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ==========================================
    # 3. 初始化 vLLM（GPU 1）
    # ==========================================
    print(f"在 {config.vllm_device} 上初始化 vLLM...")
    vllm_engine = init_vllm(
        model_id=config.model_path,
        device=config.vllm_device,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization
    )
    print("vLLM 初始化完成")
    
    # ==========================================
    # 4. 准备数据集
    # ==========================================
    train_dataset = GSM8KDataset(config.train_data_path, tokenizer)
    eval_dataset = GSM8KDataset(config.eval_data_path, tokenizer, max_samples=config.max_eval_samples)
    eval_data_raw = eval_dataset.data  # 用于 vLLM 验证
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        drop_last=True
    )
    
    # ==========================================
    # 5. 初始化优化器
    # ==========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ==========================================
    # 6. 训练循环
    # ==========================================
    global_step = 0
    model.train()

    print("=" * 60)
    print("开始 SFT 训练 (快速双 GPU 模式)...")
    print(f"训练 GPU: {config.train_device}")
    print(f"vLLM GPU: {config.vllm_device}")
    print("=" * 60)

    for epoch in range(config.epochs):
        print(f"\n{'='*20} Epoch {epoch + 1}/{config.epochs} {'='*20}")
        epoch_loss = 0.0
        num_batches = 0

        for idx, batch in enumerate(train_dataloader):
            # 将数据移动到训练 GPU
            input_ids = batch['input_ids'].to(config.train_device)
            labels = batch['labels'].to(config.train_device)
            response_mask = batch['response_mask'].to(config.train_device)

            num_response_tokens = response_mask.sum(dim=1).float()

            # 前向传播
            log_probs_dict = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False
            )
            policy_log_probs = log_probs_dict['log_probs']

            # 计算损失并反向传播
            scaled_loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=num_response_tokens.mean().item()
            )

            epoch_loss += metadata['unscaled_loss'].item()
            num_batches += 1

            # 梯度累积更新
            if (idx + 1) % config.gradient_accumulation_steps == 0:
                # 记录梯度范数（裁剪前）
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # 记录训练指标到 TensorBoard
                current_loss = metadata['unscaled_loss'].item()
                current_lr = optimizer.param_groups[0]['lr']

                writer.add_scalar("train/loss", current_loss, global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, global_step)

                # 计算并记录 perplexity（使用 math.exp 避免数值溢出）
                try:
                    # 限制 loss 的最大值，避免 PPL 过大导致 TensorBoard 无法正常显示
                    safe_loss = min(current_loss, 15.0)
                    perplexity = math.exp(safe_loss)
                except OverflowError:
                    perplexity = 0.0

                # 只记录有效的 perplexity 值
                if perplexity > 0.0:
                    writer.add_scalar("train/perplexity", perplexity, global_step)

                # 日志
                if global_step % config.log_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"[Step {global_step}] Loss: {current_loss:.4f} (avg: {avg_loss:.4f}), "
                          f"PPL: {perplexity:.2f}, GradNorm: {grad_norm:.4f}")

                # 快速 vLLM 验证（内存传输权重）
                if global_step % config.eval_steps == 0 and global_step > 0:
                    print(f"\n[Step {global_step}] 快速 vLLM 验证（内存传输权重）...")

                    # 直接将权重加载到 vLLM（无需磁盘 I/O）
                    load_policy_into_vllm_instance(model, vllm_engine)

                    # 执行验证
                    eval_results = evaluate_with_vllm(
                        vllm_engine,
                        eval_data_raw,
                        max_new_tokens=config.max_new_tokens
                    )

                    # 记录验证指标到 TensorBoard
                    writer.add_scalar("eval/accuracy", eval_results['accuracy'], global_step)
                    writer.add_scalar("eval/format_rate", eval_results['format_rate'], global_step)

                    print(f"验证结果: 准确率={eval_results['accuracy']*100:.2f}%, "
                          f"格式率={eval_results['format_rate']*100:.2f}%")

                    print("继续训练...\n")

                # 小批量生成监控（使用训练模型）
                if global_step % config.log_steps == 0 and global_step > 0:
                    sample_prompts = [eval_dataset[i]['prompt'] for i in range(min(4, len(eval_dataset)))]
                    sample_gts = [eval_dataset[i]['label'] for i in range(min(4, len(eval_dataset)))]

                    metrics = log_generations(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=sample_prompts,
                        ground_truths=sample_gts,
                        reward_fn=r1_zero_reward_fn,
                        step=global_step,
                        max_new_tokens=config.max_new_tokens
                    )

                    # 记录生成监控指标到 TensorBoard
                    writer.add_scalar("eval/avg_format_reward", metrics["eval/avg_format_reward"], global_step)
                    writer.add_scalar("eval/avg_answer_reward", metrics["eval/avg_answer_reward"], global_step)
                    writer.add_scalar("eval/avg_response_length", metrics["eval/avg_response_length"], global_step)
                    writer.add_scalar("eval/avg_entropy", metrics["eval/avg_entropy"], global_step)

                # 保存检查点
                if global_step % config.save_steps == 0 and global_step > 0:
                    ckpt_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"[Step {global_step}] 模型已保存到 {ckpt_path}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)
        print(f"Epoch {epoch + 1} 完成, 平均损失: {avg_epoch_loss:.4f}")
    
    # ==========================================
    # 7. 保存最终模型
    # ==========================================
    print(f"\n训练完成! 保存最终模型到 {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 关闭 TensorBoard writer
    writer.close()
    print(f"TensorBoard 日志已保存到: {config.log_dir}")
    print("使用命令查看: tensorboard --logdir=" + config.log_dir)

    print("SFT 训练全部完成!")

if __name__ == "__main__":
    train()
