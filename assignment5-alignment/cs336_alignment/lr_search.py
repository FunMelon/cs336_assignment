"""
学习率搜索脚本 - 基于 GRPO 训练流程

在多个学习率上各运行少量 GRPO 步数，记录训练损失与验证准确率，
帮助快速确定最佳学习率区间。

用法:
    python lr_search.py
"""
import os
import json
import random
import copy
from typing import Literal
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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
# 搜索配置
# ==========================================
class LRSearchConfig:
    # 路径配置（与 train_grpo.py 保持一致）
    model_path = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/SFT"
    train_data_path = "dataset/gsm8k/processed_with_label/train.jsonl"
    eval_data_path = "dataset/gsm8k/processed_with_label/test.jsonl"
    output_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/LR_SEARCH"
    log_dir = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/LR_SEARCH/logs"
    prompt_template_path = "prompts/r1_zero.prompt"

    # ---- 学习率搜索空间 ----
    learning_rates: list[float] = [5e-7, 1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4]

    # ---- 每个学习率跑多少步 ----
    n_grpo_steps: int = 6  # 每个 lr 的 GRPO 步数（少量即可观察趋势）

    # GRPO 核心参数（与 train_grpo.py 默认值一致）
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 32
    epochs_per_rollout_batch: int = 2

    # 采样配置
    sampling_temperature: float = 0.7
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 512

    # 损失函数配置
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip"
    use_std_normalization: bool = False
    use_length_normalization: bool = True
    cliprange: float = 0.2

    # 优化器配置
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # 其他
    advantage_eps: float = 1e-6

    # 验证配置
    max_eval_samples: int = 1024  # 验证样本数

    # GPU 配置
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    vllm_gpu_memory_utilization: float = 0.85


# ==========================================
# 数据集（复用 train_grpo.py 的逻辑）
# ==========================================
class GSM8KDataset(Dataset):
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
            'label': item.get('label', ''),
        }


def load_prompt_template(template_path: str) -> str:
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, question: str) -> str:
    return template.format(question=question)


# ==========================================
# vLLM 工具函数（与 train_grpo.py 一致）
# ==========================================
def init_vllm(model_id, device, seed=42, gpu_memory_utilization=0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
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
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def rollout_with_vllm(llm, prompts, group_size, sampling_params):
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * group_size)
    outputs = llm.generate(repeated_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return repeated_prompts, responses


def evaluate_with_vllm(llm, eval_data, prompt_template, sampling_params):
    prompts = [format_prompt(prompt_template, item['question']) for item in eval_data]
    labels = [item.get('label', '') for item in eval_data]
    outputs = llm.generate(prompts, sampling_params)

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
        'accuracy': total_answer_reward / n_samples if n_samples > 0 else 0,
        'format_rate': total_format_reward / n_samples if n_samples > 0 else 0,
        'mean_reward': total_reward / n_samples if n_samples > 0 else 0,
        'total_samples': n_samples,
    }


def infinite_data_iterator(dataset, batch_size):
    indices = list(range(len(dataset)))
    while True:
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) == batch_size:
                yield [dataset[idx] for idx in batch_indices]


# ==========================================
# 单次学习率实验
# ==========================================
def run_single_lr_experiment(
    lr: float,
    config: LRSearchConfig,
    initial_state_dict: dict,
    tokenizer,
    vllm_engine,
    rollout_sampling_params,
    eval_sampling_params,
    train_dataset,
    eval_data_raw,
    prompt_template: str,
    writer: SummaryWriter,
    lr_idx: int,
) -> dict:
    """
    用给定的学习率从初始权重开始训练 n_grpo_steps 步，
    返回训练过程中的指标。
    """
    print(f"\n{'='*60}")
    print(f"  学习率实验 [{lr_idx+1}/{len(config.learning_rates)}]: lr = {lr}")
    print(f"{'='*60}")

    # 派生参数
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size
    n_optimizer_steps_per_epoch = config.rollout_batch_size // config.train_batch_size

    # 重新加载初始权重（保证每个 lr 起点一致）
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(config.train_device)
    policy.load_state_dict(initial_state_dict)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    data_iterator = infinite_data_iterator(train_dataset, n_prompts_per_rollout_batch)

    # 记录每步指标
    step_metrics = []

    for grpo_step in range(1, config.n_grpo_steps + 1):
        # --- 同步权重到 vLLM ---
        load_policy_into_vllm_instance(policy, vllm_engine)

        # --- 采样 prompts ---
        batch_data = next(data_iterator)
        questions = [item['question'] for item in batch_data]
        ground_truths = [item['label'] for item in batch_data]
        prompts = [format_prompt(prompt_template, q) for q in questions]

        # --- Rollout ---
        repeated_prompts, rollout_responses = rollout_with_vllm(
            llm=vllm_engine,
            prompts=prompts,
            group_size=config.group_size,
            sampling_params=rollout_sampling_params,
        )

        repeated_ground_truths = []
        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * config.group_size)

        # --- 计算优势 ---
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )
        advantages = advantages.to(config.train_device)
        raw_rewards = raw_rewards.to(config.train_device)

        # --- 预计算旧 log probs（off-policy） ---
        old_log_probs_all = None
        if config.loss_type == "grpo_clip" and config.epochs_per_rollout_batch > 1:
            policy.eval()
            with torch.inference_mode():
                tokenized = tokenize_prompt_and_output(
                    repeated_prompts, rollout_responses, tokenizer,
                )
                input_ids = tokenized['input_ids'].to(config.train_device)
                labels_tok = tokenized['labels'].to(config.train_device)

                all_old_log_probs = []
                for i in range(0, config.rollout_batch_size, micro_train_batch_size):
                    batch_input_ids = input_ids[i:i + micro_train_batch_size]
                    batch_labels = labels_tok[i:i + micro_train_batch_size]
                    batch_old_lp = get_response_log_probs(
                        model=policy,
                        input_ids=batch_input_ids,
                        labels=batch_labels,
                        return_token_entropy=False,
                    )
                    all_old_log_probs.append(batch_old_lp['log_probs'].detach())
                old_log_probs_all = torch.cat(all_old_log_probs, dim=0)
            policy.train()

        # --- 训练 ---
        policy.train()
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer,
        )
        input_ids_all = tokenized['input_ids'].to(config.train_device)
        labels_all = tokenized['labels'].to(config.train_device)
        response_mask_all = tokenized['response_mask'].to(config.train_device)

        total_loss = 0.0
        total_optimizer_steps = 0
        grad_norm = 0.0

        for epoch in range(config.epochs_per_rollout_batch):
            indices = list(range(config.rollout_batch_size))
            random.shuffle(indices)
            global_micro_idx = 0

            for opt_step in range(n_optimizer_steps_per_epoch):
                optimizer.zero_grad()
                for accum_idx in range(config.gradient_accumulation_steps):
                    start_idx = global_micro_idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size
                    batch_indices = indices[start_idx:end_idx]

                    input_ids = input_ids_all[batch_indices]
                    labels = labels_all[batch_indices]
                    response_mask = response_mask_all[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_raw_rewards = raw_rewards[batch_indices]

                    log_probs_dict = get_response_log_probs(
                        model=policy,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=False,
                    )
                    policy_log_probs = log_probs_dict['log_probs']

                    batch_old_log_probs = None
                    if old_log_probs_all is not None:
                        batch_old_log_probs = old_log_probs_all[batch_indices]

                    scaled_loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=batch_raw_rewards.unsqueeze(-1) if config.loss_type == "no_baseline" else None,
                        advantages=batch_advantages.unsqueeze(-1) if config.loss_type != "no_baseline" else None,
                        old_log_probs=batch_old_log_probs,
                        cliprange=config.cliprange if config.loss_type == "grpo_clip" else None,
                        use_length_normalization=config.use_length_normalization,
                    )
                    total_loss += loss_metadata['unscaled_loss'].item()
                    global_micro_idx += 1

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=config.max_grad_norm,
                )
                optimizer.step()
                total_optimizer_steps += 1

        n_total_microbatches = total_optimizer_steps * config.gradient_accumulation_steps
        avg_loss = total_loss / n_total_microbatches if n_total_microbatches > 0 else 0.0

        # TensorBoard 记录（tag 中带上 lr 信息）
        lr_tag = f"lr_{lr:.0e}"
        global_step = (lr_idx * config.n_grpo_steps) + grpo_step
        writer.add_scalar(f"{lr_tag}/train_loss", avg_loss, grpo_step)
        writer.add_scalar(f"{lr_tag}/train_reward_mean", reward_metadata['reward/mean'], grpo_step)
        writer.add_scalar(f"{lr_tag}/train_grad_norm", grad_norm, grpo_step)

        print(f"  [lr={lr:.1e}] Step {grpo_step}/{config.n_grpo_steps} | "
              f"Loss: {avg_loss:.4f} | "
              f"Reward: {reward_metadata['reward/mean']:.4f} | "
              f"GradNorm: {grad_norm:.2f}")

        step_metrics.append({
            'step': grpo_step,
            'loss': avg_loss,
            'reward_mean': reward_metadata['reward/mean'],
            'reward_std': reward_metadata['reward/std'],
            'format_reward': reward_metadata['reward/format_mean'],
            'answer_reward': reward_metadata['reward/answer_mean'],
            'grad_norm': float(grad_norm),
        })

    # --- 最终验证 ---
    print(f"  [lr={lr:.1e}] 进行最终验证...")
    load_policy_into_vllm_instance(policy, vllm_engine)
    eval_results = evaluate_with_vllm(
        llm=vllm_engine,
        eval_data=eval_data_raw,
        prompt_template=prompt_template,
        sampling_params=eval_sampling_params,
    )
    print(f"  [lr={lr:.1e}] 验证结果: 准确率={eval_results['accuracy']*100:.2f}%, "
          f"格式率={eval_results['format_rate']*100:.2f}%")

    writer.add_scalar(f"{lr_tag}/eval_accuracy", eval_results['accuracy'], config.n_grpo_steps)
    writer.add_scalar(f"{lr_tag}/eval_format_rate", eval_results['format_rate'], config.n_grpo_steps)

    # 清理 GPU 显存
    del policy, optimizer
    torch.cuda.empty_cache()

    return {
        'lr': lr,
        'step_metrics': step_metrics,
        'final_eval': eval_results,
    }


# ==========================================
# 主函数
# ==========================================
def main():
    config = LRSearchConfig()

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    # 加载 prompt 模板
    prompt_template = load_prompt_template(config.prompt_template_path)

    # 加载初始模型权重（只加载一次，后续每个 lr 实验复用）
    print("加载初始模型权重...")
    init_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    initial_state_dict = copy.deepcopy(init_model.state_dict())
    del init_model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 初始化 vLLM（只初始化一次）
    print(f"在 {config.vllm_device} 上初始化 vLLM...")
    vllm_engine = init_vllm(
        model_id=config.model_path,
        device=config.vllm_device,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )

    from vllm import SamplingParams
    rollout_sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        top_p=1.0,
        min_tokens=config.sampling_min_tokens,
        max_tokens=config.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=config.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 加载数据
    train_dataset = GSM8KDataset(config.train_data_path)
    eval_dataset = GSM8KDataset(config.eval_data_path, max_samples=config.max_eval_samples)
    eval_data_raw = eval_dataset.data

    # 初始验证（基线）
    print("\n进行初始验证（基线）...")
    # 先把初始权重加载到一个临时 policy 以同步给 vLLM
    tmp_policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(config.train_device)
    load_policy_into_vllm_instance(tmp_policy, vllm_engine)
    del tmp_policy
    torch.cuda.empty_cache()

    baseline_eval = evaluate_with_vllm(
        llm=vllm_engine,
        eval_data=eval_data_raw,
        prompt_template=prompt_template,
        sampling_params=eval_sampling_params,
    )
    print(f"基线准确率: {baseline_eval['accuracy']*100:.2f}%, "
          f"格式率: {baseline_eval['format_rate']*100:.2f}%")

    # ==========================================
    # 遍历所有学习率
    # ==========================================
    all_results = []

    for lr_idx, lr in enumerate(config.learning_rates):
        # 每个实验重置随机种子，保证数据采样一致
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        result = run_single_lr_experiment(
            lr=lr,
            config=config,
            initial_state_dict=initial_state_dict,
            tokenizer=tokenizer,
            vllm_engine=vllm_engine,
            rollout_sampling_params=rollout_sampling_params,
            eval_sampling_params=eval_sampling_params,
            train_dataset=train_dataset,
            eval_data_raw=eval_data_raw,
            prompt_template=prompt_template,
            writer=writer,
            lr_idx=lr_idx,
        )
        all_results.append(result)

    writer.close()

    # ==========================================
    # 汇总结果
    # ==========================================
    print("\n" + "=" * 70)
    print("  学习率搜索结果汇总")
    print("=" * 70)
    print(f"  基线准确率: {baseline_eval['accuracy']*100:.2f}%")
    print(f"  每个学习率训练 {config.n_grpo_steps} 步 GRPO")
    print("-" * 70)
    print(f"  {'LR':>12s} | {'最终Loss':>10s} | {'最终Reward':>10s} | {'验证准确率':>10s} | {'格式率':>10s}")
    print("-" * 70)

    best_lr = None
    best_acc = -1.0

    for r in all_results:
        lr = r['lr']
        final_loss = r['step_metrics'][-1]['loss']
        final_reward = r['step_metrics'][-1]['reward_mean']
        eval_acc = r['final_eval']['accuracy']
        eval_fmt = r['final_eval']['format_rate']

        marker = ""
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_lr = lr
            marker = " <-- best"

        print(f"  {lr:>12.1e} | {final_loss:>10.4f} | {final_reward:>10.4f} | "
              f"{eval_acc*100:>9.2f}% | {eval_fmt*100:>9.2f}%{marker}")

    print("-" * 70)
    print(f"  最佳学习率: {best_lr:.1e}  (验证准确率: {best_acc*100:.2f}%)")
    print("=" * 70)

    # 保存结果到 JSON
    results_path = os.path.join(config.output_dir, "lr_search_results.json")
    save_data = {
        'baseline': baseline_eval,
        'config': {
            'learning_rates': config.learning_rates,
            'n_grpo_steps': config.n_grpo_steps,
            'loss_type': config.loss_type,
            'rollout_batch_size': config.rollout_batch_size,
            'group_size': config.group_size,
            'train_batch_size': config.train_batch_size,
        },
        'results': all_results,
        'best_lr': best_lr,
        'best_accuracy': best_acc,
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")
    print(f"TensorBoard 日志: {config.log_dir}")
    print(f"  tensorboard --logdir={config.log_dir}")


if __name__ == "__main__":
    main()
