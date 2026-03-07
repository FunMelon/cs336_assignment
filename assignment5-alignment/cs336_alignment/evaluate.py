#!/usr/bin/env python3
"""
评估 GSM8K 推理结果的脚本。
直接读取推理结果中的 label 字段作为 Ground Truth，无需参考文件。
"""

import json
import argparse
import os
from typing import List
from tqdm import tqdm
from drgrpo_grader import r1_zero_reward_fn

def load_data(path: str) -> List[dict]:
    """从jsonl文件中加载数据。"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K inference results using embedded labels.")
    
    parser.add_argument("--prediction_file", type=str, default="./out/inference_output.jsonl", help="Path to the prediction file (jsonl) containing 'label' field")

    args = parser.parse_args()

    # 1. 加载预测结果
    print(f"正在加载预测结果: {args.prediction_file}")
    if not os.path.exists(args.prediction_file):
        print(f"Error: Prediction file {args.prediction_file} does not exist.")
        return
        
    predictions = load_data(args.prediction_file)
    print(f"已加载 {len(predictions)} 条预测")
    
    # 2. 评估
    metrics_aggregate = {
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "reward": 0.0
    }
    
    results_details = []
    n_evaluated = 0
    n_missing_label = 0
    
    print("正在评估...")
    for i, pred in enumerate(tqdm(predictions)):
        # 提取模型生成内容 (Answer 字段)
        generation = pred.get('answer', '')
        
        # 提取真实答案 (从 label 字段)
        ground_truth = pred.get('label')
        
        if ground_truth is None:
            n_missing_label += 1
            # 可以选择跳过或报错，这里选择跳过并记录
            continue
            
        # 计算 Reward
        # r1_zero_reward_fn(completion, solution, fast=True)
        metrics = r1_zero_reward_fn(generation, ground_truth, fast=True)
        
        # 累加指标
        for key in metrics_aggregate:
            metrics_aggregate[key] += metrics.get(key, 0.0)
            
        results_details.append({
            "id": i,
            "question": pred.get('question', ''),
            "generation": generation,
            "ground_truth": ground_truth,
            "metrics": metrics
        })
        n_evaluated += 1

    if n_missing_label > 0:
        print(f"Warning: Skipped {n_missing_label} examples due to missing 'label' field.")

    # 3. 计算平均值
    if n_evaluated > 0:
        metrics_avg = {k: v / n_evaluated for k, v in metrics_aggregate.items()}
    else:
        metrics_avg = metrics_aggregate

    # 4. 输出摘要
    print("\n" + "="*60)
    print("评估摘要")
    print("="*60)
    print(f"评估样本数量: {n_evaluated}")
    print(f"格式奖励 (Format Reward): {metrics_avg['format_reward']:.4f}")
    print(f"答案奖励 (Answer Reward): {metrics_avg['answer_reward']:.4f}")
    print(f"总体奖励 (Total Reward):  {metrics_avg['reward']:.4f}")
    print(f"准确率 (Accuracy):      {metrics_avg['reward']*100:.2f}%")
    print("="*60)
    
if __name__ == "__main__":
    main()
