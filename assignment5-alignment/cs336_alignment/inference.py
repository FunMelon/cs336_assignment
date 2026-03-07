#!/usr/bin/env python3
"""
使用 vLLM 在 GSM8K 上进行推理的脚本。
结果保存为与训练/测试集一致的 JSONL 格式。
"""

import json
import os
import argparse
from typing import List, Optional
from pathlib import Path

from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_prompt_template(prompt_path: str) -> str:
    """从文件中加载提示模板。"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_gsm8k_data(data_path: str) -> List[dict]:
    """从jsonl文件中加载GSM8K测试样本。"""
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples

def format_prompt(question: str, prompt_template: str) -> str:
    """使用提示模板格式化问题。"""
    return prompt_template.format(question=question)

def main():
    parser = argparse.ArgumentParser(description="Run inference on GSM8K using vLLM.")
    
    # 默认路径设置
    default_model_path = "/root/paddlejob/workspace/env_run/theft/cs336_data/model/Qwen2.5-Math-1.5B"
    default_data_path = "./dataset/gsm8k/processed_with_label/test.jsonl"
    
    # 获取当前脚本所在目录，以便定位 prompt 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_prompt_path = os.path.join(script_dir, "prompts", "r1_zero.prompt")
    
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the model")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the input data (jsonl)")
    parser.add_argument("--prompt_path", type=str, default=default_prompt_path, help="Path to the prompt template")
    parser.add_argument("--output_file", type=str, default="./out/inference_output.jsonl", help="Path to save the output jsonl")
    
    args = parser.parse_args()

    # 1. 加载数据
    print(f"正在加载数据: {args.data_path}")
    examples = load_gsm8k_data(args.data_path)
    print(f"已加载 {len(examples)} 个样本")

    # 2. 加载 Prompt 模板
    print(f"正在加载提示模板: {args.prompt_path}")
    prompt_template = load_prompt_template(args.prompt_path)
    
    # 3. 格式化 Prompts
    print("正在格式化提示...")
    prompts = [format_prompt(ex['question'], prompt_template) for ex in examples]

    # 4. 设置采样参数 (保持与 baseline 一致)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        stop=["</answer>", "User:", "\n\nUser:"],
        include_stop_str_in_output=True
    )

    # 5. 加载模型
    print(f"正在加载模型: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    # 6. 执行推理
    print(f"开始生成 {len(prompts)} 条回复...")
    outputs = llm.generate(prompts, sampling_params)

    # 7. 保存结果
    print(f"正在保存结果到: {args.output_file}")
    
    # 确保存储目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # 后处理：修复标签格式 (与 baseline 一致)
            generated_text = generated_text.replace("</think>\n<answer>", "</think> <answer>")
            
            # 构造输出对象，保持 dataset 格式 {"question": ..., "answer": ...}
            # 这里 "answer" 字段存放的是模型的生成结果
            record = {
                "question": examples[i]['question'],
                "answer": generated_text,
                "label": examples[i]['label']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("推理完成。")

if __name__ == "__main__":
    main()
