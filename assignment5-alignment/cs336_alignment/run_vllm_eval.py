"""
vLLM 验证脚本 - 运行在 GPU 1
由主训练脚本通过 subprocess 调用
"""
import argparse
import json
import sys
import os

# 确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn, extract_answer

def load_eval_data(data_path, max_samples=None):
    """加载验证数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            data.append(item)
    return data

def main():
    parser = argparse.ArgumentParser(description="vLLM 验证脚本")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--eval_data_path", type=str, required=True, help="验证数据路径")
    parser.add_argument("--max_samples", type=int, default=200, help="最大验证样本数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU 显存利用率")
    args = parser.parse_args()
    
    print(f"初始化 vLLM，加载模型: {args.ckpt_path}")
    
    # 初始化 vLLM
    # 注意：主进程已经设置了 CUDA_VISIBLE_DEVICES="1"
    # 所以这里 vLLM 会使用第二张卡
    llm = LLM(
        model=args.ckpt_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    
    # 加载验证数据
    print(f"加载验证数据: {args.eval_data_path}")
    eval_data = load_eval_data(args.eval_data_path, args.max_samples)
    print(f"共 {len(eval_data)} 条验证样本")
    
    # 准备 prompts
    prompts = [item['question'] for item in eval_data]
    
    # 获取 ground truth labels
    labels = []
    for item in eval_data:
        if 'label' in item:
            labels.append(item['label'])
        else:
            # 从 answer 中提取 #### 后面的数字
            answer = item.get('answer', '')
            if '####' in answer:
                label = answer.split('####')[-1].strip()
            else:
                label = ''
            labels.append(label)
    
    # 批量生成
    print("开始批量生成...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 评估结果
    total_correct = 0
    total_format_correct = 0
    total_samples = len(outputs)
    
    detailed_results = []
    
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        ground_truth = labels[i]
        
        # 使用 reward 函数评估
        reward_dict = r1_zero_reward_fn(response, ground_truth)
        
        is_correct = reward_dict['answer_reward'] > 0
        is_format_correct = reward_dict['format_reward'] > 0
        
        if is_correct:
            total_correct += 1
        if is_format_correct:
            total_format_correct += 1
        
        detailed_results.append({
            'prompt': prompts[i][:100] + '...',
            'response': response[:200] + '...' if len(response) > 200 else response,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'is_format_correct': is_format_correct,
        })
    
    # 计算指标
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    format_rate = total_format_correct / total_samples if total_samples > 0 else 0
    
    # 打印部分结果用于调试
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"正确答案数: {total_correct}")
    print(f"格式正确数: {total_format_correct}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"格式正确率: {format_rate * 100:.2f}%")
    
    # 打印几个示例
    print("\n" + "-" * 40)
    print("示例结果:")
    print("-" * 40)
    for i, res in enumerate(detailed_results[:3]):
        print(f"\n[样本 {i+1}]")
        print(f"问题: {res['prompt']}")
        print(f"生成: {res['response']}")
        print(f"标答: {res['ground_truth']}")
        print(f"正确: {res['is_correct']}, 格式: {res['is_format_correct']}")
    
    # 输出结果供主进程解析
    # 使用特殊前缀便于解析
    result = {
        'accuracy': accuracy,
        'format_rate': format_rate,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'total_format_correct': total_format_correct,
    }
    print(f"\nRESULT:{json.dumps(result)}")

if __name__ == "__main__":
    main()
