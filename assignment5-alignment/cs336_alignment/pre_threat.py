import json
import os
import glob
import argparse

def extract_ground_truth(answer_str: str) -> str:
    """
    从GSM8K原始答案字段中提取真实答案（数值）。
    GSM8K答案格式通常为: "Thinking Process....\n#### Final Answer"
    """
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()

def format_example(example):
    """
    转换单个样本的格式。
    1. 提取 label (正确答案)。
    2. 将 answer 转换为 "<think> ... </think> <answer> ... </answer>" 格式。
    """
    if 'answer' not in example:
        return example
    
    answer_raw = example['answer']
    
    # 1. 提取并添加 label 字段
    label = extract_ground_truth(answer_raw)
    
    # 2. 转换 answer 格式
    if '####' in answer_raw:
        parts = answer_raw.split('####')
        # 取最后一部分作为答案，前面作为思考过程
        answer_content = parts[-1].strip()
        think_content = "####".join(parts[:-1]).strip()
        new_answer = f"<think> {think_content} </think> <answer> {answer_content} </answer>"
    else:
        # 如果没有找到分隔符，保持原样（或者根据需要处理）
        new_answer = answer_raw

    # 创建新字典
    new_example = example.copy()
    new_example['answer'] = new_answer
    new_example['label'] = label  # 新增字段
    
    return new_example

def process_file(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                new_data = format_example(data)
                fout.write(json.dumps(new_data) + '\n')
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {input_file}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Convert GSM8K dataset format and add label.")
    
    # 默认路径
    default_input_dir = "./dataset/gsm8k"
    default_output_dir = os.path.join(default_input_dir, "processed_with_label") 
    
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="Input directory containing .jsonl files")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory to save processed files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        return
        
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} .jsonl files in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    for jsonl_file in jsonl_files:
        filename = os.path.basename(jsonl_file)
        
        # 避免递归处理输出目录
        if os.path.abspath(jsonl_file).startswith(os.path.abspath(args.output_dir)):
            continue
            
        output_file = os.path.join(args.output_dir, filename)
        process_file(jsonl_file, output_file)
        
    print("Done.")

if __name__ == "__main__":
    main()
