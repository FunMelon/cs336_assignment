from vllm import LLM, SamplingParams

# 示例提示词。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建采样参数对象，遇到换行符时停止生成。
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

# 创建 LLM 模型（使用本地模型路径）。
llm = LLM(model="/root/paddlejob/workspace/env_run/theft/cs336_data/model/Qwen2.5-Math-1.5B")

# 从提示词生成文本。输出是 RequestOutput 对象的列表，
# 包含提示词、生成的文本和其他信息。
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")