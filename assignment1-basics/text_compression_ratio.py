# 测试不同分词器在不同数据集上的压缩率
from cs336_basics import Tokenizer 
import random

# -----------------------------
# 1. 加载样本文本
# -----------------------------
# 假设每个数据集是一个 jsonl 或 txt，每行一条文档
def load_texts(path, n=10):
    """随机采样 n 条文档"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return random.sample(lines, n) if len(lines) >= n else lines


# -----------------------------
# 2. 加载两个分词器
# -----------------------------
def load_tokenizer(vocab_path, merges_path):
    return Tokenizer.from_files(vocab_path, merges_path)


tokenizer_tiny = load_tokenizer("./data/vocab/ts-t/vocab.json", "./data/vocab/ts-t/merges.txt")
tokenizer_owt = load_tokenizer("./data/vocab/owt/vocab.json", "./data/vocab/owt/merges.txt")
tokenizer_owt_hf = load_tokenizer("./data/vocab/owt-t/vocab.json", "./data/vocab/owt-t/merges.txt")
# -----------------------------
# 3. 加载 TinyStories 和 OWT 数据样本
# -----------------------------
samples_tiny = load_texts("./data/vocab/ts-t/merges.txt", n=10)
samples_owt = load_texts("./data/vocab/owt/merges.txt", n=10)

# -----------------------------
# 4. 定义压缩率计算函数
# -----------------------------
def compression_ratio(tokenizer, samples):
    """计算平均压缩率（bytes/token）"""
    total_bytes = 0
    total_tokens = 0
    for text in samples:
        encoded = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(encoded)
    return total_bytes / total_tokens if total_tokens else float("inf")


# -----------------------------
# 5. 计算各模型压缩率
# -----------------------------
ratio_tiny_tiny = compression_ratio(tokenizer_tiny, samples_tiny)
ratio_tiny_owt = compression_ratio(tokenizer_tiny, samples_owt)
ratio_owt_tiny = compression_ratio(tokenizer_owt, samples_tiny)
ratio_owt_owt = compression_ratio(tokenizer_owt, samples_owt)
ratio_owt_hf_tiny = compression_ratio(tokenizer_owt_hf, samples_tiny)
ratio_owt_hf_owt = compression_ratio(tokenizer_owt_hf, samples_owt)

# -----------------------------
# 6. 打印结果
# -----------------------------
print("Compression ratio (bytes/token):")
print(f"TinyStories tokenizer on TinyStories:     {ratio_tiny_tiny:.3f}")
print(f"TinyStories tokenizer on OpenWebText:     {ratio_tiny_owt:.3f}")
print(f"OpenWebText tokenizer on TinyStories:     {ratio_owt_tiny:.3f}")
print(f"OpenWebText tokenizer on OpenWebText:     {ratio_owt_owt:.3f}")
print(f"OpenWebText-HF tokenizer on TinyStories:  {ratio_owt_hf_tiny:.3f}")
print(f"OpenWebText-HF tokenizer on OpenWebText:  {ratio_owt_hf_owt:.3f}")
