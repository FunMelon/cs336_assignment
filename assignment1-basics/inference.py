# 推理代码
import torch
import time
from cs336_basics import Transformer, Tokenizer

# 用户参数
prompt = "As a leading expert in artificial intelligence,"
max_len = 512
temperature = 0.8
top_p = 0.9
# 加载参数
checkpoint_path = "./out/owt/checkpoint.pth"
vocab_path = "./data/vocab/owt-t/vocab.json"
merges_path = "./data/vocab/owt-t/merges.txt"
special_tokens = ["<|endoftext|>"]
# 模型参数
vocab_size = 32000
context_length = 512
d_model = 512
nhead = 16
num_layers = 4
d_ff = 1344
rope_theta = 10000.0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32


if __name__ == "__main__":
    start_time = time.time()
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    tokenizer_load_time = time.time()
    print(f"Tokenizer loaded in {tokenizer_load_time - start_time:.2f} seconds.")

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        device=device,
        rope_theta=rope_theta,
        tokenizer=tokenizer,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device=device, dtype=dtype)
    model_load_time = time.time()
    print(f"Model loaded in {model_load_time - tokenizer_load_time:.2f} seconds.")

    ans = model.generate_text(
        prompt,
        max_length=max_len,
        temperature=temperature,
        top_p=top_p,
    )
    generation_time = time.time()
    print(f"Text generated in {generation_time - model_load_time:.2f} seconds.")
    print("Generated Text:")
    print("===================================")
    print(ans)
    print("===================================")
