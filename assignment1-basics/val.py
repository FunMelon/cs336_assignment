# 验证脚本
import numpy as np
import tqdm
import torch
from cs336_basics import Transformer, cross_entropy_loss, get_batch

# 路径配置
checkpoint_path = "./out/owt/checkpoint.pth" 
valid_dataset_path = "../../cs336_data/id/owt-v-id/owt_valid.bin"

# 模型超参数
vocab_size = 32000
context_length = 1024
d_model = 768
nhead = 12
num_layers = 12
d_ff = 2048
logit_cap = 30.0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
batch_size = 64


valid_dataset = np.memmap(valid_dataset_path, dtype=np.uint16, mode="r")
model = Transformer(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    d_ff=d_ff,
    logit_cap=logit_cap,
    device=device,
)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device=device, dtype=dtype)
model.eval()

total_loss = 0.0
total_batches = 0
with torch.no_grad():
    for i in tqdm.tqdm(range(0, len(valid_dataset) - context_length, batch_size * context_length)):
        input_batch, target_batch = get_batch(
            valid_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        input_batch = input_batch.to(dtype=torch.int)
        target_batch = target_batch.to(dtype=torch.int)
        logits = model(input_batch)
        loss = cross_entropy_loss(logits, target_batch)
        total_loss += loss.item()
        total_batches += 1

avg_loss = total_loss / total_batches
print(f"Validation average loss: {avg_loss:.4f}")
