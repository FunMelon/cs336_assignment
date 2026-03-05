import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time


class SimpleModel(nn.Module):
    """简单的两层MLP模型"""
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class RandomDataset(Dataset):
    """生成随机数据的数据集"""
    def __init__(self, num_samples=1000, input_size=784, num_classes=10):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机输入和标签
        x = torch.randn(self.input_size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def setup(rank, world_size):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的编号（0 到 world_size-1）
        world_size: 总进程数（设备数）
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # 使用NCCL后端（GPU）或GLOO后端（CPU）
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def naive_ddp_step(model, data, target, criterion, optimizer, rank, world_size):
    """
    朴素DDP的单步训练
    
    Args:
        model: 模型
        data: 输入数据（已经是当前rank的子集）
        target: 标签（已经是当前rank的子集）
        criterion: 损失函数
        optimizer: 优化器
        rank: 当前进程编号
        world_size: 总进程数
    
    Returns:
        loss: 损失值
    """
    # ==================== 步骤1：数据已经通过DistributedSampler切分 ====================
    # DataLoader + DistributedSampler 自动确保每个设备收到不重叠的 n/d 个样本
    
    # ==================== 步骤2：独立的前向与反向传播 ====================
    # 每个设备独立进行前向传播
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    
    # 每个设备独立进行反向传播，计算基于本地数据的梯度
    loss.backward()
    
    # 此时每个设备的梯度只基于自己看到的 n/d 个样本
    
    # ==================== 步骤3：梯度同步 (All-Reduce) ====================
    # 对所有参数的梯度进行 all-reduce 操作
    # all-reduce 会将所有设备的梯度求和，然后除以 world_size 得到平均梯度
    for param in model.parameters():
        if param.grad is not None:
            # all_reduce 默认是求和操作
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # 除以设备数得到平均梯度
            param.grad /= world_size
    
    # 执行完毕后，每个设备上的梯度都是全局 n 个样本的平均梯度
    
    # ==================== 步骤4：优化器更新 ====================
    # 每个设备使用相同的平均梯度独立更新参数
    # 因为初始参数相同、梯度相同，所以更新后的参数也保持一致
    optimizer.step()
    
    return loss.item()


def train_epoch(rank, world_size, model, dataloader, criterion, optimizer, epoch):
    """
    训练一个epoch
    
    Args:
        rank: 当前进程编号
        world_size: 总进程数
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epoch: 当前epoch编号
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 设置当前epoch（用于DistributedSampler的shuffle）
    dataloader.sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 将数据移到对应的设备
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        data, target = data.to(device), target.to(device)
        
        # 执行朴素DDP训练步骤
        loss = naive_ddp_step(model, data, target, criterion, optimizer, rank, world_size)
        
        total_loss += loss
        num_batches += 1
        
        # 每10个batch打印一次
        if batch_idx % 10 == 0:
            print(f"[Rank {rank}] Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss:.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def verify_model_sync(rank, world_size, model):
    """
    验证所有设备上的模型参数是否同步
    
    Args:
        rank: 当前进程编号
        world_size: 总进程数
        model: 模型
    """
    for name, param in model.named_parameters():
        # 收集所有rank的参数
        param_list = [torch.zeros_like(param) for _ in range(world_size)]
        dist.all_gather(param_list, param)
        
        # 在rank 0上检查是否所有参数都相同
        if rank == 0:
            all_equal = all(torch.allclose(param_list[0], p, rtol=1e-5) for p in param_list[1:])
            status = "✓ Synced" if all_equal else "✗ NOT Synced"
            print(f"  {name}: {status}")


def train_naive_ddp(rank, world_size, num_epochs=3, batch_size=32):
    """
    朴素DDP训练主函数
    
    Args:
        rank: 当前进程编号
        world_size: 总进程数
        num_epochs: 训练轮数
        batch_size: 批大小（全局）
    """
    print(f"[Rank {rank}] Starting Naive DDP training")
    
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 创建模型
    model = SimpleModel().to(device)
    
    # 确保所有设备的初始模型参数相同
    # 从rank 0广播参数到所有其他设备
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Naive DDP Training Configuration")
        print("=" * 80)
        print(f"World Size (Devices): {world_size}")
        print(f"Global Batch Size: {batch_size}")
        print(f"Local Batch Size per Device: {batch_size // world_size}")
        print(f"Number of Epochs: {num_epochs}")
        print(f"Device Type: {device.type}")
        print("=" * 80 + "\n")
    
    # 创建数据集和数据加载器
    dataset = RandomDataset(num_samples=1000)
    
    # ==================== 关键：使用DistributedSampler进行数据切分 ====================
    # DistributedSampler 确保每个设备获得不重叠的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # 总设备数
        rank=rank,                 # 当前设备编号
        shuffle=True               # 每个epoch打乱数据
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size // world_size,  # 每个设备的本地批大小
        sampler=sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        
        avg_loss = train_epoch(rank, world_size, model, dataloader, criterion, optimizer, epoch)
        
        epoch_time = time.time() - start_time
        
        if rank == 0:
            print(f"\n[Rank {rank}] Epoch {epoch} completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f}")
            
            # 验证模型参数在所有设备上是否同步
            print(f"\nVerifying model synchronization after epoch {epoch}:")
            verify_model_sync(rank, world_size, model)
            print()
    
    # 最终验证
    if rank == 0:
        print("\n" + "=" * 80)
        print("Training Completed! Final Model Synchronization Check:")
        print("=" * 80)
        verify_model_sync(rank, world_size, model)
        print("=" * 80 + "\n")
    
    # 清理分布式环境
    cleanup()


def main():
    """主函数"""
    world_size = 2  # 使用2个进程（设备）
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        world_size = min(world_size, num_gpus)
    else:
        print("CUDA not available, using CPU")
    
    print(f"Launching {world_size} processes for Naive DDP training...\n")
    
    # 使用spawn启动多进程训练
    mp.spawn(
        fn=train_naive_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    main()
