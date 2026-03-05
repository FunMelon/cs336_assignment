import os
import torch
import torch.distributed as dist  # PyTorch 分布式通信库
import torch.multiprocessing as mp  # PyTorch 多进程库

def setup(rank, world_size):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的编号（从0开始）
        world_size: 总进程数
    """
    # 设置主节点地址为本地
    os.environ["MASTER_ADDR"] = "localhost"
    # 设置主节点通信端口
    os.environ["MASTER_PORT"] = "29500"
    # 初始化进程组，使用gloo后端（适用于CPU和GPU）
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    """
    分布式演示函数：展示all-reduce集合通信操作
    
    Args:
        rank: 当前进程的编号
        world_size: 总进程数
    """
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 每个进程生成一个随机整数张量（3个元素，范围0-9）
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    
    # all-reduce操作：将所有进程的data张量求和，结果广播到每个进程
    # async_op=False表示同步执行，等待操作完成后再继续
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    # 设置总进程数为6（模拟6个并行工作进程）
    world_size = 6
    # 使用spawn方式启动多个进程
    # fn: 每个进程要执行的函数
    # args: 传递给函数的额外参数
    # nprocs: 启动的进程数量
    # join: 等待所有进程执行完毕
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)