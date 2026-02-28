# CS336的课后作业
## A1 构建 Transformer LM 模型
- `assignment1-basics/cs336_basics/bpe/bpe_trainer.py`: 并行分词器训练脚本；
- `assignment1-basics/cs336_basics/bpe/streaming_bpe_trainer.py`: 流式分词器训练脚本；
- `assignment1-basics/cs336_basics/bpe/profile_bpe_trainer.py`: cProfile脚本；
- `assignment1-basics/text_compression_ratio.py`: 压缩率测试脚本；
- `assignment1-basics/text2int.py`: 字符数据集转id数据集的脚本；
- `assignment1-basics/train.py`: LM训练脚本（单卡）；
- `assignment1-basics/train_distributed.py`: LM分布式训练脚本（多卡DDP）；
- `assignment1-basics/run_distributed.sh`: 分布式训练启动脚本；
- `assignment1-basics/plot_loss.py`: 绘图脚本；
- `assignment1-basics/inference.py`: 推理对话脚本；
- `assignment1-basics/LR_range_test.py`: 学习率查找脚本；

### BPE编码器
- [x] 实现bpe编码器的训练逻辑；
- [x] 实现并行的预分词过程（无法通过测试代码可以尝试改小并行进程数）；
- [x] 实现分词器类使用分词结果；
- [x] 词表保存磁盘格式兼容huggingface格式；
- [x] 解决openweb大规模数据集在有限内存训练的问题（流式预分词）；
- [x] 修复流式预分词的并行化问题
- [x] 修复多线程cProfile的pickle问题

### Transformer语言模型架构
- [x] 实现线性层和嵌入层；
- [x] 实现RMSNorm；
- [x] 实现SiLU和SwiGLU的FFN层；
- [x] 实现RoPE编码；
- [x] 实现softmax函数；
- [x] 实现缩放点积注意力；
- [x] 实现多头自注意力；
- [x] 实现大矩阵乘法计算多头自注意力；
- [x] 实现transformer块；
- [x] 搭建完整的transformer语言模型；
- [x] 实现模型的参数量估计函数和计算量估计函数；
- [ ] 修正存在错误的显存和FLOPs计算函数；
- [x] 增加梯度累计；

### 训练Transformer语言模型
- [x] 实现交叉熵函数；
- [x] 实现AdamW优化器；
- [x] 实现带预热的余弦学习率调度；
- [x] 实现梯度剪裁；

### 训练循环
- [x] 实现数据加载器；
- [x] 实现checkpointing的保存和加载；
- [x] 实现完整的训练脚本；

### 分布式训练 (DDP，大多数内容由vibe-coding实现) 
- [x] 实现基于 PyTorch DDP 的多卡数据并行训练；
- [x] 支持 torchrun 和 mp.spawn 两种启动方式；
- [x] 实现分布式检查点的保存和加载（rank 0 保存，广播同步）；
- [x] 实现跨进程的路径信息同步（固定长度 tensor 广播）；
- [x] 按 GPU 数量自动缩放迭代次数，保持总样本量不变；
- [x] 配置 NCCL 环境变量，支持单机多卡通信；

### 生成文本
- [x] 搭建 generate text 过程；
- [x] 增加采样规则，实现top-p, temperature等功能
- [ ] 增加top-k采样规则

### 实验
- [x] 实现日志检查和损失曲线绘制功能；
- [x] 实现推理功能；
- [x] 增加学习率查找脚本；

### 扩展功能
- [x] DDP多卡训练；
- [x] 开启pytorch compile，并使用矩阵乘法加速；
- [x] 实现Muon优化器和AdamW优化器混合（3.75 → 3.73）；
- [x] 增加QK-Norm正则化项（3.73 → 3.69）；
- [x] 增加logit softcapping功能；
- [x] 扩大了模型参数（45.2M → 134M，3.10）；
- [x] 增加了early stopping功能；