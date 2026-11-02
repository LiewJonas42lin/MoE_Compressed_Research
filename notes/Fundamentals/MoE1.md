# 初次学习

## Neurons 基本神经元
\[
\boxed{y=\operatorname{ReLU}(w_1x_1+w_2x_2+b)}
\]
W -> weight  线性变化+偏置 再非线性激活

多个神经元不同权重 -> 学习不同特征  --> 所以有了矩阵乘法
\[
\begin{aligned}
x&=[2,1]\\
W_1&=
\begin{bmatrix}
1&-1&0\\
0&2&1
\end{bmatrix}\\
h&=\operatorname{ReLU}(xW_1)=[2,0,1]
\end{aligned}
\]

## 为什么 MoE 专家通常是 FFN？
Input -> Embedding -> Attention -> 残差连接处理与归一化 -> 再前馈: ReLU
前馈:数据沿着一个方向流动：
\[
Input \rightarrow W_1 \rightarrow ReLU
\rightarrow W_2 \rightarrow Output
\]

多头: head-->一种视角 layer? 和MoE的expert?

Attention负责不同Token之间的信息交互，获得全局信息
FFN负责[逐个]对每个Token的特征进行**非线性变换**->升维再降维

FFN有一个特点:FFN天然适合被拆分成多个独立的网络
->Router为每个Token选择专家
\[
FFN_i(x)=W_{2,i}\,\sigma(W_{1,i}x+b_{1,i})+b_{2,i}
\]
不同专家可以学习不同的特征变换,不是全都用
--> MoE核心优势:
**增加模型参数容量，同时保持相对稀疏的计算**

反向传播：模型计算预测结果，与正确答案比较，得到Loss(概率差距大小得到不同调整)

训练时Router为不同Token选择专家，专家会根据自己的训练损失获得梯度更新
Router(Router本身也是一个可训练的神经网络模块)决定哪些Expert获得训练机会
Gradient(告诉模型权重应该朝哪个方向调整，以及当前的局部变化有多敏感)决定这些Expert的权重如何变化，逐渐形成不同的专家行为

## 一个角度

FFN内部不同神经元的激活具有稀疏性，很多输入只会激活其中一小部分神经元。
研究者尝试把原本的 FFN 划分成多个功能分区，再通过 Router 选择需要的部分


## 任务
**Mixed-Bit Quantization**
同等量化方式，不同的输出误差 [权重分布、接收的输入、激活模式...]
eg.专家激活得更多 ≠ 更难压缩

Core: Quantization Sensitivity
不从零设计、训练一个MoE大模型，而是研究如何压缩已经训练好的MoE，以及如何评价压缩效果PTQ

Minimum requirements:
- 加载一个预训练 MoE 模型
- 了解参数 BF16 INT8 INT4
- 复现现有的 INT8 / INT4 量化策略
- 统计各 Expert 的 Routing Frequency
- 测量各 Expert 的 Quantization Sensitivity
- 比较不同 Mixed-Bit Allocation 策略
- 测试模型精度、存储占用以及实际推理表现

one method: Model Loss Increase（模型损失增量）