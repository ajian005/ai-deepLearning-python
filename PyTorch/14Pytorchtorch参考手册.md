## Pytorch torch 参考手册

PyTorch 软件包包含了用于多维张量的数据结构，并定义了在这些张量上执行的数学运算。此外，它还提供了许多实用工具，用于高效地序列化张量和任意类型的数据，以及其他有用的工具。

它还有一个 CUDA 版本，可以让你在计算能力 >= 3.0 的 NVIDIA GPU 上运行张量计算。

## PyTorch torch API 手册

PyTorch torch API 手册
|类别|	API|	描述|
|----|----|----|
|Tensors|	is_tensor(obj)|	检查 obj 是否为 PyTorch 张量。|
||is_storage(obj)|	检查 obj 是否为 PyTorch 存储对象。|
||is_complex(input)	|检查 input 数据类型是否为复数数据类型。|
||is_conj(input)	|检查 input 是否为共轭张量。|
||is_floating_point(input)	|检查 input 数据类型是否为浮点数据类型。|
||is_nonzero(input)	|检查 input 是否为非零单一元素张量。|
||set_default_dtype(d)	|设置默认浮点数据类型为 d。|
||get_default_dtype()	|获取当前默认浮点 torch.dtype。|
||set_default_device(device)|	设置默认 torch.Tensor 分配的设备为 device。|
||get_default_device()	|获取默认 torch.Tensor 分配的设备。|
||numel(input)	|返回 input 张量中的元素总数。|
|Creation Ops	|tensor(data)	|通过复制 data 构造无自动梯度历史的张量。|
||sparse_coo_tensor(indices, values)|	在指定的 indices 处构造稀疏张量，具有指定的值。|
||as_tensor(data)|	将 data 转换为张量，共享数据并尽可能保留自动梯度历史。|
||zeros(size)|	返回一个用标量值 0 填充的张量，形状由 size 定义。|
||ones(size)|	返回一个用标量值 1 填充的张量，形状由 size 定义。|
||arange(start, end, step)	|返回一个 1-D 张量，包含从 start 到 end 的值，步长为 step。|
||rand(size)|	返回一个从 [0, 1) 区间均匀分布的随机数填充的张量。|
||randn(size)|	返回一个从标准正态分布填充的张量。|
|Math operations	|add(input, other, alpha)|	将 other（由 alpha 缩放）加到 input 上。|
||mul(input, other)|	将 input 与 other 相乘。|
||matmul(input, other)	|执行 input 和 other 的矩阵乘法。|
||mean(input, dim)	|计算 input 在维度 dim 上的均值。|
||sum(input, dim)	|计算 input 在维度 dim 上的和。|
||max(input, dim)	|返回 input 在维度 dim 上的最大值。|
||min(input, dim)	|返回 input 在维度 dim 上的最小值。|
## Tensor 创建

|函数|	描述|
|----|----|
|torch.tensor(data, dtype, device, requires_grad)|	从数据创建张量。|
|torch.as_tensor(data, dtype, device)|	将数据转换为张量（共享内存）。|
|torch.from_numpy(ndarray)	|从 NumPy 数组创建张量（共享内存）。|
|torch.zeros(*size, dtype, device, requires_grad)|	创建全零张量。|
|torch.ones(*size, dtype, device, requires_grad)|	创建全一张量。|
|torch.empty(*size, dtype, device, requires_grad)|	创建未初始化的张量。|
|torch.arange(start, end, step, dtype, device, requires_grad)|	创建等差序列张量。|
|torch.linspace(start, end, steps, dtype, device, requires_grad)|	创建等间隔序列张量。|
|torch.logspace(start, end, steps, base, dtype, device, requires_grad)	|创建对数间隔序列张量。|
|torch.eye(n, m, dtype, device, requires_grad)|	创建单位矩阵。|
|torch.full(size, fill_value, dtype, device, requires_grad)	|创建填充指定值的张量。|
|torch.rand(*size, dtype, device, requires_grad)|	创建均匀分布随机张量（范围 [0, 1)）。|
|torch.randn(*size, dtype, device, requires_grad)|	创建标准正态分布随机张量。|
|torch.randint(low, high, size, dtype, device, requires_grad)	|创建整数随机张量。|
|torch.randperm(n, dtype, device, requires_grad)|	创建 0 到 n-1 的随机排列。|


## Tensor 操作
|函数|	描述|
|----|----|
|torch.cat(tensors, dim)|	沿指定维度连接张量。|
|torch.stack(tensors, dim)	|沿新维度堆叠张量。|
|torch.split(tensor, split_size, dim)|	将张量沿指定维度分割。|
|torch.chunk(tensor, chunks, dim)|	将张量沿指定维度分块。|
|torch.reshape(input, shape)|	改变张量的形状。|
|torch.transpose(input, dim0, dim1)	|交换张量的两个维度。|
|torch.squeeze(input, dim)	|移除大小为 1 的维度。|
|torch.unsqueeze(input, dim)	|在指定位置插入大小为 1 的维度。|
|torch.expand(input, size)|	扩展张量的尺寸。|
|torch.narrow(input, dim, start, length)	|返回张量的切片。|
|torch.permute(input, dims)|	重新排列张量的维度。|
|torch.masked_select(input, mask)	|根据布尔掩码选择元素。|
|torch.index_select(input, dim, index)	|沿指定维度选择索引对应的元素。|
|torch.gather(input, dim, index)	|沿指定维度收集指定索引的元素。|
|torch.scatter(input, dim, index, src)	|将 src 的值散布到 input 的指定位置。|
|torch.nonzero(input)|	返回非零元素的索引。    |


### 数学运算
|函数|	描述|
|----|----|
|torch.add(input, other)	|逐元素加法。|
|torch.sub(input, other)	|逐元素减法。|
|torch.mul(input, other)	|逐元素乘法。|
|torch.div(input, other)	|逐元素除法。|
|torch.matmul(input, other)	|矩阵乘法。|
|torch.pow(input, exponent)	|逐元素幂运算。|
|torch.sqrt(input)	|逐元素平方根。|
|torch.exp(input)	|逐元素指数函数。|
|torch.log(input)	|逐元素自然对数。|
|torch.sum(input, dim)	|沿指定维度求和。|
|torch.mean(input, dim)	|沿指定维度求均值。|
|torch.max(input, dim)	|沿指定维度求最大值。|
|torch.min(input, dim)	|沿指定维度求最小值。|
|torch.abs(input)|	逐元素绝对值。|
|torch.clamp(input, min, max)	|将张量值限制在指定范围内。|
|torch.round(input)	|逐元素四舍五入。|
|torch.floor(input)|	逐元素向下取整。|
|torch.ceil(input)	|逐元素向上取整。|


### 随机数生成
|函数|	描述|
|----|----|
|torch.manual_seed(seed)|	设置随机种子。|
|torch.initial_seed()|	返回当前随机种子。|
|torch.rand(*size)|	创建均匀分布随机张量（范围 [0, 1)）。|
|torch.randn(*size)|	创建标准正态分布随机张量。|
|torch.randint(low, high, size)|	创建整数随机张量。|
|torch.randperm(n)	| 返回 0 到 n-1 的随机排列。  |


## 线性代数
|函数|	描述|
|----|----|
|torch.dot(input, other)|	计算两个向量的点积。|
|torch.mm(input, mat2)	|矩阵乘法。|
|torch.bmm(input, mat2)|	批量矩阵乘法。|
|torch.eig(input)	|计算矩阵的特征值和特征向量。|
|torch.svd(input)	|计算矩阵的奇异值分解。|
|torch.inverse(input)	|计算矩阵的逆。|
|torch.det(input)|	计算矩阵的行列式。|
|torch.trace(input)	|计算矩阵的迹。|


## 设备管理
|函数|	描述|
|----|----|
|torch.cuda.is_available()|	检查 CUDA 是否可用。|
|torch.device(device)|	创建一个设备对象（如 'cpu' 或 'cuda:0'）。|
|torch.to(device)|	将张量移动到指定设备。|

实例
```python

import torch

# 创建张量
x = torch.tensor([1, 2, 3])
y = torch.zeros(2, 3)

# 数学运算
z = torch.add(x, 1)  # 逐元素加 1
print(z)

# 索引和切片
mask = x > 1
selected = torch.masked_select(x, mask)
print(selected)

# 设备管理
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = x.to(device)
    print(x.device)

```