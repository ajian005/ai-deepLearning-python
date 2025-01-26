## PyTorch 张量（Tensor）
张量是一个多维数组，可以是标量、向量、矩阵或更高维度的数据结构。

在 PyTorch 中，张量（Tensor）是数据的核心表示形式，类似于 NumPy 的多维数组，但具有更强大的功能，例如支持 GPU 加速和自动梯度计算。

张量支持多种数据类型（整型、浮点型、布尔型等）。

张量可以存储在 CPU 或 GPU 中，GPU 张量可显著加速计算。

下图展示了不同维度的张量（Tensor）在 PyTorch 中的表示方法：
![alt text](../src/images/1D-5DTensor_D5ZvufDS38WkhK9rK32hQ.jpg)

说明：

 - **1D Tensor / Vector（一维张量/向量）**: 最基本的张量形式，可以看作是一个数组，图中的例子是一个包含 10 个元素的向量。
 - **2D Tensor / Matrix（二维张量/矩阵）**: 二维数组，通常用于表示矩阵，图中的例子是一个 4x5 的矩阵，包含了 20 个元素。
 - **3D Tensor / Cube（三维张量/立方体）**: 三维数组，可以看作是由多个矩阵堆叠而成的立方体，图中的例子展示了一个 3x4x5 的立方体，其中每个 5x5 的矩阵代表立方体的一个"层"。
 - **4D Tensor / Vector of Cubes（四维张量/立方体向量）**: 四维数组，可以看作是由多个立方体组成的向量，图中的例子没有具体数值，但可以理解为一个包含多个 3D 张量的集合。
 - **5D Tensor / Matrix of Cubes（五维张量/立方体矩阵）**: 五维数组，可以看作是由多个4D张量组成的矩阵，图中的例子同样没有具体数值，但可以理解为一个包含多个 4D 张量的集合。

 ***

 ## 创建张量
张量创建的方式有：


|方法 |	说明|示例代码|
|----|----|----|
|torch.tensor(data)	|从 Python 列表或 NumPy 数组创建张量。	|x = torch.tensor([[1, 2], [3, 4]])|
|torch.zeros(size)	|创建一个全为零的张量。	|x = torch.zeros((2, 3))|
|torch.ones(size)	|创建一个全为 1 的张量。	|x = torch.ones((2, 3))|
|torch.empty(size)	|创建一个未初始化的张量。	|x = torch.empty((2, 3))|
|torch.rand(size)	|创建一个服从均匀分布的随机张量，值在 [0, 1)。	|x = torch.rand((2, 3))|
|torch.randn(size)	|创建一个服从正态分布的随机张量，均值为 0，标准差为 1。	|x = torch.randn((2, 3))|
|torch.arange(start, end, step)	|创建一个一维序列张量，类似于 Python 的 range。	|x = torch.arange(0, 10, 2)|
|torch.linspace(start, end, steps)	|创建一个在指定范围内等间隔的序列张量。	|x = torch.linspace(0, 1, 5)|
|torch.eye(size)	|创建一个单位矩阵（对角线为 1，其他为 0）。	|x = torch.eye(3)|
|torch.from_numpy(ndarray)	|将 NumPy 数组转换为张量。	|x = torch.from_numpy(np.array([1, 2, 3]))|

使用 **torch.tensor()** 函数，你可以将一个列表或数组转换为张量：



实例
```python
import torch

tensor = torch.tensor([1, 2, 3])
print(tensor)
```
输出如下：
```
tensor([1, 2, 3])
```

如果你有一个 NumPy 数组，可以使用 torch.from_numpy() 将其转换为张量：


实例
```python
import numpy as np

np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(tensor)
```
输出如下：
```
tensor([1, 2, 3])
```



创建 2D 张量（矩阵）：

实例
```python
import torch

tensor_2d = torch.tensor([
    [-9, 4, 2, 5, 7],
    [3, 0, 12, 8, 6],
    [1, 23, -6, 45, 2],
    [22, 3, -1, 72, 6]
])
print("2D Tensor (Matrix):\n", tensor_2d)
print("Shape:", tensor_2d.shape)  # 形状
```
输出如下：
```
2D Tensor (Matrix):
 tensor([[-9,  4,  2,  5,  7],
        [ 3,  0, 12,  8,  6],
        [ 1, 23, -6, 45,  2],
        [22,  3, -1, 72,  6]])
Shape: torch.Size([4, 5])
```



其他维度的创建：
```python
# 创建 3D 张量（立方体）
tensor_3d = torch.stack([tensor_2d, tensor_2d + 10, tensor_2d - 5])  # 堆叠 3 个 2D 张量
print("3D Tensor (Cube):\n", tensor_3d)
print("Shape:", tensor_3d.shape)  # 形状

# 创建 4D 张量（向量的立方体）
tensor_4d = torch.stack([tensor_3d, tensor_3d + 100])  # 堆叠 2 个 3D 张量
print("4D Tensor (Vector of Cubes):\n", tensor_4d)
print("Shape:", tensor_4d.shape)  # 形状

# 创建 5D 张量（矩阵的立方体）
tensor_5d = torch.stack([tensor_4d, tensor_4d + 1000])  # 堆叠 2 个 4D 张量
print("5D Tensor (Matrix of Cubes):\n", tensor_5d)
print("Shape:", tensor_5d.shape)  # 形状
```

## 张量的属性
张量的属性如下表:


|方法 |	说明|示例|
|----|----|----|
|.shape	|获取张量的形状	|tensor.shape|
|.size()|获取张量的形状|tensor.size()|
|.dtype	|获取张量的数据类型|tensor.dtype|
|.device|查看张量所在的设备 (CPU/GPU)|tensor.device|
|.dim()	|获取张量的维度数|tensor.dim()|
|.requires_grad|是否启用梯度计算|tensor.requires_grad|
|.numel()|获取张量中的元素总数|tensor.numel()|
|.is_cuda|检查张量是否在 GPU 上	|tensor.is_cuda|
|.T	|获取张量的转置（适用于 2D 张量）|tensor.T |
|.item()|获取单元素张量的值|tensor.item()|
|.is_contiguous()|检查张量是否连续存储|tensor.is_contiguous()|

实例
```python
import torch

# 创建一个 2D 张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 张量的属性
print("Tensor:\n", tensor)
print("Shape:", tensor.shape)  # 获取形状
print("Size:", tensor.size())  # 获取形状（另一种方法）
print("Data Type:", tensor.dtype)  # 数据类型
print("Device:", tensor.device)  # 设备
print("Dimensions:", tensor.dim())  # 维度数
print("Total Elements:", tensor.numel())  # 元素总数
print("Requires Grad:", tensor.requires_grad)  # 是否启用梯度
print("Is CUDA:", tensor.is_cuda)  # 是否在 GPU 上
print("Is Contiguous:", tensor.is_contiguous())  # 是否连续存储

# 获取单元素值
single_value = torch.tensor(42)
print("Single Element Value:", single_value.item())

# 转置张量
tensor_T = tensor.T
print("Transposed Tensor:\n", tensor_T)
```

输出结果：
```
Tensor:
 tensor([[1., 2., 3.],
         [4., 5., 6.]])
Shape: torch.Size([2, 3])
Size: torch.Size([2, 3])
Data Type: torch.float32
Device: cpu
Dimensions: 2
Total Elements: 6
Requires Grad: False
Is CUDA: False
Is Contiguous: True
Single Element Value: 42
Transposed Tensor:
 tensor([[1., 4.],
         [2., 5.],
         [3., 6.]])
```

***

##张量的操作
张量操作方法说明如下。

**基础操作**：
|操作|说明|示例代码|
|----|----|----|
|+, -, *, /	|元素级加法、减法、乘法、除法。	|z = x + y|
|torch.matmul(x, y)|矩阵乘法。|	z = torch.matmul(x, y)|
|torch.dot(x, y) |向量点积（仅适用于 1D 张量）。|z = torch.dot(x, y)|
|torch.sum(x)	|求和。	|z = torch.sum(x)|
|torch.mean(x)	|求均值。|z = torch.mean(x)|
|torch.max(x)	|求最大值。|z = torch.max(x)|
|torch.min(x)	|求最小值。|z = torch.min(x)|
|torch.argmax(x, dim)|返回最大值的索引（指定维度）。|z = torch.argmax(x, dim=1)|
|torch.softmax(x, dim)|计算 softmax（指定维度）。|z = torch.softmax(x, dim=1)|


**形状操作**
|操作|说明|示例代码|
|----|----|----|
|x.view(shape)|改变张量的形状（不改变数据）。|z = x.view(3, 4)|
|x.reshape(shape)|类似于 view，但更灵活。  |z = x.reshape(3, 4)|
|x.t()	|转置矩阵。	|z = x.t()|
|x.unsqueeze(dim) |	在指定维度添加一个维度。|z = x.unsqueeze(0) |
|x.squeeze(dim)	|去掉指定维度为 1 的维度。	|z = x.squeeze(0) |
|torch.cat((x, y), dim)	|按指定维度连接多个张量。	|z = torch.cat((x, y), dim=1)|

输出结果：
```
原始张量:
 tensor([[1., 2., 3.],
         [4., 5., 6.]])

【索引和切片】
获取第一行: tensor([1., 2., 3.])
获取第一行第一列的元素: tensor(1.)
获取第二列的所有元素: tensor([2., 5.])

【形状变换】
改变形状后的张量:
 tensor([[1., 2.],
         [3., 4.],
         [5., 6.]])
展平后的张量:
 tensor([1., 2., 3., 4., 5., 6.])

【数学运算】
张量加 10:
 tensor([[11., 12., 13.],
         [14., 15., 16.]])
张量乘 2:
 tensor([[ 2.,  4.,  6.],
         [ 8., 10., 12.]])
张量元素的和: 21.0

【与其他张量操作】
另一个张量:
 tensor([[1., 1., 1.],
         [1., 1., 1.]])
矩阵乘法结果:
 tensor([[ 6.,  6.],
         [15., 15.]])

【条件判断和筛选】
大于 3 的元素的布尔掩码:
 tensor([[False, False, False],
         [ True,  True,  True]])
大于 3 的元素:
 tensor([4., 5., 6.])
 ```

 ## 张量的 GPU 加速

将张量转移到 GPU：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1.0, 2.0, 3.0], device=device)
```
检查 GPU 是否可用：
```python
torch.cuda.is_available()  # 返回 True 或 False
```


## 张量与 NumPy 的互操作
张量与 NumPy 的互操作如下表所示：

|操作|	说明|	示例代码|
|----|----|----|
|torch.from_numpy(ndarray)| 将 NumPy 数组转换为张量。|x = torch.from_numpy(np_array) |
|x.numpy() | 将张量转换为 NumPy 数组（仅限 CPU 张量）。| np_array = x.numpy() |

实例
```python
import torch
import numpy as np

# 1. NumPy 数组转换为 PyTorch 张量
print("1. NumPy 转为 PyTorch 张量")
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy 数组:\n", numpy_array)

# 使用 torch.from_numpy() 将 NumPy 数组转换为张量
tensor_from_numpy = torch.from_numpy(numpy_array)
print("转换后的 PyTorch 张量:\n", tensor_from_numpy)

# 修改 NumPy 数组，观察张量的变化（共享内存）
numpy_array[0, 0] = 100
print("修改后的 NumPy 数组:\n", numpy_array)
print("PyTorch 张量也会同步变化:\n", tensor_from_numpy)

# 2. PyTorch 张量转换为 NumPy 数组
print("\n2. PyTorch 张量转为 NumPy 数组")
tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
print("PyTorch 张量:\n", tensor)

# 使用 tensor.numpy() 将张量转换为 NumPy 数组
numpy_from_tensor = tensor.numpy()
print("转换后的 NumPy 数组:\n", numpy_from_tensor)

# 修改张量，观察 NumPy 数组的变化（共享内存）
tensor[0, 0] = 77
print("修改后的 PyTorch 张量:\n", tensor)
print("NumPy 数组也会同步变化:\n", numpy_from_tensor)

# 3. 注意：不共享内存的情况（需要复制数据）
print("\n3. 使用 clone() 保证独立数据")
tensor_independent = torch.tensor([[13, 14, 15], [16, 17, 18]], dtype=torch.float32)
numpy_independent = tensor_independent.clone().numpy()  # 使用 clone 复制数据
print("原始张量:\n", tensor_independent)
tensor_independent[0, 0] = 0  # 修改张量数据
print("修改后的张量:\n", tensor_independent)
print("NumPy 数组（不会同步变化）:\n", numpy_independent)
```

输出结果：
```
1. NumPy 转为 PyTorch 张量
NumPy 数组:
 [[1 2 3]
 [4 5 6]]
转换后的 PyTorch 张量:
 tensor([[1, 2, 3],
         [4, 5, 6]])

修改后的 NumPy 数组:
 [[100   2   3]
 [  4   5   6]]
PyTorch 张量也会同步变化:
 tensor([[100,   2,   3],
         [  4,   5,   6]])

2. PyTorch 张量转为 NumPy 数组
PyTorch 张量:
 tensor([[ 7.,  8.,  9.],
         [10., 11., 12.]])
转换后的 NumPy 数组:
 [[ 7.  8.  9.]
 [10. 11. 12.]]

修改后的 PyTorch 张量:
 tensor([[77.,  8.,  9.],
         [10., 11., 12.]])
NumPy 数组也会同步变化:
 [[77.  8.  9.]
 [10. 11. 12.]]

3. 使用 clone() 保证独立数据
原始张量:
 tensor([[13., 14., 15.],
         [16., 17., 18.]])
修改后的张量:
 tensor([[ 0., 14., 15.],
         [16., 17., 18.]])
NumPy 数组（不会同步变化）:
 [[13. 14. 15.]
 [16. 17. 18.]]
 ```

 