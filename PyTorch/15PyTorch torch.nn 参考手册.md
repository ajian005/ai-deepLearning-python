## PyTorch torch.nn 参考手册
PyTorch 的 torch.nn 模块是构建和训练神经网络的核心模块，它提供了丰富的类和函数来定义和操作神经网络。

以下是 torch.nn 模块的一些关键组成部分及其功能：

### 1、nn.Module 类：

nn.Module 是所有自定义神经网络模型的基类。用户通常会从这个类派生自己的模型类，并在其中定义网络层结构以及前向传播函数（forward pass）。
### 2、预定义层（Modules）：

包括各种类型的层组件，例如卷积层（nn.Conv1d, nn.Conv2d, nn.Conv3d）、全连接层（nn.Linear）、激活函数（nn.ReLU, nn.Sigmoid, nn.Tanh）等。
### 3、容器类：

nn.Sequential：允许将多个层按顺序组合起来，形成简单的线性堆叠网络。
nn.ModuleList 和 nn.ModuleDict：可以动态地存储和访问子模块，支持可变长度或命名的模块集合。
### 4、损失函数（Loss Functions）：

torch.nn 包含了一系列用于衡量模型预测与真实标签之间差异的损失函数，例如均方误差损失（nn.MSELoss）、交叉熵损失（nn.CrossEntropyLoss）等。
### 5、实用函数接口（Functional Interface）：

nn.functional（通常简写为 F），包含了许多可以直接作用于张量上的函数，它们实现了与层对象相同的功能，但不具有参数保存和更新的能力。例如，可以使用 F.relu() 直接进行 ReLU 操作，或者 F.conv2d() 进行卷积操作。
### 6、初始化方法：

torch.nn.init 提供了一些常用的权重初始化策略，比如 Xavier 初始化 (nn.init.xavier_uniform_()) 和 Kaiming 初始化 (nn.init.kaiming_uniform_())，这些对于成功训练神经网络至关重要。

## PyTorch torch.nn 模块参考手册

### 神经网络容器
|类/函数|	描述|
|----|----|
|torch.nn.Module|	所有神经网络模块的基类。|
|torch.nn.Sequential(*args)	|按顺序组合多个模块。|
|torch.nn.ModuleList(modules)	|将子模块存储在列表中。|
|torch.nn.ModuleDict(modules)	|将子模块存储在字典中。|
|torch.nn.ParameterList(parameters)	|将参数存储在列表中。|
|torch.nn.ParameterDict(parameters)	|将参数存储在字典中。|

### 线性层
|类/函数|	描述|
|----|----|
|torch.nn.Linear(in_features, out_features)|全连接层。|
|torch.nn.Bilinear(in1_features, in2_features, out_features)	|双线性层。|


### 卷积神经网络层
|类/函数|	描述|
|----|----|
|torch.nn.Conv1d(in_channels, out_channels, kernel_size)|	一维卷积层。|
|torch.nn.Conv2d(in_channels, out_channels, kernel_size)|	二维卷积层。|
|torch.nn.Conv3d(in_channels, out_channels, kernel_size)|	三维卷积层。|
|torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size)|	一维转置卷积层。|
|torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size)|	二维转置卷积层。|
|torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size)|	三维转置卷积层。|


### 池化层
|类/函数|	描述|
|----|----|
|torch.nn.MaxPool1d(kernel_size)|	一维最大池化层。|
|torch.nn.MaxPool2d(kernel_size)|	二维最大池化层。|
|torch.nn.MaxPool3d(kernel_size)|	三维最大池化层。|
|torch.nn.AvgPool1d(kernel_size)|	一维平均池化层。|
|torch.nn.AvgPool2d(kernel_size)|	二维平均池化层。|
|torch.nn.AvgPool3d(kernel_size)|	三维平均池化层。|
|torch.nn.AdaptiveMaxPool1d(output_size)|	一维自适应最大池化层。|
|torch.nn.AdaptiveAvgPool1d(output_size)|	一维自适应平均池化层。|
|torch.nn.AdaptiveMaxPool2d(output_size)|	二维自适应最大池化层。|
|torch.nn.AdaptiveAvgPool2d(output_size)|	二维自适应平均池化层。|
|torch.nn.AdaptiveMaxPool3d(output_size)|	三维自适应最大池化层。|
|torch.nn.AdaptiveAvgPool3d(output_size)|	三维自适应平均池化层。|


### 激活函数
|类/函数|	描述|
|----|----|
|torch.nn.ReLU()|	ReLU 激活函数。|
|torch.nn.Sigmoid()|	Sigmoid 激活函数。|
|torch.nn.Tanh()|	Tanh 激活函数。|
|torch.nn.Softmax(dim)|	Softmax 激活函数。|
|torch.nn.LogSoftmax(dim)|	LogSoftmax 激活函数。|
|torch.nn.LeakyReLU(negative_slope)	|LeakyReLU 激活函数。|
|torch.nn.ELU(alpha)|	ELU 激活函数。|
|torch.nn.SELU()|	SELU 激活函数。|
|torch.nn.GELU()|	GELU 激活函数。|


### 损失函数
|类/函数|	描述|
|----|----|
|torch.nn.MSELoss()	|均方误差损失。|
|torch.nn.L1Loss()|	L1 损失。|
|torch.nn.CrossEntropyLoss()|	交叉熵损失。|
|torch.nn.NLLLoss()	|负对数似然损失。|
|torch.nn.BCELoss()|	二分类交叉熵损失。|
|torch.nn.BCEWithLogitsLoss()|	带 Sigmoid 的二分类交叉熵损失。|
|torch.nn.KLDivLoss()|	KL 散度损失。|
|torch.nn.HingeEmbeddingLoss()	|铰链嵌入损失。|
|torch.nn.MultiMarginLoss()|	多分类间隔损失。|
|torch.nn.SmoothL1Loss()|	平滑 L1 损失。|


### 归一化层
|类/函数|	描述|
|----|----|
|torch.nn.BatchNorm1d(num_features)|	一维批归一化层。|
|torch.nn.BatchNorm2d(num_features)	|二维批归一化层。|
|torch.nn.BatchNorm3d(num_features)|	三维批归一化层。|
|torch.nn.LayerNorm(normalized_shape)|	层归一化。|
|torch.nn.InstanceNorm1d(num_features)|	一维实例归一化层。|
|torch.nn.InstanceNorm2d(num_features)|	二维实例归一化层。|
|torch.nn.InstanceNorm3d(num_features)	|三维实例归一化层。|
|torch.nn.GroupNorm(num_groups, num_channels)	|组归一化。|


### 循环神经网络层
|类/函数|	描述|
|----|----|
|torch.nn.RNN(input_size, hidden_size)|	简单 RNN 层。|
|torch.nn.LSTM(input_size, hidden_size)|	LSTM 层。|
|torch.nn.GRU(input_size, hidden_size)|	GRU 层。|
|torch.nn.RNNCell(input_size, hidden_size)|	简单 RNN 单元。|
|torch.nn.LSTMCell(input_size, hidden_size)|	LSTM 单元。|
|torch.nn.GRUCell(input_size, hidden_size)|	GRU 单元。|


### 嵌入层
|类/函数|	描述|
|----|----|
|torch.nn.Embedding(num_embeddings, embedding_dim)|	嵌入层。|

### Dropout 层
|类/函数|	描述|
|----|----|
|torch.nn.Dropout(p)	|Dropout 层。|
|torch.nn.Dropout2d(p)	|2D Dropout 层。|
|torch.nn.Dropout3d(p)	|3D Dropout 层。|

### 实用函数
|函数|	描述|
|----|----|
|torch.nn.functional.relu(input)|	应用 ReLU 激活函数。|
|torch.nn.functional.sigmoid(input)|	应用 Sigmoid 激活函数。|
|torch.nn.functional.softmax(input, dim)|	应用 Softmax 激活函数。|
|torch.nn.functional.cross_entropy(input, target)|	计算交叉熵损失。|
|torch.nn.functional.mse_loss(input, target)|	计算均方误差损失。|


实例
```python
import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型和输入
model = SimpleNet()
input = torch.randn(5, 10)
output = model(input)
print(output)
```