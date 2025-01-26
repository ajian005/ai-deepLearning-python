import torch

# 当前安装的 PyTorch 库的版本
print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torch.cuda.is_available())


## 一个简单的实例，构建一个随机初始化的张量：
x = torch.rand(5, 3)
print(x)

