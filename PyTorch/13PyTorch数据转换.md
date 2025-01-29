## PyTorch 数据转换

在 PyTorch 中，数据转换（Data Transformation） 是一种在加载数据时对数据进行处理的机制，将原始数据转换成适合模型训练的格式，主要通过 torchvision.transforms 提供的工具完成。

数据转换不仅可以实现基本的数据预处理（如归一化、大小调整等），还能帮助进行数据增强（如随机裁剪、翻转等），提高模型的泛化能力。

#### 为什么需要数据转换？
##### 数据预处理：

- 调整数据格式、大小和范围，使其适合模型输入。
- 例如，图像需要调整为固定大小、张量格式并归一化到 [0,1]。
##### 数据增强：

- 在训练时对数据进行变换，以增加多样性。
- 例如，通过随机旋转、翻转和裁剪增加数据样本的变种，避免过拟合。
##### 灵活性：

- 通过定义一系列转换操作，可以动态地对数据进行处理，简化数据加载的复杂度。

在 PyTorch 中，torchvision.transforms 模块提供了多种用于图像处理的变换操作。


#### 基础变换操作
|变换函数名称|	描述|	实例|
|----|----|----|
|transforms.ToTensor()	|将PIL图像或NumPy数组转换为PyTorch张量，并自动将像素值归一化到 [0, 1]。	|transform = transforms.ToTensor()|
|transforms.Normalize(mean, std)|	对图像进行标准化，使数据符合零均值和单位方差。	|transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])|
|transforms.Resize(size)|	调整图像尺寸，确保输入到网络的图像大小一致。	|transform = transforms.Resize((256, 256))|
|transforms.CenterCrop(size)	|从图像中心裁剪指定大小的区域。	|transform = transforms.CenterCrop(224)|


##### 1、ToTensor

将 PIL 图像或 NumPy 数组转换为 PyTorch 张量。

同时将像素值从 [0, 255] 归一化为 [0, 1]。

```python
from torchvision import transforms

transform = transforms.ToTensor()
```
##### 2、Normalize

对数据进行标准化，使其符合特定的均值和标准差。

通常用于图像数据，将其像素值归一化为零均值和单位方差。
```python
transform = transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
```

##### 3、Resize

调整图像的大小。
```python
transform = transforms.Resize((128, 128))  # 将图像调整为 128x128
```
##### 4、CenterCrop

从图像中心裁剪指定大小的区域。
```python
transform = transforms.CenterCrop(128)  # 裁剪 128x128 的区域
```

#### 数据增强操作
|变换函数名称|	描述|	实例|
|----|----|----|
|transforms.RandomHorizontalFlip(p)|	随机水平翻转图像。	|transform = transforms.RandomHorizontalFlip(p=0.5)|
|transforms.RandomRotation(degrees)	|随机旋转图像。	|transform = transforms.RandomRotation(degrees=45)|
|transforms.ColorJitter(brightness, contrast, saturation, hue)	|调整图像的亮度、对比度、饱和度和色调。	|transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)|
|transforms.RandomCrop(size)|	随机裁剪指定大小的区域。|	transform = transforms.RandomCrop(224)|
|transforms.RandomResizedCrop(size)	|随机裁剪图像并调整到指定大小。	|transform = transforms.RandomResizedCrop(224)|

##### 1、RandomCrop

从图像中随机裁剪指定大小。
```python
transform = transforms.RandomCrop(128)
```

##### 2、RandomHorizontalFlip

以一定概率水平翻转图像。
```python
transform = transforms.RandomHorizontalFlip(p=0.5)  # 50% 概率翻转
```

##### 3、RandomRotation

随机旋转一定角度。
```python
transform = transforms.RandomRotation(degrees=30)  # 随机旋转 -30 到 +30 度
```

##### 4、ColorJitter

随机改变图像的亮度、对比度、饱和度或色调。
```python
transform = transforms.ColorJitter(brightness=0.5, contrast=0.5)
```


#### 组合变换
|变换函数名称|	描述|	实例|
|----|----|----|
|transforms.Compose()|	将多个变换组合在一起，按照顺序依次应用。	|transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((256, 256))])|

通过 transforms.Compose 将多个变换组合起来。
```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

#### 自定义转换

如果 transforms 提供的功能无法满足需求，可以通过自定义类或函数实现。

实例
```python
class CustomTransform:
    def __call__(self, x):
        # 这里可以自定义任何变换逻辑
        return x * 2

transform = CustomTransform()
```

## 实例
### 对图像数据集应用转换
加载 MNIST 数据集，并应用转换。
实例
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 使用 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 查看转换后的数据
for images, labels in train_loader:
    print("图像张量大小:", images.size())  # [batch_size, 1, 128, 128]
    break
```
输出结果为：
```
图像张量大小: torch.Size([32, 1, 128, 128])
```

可视化转换效果
以下代码展示了原始数据和经过转换后的数据对比。


### 可视化转换效果
以下代码展示了原始数据和经过转换后的数据对比。
实例

```python
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import datasets, transforms


# 原始和增强后的图像可视化
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# 加载数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_augment)

# 显示图像
def show_images(dataset):
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        image, label = dataset[i]
        axs[i].imshow(image.squeeze(0), cmap='gray')  # 将 (1, H, W) 转为 (H, W)
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')
    plt.show()

show_images(dataset)
```

显示如下所示：

![alt text](../src/images/digit_image.png)





