import torch
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable

# torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 将PILImage转换为张量
    torchvision.transforms.Normalize((0.5,), (0.5,))  # 将[0, 1]归一化到[-1, 1]
    # 前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差
])
# 上述代码我们可以将transforms.Compose()看作一种容器，它能够同时对多种数据变换进行组合。
# 传入的参数是一个列表，列表中的元素就是对载入数据进行的变换操作。

data_train = torchvision.datasets.MNIST(root='./data/',
                                        transform=transform,
                                        train=True,
                                        download=True)

data_test = torchvision.datasets.MNIST(root='./data/',
                                       transform=transform,
                                       train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,  # 每个batch载入的图片数量，默认为1,这里设置为64
                                                shuffle=True,
                                                # num_workers=2#载入训练数据所需的子任务数
                                                )

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True,
                                               # num_workers=2
                                               )

# 这里, iter和next获取一个批次的图片数据和其对应的图片标签, 再使用torchvision.utils.make_grid将一个批次的图片构造成网格模式, 经过torchvision.utils.make_grid后图片维度变为channel, h, w三维,
# 因为要用matplotlib将图片显示, 我们要使用的数据要是数组且维度为（height, weight, channel）即色彩通道在最后, 因此我们需要用numpy和transpose完成原始数据类型的转换和数据维度的交换.

# 获取一个批次的图像和标签
images, labels = next(iter(data_loader_train))

# # 将单通道图像扩展为三通道
# images = images.expand(-1, 3, -1, -1)

# 将图像批次转换为网格
img = torchvision.utils.make_grid(images)

# 将张量转换为NumPy数组并调整通道顺序
img = img.numpy().transpose((1, 2, 0))

# 反归一化并显示图像
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)
plt.show()

import math, torch, numpy as np
import torch.nn as nn


# 实现卷积神经网络模型搭建：
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 构建卷积层之后的全连接层以及分类器
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dense = nn.Sequential(
            nn.Linear(14 * 14 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


net = Net()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
print(net)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)
    for data in data_loader_train:
        x_train, y_train = data
        # # 将单通道图像扩展为三通道
        # x_train = x_train.expand(-1, 3, -1, -1)
        x_train, y_train = Variable(x_train), Variable(y_train)
        output = net(x_train)
        _, predicted = torch.max(output.data, 1)
        optimizer.zero_grad()
        loss = cost(output, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_corrects += torch.sum(predicted == y_train.data)
    testing_corrects = 0
    for data in data_loader_test:
        x_test, y_test = data
        # # 将单通道图像扩展为三通道
        # x_test = x_test.expand(-1, 3, -1, -1)
        x_test, y_test = Variable(x_test), Variable(y_test)
        output = net(x_test)
        _, predicted = torch.max(output.data, 1)
        testing_corrects += torch.sum(predicted == y_test.data)
    print("Loss is:{:4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss / len(data_train), 100 * running_corrects / len(data_train)
                                                                                   , 100 * testing_corrects / len(data_test)))




data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)
x_test, y_test = next(iter(data_loader_test))
inputs = Variable(x_test)
predicted = net(inputs)
_, predicted = torch.max(predicted, 1)

print("Predict Label is:", [i for i in predicted.data])
print("Real Label is:", [i for i in y_test])
img = torchvision.utils.make_grid(x_test)
img = img.numpy().transpose((1, 2, 0))

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
plt.imshow(img)

