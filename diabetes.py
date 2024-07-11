import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, encoding='utf8')
        self.len = xy.shape[0]  # shape[0]是矩阵的行数, shape[1]是矩阵的列数
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 获取数据索引
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 获取数据总量
    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers=2 为多线程


# 搭建FNN网络模型
class FNNModel(nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()
        self.linear1 = nn.Linear(8, 6)  # 输入数据的特征有8个，也就是有8个维度，随后将其降维到6维
        self.linear2 = nn.Linear(6, 4)  # 6维降到4维
        self.linear3 = nn.Linear(4, 2)  # 4维降到2维
        self.linear4 = nn.Linear(2, 1)  # 2维降到1维
        self.sigmoid = nn.Sigmoid()  # 可以视其为网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = FNNModel()

criterion = nn.BCELoss(reduction='mean')  # 返回损失的平均值
optimizer = optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

# 开始训练
if __name__ == '__main__':
    for epoch in range(100):
        # i 是一个epoch中第几次迭代，一共756条数据，每一个mini_batch为32，所以一个和epoch需要迭代23次
        # data获取的数据为(x, y)
        loss_one_epoch = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            print(inputs, labels)
            y_pred = model(inputs)  # 实则传入输入数据inputs，调用forward方法进行前向传播
            loss = criterion(y_pred, labels)
            loss_one_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss_one_epoch / 23)
        epoch_list.append(epoch)


    plt.plot(epoch_list, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
