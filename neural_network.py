import torch
from torch import nn
#
# batch_n = 100  # 一个批次输入数据的数量
# hidden_layer = 100  # 隐藏层神经元的数量
# input_data = 1000  # 每个数据的特征维度
# output_data = 10  # 输出数据的维度
#
# # 生成随机输入和输出数据。x 是形状为 (100, 1000) 的随机输入数据，y 是形状为 (100, 10) 的随机输出数据。
# x = torch.randn(batch_n, input_data)
# y = torch.randn(batch_n, output_data)
#
# # 初始化权重 w1 和 w2，分别用于输入层到隐藏层和隐藏层到输出层的映射。w1 形状为 (1000, 100)，w2 形状为 (100, 10)。
# w1 = torch.randn(input_data, hidden_layer)
# w2 = torch.randn(hidden_layer, output_data)
#
# epoch_count = 20  # 训练的轮数
# learning_rate = 1e-6  # 学习率
#
# for epoch in range(epoch_count):
#     # 计算隐藏层的线性变换，h1 的形状为 (100, 100)。
#     h1 = x.mm(w1)  # (100, 1000) * (1000, 100) --> (100, 100)
#     # 打印 h1 的形状，用于调试。
#     print(h1.shape)
#     # 使用 clamp(min=0) 就相当于应用 ReLU 激活函数，将 h1 中所有小于 0 的值置为 0。
#     """
#     clamp 和 clamp_ 都是用于限制张量的值在某个范围内的函数，但它们有一些关键的区别：
#         clamp 会返回一个新的张量，原始张量不会被修改。
#         clamp_ 是一个原地操作（in-place operation），会直接修改原始张量。
#     """
#     h1 = h1.clamp(min=0)
#     # 计算输出层的线性变换，y_pred 的形状为 (100, 10)。
#     y_pred = h1.mm(w2)
#     # 计算预测值 y_pred 和真实值 y 之间的平方误差损失。
#     loss = (y_pred - y).pow(2).sum()
#     # 打印当前轮次的损失值。
#     print("epoch: {}, loss: {}".format(epoch, loss))
#     # 计算损失函数对预测值 y_pred 的梯度。
#     grad_y_pred = 2 * (y_pred - y)
#     # 计算损失函数对权重 w2 的梯度。
#     grad_w2 = h1.t().mm(grad_y_pred)
#     # 计算损失函数对隐藏层输出 h1 的梯度。首先复制 grad_y_pred，然后计算其与 w2 转置的矩阵乘法。
#     grad_h = grad_y_pred.clone()
#     grad_h = grad_h.mm(w2.t())
#     # 将 grad_h 中小于 0 的值置为 0，对应 ReLU 激活函数的梯度。
#     # ReLU（Rectified Linear Unit，修正线性单元）是一种常用的激活函数，广泛应用于神经网络中。ReLU 的定义非常简单：ReLU(x)=max(0,x)
#     grad_h.clamp_(min=0)
#     # 计算损失函数对权重 w1 的梯度。
#     grad_w1 = x.t().mm(grad_h)
#
#     # 更新权重 w1 和 w2，使用学习率 lr 乘以梯度。
#     w1 = w1 - learning_rate * grad_w1
#     w2 = w2 - learning_rate * grad_w2
#     print("w1: {}, w2: {}".format(w1, w2))
#     """
#     这个简单的神经网络训练循环使用了全连接层和 ReLU 激活函数，通过梯度下降法进行权重更新，以最小化平方误差损失函数。
#     """


from torch.autograd import Variable

#
# batch_size = 100
# hidden_layer = 100
# input_data = 1000
# output_data = 10
#
# # 用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是False，表示该变量在进行自动梯度计算的过程中不会保留梯度值
# x = Variable(torch.randn(batch_size, input_data), requires_grad=False)
# y = Variable(torch.randn(batch_size, output_data), requires_grad=False)
#
# w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
# w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)
#
# epoch_num = 50
# learning_rate = 1e-6
#
# for epoch in range(epoch_num):
#     h1 = x.mm(w1)
#     print(h1.shape)
#     h1 = h1.clamp(min=0)
#     y_pred = h1.mm(w2)
#     loss = (y_pred - y).pow(2).sum()
#     print("epoch: {}, loss: {:.4f}".format(epoch, loss))
#     # 后向传播
#     loss.backward()
#     w1.data -= learning_rate * w1.grad.data
#     w2.data -= learning_rate * w2.grad.data
#
#     w1.grad.data.zero_()
#     w2.grad.data.zero_()

#
# batch_size = 64
# hidden_size = 100
# input_size = 1000
# output_size = 10
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#     def forward(self, input, w1, w2):
#         x = torch.mm(input, w1)
#         x = torch.clamp(x, min=0)
#         x = torch.mm(x, w2)
#         return x
#
#     def backward(self, grad_output):
#         pass
#
#
# net = Net()
# x = Variable(torch.randn(batch_size, input_size), requires_grad=False)
# y = Variable(torch.randn(batch_size, output_size), requires_grad=False)
#
# w1 = Variable(torch.randn(input_size, hidden_size), requires_grad=True)
# w2 = Variable(torch.randn(hidden_size, output_size), requires_grad=True)
#
# epoch_num = 30
# learning_rate = 1e-6
#
# for epoch in range(epoch_num):
#     y_pred = net(x, w1, w2)
#     loss = (y_pred - y).pow(2).sum()
#     print("epoch %d, loss %f" % (epoch, loss.data))
#     loss.backward()
#     w1.data -= learning_rate * w1.grad.data
#     w2.data -= learning_rate * w2.grad.data
#
#     w1.grad.data.zero_()
#     w2.grad.data.zero_()

loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
# loss_function = nn.CrossEntropyLoss()
x = Variable(torch.randn(100, 100))
y = Variable(torch.randn(100, 100))
loss = loss_function(x, y)

batch_size = 100
hidden_size = 100
input_size = 1000
output_size = 10

x = Variable(torch.randn(batch_size, input_size), requires_grad=False)
y = Variable(torch.randn(batch_size, output_size), requires_grad=False)

models = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
# nn.Sequential括号内就是我们搭建的神经网络模型的具体结构，Linear完成从隐藏层到输出层的线性变换，再用ReLU激活函数激活
# nn.Sequential类是torch.nn中的一种序列容器，通过在容器中嵌套各种实现神经网络模型的搭建，
# 最主要的是，参数会按照我们定义好的序列自动传递下去。

epochs = 10000
learning_rate = 1e-6

optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
# 使用torch.optim.Adam类作为我们模型参数的优化函数，这里输入的是：被优化的参数和学习率的初始值。
# 因为我们需要优化的是模型中的全部参数，所以传递的参数是models.parameters()

for epoch in range(epochs):
    y_pred = models(x)

    loss = loss_function(y_pred, y)
    if epoch % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss', '%.4f' % loss)
    # models.zero_grad()
    optimizer.zero_grad()   # 将模型参数的梯度归0

    loss.backward()
    optimizer.step()    # 使用计算得到的梯度值对各个节点的参数进行梯度更新。

    # for param in models.parameters():
    #     param.data -= learning_rate * param.grad.data
