# import numpy as np
# import torch
#
# # 标量Tensor求导
# # 求 f(x) = a*x**2 + b*x + c 的导数
# x = torch.tensor(-2.0, requires_grad=True)
# a = torch.tensor(1.0)
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# y = a * torch.pow(x, 2) + b * x + c
# y.backward()  # backward求得的梯度会存储在自变量x的grad属性中
# dy_dx = x.grad
# print(dy_dx)
#
# # 非标量Tensor求导
# # 求 f(x) = a*x**2 + b*x + c 的导数
# x = torch.tensor([[-2.0, -1.0], [0.0, 1.0]], requires_grad=True)
# a = torch.tensor(1.0)
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])   # 使用链式法则，并且在 y.backward() 中传递一个梯度张量 gradient。这个 gradient 张量是一个与 y 形状相同的张量，通常它被设置为全1的张量，表示对每个元素的导数都求一次
# y = a * torch.pow(x, 2) + b * x + c
# y.backward(gradient=gradient)
# dy_dx = x.grad
# print(dy_dx)
#
# # 使用标量求导方式解决非标量求导
# # 求 f(x) = a*x**2 + b*x + c 的导数
# x = torch.tensor([[-2.0, -1.0], [0.0, 1.0]], requires_grad=True)
# a = torch.tensor(1.0)
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
# y = a * torch.pow(x, 2) + b * x + c
# z = torch.sum(y * gradient)
# z.backward()
# dy_dx = x.grad
# print(dy_dx)


# import torch
#
# # 单个自变量求导
# # 求 f(x) = a*x**4 + b*x + c 的导数
# x = torch.tensor(1.0, requires_grad=True)
# a = torch.tensor(1.0)
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# y = a * torch.pow(x, 4) + b * x + c
# # create_graph设置为True,允许创建更高阶级的导数
# # 求一阶导
# dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
# # 求二阶导
# dy2_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
# # 求三阶导
# dy3_dx3 = torch.autograd.grad(dy2_dx2, x)[0]
# print(dy_dx.data, dy2_dx2.data, dy3_dx3)
#
# # 多个自变量求偏导
# x1 = torch.tensor(1.0, requires_grad=True)
# x2 = torch.tensor(2.0, requires_grad=True)
# y1 = x1 * x2
# y2 = x1 + x2
# # 只有一个因变量,正常求偏导
# dy1_dx1, dy1_dx2 = torch.autograd.grad(outputs=y1, inputs=[x1, x2], retain_graph=True)
# print(dy1_dx1, dy1_dx2)
# # 若有多个因变量，则对于每个因变量,会将求偏导的结果加起来
# dy1_dx, dy2_dx = torch.autograd.grad(outputs=[y1, y2], inputs=[x1, x2])
# print(dy1_dx, dy2_dx)


# 例2-1-3 利用自动微分和优化器求最小值
import numpy as np
import torch

# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
optimizer = torch.optim.SGD(params=[x], lr=0.01)  # SGD为随机梯度下降
print(x)
print(x.data)
print(optimizer)


def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return result


for i in range(500):
    optimizer.zero_grad()  # 将模型的参数初始化为0
    y = f(x)
    y.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新所有的参数
    print("y=", y.data, ";", "x=", x.data)
